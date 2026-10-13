"""Setup shared by the gist trainers (gist_cpt_trainer.py, gist_sft_trainer*.py,
gist_trainer_long.py)."""

import argparse
import os

import datasets
import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

from gist_utils import Global_data, add_gist_args, apply_gist_args, gist_token_names
from src.data.gist_input_preprocessor import (
    custom_collate_chunkaug,
    gist_attention_preprocessor,
    link_custom_collate_chunkaug,
    meta_custom_collate_chunkaug,
    meta_link_custom_collate_chunkaug,
)

data_dir = str(os.environ.get("DATA_DIR", "../"))

FT_QA_DATA = "gist-sparse-attention/GSA-FT-Llama-3.2-1B-data"

SFT_SCHEDULE = dict(
    num_train_epochs=2,
    warmup_ratio=0.1,
    learning_rate=5e-6,
    eval_strategy="epoch",
    save_total_limit=3,
)


class GistTrainer(Trainer):
    """Also logs the loss components the gist models accumulate in Global_data."""

    def log(self, logs, start_time=None):
        if self.model.training:
            if Global_data.last_loss_ce:
                logs["loss_ce"] = np.sum(Global_data.last_loss_ce) / self.args.logging_steps
            if Global_data.last_loss_align:
                logs["loss_align"] = np.sum(Global_data.last_loss_align) / self.args.logging_steps
        Global_data.last_loss_ce = []
        Global_data.last_loss_align = []
        super().log(logs, start_time)


def make_parser(batch_size_per_device: int = 2) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gist on Long Context Training")
    add_gist_args(parser)
    parser.add_argument("--batch_size_per_device", type=int, default=batch_size_per_device)
    return parser


def add_sft_args(parser):
    parser.add_argument("--start_model", type=str, required=True,
                        help="Gist-pretrained checkpoint to start from.")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Name of the output directory and wandb run.")
    return parser


def configure(args) -> int:
    """Apply the gist flags to Global_data; return the number of gist levels."""
    num_gist_level = apply_gist_args(args)
    print(f"Batch size per device: {args.batch_size_per_device}")
    print(f"Meta gist: {args.meta_gist}")
    print(f"Selective training: {args.selective}")
    print(f"Top k for selective training: {Global_data.top_k}")
    print(f"Top p for selective training: {Global_data.top_p}")
    print(f"Sink size for gist attention: {Global_data.sink_size}")
    print(f"Grouped-union GQA enabled: {Global_data.use_nsa_gqa}")
    print(f"Chunk sizes for gist attention: {Global_data.chunk_size}")
    print(f"Number of previous chunks to attend to: {Global_data.num_previous_chunks}")
    return num_gist_level


def chunk_tag(meta_gist: int, sep: str = "_") -> str:
    """'chunk8', or 'chunk4_chunk4' with meta gists, as in run and dataset names."""
    sizes = Global_data.chunk_size[:2] if meta_gist else Global_data.chunk_size[:1]
    return sep.join(f"chunk{c}" for c in sizes)


def load_model(model_cls, path, selective: bool, attn_implementation: str = "sdpa"):
    """bf16 model and left-padding tokenizer; pad falls back to eos."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = model_cls.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=attn_implementation
    )
    model.selective = selective
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer, model


def set_gist_tokens(tokenizer, gist_token_ids):
    Global_data.gist_token_id = gist_token_ids
    Global_data.gist_token = gist_token_names(len(gist_token_ids))
    Global_data.pad_token_id = tokenizer.pad_token_id


def load_qa(tokenizer, gist_token_ids, meta_gist: int, link: bool = False):
    """Train and test splits of the QA finetuning data, gist-processed."""
    preprocessor = gist_attention_preprocessor(
        tokenizer=tokenizer,
        max_len=4096,
        do_shuffle=True,
        chunk_size=Global_data.chunk_size,
        attention_sink_size=Global_data.sink_size,
        gist_token_ids=gist_token_ids,
        pad_token=Global_data.pad_token_id,
        meta_gist=meta_gist,
        interleaved=False,
        query_first=False,
        use_semantic_chunking=True,
    )
    process = preprocessor.gist_process_qa_link if link else preprocessor.gist_process_qa
    data = datasets.load_dataset(FT_QA_DATA)
    return [
        data[split].map(
            process,
            remove_columns=["question", "answers", "generated", "documents"],
            num_proc=8,
            batched=False,
            load_from_cache_file=True,
        )
        for split in ("train", "test")
    ]


def collator(meta_gist: int, link: bool = False):
    if link:
        return meta_link_custom_collate_chunkaug if meta_gist else link_custom_collate_chunkaug
    return meta_custom_collate_chunkaug if meta_gist else custom_collate_chunkaug


def training_arguments(args, run_name: str, global_batch_size: int, **kwargs):
    """Arguments common to all stages; `kwargs` sets the stage-specific schedule."""
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = global_batch_size // num_gpus // args.batch_size_per_device
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    return TrainingArguments(
        output_dir=f"{data_dir}/training_res/{run_name}",
        report_to="wandb",
        run_name=run_name,
        per_device_train_batch_size=args.batch_size_per_device,
        per_device_eval_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_dir="training_res/logs",
        logging_steps=10,
        save_steps=1000,
        lr_scheduler_type="cosine",
        bf16=True,
        do_eval=True,
        gradient_checkpointing=True,
        # Non-reentrant: the reentrant default reruns the forward under no_grad, which
        # compiles flex_attention a second time (grad_mode guard).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        eval_on_start=False,
        seed=42,
        **kwargs,
    )
