import os
from typing import Tuple
import argparse
import json
import datasets
import torch
import numpy as np
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from functools import partial

from src.data.gist_input_preprocessor import custom_collate_chunkaug, gist_attention_preprocessor, meta_custom_collate_chunkaug
from transformers import Trainer

from gist_utils import Global_data
import gist_llama
from gist_llama import GistLlamaForCausalLM

data_dir = str(os.environ.get("DATA_DIR", "../"))


class GistTrainer(Trainer):
    """Custom Trainer that logs loss components from Global_data."""

    def log(self, logs, start_time = None):
        """Override log method to add loss components."""
        # Add loss components if available
        
        if self.model.training:
            if hasattr(Global_data, 'last_loss_ce') and len(Global_data.last_loss_ce) > 0:
                logs["loss_ce"] = np.sum(Global_data.last_loss_ce) / self.args.logging_steps
            if hasattr(Global_data, 'last_loss_align') and len(Global_data.last_loss_align) > 0:
                logs["loss_align"] = np.sum(Global_data.last_loss_align) / self.args.logging_steps

        Global_data.last_loss_ce = []
        Global_data.last_loss_align = []

        # Call parent log method
        super().log(logs, start_time)


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: gist_attention_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name == "text_multichunk":
        preprocessor_fn = preprocessor.gist_process_pretraining_multichunk_completion_compress
        data_path = "processed/fineweb/text_min2048"
    else:
        raise NotImplementedError()
    remove_columns = [
        "text", "id", "dump", "url", "date",
        "file_path", "language", "language_score", "token_count",
    ]
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)

    # streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    streaming_train_dataset = data_component["train"]
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=100,
        # batched=True,
        load_from_cache_file=False
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=100,
        # batched=True,
        load_from_cache_file=False
    )

    return training_data, eval_data


def main():

    parser = argparse.ArgumentParser(description="Gist on Long Context Training")
    parser.add_argument("--meta_gist", type=int, default=0, help="Use meta gist or not.")
    parser.add_argument("--batch_size_per_device", type=int, default=2, help="Batch size.")
    parser.add_argument(
        "--chunk_size",
        type=int,
        nargs="+",
        default=[8],
        help="Chunk size for gist attention per layer."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[3],
        help="Top-k values for selective training."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0,
        help="Top-p value for selective training (nucleus sampling)."
    )
    parser.add_argument("--num_gist_level", type=int, default=2, help="Number of levels of gist tokens.")
    parser.add_argument("--sink_size", type=int, default=3, help="Sink size for gist attention.")
    parser.add_argument(
        "--selective",
        action="store_true",
        help="Enable selective training."
    )
    parser.add_argument("--num_previous_chunks", type=int, default=1, help="Number of previous chunks to attend to.")
    args = parser.parse_args()

    ####################################
    # Configurations
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    batch_size_per_device = args.batch_size_per_device
    gradient_accumulation_steps = 128 // num_gpus // batch_size_per_device
    meta_gist = args.meta_gist
    selective = args.selective
    Global_data.top_k = args.top_k
    Global_data.top_p = args.top_p
    Global_data.sink_size = args.sink_size
    Global_data.use_nsa_gqa = True
    Global_data.use_grouped_sdpa_optimization = False
    Global_data.chunk_size = args.chunk_size
    Global_data.num_previous_chunks = args.num_previous_chunks

    if len(Global_data.top_k) == 1:
        Global_data.top_k = Global_data.top_k * 3
    elif len(Global_data.top_k) == 2:
        Global_data.top_k = Global_data.top_k + [Global_data.top_k[-1]]
    elif len(Global_data.top_k) > 3:
        Global_data.top_k = Global_data.top_k[:3]

    if args.meta_gist == 0:
        num_gist_level = 1
    else:
        num_gist_level = args.num_gist_level

    # Print configurations
    print(f"Batch size per device: {batch_size_per_device}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Meta gist: {meta_gist}")
    print(f"Selective training: {selective}")
    print(f"Top k for selective training: {Global_data.top_k}")   
    print(f"Top p for selective training: {Global_data.top_p}")
    print(f"Sink size for gist attention: {Global_data.sink_size}") 
    print(f"Grouped-union GQA enabled: {Global_data.use_nsa_gqa}")
    print(f"Chunk sizes for gist attention: {Global_data.chunk_size}")
    print(f"Number of previous chunks to attend to: {Global_data.num_previous_chunks}")
    ####################################

    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    global_model = GistLlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        dtype=torch.bfloat16,
        attn_implementation='sdpa',
        # use_flash_attention_2=True,
    )
    global_model.selective = selective

    if global_tokenizer.pad_token is None:
        global_tokenizer.pad_token = global_tokenizer.eos_token
        global_tokenizer.pad_token_id = global_tokenizer.eos_token_id
    global_tokenizer.padding_side = "left"
    
    # ==== BEGIN GIST CHANGES ====
    # Check if gist token has already been added to the model (e.g. because
    # we're resuming from a checkpoint.)
    if len(global_tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 3:
        assert (
            global_model.model.embed_tokens.weight.shape[0]
            == len(global_tokenizer)
        )
        assert global_model.lm_head.weight.shape[0] == len(global_tokenizer)
    else:
        # Initialize gist token
        global_tokenizer.add_special_tokens({"additional_special_tokens": ["<gist>", "<Gist>", "<MGist>"]})
        global_model.resize_token_embeddings(len(global_tokenizer))
        # Set new word embedding to average of existing word embeddings. For why,
        # see https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        if True:
            with torch.no_grad():
                # Get the average of existing embeddings
                avg_embedding = global_model.model.embed_tokens.weight[:-3].mean(0)
                avg_lm_head = global_model.lm_head.weight[:-3].mean(0)
                
                # Set both new tokens to the average
                global_model.model.embed_tokens.weight[-3] = avg_embedding  # <gist>
                global_model.model.embed_tokens.weight[-2] = avg_embedding  # <Gist>
                global_model.model.embed_tokens.weight[-1] = avg_embedding  # <Meta_Gist>
                global_model.lm_head.weight[-3] = avg_lm_head  # <gist>
                global_model.lm_head.weight[-2] = avg_lm_head  # <Gist>
                global_model.lm_head.weight[-1] = avg_lm_head  # <Meta_Gist>

    gist_token_ids = global_tokenizer.additional_special_tokens_ids[:num_gist_level]
    gist_token = global_tokenizer.additional_special_tokens[:num_gist_level]
    Global_data.gist_token_id = gist_token_ids
    Global_data.gist_token = gist_token
    Global_data.pad_token_id = global_tokenizer.pad_token_id

    preprocessor = gist_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        do_shuffle=True,
        chunk_size=Global_data.chunk_size,
        attention_sink_size=Global_data.sink_size,
        gist_token_ids=gist_token_ids,
        pad_token=Global_data.pad_token_id,
        meta_gist=meta_gist,
    )

    # train_dataset, eval_dataset = load_from_disk_then_process("text_multichunk", preprocessor)

    if meta_gist == 0:
        dataset_name = f"gist-sparse-attention/GSA-PT-Llama-3.2-1B-chunk{Global_data.chunk_size[0]}-data"
    elif meta_gist == 1:
        dataset_name = f"gist-sparse-attention/GSA-PT-Llama-3.2-1B-chunk{Global_data.chunk_size[0]}-chunk{Global_data.chunk_size[1]}-data"

    train_dataset = datasets.load_dataset("arrow", data_files={"train": f"hf://datasets/{dataset_name}/train/data-*.arrow"}, split="train")
    eval_dataset = datasets.load_dataset("arrow", data_files={"eval": f"hf://datasets/{dataset_name}/eval/data-*.arrow"}, split="eval")
    
    if meta_gist:
        out_dir = f"gist_llama-3.2-1b_meta{meta_gist}_selective{selective}_topp{args.top_p}_batch{batch_size_per_device}_chunk{Global_data.chunk_size[0]}_chunk{Global_data.chunk_size[1]}"
    else:
        out_dir = f"gist_llama-3.2-1b_meta{meta_gist}_selective{selective}_topp{args.top_p}_batch{batch_size_per_device}_chunk{Global_data.chunk_size[0]}"

    training_args = TrainingArguments(
        output_dir=f"{data_dir}/training_res/{out_dir}",
        report_to="wandb",
        run_name=out_dir,
        per_device_train_batch_size=batch_size_per_device,
        num_train_epochs=1,
        logging_dir="training_res/logs",
        logging_steps=10,
        save_steps=1000,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.05,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=1e-5,
        do_eval=True,
        per_device_eval_batch_size=batch_size_per_device,
        eval_strategy="steps",  # Add this line
        eval_steps=2000,
        gradient_checkpointing=True,
        save_total_limit=6,
        remove_unused_columns=False,
        eval_on_start=False,
        seed = 42
    )

    trainer = GistTrainer(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = custom_collate_chunkaug if meta_gist == 0 else meta_custom_collate_chunkaug,
    )

    trainer.train()

if __name__ == "__main__":
    main()
