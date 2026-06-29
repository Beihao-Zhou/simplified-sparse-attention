import os
import glob
import re
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
from transformers.trainer import is_datasets_available

from gist_utils import Global_data, StrideGroupedSampler
import gist_qwen2
from gist_qwen2 import GistQwen2ForCausalLM

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


    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset

        group_by_stride = "relaxed"
        
        # Build the sampler.
        if group_by_stride is not None:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = train_dataset['length']
            else:
                lengths = None

            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None

            return StrideGroupedSampler(
                batch_size=self.args.train_batch_size * self.args.world_size,
                group=group_by_stride,
                sort=None,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return super()._get_train_sampler(train_dataset)


def main():

    parser = argparse.ArgumentParser(description="Gist on Long Context Training")
    parser.add_argument("--meta_gist", type=int, default=0, help="Use meta gist or not.")
    parser.add_argument("--batch_size_per_device", type=int, default=1, help="Batch size.")
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
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="stage2.json"
    )
    parser.add_argument(
        "--ft",
        action="store_true"
    )
    
    args = parser.parse_args()

    ####################################
    # Configurations
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    batch_size_per_device = args.batch_size_per_device
    gradient_accumulation_steps = 8 // num_gpus // batch_size_per_device
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


    def get_latest_ckpt(base_dir, pattern="checkpoint-11*"):
        ckpts = glob.glob(os.path.join(base_dir, pattern))
        ckpts = [p for p in ckpts if os.path.isdir(p)]
        if not ckpts:
            raise ValueError(f"No checkpoints found in {base_dir} with pattern {pattern}")

        def extract_step(p):
            m = re.search(r"checkpoint-(\d+)", p)
            return int(m.group(1)) if m else -1

        return max(ckpts, key=extract_step)


    if args.ft:
        if args.meta_gist:
            base_dir = f"{data_dir}/training_res/long_gist_qwen2-7b_meta{meta_gist}_selectiveFalse_batch{batch_size_per_device}_chunk{Global_data.chunk_size[0]}_chunk{Global_data.chunk_size[1]}"
            input_path = get_latest_ckpt(base_dir)
            print(input_path)

            out_path = f"ft-long_gist_qwen2-7b_meta{meta_gist}_selective{selective}_batch{batch_size_per_device}_chunk{Global_data.chunk_size[0]}_chunk{Global_data.chunk_size[1]}"
        else:
            base_dir = f"{data_dir}/training_res/long_gist_qwen2-7b_meta{meta_gist}_selectiveFalse_batch{batch_size_per_device}_chunk{Global_data.chunk_size[0]}"
            input_path = get_latest_ckpt(base_dir)
            print(input_path)

            out_path = f"ft-long_gist_qwen2-7b_meta{meta_gist}_selective{selective}_batch{batch_size_per_device}_chunk{Global_data.chunk_size[0]}"
    else:
        input_path = "Qwen/Qwen2-7B-Instruct"
        if args.meta_gist:
            out_path = f"long_gist_qwen2-7b_meta{meta_gist}_selectiveFalse_batch{batch_size_per_device}_chunk{Global_data.chunk_size[0]}_chunk{Global_data.chunk_size[1]}"
        else:
            out_path = f"long_gist_qwen2-7b_meta{meta_gist}_selectiveFalse_batch{batch_size_per_device}_chunk{Global_data.chunk_size[0]}"

    global_tokenizer = AutoTokenizer.from_pretrained(input_path)
    global_model = GistQwen2ForCausalLM.from_pretrained(
        input_path,
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
    if len(global_tokenizer) == gist_qwen2.PRETRAINED_VOCAB_SIZE + 3:
        assert (
            global_model.model.embed_tokens.weight.shape[0]
            == len(global_tokenizer)
        )
        assert global_model.lm_head.weight.shape[0] == len(global_tokenizer)
    else:
        print("ATTENTION! Not the Gist pretrained model!!")
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

    if args.ft == False:
        if args.meta_gist:
            dataset_name = f"gist-sparse-attention/GSA-PT-Qwen2-7B-Instruct-chunk{Global_data.chunk_size[0]}-chunk{Global_data.chunk_size[1]}-data"
        else:
            dataset_name = f"gist-sparse-attention/GSA-PT-Qwen2-7B-Instruct-chunk{Global_data.chunk_size[0]}-data"
    else:
        if args.meta_gist:
            dataset_name = f"gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk{Global_data.chunk_size[0]}-chunk{Global_data.chunk_size[1]}-data"
        else:
            dataset_name = f"gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk{Global_data.chunk_size[0]}-data"
    train_dataset = datasets.load_dataset("arrow", data_files={"train": f"hf://datasets/{dataset_name}/data-*.arrow"}, split="train")
    eval_dataset = None
    
    training_args = TrainingArguments(
        output_dir=f"{data_dir}/training_res/{out_path}",
        report_to="wandb",
        run_name=out_path,
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
        learning_rate=5e-6 if args.ft else 1e-5,
        do_eval=True,
        per_device_eval_batch_size=batch_size_per_device,
        gradient_checkpointing=True,
        save_total_limit=6,
        remove_unused_columns=False,
        eval_on_start=False,
        deepspeed=args.deepspeed,
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
