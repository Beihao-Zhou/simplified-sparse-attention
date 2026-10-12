"""Long-context training of Qwen2: continued pretraining, then selective finetuning (--ft)."""

import glob
import os
import re

import datasets

from gist_qwen2 import GistQwen2ForCausalLM
from gist_training import (
    GistTrainer,
    chunk_tag,
    collator,
    configure,
    data_dir,
    load_model,
    make_parser,
    set_gist_tokens,
    training_arguments,
)
from gist_utils import StrideGroupedSampler, add_gist_tokens


class StrideGroupedTrainer(GistTrainer):
    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        lengths = train_dataset["length"] if isinstance(train_dataset, datasets.Dataset) else None
        tokenizer = self.tokenizer
        return StrideGroupedSampler(
            batch_size=self.args.train_batch_size * self.args.world_size,
            group="relaxed",
            sort=None,
            dataset=train_dataset,
            lengths=lengths,
            model_input_name=tokenizer.model_input_names[0] if tokenizer is not None else None,
        )


def get_latest_ckpt(base_dir, pattern="checkpoint-11*"):
    ckpts = [p for p in glob.glob(os.path.join(base_dir, pattern)) if os.path.isdir(p)]
    if not ckpts:
        raise ValueError(f"No checkpoints found in {base_dir} with pattern {pattern}")
    return max(ckpts, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))


def main():
    parser = make_parser(batch_size_per_device=1)
    parser.add_argument("--deepspeed", type=str, default="stage2.json")
    parser.add_argument("--ft", action="store_true")
    args = parser.parse_args()

    num_gist_level = configure(args)

    model_tag = "qwen2-7b"
    stage1_name = (f"long_gist_{model_tag}_meta{args.meta_gist}_selectiveFalse"
                   f"_batch{args.batch_size_per_device}_{chunk_tag(args.meta_gist)}")
    if args.ft:
        input_path = get_latest_ckpt(f"{data_dir}/training_res/{stage1_name}")
        print(input_path)
        run_name = (f"ft-long_gist_{model_tag}_meta{args.meta_gist}_selective{args.selective}"
                    f"_batch{args.batch_size_per_device}_{chunk_tag(args.meta_gist)}")
    else:
        input_path = "Qwen/Qwen2-7B-Instruct"
        run_name = stage1_name

    tokenizer, model = load_model(GistQwen2ForCausalLM, input_path, args.selective)
    gist_token_ids = add_gist_tokens(tokenizer, model, num_gist_level)
    set_gist_tokens(tokenizer, gist_token_ids)

    stage = "FT" if args.ft else "PT"
    dataset_name = (f"gist-sparse-attention/GSA-{stage}-Qwen2-7B-Instruct-"
                    f"{chunk_tag(args.meta_gist, '-')}-data")
    train_dataset = datasets.load_dataset(
        "arrow", data_files={"train": f"hf://datasets/{dataset_name}/data-*.arrow"},
        split="train",
    )

    training_args = training_arguments(
        args, run_name, global_batch_size=8,
        num_train_epochs=1,
        warmup_ratio=0.05,
        weight_decay=0.1,
        learning_rate=5e-6 if args.ft else 1e-5,
        save_total_limit=6,
        deepspeed=args.deepspeed,
    )
    StrideGroupedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator(args.meta_gist),
    ).train()


if __name__ == "__main__":
    main()
