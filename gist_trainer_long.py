"""Long-context training of Qwen2: continued pretraining, then selective finetuning (--ft)."""

import glob
import os
import re

import datasets
from transformers import AutoConfig

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
from gist_utils import Global_data, StrideGroupedSampler, add_gist_tokens

# <gist>, <Gist>, <MGist> in the released GSA-*-Qwen2-7B-Instruct data (Qwen2 tokenizer).
RELEASED_DATA_GIST_IDS = [151646, 151647, 151648]


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
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Load a local dataset (Dataset.save_to_disk) instead of the released HF data, "
             "e.g. one produced by data_process/convert_cpt_to_stream.py.",
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2-7B-Instruct",
        help="HF model to start continued pretraining from (ignored with --ft, which "
             "resumes from the latest stage-1 checkpoint). Must be a Qwen2-architecture "
             "model, e.g. Qwen/Qwen2.5-7B-Instruct-1M.",
    )
    parser.add_argument(
        "--raw_window",
        type=int,
        default=0,
        help="Raw tokens within this many positions stay exactly visible (0 = off). "
             "Use with streaming-gist data, where gists also cover the generated text.",
    )
    parser.add_argument(
        "--train_attn",
        choices=["dense", "flex"],
        default="dense",
        help="dense: materialize [B,1,T,T] gist masks (original path, ~20K max). "
             "flex: FlexAttention with O(T) masks (gist_flex_attention.py), for long "
             "sequences; non-selective training only.",
    )
    args = parser.parse_args()

    Global_data.train_attn = args.train_attn
    Global_data.raw_window = args.raw_window
    if args.train_attn == "flex":
        if args.selective:
            raise ValueError("--train_attn flex does not support --selective yet")
        import gist_flex_attention

        attn_implementation = gist_flex_attention.GIST_FLEX
    else:
        attn_implementation = "sdpa"
    num_gist_level = configure(args)

    model_tag = ("qwen2-7b" if args.base_model == "Qwen/Qwen2-7B-Instruct"
                 else args.base_model.split("/")[-1].lower())
    stage1_name = (f"long_gist_{model_tag}_meta{args.meta_gist}_selectiveFalse"
                   f"_batch{args.batch_size_per_device}_{chunk_tag(args.meta_gist)}")
    if args.ft:
        input_path = get_latest_ckpt(f"{data_dir}/training_res/{stage1_name}")
        print(input_path)
        run_name = (f"ft-long_gist_{model_tag}_meta{args.meta_gist}_selective{args.selective}"
                    f"_batch{args.batch_size_per_device}_{chunk_tag(args.meta_gist)}")
    else:
        input_path = args.base_model
        run_name = stage1_name

    model_type = AutoConfig.from_pretrained(input_path).model_type
    if model_type != "qwen2":
        raise NotImplementedError(
            f"{input_path} is {model_type!r}; GistQwen2 needs a Qwen2-architecture model "
            "(Qwen3 adds q/k norms and needs its own gist port)")
    tokenizer, model = load_model(
        GistQwen2ForCausalLM, input_path, args.selective, attn_implementation
    )
    gist_token_ids = add_gist_tokens(tokenizer, model, num_gist_level)
    set_gist_tokens(tokenizer, gist_token_ids)

    if args.dataset_path:
        train_dataset = datasets.load_from_disk(args.dataset_path)
    else:
        if num_gist_level > 2 or gist_token_ids != RELEASED_DATA_GIST_IDS[:num_gist_level]:
            # Released data: <= 2 levels, Qwen2 gist ids (other tokens in Qwen2.5/Qwen3).
            raise ValueError(
                f"{num_gist_level}-level gists with ids {gist_token_ids} do not match the "
                f"released data (<= 2 levels, ids {RELEASED_DATA_GIST_IDS[:2]}); pass a "
                "converted --dataset_path")
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
