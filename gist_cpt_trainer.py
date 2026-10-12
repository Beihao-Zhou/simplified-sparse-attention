"""Stage 1: continued pretraining of Llama-3.2-1B with gist tokens."""

import datasets

from gist_llama import GistLlamaForCausalLM
from gist_training import (
    GistTrainer,
    chunk_tag,
    collator,
    configure,
    load_model,
    make_parser,
    set_gist_tokens,
    training_arguments,
)
from gist_utils import add_gist_tokens


def main():
    args = make_parser().parse_args()
    num_gist_level = configure(args)

    tokenizer, model = load_model(GistLlamaForCausalLM, "meta-llama/Llama-3.2-1B", args.selective)
    gist_token_ids = add_gist_tokens(tokenizer, model, num_gist_level)
    set_gist_tokens(tokenizer, gist_token_ids)

    dataset_name = (
        f"gist-sparse-attention/GSA-PT-Llama-3.2-1B-{chunk_tag(args.meta_gist, '-')}-data"
    )
    train_dataset = datasets.load_dataset(
        "arrow", data_files={"train": f"hf://datasets/{dataset_name}/train/data-*.arrow"},
        split="train",
    )
    eval_dataset = datasets.load_dataset(
        "arrow", data_files={"eval": f"hf://datasets/{dataset_name}/eval/data-*.arrow"},
        split="eval",
    )

    run_name = (
        f"gist_llama-3.2-1b_meta{args.meta_gist}_selective{args.selective}_topp{args.top_p}"
        f"_batch{args.batch_size_per_device}_{chunk_tag(args.meta_gist)}"
    )
    training_args = training_arguments(
        args, run_name, global_batch_size=128,
        num_train_epochs=1,
        warmup_ratio=0.05,
        weight_decay=0.1,
        learning_rate=1e-5,
        eval_strategy="steps",
        eval_steps=2000,
        save_total_limit=6,
    )
    GistTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator(args.meta_gist),
    ).train()


if __name__ == "__main__":
    main()
