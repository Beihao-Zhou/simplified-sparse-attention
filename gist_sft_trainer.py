"""Stage 2: selective finetuning of a gist-pretrained Llama on QA."""

from transformers import Trainer

from gist_llama import GistLlamaForCausalLM
from gist_training import (
    SFT_SCHEDULE,
    add_sft_args,
    collator,
    configure,
    load_model,
    load_qa,
    make_parser,
    set_gist_tokens,
    training_arguments,
)
from gist_utils import add_gist_tokens


def main():
    args = add_sft_args(make_parser()).parse_args()
    num_gist_level = configure(args)

    print(f"Starting model checkpoint: {args.start_model}")
    tokenizer, model = load_model(GistLlamaForCausalLM, args.start_model, args.selective)
    # Warns and adds fresh gist tokens if start_model is not gist-pretrained.
    gist_token_ids = add_gist_tokens(tokenizer, model, num_gist_level)
    set_gist_tokens(tokenizer, gist_token_ids)

    train_dataset, eval_dataset = load_qa(tokenizer, gist_token_ids, args.meta_gist)
    Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments(args, args.output_name, global_batch_size=64, **SFT_SCHEDULE),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator(args.meta_gist),
    ).train()


if __name__ == "__main__":
    main()
