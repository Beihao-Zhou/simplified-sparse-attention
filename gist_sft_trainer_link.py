"""Stage 2 for RAG: selective finetuning with <link> tokens after each document."""

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
from gist_utils import Global_data, add_link_tokens, check_gist_model, link_token_names


def main():
    parser = add_sft_args(make_parser())
    parser.add_argument("--link_num", type=int, default=5, help="Number of link tokens.")
    args = parser.parse_args()
    num_gist_level = configure(args)

    print(f"Starting model checkpoint: {args.start_model}")
    tokenizer, model = load_model(GistLlamaForCausalLM, args.start_model, args.selective)
    gist_token_ids = check_gist_model(tokenizer, model, num_gist_level)
    set_gist_tokens(tokenizer, gist_token_ids)
    Global_data.link_token_id = add_link_tokens(tokenizer, model, args.link_num)
    Global_data.link_token = link_token_names(args.link_num)

    train_dataset, eval_dataset = load_qa(tokenizer, gist_token_ids, args.meta_gist, link=True)
    Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments(args, args.output_name, global_batch_size=64, **SFT_SCHEDULE),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator(args.meta_gist, link=True),
    ).train()


if __name__ == "__main__":
    main()
