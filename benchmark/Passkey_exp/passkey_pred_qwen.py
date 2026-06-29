import warnings

warnings.filterwarnings("once")
import os
from datasets import load_dataset, load_from_disk
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor
)
from tqdm import tqdm
import numpy as np
import random
import argparse
import time
import sys
import gist_qwen2
from gist_qwen2 import GistQwen2ForCausalLM
from src.data import gist
from gist_utils import Global_data

import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Define the model path and the corresponding prompt template
MODEL_CONFIGS = {
    "qwen2-7b-instruct": dict(
        path="Qwen/Qwen2-7B-Instruct",
    ),
    "gsa-pt-qwen2-chunk8": dict(
        path="gist-sparse-attention/GSA-PT-Qwen2-7B-Instruct-chunk8",
    ),
    "gsa-pt-qwen2-chunk16": dict(
        path="gist-sparse-attention/GSA-PT-Qwen2-7B-Instruct-chunk16",
    ),
    "gsa-pt-qwen2-chunk32": dict(
        path="gist-sparse-attention/GSA-PT-Qwen2-7B-Instruct-chunk32",
    ),
    "gsa-pt-qwen2-chunk4-chunk4": dict(
        path="gist-sparse-attention/GSA-PT-Qwen2-7B-Instruct-chunk4-chunk4",
    ),
    "gsa-pt-qwen2-chunk8-chunk4": dict(
        path="gist-sparse-attention/GSA-PT-Qwen2-7B-Instruct-chunk8-chunk4",
    ),
    "gsa-ft-qwen2-chunk8": dict(
        path="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk8",
    ),
    "gsa-ft-qwen2-chunk16": dict(
        path="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk16",
    ),
    "gsa-ft-qwen2-chunk32": dict(
        path="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk32",
    ),
    "gsa-ft-qwen2-chunk4-chunk4": dict(
        path="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk4-chunk4",
    ),
    "gsa-ft-qwen2-chunk8-chunk4": dict(
        path="gist-sparse-attention/GSA-FT-Qwen2-7B-Instruct-chunk8-chunk4",
    ),
}


def generate_prompt_landmark(n_garbage, loc):
    """Generates a text file and inserts an passkey at a random position."""

    n_garbage_prefix = int(n_garbage * loc)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = (
        f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    )
    final_question = "What is the pass key? The pass key is"
    lines = [task_description, garbage_prefix, information_line, garbage_suffix]
    return "\n".join(lines), final_question, str(pass_key)


def passkey_retrieval_test(
    args,
    model,
    tokenizer,
    device,
    n_garbage=10000,
    loc=0.5,
    use_3_stages_gen=False,
):

    prompt, prompt_postfix, answer = generate_prompt_landmark(n_garbage, loc)
    prompt += prompt_postfix

    inputs = tokenizer([prompt])
    start = tokenizer(["There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."]).input_ids[0]
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:]  # drop BOS
    # answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, :]  # drop BOS
    old_context_length = len(inputs.input_ids[0])
    max_new_tokens = answer_ids.shape[-1]

    if args.meta_gist == 0:
        num_gist_level = 1
    else:
        num_gist_level = args.num_gist_level

    output_sequence = inputs.input_ids[0][:len(start)]

    # Process document chunks with GIST tokens
    if num_gist_level == 1:
        doc_with_gist = gist.insert_gist_tokens_word_aware(
            inputs.input_ids[0][len(start):-10],
            Global_data.gist_token_id,
            Global_data.chunk_size[0],
            tokenizer,
            Global_data.sink_size,
        )
    else:
        doc_with_gist = gist.insert_hierarchical_gist_tokens(
            inputs.input_ids[0][len(start):-10],
            Global_data.gist_token_id,
            Global_data.chunk_size[0],
            Global_data.chunk_size[1],
            tokenizer,
            Global_data.sink_size,
        )
    output_sequence.extend(doc_with_gist)
    output_sequence.extend(inputs.input_ids[0][-10:])
    input_ids = torch.tensor([output_sequence])
    attention_mask = torch.tensor([[1 for _ in output_sequence]])
    
    if num_gist_level > 1:
        gist_fn = gist.make_hierarchical_gist_mask_greedy
    else:
        gist_fn = gist.make_gist_mask
    attention_mask_gist = gist_fn(
        inputs=input_ids,
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=-1,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=tokenizer.pad_token_id,
        add_raw=Global_data.add_raw,
    )

    context_length = input_ids.shape[-1]

    inputs = {
        "input_ids": input_ids.to(model.device),
        "attention_mask": attention_mask.to(model.device),
        "attention_mask_gist": attention_mask_gist.to(model.device),
    }

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens + 1,
        do_sample=False,
        temperature=1e-9,
        pad_token_id=tokenizer.eos_token_id,
    )

    model_answer = output[0, -answer_ids.shape[-1] :].cpu()
    print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    is_correct = (model_answer == answer_ids[0]).all().item()
    print(
        f"context_length, {old_context_length}, The model output is '{tokenizer.decode(output[0, context_length:].cpu())}'. The model answer is '{tokenizer.decode(model_answer.cpu())}', is_correct : {is_correct}"
    )

    return is_correct, context_length


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="qwen-32",
    )
    ap.add_argument("--name", type=str, default="default")
    ap.add_argument("--num-tests", type=int, default=20)
    ap.add_argument(
        "--top_k",
        type=int,
        nargs="+",
        default=[3],
        help="Top-k values for selective training."
    )
    ap.add_argument(
        "--top_p",
        type=float,
        default=0,
        help="Top-p value for selective training (nucleus sampling)."
    )
    ap.add_argument("--sink_size", type=int, default=3, help="Sink size for gist attention.")
    ap.add_argument("--meta_gist", type=int, default=0, help="Use meta gist or not.")
    ap.add_argument(
        "--chunk_size",
        type=int,
        nargs="+",
        default=[8],
        help="Chunk size for gist attention per layer."
    )
    ap.add_argument("--num_gist_level", type=int, default=2, help="Number of levels of gist tokens.")
    ap.add_argument("--num_previous_chunks", type=int, default=1, help="Number of previous chunks to attend to.")
    ap.add_argument(
        "--selective",
        action="store_true",
        help="Enable selective training."
    )
    ap.add_argument("--sep_num", type=int, default=0, help="Number of separate tokens.")
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Enable adding raw input to the model inputs for loss computation."
    )
    ap.add_argument(
        "--klloss",
        action="store_true",
        help="Enable KL divergence loss between gist and raw input logits."
    )
    args = ap.parse_args()
    return args


def main():
    seed_everything(42)
    args = parse_args()

    # Define model config
    dev = torch.device("cuda")
    model_name: str = args.model
    path = MODEL_CONFIGS[model_name]["path"]
    num_tests = args.num_tests


    tokenizer = AutoTokenizer.from_pretrained(
        path, use_fast=False, local_files_only=True
    )
    
    model = GistQwen2ForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=dev,
    )
    model = model.eval()
    model.selective = args.selective
    meta_gist = args.meta_gist

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    if args.meta_gist == 0:
        num_gist_level = 1
    else:
        num_gist_level = args.num_gist_level

    if len(tokenizer) == gist_qwen2.PRETRAINED_VOCAB_SIZE + 3:
        assert (
            model.model.embed_tokens.weight.shape[0]
            == len(tokenizer)
        )
        assert model.lm_head.weight.shape[0] == len(tokenizer)

    else:
        print("ERROR! Not the Gist pretrained model!!")
        sys.exit(1)

    gist_token_ids = tokenizer.additional_special_tokens_ids[:num_gist_level]
    gist_token = tokenizer.additional_special_tokens[:num_gist_level]
    Global_data.gist_token_id = gist_token_ids
    Global_data.gist_token = gist_token
    Global_data.top_k = args.top_k
    Global_data.top_p = args.top_p
    Global_data.sink_size = args.sink_size
    Global_data.pad_token_id = tokenizer.pad_token_id
    Global_data.use_nsa_gqa = True
    Global_data.use_grouped_sdpa_optimization = False
    Global_data.chunk_size = args.chunk_size
    Global_data.num_previous_chunks = args.num_previous_chunks
    Global_data.sep_num = args.sep_num
    Global_data.add_raw = args.raw
    Global_data.kl_loss = args.klloss
    if args.klloss:
        Global_data.add_raw = True

    if len(Global_data.top_k) == 1:
        Global_data.top_k = Global_data.top_k * 3
    elif len(Global_data.top_k) == 2:
        Global_data.top_k = Global_data.top_k + [Global_data.top_k[-1]]
    elif len(Global_data.top_k) > 3:
        Global_data.top_k = Global_data.top_k[:3]


    # This is a rough ratio to control the number of texts and tokens
    N_GARBAGES = [230000]
    if "N_GARBAGES" in os.environ:
        N_GARBAGES = [int(os.environ["N_GARBAGES"])]
    for n_garbage in N_GARBAGES:
        # 38000 76000 114000
        passed_tests = 0
        for i in tqdm(range(num_tests)):
            loc = i / num_tests
            is_correct, len_tokens = passkey_retrieval_test(
                args,
                model,
                tokenizer,
                dev,
                n_garbage=n_garbage,
                loc=loc,
            )
            passed_tests += is_correct

        accuracy = float(passed_tests) / num_tests

        with open("./passkey.jsonl", "a") as fp:
            json.dump(
                {
                    "model": model_name,
                    "name": args.name,
                    "length": len_tokens,
                    "accuracy": accuracy,
                },
                fp,
            )
            fp.write("\n")


if __name__ == "__main__":
    main()
