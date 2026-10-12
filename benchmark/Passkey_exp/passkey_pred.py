import argparse
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from gist_llama import GistLlamaForCausalLM
from gist_qwen2 import GistQwen2ForCausalLM
from gist_utils import (
    Global_data,
    add_gist_args,
    apply_gist_args,
    check_gist_model,
    gist_token_names,
)
from src.data import gist

warnings.filterwarnings("once")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Keyed by the checkpoint's config.model_type.
GIST_MODELS = {"llama": GistLlamaForCausalLM, "qwen2": GistQwen2ForCausalLM}

# Define the model path and the corresponding prompt template
MODEL_CONFIGS = {
    "llama-3.2-1b": dict(
        path="meta-llama/Llama-3.2-1B-Instruct",
    ),
    "gsa-pt-llama3.2-chunk8": dict(
        path="gist-sparse-attention/GSA-PT-Llama-3.2-1B-chunk8",
    ),
    "gsa-pt-llama3.2-chunk16": dict(
        path="gist-sparse-attention/GSA-PT-Llama-3.2-1B-chunk16",
    ),
    "gsa-pt-llama3.2-chunk4-chunk4": dict(
        path="gist-sparse-attention/GSA-PT-Llama-3.2-1B-chunk4-chunk4",
    ),
    "gsa-ft-llama3.2-chunk8": dict(
        path="gist-sparse-attention/GSA-FT-Llama-3.2-1B-chunk8",
    ),
    "gsa-ft-llama3.2-chunk16": dict(
        path="gist-sparse-attention/GSA-FT-Llama-3.2-1B-chunk16",
    ),
    "gsa-ft-llama3.2-chunk4-chunk4": dict(
        path="gist-sparse-attention/GSA-FT-Llama-3.2-1B-chunk4-chunk4",
    ),
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

    num_gist_level = len(Global_data.gist_token_id)
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
    ap.add_argument("--model", type=str, required=True, choices=MODEL_CONFIGS)
    ap.add_argument("--name", type=str, default="default")
    ap.add_argument("--num-tests", type=int, default=20)
    ap.add_argument("--sep_num", type=int, default=0, help="Number of separate tokens.")
    add_gist_args(ap)
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

    model = GIST_MODELS[AutoConfig.from_pretrained(path).model_type].from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map=dev,
    )
    model = model.eval()
    model.selective = args.selective

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    num_gist_level = apply_gist_args(args)
    try:
        Global_data.gist_token_id = check_gist_model(tokenizer, model, num_gist_level)
    except ValueError as e:
        print(f"ERROR! Not the Gist pretrained model!! {e}")
        sys.exit(1)
    Global_data.gist_token = gist_token_names(num_gist_level)
    Global_data.pad_token_id = tokenizer.pad_token_id
    Global_data.sep_num = args.sep_num

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
