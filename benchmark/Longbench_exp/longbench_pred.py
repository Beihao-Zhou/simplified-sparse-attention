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
from gist_llama import GistLlamaForCausalLM
import gist_llama
from src.data import gist
from gist_utils import Global_data

import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def make_chunk_aug_mask(*args, **kwargs):
    raise NotImplementedError(
        "The legacy append-format LongBench path depends on src.data.attention_mask, "
        "which is not part of this release. Use the gist/Qwen LongBench path instead."
    )


def parse_args(cmd_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="llama-3.2-1b",
        choices=[
            "llama-3.2-1b",
        ],
    )
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="The path to the `checkpoint.pt` file.",
    )
    ap.add_argument("--name", type=str, default="default")
    ap.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "qasper",
            "narrativeqa",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ],
    )
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
    args = ap.parse_args(cmd_args)
    args.e = False
    return args


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    out_path,
):
    start = 0
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            start = len(list(f))
    if start >= len(data):
        return
    print(out_path, start)

    data_ = data[start:]
    for json_obj in tqdm(data_):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if "gist" not in model_name and "append" not in model_name:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        elif "append" in model_name:
            pad_token = 128004
            chunk_size = 100 * (args.chunk_size[0] // 4)
            chunk_end_token = 128253
            global_start_token = 128254
            global_end_token = 128255
            compress_tokens = list(range(128011, 128036))

            input_ids = []
            segment_ids_1 = []
            segment_ids_2 = []
            position_ids = []
            chunk_index_ids = []

            all_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            sys_prompt = prompt_format.split("\n", 1)[0]
            after_brace = prompt_format.split("}", 1)[1]
            sys_prompt_ids = tokenizer(sys_prompt, add_special_tokens=False).input_ids
            after_brace = after_brace.format(**json_obj)
            prompt_ids = tokenizer(after_brace, add_special_tokens=False).input_ids
            all_ids = all_ids[:-len(prompt_ids)]

            sys_ids = tokenizer("<|begin_of_text|>", add_special_tokens=False).input_ids + [global_start_token]
            sys_len = len(sys_ids)

            input_ids.extend(sys_ids)
            segment_ids_1.extend([0] * sys_len)
            segment_ids_2.extend([3] * sys_len)
            position_ids.extend(list(range(sys_len)))
            chunk_index_ids.extend([-1] * sys_len)

            current_index = sys_len

            tem_id = all_ids + [chunk_end_token]
            chunk_idx = 1
            for idx in range(0, len(tem_id), chunk_size):
                chunk_id = tem_id[idx : idx + chunk_size]
                chunk_len = len(chunk_id)
                
                segment_ids_1.extend([chunk_idx] * (chunk_len + 1 + len(compress_tokens)))
                segment_ids_2.extend([1] * (chunk_len + 1) + [2] * len(compress_tokens))
                chunk_index_ids.extend([0] * (chunk_len + 1 + len(compress_tokens)))
                position_ids.extend(list(range(current_index - chunk_len - 1, current_index + len(compress_tokens))))
                input_ids.extend(chunk_id + [chunk_end_token] + compress_tokens)

                current_index += len(compress_tokens)
                chunk_idx += 1
            
            user_id = [global_end_token] + prompt_ids
            user_len = len(user_id)
            segment_ids_1.extend([0] * user_len)
            segment_ids_2.extend([3] * user_len)
            chunk_index_ids.extend([-1] * user_len)
            position_ids.extend(list(range(current_index, current_index + user_len)))
            input_ids.extend(user_id)
            current_index += user_len

            attention_mask = torch.tensor([[1 for _ in input_ids]], device = model.device)
            input_ids = torch.tensor([input_ids], device = model.device)
            position_ids = torch.tensor([position_ids], device=model.device)
            segment_ids_1 = torch.tensor([segment_ids_1])
            segment_ids_2 = torch.tensor([segment_ids_2])
            chunk_index_ids = torch.tensor([chunk_index_ids])

            attention_mask_gist = make_chunk_aug_mask(
                source_seg1=segment_ids_1,
                target_seg1=segment_ids_1,
                source_seg2=segment_ids_2,
                target_seg2=segment_ids_2,
                source_chunk=chunk_index_ids,
                target_chunk=chunk_index_ids,
                dtype=torch.bfloat16,
                add_causal_lm_mask=True
            ).unsqueeze(1).to(model.device)

            attention_mask_gist.masked_fill_(attention_mask_gist > -1, 1)
            attention_mask_gist.masked_fill_(attention_mask_gist < -1, 0)

            input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "attention_mask_gist": attention_mask_gist,
            }

        else:
            output_sequence = [tokenizer.bos_token_id]
            input_ids = tokenizer(prompt, add_special_tokens=False).input_ids

            sys_prompt = prompt_format.split("\n", 1)[0]
            after_brace = prompt_format.split("}", 1)[1]
            sys_prompt_ids = tokenizer(sys_prompt, add_special_tokens=False).input_ids
            after_brace = after_brace.format(**json_obj)
            prompt_ids = tokenizer(after_brace, add_special_tokens=False).input_ids
            input_ids = input_ids[:-len(prompt_ids)]

            if args.meta_gist == 0:
                num_gist_level = 1
            else:
                num_gist_level = args.num_gist_level

            # Process document chunks with GIST tokens
            if num_gist_level == 1:
                doc_with_gist = gist.insert_gist_tokens_word_aware(
                    input_ids,
                    Global_data.gist_token_id,
                    Global_data.chunk_size[0],
                    tokenizer,
                    Global_data.sink_size - 1,
                )
            else:
                doc_with_gist = gist.insert_hierarchical_gist_tokens(
                    input_ids,
                    Global_data.gist_token_id,
                    Global_data.chunk_size[0],
                    Global_data.chunk_size[1],
                    tokenizer,
                    Global_data.sink_size - 1,
                )
            output_sequence.extend(doc_with_gist)
            output_sequence.extend(prompt_ids)
            input_ids = torch.tensor([output_sequence], device = model.device)
            attention_mask = torch.tensor([[1 for _ in output_sequence]], device = model.device)
            
            if num_gist_level > 1:
                gist_fn = gist.make_hierarchical_gist_mask_greedy
            else:
                gist_fn = gist.make_gist_mask
            attention_mask_gist = gist_fn(
                inputs=input_ids,
                gist_token=Global_data.gist_token_id,
                chunk_size=Global_data.chunk_size[0],
                attention_sink_size=Global_data.sink_size,
                num_previous_chunks=Global_data.num_previous_chunks,
                pad_token=128001,
                add_raw=Global_data.add_raw,
            ).to(model.device)

            # #######################
            if Global_data.add_raw:
                attention_mask_gist = attention_mask_gist[0]

            # #######################

            input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "attention_mask_gist": attention_mask_gist,
            }

        context_length = input["input_ids"].shape[-1]
        cur_max_gen = max_gen

        print(f"Context length: {context_length}, Max generation length: {cur_max_gen}")

        from transformers import StoppingCriteria, StoppingCriteriaList
        import time
        class TimingCriteria(StoppingCriteria):
            def __init__(self):
                self.timings = [time.time()]

            def __call__(self, input_ids, scores, **kwargs):
                self.timings.append(time.time())
                return False  # Never actually stop early


        timer = TimingCriteria()

        if (
            dataset in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=cur_max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([timer])
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=cur_max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([timer])
            )[0]

        # token_latencies = [t2 - t1 for t1, t2 in zip(timer.timings, timer.timings[1:])]
        # print(f"Prefill latencies: {token_latencies[0]+token_latencies[1]}, Decode latencies: {np.mean(token_latencies[2:])}")

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "length": json_obj["length"],
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    if "gist" not in model_name and "append" not in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model = model.eval()

    else:
        model = GistLlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
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

        if len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 3:
            assert (
                model.model.embed_tokens.weight.shape[0]
                == len(tokenizer)
            )
            assert model.lm_head.weight.shape[0] == len(tokenizer)

        else:
            print("WARNNING! Not the Gist pretrained model!!")

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
    
    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("longbench_config/model2path.json", "r"))
    model2maxlen = json.load(open("longbench_config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    model_path = args.ckpt_path if args.ckpt_path is not None else model2path[model_name]
    # define your model
    max_length = model2maxlen[model_name]
    model, tokenizer = load_model_and_tokenizer(
        model_path, model_name, device, args
    )

    datasets = args.datasets
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("longbench_config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench_config/dataset2maxlen.json", "r"))
    # predict on each dataset
    for dataset in datasets:
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            os.makedirs(f"pred_e/{model_name}/{args.name}", exist_ok=True)
            out_path = f"pred_e/{model_name}/{args.name}/{dataset}.jsonl"
        else:
            data = load_from_disk(f"LongBench/{dataset}")
            os.makedirs(f"pred/{model_name}/{args.name}", exist_ok=True)
            out_path = f"pred/{model_name}/{args.name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        get_pred(
            model,
            tokenizer,
            list(data),
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            out_path,
        )
