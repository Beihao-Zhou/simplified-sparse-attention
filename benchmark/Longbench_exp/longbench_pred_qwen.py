import os
from datasets import load_dataset, load_from_disk
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)


class _SuppressGistTokens(LogitsProcessor):
    def __init__(self, gist_token_ids):
        self.ids = [int(g) for g in (gist_token_ids or [])]

    def __call__(self, input_ids, scores):
        if self.ids:
            scores[:, self.ids] = float("-inf")
        return scores
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
from chat import apply_chat_template

import os
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


DATASET2CATEGORY = {
    "narrativeqa": "EN Single-Doc QA",
    "qasper": "EN Single-Doc QA",
    "multifieldqa_en": "EN Single-Doc QA",
    "multifieldqa_zh": "CN Single-Doc QA",
    "hotpotqa": "EN Multi-Doc QA",
    "2wikimqa": "EN Multi-Doc QA",
    "musique": "EN Multi-Doc QA",
    "dureader": "CN Multi-Doc QA",
    "gov_report": "EN Summarization",
    "qmsum": "EN Summarization",
    "multi_news": "EN Summarization",
    "vcsum": "CN Summarization",
    "trec": "EN Few-Shot Learning",
    "triviaqa": "EN Few-Shot Learning",
    "samsum": "EN Few-Shot Learning",
    "lsht": "CN Few-Shot Learning",
    "passage_retrieval_en": "EN Synthetic Task",
    "passage_count": "EN Synthetic Task",
    "passage_retrieval_zh": "CN Synthetic Task",
    "lcc": "Code Completion",
    "repobench-p": "Code Completion",
}


def parse_args(cmd_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="qwen2-7b-instruct",
        choices=[
            "llama-3.2-1b",
            "qwen2-7b-instruct",
            "gsa-pt-qwen2-chunk8",
            "gsa-pt-qwen2-chunk16",
            "gsa-pt-qwen2-chunk32",
            "gsa-pt-qwen2-chunk4-chunk4",
            "gsa-pt-qwen2-chunk8-chunk4",
            "gsa-ft-qwen2-chunk8",
            "gsa-ft-qwen2-chunk16",
            "gsa-ft-qwen2-chunk32",
            "gsa-ft-qwen2-chunk4-chunk4",
            "gsa-ft-qwen2-chunk8-chunk4",
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


class CudaTimingCriteria:
    """A StoppingCriteria-shaped object that records cuda Events around
    each generate step. Subclasses StoppingCriteria's interface (callable);
    we don't import it to avoid an extra import for a duck-typed protocol.
    """

    def __init__(self):
        self.events = []
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append(e)

    def __call__(self, input_ids, scores, **kwargs):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append(e)
        return False  # never actually stop early

    def deltas_ms(self):
        if len(self.events) < 2:
            return []
        torch.cuda.synchronize()
        return [
            self.events[i - 1].elapsed_time(self.events[i])
            for i in range(1, len(self.events))
        ]

    def prefill_and_tpot_ms(self):
        ds = self.deltas_ms()
        if not ds:
            return 0.0, 0.0
        prefill = ds[0]
        tpot = (sum(ds[1:]) / len(ds[1:])) if len(ds) > 1 else 0.0
        return prefill, tpot


_BENCH_RUNS = {
    "candidate_label": "sparse",
    "prefill_ms": [],
    "ttft_ms": [],
    "tpot_ms": [],
    "correctness": True,
    "first_mismatch": None,
    "peak_vram_mb": 0.0,
}


def _set_attn_impl(impl: str):
    os.environ["ATTN_IMPL"] = impl


def _generate_one(model, gen_kwargs):
    from transformers import StoppingCriteriaList
    timer = CudaTimingCriteria()
    if os.environ.get("PROFILE_DECODE", "0") == "1":
        import torch.profiler as tp
        # warm up once (build kernels / compile) so the profile is steady-state
        with torch.no_grad():
            model.generate(**gen_kwargs, stopping_criteria=StoppingCriteriaList([CudaTimingCriteria()]))
        with tp.profile(activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA],
                        record_shapes=False) as prof:
            out = model.generate(
                **gen_kwargs,
                stopping_criteria=StoppingCriteriaList([timer]),
            )[0]
        ka = prof.key_averages()
        print("\n===== PROFILE_DECODE: top 30 by self CUDA time =====")
        print(ka.table(sort_by="self_cuda_time_total", row_limit=30))
        print("\n===== PROFILE_DECODE: top 30 by self CPU time (launch overhead) =====")
        print(ka.table(sort_by="self_cpu_time_total", row_limit=30))
        print("\n===== PROFILE_DECODE: top 30 by count (op launches) =====")
        print(ka.table(sort_by="count", row_limit=30))
        prefill, tpot = timer.prefill_and_tpot_ms()
        return out, prefill, tpot
    out = model.generate(
        **gen_kwargs,
        stopping_criteria=StoppingCriteriaList([timer]),
    )[0]
    prefill, tpot = timer.prefill_and_tpot_ms()
    return out, prefill, tpot


def _build_gen_kwargs(input_dict, dataset, cur_max_gen, tokenizer, context_length):
    base = dict(
        **input_dict,
        max_new_tokens=cur_max_gen,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
        logits_processor=LogitsProcessorList([_SuppressGistTokens(Global_data.gist_token_id)]),
    )
    if dataset in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]:
        base["min_length"] = context_length + 1
        base["eos_token_id"] = [
            tokenizer.eos_token_id,
            tokenizer.encode("\n", add_special_tokens=False)[-1],
        ]
    return base


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
    max_samples = int(os.environ.get("MAX_SAMPLES", "5"))
    start_idx = int(os.environ.get("START_IDX", "0"))
    candidate = os.environ.get("ATTN_IMPL", "sparse")
    _BENCH_RUNS["candidate_label"] = candidate

    start = start_idx
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            start = len(list(f))
    if start >= len(data):
        return

    data_ = data[start:start + max_samples]
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
        
        
        # in fewshot learning and code completion we do not need chat template
        if "ft-" in args.ckpt_path.lower() and not any(x in DATASET2CATEGORY[dataset] for x in ["Few-Shot Learning", "Code Completion"]):
            encoded = apply_chat_template(
                "hf", 
                messages=[{'role': 'user', 'content': prompt}],
                tokenizer=tokenizer,
                add_generation_prompt=True,
            ).encoded
        else:
            encoded = tokenizer(prompt)
        
        if "gist" not in args.ckpt_path.lower():
            input = {'input_ids': None, 'attention_mask': None}
            input["input_ids"] = torch.tensor([encoded["input_ids"]], device = model.device)
            input["attention_mask"] = torch.tensor([encoded["attention_mask"]], device = model.device)

            # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        else:
            output_sequence = []
            input_ids = encoded["input_ids"]
            after_brace = prompt_format.split("}", 1)[1]
            after_brace = after_brace.format(**json_obj)
            real_prompt = tokenizer.decode(input_ids)
            char_pos = real_prompt.find(after_brace)
            assert char_pos != -1
            prompt_left = real_prompt[char_pos:]
            prompt_ids = tokenizer(prompt_left, add_special_tokens=False).input_ids
            input_ids = input_ids[:-len(prompt_ids)]

            # Optional: repeat the document to inflate context length for
            # large-k_len benchmarking (env CONTEXT_REPEAT, default 1 = off).
            _crep = int(os.environ.get("CONTEXT_REPEAT", "1"))
            if _crep > 1:
                input_ids = list(input_ids) * _crep

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
                    Global_data.sink_size,
                )
            else:
                doc_with_gist = gist.insert_hierarchical_gist_tokens(
                    input_ids,
                    Global_data.gist_token_id,
                    Global_data.chunk_size[0],
                    Global_data.chunk_size[1],
                    tokenizer,
                    Global_data.sink_size,
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
                pad_token=tokenizer.pad_token_id,
                add_raw=Global_data.add_raw,
            ).to(model.device)

            input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "attention_mask_gist": attention_mask_gist,
            }

        context_length = input["input_ids"].shape[-1]
        cur_max_gen = max_gen

        print(f"Context length: {context_length}, Max generation length: {cur_max_gen}")

        gen_kwargs = _build_gen_kwargs(input, dataset, cur_max_gen, tokenizer, context_length)

        torch.cuda.reset_peak_memory_stats()

        _set_attn_impl(candidate)
        torch.manual_seed(0)
        output, prefill_ms, tpot_ms = _generate_one(model, gen_kwargs)
        _BENCH_RUNS["correctness"] = None
        _BENCH_RUNS["token_match"] = None

        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        _BENCH_RUNS["peak_vram_mb"] = max(_BENCH_RUNS["peak_vram_mb"], peak)
        _BENCH_RUNS["prefill_ms"].append(prefill_ms)
        _BENCH_RUNS["ttft_ms"].append(prefill_ms)
        _BENCH_RUNS["tpot_ms"].append(tpot_ms)

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


def load_model_and_tokenizer(path, device, args):
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    if "gist" not in path:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model = model.eval()

    else:
        model = GistQwen2ForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model = model.eval()
        model.selective = args.selective

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
    
    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("benchmark/Longbench_exp/longbench_config/model2path.json", "r"))
    model2maxlen = json.load(open("benchmark/Longbench_exp/longbench_config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    model_path = args.ckpt_path if args.ckpt_path is not None else model2path[model_name]
    # define your model
    max_length = model2maxlen[model_name]
    model, tokenizer = load_model_and_tokenizer(
        model_path, device, args
    )

    datasets = args.datasets
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("benchmark/Longbench_exp/longbench_config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/Longbench_exp/longbench_config/dataset2maxlen.json", "r"))
    # predict on each dataset
    for dataset in datasets:
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            os.makedirs(f"pred_e/{model_name}/{args.name}", exist_ok=True)
            out_path = f"pred_e/{model_name}/{args.name}/{dataset}.jsonl"
        else:
            data = load_from_disk(f"benchmark/Longbench_exp/LongBench/{dataset}")
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

    # ---- Final metrics block (parsed by run_longbench.sh extractor) ----
    runs = _BENCH_RUNS
    n = len(runs["prefill_ms"])
    avg_prefill = sum(runs["prefill_ms"]) / n if n else 0.0
    avg_ttft    = sum(runs["ttft_ms"])    / n if n else 0.0
    avg_tpot    = sum(runs["tpot_ms"])    / n if n else 0.0
    print("---")
    print(f"candidate:        {runs['candidate_label']}")
    print(f"prefill_ms:       {avg_prefill:.4f}")
    print(f"ttft_ms:          {avg_ttft:.4f}")
    print(f"tpot_ms:          {avg_tpot:.4f}")
    _sel = os.environ.get("DECODE_SELECT", "head")
    if runs["candidate_label"] == "ref":
        _corr = "n/a (eager reference run)"
    elif _sel == "head":
        _corr = "head selection = allclose-exact vs eager (run check_correctness.sh to verify)"
    else:
        _corr = f"{_sel} selection = different algorithm, F1-validated not allclose"
    print(f"correctness:      not checked per-run; {_corr}")
    print(f"peak_vram_mb:     {runs['peak_vram_mb']:.1f}")
    if n >= 2:
        import statistics as _st
        ssp = runs["prefill_ms"][1:]; sst = runs["tpot_ms"][1:]  # exclude cold sample 0
        print(f"prefill_ms_median: {_st.median(runs['prefill_ms']):.4f}")
        print(f"tpot_ms_median:    {_st.median(runs['tpot_ms']):.4f}")
        print(f"prefill_ms_steady: {sum(ssp)/len(ssp):.4f}  (mean excl. cold sample 0)")
        print(f"tpot_ms_steady:    {sum(sst)/len(sst):.4f}  (mean excl. cold sample 0)")
    print(f"per_sample_prefill_ms: {[round(x,4) for x in runs['prefill_ms']]}")
    print(f"per_sample_tpot_ms:    {[round(x,4) for x in runs['tpot_ms']]}")
    if runs["first_mismatch"] is not None:
        print(f"first_mismatch:   {runs['first_mismatch']}")
