"""Multi-document RAG evaluation (best-subspan EM).

Modes: `base` runs a plain HF model on the raw documents; `gist` compresses the system prompt
and documents with gist tokens; `link` compresses the documents and appends link tokens after
each one (checkpoints from gist_sft_trainer_link.py).
"""

import argparse
import json
import os
import string
import sys
from dataclasses import dataclass
from operator import itemgetter
from typing import Callable, Dict, List, Optional

import regex
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from gist_llama import GistLlamaForCausalLM
from gist_utils import (
    Global_data,
    add_gist_args,
    apply_gist_args,
    check_gist_model,
    find_link_token_ids,
    gist_token_names,
    link_token_names,
)
from src.data import gist

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "You are a intelligent AI assistant. Please answer questions based on the user's instruction. "
    "Below are some reference documents that may help you in answering the user's question.\n\n"
)
QUERY_HEADER = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
PAD_TOKEN = 128001


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_eval_split(name):
    return load_jsonl(f"{os.environ.get('DATA_DIR', '../')}/rag/{name}_eval/dataset")


@dataclass(frozen=True)
class RagDataset:
    load: Callable[[], List[dict]]
    # Result file stem per mode, formatted with name, qf (query_first) and pos.
    results: Dict[str, str]
    # Progress-print prefix ("<label>_link" / "<label>_nolink"); None prints "Correct progress".
    label: Optional[str]
    doc_key: str = "documents"
    text_key: str = "text"
    max_docs: Optional[int] = None
    answers: Callable[[dict], List[str]] = itemgetter("answers")
    # --pos moves the first (gold) document to this index; nq-open-10_0 lists the gold first.
    has_pos: bool = False
    # Single-level gist mode inserts word-aware even for checkpoints without "new" in the path.
    gist_word_aware: bool = False


DATASETS = {
    "nq": RagDataset(
        load=lambda: load_jsonl(os.path.join(HERE, "nq-open-10_0.jsonl")),
        results={
            "base": "base_nolink_{name}_qf{qf}_NQ_at{pos}",
            "gist": "nolink_select_{name}_NQ_at{pos}",
            "link": "base_select_{name}_qf{qf}_NQ_at{pos}",
        },
        label=None,
        doc_key="ctxs",
        max_docs=10,
        has_pos=True,
        gist_word_aware=True,
    ),
    "tqa": RagDataset(
        load=lambda: load_eval_split("tqa"),
        results={
            "base": "tqa_nolink_base_{name}",
            "gist": "tqa_nolink_select_{name}",
            "link": "tqa_full_ckpt_base_select_{name}",
        },
        label="TQA",
        max_docs=10,
    ),
    "hqa": RagDataset(
        load=lambda: load_eval_split("hqa"),
        results={
            "base": "hqa2_nolink_base_{name}",
            "gist": "hqa2_nolink_select_{name}",
            "link": "hqa2_full_ckpt_base_select_{name}",
        },
        label="HQA",
    ),
    "wiki": RagDataset(
        load=lambda: load_eval_split("2wiki"),
        results={
            "base": "wiki_nolink-base_{name}",
            "gist": "wiki_nolink-select_{name}",
            "link": "wiki_select_{name}",
        },
        label="Wiki",
    ),
    "musique": RagDataset(
        load=lambda: load_dataset("dgslibisey/MuSiQue", split="validation"),
        results={
            "base": "mqa_nolink_base_{name}",
            "gist": "mqa_nolink_select_{name}",
            "link": "mqa_full_ckpt_base_select_{name}",
        },
        label="MQA",
        doc_key="paragraphs",
        text_key="paragraph_text",
        answers=lambda sample: [sample["answer"]],
    ),
}


def normalize_answer(s) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        if normalize_answer(ground_truth).lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False).input_ids


def encode_docs(tokenizer, docs):
    return [
        encode(tokenizer, f"Document [{j + 1}](Title: {doc['title']}) {doc['text']}\n")
        for j, doc in enumerate(docs)
    ]


def assemble(tokenizer, system_ids, doc_ids, question, query_first, device):
    seq = [tokenizer.bos_token_id, *system_ids]
    if query_first:
        seq += encode(tokenizer, QUERY_HEADER + question)
    for ids in doc_ids:
        seq += ids
    seq += encode(tokenizer, question + "\n") + encode(tokenizer, "Answer:\n")
    input_ids = torch.tensor([seq], device=device)
    return input_ids, torch.ones_like(input_ids)


def insert_gist(tokenizer, ids, sink_size, word_aware):
    gist_ids, chunk_size = Global_data.gist_token_id, Global_data.chunk_size
    if len(gist_ids) > 1:
        return gist.insert_hierarchical_gist_tokens(
            ids, gist_ids, chunk_size[0], chunk_size[1], tokenizer, sink_size
        )
    if word_aware:
        return gist.insert_gist_tokens_word_aware(
            ids, gist_ids, chunk_size[0], tokenizer, sink_size
        )
    return gist.insert_gist_tokens_by_token_count(ids, gist_ids, chunk_size[0], sink_size)


def gist_mask(input_ids, **kwargs):
    if len(Global_data.gist_token_id) > 1:
        mask_fn = gist.make_hierarchical_gist_mask_greedy
    else:
        mask_fn = gist.make_gist_mask
    return mask_fn(
        inputs=input_ids,
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=PAD_TOKEN,
        **kwargs,
    ).to(input_ids.device)


def move_links_after_last_gist(input_ids, mask):
    """Reorder tokens, and the [B, H, L, L] mask rows/columns, so the link tokens sit right
    after the last gist token."""
    device = input_ids.device
    link_ids = torch.tensor(Global_data.link_token_id, device=device)
    is_link = torch.isin(input_ids, link_ids)
    is_gist = torch.isin(input_ids, torch.tensor(Global_data.gist_token_id, device=device))
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    last_gist = torch.where(is_gist, positions, torch.tensor(-1, device=device))
    last_gist = last_gist.max(dim=1, keepdim=True)[0]
    link_cumsum = is_link.cumsum(dim=1)
    total_links = link_cumsum[:, -1].unsqueeze(1)
    new_positions = torch.where(
        is_link,
        last_gist - (total_links - link_ids.shape[0]) + link_cumsum,
        torch.where(positions > last_gist, positions, positions - link_cumsum),
    )
    order = new_positions.argsort(dim=1)
    B, H, L, _ = mask.shape
    mask = mask.gather(3, order[:, None, None, :].expand(B, H, L, L))
    mask = mask.gather(2, order[:, None, :, None].expand(B, H, L, L))
    return input_ids.gather(1, order), mask


def base_inputs(tokenizer, args, docs, question, device):
    input_ids, attention_mask = assemble(
        tokenizer,
        encode(tokenizer, SYSTEM),
        encode_docs(tokenizer, docs),
        question,
        args.query_first,
        device,
    )
    return dict(input_ids=input_ids, attention_mask=attention_mask, num_beams=1, temperature=1.0)


def gist_inputs(tokenizer, args, docs, question, device):
    word_aware = DATASETS[args.dataset].gist_word_aware or "new" in args.ckpt_path
    system_ids = insert_gist(
        tokenizer, encode(tokenizer, SYSTEM), Global_data.sink_size - 1, word_aware
    )
    doc_ids = [insert_gist(tokenizer, ids, 0, word_aware) for ids in encode_docs(tokenizer, docs)]
    input_ids, attention_mask = assemble(
        tokenizer, system_ids, doc_ids, question, args.query_first, device
    )
    mask = gist_mask(
        input_ids, attention_sink_size=Global_data.sink_size, add_raw=Global_data.add_raw
    )
    if Global_data.add_raw:
        mask = mask[0]
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        attention_mask_gist=mask,
        num_beams=1,
        temperature=1.0,
    )


def link_inputs(tokenizer, args, docs, question, device):
    word_aware = "new" in args.ckpt_path
    doc_ids = [
        insert_gist(tokenizer, ids, Global_data.sink_size, word_aware) + Global_data.link_token_id
        for ids in encode_docs(tokenizer, docs)
    ]
    input_ids, attention_mask = assemble(
        tokenizer, encode(tokenizer, SYSTEM), doc_ids, question, args.query_first, device
    )
    mask = gist_mask(input_ids, attention_sink_size=-1, link_tokens=Global_data.link_token_id)
    input_ids, mask = move_links_after_last_gist(input_ids, mask)
    return dict(
        input_ids=input_ids, attention_mask=attention_mask, attention_mask_gist=mask, use_cache=True
    )


# Each mode builds the model.generate kwargs for one sample.
MODES = {"base": base_inputs, "gist": gist_inputs, "link": link_inputs}


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    model_cls = AutoModelForCausalLM if args.mode == "base" else GistLlamaForCausalLM
    model = model_cls.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
    model.to("cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    if args.mode == "base":
        return tokenizer, model

    model.selective = args.selective
    num_gist_level = apply_gist_args(args)
    try:
        Global_data.gist_token_id = check_gist_model(tokenizer, model, num_gist_level)
        if args.mode == "link":
            Global_data.link_token_id = find_link_token_ids(tokenizer, args.link_num)
    except ValueError as e:
        print(f"ERROR! Not the Gist pretrained model!! {e}")
        sys.exit(1)
    Global_data.gist_token = gist_token_names(num_gist_level)
    if args.mode == "link":
        Global_data.link_token = link_token_names(args.link_num)
    Global_data.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--name", type=str, default="base")
    parser.add_argument("--query_first", action="store_true", help="Place query before documents.")
    parser.add_argument("--pos", type=int, help="nq only (required): index of the gold document.")
    parser.add_argument("--link_num", type=int, default=5, help="Number of link tokens (link).")
    add_gist_args(parser)
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    if (args.pos is not None) != dataset.has_pos:
        parser.error("--pos is required for nq and unsupported for the other datasets")

    data = dataset.load()
    tokenizer, model = load_model(args)
    build_inputs = MODES[args.mode]
    if dataset.label is None:
        progress = "Correct progress"
    else:
        progress = f"{dataset.label}_{'link' if args.mode == 'link' else 'nolink'}"

    total_num = len(data)
    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples is not None:
        total_num = min(total_num, max(1, int(max_samples)))
    correct_num = 0
    res_list = []
    for i in tqdm(range(total_num)):
        print("Processing sample:", str(i))
        sample = data[i]
        docs = [
            {"title": doc["title"], "text": doc[dataset.text_key]}
            for doc in sample[dataset.doc_key][: dataset.max_docs]
        ]
        if args.pos:
            docs.insert(args.pos, docs.pop(0))

        generate_kwargs = build_inputs(tokenizer, args, docs, sample["question"], model.device)
        with torch.no_grad():
            generated = model.generate(**generate_kwargs, max_new_tokens=200, do_sample=False)
        response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        response = response.split("Answer:\n")[-1]
        print(response)

        gold = dataset.answers(sample)
        score = best_subspan_em(response, gold)
        correct_num += int(score)
        res_list.append(
            {
                "id": str(i),
                "question": sample["question"],
                "response": response,
                "gold_answer": gold,
                "Score": score,
            }
        )
        print(progress, correct_num / (i + 1))

    accuracy = correct_num / total_num
    stem = dataset.results[args.mode].format(name=args.name, qf=args.query_first, pos=args.pos)
    file_name = f"result/{stem}_{accuracy}.jsonl"
    os.makedirs("result", exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        for entry in res_list:
            f.write(json.dumps(entry) + "\n")
    print(f"Dumped at {file_name}")


if __name__ == "__main__":
    main()
