import os
import torch
import string
import argparse
import json
import regex
import sys
import datasets
from datasets import load_dataset
from tqdm import tqdm

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from gist_llama import GistLlamaForCausalLM
import gist_llama
from src.data import gist
from gist_utils import Global_data


def filter_kv(past_key_values, mask):
    num_layers = len(past_key_values)
    filtered_past_key_values = ()

    for layer_id in range(num_layers):
        tem_key = past_key_values[layer_id][0]
        tem_value = past_key_values[layer_id][1]

        filtered_key = tem_key[:, :, mask, :]
        filtered_value = tem_value[:, :, mask, :]

        filtered_past_key_values += ((filtered_key, filtered_value),)

    return filtered_past_key_values

def filter_id(input_ids, mask):  

    return input_ids[:, mask]

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

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=None,
    help="The path to the `checkpoint.pt` file.",
)
parser.add_argument(
    "--name",
    type=str,
    default="base"
)
parser.add_argument(
    "--top_k",
    type=int,
    nargs="+",
    default=[3],
    help="Top-k values for selective training."
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0,
    help="Top-p value for selective training (nucleus sampling)."
)
parser.add_argument("--num_gist_level", type=int, default=2, help="Number of levels of gist tokens.")
parser.add_argument("--sink_size", type=int, default=3, help="Sink size for gist attention.")
parser.add_argument("--meta_gist", type=int, default=0, help="Use meta gist or not.")
parser.add_argument(
    "--chunk_size",
    type=int,
    nargs="+",
    default=[8],
    help="Chunk size for gist attention per layer."
)
parser.add_argument(
    "--selective",
    action="store_true",
    help="Enable selective training."
)
parser.add_argument("--query_first", action="store_true", help="Place query before documents if set.")
parser.add_argument("--num_previous_chunks", type=int, default=1, help="Number of previous chunks to attend to.")
parser.add_argument(
    "--raw",
    action="store_true",
    help="Enable adding raw input to the model inputs for loss computation."
)
parser.add_argument(
    "--klloss",
    action="store_true",
    help="Enable KL divergence loss between gist and raw input logits."
)

args = parser.parse_args()

data_dir = str(os.environ.get("DATA_DIR", "../"))

def load_jsonline(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(i) for i in f]
    
data = []
data.extend(load_jsonline(fp=f"{data_dir}/rag/tqa_eval/dataset"))

tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
model = GistLlamaForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
model.to('cuda')
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

query_first = args.query_first


system = (
    "You are a intelligent AI assistant. Please answer questions based on the user's instruction. "
    "Below are some reference documents that may help you in answering the user's question.\n\n"
)
total_num = len(data)
correct_num = 0
res_list = []

for i in tqdm(range(total_num)):
    print("Processing sample:", str(i))
    doc_list = []

    for k in range(0,10):
        title = data[i]["documents"][k]["title"]
        text = data[i]["documents"][k]["text"]
        doc_list.append({'title': title, 'text':text})

    output_sequence = [tokenizer.bos_token_id]
    system_input_ids = tokenizer(system, add_special_tokens=False).input_ids

    # Process document chunks with GIST tokens
    if num_gist_level == 1:
        if "new" in args.ckpt_path:
            system_input_ids = gist.insert_gist_tokens_word_aware(
                system_input_ids,
                Global_data.gist_token_id,
                Global_data.chunk_size[0],
                tokenizer,
                Global_data.sink_size - 1,
            )
        else:
            system_input_ids = gist.insert_gist_tokens_by_token_count(
                system_input_ids,
                Global_data.gist_token_id,
                Global_data.chunk_size[0],
                Global_data.sink_size - 1,
            )
    else:
        system_input_ids = gist.insert_hierarchical_gist_tokens(
            system_input_ids,
            Global_data.gist_token_id,
            Global_data.chunk_size[0],
            Global_data.chunk_size[1],
            tokenizer,
            Global_data.sink_size - 1,
        )

    output_sequence.extend(system_input_ids)

    if query_first:
        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + data[i]['question']
        user_id = tokenizer(user, add_special_tokens=False).input_ids
        output_sequence.extend(user_id)

    for j in range(len(doc_list)):
        
        title = doc_list[j]['title']
        text = doc_list[j]['text']
        doc_text = f"Document [{j+1}](Title: {title}) {text}\n"
        doc_tokens = tokenizer(doc_text, add_special_tokens=False).input_ids

        # Process document chunks with GIST tokens
        if num_gist_level == 1:
            if "new" in args.ckpt_path:
                doc_with_gist = gist.insert_gist_tokens_word_aware(
                    doc_tokens,
                    Global_data.gist_token_id,
                    Global_data.chunk_size[0],
                    tokenizer,
                    0,
                )
            else:
                doc_with_gist = gist.insert_gist_tokens_by_token_count(
                    doc_tokens,
                    Global_data.gist_token_id,
                    Global_data.chunk_size[0],
                    0,
                )
        else:
            doc_with_gist = gist.insert_hierarchical_gist_tokens(
                doc_tokens,
                Global_data.gist_token_id,
                Global_data.chunk_size[0],
                Global_data.chunk_size[1],
                tokenizer,
                0,
            )

        output_sequence.extend(doc_with_gist)

    user = data[i]['question'] + "\n"
    user_id = tokenizer(user, add_special_tokens=False).input_ids
    output_sequence.extend(user_id)

    prefix_ids = tokenizer("Answer:\n", add_special_tokens=False).input_ids
    output_sequence.extend(prefix_ids)

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


    with torch.no_grad():
        generated = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask, 
                                   attention_mask_gist=attention_mask_gist,
                                   max_new_tokens=200,
                                    num_beams=1,
                                    do_sample=False,
                                    temperature=1.0
                                    )
    
        generated_seq = tokenizer.batch_decode(generated, skip_special_tokens=True)

        response = generated_seq[0].split('Answer:\n')[-1]
        print(response)

        score = best_subspan_em(response, data[i]["answers"])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data[i]["question"], "response": response, "gold_answer": data[i]["answers"], "Score": score})
        print("TQA_nolink", correct_num / (i+1))


accuracy = correct_num / total_num

file_name = f"result/tqa_nolink_select_{args.name}_{accuracy}.jsonl"
if not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))

with open(file_name, "w", encoding="utf-8") as f:
    for entry in res_list:
        json_line = json.dumps(entry)
        f.write(json_line + "\n")

print(f"Dumped at {file_name}")