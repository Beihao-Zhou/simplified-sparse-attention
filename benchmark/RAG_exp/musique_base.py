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
parser.add_argument("--query_first", action="store_true", help="Place query before documents if set.")
args = parser.parse_args()

data = load_dataset("dgslibisey/MuSiQue", split="validation")

tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
model.to('cuda')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

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

    for j in range(len(data[i]['paragraphs'])):
        title = data[i]['paragraphs'][j]['title']
        text = data[i]['paragraphs'][j]['paragraph_text']
        doc_list.append({'title': title, 'text':text})

    output_sequence = [tokenizer.bos_token_id]
    system_input_ids = tokenizer(system, add_special_tokens=False).input_ids
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

        output_sequence.extend(doc_tokens)

    user = data[i]['question'] + "\n"
    user_id = tokenizer(user, add_special_tokens=False).input_ids
    output_sequence.extend(user_id)

    prefix_ids = tokenizer("Answer:\n", add_special_tokens=False).input_ids
    output_sequence.extend(prefix_ids)

    input_ids = torch.tensor([output_sequence], device = model.device)
    attention_mask = torch.tensor([[1 for _ in output_sequence]], device = model.device)


    with torch.no_grad():
        generated = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask, 
                                   max_new_tokens=200,
                                    num_beams=1,
                                    do_sample=False,
                                    temperature=1.0
                                    )
    
        generated_seq = tokenizer.batch_decode(generated, skip_special_tokens=True)

        response = generated_seq[0].split('Answer:\n')[-1]
        print(response)

        score = best_subspan_em(response, [data[i]['answer']])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data[i]["question"], "response": response, "gold_answer": [data[i]['answer']], "Score": score})
        print("MQA_nolink", correct_num / (i+1))
    

accuracy = correct_num / total_num

file_name = f"result/mqa_nolink_base_{args.name}_{accuracy}.jsonl"
if not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))

with open(file_name, "w", encoding="utf-8") as f:
    for entry in res_list:
        json_line = json.dumps(entry)
        f.write(json_line + "\n")

print(f"Dumped at {file_name}")