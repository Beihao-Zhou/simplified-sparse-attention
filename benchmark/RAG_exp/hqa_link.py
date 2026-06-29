import os
import torch
import string
import argparse
import json
import regex
import sys
import datasets
from tqdm import tqdm

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from gist_llama import GistLlamaForCausalLM
import gist_llama
from src.data import gist
from gist_utils import Global_data


def _interleave_link_tokens(doc_tokens: list[int], link_tokens: list[int]) -> list[int]:
    """
    Interleave link tokens evenly throughout doc_tokens, with the last link at the end.

    Args:
        doc_tokens: The document tokens (with gist tokens already inserted)
        link_tokens: The link tokens to interleave

    Returns:
        A new list with link tokens evenly distributed throughout doc_tokens
    """
    if not link_tokens:
        return doc_tokens

    doc_len = len(doc_tokens)
    num_links = len(link_tokens)

    # Calculate the spacing between link tokens
    if num_links == 1:
        # Insert single link token at the end
        return doc_tokens + link_tokens

    # For multiple link tokens, distribute evenly with last one at the end
    result = []
    interval = doc_len / num_links

    link_idx = 0
    for i, token in enumerate(doc_tokens):
        # Check if we should insert a link token before this position
        # Reserve the last link token for the end
        if link_idx < num_links - 1 and i >= int((link_idx + 1) * interval):
            result.extend(link_tokens[link_idx:link_idx+1])
            link_idx += 1
        result.append(token)

    # Add the last link token at the end
    if link_idx < num_links:
        result.extend(link_tokens[link_idx:])

    return result


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
parser.add_argument("--link_num", type=int, default=5, help="Number of link tokens to use.")
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
data.extend(load_jsonline(fp=f"{data_dir}/rag/hqa_eval/dataset"))

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

if len(tokenizer) == gist_llama.PRETRAINED_VOCAB_SIZE + 3 + args.link_num:
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
link_token_ids = tokenizer.additional_special_tokens_ids[-args.link_num:]
link_token = tokenizer.additional_special_tokens[-args.link_num:]
Global_data.gist_token_id = gist_token_ids
Global_data.gist_token = gist_token
Global_data.top_k = args.top_k
Global_data.top_p = args.top_p
Global_data.sink_size = args.sink_size
Global_data.pad_token_id = tokenizer.pad_token_id
Global_data.link_token_id = link_token_ids
Global_data.link_token = link_token
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

    for k in range(len(data[i]['documents'])):
        title = data[i]["documents"][k]["title"]
        text = data[i]["documents"][k]["text"]
        doc_list.append({'title': title, 'text':text})

    output_sequence = [tokenizer.bos_token_id]
    system_input_ids = tokenizer(system, add_special_tokens=False).input_ids
    output_sequence.extend(system_input_ids)

    if query_first:
        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + data[i]['question']
        user_id = tokenizer(user, add_special_tokens=False).input_ids
        output_sequence.extend(user_id)

    link_tokens = Global_data.link_token_id

    for j in range(len(doc_list)):
        local_link_tokens = link_tokens
        
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
                    Global_data.sink_size,
                )
            else:
                doc_with_gist = gist.insert_gist_tokens_by_token_count(
                    doc_tokens,
                    Global_data.gist_token_id,
                    Global_data.chunk_size[0],
                    Global_data.sink_size,
                )
        else:
            doc_with_gist = gist.insert_hierarchical_gist_tokens(
                doc_tokens,
                Global_data.gist_token_id,
                Global_data.chunk_size[0],
                Global_data.chunk_size[1],
                tokenizer,
                Global_data.sink_size,
            )

        output_sequence.extend(doc_with_gist)
        output_sequence.extend(local_link_tokens)

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
        attention_sink_size=-1,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=128001,
        link_tokens=Global_data.link_token_id,
    ).to(model.device)


    # Move link tokens to the end of the compressed context (right after the last gist token)
    # Also move the attention mask accordingly
    device = model.device
    link_token_tensor = torch.tensor(Global_data.link_token_id, device=device)
    gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=device)

    batch_size, seq_len = input_ids.shape

    # Identify link and gist token positions for all batches (vectorized)
    is_link = torch.isin(input_ids, link_token_tensor)  # [B, S]
    is_gist = torch.isin(input_ids, gist_token_tensor)  # [B, S]

    # Create position indices
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]

    # Find last gist position for each batch
    # Set positions to -1 where there's no gist, then take max
    gist_positions = torch.where(is_gist, positions, torch.tensor(-1, device=device))
    last_gist_pos = gist_positions.max(dim=1, keepdim=True)[0]  # [B, 1]

    # Create new position indices for reordering
    # Priority order:
    # 1. Non-link tokens before or at last_gist_pos: keep original order (priority 0 to last_gist_pos)
    # 2. Link tokens: move right after last gist (priority last_gist_pos + 1 to last_gist_pos + num_links)
    # 3. Non-link tokens after last_gist_pos: shift to make room for links (priority last_gist_pos + num_links + 1 onwards)

    # Calculate number of link tokens before each position (cumulative count)
    link_cumsum = is_link.cumsum(dim=1)  # [B, S]
    total_links = link_cumsum[:, -1].unsqueeze(1)  # [B, 1]

    is_after_gist = positions > last_gist_pos  # [B, S]

    # Initialize new positions
    new_positions = torch.zeros_like(positions, dtype=torch.long)

    # Link tokens: position right after last gist (11, 12, 13, ... if last_gist_pos=10)
    new_positions = torch.where(
        is_link,
        last_gist_pos - (total_links - link_token_tensor.shape[0]) + link_cumsum,  # link_cumsum gives us 1, 2, 3, ... for each link
        new_positions
    )

    new_positions = torch.where(
        (~is_link) & (~is_after_gist),
        positions - link_cumsum,
        new_positions
    )

    new_positions = torch.where(
        (~is_link) & is_after_gist,
        positions,
        new_positions
    )

    # Sort indices based on new_positions
    sorted_indices = new_positions.argsort(dim=1)  # [B, S]

    # Reorder all tensors using advanced indexing
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)

    input_ids = input_ids[batch_indices, sorted_indices]

    # Reorder the attention_mask_gist to match the moved tokens
    # attention_mask_gist has shape [B, H, L, L], need to reorder both rows and columns
    B, H, L, _ = attention_mask_gist.shape
    # Expand indices for gathering columns (last dim): (B, L) -> (B, H, L, L)
    col_indices = sorted_indices.unsqueeze(1).unsqueeze(1).expand(B, H, L, L)
    # Expand indices for gathering rows (second to last dim): (B, L) -> (B, H, L, L)
    row_indices = sorted_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, L, L)
    # Reorder columns (source dimension)
    attention_mask_gist = attention_mask_gist.gather(3, col_indices)
    # Reorder rows (target dimension)
    attention_mask_gist = attention_mask_gist.gather(2, row_indices)


    with torch.no_grad():
        generated = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask, 
                                   attention_mask_gist=attention_mask_gist,
                                   max_new_tokens=200,
                                    do_sample=False,
                                    use_cache=True)
    
        generated_seq = tokenizer.batch_decode(generated, skip_special_tokens=True)

        response = generated_seq[0].split('Answer:\n')[-1]
        print(response)

        score = best_subspan_em(response, data[i]["answers"])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data[i]["question"], "response": response, "gold_answer": data[i]["answers"], "Score": score})
        print("HQA_link", correct_num / (i+1))
    
accuracy = correct_num / total_num

file_name = f"result/hqa2_full_ckpt_base_select_{args.name}_{accuracy}.jsonl"
if not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))

with open(file_name, "w", encoding="utf-8") as f:
    for entry in res_list:
        json_line = json.dumps(entry)
        f.write(json_line + "\n")

print(f"Dumped at {file_name}")