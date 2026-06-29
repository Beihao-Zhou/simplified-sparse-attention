import random
from typing import Dict
import math
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import torch
import src.data.gist as gist
from gist_utils import Global_data
from transformers import PreTrainedTokenizerBase

general_prompts = [
    "You are an AI assistant. Provide helpful, accurate, and clear answers. When uncertain, explain your reasoning or request clarification.",
    "You are an AI assistant. Focus on achieving the user's goal in each interaction. Use concise yet informative explanations.",
    "You are an AI assistant. Speak clearly and stay consistent with prior statements. If you need more information, politely ask for it.",
    "You are an AI assistant. Provide truthful, well-sourced information whenever possible. Acknowledge any limitations and avoid speculation if unsure."
]
qa_prompts = [
    "You are an AI assistant. Use the provided documents to answer the user’s question. If the information is insufficient, acknowledge the gap or request clarification.",
    "You are an AI assistant. Always ground your answers in the retrieved documents and do not add unsupported details. If the documents lack sufficient information, indicate that.",
    "You are an AI assistant. Rely solely on the given documents for evidence when answering questions. When necessary, cite or paraphrase the document content accurately.",
    "You are an AI assistant. Base your replies on the retrieved documents, ensuring completeness and correctness. Ask for more details if the documents do not cover the question fully."
]
summary_prompts = [
    "You are an AI assistant. Read the provided text and produce a concise summary. Capture the main points without unnecessary detail.",
    "You are an AI assistant. Summarize the essential ideas from the given text. Avoid minor details and focus on critical insights.",
    "You are an AI assistant. Provide a brief, high-level overview of the text. Ensure clarity and coherence, prioritizing key themes.",
    "You are an AI assistant. Summarize the text clearly and logically. Organize the main ideas in a coherent sequence."
]


class gist_attention_preprocessor():
    '''
    Apply one piece of memory to non-memory use samples to enable batch forward pass for calculating KV.
    '''
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        do_shuffle: bool,
        chunk_size: list[int] = [8],
        attention_sink_size:int = 3,
        gist_token_ids: list[int] = [128011, 128012, 128013, 128014, 128015],
        pad_token: int = 128004,
        meta_gist: int = 1,
        interleaved: bool = False,
        query_first: bool = False,
        use_semantic_chunking: bool = False,
        chunk_size_flexibility: float = 0.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.do_shuffle = do_shuffle
        self.chunk_size = chunk_size
        self.attention_sink_size = attention_sink_size
        self.gist_token_ids = gist_token_ids
        self.pad_token = pad_token
        self.meta_gist = meta_gist
        self.interleaved = interleaved
        self.query_first = query_first
        self.use_semantic_chunking = use_semantic_chunking  # Not used
        self.chunk_size_flexibility = chunk_size_flexibility

    def _interleave_link_tokens(self, doc_tokens: list[int], link_tokens: list[int]) -> list[int]:
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

    def gist_process_qa_link(
        self,
        example
    ):
        output_sequence = [self.tokenizer.bos_token_id]
        labels = [-100]

        system = random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        labels.extend([-100] * sys_len)

        if self.query_first:
            user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question']
            user_id = self.tokenizer(user, add_special_tokens=False).input_ids
            user_len = len(user_id)
            labels.extend([-100] * user_len)
            output_sequence.extend(user_id)

        doc_list = []

        for k in range(len(example['documents'])):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text': text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        link_tokens = Global_data.link_token_id

        for j in range(len(doc_list)):
            local_link_tokens = link_tokens
            # if len(link_tokens) > 5:
            #     local_link_tokens = link_tokens[j * 5 : (j + 1) * 5]

            title = doc_list[j]['title']
            text = doc_list[j]['text']
            doc_text = f"Document [{j+1}](Title: {title}) {text}\n"
            doc_tokens = self.tokenizer(doc_text, add_special_tokens=False).input_ids

            # Process document chunks with GIST tokens
            if self.meta_gist == 0:
                doc_with_gist = gist.insert_gist_tokens_word_aware(
                    doc_tokens,
                    self.gist_token_ids,
                    self.chunk_size[0],
                    self.tokenizer,
                    self.attention_sink_size,
                )
            elif self.meta_gist == 1:
                doc_with_gist = gist.insert_hierarchical_gist_tokens(
                    doc_tokens,
                    self.gist_token_ids,
                    self.chunk_size[0],
                    self.chunk_size[1],
                    self.tokenizer,
                    self.attention_sink_size,
                )

            if self.interleaved:
                # Interleave link tokens evenly into doc_with_gist
                interleaved_tokens = self._interleave_link_tokens(doc_with_gist, local_link_tokens)
                output_sequence.extend(interleaved_tokens)
                labels.extend([-100] * len(interleaved_tokens))
            else:
                output_sequence.extend(doc_with_gist)
                labels.extend([-100] * len(doc_with_gist))
                output_sequence.extend(local_link_tokens)
                labels.extend([-100] * len(local_link_tokens))

        user = example['question'] + "\nAnswer:\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        labels.extend([-100] * user_len)
        output_sequence.extend(user_id)

        ans_id = self.tokenizer(example['generated'], add_special_tokens=False).input_ids
        labels.extend(ans_id)
        output_sequence.extend(ans_id)

        labels.extend([self.tokenizer.eos_token_id])
        output_sequence.extend([self.tokenizer.eos_token_id])

        return {
            "input_ids": output_sequence,
            "labels": labels,
            "attention_mask": [1 for _ in output_sequence],
        }

    def gist_process_qa(
        self,
        example
    ):
        output_sequence = [self.tokenizer.bos_token_id]
        labels = [-100]

        system = random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids

        # Process document chunks with GIST tokens
        if self.meta_gist == 0:
            system_input_ids = gist.insert_gist_tokens_word_aware(
                system_input_ids,
                self.gist_token_ids,
                self.chunk_size[0],
                self.tokenizer,
                Global_data.sink_size - 1,
            )
        else:
            system_input_ids = gist.insert_hierarchical_gist_tokens(
                system_input_ids,
                self.gist_token_ids,
                self.chunk_size[0],
                self.chunk_size[1],
                self.tokenizer,
                Global_data.sink_size - 1,
            )

        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        labels.extend([-100] * sys_len)

        if self.query_first:
            user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question']
            user_id = self.tokenizer(user, add_special_tokens=False).input_ids
            user_len = len(user_id)
            labels.extend([-100] * user_len)
            output_sequence.extend(user_id)

        doc_list = []

        for k in range(len(example['documents'])):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text': text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        for j in range(len(doc_list)):

            title = doc_list[j]['title']
            text = doc_list[j]['text']
            doc_text = f"Document [{j+1}](Title: {title}) {text}\n"
            doc_tokens = self.tokenizer(doc_text, add_special_tokens=False).input_ids

            # Process document chunks with GIST tokens
            if self.meta_gist == 0:
                doc_with_gist = gist.insert_gist_tokens_word_aware(
                    doc_tokens,
                    self.gist_token_ids,
                    self.chunk_size[0],
                    self.tokenizer,
                    0,
                )
            elif self.meta_gist == 1:
                doc_with_gist = gist.insert_hierarchical_gist_tokens(
                    doc_tokens,
                    self.gist_token_ids,
                    self.chunk_size[0],
                    self.chunk_size[1],
                    self.tokenizer,
                    0,
                )

            output_sequence.extend(doc_with_gist)
            labels.extend([-100] * len(doc_with_gist))

        user = example['question'] + "\nAnswer:\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        labels.extend([-100] * user_len)
        output_sequence.extend(user_id)

        ans_id = self.tokenizer(example['generated'], add_special_tokens=False).input_ids
        labels.extend(ans_id)
        output_sequence.extend(ans_id)

        labels.extend([self.tokenizer.eos_token_id])
        output_sequence.extend([self.tokenizer.eos_token_id])

        return {
            "input_ids": output_sequence,
            "labels": labels,
            "attention_mask": [1 for _ in output_sequence],
        }

    def process_qa(
        self,
        example
    ):
        
        output_sequence = [self.tokenizer.bos_token_id]
        labels = [-100]

        system = random.choice(qa_prompts)
        system_input_ids = self.tokenizer(system, add_special_tokens=False).input_ids
        sys_len = len(system_input_ids)

        output_sequence.extend(system_input_ids)
        labels.extend([-100] * sys_len)

        if self.query_first:
            user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question']
            user_id = self.tokenizer(user, add_special_tokens=False).input_ids
            user_len = len(user_id)
            labels.extend([-100] * user_len)
            output_sequence.extend(user_id)

        doc_list = []

        for k in range(len(example['documents'])):
            title = example['documents'][k]['title']
            text = example['documents'][k]['text']
            doc_list.append({'title': title, 'text': text})

        if self.do_shuffle:
            random.shuffle(doc_list)

        for j in range(len(doc_list)):

            title = doc_list[j]['title']
            text = doc_list[j]['text']
            doc_text = f"Document [{j+1}](Title: {title}) {text}\n"
            doc_tokens = self.tokenizer(doc_text, add_special_tokens=False).input_ids
            output_sequence.extend(doc_tokens)
            labels.extend([-100] * len(doc_tokens))

        user = example['question'] + "\nAnswer:\n"
        user_id = self.tokenizer(user, add_special_tokens=False).input_ids
        user_len = len(user_id)
        labels.extend([-100] * user_len)
        output_sequence.extend(user_id)

        ans_id = self.tokenizer(example['generated'], add_special_tokens=False).input_ids
        labels.extend(ans_id)
        output_sequence.extend(ans_id)

        labels.extend([self.tokenizer.eos_token_id])
        output_sequence.extend([self.tokenizer.eos_token_id])

        return {
            "input_ids": output_sequence,
            "labels": labels,
            "attention_mask": [1 for _ in output_sequence],
        }

def custom_collate_chunkaug(batch):
    model_inputs = defaultdict(list)
    for instance in batch:
        model_inputs["input_ids"].append(instance['input_ids'])
        model_inputs["labels"].append(instance['labels'])
        model_inputs["attention_mask"].append(instance['attention_mask'])

    # Left-pad inputs, convert to tensor.
    for key, value in model_inputs.items():
        if key == "labels":
            pad_token_id = -100
        else:
            pad_token_id = Global_data.pad_token_id
        # To left-pad inputs, reverse, then right-pad, then reverse.
        value_tensors = [torch.tensor(v[::-1]) for v in value]
        model_inputs[key] = torch.fliplr(
            pad_sequence(
                value_tensors,
                batch_first=True,
                padding_value=pad_token_id,
            )
        )

    gist_fn = gist.make_gist_mask
    model_inputs["attention_mask_gist"] = gist_fn(
        inputs=model_inputs["input_ids"],
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=Global_data.sink_size,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=Global_data.pad_token_id,
        meta_gist=0,
        add_raw=Global_data.add_raw,
    )

    return dict(model_inputs)


def collate_chunkaug(batch):
    model_inputs = defaultdict(list)
    for instance in batch:
        model_inputs["input_ids"].append(instance['input_ids'])
        model_inputs["labels"].append(instance['labels'])
        model_inputs["attention_mask"].append(instance['attention_mask'])

    # Left-pad inputs, convert to tensor.
    for key, value in model_inputs.items():
        if key == "labels":
            pad_token_id = -100
        else:
            pad_token_id = Global_data.pad_token_id
        # To left-pad inputs, reverse, then right-pad, then reverse.
        value_tensors = [torch.tensor(v[::-1]) for v in value]
        model_inputs[key] = torch.fliplr(
            pad_sequence(
                value_tensors,
                batch_first=True,
                padding_value=pad_token_id,
            )
        )

    return dict(model_inputs)


def meta_custom_collate_chunkaug(batch):
    model_inputs = defaultdict(list)
    for instance in batch:
        model_inputs["input_ids"].append(instance['input_ids'])
        model_inputs["labels"].append(instance['labels'])
        model_inputs["attention_mask"].append(instance['attention_mask'])

    # Left-pad inputs, convert to tensor.
    for key, value in model_inputs.items():
        if key == "labels":
            pad_token_id = -100
        else:
            pad_token_id = Global_data.pad_token_id
        # To left-pad inputs, reverse, then right-pad, then reverse.
        value_tensors = [torch.tensor(v[::-1]) for v in value]
        model_inputs[key] = torch.fliplr(
            pad_sequence(
                value_tensors,
                batch_first=True,
                padding_value=pad_token_id,
            )
        )

    model_inputs["attention_mask_gist"] = gist.make_hierarchical_gist_mask_greedy(
        inputs=model_inputs["input_ids"],
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=Global_data.sink_size,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=Global_data.pad_token_id,
        add_raw=Global_data.add_raw,
    )

    return dict(model_inputs)


def link_custom_collate_chunkaug(batch):
    model_inputs = defaultdict(list)
    for instance in batch:
        model_inputs["input_ids"].append(instance['input_ids'])
        model_inputs["labels"].append(instance['labels'])
        model_inputs["attention_mask"].append(instance['attention_mask'])

    # Left-pad inputs, convert to tensor.
    for key, value in model_inputs.items():
        if key == "labels":
            pad_token_id = -100
        else:
            pad_token_id = Global_data.pad_token_id
        # To left-pad inputs, reverse, then right-pad, then reverse.
        value_tensors = [torch.tensor(v[::-1]) for v in value]
        model_inputs[key] = torch.fliplr(
            pad_sequence(
                value_tensors,
                batch_first=True,
                padding_value=pad_token_id,
            )
        )

    gist_fn = gist.make_gist_mask
    model_inputs["attention_mask_gist"] = gist_fn(
        inputs=model_inputs["input_ids"],
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=-1,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=Global_data.pad_token_id,
        meta_gist=0,
        link_tokens=Global_data.link_token_id,
    )

    # Move link tokens to the end of the compressed context (right after the last gist token)
    # Also move the attention mask accordingly
    device = model_inputs["input_ids"].device
    link_token_tensor = torch.tensor(Global_data.link_token_id, device=device)
    gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=device)

    batch_size, seq_len = model_inputs["input_ids"].shape

    # Identify link and gist token positions for all batches (vectorized)
    is_link = torch.isin(model_inputs["input_ids"], link_token_tensor)  # [B, S]
    is_gist = torch.isin(model_inputs["input_ids"], gist_token_tensor)  # [B, S]

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

    model_inputs["input_ids"] = model_inputs["input_ids"][batch_indices, sorted_indices]

    # Reorder the attention_mask_gist to match the moved tokens
    # attention_mask_gist has shape [B, H, L, L], need to reorder both rows and columns
    B, H, L, _ = model_inputs["attention_mask_gist"].shape
    # Expand indices for gathering columns (last dim): (B, L) -> (B, H, L, L)
    col_indices = sorted_indices.unsqueeze(1).unsqueeze(1).expand(B, H, L, L)
    # Expand indices for gathering rows (second to last dim): (B, L) -> (B, H, L, L)
    row_indices = sorted_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, L, L)
    # Reorder columns (source dimension)
    model_inputs["attention_mask_gist"] = model_inputs["attention_mask_gist"].gather(3, col_indices)
    # Reorder rows (target dimension)
    model_inputs["attention_mask_gist"] = model_inputs["attention_mask_gist"].gather(2, row_indices)

    return dict(model_inputs)


def meta_link_custom_collate_chunkaug(batch):
    model_inputs = defaultdict(list)
    for instance in batch:
        model_inputs["input_ids"].append(instance['input_ids'])
        model_inputs["labels"].append(instance['labels'])
        model_inputs["attention_mask"].append(instance['attention_mask'])

    # Left-pad inputs, convert to tensor.
    for key, value in model_inputs.items():
        if key == "labels":
            pad_token_id = -100
        else:
            pad_token_id = Global_data.pad_token_id
        # To left-pad inputs, reverse, then right-pad, then reverse.
        value_tensors = [torch.tensor(v[::-1]) for v in value]
        model_inputs[key] = torch.fliplr(
            pad_sequence(
                value_tensors,
                batch_first=True,
                padding_value=pad_token_id,
            )
        )

    gist_fn = gist.make_hierarchical_gist_mask_greedy
    model_inputs["attention_mask_gist"] = gist_fn(
        inputs=model_inputs["input_ids"],
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=-1,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=Global_data.pad_token_id,
        link_tokens=Global_data.link_token_id,
    )

    # Move link tokens to the end of the compressed context (right after the last gist token)
    # Also move the attention mask accordingly
    device = model_inputs["input_ids"].device
    link_token_tensor = torch.tensor(Global_data.link_token_id, device=device)
    gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=device)

    batch_size, seq_len = model_inputs["input_ids"].shape

    # Identify link and gist token positions for all batches (vectorized)
    is_link = torch.isin(model_inputs["input_ids"], link_token_tensor)  # [B, S]
    is_gist = torch.isin(model_inputs["input_ids"], gist_token_tensor)  # [B, S]

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

    model_inputs["input_ids"] = model_inputs["input_ids"][batch_indices, sorted_indices]

    # Reorder the attention_mask_gist to match the moved tokens
    # attention_mask_gist has shape [B, H, L, L], need to reorder both rows and columns
    B, H, L, _ = model_inputs["attention_mask_gist"].shape
    # Expand indices for gathering columns (last dim): (B, L) -> (B, H, L, L)
    col_indices = sorted_indices.unsqueeze(1).unsqueeze(1).expand(B, H, L, L)
    # Expand indices for gathering rows (second to last dim): (B, L) -> (B, H, L, L)
    row_indices = sorted_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, L, L)
    # Reorder columns (source dimension)
    model_inputs["attention_mask_gist"] = model_inputs["attention_mask_gist"].gather(3, col_indices)
    # Reorder rows (target dimension)
    model_inputs["attention_mask_gist"] = model_inputs["attention_mask_gist"].gather(2, row_indices)

    return dict(model_inputs)
