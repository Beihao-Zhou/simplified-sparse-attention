import random

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

import src.data.gist as gist
from gist_utils import Global_data

summary_prompts = [
    "You are an AI assistant. Read the provided text and produce a concise summary. Capture the main points without unnecessary detail.",
    "You are an AI assistant. Summarize the essential ideas from the given text. Avoid minor details and focus on critical insights.",
    "You are an AI assistant. Provide a brief, high-level overview of the text. Ensure clarity and coherence, prioritizing key themes.",
    "You are an AI assistant. Summarize the text clearly and logically. Organize the main ideas in a coherent sequence."
]
qa_prompts = [
    "You are an AI assistant. Use the provided documents to answer the user’s question. If the information is insufficient, acknowledge the gap or request clarification.",
    "You are an AI assistant. Always ground your answers in the retrieved documents and do not add unsupported details. If the documents lack sufficient information, indicate that.",
    "You are an AI assistant. Rely solely on the given documents for evidence when answering questions. When necessary, cite or paraphrase the document content accurately.",
    "You are an AI assistant. Base your replies on the retrieved documents, ensuring completeness and correctness. Ask for more details if the documents do not cover the question fully."
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

        if num_links == 1:
            return doc_tokens + link_tokens

        result = []
        interval = doc_len / num_links

        link_idx = 0
        for i, token in enumerate(doc_tokens):
            # Reserve the last link token for the end
            if link_idx < num_links - 1 and i >= int((link_idx + 1) * interval):
                result.extend(link_tokens[link_idx:link_idx+1])
                link_idx += 1
            result.append(token)

        if link_idx < num_links:
            result.extend(link_tokens[link_idx:])

        return result

    def _tokenize(self, text):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _insert_gists(self, token_ids, attention_sink_size):
        if self.meta_gist == 0:
            return gist.insert_gist_tokens_word_aware(
                token_ids,
                self.gist_token_ids,
                self.chunk_size[0],
                self.tokenizer,
                attention_sink_size,
            )
        return gist.insert_hierarchical_gist_tokens(
            token_ids,
            self.gist_token_ids,
            self.chunk_size[0],
            self.chunk_size[1],
            self.tokenizer,
            attention_sink_size,
        )

    def _qa_segments(self, example):
        """(kind, token ids) for each part of a QA prompt, in order; kind is one of
        system / query / doc / answer, and the answer ends with EOS."""
        segments = [("system", self._tokenize(random.choice(qa_prompts)))]
        if self.query_first:
            user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + example['question']
            segments.append(("query", self._tokenize(user)))

        doc_list = [{'title': d['title'], 'text': d['text']} for d in example['documents']]
        if self.do_shuffle:
            random.shuffle(doc_list)
        for j, doc in enumerate(doc_list):
            doc_text = f"Document [{j+1}](Title: {doc['title']}) {doc['text']}\n"
            segments.append(("doc", self._tokenize(doc_text)))

        segments.append(("query", self._tokenize(example['question'] + "\nAnswer:\n")))
        answer = self._tokenize(example['generated']) + [self.tokenizer.eos_token_id]
        segments.append(("answer", answer))
        return segments

    def _assemble(self, segments):
        """BOS + segments; loss only on the answer."""
        output_sequence = [self.tokenizer.bos_token_id]
        labels = [-100]
        for kind, ids in segments:
            output_sequence.extend(ids)
            labels.extend(ids if kind == "answer" else [-100] * len(ids))
        return {
            "input_ids": output_sequence,
            "labels": labels,
            "attention_mask": [1 for _ in output_sequence],
        }

    def gist_process_qa_link(self, example):
        link_tokens = Global_data.link_token_id
        segments = []
        for kind, ids in self._qa_segments(example):
            if kind == "doc":
                ids = self._insert_gists(ids, self.attention_sink_size)
                if self.interleaved:
                    ids = self._interleave_link_tokens(ids, link_tokens)
                else:
                    ids = ids + list(link_tokens)
            segments.append((kind, ids))
        return self._assemble(segments)

    def gist_process_qa(self, example):
        sink = {"system": Global_data.sink_size - 1, "doc": 0}
        return self._assemble([
            (kind, self._insert_gists(ids, sink[kind]) if kind in sink else ids)
            for kind, ids in self._qa_segments(example)
        ])

    def process_qa(self, example):
        return self._assemble(self._qa_segments(example))

def _left_pad(batch):
    """input_ids / labels / attention_mask of a batch as left-padded tensors."""
    model_inputs = {}
    for key in ("input_ids", "labels", "attention_mask"):
        pad_token_id = -100 if key == "labels" else Global_data.pad_token_id
        # To left-pad inputs, reverse, then right-pad, then reverse.
        value_tensors = [torch.tensor(instance[key][::-1]) for instance in batch]
        model_inputs[key] = torch.fliplr(
            pad_sequence(value_tensors, batch_first=True, padding_value=pad_token_id)
        )
    return model_inputs


def _attach_gist_mask(model_inputs, mask_fn, **mask_kwargs):
    """Add attention_mask_gist to a left-padded batch."""
    model_inputs["attention_mask_gist"] = mask_fn(
        inputs=model_inputs["input_ids"],
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=Global_data.sink_size,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=Global_data.pad_token_id,
        add_raw=Global_data.add_raw,
        **mask_kwargs,
    )
    return dict(model_inputs)


def _link_collate(batch, mask_fn, **mask_kwargs):
    """Left-pad, build the link-aware gist mask, then move the link tokens (and their mask
    rows/columns) right after the last gist token."""
    model_inputs = _left_pad(batch)
    model_inputs["attention_mask_gist"] = mask_fn(
        inputs=model_inputs["input_ids"],
        gist_token=Global_data.gist_token_id,
        chunk_size=Global_data.chunk_size[0],
        attention_sink_size=-1,
        num_previous_chunks=Global_data.num_previous_chunks,
        pad_token=Global_data.pad_token_id,
        link_tokens=Global_data.link_token_id,
        **mask_kwargs,
    )

    device = model_inputs["input_ids"].device
    link_token_tensor = torch.tensor(Global_data.link_token_id, device=device)
    gist_token_tensor = torch.tensor(Global_data.gist_token_id, device=device)
    batch_size, seq_len = model_inputs["input_ids"].shape

    is_link = torch.isin(model_inputs["input_ids"], link_token_tensor)  # [B, S]
    is_gist = torch.isin(model_inputs["input_ids"], gist_token_tensor)  # [B, S]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]

    gist_positions = torch.where(is_gist, positions, torch.tensor(-1, device=device))
    last_gist_pos = gist_positions.max(dim=1, keepdim=True)[0]  # [B, 1]

    # Sort keys: non-link tokens up to the last gist keep their order, link tokens follow
    # right after it, the remaining tokens keep their positions.
    link_cumsum = is_link.cumsum(dim=1)  # [B, S]
    total_links = link_cumsum[:, -1].unsqueeze(1)  # [B, 1]
    is_after_gist = positions > last_gist_pos  # [B, S]

    new_positions = torch.zeros_like(positions, dtype=torch.long)
    new_positions = torch.where(
        is_link,
        last_gist_pos - (total_links - link_token_tensor.shape[0]) + link_cumsum,
        new_positions
    )
    new_positions = torch.where((~is_link) & (~is_after_gist), positions - link_cumsum, new_positions)
    new_positions = torch.where((~is_link) & is_after_gist, positions, new_positions)

    sorted_indices = new_positions.argsort(dim=1)  # [B, S]
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
    model_inputs["input_ids"] = model_inputs["input_ids"][batch_indices, sorted_indices]

    # Reorder both key (columns) and query (rows) dims of the [B, H, L, L] mask.
    B, H, L, _ = model_inputs["attention_mask_gist"].shape
    col_indices = sorted_indices.unsqueeze(1).unsqueeze(1).expand(B, H, L, L)
    row_indices = sorted_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, L, L)
    model_inputs["attention_mask_gist"] = model_inputs["attention_mask_gist"].gather(3, col_indices)
    model_inputs["attention_mask_gist"] = model_inputs["attention_mask_gist"].gather(2, row_indices)

    return dict(model_inputs)


def custom_collate_chunkaug(batch):
    return _attach_gist_mask(_left_pad(batch), gist.make_gist_mask, meta_gist=0)


def collate_chunkaug(batch):
    return _left_pad(batch)


def meta_custom_collate_chunkaug(batch):
    return _attach_gist_mask(_left_pad(batch), gist.make_hierarchical_gist_mask_greedy)


def link_custom_collate_chunkaug(batch):
    return _link_collate(batch, gist.make_gist_mask, meta_gist=0)


def meta_link_custom_collate_chunkaug(batch):
    return _link_collate(batch, gist.make_hierarchical_gist_mask_greedy)
