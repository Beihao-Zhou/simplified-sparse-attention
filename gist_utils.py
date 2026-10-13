"""Miscellaneous utils."""


import itertools
from typing import Any, List

import math
import torch
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Sampler, Dataset
from transformers.tokenization_utils import BatchEncoding


class Global_data:
    gist_token_id = None
    gist_token = None
    link_token_id = None
    link_token = None
    sep_token_id = None
    sep_token = None
    pad_token_id = None
    sink_size = 0
    chunk_size=[8]
    top_k = [3,3,3]
    top_p = -1.0
    num_previous_chunks = 1
    # "dense": [B,1,T,T] masks (~20K max); "flex": gist_flex_attention.py (128K+).
    train_attn = "dense"
    # Recent raw tokens kept exactly visible (R6 in src/data/gist_flex_mask.py); 0 = off.
    # Training only so far: inference ignores it.
    raw_window = 0
    # Non-selective training computes the LM loss in chunks of this many tokens instead of
    # materializing [B, T, vocab] fp32 logits; 0 = off.
    loss_chunk_size = 0
    inference_mode = "default"
    sparse_budget = 64  # Per-query budget for sparse attention (max keys to attend to)
    last_pos = None
    sep_num = 0
    add_raw = False
    kl_loss = False
    alpha = 1e-4  # Weight for the KL alignment loss
    modify_at_step = -1  # Step at which to start selective
    comp_factor = -1

    last_loss_ce = []  # Cross-entropy loss for the current batch
    last_loss_align = []  # Align loss for the current batch

    # Grouped-union GQA configuration
    use_nsa_gqa = True  # Union selected KVs across query heads within each GQA group
    use_grouped_sdpa_optimization = False  # Enable SDPA optimization for shared masks


# Names of the released checkpoints; deeper levels are <G3>, <G4>, ...
GIST_TOKENS = ["<gist>", "<Gist>", "<MGist>"]


def gist_token_names(num_levels: int = 3) -> List[str]:
    """Summary-token names for levels 0..num_levels-1."""
    return GIST_TOKENS[:num_levels] + [f"<G{lvl}>" for lvl in range(3, num_levels)]


def find_gist_token_ids(tokenizer, num_levels: int = 3) -> Optional[List[int]]:
    """Ids of the level 0..num_levels-1 summary tokens, or None if the tokenizer has none.

    By name, not vocab size: sizes differ across Qwen2/2.5/3, and a size check that
    misfired on resume re-initialized the trained gist embeddings.
    """
    names = gist_token_names(num_levels)
    vocab = tokenizer.get_vocab()
    missing = [t for t in names if t not in vocab]
    if len(missing) == len(names):
        return None
    if missing:
        raise ValueError(f"tokenizer lacks {missing}; add them with add_gist_tokens")
    return [vocab[t] for t in names]


def check_gist_model(tokenizer, model, num_levels: int = 3) -> List[int]:
    """Gist ids of a gist-trained checkpoint; embeddings must cover the tokenizer."""
    ids = find_gist_token_ids(tokenizer, num_levels)
    if ids is None:
        raise ValueError(f"not a gist checkpoint: tokenizer has none of {gist_token_names(num_levels)}")
    for w in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
        if w.shape[0] != len(tokenizer):
            raise ValueError(f"embedding rows {w.shape[0]} != len(tokenizer) {len(tokenizer)}")
    return ids


def _add_tokens(tokenizer, model, missing, all_names, keep_names, replace):
    """Add `missing` special tokens; new rows get the mean of the pre-existing rows not in
    `keep_names` (https://nlp.stanford.edu/~johnhew/vocab-expansion.html)."""
    vocab = tokenizer.get_vocab()
    n_before = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": all_names},
                                 replace_additional_special_tokens=replace)
    model.resize_token_embeddings(len(tokenizer))
    new_ids = tokenizer.convert_tokens_to_ids(missing)
    keep = torch.ones(n_before, dtype=torch.bool)
    keep[[vocab[t] for t in keep_names if t in vocab]] = False
    with torch.no_grad():
        # With tied embeddings (e.g. Qwen3-4B) both are the same tensor.
        for w in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
            w[new_ids] = w[:n_before][keep.to(w.device)].mean(0)


def add_gist_tokens(tokenizer, model, num_levels: int = 3) -> List[int]:
    """Add missing summary tokens (always levels 0-2); return the level 0..num_levels-1 ids.

    Existing gist rows are kept, so a checkpoint can be deepened.
    """
    names = gist_token_names(max(3, num_levels))
    missing = [t for t in names if t not in tokenizer.get_vocab()]
    if missing:
        print(f"ATTENTION! Adding gist tokens {missing}.")
        _add_tokens(tokenizer, model, missing, names, names, replace=True)
    return check_gist_model(tokenizer, model, num_levels)


def link_token_names(n: int) -> List[str]:
    return [f"<link{i + 1}>" for i in range(n)]


def add_link_tokens(tokenizer, model, n: int) -> List[int]:
    """Add missing <link1>..<linkN> (multi-document RAG); return their ids."""
    names = link_token_names(n)
    missing = [t for t in names if t not in tokenizer.get_vocab()]
    if missing:
        _add_tokens(tokenizer, model, missing, names, [], replace=False)
    return tokenizer.convert_tokens_to_ids(names)


def find_link_token_ids(tokenizer, n: int) -> List[int]:
    names = link_token_names(n)
    vocab = tokenizer.get_vocab()
    missing = [t for t in names if t not in vocab]
    if missing:
        raise ValueError(f"tokenizer lacks link tokens {missing}")
    return [vocab[t] for t in names]


def add_gist_args(parser):
    """Gist-attention flags shared by the trainers and evaluation scripts."""
    g = parser.add_argument_group("gist attention")
    g.add_argument("--chunk_size", type=int, nargs="+", default=[8],
                   help="L, then the grouping factor J per meta-gist level.")
    g.add_argument("--meta_gist", type=int, default=0, help="Use meta gists (hierarchical).")
    g.add_argument("--num_gist_level", type=int, default=2, help="Gist levels with --meta_gist.")
    g.add_argument("--sink_size", type=int, default=3)
    g.add_argument("--num_previous_chunks", type=int, default=1)
    g.add_argument("--top_k", type=int, nargs="+", default=[3], help="Selective unfolding top-k per level.")
    g.add_argument("--top_p", type=float, default=0)
    g.add_argument("--selective", action="store_true", help="Selective unfolding.")
    g.add_argument("--raw", action="store_true", help="Add the raw input for the KL-alignment loss.")
    g.add_argument("--klloss", action="store_true", help="KL loss between gist and raw logits.")
    return parser


def apply_gist_args(args) -> int:
    """Copy the add_gist_args flags into Global_data; return the number of gist levels."""
    Global_data.chunk_size = args.chunk_size
    Global_data.sink_size = args.sink_size
    Global_data.num_previous_chunks = args.num_previous_chunks
    Global_data.top_k = (args.top_k + args.top_k[-1:] * 3)[:3]
    Global_data.top_p = args.top_p
    Global_data.kl_loss = args.klloss
    Global_data.add_raw = args.raw or args.klloss
    Global_data.use_nsa_gqa = True
    Global_data.use_grouped_sdpa_optimization = False
    return args.num_gist_level if args.meta_gist else 1


def first_mismatch(a: List[Any], b: List[Any], window: int = 10):
    """Returns first mismatch as well as sublists for debugging."""
    for i, (x, y) in enumerate(itertools.zip_longest(a, b)):
        if x != y:
            window_slice = slice(i - window, i + window)
            return (x, y), (a[window_slice], b[window_slice])
    return None


def split_at_shared_prefix(string1, string2):
    """
    Split string1 into two parts: the shared prefix with string2, and the remainder.
    
    Args:
        string1: The string to split
        string2: The string to compare against
    
    Returns:
        tuple: (shared_part, non_shared_part)
    """
    # Find the length of the shared prefix
    shared_length = 0
    min_length = min(len(string1), len(string2))
    
    for i in range(min_length):
        if string1[i] == string2[i]:
            shared_length += 1
        else:
            break
    
    # Split the first string
    shared_part = string1[:shared_length]
    non_shared_part = string1[shared_length:]
    
    return shared_part, non_shared_part



class StrideGroupedSampler(Sampler):
    """Group """

    def __init__(
        self,
        batch_size: int,
        group: str,
        sort: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        
        if group is None:
            raise ValueError("Group cannot be None!")

        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            print(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        indices = list(range(len(lengths)))

        indice_stride_pairs = list(zip(indices, lengths))
        # NOTE: shuffle the indices in advance, otherwise the randomness may be lost when all length are equal
        random.shuffle(indice_stride_pairs)

        # sort data according to the number of strides
        indice_stride_pairs = sorted(indice_stride_pairs, key=lambda x: x[1])

        # group data instances with the same number of strides into the same batch
        batches = []
        batch = []
        prev_num_stride = None
        for index, num_stride in indice_stride_pairs:
            if num_stride != prev_num_stride:
                # in strict mode, all instances in the batch are forced to have the same number of strides
                if group == "strict":
                    batch.clear()
                elif group == "relaxed":
                    pass
                else:
                    raise ValueError(f"Group method {group} must be in None, strict, relaxed!")

            batch.append(index)
            prev_num_stride = num_stride

            if len(batch) == batch_size:
                batches.append((batch.copy(), num_stride))
                batch.clear()

        if len(batch) and group == "relaxed":
            batches.append((batch.copy(), num_stride))

        if sort is None:
            random.shuffle(batches)
        elif sort == "ascend":
            batches = sorted(batches, key=lambda x: x[1])
        elif sort == "descend":
            batches = sorted(batches, key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Sort method {sort} must be in None, ascend, descend!")

        batches = [x[0] for x in batches]
        self.indices = sum(batches, [])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)
