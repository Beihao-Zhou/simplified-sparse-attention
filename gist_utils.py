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
