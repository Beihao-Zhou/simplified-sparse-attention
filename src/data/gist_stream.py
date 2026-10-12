"""Position-only gist insertion, shared by training data and decode-time insertion.

The original insertion looks ahead to word boundaries and appends trailing gists,
neither of which decoding can do. The rule here can:

* the first `sink` tokens are never chunked;
* a level-0 gist follows every `chunk_size` raw tokens;
* a level-l gist follows every J_l level-(l-1) gists (`group`: one int or a list);
* nothing else, so gisting a prefix gives a prefix of gisting any extension.

The offline helpers wrap `OnlineGistInserter`, so training and decoding cannot drift.
"""

from typing import List, Optional, Sequence, Tuple, Union

Groups = Union[int, Sequence[int]]


def level_groups(group: Groups, num_levels: int) -> List[int]:
    """[J_1, ..., J_{num_levels-1}]; an int applies to all levels, a short list repeats its last."""
    groups = [group] if isinstance(group, int) else list(group)
    if num_levels > 1 and not groups:
        raise ValueError("group must not be empty")
    groups = (groups + groups[-1:] * num_levels)[: max(num_levels - 1, 0)]
    if any(g <= 0 for g in groups):
        raise ValueError("group factors must be positive")
    return groups


class OnlineGistInserter:
    """Decides, token by token, which summary tokens to append after each raw token."""

    def __init__(self, gist_ids: Sequence[int], chunk_size: int, group: Groups, sink: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.gist_ids = list(gist_ids)
        self.chunk_size = chunk_size
        self.groups = level_groups(group, len(self.gist_ids))
        self.sink = sink
        self.n_raw = 0
        self.raw_in_chunk = 0                   # in the open level-0 chunk
        self.counts = [0] * len(self.gist_ids)  # summary tokens emitted per level

    def push(self, token: int) -> List[int]:
        """Account for one raw token; return the summary tokens to append after it."""
        if token in self.gist_ids:
            raise ValueError("push() takes raw tokens; summary tokens are emitted by the inserter")
        self.n_raw += 1
        if self.n_raw <= self.sink:
            return []
        self.raw_in_chunk += 1
        if self.raw_in_chunk < self.chunk_size:
            return []
        self.raw_in_chunk = 0
        out = [self.gist_ids[0]]
        self.counts[0] += 1
        for lvl in range(1, len(self.gist_ids)):
            if self.counts[lvl - 1] % self.groups[lvl - 1]:
                break
            out.append(self.gist_ids[lvl])
            self.counts[lvl] += 1
        return out


def insert_gists_with_map(
    token_ids: Sequence[int],
    gist_ids: Sequence[int],
    chunk_size: int,
    group: Groups,
    sink: int = 0,
    inserter: Optional[OnlineGistInserter] = None,
) -> Tuple[List[int], List[int]]:
    """Insert summary tokens by position.

    Returns (tokens, src) where src[i] is the index in token_ids of output token i,
    or -1 for an inserted summary token. Pass an existing `inserter` to continue a
    stream (e.g. a later turn) from where it stopped.
    """
    ins = inserter or OnlineGistInserter(gist_ids, chunk_size, group, sink)
    out, src = [], []
    for i, t in enumerate(token_ids):
        out.append(t)
        src.append(i)
        for g in ins.push(t):
            out.append(g)
            src.append(-1)
    return out, src


def insert_gists_by_position(
    token_ids: Sequence[int],
    gist_ids: Sequence[int],
    chunk_size: int,
    group: Groups,
    sink: int = 0,
) -> List[int]:
    """`insert_gists_with_map` without the source map."""
    return insert_gists_with_map(token_ids, gist_ids, chunk_size, group, sink)[0]


def stream_labels(tokens: Sequence[int], gist_ids: Sequence[int], loss_mask: Optional[Sequence[bool]] = None) -> List[int]:
    """-100 on summary tokens (inserted, never generated); raw tokens are predicted,
    including those right after a gist. loss_mask further restricts raw tokens."""
    gist = set(gist_ids)
    labels = []
    for i, t in enumerate(tokens):
        keep = t not in gist and (loss_mask is None or loss_mask[i])
        labels.append(t if keep else -100)
    return labels
