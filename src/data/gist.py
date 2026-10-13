"""Gist token insertion and gist attention masks."""

import re
from typing import Optional, Tuple

import torch

# --- Gist insertion ---


def insert_gist_tokens_by_token_count(
    token_ids: list,
    gist_token_ids: list,
    chunk_size: int,
    attention_sink_size: int = 0,
) -> list:
    """Insert gist token every chunk_size tokens.

    Args:
        token_ids: List of token IDs
        gist_token_ids: The gist token IDs to insert
        chunk_size: Insert gist after every chunk_size tokens
        attention_sink_size: Don't insert gist in the first N tokens (they're attention sink)
    """
    assert chunk_size > 0, "chunk_size must be positive."
    gist_token_id = gist_token_ids[0]

    result = []
    for i, token_id in enumerate(token_ids):
        result.append(token_id)

        # Don't insert gist tokens in the attention sink region
        if i + 1 < attention_sink_size:
            continue

        # Insert gist token after every chunk_size tokens (but not at the very end)
        # Adjust for the attention sink offset
        adjusted_position = i + 1 - attention_sink_size
        if (
            adjusted_position > 0
            and adjusted_position % chunk_size == 0
            and (i + 1) < len(token_ids)
        ):
            result.append(gist_token_id)

    # Add final gist token at the end
    if len(token_ids) > 0:
        result.append(gist_token_id)

    return result


def insert_gist_tokens_word_aware(
    token_ids: list,
    gist_token_ids: list,
    chunk_size: int,
    tokenizer,
    attention_sink_size: int = 0,
    max_shift: int = None,
) -> list:
    """Insert gist tokens every chunk_size tokens, avoiding mid-word splits.

    This is a simplified version that checks if the insertion point would split
    a word. If so, it shifts forward until it finds a word boundary (a token
    that starts with a space or punctuation).

    Args:
        token_ids: List of token IDs
        gist_token_ids: The gist token IDs to insert
        chunk_size: Insert gist after every chunk_size tokens
        tokenizer: Tokenizer with decode method to check word boundaries
        attention_sink_size: Don't insert gist in the first N tokens
        max_shift: Maximum positions to shift forward (default: chunk_size//2)

    Returns:
        List of token IDs with gist tokens inserted at word boundaries

    Example:
        Without word awareness: [The, beau, GIST, tiful, cat]  # splits "beautiful"
        With word awareness:    [The, beautiful, GIST, cat]    # keeps word intact
    """
    assert chunk_size > 0, "chunk_size must be positive."

    if max_shift is None:
        max_shift = chunk_size // 2

    gist_token_id = gist_token_ids[0]

    def is_word_boundary(pos):
        """Check if position is at a word boundary (token starts with space/punctuation)."""
        if pos >= len(token_ids) or pos == 0:
            return True  # Beginning/end are always boundaries

        try:
            decoded = tokenizer.decode([token_ids[pos]], skip_special_tokens=False)
            if not decoded:
                return True
            first_char = decoded[0]
            return first_char in " \t\n\r" or first_char in ".,;:!?-—()[]{}\"'"
        except Exception:
            return True  # If decode fails, assume it's a boundary

    result = []
    i = 0
    cur_gist_pos = 0

    while i < len(token_ids):
        result.append(token_ids[i])

        # Don't insert gist tokens in the attention sink region
        if i + 1 < attention_sink_size:
            i += 1
            continue

        adjusted_position = i + 1 - attention_sink_size

        if (
            adjusted_position > 0
            and adjusted_position == chunk_size + cur_gist_pos
            and (i + 1) < len(token_ids)
        ):
            insert_pos = i + 1

            # If we're mid-word, shift forward to find word boundary
            if not is_word_boundary(insert_pos):
                for shift in range(1, max_shift + 1):
                    candidate_pos = insert_pos + shift
                    if candidate_pos >= len(token_ids):
                        break
                    if is_word_boundary(candidate_pos):
                        for j in range(shift):
                            if i + 1 + j < len(token_ids):
                                i += 1
                                result.append(token_ids[i])
                        break

            cur_gist_pos = i + 1 - attention_sink_size
            result.append(gist_token_id)

        i += 1

    # Add final gist token at the end
    if len(result) > 0 and result[-1] != gist_token_id:
        result.append(gist_token_id)

    return result


def insert_hierarchical_gist_tokens(
    token_ids: list,
    gist_token_ids: list,
    chunk_size: int,
    meta_chunk_size: int = 0,
    tokenizer=None,
    attention_sink_size: int = 0,
) -> list:
    """Insert hierarchical gist tokens with multiple levels.

    For example, with gist_token_ids=[g, G, GG], chunk_size=8 and meta_chunk_size=4:
    - Insert 'g' after every chunk_size regular tokens (word-aware)
    - Insert 'G' after every meta_chunk_size 'g' tokens
    - Insert 'GG' after every meta_chunk_size 'G' tokens
    - And so on for additional levels

    Args:
        token_ids: List of token IDs
        gist_token_ids: List of hierarchical gist token IDs [level0, level1, level2, ...]
        chunk_size: Insert a level-0 gist after every chunk_size raw tokens
        meta_chunk_size: Insert a higher-level gist after every meta_chunk_size lower-level gists
        attention_sink_size: Don't insert gist in the first N tokens (they're attention sink)

    Returns:
        Token list with hierarchical gist tokens inserted
    """
    if not gist_token_ids:
        return token_ids

    result = insert_gist_tokens_word_aware(
        token_ids,
        gist_token_ids,
        chunk_size,
        tokenizer,
        attention_sink_size,
    )

    for level in range(1, len(gist_token_ids)):
        lower_gist_token = gist_token_ids[level - 1]
        higher_gist_token = gist_token_ids[level]

        new_result = []
        lower_gist_count = 0

        for i, token_id in enumerate(result):
            new_result.append(token_id)

            if token_id == lower_gist_token:
                lower_gist_count += 1

                if lower_gist_count % meta_chunk_size == 0 and (i + 1) < len(result):
                    new_result.append(higher_gist_token)

        # Add final gist token at the end
        if len(token_ids) > 0:
            new_result.append(higher_gist_token)

        result = new_result

    return result


def _get_punctuation_token_ids(tokenizer) -> dict:
    """Extract punctuation token IDs from tokenizer.

    Returns:
        Dictionary with 'sentence', 'clause', and 'newline' token ID sets
    """
    try:
        # Common sentence-ending punctuation
        sentence_tokens = set()
        for punct in ['.', '!', '?', '.\n', '!\n', '?\n']:
            try:
                encoded = tokenizer.encode(punct, add_special_tokens=False)
                if encoded:
                    sentence_tokens.add(encoded[0])
            except:
                pass

        # Common clause-ending punctuation
        clause_tokens = set()
        for punct in [',', ';', ':', ',\n', ';\n', ':\n']:
            try:
                encoded = tokenizer.encode(punct, add_special_tokens=False)
                if encoded:
                    clause_tokens.add(encoded[0])
            except:
                pass

        # Newline tokens
        newline_tokens = set()
        for nl in ['\n', '\n\n', '\r\n']:
            try:
                encoded = tokenizer.encode(nl, add_special_tokens=False)
                if encoded:
                    newline_tokens.update(encoded)
            except:
                pass

        return {
            'sentence': sentence_tokens,
            'clause': clause_tokens,
            'newline': newline_tokens
        }
    except:
        # If tokenizer doesn't support encode, return empty sets
        return {'sentence': set(), 'clause': set(), 'newline': set()}


def _is_word_boundary(token_ids: list, pos: int, tokenizer) -> bool:
    """Check if position is a word boundary using token decoding.

    Args:
        token_ids: List of token IDs
        pos: Position to check
        tokenizer: Tokenizer for decoding

    Returns:
        True if this position starts a new word (token starts with space)
    """
    if pos >= len(token_ids) or pos == 0:
        return False

    try:
        # Decode the token at this position
        decoded = tokenizer.decode([token_ids[pos]], skip_special_tokens=False)
        # Check if it starts with whitespace (indicates word boundary)
        return len(decoded) > 0 and decoded[0] in [' ', '\t', '\n']
    except:
        return False


def _score_position_token_based(
    token_ids: list,
    pos: int,
    target_pos: int,
    punct_tokens: dict
) -> float:
    """Score a position based on token-level features (fast path).

    Args:
        token_ids: List of token IDs
        pos: Position to score
        target_pos: Target position (preference for positions closer to this)
        punct_tokens: Dictionary of punctuation token ID sets

    Returns:
        Score for this position (higher is better)
    """
    if pos >= len(token_ids):
        return 0.0

    token_id = token_ids[pos]
    base_score = 0.0

    # Sentence boundary (highest priority)
    if token_id in punct_tokens['sentence'] or token_id in punct_tokens['newline']:
        base_score = 100.0
    # Clause boundary (medium priority)
    elif token_id in punct_tokens['clause']:
        base_score = 50.0

    # Add distance penalty (prefer positions closer to target)
    distance_bonus = 1.0 / (1.0 + abs(pos - target_pos))

    return base_score + distance_bonus


def _score_position_decode_based(
    token_ids: list,
    pos: int,
    target_pos: int,
    tokenizer,
    context_window: int = 5
) -> float:
    """Score a position by decoding context and checking for semantic boundaries.

    Args:
        token_ids: List of token IDs
        pos: Position to score
        target_pos: Target position
        tokenizer: Tokenizer for decoding
        context_window: Number of tokens to decode around position

    Returns:
        Score for this position
    """
    if pos >= len(token_ids) or pos == 0:
        return 0.0

    try:
        # Decode a context window around this position
        start_idx = max(0, pos - context_window)
        end_idx = min(len(token_ids), pos + context_window)
        context = tokenizer.decode(token_ids[start_idx:end_idx], skip_special_tokens=False)

        # Calculate position in decoded text
        before_text = tokenizer.decode(token_ids[start_idx:pos], skip_special_tokens=False)
        split_pos = len(before_text)

        base_score = 0.0

        # Check for sentence boundaries
        # Pattern: sentence-ending punctuation followed by space and capital letter
        if split_pos > 0 and split_pos < len(context):
            before = context[max(0, split_pos-2):split_pos]
            after = context[split_pos:min(len(context), split_pos+2)]

            # Sentence boundary: . ! ? followed by space/newline and capital
            if re.search(r'[.!?]\s*$', before) and re.search(r'^\s*[A-Z]', after):
                base_score = 100.0
            # Paragraph break
            elif '\n\n' in context[max(0, split_pos-2):min(len(context), split_pos+2)]:
                base_score = 100.0
            # Clause boundary: , ; : followed by space
            elif re.search(r'[,;:]\s*$', before) and re.search(r'^\s+', after):
                base_score = 50.0
            # Single newline
            elif '\n' in context[max(0, split_pos-1):min(len(context), split_pos+1)]:
                base_score = 70.0

        # Distance bonus
        distance_bonus = 1.0 / (1.0 + abs(pos - target_pos))

        return base_score + distance_bonus
    except:
        return 0.0


def _find_best_boundary(
    token_ids: list,
    window_start: int,
    window_end: int,
    target_pos: int,
    tokenizer,
    punct_tokens: dict
) -> int:
    """Find the best position to insert a gist token within the search window.

    Uses a hybrid approach:
    1. Fast token-based scoring for all positions
    2. Decode-based fallback if no good boundaries found

    Args:
        token_ids: List of token IDs
        window_start: Start of search window
        window_end: End of search window
        target_pos: Target position (ideally where gist should go)
        tokenizer: Tokenizer for decoding
        punct_tokens: Dictionary of punctuation token ID sets

    Returns:
        Best position to insert gist token
    """
    scores = {}

    # Phase 1: Token-based scoring (fast)
    for pos in range(window_start, min(window_end, len(token_ids))):
        score = _score_position_token_based(token_ids, pos, target_pos, punct_tokens)
        if score > 1.0:  # Only keep positions with meaningful scores
            scores[pos] = score

    # Phase 2: If no good boundaries found, use decode-based scoring
    max_score = max(scores.values()) if scores else 0.0

    if max_score < 50.0:  # No sentence or clause boundaries found via tokens
        # Check word boundaries for all positions
        for pos in range(window_start, min(window_end, len(token_ids))):
            if _is_word_boundary(token_ids, pos, tokenizer):
                scores[pos] = 10.0 + 1.0 / (1.0 + abs(pos - target_pos))

    # Phase 3: Decode-based fallback for positions near target if still no good boundaries
    if max(scores.values()) if scores else 0.0 < 50.0:
        # Only decode a small window around the target for efficiency
        decode_window_start = max(window_start, target_pos - 10)
        decode_window_end = min(window_end, target_pos + 10)

        for pos in range(decode_window_start, min(decode_window_end, len(token_ids))):
            decode_score = _score_position_decode_based(token_ids, pos, target_pos, tokenizer)
            if decode_score > scores.get(pos, 0.0):
                scores[pos] = decode_score

    # Select position with highest score
    if scores:
        best_pos = max(scores.items(), key=lambda x: x[1])[0]
        return best_pos
    else:
        # Fallback: closest to target
        return min(range(window_start, min(window_end, len(token_ids))),
                   key=lambda x: abs(x - target_pos),
                   default=target_pos)


def insert_gist_tokens_semantic(
    token_ids: list,
    gist_token_ids: list,
    chunk_size: int,
    tokenizer,
    attention_sink_size: int = 0,
    chunk_size_flexibility: float = 0.5,
) -> list:
    """Insert gist tokens at semantic boundaries with flexible chunk sizes.

    This function respects semantic boundaries (sentences, clauses, words) when
    inserting gist tokens, rather than inserting at fixed intervals. It allows
    chunk sizes to vary within a range to find better insertion points.

    Args:
        token_ids: List of token IDs
        gist_token_ids: The gist token IDs to insert
        chunk_size: Target chunk size (actual chunks will be chunk_size ± flexibility)
        tokenizer: Tokenizer for decoding (must have encode/decode methods)
        attention_sink_size: Don't insert gist in the first N tokens (they're attention sink)
        chunk_size_flexibility: Flexibility as fraction of chunk_size (0.5 = ±50%)

    Returns:
        List of token IDs with gist tokens inserted at semantic boundaries

    Example:
        >>> # Instead of breaking mid-sentence:
        >>> # [The, quick, brown, GIST, fox, jumps, over, GIST, ...]
        >>> # Respects boundaries:
        >>> # [The, quick, brown, fox, ., GIST, Jumps, over, the, dog, ., GIST, ...]
    """
    assert chunk_size > 0, "chunk_size must be positive."
    assert 0 <= chunk_size_flexibility <= 1, "chunk_size_flexibility must be in [0, 1]"

    if len(token_ids) == 0:
        return []

    gist_token_id = gist_token_ids[0]

    # Pre-compute punctuation token IDs
    punct_tokens = _get_punctuation_token_ids(tokenizer)

    # Calculate flexibility window
    flexibility_range = int(chunk_size * chunk_size_flexibility)

    result = []
    current_pos = 0

    # Add attention sink tokens without gist
    while current_pos < min(attention_sink_size, len(token_ids)):
        result.append(token_ids[current_pos])
        current_pos += 1

    # Process remaining tokens with semantic boundary detection
    while current_pos < len(token_ids):
        # Calculate target position for next gist token
        target_pos = current_pos + chunk_size

        # Define search window
        window_start = max(current_pos + 1, target_pos - flexibility_range)
        window_end = min(len(token_ids), target_pos + flexibility_range + 1)

        # If we're near the end, adjust to avoid inserting gist at the very end
        if window_end >= len(token_ids) - 1:
            # Add remaining tokens
            while current_pos < len(token_ids):
                result.append(token_ids[current_pos])
                current_pos += 1
            break

        # Find best boundary position in window
        best_pos = _find_best_boundary(
            token_ids,
            window_start,
            window_end,
            target_pos,
            tokenizer,
            punct_tokens
        )

        # Add tokens up to and including the best position
        while current_pos < best_pos:
            result.append(token_ids[current_pos])
            current_pos += 1

        # Insert gist token after the boundary
        result.append(gist_token_id)

    # Add final gist token at the end
    if len(result) > 0 and result[-1] != gist_token_id:
        result.append(gist_token_id)

    return result


def get_gist_index(
    input_ids: torch.Tensor, gist_token: int, raise_if_no_tokens: bool = False
) -> Tuple[Optional[Tuple[int]], Optional[Tuple[int]]]:
    """Finds the contiguous spans of gist tokens in input_ids.

    Args:
        input_ids: tensor of input ids.
        gist_token: value of gist token.
        raise_if_no_tokens: raise an error if there are no gist tokens.

    Returns:
        A list of (start, end) spans of gist tokens, with exclusive end, if they
        exist, otherwise (None, None) if raise_if_no_tokens is False.

    Raises:
        ValueError: If no gist tokens are found and raise_if_no_tokens is True.
    """
    gist_indices = (input_ids == gist_token).nonzero().squeeze(-1)
    if len(gist_indices) == 0:
        if raise_if_no_tokens:
            raise ValueError(f"Could not find gist token {gist_token} in {input_ids}")
        return (None, None)

    gist_positions = gist_indices.tolist()
    spans = []
    start = prev = gist_positions[0]
    for curr in gist_positions[1:]:
        if curr != prev + 1:
            spans.append((start, prev + 1))
            start = curr
        prev = curr
    spans.append((start, prev + 1))
    return spans


# --- Attention masks ---


def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum from right to left.

    See https://github.com/pytorch/pytorch/issues/33520.

    Args:
        x: a tensor of shape (batch_size, seq_len)
    Returns:
        A tensor of shape (batch_size, seq_len) where each element is the sum of
        all elements to the right of it.
    """
    return x + torch.sum(x, dim=-1, keepdim=True) - torch.cumsum(x, dim=-1)


def make_mask_pre_first_gist(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Returns a mask where all tokens prior to the first gist token are masked out.
    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        gist_token: the integer id of the gist token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    mask = (inputs == gist_token).cumsum(-1) >= 1
    if pad_token is not None:
        mask = mask & (inputs != pad_token)
    return mask.type(dtype)


def make_mask_post_last_gist(
    inputs: torch.Tensor,
    gist_token: int,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Returns a mask where all tokens after the last gist token are masked out.
    Computes the same as mask_pre_first_gist_token but reverses the
    sequence before and after the cumsum.
    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        gist_token: the integer id of the gist token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    mask = reverse_cumsum(inputs == gist_token) >= 1
    if pad_token is not None:
        mask = mask & (inputs != pad_token)
    return mask.type(dtype)


def _chunk_ids(is_boundary: torch.Tensor) -> torch.Tensor:
    """Chunk index per token; a boundary gist belongs to the chunk it closes."""
    chunk_ids = is_boundary.cumsum(-1)
    return torch.clamp(torch.where(is_boundary, chunk_ids - 1, chunk_ids), min=0)


def _attention_sink_mask(
    inputs: torch.Tensor,
    first_gist_tokens: list,
    chunk_size: int,
    attention_sink_size: int,
) -> torch.Tensor:
    """(B, T, T) keys every query may attend to: the first attention_sink_size tokens, or, if
    attention_sink_size < 0, the tokens more than chunk_size before the first first_gist_tokens.
    """
    batch_size, seq_len = inputs.shape
    positions = torch.arange(seq_len, device=inputs.device)
    if attention_sink_size < 0:
        is_first_gist = torch.isin(inputs, torch.tensor(first_gist_tokens, device=inputs.device))
        first_gist_pos = torch.where(
            is_first_gist, positions[None, :], torch.tensor(seq_len, device=inputs.device)
        ).min(dim=1, keepdim=True)[0]
        is_sink = positions[None, :] < first_gist_pos - chunk_size
    elif attention_sink_size > 0:
        is_sink = positions[None, :] < attention_sink_size
    else:
        return torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=inputs.device)
    return is_sink[:, None, :].expand(-1, seq_len, -1)


def make_link_section_sink_mask(
    is_pre_gist_window: torch.Tensor,
    is_link_boundary: torch.Tensor,
    attention_sink_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Mark the first N tokens of each link section.

    Args:
        is_pre_gist_window: Boolean tensor of shape (batch_size, seq_len) indicating
                           tokens before the first link section
        is_link_boundary: Boolean tensor of shape (batch_size, seq_len) indicating
                         the LAST token of each link section (needs to shift right by 1)
        attention_sink_size: Number of tokens to mark at the start of each section
        device: Device for tensor operations

    Returns:
        Boolean tensor of shape (batch_size, seq_len) where True indicates the first
        N tokens of each link section
    """
    batch_size, seq_len = is_pre_gist_window.shape
    positions = torch.arange(seq_len, device=device)

    # Find the end position of pre_gist_window (start of first link section)
    first_link_section_start = torch.where(
        ~is_pre_gist_window, positions[None, :], torch.tensor(seq_len, device=device)
    ).min(dim=1, keepdim=True)[0]

    # is_link_boundary marks the LAST token of a section; shift to the FIRST of the next one.
    is_link_boundary_shifted = torch.zeros_like(is_link_boundary)
    is_link_boundary_shifted[:, 1:] = is_link_boundary[:, :-1]

    is_section_start = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    is_section_start.scatter_(1, first_link_section_start, True)
    is_section_start = is_section_start | is_link_boundary_shifted

    # Drop the last section start of each row
    last_true_pos = torch.where(
        is_section_start, positions[None, :], torch.tensor(-1, device=device)
    ).max(dim=1, keepdim=True)[0]
    is_section_start = is_section_start & (positions[None, :] != last_true_pos)

    is_first_n = is_section_start
    for n in range(1, attention_sink_size):
        shifted_starts = torch.zeros_like(is_section_start)
        shifted_starts[:, n:] = is_section_start[:, :-n]
        is_first_n = is_first_n | shifted_starts

    return is_first_n


def _apply_link_mask(
    mask: torch.Tensor,
    inputs: torch.Tensor,
    gist_token: list,
    link_tokens: list,
    chunk_size: int,
    attention_sink_size: int,
    first_gist_tokens: list,
) -> torch.Tensor:
    """Restrict a (B, 1, T, T) gist mask to link sections (multi-document RAG).

    Each link token closes a document section: until the last link token, tokens only see
    their own section; tokens after it see everything, including the link tokens. The
    window ending chunk_size + attention_sink_size before the first first_gist_tokens gist
    acts as a global sink.
    """
    seq_len = inputs.shape[1]
    device = inputs.device
    positions = torch.arange(seq_len, device=device)
    causal_mask = positions[:, None] >= positions[None, :]

    link_token_tensor = torch.tensor(link_tokens, device=device)
    is_any_link = torch.isin(inputs, link_token_tensor)
    is_any_gist = torch.isin(inputs, torch.tensor(gist_token, device=device))

    # Link tokens attend to all previous gist and link tokens
    link_attend_mask = (
        is_any_link[:, :, None] & (is_any_gist | is_any_link)[:, None, :] & causal_mask[None]
    )

    last_link_pos = torch.where(
        is_any_link, positions[None, :], torch.tensor(-1, device=device)
    ).max(dim=1, keepdim=True)[0]
    after_last_link = positions[None, :] > last_link_pos
    can_attend_to_link = after_last_link[:, :, None] & is_any_link[:, None, :]

    is_first_gist = torch.isin(inputs, torch.tensor(first_gist_tokens, device=device))
    first_gist_pos = torch.where(
        is_first_gist, positions[None, :], torch.tensor(seq_len, device=device)
    ).min(dim=1, keepdim=True)[0]
    is_pre_gist_window = positions[None, :] < first_gist_pos - chunk_size - attention_sink_size

    pre_gist_window_mask = is_pre_gist_window[:, None, :].expand(-1, seq_len, -1)
    mask = mask & ~pre_gist_window_mask[:, None, :, :]
    pre_gist_window_mask = pre_gist_window_mask & (
        after_last_link[:, :, None] | is_any_link[:, :, None] | is_pre_gist_window[:, :, None]
    )

    # Section boundaries: the last id of each group of 5 link tokens, else the largest id.
    if len(link_tokens) > 5:
        is_link_boundary = torch.isin(inputs, link_token_tensor.reshape(-1, 5)[:, -1])
    else:
        is_link_boundary = inputs == link_token_tensor.max().item()

    link_sections = is_link_boundary.cumsum(-1)
    same_link_section = (
        (link_sections[:, :, None] == link_sections[:, None, :])
        & ~(is_any_link[:, None, :])
        & ~(is_any_link[:, :, None])
    )
    link_section_mask = same_link_section | after_last_link[:, :, None]

    # The first attention_sink_size tokens of each section are visible to the whole section.
    link_section_sink_mask = make_link_section_sink_mask(
        is_pre_gist_window, is_link_boundary, attention_sink_size, device=device
    )
    link_section_sink_mask = link_section_sink_mask[:, None, :] & causal_mask[None, :, :]
    link_section_sink_mask = (link_section_sink_mask & link_section_mask) | (
        link_section_sink_mask & (after_last_link[:, :, None] | is_any_link[:, :, None])
    )

    return (
        mask & link_section_mask[:, None, :, :]
        | link_attend_mask[:, None, :, :]
        | can_attend_to_link[:, None, :, :]
        | pre_gist_window_mask[:, None, :, :]
        | link_section_sink_mask[:, None, :, :]
    )


def make_gist_mask(
    inputs: torch.Tensor,
    gist_token: list,
    chunk_size: int,
    attention_sink_size: int = 0,
    num_previous_chunks: int = 0,
    pad_token: Optional[int] = None,
    dtype=torch.bool,
    meta_gist: int = 0,
    link_tokens: Optional[list] = None,
    add_raw: bool = False,
) -> torch.Tensor:
    """Creates a 4D gist mask.
    Here, tokens after the last gist cannot attend to tokens prior to the first
    gist.
    Additionally, tokens *before* the last gist cannot attend to tokens *after*
    the last gist.

    Example with chunk_size=-1, where G is the gist token:

      a b c G d
    a 1 1 1 1 0
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0
    d 0 0 0 1 1

    Example with chunk_size=3:

      1 2 3 G 4 5 6 G 7 8 G
    1 1 1 1 1 0 0 0 0 0 0 0
    2 1 1 1 1 0 0 0 0 0 0 0
    3 1 1 1 1 0 0 0 0 0 0 0
    G 1 1 1 1 0 0 0 0 0 0 0
    4 0 0 0 1 1 1 1 1 0 0 0
    5 0 0 0 1 1 1 1 1 0 0 0
    6 0 0 0 1 1 1 1 1 0 0 0
    G 0 0 0 1 1 1 1 1 0 0 0
    7 0 0 0 1 0 0 0 1 1 1 1
    8 0 0 0 1 0 0 0 1 1 1 1
    G 0 0 0 1 0 0 0 1 1 1 1

    Args:
        inputs: Tensor of shape (batch_size, seq_len)
        gist_token: The list of gist token IDs
        chunk_size: Size of each chunk
        attention_sink_size: Number of initial tokens that all tokens can attend to (e.g., 4)
        num_previous_chunks: Number of previous chunks that all tokens can attend to (e.g., 1)
        pad_token: Optional padding token to mask out
        dtype: Output dtype

    Returns:
        Mask of shape (batch_size, 1, seq_len, seq_len)
    """
    assert chunk_size != -1, "chunk_size=-1 is deprecated."
    seq_len = inputs.shape[1]

    is_gist = inputs == gist_token[0]
    chunk_ids = _chunk_ids(is_gist)

    same_chunk = chunk_ids[:, :, None] == chunk_ids[:, None, :]
    chunk_diff = chunk_ids[:, :, None] - chunk_ids[:, None, :]
    within_n_chunks = (chunk_diff > 0) & (chunk_diff <= num_previous_chunks)

    positions = torch.arange(seq_len, device=inputs.device)
    causal_mask = positions[:, None] >= positions[None, :]  # query x key
    can_attend_to_prev_gist = is_gist[:, None, :] & causal_mask[None, :, :]

    attention_sink_mask = _attention_sink_mask(inputs, gist_token, chunk_size, attention_sink_size)

    mask = same_chunk | within_n_chunks | can_attend_to_prev_gist
    mask = mask[:, None, :, :]

    if link_tokens is not None:
        mask = _apply_link_mask(
            mask, inputs, gist_token, link_tokens, chunk_size, attention_sink_size, gist_token
        )

    if meta_gist == 0:  # l1 only: no token attends to meta-gist tokens
        for meta_gist_token in gist_token[1:]:
            mask = mask & ~(inputs == meta_gist_token)[:, None, None, :]
    elif meta_gist == 1:  # l2 only
        meta_gist_token = gist_token[1]
        pre_gist_mask = make_mask_post_last_gist(inputs, meta_gist_token, dtype=torch.bool)[
            :, None, None
        ]
        post_gist_mask = make_mask_pre_first_gist(inputs, meta_gist_token, dtype=torch.bool)[
            :, None, None
        ]
        pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))
        mask = mask & torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)
    elif meta_gist == 3:  # l1+l2+raw: queries after the last meta-gist attend to everything
        is_meta_gist = inputs == gist_token[1]
        last_meta_gist_pos = torch.where(
            is_meta_gist, positions[None, :], torch.tensor(-1, device=inputs.device)
        ).max(dim=1, keepdim=True)[0]
        is_after = (positions[None, :] > last_meta_gist_pos)[:, None, :, None]
        mask = torch.where(is_after, torch.ones_like(mask), mask)

    if link_tokens is None:
        mask = mask | attention_sink_mask[:, None, :, :]

    # If there are no gist tokens in an example, don't modify the mask
    has_gist = torch.isin(inputs, torch.tensor(gist_token, device=inputs.device)).any(-1)
    mask = torch.where(has_gist[:, None, None, None], mask, True)
    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]

    if add_raw:
        mask_0 = torch.ones_like(mask).masked_fill(is_gist[:, None, None, :], False)
        return torch.stack([mask_0, mask]).type(dtype)

    return mask.type(dtype)


def make_hierarchical_gist_mask_greedy(
    inputs: torch.Tensor,
    gist_token: list,
    chunk_size: int,
    attention_sink_size: int = 0,
    num_previous_chunks: int = 0,
    pad_token: Optional[int] = None,
    add_raw: bool = False,
    link_tokens: Optional[list] = None,
    dtype=torch.bool,
) -> torch.Tensor:
    """Creates a 4D hierarchical gist mask with support for multiple levels.

    Following the pattern of make_gist_mask, for each level i:
    - When processing level i, we only keep level-i gisting tokens and level-(i-1) tokens
    - If i=1, level-(i-1) tokens are raw tokens
    - Level i gist tokens can attend to:
      1. Level (i-1) tokens in the same chunk
      2. Previous level i gisting tokens AFTER the closest higher-level gist token
      3. The closest previous highest-level (largest level) gist token

    IMPORTANT:
    - Tokens can only attend to previous gist tokens of the same level that come
      AFTER the closest higher-level gist token. This ensures hierarchical sections are isolated.
    - ALL tokens at ALL levels can attend to gist tokens at the highest level that exists so far.
      "Highest level so far" means: for each position, look at all previous gist tokens and
      attend to those at the highest level among them. This provides a dynamic anchor point.
      Example: if only level-0 and level-1 gist tokens have appeared, attend to level-1;
               if level-2 gist tokens appear later, subsequent tokens attend to level-2.
    - BLOCKING RULE: If a token's level is lower than the highest-level gist token seen so far,
      ALL tokens BEFORE that highest-level gist token are blocked (not visible).
      The higher-level gist acts as a barrier, blocking access to everything before it.

    Example with 2 levels (g=level0, G=level1), chunk_size=3:

      1 2 3 g 4 5 6 g 7 8 9 g G 10 11 12 g 13 14 15 g

    For level 0 (g tokens and raw tokens):
      - Before G (positions 1-12): highest level so far is level-0, so no blocking
        Normal chunked attention: tokens attend to same chunk + previous g tokens
      - After G (positions 13-18): highest level so far is level-1 (G)
        * Can attend to G (the highest-level gist)
        * Can attend to tokens in same chunk (e.g., 13-15)
        * Can attend to g tokens after G (e.g., the g at position 16)
        * CANNOT attend to anything before G (positions 1-12 are blocked by G)

    For level 1 (G tokens):
      - When processing G tokens, ignore raw tokens, only keep g and G tokens
      - G at position 13 can attend to:
        * g tokens in current chunk (the 3rd 'g' at position 12)
        * Previous G tokens (none in this case)
      - If there were a level-2 gist GG before G:
        * G would attend to GG
        * G could NOT attend to anything before GG (blocked)

    Args:
        inputs: Tensor of shape (batch_size, seq_len)
        gist_token: List of gist token IDs [level0, level1, level2, ...]
        chunk_size: Size of each chunk (same across all levels)
        attention_sink_size: Number of initial tokens that all tokens can attend to
        num_previous_chunks: Number of previous chunks that tokens can attend to
        pad_token: Optional padding token to mask out
        dtype: Output dtype

    Returns:
        Mask of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, seq_len = inputs.shape
    device = inputs.device
    num_levels = len(gist_token)
    gist_masks = [inputs == token_id for token_id in gist_token]
    gist_token_tensor = torch.tensor(gist_token, device=device)

    positions = torch.arange(seq_len, device=device)
    causal_mask = positions[:, None] >= positions[None, :]

    attention_sink_mask = _attention_sink_mask(
        inputs, gist_token[:1], chunk_size, attention_sink_size
    )

    # Every token attends to earlier gists of each level within its section of higher-level
    # gists; the highest level is unsectioned.
    final_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
    for level in range(num_levels - 1, -1, -1):
        attend_to_this_level = gist_masks[level][:, None, :] & causal_mask[None, :, :]
        if level < num_levels - 1:
            higher_sections = torch.isin(inputs, gist_token_tensor[level + 1 :]).cumsum(-1)
            attend_to_this_level = attend_to_this_level & (
                higher_sections[:, :, None] == higher_sections[:, None, :]
            )
        final_mask = final_mask | attend_to_this_level

    # Level 0: raw tokens and level-0 gists, chunked by level-0 gists
    chunk_ids = _chunk_ids(gist_masks[0])
    same_chunk = chunk_ids[:, :, None] == chunk_ids[:, None, :]
    chunk_diff = chunk_ids[:, :, None] - chunk_ids[:, None, :]
    within_n_chunks = (chunk_diff > 0) & (chunk_diff <= num_previous_chunks)
    is_higher_level_gist = torch.isin(inputs, gist_token_tensor[1:])
    # Raw and level-0 keys only: a higher gist right after a level-0 gist falls in the next
    # chunk, and with 3+ levels it would stay visible after the gist above it closed it.
    is_level0 = ~is_higher_level_gist
    final_mask = final_mask | ((same_chunk | within_n_chunks) & is_level0[:, :, None] & is_level0[:, None, :])

    # Level i: level-i gists attend to the level-(i-1) gists of their chunk
    for level in range(1, num_levels):
        is_current_level_gist = gist_masks[level]
        is_prev_level_gist = gist_masks[level - 1]

        chunk_ids = _chunk_ids(is_current_level_gist)
        same_chunk = chunk_ids[:, :, None] == chunk_ids[:, None, :]
        level_i_mask = (
            same_chunk & is_prev_level_gist[:, None, :] & is_current_level_gist[:, :, None]
        )
        final_mask = final_mask | level_i_mask

        if num_previous_chunks > 0:
            chunk_diff = chunk_ids[:, :, None] - chunk_ids[:, None, :]
            within_n_chunks = (
                (chunk_diff > 0)
                & (chunk_diff <= num_previous_chunks)
                & is_prev_level_gist[:, None, :]
                & causal_mask[None, :, :]
            )
            is_higher_level_gist = torch.isin(inputs, gist_token_tensor[level + 1 :])
            final_mask = final_mask | (within_n_chunks & (~is_higher_level_gist)[:, :, None])

    if link_tokens is None:
        final_mask = final_mask | attention_sink_mask

    final_mask = final_mask[:, None, :, :]

    if link_tokens is not None:
        final_mask = _apply_link_mask(
            final_mask,
            inputs,
            gist_token,
            link_tokens,
            chunk_size,
            attention_sink_size,
            gist_token[:1],
        )

    # If there are no gist tokens in an example, return full causal mask
    has_gist = torch.isin(inputs, gist_token_tensor).any(-1)[:, None, None, None]
    full_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    final_mask = torch.where(has_gist, final_mask, full_mask)

    if pad_token is not None:
        final_mask = final_mask & (inputs != pad_token)[:, None, None]

    if add_raw:
        is_any_gist = torch.isin(inputs, gist_token_tensor)[:, None, None, :]
        final_mask_0 = torch.ones_like(final_mask).masked_fill(is_any_gist, False)
        return torch.stack([final_mask_0, final_mask]).type(dtype)

    return final_mask.type(dtype)


def add_raw_window(
    mask: torch.Tensor,
    inputs: torch.Tensor,
    gist_token: list,
    raw_window: int,
    pad_token: Optional[int] = None,
) -> torch.Tensor:
    """Dense counterpart of R6 in src/data/gist_flex_mask.py, on a bool [B, 1, T, T] mask."""
    if raw_window <= 0:
        return mask
    T = inputs.shape[1]
    band = torch.ones(T, T, dtype=torch.bool, device=inputs.device).tril()
    band &= ~band.tril(-raw_window)  # 0 <= q - k < raw_window
    is_raw = ~torch.isin(inputs, torch.tensor(list(gist_token), device=inputs.device))
    if pad_token is not None:
        is_raw = is_raw & (inputs != pad_token)
    win = band[None] & is_raw[:, :, None] & is_raw[:, None, :]
    return mask | win[:, None]


def make_hierarchical_gist_mask(
    inputs: torch.Tensor,
    gist_token: list,
    chunk_size: int,
    attention_sink_size: int = 0,
    num_previous_chunks: int = 0,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Creates a 4D hierarchical gist mask with support for multiple levels.

    Args:
        inputs: Tensor of shape (batch_size, seq_len)
        gist_token: List of gist token IDs [level0, level1, level2, ...]
        chunk_size: Size of each chunk (same across all levels)
        attention_sink_size: Number of initial tokens that all tokens can attend to
        num_previous_chunks: Number of previous chunks that tokens can attend to
        pad_token: Optional padding token to mask out
        dtype: Output dtype

    Returns:
        Mask of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, seq_len = inputs.shape
    device = inputs.device

    # Initialize final mask as zeros (will combine all level masks)
    final_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)

    # Identify all gist tokens at each level
    gist_masks = []
    for level, token_id in enumerate(gist_token):
        is_gist = (inputs == token_id)  # (batch_size, seq_len)
        gist_masks.append(is_gist)

    # Attention sink: all tokens can attend to first N tokens
    attention_sink_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
    if attention_sink_size > 0:
        assert pad_token is not None, "pad_token must be provided when using attention_sink_size > 0"
        is_not_padding = (inputs != pad_token)
        non_padding_cumsum = is_not_padding.cumsum(dim=1)
        is_sink_token = (non_padding_cumsum <= attention_sink_size) & is_not_padding
        attention_sink_mask = is_sink_token[:, None, :].expand(-1, seq_len, -1)

    positions = torch.arange(seq_len, device=device)
    causal_mask = positions[:, None] >= positions[None, :]

    # Recursively build hierarchical gist attendance

    # Level 0: Process raw tokens and level-0 gist tokens
    # This follows the same pattern as make_gist_mask for a single level
    is_level0_gist = gist_masks[0]  # (batch_size, seq_len)

    # Assign chunk IDs based on level 0 gist tokens
    chunk_ids = is_level0_gist.cumsum(-1)
    chunk_ids_adjusted = torch.where(is_level0_gist, chunk_ids - 1, chunk_ids)
    chunk_ids_adjusted = torch.clamp(chunk_ids_adjusted, min=0)

    # Tokens in same chunk can attend to each other
    same_chunk = chunk_ids_adjusted[:, :, None] == chunk_ids_adjusted[:, None, :]

    # Tokens can attend to tokens in previous N chunks
    chunk_diff = chunk_ids_adjusted[:, :, None] - chunk_ids_adjusted[:, None, :]
    within_n_chunks = (chunk_diff > 0) & (chunk_diff <= num_previous_chunks)

    can_attend_to_prev_gist = is_level0_gist[:, None, :] & causal_mask[None, :, :]

    is_higher_level_gist = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    for level in range(1, len(gist_token)):
        is_higher_level_gist = is_higher_level_gist | gist_masks[level]
    is_level0_token = ~is_higher_level_gist

    attend_to_prev_level_in_chunk = same_chunk & is_level0_token[:, None, :]

    # Level 0 tokens can attend to: same chunk, previous chunks, previous level-0 gist
    level0_mask = (attend_to_prev_level_in_chunk | within_n_chunks | can_attend_to_prev_gist) & is_level0_token[:, :, None]
    final_mask = final_mask | level0_mask

    # Process higher levels (level 1, 2, 3, ...)
    for level in range(1, len(gist_token)):
        is_current_level_gist = gist_masks[level]  # Level i gist tokens
        is_prev_level_gist = gist_masks[level - 1]  # Level i-1 gist tokens (treated as "raw tokens")

        # Assign chunk IDs based on level i gist tokens
        # This is the same pattern as level 0, but using level i gist tokens as chunk boundaries
        chunk_ids = is_current_level_gist.cumsum(-1)
        chunk_ids_adjusted = torch.where(is_current_level_gist, chunk_ids - 1, chunk_ids)
        chunk_ids_adjusted = torch.clamp(chunk_ids_adjusted, min=0)

        # Same chunk: level i gist tokens attend to level (i-1) tokens in same chunk
        same_chunk = chunk_ids_adjusted[:, :, None] == chunk_ids_adjusted[:, None, :]
        can_attend_to_prev_gist = is_current_level_gist[:, None, :] & causal_mask[None, :, :]

        attend_to_prev_level_in_chunk = same_chunk & is_prev_level_gist[:, None, :]

        level_i_mask = (attend_to_prev_level_in_chunk | can_attend_to_prev_gist) & is_current_level_gist[:, :, None]
        final_mask = final_mask | level_i_mask

    # For all the tokens after the last highest-level gist token, only allow them to attend to the highest-level gist token
    highest_level_gist = gist_masks[-1]  # (batch_size, seq_len)
    positions = torch.arange(seq_len, device=device)
    gist_positions = torch.where(highest_level_gist, positions[None, :], torch.tensor(-1, device=device))
    # Get the maximum position (last gist token) for each batch
    last_gist_positions = gist_positions.max(dim=1)[0]  # (batch_size,)
    # Create mask for tokens after last highest-level gist
    is_after_last_gist = positions[None, :] > last_gist_positions[:, None]
    final_mask = final_mask & ~is_after_last_gist[:, :, None]
    after_gist_mask = is_after_last_gist[:, :, None] & (highest_level_gist | is_after_last_gist)[:, None, :] & causal_mask[None, :, :]
    final_mask = final_mask | after_gist_mask

    # Add attention sink
    final_mask = final_mask | attention_sink_mask

    # Add batch and head dimensions
    final_mask = final_mask[:, None, :, :]  # (batch_size, 1, seq_len, seq_len)

    # If there are no gist tokens in an example, return full causal mask
    has_gist = torch.isin(inputs, torch.tensor(gist_token, device=device)).any(-1)[:, None, None, None]
    full_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    final_mask = torch.where(has_gist, final_mask, full_mask)

    # Mask out padding tokens
    if pad_token is not None:
        final_mask = final_mask & (inputs != pad_token)[:, None, None]

    return final_mask.type(dtype)


def make_raw_mask(
    inputs: torch.Tensor,
    gist_token: list,
    chunk_size: int = 0,
    attention_sink_size: int = 0,
    num_previous_chunks: int = 0,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
):
    """Creates a 4D pos control mask.
    Returns all ones (unaffected mask).

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        gist_token: the list of integer ids of the gist tokens.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, seq_len = inputs.shape
    mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)

    meta_gist_tokens = torch.tensor(gist_token, device=inputs.device)
    is_meta_gist = torch.isin(inputs, meta_gist_tokens)  # Shape: (batch_size, seq_len)
    # Set all columns corresponding to meta-gist tokens to 0
    meta_gist_mask = ~is_meta_gist[:, None, None, :]  # Shape: (batch_size, 1, 1, seq_len)
    mask = mask & meta_gist_mask

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def print_mask(mask, inputs, gist_token):
    """Pretty print with fixed-width columns."""
    seq_len = mask.shape[-1]
    tokens = inputs[0].tolist()
    
    # Fixed column width
    col_width = 3
    
    # Print header
    print("    ", end="")
    for token in tokens:
        if token == gist_token[0]:
            label = "g"
        elif len(gist_token) > 1 and token == gist_token[1]:
            label = "G"
        else:
            label = str(token)
        print(f"{label:>{col_width}}", end="")
    print()
    
    # Print each row
    for i, token in enumerate(tokens):
        # Row label
        if token == gist_token[0]:
            label = "g"
        elif len(gist_token) > 1 and token == gist_token[1]:
            label = "G"
        else:
            label = str(token)
        print(f"{label:>3} ", end="")
        
        # Mask values
        for j in range(seq_len):
            val = mask[0, 0, i, j].item()
            print(f"{val:>{col_width}}", end="")
        print()
