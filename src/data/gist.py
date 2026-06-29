"""Utilities for gist mask generation."""


from typing import Dict, Set, List, Optional, Tuple
import re

import torch


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
        adjusted_position = (i + 1 - attention_sink_size)
        if adjusted_position > 0 and adjusted_position % chunk_size == 0 and (i + 1) < len(token_ids):
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
            # Decode the token at this position
            decoded = tokenizer.decode([token_ids[pos]], skip_special_tokens=False)
            # Word boundary: starts with space, newline, tab, or is punctuation
            if not decoded:
                return True
            first_char = decoded[0]
            return first_char in ' \t\n\r' or first_char in '.,;:!?-—()[]{}"\''
        except:
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

        # Check if we should insert a gist token
        adjusted_position = (i + 1 - attention_sink_size)

        if adjusted_position > 0 and adjusted_position == chunk_size + cur_gist_pos and (i + 1) < len(token_ids):
            # Target position for gist insertion
            insert_pos = i + 1

            # If we're mid-word, shift forward to find word boundary
            if not is_word_boundary(insert_pos):
                # Shift forward up to max_shift positions
                for shift in range(1, max_shift + 1):
                    candidate_pos = insert_pos + shift
                    if candidate_pos >= len(token_ids):
                        break
                    if is_word_boundary(candidate_pos):
                        # Found a word boundary, add tokens up to here
                        for j in range(shift):
                            if i + 1 + j < len(token_ids):
                                i += 1
                                result.append(token_ids[i])
                        break

            # Insert gist token
            cur_gist_pos = (i + 1 - attention_sink_size)
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

    For example, with gist_token_ids=[g, G, GG] and chunk_size=8:
    - Insert 'g' after every chunk_size regular tokens
    - Insert 'G' after every chunk_size 'g' tokens
    - Insert 'GG' after every chunk_size 'G' tokens
    - And so on for additional levels

    Args:
        token_ids: List of token IDs
        gist_token_ids: List of hierarchical gist token IDs [level0, level1, level2, ...]
        chunk_size: Insert higher-level gist after every chunk_size lower-level gist tokens
        attention_sink_size: Don't insert gist in the first N tokens (they're attention sink)

    Returns:
        Token list with hierarchical gist tokens inserted
    """
    if not gist_token_ids:
        return token_ids

    # Start with level 0: insert base gist tokens
    # result = insert_gist_tokens_by_token_count(
    result = insert_gist_tokens_word_aware(
        token_ids,
        gist_token_ids,
        chunk_size,
        tokenizer,
        attention_sink_size
    )

    # For each additional level, insert higher-level gist tokens after every chunk_size
    # occurrences of the previous level's gist token
    for level in range(1, len(gist_token_ids)):
        lower_gist_token = gist_token_ids[level - 1]
        higher_gist_token = gist_token_ids[level]

        new_result = []
        lower_gist_count = 0

        for i, token_id in enumerate(result):
            new_result.append(token_id)

            # Count occurrences of lower-level gist tokens
            if token_id == lower_gist_token:
                lower_gist_count += 1

                # Insert higher-level gist after every meta_chunk_size lower-level gist tokens
                if lower_gist_count % meta_chunk_size == 0 and (i + 1) < len(result):
                    new_result.append(higher_gist_token)

        # Add final gist token at the end
        if len(token_ids) > 0:
            new_result.append(higher_gist_token)

        result = new_result

    return result


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

    # Identify all gist tokens at each level
    gist_masks = []
    for level, token_id in enumerate(gist_token):
        is_gist = (inputs == token_id)  # (batch_size, seq_len)
        gist_masks.append(is_gist)

    positions = torch.arange(seq_len, device=device)
    causal_mask = positions[:, None] >= positions[None, :]

    # Attention sink: all tokens can attend to first N tokens
    attention_sink_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
    if attention_sink_size < 0:  # Dynamic attention sink based on first gist token position
        is_any_l1_gist = torch.isin(inputs, torch.tensor(gist_token[0], device=inputs.device))  # (batch_size, seq_len)
        first_l1_gist_positions = torch.where(is_any_l1_gist, positions[None, :], torch.tensor(seq_len, device=inputs.device))
        first_gist_pos = first_l1_gist_positions.min(dim=1, keepdim=True)[0]  # (batch_size, 1)
        # Tokens before (first_gist_pos - chunk_size) can be attended by all tokens
        pre_gist_window_end = first_gist_pos - chunk_size
        is_pre_gist_window = positions[None, :] < pre_gist_window_end  # (batch_size, seq_len)
        attention_sink_mask = is_pre_gist_window[:, None, :].expand(-1, seq_len, -1)
    elif attention_sink_size > 0:  # Use given attention_sink_size
        is_pre_gist_window = positions[None, :] < attention_sink_size  # (batch_size, seq_len)
        # All tokens can attend to pre-gist window tokens
        attention_sink_mask = is_pre_gist_window[:, None, :].expand(-1, seq_len, -1)

    # Recursively build hierarchical gist attendance:
    # 1. First, find the highest-level gist token so far for each position and set to 1
    # 2. Then for each lower level, only attend to gist tokens between current token
    #    and the closest higher-level gist token
    # This is done recursively from highest to lowest level

    hierarchical_gist_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)

    # Start from highest level (len(gist_token) - 1) down to level 0
    for level in range(len(gist_token) - 1, -1, -1):
        is_this_level = gist_masks[level]  # (batch_size, seq_len)

        if level == len(gist_token) - 1:
            # Highest level: all tokens can attend to all previous highest-level gist tokens
            attend_to_this_level = is_this_level[:, None, :] & causal_mask[None, :, :]
        else:
            # Lower levels: only attend to gist tokens between current position and closest higher-level gist
            # Find sections defined by any higher-level gist tokens
            is_any_higher_level = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            for higher_level in range(level + 1, len(gist_token)):
                is_any_higher_level = is_any_higher_level | gist_masks[higher_level]

            # Create section IDs based on higher-level gist tokens
            # Tokens in the same section can attend to this level's gist tokens
            higher_sections = is_any_higher_level.cumsum(-1)  # (batch_size, seq_len)
            same_higher_section = higher_sections[:, :, None] == higher_sections[:, None, :]

            # Can attend to this level's gist tokens only within the same higher-level section
            attend_to_this_level = is_this_level[:, None, :] & causal_mask[None, :, :] & same_higher_section

        hierarchical_gist_mask = hierarchical_gist_mask | attend_to_this_level

    can_attend_to_highest_gist = hierarchical_gist_mask

    final_mask = can_attend_to_highest_gist.clone()

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

    gist_token_tensor = torch.tensor(gist_token, device=inputs.device)
    is_higher_level_gist = torch.isin(inputs, gist_token_tensor[1:])  # (batch_size, seq_len)
    raw_l0_mask = (same_chunk | within_n_chunks) & (~is_higher_level_gist)[:, :, None]
    final_mask = final_mask | raw_l0_mask

    # Process higher levels (level 1, 2, 3, ...)
    for level in range(1, len(gist_token)):
        # When processing level i:
        # - Only keep level-i gisting tokens and level-(i-1) gisting tokens
        # - Ignore all other tokens

        is_current_level_gist = gist_masks[level]  # Level i gist tokens
        is_prev_level_gist = gist_masks[level - 1]  # Level i-1 gist tokens (treated as "raw tokens")

        # Assign chunk IDs based on level i gist tokens
        # This is the same pattern as level 0, but using level i gist tokens as chunk boundaries
        chunk_ids = is_current_level_gist.cumsum(-1)
        chunk_ids_adjusted = torch.where(is_current_level_gist, chunk_ids - 1, chunk_ids)
        chunk_ids_adjusted = torch.clamp(chunk_ids_adjusted, min=0)

        # Same chunk: level i gist tokens attend to level (i-1) tokens in same chunk
        same_chunk = chunk_ids_adjusted[:, :, None] == chunk_ids_adjusted[:, None, :]
        attend_to_prev_level_in_chunk = same_chunk & is_prev_level_gist[:, None, :]

        level_i_mask = attend_to_prev_level_in_chunk & is_current_level_gist[:, :, None]
        final_mask = final_mask | level_i_mask

        if num_previous_chunks > 0:
            # Tokens can attend to tokens in previous N chunks
            chunk_diff = chunk_ids_adjusted[:, :, None] - chunk_ids_adjusted[:, None, :]
            within_n_chunks = (chunk_diff > 0) & (chunk_diff <= num_previous_chunks) & is_prev_level_gist[:, None, :] & causal_mask[None, :, :]

            is_higher_level_gist = torch.isin(inputs, gist_token_tensor[level+1:])  # (batch_size, seq_len)
            final_mask = final_mask | (within_n_chunks & (~is_higher_level_gist)[:, :, None])
    
    # Add attention sink
    if link_tokens is None:
        final_mask = final_mask | attention_sink_mask

    # Add batch and head dimensions
    final_mask = final_mask[:, None, :, :]  # (batch_size, 1, seq_len, seq_len)

    if link_tokens is not None:
        # Identify link token positions
        link_token_tensor = torch.tensor(link_tokens, device=inputs.device)
        is_any_link = torch.isin(inputs, link_token_tensor)  # (batch_size, seq_len)
        # Find positions after the last link token for each batch
        # All tokens after the last link can attend to link tokens
        gist_token_tensor = torch.tensor(gist_token, device=inputs.device)
        is_any_gist = torch.isin(inputs, gist_token_tensor)  # (batch_size, seq_len)

        # Link tokens can attend to all previous tokens (causal)
        link_attend_mask = is_any_link[:, :, None] & (is_any_gist | is_any_link)[:, None, :] & causal_mask[None, :, :]  # (batch_size, seq_len, seq_len)

        # Get last gist position per batch (-1 if no gist exists)
        link_positions = torch.where(is_any_link, positions[None, :], torch.tensor(-1, device=inputs.device))
        last_link_pos = link_positions.max(dim=1, keepdim=True)[0]  # (batch_size, 1)

        # Tokens after last gist (excluding gist itself) can attend to link tokens
        after_last_link = positions[None, :] > last_link_pos  # (batch_size, seq_len)
        can_attend_to_link = after_last_link[:, :, None] & is_any_link[:, None, :]  # (batch_size, seq_len, seq_len)

        # Tokens before or at last gist CANNOT attend to link tokens
        # This is already handled by NOT adding them to the mask

        # For the tokens before the first gist token window (position of first gist token - chunk_size),
        # they can be attended by all the tokens like the sink tokens
        # Find the first gist token position for each batch
        is_any_l1_gist = torch.isin(inputs, torch.tensor(gist_token[0], device=inputs.device))  # (batch_size, seq_len)
        first_l1_gist_positions = torch.where(is_any_l1_gist, positions[None, :], torch.tensor(seq_len, device=inputs.device))
        first_gist_pos = first_l1_gist_positions.min(dim=1, keepdim=True)[0]  # (batch_size, 1)
        # Tokens before (first_gist_pos - chunk_size) can be attended by all tokens
        pre_gist_window_end = first_gist_pos - chunk_size - attention_sink_size
        is_pre_gist_window = positions[None, :] < pre_gist_window_end  # (batch_size, seq_len)

        # All tokens can attend to pre-gist window tokens
        pre_gist_window_mask = is_pre_gist_window[:, None, :]  # (batch_size, 1, seq_len)
        pre_gist_window_mask = pre_gist_window_mask.expand(-1, seq_len, -1)  # (batch_size, seq_len, seq_len)
        final_mask = final_mask & ~pre_gist_window_mask[:, None, :, :]
        pre_gist_window_mask = pre_gist_window_mask & (after_last_link[:, :, None] | is_any_link[:, :, None] | is_pre_gist_window[:, :, None])

        # For each link token (if it is the list, consider the smallest id as the link token),
        # it separates the strings, which means, for tokens after the link token,
        # they cannot attend to tokens before the link token, until the last link token.

        # Use the smallest/largest link token ID to identify link token boundaries
        if len(link_tokens) > 5:
            boundary_link_token_id = link_token_tensor.reshape(-1, 5)[:, -1]
            is_link_boundary = torch.isin(inputs, boundary_link_token_id)
        else:
            link_token_id = link_token_tensor.max().item()
            is_link_boundary = (inputs == link_token_id)  # (batch_size, seq_len)

        # Assign section IDs based on link token boundaries
        # Each link token increments the section ID
        link_sections = is_link_boundary.cumsum(-1)  # (batch_size, seq_len)

        # Tokens can only attend to tokens in the same or later link section
        # (i.e., after the same link boundary or later)
        same_link_section = (link_sections[:, :, None] == link_sections[:, None, :]) & ~(is_any_link[:, None, :]) & ~(is_any_link[:, :, None]) # (batch_size, seq_len, seq_len)

        # BUT: tokens after the last link token can attend to ALL previous tokens
        # Apply section restriction only for query tokens that are before or at the last link
        link_section_mask = same_link_section | after_last_link[:, :, None]   # (batch_size, seq_len, seq_len)

        # For each link section, apply the attention sink, which means, the first attention_sink_size tokens can be attended by all tokens in that link section
        link_section_sink_mask = make_link_section_sink_mask(
            is_pre_gist_window,
            is_link_boundary,
            attention_sink_size,
            device=inputs.device
        )
        link_section_sink_mask = link_section_sink_mask[:, None, :] & causal_mask[None, :, :]
        link_section_sink_mask_1 = link_section_sink_mask & link_section_mask
        link_section_sink_mask_2 = link_section_sink_mask & (after_last_link[:, :, None] | is_any_link[:, :, None])
        link_section_sink_mask = link_section_sink_mask_1 | link_section_sink_mask_2

        # Add link token attention to the mask
        final_mask = final_mask & link_section_mask[:, None, :, :] | link_attend_mask[:, None, :, :] | can_attend_to_link[:, None, :, :] | pre_gist_window_mask[:, None, :, :] | link_section_sink_mask[:, None, :, :]

    # If there are no gist tokens in an example, return full causal mask
    has_gist = torch.isin(inputs, torch.tensor(gist_token, device=device)).any(-1)[:, None, None, None]
    full_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)
    final_mask = torch.where(has_gist, final_mask, full_mask)

    # Mask out padding tokens
    if pad_token is not None:
        final_mask = final_mask & (inputs != pad_token)[:, None, None]

    if add_raw:
        gist_token_tensor = torch.tensor(gist_token, device=inputs.device)
        is_any_gist = torch.isin(inputs, gist_token_tensor)[:, None, None, :]
        final_mask_0 = torch.ones_like(final_mask).masked_fill(is_any_gist, False)

        # last_meta_gist = gist_masks[-1] # [B, S]
        # # Find last meta-gist position for each batch element
        # positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]
        # meta_gist_positions = torch.where(last_meta_gist, positions, torch.tensor(-1, device=device))
        # last_meta_gist_pos = meta_gist_positions.max(dim=1)[0]  # [B]

        # # Create mask for positions to copy (before and including last meta-gist token)
        # # mask shape: [B, 1, S, 1] for broadcasting
        # change_mask = (positions[:, None, :, None] > last_meta_gist_pos[:, None, None, None])  # [B, 1, S, 1]
        # change_mask_l0 = change_mask & gist_masks[0][:, None, None, :]  # [B, 1, S, S]
        # change_mask_l1 = change_mask & last_meta_gist[:, None, None, :]  # [B, 1, S, S]
        # # final_mask_ = final_mask.masked_fill(change_mask_l1, False).masked_fill(change_mask_l0, True)
        # # final_mask_1 = final_mask.masked_fill(change_mask_l0, True)
        # final_mask_2 = final_mask.masked_fill(change_mask, True).masked_fill(change_mask_l1, False).masked_fill(change_mask_l0, False)

        return torch.stack([final_mask_0, final_mask]).type(dtype)
        # return final_mask_2.type(dtype)

    return final_mask.type(dtype)


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
        ~is_pre_gist_window,
        positions[None, :],
        torch.tensor(seq_len, device=device)
    ).min(dim=1, keepdim=True)[0]  # (batch_size, 1)

    # Shift is_link_boundary right by 1 position since it marks the LAST token of each section
    # We need to mark the FIRST token of the NEXT section
    is_link_boundary_shifted = torch.zeros_like(is_link_boundary)
    is_link_boundary_shifted[:, 1:] = is_link_boundary[:, :-1]  # Shift right by 1

    # Create a mask for section boundaries
    is_section_start = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    is_section_start.scatter_(1, first_link_section_start, True)
    is_section_start = is_section_start | is_link_boundary_shifted

    # Set the last True to False for each batch
    # Find the position of the last True in each batch
    last_true_pos = torch.where(
        is_section_start,
        positions[None, :],
        torch.tensor(-1, device=device)
    ).max(dim=1, keepdim=True)[0]  # (batch_size, 1)

    # Create a mask where the last True position is False
    is_not_last_true = positions[None, :] != last_true_pos  # (batch_size, seq_len)
    is_section_start = is_section_start & is_not_last_true

    # Shfit the section starts to mark the first N tokens of each section
    is_first_n = is_section_start
    for n in range(1, attention_sink_size):
        shifted_starts = torch.zeros_like(is_section_start)
        shifted_starts[:, n:] = is_section_start[:, :-n]  # Shift right by n
        is_first_n = is_first_n | shifted_starts
        
    return is_first_n


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
    # Attention mask for tokens before the last gist token.
    # Don't pass the pad token through for these first two masks, since we mask
    # out padding later.
    if chunk_size == -1:
        assert False, "chunk_size=-1 is deprecated."
    else:
        batch_size, seq_len = inputs.shape
        device = inputs.device

        # Chunked gisting behavior
        is_gist = (inputs == gist_token[0])
        
        # Assign chunk IDs: gist tokens belong to the chunk before them
        # Use cumsum of gist positions, so tokens after a gist get the next chunk ID
        chunk_ids = is_gist.cumsum(-1)  # Shape: (batch_size, seq_len)
        
        # For gist tokens themselves, assign them to the previous chunk
        # (decrement their chunk_id by 1, but keep >= 0)
        chunk_ids_adjusted = torch.where(is_gist, chunk_ids - 1, chunk_ids)
        chunk_ids_adjusted = torch.clamp(chunk_ids_adjusted, min=0)
        
        # Tokens can attend to others in the same chunk
        same_chunk = chunk_ids_adjusted[:, :, None] == chunk_ids_adjusted[:, None, :]
        
        # Tokens can attend to tokens in the previous N chunks        
        chunk_diff = chunk_ids_adjusted[:, :, None] - chunk_ids_adjusted[:, None, :]
        within_n_chunks = (chunk_diff > 0) & (chunk_diff <= num_previous_chunks)

        # Tokens can attend to all previous or current-position gist tokens
        positions = torch.arange(seq_len, device=inputs.device)
        causal_mask = positions[:, None] >= positions[None, :]  # query x key
        can_attend_to_prev_gist = is_gist[:, None, :] & causal_mask[None, :, :]

        # Attention sink: all tokens can attend to first N tokens
        attention_sink_mask = torch.zeros_like(same_chunk)
        if attention_sink_size < 0:  # Dynamic attention sink based on first gist token position
            gist_token_tensor = torch.tensor(gist_token, device=inputs.device)
            is_any_gist = torch.isin(inputs, gist_token_tensor)  # (batch_size, seq_len)
            first_l1_gist_positions = torch.where(is_any_gist, positions[None, :], torch.tensor(seq_len, device=inputs.device))
            first_gist_pos = first_l1_gist_positions.min(dim=1, keepdim=True)[0]  # (batch_size, 1)
            # Tokens before (first_gist_pos - chunk_size) can be attended by all tokens
            pre_gist_window_end = first_gist_pos - chunk_size
            is_pre_gist_window = positions[None, :] < pre_gist_window_end  # (batch_size, seq_len)

            # All tokens can attend to pre-gist window tokens
            attention_sink_mask = is_pre_gist_window[:, None, :]  # (batch_size, 1, seq_len)
            attention_sink_mask = attention_sink_mask.expand(-1, seq_len, -1)
        elif attention_sink_size > 0:  # Use given attention_sink_size
            is_pre_gist_window = positions[None, :] < attention_sink_size  # (batch_size, seq_len)
            # All tokens can attend to pre-gist window tokens
            attention_sink_mask = is_pre_gist_window[:, None, :]  # (batch_size, 1, seq_len)
            attention_sink_mask = attention_sink_mask.expand(-1, seq_len, -1)
                
        # Combine: same chunk OR attending to previous gist
        mask = same_chunk | within_n_chunks | can_attend_to_prev_gist
        mask = mask[:, None, :, :]

        if link_tokens is not None:
            # Identify link token positions
            link_token_tensor = torch.tensor(link_tokens, device=inputs.device)
            is_any_link = torch.isin(inputs, link_token_tensor)  # (batch_size, seq_len)
            # Find positions after the last link token for each batch
            # All tokens after the last link can attend to link tokens
            gist_token_tensor = torch.tensor(gist_token, device=inputs.device)
            is_any_gist = torch.isin(inputs, gist_token_tensor)  # (batch_size, seq_len)

            # Link tokens can attend to all previous tokens (causal)
            link_attend_mask = is_any_link[:, :, None] & (is_any_gist | is_any_link)[:, None, :] & causal_mask[None, :, :]  # (batch_size, seq_len, seq_len)

            # Get last gist position per batch (-1 if no gist exists)
            link_positions = torch.where(is_any_link, positions[None, :], torch.tensor(-1, device=inputs.device))
            last_link_pos = link_positions.max(dim=1, keepdim=True)[0]  # (batch_size, 1)

            # Tokens after last gist (excluding gist itself) can attend to link tokens
            after_last_link = positions[None, :] > last_link_pos  # (batch_size, seq_len)
            can_attend_to_link = after_last_link[:, :, None] & is_any_link[:, None, :]  # (batch_size, seq_len, seq_len)

            # Tokens before or at last gist CANNOT attend to link tokens
            # This is already handled by NOT adding them to the mask

            # For the tokens before the first gist token window (position of first gist token - chunk_size),
            # they can be attended by all the tokens like the sink tokens
            # Find the first gist token position for each batch
            first_l1_gist_positions = torch.where(is_any_gist, positions[None, :], torch.tensor(seq_len, device=inputs.device))
            first_gist_pos = first_l1_gist_positions.min(dim=1, keepdim=True)[0]  # (batch_size, 1)
            # Tokens before (first_gist_pos - chunk_size) can be attended by all tokens
            pre_gist_window_end = first_gist_pos - chunk_size - attention_sink_size
            is_pre_gist_window = positions[None, :] < pre_gist_window_end  # (batch_size, seq_len)

            # All tokens can attend to pre-gist window tokens
            pre_gist_window_mask = is_pre_gist_window[:, None, :]  # (batch_size, 1, seq_len)
            pre_gist_window_mask = pre_gist_window_mask.expand(-1, seq_len, -1)  # (batch_size, seq_len, seq_len)
            mask  = mask & ~pre_gist_window_mask[:, None, :, :]
            pre_gist_window_mask = pre_gist_window_mask & (after_last_link[:, :, None] | is_any_link[:, :, None] | is_pre_gist_window[:, :, None])

            # For each link token (if it is the list, consider the smallest id as the link token),
            # it separates the strings, which means, for tokens after the link token,
            # they cannot attend to tokens before the link token, until the last link token.

            # Use the smallest/largest link token ID to identify link token boundaries
            if len(link_tokens) > 5:
                boundary_link_token_id = link_token_tensor.reshape(-1, 5)[:, -1]
                is_link_boundary = torch.isin(inputs, boundary_link_token_id)
            else:
                link_token_id = link_token_tensor.max().item()
                is_link_boundary = (inputs == link_token_id)  # (batch_size, seq_len)

            # Assign section IDs based on link token boundaries
            # Each link token increments the section ID
            link_sections = is_link_boundary.cumsum(-1)  # (batch_size, seq_len)

            # Tokens can only attend to tokens in the same or later link section
            # (i.e., after the same link boundary or later)
            same_link_section = (link_sections[:, :, None] == link_sections[:, None, :]) & ~(is_any_link[:, None, :]) & ~(is_any_link[:, :, None]) # (batch_size, seq_len, seq_len)

            # BUT: tokens after the last link token can attend to ALL previous tokens
            # Apply section restriction only for query tokens that are before or at the last link
            link_section_mask = same_link_section | after_last_link[:, :, None]   # (batch_size, seq_len, seq_len)

            # For each link section, apply the attention sink, which means, the first attention_sink_size tokens can be attended by all tokens in that link section
            link_section_sink_mask = make_link_section_sink_mask(
                is_pre_gist_window,
                is_link_boundary,
                attention_sink_size,
                device=inputs.device
            )
            link_section_sink_mask = link_section_sink_mask[:, None, :] & causal_mask[None, :, :]
            link_section_sink_mask_1 = link_section_sink_mask & link_section_mask
            link_section_sink_mask_2 = link_section_sink_mask & (after_last_link[:, :, None] | is_any_link[:, :, None])
            link_section_sink_mask = link_section_sink_mask_1 | link_section_sink_mask_2

            # Add link token attention to the mask
            mask = mask & link_section_mask[:, None, :, :] | link_attend_mask[:, None, :, :] | can_attend_to_link[:, None, :, :] | pre_gist_window_mask[:, None, :, :] | link_section_sink_mask[:, None, :, :]

        if meta_gist == 0:  # l1 only
            if len(gist_token) > 1:
                for m_i in range(1, len(gist_token)):
                    meta_gist_token = gist_token[m_i]
                    is_meta_gist = (inputs == meta_gist_token)  # Shape: (batch_size, seq_len)
                    # Set all columns corresponding to meta-gist tokens to 0
                    # This means no token can attend to meta-gist tokens
                    meta_gist_mask = ~is_meta_gist[:, None, None, :]  # Shape: (batch_size, 1, 1, seq_len)
                    mask = mask & meta_gist_mask
        elif meta_gist == 1:  # l2 only
            # Handle multiple gist tokens by expanding the mask to include all gist tokens
            # assert len(gist_token) == 2, "Currently only support two gist tokens: "
            # Identify meta gist tokens (second gist token type)
            meta_gist_token = gist_token[1]
            # Original behavior
            pre_gist_mask = make_mask_post_last_gist(inputs, meta_gist_token, dtype=torch.bool)[
                :, None, None
            ]
            # Attention mask for tokens after the last gist token.
            post_gist_mask = make_mask_pre_first_gist(inputs, meta_gist_token, dtype=torch.bool)[
                :, None, None
            ]
            # Construct time masks by permuting to time dimension.
            pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))
            meta_gist_mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)
            mask = mask & meta_gist_mask
        elif meta_gist == 3:  # l1+l2+raw
            meta_gist_token = gist_token[1]
            is_meta_gist = (inputs == meta_gist_token)  # Shape: (batch_size, seq_len)
            
            # Find the position of the last meta-gist token for each batch
            seq_len = inputs.size(1)
            positions = torch.arange(seq_len, device=inputs.device)
            
            # Get last meta-gist position per batch (-1 if no meta-gist exists)
            meta_gist_positions = torch.where(is_meta_gist, positions[None, :], torch.tensor(-1, device=inputs.device))
            last_meta_gist_pos = meta_gist_positions.max(dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)
            
            # Tokens after last meta-gist can attend to everything
            after_meta_gist = positions[None, :] > last_meta_gist_pos  # Shape: (batch_size, seq_len)
            
            # Where query is after meta-gist, set entire row to True
            is_after = after_meta_gist[:, None, :, None]  # Shape: (batch_size, 1, seq_len, 1)
            mask = torch.where(is_after, torch.ones_like(mask), mask)

        if link_tokens is None:
            mask = mask | attention_sink_mask[:, None, :, :]

    # If there are no gist tokens in an example, don't modify the mask
    # has_gist = (inputs == gist_token[0]).any(-1)[:, None, None, None]
    has_gist = torch.isin(inputs, torch.tensor(gist_token, device=inputs.device)).any(-1)[:, None, None, None]
    mask = torch.where(has_gist, mask, True)
    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]

    if add_raw:
        change_mask_l0_ = is_gist[:, None, None, :]
        mask_0 = torch.ones_like(mask).masked_fill(change_mask_l0_, False)

        # # Find last meta-gist position for each batch element
        # positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]
        # gist_positions = torch.where(is_gist, positions, torch.tensor(-1, device=device))
        # last_gist_pos = gist_positions.max(dim=1)[0]  # [B]

        # # Create mask for positions to copy (before and including last meta-gist token)
        # # mask shape: [B, 1, S, 1] for broadcasting
        # change_mask = (positions[:, None, :, None] > last_gist_pos[:, None, None, None])  # [B, 1, S, 1]
        # change_mask_l0 = change_mask & is_gist[:, None, None, :]
        # mask_1 = mask.masked_fill(change_mask, True).masked_fill(change_mask_l0, False)

        return torch.stack([mask_0, mask]).type(dtype)
        # return mask_.type(dtype)
    
    return mask.type(dtype)


def make_neg_control_mask(
    inputs: torch.Tensor,
    gist_token: list,
    chunk_size: int,
    attention_sink_size: int = 0,
    num_previous_chunks: int = 0,
    pad_token: Optional[int] = None,
    dtype=torch.int64,
):
    """Creates a 4D neg control mask.
    Here, tokens after the last gist cannot attend to any gist tokens (or prior).

    Example, where G is the gist token:

      a b c G d
    a 1 1 1 1 0
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0
    d 0 0 0 0 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        gist_token: the list of integer ids of the gist tokens.
        chunk_size: Not used in this function.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Attention mask for tokens before the last gist token.
    # Don't pass the pad token through for these first two masks, since we mask
    # out padding later.
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token[0], dtype=torch.bool)[
        :, None, None
    ]
    # Attention mask for tokens after the last gist token. This creates a mask
    # that is zero for all tokens up to and including the last gist token.
    post_gist_mask = torch.logical_not(pre_gist_mask)
    # Construct time masks by permuting to time dimension.
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)
    # If there are no gist tokens in an example, don't modify the mask (return
    # all ones)
    has_gist = (inputs == gist_token[0]).any(-1)[:, None, None, None]
    mask = torch.where(has_gist, mask, True)

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def make_pos_control_mask(
    inputs: torch.Tensor,
    gist_token: list,
    chunk_size: int = -1,
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
        chunk_size: Not used in this function.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    del gist_token
    batch_size, seq_len = inputs.shape
    mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


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


def get_gist_index(
    input_ids: torch.Tensor, gist_token: int, raise_if_no_tokens: bool = False
) -> Tuple[Optional[Tuple[int]], Optional[Tuple[int]]]:
    """Finds the start and end of the gist span in input_ids.

    Args:
        input_ids: tensor of input ids.
        gist_token: value of gist token.
        raise_if_no_tokens: raise an error if there are no gist tokens.

    Returns:
        (start, end) of gist token(s), with exclusive end, if they exist,
        otherwise (None, None) if raise_if_no_tokens is False (raises
        error if True).

    Raises:
        RuntimeError: If the gist tokens in the input are not a contiguous span.
        ValueError: If no gist tokens are found and raise_if_no_tokens is True.
    """
    gist_indices = (input_ids == gist_token).nonzero().squeeze(-1)
    if len(gist_indices) == 0:
        if raise_if_no_tokens:
            raise ValueError(f"Could not find gist token {gist_token} in {input_ids}")
        return (None, None)
    # Assert that the gist indices are a single continuous sequence.
    # _assert_continguous_span(gist_indices)

    # Group contiguous gist indices
    gist_positions = gist_indices.tolist()
    spans = []
    
    start = gist_positions[0]
    prev = gist_positions[0]
    
    for i in range(1, len(gist_positions)):
        curr = gist_positions[i]
        # If not contiguous, close the current span and start a new one
        if curr != prev + 1:
            spans.append((start, prev + 1))
            start = curr
        prev = curr
    
    # Add the last span
    spans.append((start, prev + 1))
    
    return spans


def get_first_pad_index(input_ids: torch.Tensor, pad_token: int) -> int:
    """Finds the index of the first pad token in input_ids.

    Args:
        input_ids: tensor of input ids.
        pad_token: value of pad token.

    Returns:
        index of pad token if exists, otherwise len(input_ids).
    """
    pad_indices = (input_ids == pad_token).nonzero()
    if len(pad_indices) == 0:
        return len(input_ids)
    return pad_indices[0].item()


def _assert_continguous_span(gist_indices: torch.Tensor):
    """Assert that the gist indices form a contiguous span."""
    gist_start = gist_indices[0]
    gist_indices_arange = torch.arange(
        start=gist_start,
        end=gist_start + len(gist_indices),
        device=gist_indices.device,
    )
    if not (gist_indices == gist_indices_arange).all():
        raise RuntimeError(f"gist tokens do not form a contiguous span: {gist_indices}")






# --- Test --- 

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


if __name__ == "__main__":

    def test_semantic_gist_insertion():
        """Test semantic-aware gist token insertion."""
        print("\n" + "="*80)
        print("Testing semantic gist token insertion")
        print("="*80)

        # Mock tokenizer for testing
        class MockTokenizer:
            def __init__(self):
                # Create a simple token mapping
                self.token_map = {
                    1: "The", 2: " quick", 3: " brown", 4: " fox", 5: ".",
                    6: " Jumps", 7: " over", 8: " the", 9: " lazy", 10: " dog", 11: ".",
                    12: " She", 13: " sells", 14: " sea", 15: "shells", 16: ",",
                    17: " by", 18: " the", 19: " sea", 20: "shore", 21: ".",
                    22: "\n", 23: " A", 24: " new", 25: " paragraph", 26: " here", 27: "."
                }
                self.reverse_map = {v: k for k, v in self.token_map.items()}

            def encode(self, text, add_special_tokens=False):
                # Simple encoding for punctuation
                if text == '.':
                    return [5]
                elif text == ',':
                    return [16]
                elif text == '\n':
                    return [22]
                return []

            def decode(self, token_ids, skip_special_tokens=False):
                return ''.join([self.token_map.get(tid, f"[{tid}]") for tid in token_ids])

        tokenizer = MockTokenizer()

        # Test case 1: Simple sentence boundaries
        print("\nTest 1: Two sentences with fixed chunk_size=8")
        tokens_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # "The quick brown fox. Jumps over the lazy dog."
        gist_token = [999]
        chunk_size = 8

        # Original method (breaks mid-sentence)
        result_fixed = insert_gist_tokens_by_token_count(tokens_1, gist_token, chunk_size)
        print(f"Original tokens: {tokens_1}")
        print(f"Fixed insertion:  {result_fixed}")
        print(f"Decoded (fixed):  {tokenizer.decode([t for t in result_fixed if t != 999])}")
        print(f"Gist positions (fixed): {[i for i, t in enumerate(result_fixed) if t == 999]}")

        # Semantic method (respects sentence boundaries)
        result_semantic = insert_gist_tokens_semantic(tokens_1, gist_token, chunk_size, tokenizer, chunk_size_flexibility=0.5)
        print(f"Semantic insertion: {result_semantic}")
        print(f"Decoded (semantic): {tokenizer.decode([t for t in result_semantic if t != 999])}")
        print(f"Gist positions (semantic): {[i for i, t in enumerate(result_semantic) if t == 999]}")

        # Test case 2: Multiple sentences with clause boundaries
        print("\n\nTest 2: Sentence with comma (clause boundary)")
        tokens_2 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # "She sells seashells, by the seashore."
        result_semantic_2 = insert_gist_tokens_semantic(tokens_2, gist_token, chunk_size=6, tokenizer=tokenizer, chunk_size_flexibility=0.5)
        print(f"Original tokens: {tokens_2}")
        print(f"Semantic insertion: {result_semantic_2}")
        print(f"Decoded: {tokenizer.decode([t for t in result_semantic_2 if t != 999])}")
        print(f"Gist positions: {[i for i, t in enumerate(result_semantic_2) if t == 999]}")

        # Test case 3: Paragraph break
        print("\n\nTest 3: Paragraph break (newline)")
        tokens_3 = [1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27]  # "The quick brown fox.\n A new paragraph here."
        result_semantic_3 = insert_gist_tokens_semantic(tokens_3, gist_token, chunk_size=7, tokenizer=tokenizer, chunk_size_flexibility=0.5)
        print(f"Original tokens: {tokens_3}")
        print(f"Semantic insertion: {result_semantic_3}")
        print(f"Decoded: {tokenizer.decode([t for t in result_semantic_3 if t != 999])}")
        print(f"Gist positions: {[i for i, t in enumerate(result_semantic_3) if t == 999]}")

        # Test case 4: Attention sink
        print("\n\nTest 4: With attention sink (first 3 tokens protected)")
        result_semantic_4 = insert_gist_tokens_semantic(tokens_1, gist_token, chunk_size=5, tokenizer=tokenizer, attention_sink_size=3, chunk_size_flexibility=0.5)
        print(f"Original tokens: {tokens_1}")
        print(f"Semantic insertion: {result_semantic_4}")
        print(f"Decoded: {tokenizer.decode([t for t in result_semantic_4 if t != 999])}")
        print(f"Gist positions: {[i for i, t in enumerate(result_semantic_4) if t == 999]}")
        print(f"Note: First 3 tokens are attention sink (no gist inserted)")

        # Test case 5: Flexibility comparison
        print("\n\nTest 5: Flexibility comparison")
        tokens_5 = list(range(1, 21))  # 20 tokens
        print(f"Original tokens: {tokens_5}")

        for flex in [0.0, 0.3, 0.5]:
            result = insert_gist_tokens_semantic(tokens_5, gist_token, chunk_size=8, tokenizer=tokenizer, chunk_size_flexibility=flex)
            gist_positions = [i for i, t in enumerate(result) if t == 999]
            chunk_sizes = [gist_positions[0]] + [gist_positions[i] - gist_positions[i-1] for i in range(1, len(gist_positions))]
            print(f"  Flexibility={flex}: gist at {gist_positions}, chunk sizes: {chunk_sizes}")

        print("\n" + "="*80)

    def test_hierarchical_gist_insertion():
        """Test hierarchical gist token insertion."""
        # Create a simple example: 40 tokens with chunk_size=4, attention_sink=2
        tokens = list(range(1, 41))  # [1, 2, ..., 40]
        gist_token_ids = [1000, 2000, 3000]  # [g, G, GG]
        chunk_size = 4
        meta_chunk_size = 2
        attention_sink_size = 2

        result = insert_hierarchical_gist_tokens(
            tokens,
            gist_token_ids,
            chunk_size,
            meta_chunk_size,
            attention_sink_size
        )

        print("Hierarchical gist token insertion test:")
        print(f"Original tokens: {tokens[:20]}...")
        print(f"Gist token IDs: g={gist_token_ids[0]}, G={gist_token_ids[1]}, GG={gist_token_ids[2]}")
        print(f"Chunk size: {chunk_size}, Attention sink: {attention_sink_size}")
        print(f"\nResult (showing token type):")

        # Pretty print showing token types
        display = []
        for token in result:
            if token == gist_token_ids[0]:
                display.append('g')
            elif token == gist_token_ids[1]:
                display.append('G')
            elif token == gist_token_ids[2]:
                display.append('GG')
            else:
                display.append('x')

        # Print in groups of 20 for readability
        for i in range(0, len(display), 20):
            print(''.join(f'{d:>4}' for d in display[i:i+20]))

        print(f"\nTotal tokens: {len(result)} (original: {len(tokens)})")

        # Count each type
        g_count = sum(1 for t in result if t == gist_token_ids[0])
        G_count = sum(1 for t in result if t == gist_token_ids[1])
        GG_count = sum(1 for t in result if t == gist_token_ids[2])
        print(f"Gist token counts: g={g_count}, G={G_count}, GG={GG_count}")

        return result

    def test_gist_mask():
        """Test gist mask generation."""
        # Test insert_gist_tokens_by_token_count
        tokens = [10, 11, 12, 13, 14, 15, 16]
        gist_token_id = [999]
        chunk_size = 3

        result = insert_gist_tokens_by_token_count(tokens, gist_token_id, chunk_size, attention_sink_size=2)
        expected = [10, 11, 12, 13, 14, 999, 15, 16, 999]
        assert result == expected, f"Expected {expected}, got {result}"

        result = insert_gist_tokens_by_token_count(tokens, gist_token_id, chunk_size, attention_sink_size=0)
        expected = [10, 11, 12, 999, 13, 14, 15, 999, 16, 999]
        assert result == expected, f"Expected {expected}, got {result}"
        print("insert_gist_tokens_by_token_count passed.")
        # Test make_gist_mask
        expected = [0, 0, 10, 11, 12, 999, 13, 14, 15, 999, 16, 999]
        input_ids = expected + [9999] + [17] * 5  # Pad with some extra tokens
        input_tensor = torch.tensor([input_ids])
        gist_token = [999, 9999]

        # # original gisting
        # mask = make_gist_mask(input_tensor, gist_token, -1, 2, 0, meta_gist=0)
        # print_mask(mask, input_tensor, gist_token)
        # l1 only
        mask = make_gist_mask(input_tensor, gist_token, chunk_size, 2, 0, pad_token=0, meta_gist=0)
        print("L1 only mask:")
        print_mask(mask, input_tensor, gist_token)
        # l2 only
        mask = make_gist_mask(input_tensor, gist_token, chunk_size, 2, 0, pad_token=0, meta_gist=1)
        print("L2 only mask:")
        print_mask(mask, input_tensor, gist_token)
        # raw only
        mask = make_raw_mask(input_tensor, gist_token)
        print("Raw only mask:")
        print_mask(mask, input_tensor, gist_token)
        # l1+l2
        mask = make_gist_mask(input_tensor, gist_token, chunk_size, 2, 0, pad_token=0, meta_gist=2)
        print("L1+L2 mask:")
        print_mask(mask, input_tensor, gist_token)
        # l1+l2+raw
        mask = make_gist_mask(input_tensor, gist_token, chunk_size, 2, 0, pad_token=0, meta_gist=3)
        print("L1+L2+Raw mask:")
        print_mask(mask, input_tensor, gist_token)
        # input_tensor = torch.tensor([[10, 11, 12, 13, 999, 9999, 14, 15, 16]])
        # mask = make_gist_mask(input_tensor, gist_token, chunk_size, 2, 1, meta_gist=1)
        # print_mask(mask, input_tensor, gist_token)

    def test_get_gist_index():
        # Test 1: [1 2 3 4 G G 5 6 7 G G]
        input_ids1 = torch.tensor([1, 2, 3, 4, 999, 999, 5, 6, 7, 999, 999])
        result1 = get_gist_index(input_ids1, gist_token=[999])
        print(f"Test 1: {result1}")
        assert result1 == [(4, 6), (9, 11)], f"Expected [(4, 6), (9, 11)], got {result1}"
        
        # Test 2: Single gist group [1 2 G G G 3 4]
        input_ids2 = torch.tensor([1, 2, 999, 999, 999, 3, 4])
        result2 = get_gist_index(input_ids2, gist_token=[999])
        print(f"Test 2: {result2}")
        assert result2 == [(2, 5)], f"Expected [(2, 5)], got {result2}"
        
        # Test 3: Non-contiguous single gists [1 G 2 G 3 G]
        input_ids3 = torch.tensor([1, 999, 2, 999, 3, 999])
        result3 = get_gist_index(input_ids3, gist_token=[999])
        print(f"Test 3: {result3}")
        assert result3 == [(1, 2), (3, 4), (5, 6)], f"Expected [(1, 2), (3, 4), (5, 6)], got {result3}"
        
        # Test 4: No gist tokens [1 2 3 4 5]
        input_ids4 = torch.tensor([1, 2, 3, 4, 5])
        result4 = get_gist_index(input_ids4, gist_token=[999])
        print(f"Test 4: {result4}")
        assert result4 == (None, None), f"Expected (None, None), got {result4}"
        
        # Test 5: All gist tokens [G G G G]
        input_ids5 = torch.tensor([999, 999, 999, 999])
        result5 = get_gist_index(input_ids5, gist_token=[999])
        print(f"Test 5: {result5}")
        assert result5 == [(0, 4)], f"Expected [(0, 4)], got {result5}"
        
        # Test 6: Gist at end [1 2 3 4 G]
        input_ids6 = torch.tensor([1, 2, 3, 4, 999])
        result6 = get_gist_index(input_ids6, gist_token=[999])
        print(f"Test 6: {result6}")
        assert result6 == [(4, 5)], f"Expected [(4, 5)], got {result6}"

    def test_hierarchical_mask():
        """Test hierarchical gist mask generation."""
        # Create a sequence with hierarchical gist tokens
        # Pattern: [1 2 3 g 4 5 6 g 7 8 9 g G]
        # where g=999 (level 0), G=9999 (level 1)
        tokens = [0, 1, 2, 3, 999, 4, 5, 6, 999, 9999, 7, 8, 999, 9999, 9, 10, 11]
        input_tensor = torch.tensor([tokens])
        gist_tokens = [999, 9999]  # [g, G]
        chunk_size = 3

        print("="*80)
        print("Testing hierarchical gist mask")
        print("="*80)
        print(f"\nInput sequence: {tokens}")
        print("Token legend: g=999 (level 0), G=9999 (level 1)")
        print(f"Chunk size: {chunk_size}\n")

        # Generate hierarchical mask
        mask = make_hierarchical_gist_mask(
            input_tensor,
            gist_tokens,
            chunk_size,
            attention_sink_size=1,
            num_previous_chunks=0,
            pad_token=-1
        )

        print("Hierarchical Mask (each token type shown separately):")
        print("-" * 80)

        # Print mask with detailed legend
        print_mask(mask, input_tensor, gist_tokens)

        print("\n" + "="*80)
        print("Expected behavior:")
        print("="*80)
        print("Regular tokens (1-9):")
        print("  - Can attend to tokens in their own chunk")
        print("  - Can attend to all previous level 0 gist tokens (g)")
        print("\nLevel 0 gist tokens (g):")
        print("  - Already handled by the regular token mask")
        print("\nLevel 1 gist tokens (G):")
        print("  - Can attend to level 0 gist tokens (g) in the current chunk")
        print("  - Can attend to all previous level 1 gist tokens (G)")
        print("  - In this example: G can attend to the 3rd 'g' (same chunk)")
        print("="*80)

    def test_hierarchical_mask_3_levels():
        """Test with 3 levels of gist tokens."""
        # Manually create a sequence with 3 levels
        # Level 0 (g=1000): after every 2 regular tokens
        # Level 1 (G=2000): after every 2 level 0 gist tokens
        # Level 2 (GG=3000): after every 2 level 1 gist tokens
        tokens = [
            0,
            1000,
            2000,
            3000,
            1, 2, 1000,           # Chunk 1
            3, 4, 1000,           # Chunk 2
            2000,                 # L1 gist after 2 L0 gists
            5, 6, 1000,           # Chunk 3
            7, 8, 1000,           # Chunk 4
            2000,                 # L1 gist
            3000,                 # L2 gist after 2 L1 gists
            9, 1000,              # Chunk 5
            2000,                 # L1 gist
            3000,                 # L2 gist
            10, 11
        ]

        input_tensor = torch.tensor([tokens])
        gist_tokens = [1000, 2000, 3000]  # [g, G, GG]
        chunk_size = 2

        print("\n" + "="*80)
        print("Testing 3-level hierarchical gist mask")
        print("="*80)
        print(f"\nInput sequence: {tokens}")
        print("Token legend: g=1000 (L0), G=2000 (L1), GG=3000 (L2)")
        print(f"Chunk size: {chunk_size}\n")

        mask = make_hierarchical_gist_mask(
            input_tensor,
            gist_tokens,
            chunk_size,
            attention_sink_size=1,
            num_previous_chunks=0,
            pad_token=-1
        )

        print("3-Level Hierarchical Mask:")
        print("-" * 80)

        # Custom print for 3 levels
        seq_len = mask.shape[-1]
        input_list = tokens

        col_width = 4

        # Print header
        print("    ", end="")
        for token in input_list:
            if token == 1000:
                label = "g"
            elif token == 2000:
                label = "G"
            elif token == 3000:
                label = "GG"
            else:
                label = str(token)
            print(f"{label:>{col_width}}", end="")
        print()

        # Print each row
        for i, token in enumerate(input_list):
            if token == 1000:
                label = "g"
            elif token == 2000:
                label = "G"
            elif token == 3000:
                label = "GG"
            else:
                label = str(token)
            print(f"{label:>3} ", end="")

            for j in range(seq_len):
                val = mask[0, 0, i, j].item()
                print(f"{val:>{col_width}}", end="")
            print()

        print("\n" + "="*80)
        print("Expected behavior for 3 levels:")
        print("="*80)
        print("L0 gist (g): attend to raw tokens in chunk + prev g tokens")
        print("L1 gist (G): attend to g tokens in current chunk + prev G tokens")
        print("L2 gist (GG): attend to G tokens in current chunk + prev GG tokens")
        print("="*80)

    print("="*60)
    test_semantic_gist_insertion()
    print("\n" + "="*60 + "\n")
    test_hierarchical_gist_insertion()
    print("\n" + "="*60 + "\n")
    test_gist_mask()
    print("\n" + "="*60 + "\n")
    test_hierarchical_mask()
    print("\n" + "="*60 + "\n")
    test_hierarchical_mask_3_levels()
    # test_get_gist_index()