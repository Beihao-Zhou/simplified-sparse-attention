#!/usr/bin/env bash
# Run the LongBench benchmark with the OPTIMIZED SSA sparse-attention setting.
#
# Optimized = fused sparse flash-decode CUDA kernel (compact) + block-sparse
# prefill (flex), routed through the sparse path (ATTN_IMPL=sparse).
#   - decode: ~flat tpot vs k_len; 2.75x vs the SSA reference (eager) at k=36k
#   - prefill (TTFT): up to 2.9x end-to-end at k=36k, grows with context
#   - accuracy: with DECODE_SELECT=group (default) task F1 is preserved (delta
#     <= 0.3% vs reference), but group is a DIFFERENT selection algorithm —
#     F1-validated, NOT allclose. DECODE_SELECT=head is allclose-exact to eager.
#
# Knobs you can override on the command line, e.g.:
#   MAX_SAMPLES=5 CONTEXT_REPEAT=5 bash run_optimized.sh
#
#   MAX_SAMPLES   number of triviaqa samples            (default 1)
#   CONTEXT_REPEAT  repeat the document to inflate k_len (default 1; 5 ~= 36k, OOM ceiling)
#
# NOTE: the first run compiles the CUDA extension (~30s, cached afterwards) and
# triggers a one-time torch.compile for flex prefill. Subsequent runs are fast.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ATTN_IMPL=sparse
export PREFILL_KERNEL=flex
export MAX_SAMPLES="${MAX_SAMPLES:-1}"

cd "$(dirname "$0")"
bash run_longbench.sh
