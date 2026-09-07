#!/usr/bin/env bash
# compute-sanitizer over the changed CUDA code: the vectorised/grid-strided host
# gather, the separate-stream event ordering around it, and the persistent hot
# buffer.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYENV:-/tmp/pyenv457}:${REPO}"
cd "$REPO"
T=tests.test_sparse_decode_batch.SparseDecodeBatchTest
CASES="${T}.test_separate_stream_gather_matches_sync_across_batch_sizes \
       ${T}.test_offload_accepts_a_view_of_a_longer_pinned_buffer \
       ${T}.test_repeated_separate_stream_calls_survive_allocator_reuse \
       ${T}.test_hot_buffer_reuses_hits_and_evicts_only_stale_tokens"
python -m unittest $CASES > /dev/null 2>&1   # warm the JIT build outside the sanitizer
status=0
for tool in memcheck initcheck synccheck racecheck; do
  echo "===== compute-sanitizer --tool=$tool ====="
  log="$(mktemp)"
  # Capture compute-sanitizer's own exit code, not the filter's: piping straight
  # into grep would report the grep status and hide a real failure.
  if compute-sanitizer --tool="$tool" --error-exitcode 9 \
       python -m unittest $CASES > "$log" 2>&1; then
    echo "exit=0"
  else
    code=$?
    echo "exit=$code"
    status=$code
  fi
  grep -E "ERROR|error|OK|FAILED|Ran [0-9]+" "$log" | tail -8
  rm -f "$log"
  [ "$status" -eq 0 ] || { echo "SANITIZER FAILED under $tool"; exit "$status"; }
done
echo "ALL DONE"
