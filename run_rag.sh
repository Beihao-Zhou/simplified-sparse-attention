#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"


select=1
chunk=(4 4)
name="nolink-select${select}"
ckpt_path="gist-sparse-attention/GSA-FT-Llama-3.2-1B-chunk${chunk[0]}-chunk${chunk[1]}"

extra_arg=""
if [ "$select" -eq 1 ]; then
  extra_arg="--selective"
fi

echo "Running evaluation with the following settings:"
echo "Checkpoint Path: $ckpt_path"
echo "Name: $name"
echo "Chunk Sizes: ${chunk[@]}"
echo "Selective: $select"

# --dataset {nq,tqa,hqa,wiki,musique} --mode {base,gist,link}; --pos is nq-only.
for pos in 0; do
    python benchmark/RAG_exp/rag_eval.py \
        --dataset nq \
        --mode gist \
        --pos $pos \
        --ckpt_path $ckpt_path \
        --name $name \
        $extra_arg \
        --chunk_size ${chunk[@]} \
        --meta_gist 1
done
