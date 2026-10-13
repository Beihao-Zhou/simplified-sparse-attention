"""Build the LongBench folders that benchmark/Longbench_exp/longbench_pred.py loads.

Downloads THUDM/LongBench data.zip and saves each task as a datasets.load_from_disk
folder. The default output is benchmark/Longbench_exp/LongBench (gitignored):

    python data_process/build_longbench.py                      # all tasks
    python data_process/build_longbench.py --tasks triviaqa hotpotqa
"""

import argparse
import os
import zipfile

from datasets import Dataset
from huggingface_hub import hf_hub_download

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "benchmark", "Longbench_exp", "LongBench")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tasks", nargs="+", default=None, help="default: every task in data.zip")
    args = ap.parse_args()

    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        members = {os.path.splitext(os.path.basename(n))[0]: n for n in zf.namelist()
                   if n.endswith(".jsonl")}
        tasks = args.tasks or sorted(members)
        missing = [t for t in tasks if t not in members]
        if missing:
            raise SystemExit(f"not in data.zip: {missing}")
        extract_dir = os.path.join(os.path.dirname(zip_path), "longbench_jsonl")
        for task in tasks:
            path = zf.extract(members[task], extract_dir)
            ds = Dataset.from_json(path)
            ds.save_to_disk(os.path.join(args.out, task))
            print(f"{task}: {len(ds)} rows")


if __name__ == "__main__":
    main()
