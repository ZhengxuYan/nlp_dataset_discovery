#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_DATASET_NAMES = [
    "ACL-OCL/acl-anthology-corpus",
    "WINGNUS/ACL-OCL",
    "ACL-OCL/ACL-OCL-Corpus",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="List files in candidate HuggingFace ACL-OCL dataset repos.")
    parser.add_argument("--dataset-name", action="append", dest="dataset_names", default=None)
    parser.add_argument("--output-json", default="data/census/hf_acl_ocl_files.json")
    args = parser.parse_args()

    api = HfApi()
    results = {}
    for name in args.dataset_names or DEFAULT_DATASET_NAMES:
        try:
            files = api.list_repo_files(name, repo_type="dataset")
            interesting = [
                file
                for file in files
                if file.endswith((".parquet", ".jsonl", ".json", ".csv", ".arrow", ".zip", ".gz"))
            ]
            results[name] = {
                "ok": True,
                "n_files": len(files),
                "interesting_files": interesting,
            }
        except Exception as exc:  # noqa: BLE001
            results[name] = {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
