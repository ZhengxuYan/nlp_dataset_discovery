#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from scv.prior_work_builder import extract_previous_work_candidates


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Extract structured previous-work candidates from processed query papers.")
    parser.add_argument("--input", type=str, default="data/processed/final_scv_200.jsonl")
    parser.add_argument("--paper-ids", nargs="*", default=None)
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--backend-params", type=str, default=None)
    args = parser.parse_args()

    backend_params = json.loads(args.backend_params) if args.backend_params else None
    rows = extract_previous_work_candidates(
        query_jsonl_path=args.input,
        paper_ids=args.paper_ids,
        model_name=args.model,
        backend=args.backend,
        backend_params=backend_params,
    )
    print(f"Stored {len(rows)} previous-work candidate records.")


if __name__ == "__main__":
    main()
