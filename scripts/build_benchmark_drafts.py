#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from scv.prior_work_builder import build_benchmark_drafts


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build or refresh benchmark drafts from query papers and prior-paper artifacts.")
    parser.add_argument("--queries", type=str, default="data/processed/final_scv_200.jsonl")
    parser.add_argument("--paper-ids", nargs="*", default=None)
    args = parser.parse_args()

    rows = build_benchmark_drafts(query_jsonl_path=args.queries, paper_ids=args.paper_ids)
    print(f"Stored {len(rows)} benchmark draft rows.")


if __name__ == "__main__":
    main()
