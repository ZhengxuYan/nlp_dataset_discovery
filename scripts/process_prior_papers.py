#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from scv.prior_work_builder import process_prior_papers


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Process fetched prior papers into benchmark-compatible structured outputs.")
    parser.add_argument("--paper-ids", nargs="*", default=None)
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--backend-params", type=str, default=None)
    args = parser.parse_args()

    backend_params = json.loads(args.backend_params) if args.backend_params else None
    rows = process_prior_papers(
        model_name=args.model,
        backend=args.backend,
        backend_params=backend_params,
        paper_ids=args.paper_ids,
    )
    print(f"Stored {len(rows)} processed-bank records.")


if __name__ == "__main__":
    main()
