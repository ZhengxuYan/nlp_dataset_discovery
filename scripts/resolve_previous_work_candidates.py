#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from scv.prior_work_builder import resolve_previous_work_candidates


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Resolve previous-work candidates to existing records or arXiv papers.")
    parser.add_argument("--candidate-ids", nargs="*", default=None)
    args = parser.parse_args()

    rows = resolve_previous_work_candidates(candidate_ids=args.candidate_ids)
    print(f"Updated {len(rows)} previous-work candidate records.")


if __name__ == "__main__":
    main()
