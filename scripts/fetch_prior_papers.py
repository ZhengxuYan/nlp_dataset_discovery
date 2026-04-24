#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from scv.prior_work_builder import fetch_resolved_prior_papers


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch resolved prior papers from arXiv.")
    parser.add_argument("--status", nargs="*", default=["resolved_arxiv", "resolved_metadata"])
    parser.add_argument("--paper-ids", nargs="*", default=None)
    args = parser.parse_args()

    rows = fetch_resolved_prior_papers(statuses=args.status, paper_ids=args.paper_ids)
    print(f"Updated {len(rows)} paper bank records.")


if __name__ == "__main__":
    main()
