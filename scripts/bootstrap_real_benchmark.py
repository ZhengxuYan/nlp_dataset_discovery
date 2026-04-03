#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_candidate(dataset: Dict, paper: Dict, idx: int) -> Dict:
    info = dataset.get("info", dataset)
    candidate_id = f"{paper.get('arxiv_id', 'paper')}:dataset:{idx}"
    return {
        "candidate_id": candidate_id,
        "name": info.get("name", f"dataset_{idx}"),
        "acus": info.get("acus", []),
        "domain": info.get("domain", ""),
        "role": info.get("role", ""),
        "source_dataset": info.get("source_dataset", ""),
        "paper_title": paper.get("metadata", {}).get("title", paper.get("title", "")),
        "paper_id": paper.get("arxiv_id", ""),
        "is_cited": False,
        "annotation_status": "needs_review",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a real added-information benchmark scaffold from processed dataset outputs.")
    parser.add_argument("--input", type=str, default="data/processed/final_scv_200.jsonl")
    parser.add_argument("--output", type=str, default="data/benchmark/real_added_information_benchmark_template.jsonl")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    papers: List[Dict] = []
    with open(args.input, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))

    benchmark_rows = []
    for paper in papers:
        introduced = [
            dataset for dataset in paper.get("datasets", [])
            if dataset.get("info", dataset).get("is_introduced")
        ]
        for dataset_idx, dataset in enumerate(introduced):
            info = dataset.get("info", dataset)
            benchmark_rows.append({
                "example_id": f"{paper.get('arxiv_id', 'paper')}-{dataset_idx}",
                "query_dataset": {
                    "name": info.get("name", ""),
                    "acus": info.get("acus", []),
                    "paper_id": paper.get("arxiv_id", ""),
                    "paper_title": paper.get("metadata", {}).get("title", ""),
                    "domain": info.get("domain", ""),
                    "role": info.get("role", ""),
                    "source_dataset": info.get("source_dataset", ""),
                },
                "gold_prior_support_ids": [],
                "gold_prior_support_acus": info.get("previous_work_acus", []),
                "gold_added_information_label": "",
                "gold_added_information_rationale": "",
                "annotation_notes": "Populate gold prior support ids, label, and rationale after manual review.",
                "candidates": [],
            })
            if len(benchmark_rows) >= args.limit:
                break
        if len(benchmark_rows) >= args.limit:
            break

    # Populate candidate pools from all introduced datasets in the scaffolded slice.
    candidate_pool = []
    for paper in papers[: max(args.limit * 2, args.limit)]:
        introduced = [
            dataset for dataset in paper.get("datasets", [])
            if dataset.get("info", dataset).get("is_introduced")
        ]
        for dataset_idx, dataset in enumerate(introduced):
            candidate_pool.append(build_candidate(dataset, paper, dataset_idx))

    for row in benchmark_rows:
        own_paper_id = row["query_dataset"]["paper_id"]
        row["candidates"] = [
            candidate for candidate in candidate_pool
            if candidate["paper_id"] != own_paper_id
        ][:25]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        for row in benchmark_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Wrote {len(benchmark_rows)} benchmark scaffolds to {args.output}")


if __name__ == "__main__":
    main()
