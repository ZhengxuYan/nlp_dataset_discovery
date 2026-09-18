#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


VALID_GOLD = {"covered", "partially_covered", "not_evidence"}
VALID_HARD = {"valid_hard_negative", "actually_evidence", "irrelevant_not_plausible"}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value: str) -> str:
    return str(value or "").strip().lower()


def summarize(path: Path, field: str, valid: set[str]) -> dict:
    rows = read_csv(path)
    values = [norm(row.get(field, "")) for row in rows]
    invalid = [
        {
            "annotation_id": row.get("annotation_id"),
            field: row.get(field),
        }
        for row in rows
        if norm(row.get(field, "")) and norm(row.get(field, "")) not in valid
    ]
    return {
        "path": str(path),
        "rows": len(rows),
        "blank": sum(1 for value in values if not value),
        "invalid": len(invalid),
        "counts": dict(Counter(value or "<blank>" for value in values)),
        "invalid_examples": invalid[:20],
    }


def disagreement_summary(path_a: Path, path_b: Path, field: str) -> dict:
    rows_a = {row.get("annotation_id"): row for row in read_csv(path_a)}
    rows_b = {row.get("annotation_id"): row for row in read_csv(path_b)}
    common = sorted(set(rows_a) & set(rows_b))
    disagreements = []
    for ann_id in common:
        a = norm(rows_a[ann_id].get(field, ""))
        b = norm(rows_b[ann_id].get(field, ""))
        if a and b and a != b:
            disagreements.append(
                {
                    "annotation_id": ann_id,
                    "a": a,
                    "b": b,
                    "benchmark_id": rows_a[ann_id].get("benchmark_id"),
                    "query_dcu_id": rows_a[ann_id].get("query_dcu_id"),
                }
            )
    return {
        "common_rows": len(common),
        "disagreements": len(disagreements),
        "agreement_on_filled_common": (
            1.0 - len(disagreements) / max(1, sum(1 for ann_id in common if norm(rows_a[ann_id].get(field, "")) and norm(rows_b[ann_id].get(field, ""))))
        ),
        "disagreement_examples": disagreements[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check completed retrieval benchmark relabeling CSVs.")
    parser.add_argument("--dir", default="data/human_validation/retrieval_benchmark_relabeling_full")
    parser.add_argument("--annotator-a", default="annotator_a")
    parser.add_argument("--annotator-b", default="annotator_b")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    base = Path(args.dir)
    gold_a = base / f"retrieval_gold_prior_full_{args.annotator_a}.csv"
    gold_b = base / f"retrieval_gold_prior_full_{args.annotator_b}.csv"
    hard_a = base / f"retrieval_hard_negatives_full_{args.annotator_a}.csv"
    hard_b = base / f"retrieval_hard_negatives_full_{args.annotator_b}.csv"

    report = {
        "gold": {
            args.annotator_a: summarize(gold_a, "human_gold_prior_decision", VALID_GOLD),
            args.annotator_b: summarize(gold_b, "human_gold_prior_decision", VALID_GOLD),
            "agreement": disagreement_summary(gold_a, gold_b, "human_gold_prior_decision"),
        },
        "hard_negative": {
            args.annotator_a: summarize(hard_a, "human_hard_negative_decision", VALID_HARD),
            args.annotator_b: summarize(hard_b, "human_hard_negative_decision", VALID_HARD),
            "agreement": disagreement_summary(hard_a, hard_b, "human_hard_negative_decision"),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
