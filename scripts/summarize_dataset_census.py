#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    introduced_rows = [row for row in rows if row.get("is_dataset_introducing")]
    mentioned_rows = [row for row in rows if row.get("is_dataset_mentioned")]
    datasets = [dataset for row in rows for dataset in (row.get("datasets") or [])]
    return {
        "rows": len(rows),
        "dataset_mentioning_papers": len(mentioned_rows),
        "dataset_introducing_papers": len(introduced_rows),
        "introduced_datasets": len(datasets),
        "dataset_mentioning_rate": len(mentioned_rows) / len(rows) if rows else 0.0,
        "dataset_introducing_rate": len(introduced_rows) / len(rows) if rows else 0.0,
        "by_year": dict(sorted(Counter(row.get("year") for row in rows).items())),
        "dataset_introducing_by_year": dict(sorted(Counter(row.get("year") for row in introduced_rows).items())),
        "by_venue_prefix": dict(Counter(row.get("venue_prefix") for row in rows).most_common()),
        "dataset_introducing_by_venue_prefix": dict(Counter(row.get("venue_prefix") for row in introduced_rows).most_common()),
        "dataset_roles": dict(Counter(dataset.get("role") or "Unknown" for dataset in datasets).most_common()),
        "dataset_transformations": dict(Counter(dataset.get("transformation_type") or "Unknown" for dataset in datasets).most_common()),
        "dataset_confidence": dict(Counter(dataset.get("confidence") or "Unknown" for dataset in datasets).most_common()),
    }


def sample_validation(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    positives = [row for row in rows if row.get("is_dataset_introducing")]
    mentioned_nonintro = [
        row for row in rows
        if row.get("is_dataset_mentioned") and not row.get("is_dataset_introducing")
    ]
    negatives = [row for row in rows if not row.get("is_dataset_mentioned")]
    buckets = [
        ("predicted_positive", positives),
        ("dataset_mentioned_nonintro", mentioned_nonintro),
        ("predicted_negative", negatives),
    ]
    rng = random.Random(seed)
    sampled = []
    per_bucket = max(1, sample_size // len(buckets))
    for label, bucket in buckets:
        bucket = list(bucket)
        rng.shuffle(bucket)
        for row in bucket[:per_bucket]:
            sampled.append(annotation_row(row, label))
    if len(sampled) < sample_size:
        seen = {row["paper_id"] for row in sampled}
        remaining = [row for row in rows if row.get("paper_id") not in seen]
        rng.shuffle(remaining)
        sampled.extend(annotation_row(row, "fill") for row in remaining[: sample_size - len(sampled)])
    return sampled[:sample_size]


def annotation_row(row: dict[str, Any], bucket: str) -> dict[str, Any]:
    return {
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "year": row.get("year"),
        "venue_prefix": row.get("venue_prefix"),
        "sample_bucket": bucket,
        "pred_is_dataset_mentioned": row.get("is_dataset_mentioned"),
        "pred_is_dataset_introducing": row.get("is_dataset_introducing"),
        "pred_datasets": row.get("datasets") or [],
        "pred_exclusion_reason": row.get("exclusion_reason") or "",
        "gold_is_dataset_introducing": None,
        "gold_dataset_names": None,
        "notes": None,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Dataset Census Summary",
        "",
        f"- Rows: {summary['rows']}",
        f"- Dataset-mentioning papers: {summary['dataset_mentioning_papers']} ({summary['dataset_mentioning_rate']:.1%})",
        f"- Dataset-introducing papers: {summary['dataset_introducing_papers']} ({summary['dataset_introducing_rate']:.1%})",
        f"- Introduced datasets: {summary['introduced_datasets']}",
        "",
        "## Dataset-Introducing By Venue",
        "",
        "| Venue | Papers |",
        "| --- | ---: |",
    ]
    for venue, count in list(summary["dataset_introducing_by_venue_prefix"].items())[:40]:
        lines.append(f"| {venue} | {count} |")
    lines.extend(["", "## Dataset Transformations", "", "| Type | Count |", "| --- | ---: |"])
    for key, count in summary["dataset_transformations"].items():
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dataset census classifier output and prepare validation samples.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", default="data/census")
    parser.add_argument("--name", default=None)
    parser.add_argument("--validation-sample-size", type=int, default=100)
    parser.add_argument("--validation-sample-seed", type=int, default=23)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    name = args.name or Path(args.input_jsonl).stem
    output_dir = Path(args.output_dir)
    summary = summarize(rows)
    sample = sample_validation(rows, args.validation_sample_size, args.validation_sample_seed)
    paths = {
        "summary_json": output_dir / f"{name}_summary.json",
        "summary_md": output_dir / f"{name}_summary.md",
        "validation_sample": output_dir / f"{name}_validation_sample.jsonl",
    }
    write_json(paths["summary_json"], summary)
    Path(paths["summary_md"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["summary_md"]).write_text(markdown(summary), encoding="utf-8")
    write_jsonl(paths["validation_sample"], sample)
    print(json.dumps({"summary": summary, "outputs": {k: str(v) for k, v in paths.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
