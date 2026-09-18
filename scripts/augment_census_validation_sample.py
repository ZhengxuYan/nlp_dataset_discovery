#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add abstracts and ACL metadata to dataset-census validation samples.")
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--catalog-jsonl", default="data/census/acl_anthology/acl_anthology_2023_2025_all_with_abstracts.jsonl")
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    catalog = {row["paper_id"]: row for row in read_jsonl(args.catalog_jsonl)}
    output = []
    for row in read_jsonl(args.sample_jsonl):
        meta = catalog.get(row.get("paper_id"), {})
        output.append({
            "paper_id": row.get("paper_id"),
            "title": row.get("title") or meta.get("title"),
            "year": row.get("year") or meta.get("year"),
            "venue_prefix": row.get("venue_prefix") or meta.get("venue_prefix"),
            "booktitle": meta.get("booktitle") or "",
            "journal": meta.get("journal") or "",
            "url": meta.get("url") or "",
            "pdf_url": meta.get("pdf_url") or "",
            "abstract": meta.get("abstract") or "",
            "sample_bucket": row.get("sample_bucket"),
            "pred_is_dataset_mentioned": row.get("pred_is_dataset_mentioned"),
            "pred_is_dataset_introducing": row.get("pred_is_dataset_introducing"),
            "pred_datasets": row.get("pred_datasets") or [],
            "pred_exclusion_reason": row.get("pred_exclusion_reason") or "",
            "gold_is_dataset_introducing": row.get("gold_is_dataset_introducing"),
            "gold_dataset_names": row.get("gold_dataset_names"),
            "notes": row.get("notes"),
        })
    write_jsonl(args.output_jsonl, output)
    print(json.dumps({
        "rows": len(output),
        "with_abstract": sum(bool(row.get("abstract")) for row in output),
        "output_jsonl": args.output_jsonl,
    }, indent=2))


if __name__ == "__main__":
    main()
