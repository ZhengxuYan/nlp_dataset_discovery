#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CENSUS = "data/census/acl_gemini_flashlite_all.clean.jsonl"
DEFAULT_DATASET_NAMES = [
    "ACL-OCL/acl-anthology-corpus",
    "WINGNUS/ACL-OCL",
    "ACL-OCL/ACL-OCL-Corpus",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_acl_id(value: Any) -> str:
    text = str(value or "").strip()
    text = text.removeprefix("ACL:")
    text = text.removeprefix("https://aclanthology.org/")
    text = text.removeprefix("http://aclanthology.org/")
    text = text.removesuffix(".pdf")
    text = text.strip("/")
    match = re.search(r"(\d{4}\.[A-Za-z0-9_.-]+\.\d+)", text)
    if match:
        return match.group(1)
    return text


def load_positive_targets(path: str | Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("is_dataset_introducing") and row.get("datasets"):
            acl_id = normalize_acl_id(row.get("paper_id"))
            if acl_id:
                targets[acl_id] = row
    return targets


def candidate_acl_ids(row: dict[str, Any]) -> list[str]:
    keys = [
        "acl_id",
        "id",
        "paper_id",
        "anthology_id",
        "url",
        "pdf_url",
        "acl_url",
    ]
    ids = []
    for key in keys:
        if key in row:
            acl_id = normalize_acl_id(row.get(key))
            if acl_id:
                ids.append(acl_id)
    return ids


def pick_full_text(row: dict[str, Any]) -> str:
    for key in ["full_text", "text", "paper_text", "body_text", "grobid_text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def pick_abstract(row: dict[str, Any]) -> str:
    value = row.get("abstract")
    return value if isinstance(value, str) else ""


def iter_parquet_rows(paths: list[str]) -> Iterable[dict[str, Any]]:
    import pandas as pd

    for path in paths:
        frame = pd.read_parquet(path)
        for row in frame.to_dict(orient="records"):
            yield row


def iter_hf_rows(dataset_names: list[str], split: str, streaming: bool) -> tuple[str, Iterable[dict[str, Any]]]:
    from datasets import load_dataset

    last_error: Exception | None = None
    for name in dataset_names:
        try:
            dataset = load_dataset(name, split=split, streaming=streaming)
            return name, iter(dataset)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


def summarize_matches(
    rows: Iterable[dict[str, Any]],
    targets: dict[str, dict[str, Any]],
    *,
    source_name: str,
    max_rows: int | None,
    output_matches_jsonl: str | None,
    sample_limit: int,
) -> dict[str, Any]:
    target_ids = set(targets)
    matched: dict[str, dict[str, Any]] = {}
    seen_rows = 0
    column_counter: Counter[str] = Counter()
    text_lengths: list[int] = []
    samples: list[dict[str, Any]] = []

    if output_matches_jsonl:
        Path(output_matches_jsonl).unlink(missing_ok=True)

    for row in rows:
        seen_rows += 1
        column_counter.update(row.keys())
        matched_acl_id = None
        for acl_id in candidate_acl_ids(row):
            if acl_id in target_ids:
                matched_acl_id = acl_id
                break
        if matched_acl_id:
            full_text = pick_full_text(row)
            abstract = pick_abstract(row)
            target = targets[matched_acl_id]
            match_row = {
                "paper_id": f"ACL:{matched_acl_id}",
                "acl_id": matched_acl_id,
                "title": target.get("title") or row.get("title") or "",
                "year": target.get("year") or row.get("year"),
                "source": source_name,
                "full_text_char_count": len(full_text),
                "has_full_text": bool(full_text.strip()),
                "abstract": abstract or target.get("abstract") or "",
                "full_text": full_text,
            }
            matched[matched_acl_id] = match_row
            if full_text:
                text_lengths.append(len(full_text))
            if len(samples) < sample_limit:
                sample = dict(match_row)
                sample["full_text_preview"] = sample.pop("full_text")[:1500]
                samples.append(sample)
            if output_matches_jsonl:
                append_jsonl(output_matches_jsonl, [match_row])
        if max_rows is not None and seen_rows >= max_rows:
            break

    years_total = Counter(str(row.get("year") or "unknown") for row in targets.values())
    years_matched = Counter(str(targets[acl_id].get("year") or "unknown") for acl_id in matched)
    full_text_matches = [row for row in matched.values() if row["has_full_text"]]
    return {
        "source": source_name,
        "hf_or_parquet_rows_scanned": seen_rows,
        "target_positive_rows": len(targets),
        "matched_rows": len(matched),
        "matched_rows_with_full_text": len(full_text_matches),
        "coverage": len(matched) / len(targets) if targets else 0,
        "full_text_coverage": len(full_text_matches) / len(targets) if targets else 0,
        "target_year_distribution": dict(years_total),
        "matched_year_distribution": dict(years_matched),
        "top_columns_seen": column_counter.most_common(30),
        "full_text_char_count_mean": sum(text_lengths) / len(text_lengths) if text_lengths else None,
        "full_text_char_count_min": min(text_lengths) if text_lengths else None,
        "full_text_char_count_max": max(text_lengths) if text_lengths else None,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether ACL-OCL/HuggingFace full text covers our ACL dataset-introducing papers.")
    parser.add_argument("--census-jsonl", default=DEFAULT_CENSUS)
    parser.add_argument("--dataset-name", action="append", dest="dataset_names", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--parquet", action="append", default=[])
    parser.add_argument("--max-hf-rows", type=int, default=None)
    parser.add_argument("--output-json", default="data/census/acl_ocl_fulltext_coverage.json")
    parser.add_argument("--output-matches-jsonl", default=None)
    parser.add_argument("--sample-limit", type=int, default=5)
    args = parser.parse_args()

    targets = load_positive_targets(args.census_jsonl)
    if args.parquet:
        source_name = "local_parquet"
        rows = iter_parquet_rows(args.parquet)
    else:
        dataset_names = args.dataset_names or DEFAULT_DATASET_NAMES
        source_name, rows = iter_hf_rows(dataset_names, args.split, not args.no_streaming)

    summary = summarize_matches(
        rows,
        targets,
        source_name=source_name,
        max_rows=args.max_hf_rows,
        output_matches_jsonl=args.output_matches_jsonl,
        sample_limit=args.sample_limit,
    )

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
