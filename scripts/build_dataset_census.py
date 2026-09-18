#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_CATALOG = "data/processed/arxiv_nlp_conf_papers_2023_2025.csv"
DEFAULT_ANALYSIS = [
    "data/processed/arxiv_nlp_conf_papers_2023_2025_dataset_analysis(gpt-5-mini).jsonl",
]
DEFAULT_OUTPUT_DIR = "data/census"
DEFAULT_START_YEAR = 2023
DEFAULT_END_YEAR = 2025


def year_label(start_year: int, end_year: int) -> str:
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    return f"{start_year}_{end_year}"


def default_catalog(start_year: int, end_year: int) -> str:
    return f"data/processed/arxiv_nlp_conf_papers_{year_label(start_year, end_year)}.csv"


def default_analysis(start_year: int, end_year: int) -> list[str]:
    label = year_label(start_year, end_year)
    return [f"data/processed/arxiv_nlp_conf_papers_{label}_dataset_analysis(gpt-5-mini).jsonl"]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
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


def read_catalog(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def dedupe_catalog(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        paper_id = paper_id_from_catalog(row)
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        deduped.append(row)
    return deduped


def paper_id_from_catalog(row: dict[str, Any]) -> str:
    return str(row.get("arXiv ID") or row.get("arxiv_id") or row.get("id") or "").strip()


def paper_id_from_analysis(row: dict[str, Any]) -> str:
    return str(row.get("arxiv_id") or row.get("arXiv ID") or row.get("paper_id") or "").strip()


def parse_year(value: Any) -> int | None:
    if not value:
        return None
    text = str(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            return datetime.strptime(text[:10], fmt).year
        except ValueError:
            pass
    for token in text.replace("/", "-").split("-"):
        if token.isdigit() and len(token) == 4:
            year = int(token)
            if 1900 <= year <= 2100:
                return year
    return None


def catalog_year(row: dict[str, Any]) -> int | None:
    return parse_year(row.get("Publication Date") or row.get("published_date") or row.get("date"))


def analysis_year(row: dict[str, Any], catalog_by_id: dict[str, dict[str, Any]]) -> int | None:
    return (
        parse_year(row.get("published_date") or row.get("Publication Date") or row.get("date"))
        or catalog_year(catalog_by_id.get(paper_id_from_analysis(row), {}))
    )


def normalize_datasets(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        datasets = []
        for name, item in raw.items():
            if isinstance(item, dict):
                normalized = dict(item)
                normalized.setdefault("name", name)
                datasets.append(normalized)
        return datasets
    if isinstance(raw, list):
        datasets = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            info = item.get("info") if isinstance(item.get("info"), dict) else item
            normalized = dict(info)
            if "scv" in item and isinstance(item["scv"], dict):
                normalized["scv"] = item["scv"]
            datasets.append(normalized)
        return datasets
    return []


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def is_introduced_dataset(dataset: dict[str, Any]) -> bool:
    if as_bool(dataset.get("is_introduced")):
        return True
    role = str(dataset.get("role") or "").lower()
    return "main contribution" in role or "introduced" in role


def confidence_rank(value: Any) -> int:
    text = str(value or "").strip().lower()
    return {"high": 3, "medium": 2, "low": 1}.get(text, 0)


def row_score(row: dict[str, Any]) -> tuple[int, int, int]:
    datasets = normalize_datasets(row.get("datasets"))
    introduced = [dataset for dataset in datasets if is_introduced_dataset(dataset)]
    confidence = max([confidence_rank(dataset.get("confidence")) for dataset in datasets] or [0])
    return (len(introduced), len(datasets), confidence)


def load_analysis(paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            paper_id = paper_id_from_analysis(row)
            if not paper_id:
                continue
            row = dict(row)
            row["_analysis_source"] = path
            if paper_id not in by_id or row_score(row) > row_score(by_id[paper_id]):
                by_id[paper_id] = row
    return by_id


def normalize_record(row: dict[str, Any], catalog_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paper_id = paper_id_from_analysis(row)
    catalog = catalog_by_id.get(paper_id, {})
    datasets = normalize_datasets(row.get("datasets"))
    introduced = [dataset for dataset in datasets if is_introduced_dataset(dataset)]
    mentioned = bool(row.get("is_dataset_mentioned")) or bool(datasets)
    year = analysis_year(row, catalog_by_id)
    return {
        "paper_id": paper_id,
        "title": row.get("title") or catalog.get("Title") or "",
        "published_date": row.get("published_date") or catalog.get("Publication Date") or "",
        "year": year,
        "source_type": row.get("source_type") or "unknown",
        "publication_venue": row.get("publication_venue") or catalog.get("Journal Reference") or catalog.get("Comment") or "",
        "is_nlp_paper": as_bool(row.get("is_nlp_paper", True)),
        "is_dataset_mentioned": mentioned,
        "is_dataset_introducing": bool(introduced),
        "n_datasets_mentioned": len(datasets),
        "n_datasets_introduced": len(introduced),
        "introduced_datasets": [
            {
                "name": dataset.get("name") or "",
                "role": dataset.get("role") or "",
                "usage_description": dataset.get("usage_description") or "",
                "source_dataset": dataset.get("source_dataset") or "None",
                "transformation_type": dataset.get("transformation_type") or "None",
                "confidence": dataset.get("confidence") or "",
                "acus": dataset.get("acus") or [],
            }
            for dataset in introduced
        ],
        "analysis_source": row.get("_analysis_source"),
    }


def build_pending_rows(catalog_rows: Sequence[dict[str, Any]], analyzed_ids: set[str]) -> list[dict[str, Any]]:
    pending = []
    for row in catalog_rows:
        paper_id = paper_id_from_catalog(row)
        if not paper_id or paper_id in analyzed_ids:
            continue
        pending.append({
            "paper_id": paper_id,
            "title": row.get("Title") or "",
            "published_date": row.get("Publication Date") or "",
            "year": catalog_year(row),
            "abstract": row.get("Abstract") or "",
            "categories": row.get("Categories") or "",
            "primary_category": row.get("Primary Category") or "",
            "arxiv_url": row.get("arXiv URL") or "",
            "pdf_url": row.get("PDF URL") or "",
        })
    return pending


def summarize(
    raw_catalog_rows: Sequence[dict[str, Any]],
    catalog_rows: Sequence[dict[str, Any]],
    census_rows: Sequence[dict[str, Any]],
    pending_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_year_catalog = Counter(catalog_year(row) for row in catalog_rows)
    by_year_analyzed = Counter(row.get("year") for row in census_rows)
    by_year_introducing = Counter(row.get("year") for row in census_rows if row.get("is_dataset_introducing"))
    role_counts = Counter()
    transformation_counts = Counter()
    source_counts = Counter()
    confidence_counts = Counter()
    for row in census_rows:
        for dataset in row.get("introduced_datasets") or []:
            role_counts[dataset.get("role") or "Unknown"] += 1
            transformation_counts[dataset.get("transformation_type") or "None"] += 1
            source = dataset.get("source_dataset") or "None"
            source_counts["has_source_dataset" if source and source.lower() != "none" else "no_source_dataset"] += 1
            confidence_counts[dataset.get("confidence") or "Unknown"] += 1
    return {
        "raw_catalog_rows": len(raw_catalog_rows),
        "unique_catalog_papers": len(catalog_rows),
        "duplicate_catalog_rows": len(raw_catalog_rows) - len(catalog_rows),
        "analyzed_papers": len(census_rows),
        "pending_papers": len(pending_rows),
        "dataset_mentioning_papers": sum(1 for row in census_rows if row.get("is_dataset_mentioned")),
        "dataset_introducing_papers": sum(1 for row in census_rows if row.get("is_dataset_introducing")),
        "introduced_datasets": sum(int(row.get("n_datasets_introduced") or 0) for row in census_rows),
        "catalog_by_year": dict(sorted((k, v) for k, v in by_year_catalog.items() if k)),
        "analyzed_by_year": dict(sorted((k, v) for k, v in by_year_analyzed.items() if k)),
        "dataset_introducing_by_year": dict(sorted((k, v) for k, v in by_year_introducing.items() if k)),
        "introduced_dataset_roles": dict(role_counts.most_common()),
        "introduced_dataset_transformations": dict(transformation_counts.most_common()),
        "introduced_dataset_source_summary": dict(source_counts.most_common()),
        "introduced_dataset_confidence": dict(confidence_counts.most_common()),
    }


def validation_sample(census_rows: Sequence[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    positives = [row for row in census_rows if row.get("is_dataset_introducing")]
    negatives = [row for row in census_rows if not row.get("is_dataset_introducing") and not row.get("is_dataset_mentioned")]
    borderline = [
        row for row in census_rows
        if not row.get("is_dataset_introducing") and row.get("is_dataset_mentioned")
    ]
    rng = random.Random(seed)
    buckets = [
        ("predicted_positive", positives),
        ("predicted_negative", negatives),
        ("borderline_dataset_mentioned", borderline),
    ]
    per_bucket = max(1, sample_size // len(buckets))
    sampled: list[dict[str, Any]] = []
    for label, rows in buckets:
        rows = list(rows)
        rng.shuffle(rows)
        for row in rows[:per_bucket]:
            sampled.append(annotation_row(row, label))
    if len(sampled) < sample_size:
        seen = {row["paper_id"] for row in sampled}
        remaining = [row for row in census_rows if row.get("paper_id") not in seen]
        rng.shuffle(remaining)
        sampled.extend(annotation_row(row, "fill") for row in remaining[: sample_size - len(sampled)])
    return sampled[:sample_size]


def annotation_row(row: dict[str, Any], bucket: str) -> dict[str, Any]:
    return {
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "published_date": row.get("published_date"),
        "sample_bucket": bucket,
        "pred_is_dataset_mentioned": row.get("is_dataset_mentioned"),
        "pred_is_dataset_introducing": row.get("is_dataset_introducing"),
        "pred_introduced_datasets": row.get("introduced_datasets"),
        "gold_is_dataset_introducing": None,
        "gold_introduced_dataset_names": None,
        "classification_notes": None,
    }


def dataset_acu_bank(census_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in census_rows:
        for index, dataset in enumerate(row.get("introduced_datasets") or []):
            rows.append({
                "dataset_id": f"{row.get('paper_id')}::dataset::{index}",
                "paper_id": row.get("paper_id"),
                "paper_title": row.get("title"),
                "published_date": row.get("published_date"),
                "year": row.get("year"),
                "dataset_name": dataset.get("name"),
                "role": dataset.get("role"),
                "source_dataset": dataset.get("source_dataset"),
                "transformation_type": dataset.get("transformation_type"),
                "usage_description": dataset.get("usage_description"),
                "acus": dataset.get("acus") or [],
                "confidence": dataset.get("confidence"),
            })
    return rows


def markdown_report(summary: dict[str, Any], paths: dict[str, str]) -> str:
    lines = [
        "# Dataset Paper Census Status",
        "",
        f"- Raw catalog rows: {summary['raw_catalog_rows']}",
        f"- Unique catalog papers: {summary['unique_catalog_papers']}",
        f"- Duplicate catalog rows removed: {summary['duplicate_catalog_rows']}",
        f"- Analyzed papers: {summary['analyzed_papers']}",
        f"- Pending papers: {summary['pending_papers']}",
        f"- Dataset-mentioning papers in analyzed set: {summary['dataset_mentioning_papers']}",
        f"- Dataset-introducing papers in analyzed set: {summary['dataset_introducing_papers']}",
        f"- Introduced datasets in analyzed set: {summary['introduced_datasets']}",
        "",
        "## Catalog By Year",
        "",
        "| Year | Papers |",
        "| --- | ---: |",
    ]
    for year, count in summary.get("catalog_by_year", {}).items():
        lines.append(f"| {year} | {count} |")
    lines.extend([
        "",
        "## Dataset-Introducing Papers By Year",
        "",
        "| Year | Papers |",
        "| --- | ---: |",
    ])
    for year, count in summary.get("dataset_introducing_by_year", {}).items():
        lines.append(f"| {year} | {count} |")
    lines.extend([
        "",
        "## Output Files",
        "",
    ])
    for label, path in paths.items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize current NLP dataset-paper census progress.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--analysis-jsonl", nargs="+", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-sample-size", type=int, default=100)
    parser.add_argument("--validation-sample-seed", type=int, default=17)
    args = parser.parse_args()

    args.catalog = args.catalog or default_catalog(args.start_year, args.end_year)
    args.analysis_jsonl = args.analysis_jsonl or default_analysis(args.start_year, args.end_year)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_catalog_rows = read_catalog(args.catalog)
    catalog_rows = dedupe_catalog(raw_catalog_rows)
    catalog_by_id = {paper_id_from_catalog(row): row for row in catalog_rows if paper_id_from_catalog(row)}
    analysis_by_id = load_analysis(args.analysis_jsonl)

    census_rows = [
        normalize_record(row, catalog_by_id)
        for _, row in sorted(analysis_by_id.items())
    ]
    pending_rows = build_pending_rows(catalog_rows, set(analysis_by_id))
    summary = summarize(raw_catalog_rows, catalog_rows, census_rows, pending_rows)
    bank_rows = dataset_acu_bank(census_rows)
    sample_rows = validation_sample(census_rows, args.validation_sample_size, args.validation_sample_seed)

    paths = {
        "summary": str(output_dir / "census_summary.json"),
        "census_records": str(output_dir / "census_records.jsonl"),
        "pending_queue": str(output_dir / "pending_papers.jsonl"),
        "dataset_acu_bank_seed": str(output_dir / "dataset_acu_bank_seed.jsonl"),
        "human_validation_sample": str(output_dir / "dataset_census_human_validation_sample.jsonl"),
        "status_markdown": str(output_dir / "census_status.md"),
    }
    write_json(paths["summary"], summary)
    write_jsonl(paths["census_records"], census_rows)
    write_jsonl(paths["pending_queue"], pending_rows)
    write_jsonl(paths["dataset_acu_bank_seed"], bank_rows)
    write_jsonl(paths["human_validation_sample"], sample_rows)
    Path(paths["status_markdown"]).write_text(markdown_report(summary, paths), encoding="utf-8")
    print(json.dumps({"summary": summary, "outputs": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
