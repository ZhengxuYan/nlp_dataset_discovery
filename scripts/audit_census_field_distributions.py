#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


STRING_FIELDS = [
    "year",
    "role",
    "resource_type",
    "primary_use",
    "is_reusable_resource",
    "annotator_type",
    "uses_llm_synthetic_generation",
    "synthetic_human_verification",
    "release_status",
    "access_restrictions",
    "documentation_type",
    "maintenance_status",
    "confidence",
    "unit_of_analysis",
]

LIST_FIELDS = [
    "tasks",
    "domains",
    "languages",
    "modalities",
    "genres",
    "transformation_types",
    "synthetic_model_names",
]

SCALE_FIELDS = [
    "num_instances",
    "num_tokens",
    "num_documents",
    "num_dialogues",
    "num_images",
    "num_audio_hours",
    "num_languages",
    "num_domains",
]

SECTOR_ORDER = ["academic_only", "academic_industry_collab", "industry_only"]
SECTOR_LABELS = {
    "academic_only": "Academic only",
    "academic_industry_collab": "Academic + industry",
    "industry_only": "Industry only",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_paper_sector(path: Path) -> dict[str, str]:
    sectors: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            profile = row.get("institution_profile") or {}
            sectors[row.get("paper_id") or ""] = profile.get("paper_sector") or "unknown"
    return sectors


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "0.0%"


def score_value(row: dict[str, Any]) -> float | None:
    value = (row.get("profile") or {}).get("added_information_score")
    return float(value) if value is not None else None


def has_adequate_prior_set(row: dict[str, Any]) -> bool:
    return ((row.get("prior_set_assessment") or {}).get("prior_set_adequacy") in {"high", "medium"})


def bucket_instances(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "unknown"
    if value < 1_000:
        return "<1K"
    if value < 10_000:
        return "1K-10K"
    if value < 100_000:
        return "10K-100K"
    return ">=100K"


def bucket_languages(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "unknown"
    if value == 1:
        return "1 language"
    if value <= 10:
        return "2-10 languages"
    return ">10 languages"


def bucket_domains(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "unknown"
    if value == 1:
        return "1 domain"
    if value <= 5:
        return "2-5 domains"
    return ">5 domains"


def summarize_scores(
    added_rows: list[dict[str, Any]],
    bank_by_id: dict[str, dict[str, Any]],
    getter: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in added_rows:
        if not has_adequate_prior_set(row):
            continue
        bank_row = bank_by_id.get(row.get("query_bank_id") or "")
        value = score_value(row)
        if not bank_row or value is None:
            continue
        groups[getter(bank_row)].append(value)
    out = []
    for group, values in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        if not values:
            continue
        out.append(
            {
                "group": group,
                "n": len(values),
                "mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "score_1_pct": round(100 * sum(value == 1.0 for value in values) / len(values), 2),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-bank-jsonl", default="data/census/fulltext_dataset_bank.jsonl")
    parser.add_argument("--fulltext-extractions-jsonl", default="data/census/fulltext_dataset_extractions_pdf_all.jsonl")
    parser.add_argument("--added-info-2025", default="data/census/fulltext_added_information_attribution_2025_all_adequacy_v3_top20prior80acu.jsonl")
    parser.add_argument("--output-markdown", default="artifacts/paper_results/census_field_distribution_audit.md")
    parser.add_argument("--output-csv-dir", default="artifacts/paper_results/census_field_distribution_audit_tables")
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    bank_rows = read_jsonl(Path(args.dataset_bank_jsonl))
    added_rows = read_jsonl(Path(args.added_info_2025))
    bank_by_id = {row.get("bank_id"): row for row in bank_rows if row.get("bank_id")}
    paper_sector = read_paper_sector(Path(args.fulltext_extractions_jsonl))
    csv_dir = Path(args.output_csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    md: list[str] = [
        "# Census Field Distribution Audit",
        "",
        f"- Dataset rows: {len(bank_rows):,}",
        f"- 2025 added-information rows: {len(added_rows):,}",
        f"- 2025 rows after prior-coverage filter: {sum(has_adequate_prior_set(row) for row in added_rows):,}",
        "",
    ]

    for field in STRING_FIELDS:
        counts = Counter(str(row.get(field)) for row in bank_rows)
        rows = [
            {"field": field, "value": value, "count": count, "pct": round(100 * count / len(bank_rows), 3)}
            for value, count in counts.most_common(args.top_k)
        ]
        write_csv(csv_dir / f"{field}.csv", rows, ["field", "value", "count", "pct"])
        md.extend([f"## {field}", "", "| Value | Count | % |", "| --- | ---: | ---: |"])
        md.extend(f"| {row['value']} | {row['count']:,} | {row['pct']:.1f} |" for row in rows[:12])
        md.append("")

    for field in LIST_FIELDS:
        counts: Counter = Counter()
        nonempty = 0
        for row in bank_rows:
            values = row.get(field) or []
            if values:
                nonempty += 1
            counts.update(str(value) for value in values)
        rows = [
            {"field": field, "value": value, "count": count, "pct_of_rows": round(100 * count / len(bank_rows), 3)}
            for value, count in counts.most_common(args.top_k)
        ]
        write_csv(csv_dir / f"{field}.csv", rows, ["field", "value", "count", "pct_of_rows"])
        md.extend([f"## {field}", "", f"- Non-empty rows: {nonempty:,} ({pct(nonempty, len(bank_rows))})", "", "| Value | Count | Count / rows |", "| --- | ---: | ---: |"])
        md.extend(f"| {row['value']} | {row['count']:,} | {row['pct_of_rows']:.1f} |" for row in rows[:12])
        md.append("")

    scale_rows = []
    for field in SCALE_FIELDS:
        values = []
        for row in bank_rows:
            value = (row.get("scale") or {}).get(field)
            if isinstance(value, (int, float)) and value > 0:
                values.append(float(value))
        if values:
            scale_rows.append(
                {
                    "field": field,
                    "nonempty": len(values),
                    "pct": round(100 * len(values) / len(bank_rows), 3),
                    "median": round(statistics.median(values), 3),
                    "mean": round(statistics.mean(values), 3),
                }
            )
    write_csv(csv_dir / "scale_numeric_coverage.csv", scale_rows, ["field", "nonempty", "pct", "median", "mean"])
    md.extend(["## scale numeric coverage", "", "| Field | Non-empty | % | Median | Mean |", "| --- | ---: | ---: | ---: | ---: |"])
    md.extend(f"| {row['field']} | {row['nonempty']:,} | {row['pct']:.1f} | {row['median']:,} | {row['mean']:,} |" for row in scale_rows)
    md.append("")

    sector_rows = []
    for sector in SECTOR_ORDER:
        group = [row for row in bank_rows if paper_sector.get(row.get("paper_id") or "") == sector]
        if not group:
            continue
        values = [((row.get("scale") or {}).get("num_instances") or 0) for row in group if ((row.get("scale") or {}).get("num_instances") or 0) > 0]
        sector_rows.append(
            {
                "sector": SECTOR_LABELS[sector],
                "dataset_rows": len(group),
                "llm_synthetic_pct": round(100 * sum(row.get("uses_llm_synthetic_generation") is True for row in group) / len(group), 2),
                "benchmarking_pct": round(100 * sum(row.get("primary_use") == "benchmarking" for row in group) / len(group), 2),
                "open_access_pct": round(100 * sum(row.get("access_restrictions") == "open" for row in group) / len(group), 2),
                "multi_language_pct": round(100 * sum(((row.get("scale") or {}).get("num_languages") or 0) > 1 for row in group) / len(group), 2),
                "multi_domain_pct": round(100 * sum(((row.get("scale") or {}).get("num_domains") or 0) > 1 for row in group) / len(group), 2),
                "median_instances": round(statistics.median(values), 2) if values else "",
            }
        )
    write_csv(csv_dir / "sector_summary.csv", sector_rows, list(sector_rows[0].keys()) if sector_rows else ["sector"])
    md.extend(["## sector summary", "", "| Sector | Rows | LLM synthetic % | Benchmarking % | Open % | Multi-lang % | Multi-domain % | Median instances |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    md.extend(
        f"| {row['sector']} | {row['dataset_rows']:,} | {row['llm_synthetic_pct']:.1f} | {row['benchmarking_pct']:.1f} | {row['open_access_pct']:.1f} | {row['multi_language_pct']:.1f} | {row['multi_domain_pct']:.1f} | {row['median_instances']} |"
        for row in sector_rows
    )
    md.append("")

    score_specs = {
        "score_by_primary_use": lambda row: str(row.get("primary_use")),
        "score_by_sector": lambda row: SECTOR_LABELS.get(paper_sector.get(row.get("paper_id") or ""), "Other/unknown"),
        "score_by_instance_bucket": lambda row: bucket_instances((row.get("scale") or {}).get("num_instances")),
        "score_by_language_bucket": lambda row: bucket_languages((row.get("scale") or {}).get("num_languages")),
        "score_by_domain_bucket": lambda row: bucket_domains((row.get("scale") or {}).get("num_domains")),
        "score_by_source_presence": lambda row: "has named source datasets" if row.get("source_datasets") else "no named source datasets",
        "score_by_llm_synthetic": lambda row: "LLM synthetic" if row.get("uses_llm_synthetic_generation") is True else "not LLM synthetic",
    }
    for name, getter in score_specs.items():
        rows = summarize_scores(added_rows, bank_by_id, getter)
        write_csv(csv_dir / f"{name}.csv", rows, ["group", "n", "mean", "median", "score_1_pct"])
        md.extend([f"## {name}", "", "| Group | N | Mean | Median | Score=1 % |", "| --- | ---: | ---: | ---: | ---: |"])
        md.extend(f"| {row['group']} | {row['n']:,} | {row['mean']:.3f} | {row['median']:.3f} | {row['score_1_pct']:.1f} |" for row in rows[:12])
        md.append("")

    Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_markdown).write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"output_markdown": args.output_markdown, "output_csv_dir": args.output_csv_dir}, indent=2))


if __name__ == "__main__":
    main()
