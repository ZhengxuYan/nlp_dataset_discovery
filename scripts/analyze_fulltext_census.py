#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


YEAR_KEYS = ("2023", "2024", "2025")
GOVERNANCE_KEYS = (
    "ethics_discussed",
    "pii_discussed",
    "consent_discussed",
    "copyright_discussed",
    "bias_or_fairness_discussed",
)
ARTIFACT_KEYS = (
    "dataset_urls",
    "project_page_urls",
    "code_urls",
    "huggingface_ids",
    "github_repos",
    "zenodo_urls",
    "osf_urls",
    "kaggle_urls",
    "paperswithcode_urls",
    "other_urls",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def year_of(row: dict[str, Any]) -> str:
    year = row.get("year")
    return str(year)[:4] if year else "unknown"


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def counter_dict(counter: Counter, *, top_k: int | None = None) -> dict[str, int]:
    items = counter.most_common(top_k) if top_k else counter.most_common()
    return {str(key): int(value) for key, value in items}


def nested_counter_dict(counter: Counter) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for key, value in sorted(counter.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        outer, inner = key
        result[str(outer)][str(inner)] = int(value)
    return dict(result)


def artifact_counts(artifacts: Any) -> dict[str, int]:
    if not isinstance(artifacts, dict):
        return {key: 0 for key in ARTIFACT_KEYS}
    return {
        key: len(artifacts.get(key) or [])
        for key in ARTIFACT_KEYS
    }


def artifact_any(artifacts: Any, *keys: str) -> bool:
    if not isinstance(artifacts, dict):
        return False
    return any(bool(artifacts.get(key)) for key in keys)


def normalize_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def iter_datasets(rows: list[dict[str, Any]]):
    for row in rows:
        for dataset in row.get("datasets") or []:
            yield row, dataset


def summarize(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    papers_by_year: Counter = Counter()
    datasets_by_year: Counter = Counter()
    paper_sector: Counter = Counter()
    paper_sector_by_year: Counter = Counter()
    roles: Counter = Counter()
    roles_by_year: Counter = Counter()
    resource_types: Counter = Counter()
    release_status: Counter = Counter()
    release_status_by_year: Counter = Counter()
    documentation_type: Counter = Counter()
    maintenance_status: Counter = Counter()
    artifact_by_year: Counter = Counter()
    synthetic_by_year: Counter = Counter()
    annotator_type: Counter = Counter()
    transformation_types: Counter = Counter()
    acu_types: Counter = Counter()
    prior_relationships: Counter = Counter()
    prior_names: Counter = Counter()
    source_names: Counter = Counter()
    tasks: Counter = Counter()
    domains: Counter = Counter()
    languages: Counter = Counter()
    modalities: Counter = Counter()
    governance: dict[str, Counter] = {key: Counter() for key in GOVERNANCE_KEYS}
    quality_confidence: Counter = Counter()
    dataset_confidence: Counter = Counter()
    paper_warning_rows = 0
    paper_validation_error_rows = 0
    total_acus = 0
    total_prior_mentions = 0

    for row in rows:
        year = year_of(row)
        papers_by_year[year] += 1
        profile = row.get("institution_profile") or {}
        sector = profile.get("paper_sector") or "unknown"
        paper_sector[sector] += 1
        paper_sector_by_year[(year, sector)] += 1
        quality = row.get("extraction_quality") or {}
        quality_confidence[quality.get("confidence") or "unknown"] += 1
        if row.get("quality_warnings"):
            paper_warning_rows += 1
        if row.get("validation_errors"):
            paper_validation_error_rows += 1

    for row, dataset in iter_datasets(rows):
        year = year_of(row)
        datasets_by_year[year] += 1
        role = dataset.get("role") or "unknown"
        roles[role] += 1
        roles_by_year[(year, role)] += 1
        resource_types[dataset.get("resource_type") or "unknown"] += 1
        dataset_confidence[dataset.get("confidence") or "unknown"] += 1

        coverage = dataset.get("coverage") or {}
        for value in coverage.get("tasks") or []:
            tasks[normalize_text(value)] += 1
        for value in coverage.get("domains") or []:
            domains[normalize_text(value)] += 1
        for value in coverage.get("languages") or []:
            languages[normalize_text(value)] += 1
        for value in coverage.get("modality") or []:
            modalities[normalize_text(value)] += 1

        construction = dataset.get("construction") or {}
        for value in construction.get("transformation_types") or []:
            transformation_types[normalize_text(value)] += 1
        annotator_type[construction.get("annotator_type") or "unknown"] += 1
        synthetic = construction.get("synthetic_generation") or {}
        synthetic_by_year[(year, str(synthetic.get("uses_llm")))] += 1
        for source in construction.get("source_datasets") or []:
            if isinstance(source, dict):
                name = source.get("name")
            else:
                name = source
            if normalize_text(name):
                source_names[normalize_text(name)] += 1

        availability = dataset.get("availability") or {}
        release = availability.get("release_status") or "unknown"
        release_status[release] += 1
        release_status_by_year[(year, release)] += 1
        documentation_type[availability.get("documentation_type") or "unknown"] += 1
        maintenance_status[availability.get("maintenance_status") or "unknown"] += 1
        artifacts = availability.get("artifacts") or {}
        if artifact_any(artifacts, "dataset_urls"):
            artifact_by_year[(year, "dataset_url")] += 1
        if artifact_any(artifacts, "project_page_urls"):
            artifact_by_year[(year, "project_page")] += 1
        if artifact_any(artifacts, "code_urls", "github_repos"):
            artifact_by_year[(year, "github_or_code")] += 1
        if artifact_any(artifacts, "github_repos"):
            artifact_by_year[(year, "github_repo")] += 1
        if artifact_any(artifacts, "huggingface_ids"):
            artifact_by_year[(year, "huggingface")] += 1

        governance_payload = dataset.get("governance") or {}
        for key in GOVERNANCE_KEYS:
            governance[key][governance_payload.get(key) or "unknown"] += 1

        acus = dataset.get("acus") or []
        total_acus += len(acus)
        for acu in acus:
            if isinstance(acu, dict):
                acu_types[acu.get("type") or "unknown"] += 1

        priors = dataset.get("prior_dataset_mentions") or []
        total_prior_mentions += len(priors)
        for prior in priors:
            if not isinstance(prior, dict):
                continue
            prior_relationships[prior.get("relationship_type") or "unknown"] += 1
            name = prior.get("name") or prior.get("dataset_name") or prior.get("canonical_name")
            if normalize_text(name):
                prior_names[normalize_text(name)] += 1

    dataset_count = sum(datasets_by_year.values())
    released = release_status.get("released", 0)
    llm_true = sum(value for (year, label), value in synthetic_by_year.items() if label == "True")
    summary = {
        "input_rows": len(rows),
        "unique_papers": len({row.get("paper_id") for row in rows}),
        "datasets": dataset_count,
        "mean_datasets_per_paper": pct(dataset_count, len(rows)),
        "total_acus": total_acus,
        "mean_acus_per_dataset": pct(total_acus, dataset_count),
        "total_prior_dataset_mentions": total_prior_mentions,
        "quality_warning_rows": paper_warning_rows,
        "validation_error_rows": paper_validation_error_rows,
        "papers_by_year": counter_dict(papers_by_year),
        "datasets_by_year": counter_dict(datasets_by_year),
        "paper_sector": counter_dict(paper_sector),
        "paper_sector_by_year": nested_counter_dict(paper_sector_by_year),
        "roles": counter_dict(roles),
        "roles_by_year": nested_counter_dict(roles_by_year),
        "resource_types": counter_dict(resource_types),
        "release_status": counter_dict(release_status),
        "release_status_by_year": nested_counter_dict(release_status_by_year),
        "documentation_type": counter_dict(documentation_type),
        "maintenance_status": counter_dict(maintenance_status),
        "artifact_by_year": nested_counter_dict(artifact_by_year),
        "synthetic_generation_by_year": nested_counter_dict(synthetic_by_year),
        "annotator_type": counter_dict(annotator_type),
        "transformation_types": counter_dict(transformation_types),
        "governance": {key: counter_dict(value) for key, value in governance.items()},
        "prior_relationships": counter_dict(prior_relationships),
        "acu_types": counter_dict(acu_types),
        "quality_confidence": counter_dict(quality_confidence),
        "dataset_confidence": counter_dict(dataset_confidence),
        "top_tasks": counter_dict(tasks, top_k=top_k),
        "top_domains": counter_dict(domains, top_k=top_k),
        "top_languages": counter_dict(languages, top_k=top_k),
        "top_modalities": counter_dict(modalities, top_k=top_k),
        "top_prior_dataset_names": counter_dict(prior_names, top_k=top_k),
        "top_source_dataset_names": counter_dict(source_names, top_k=top_k),
        "derived_rates": {
            "released_dataset_fraction": pct(released, dataset_count),
            "llm_synthetic_dataset_fraction": pct(llm_true, dataset_count),
            "papers_with_quality_warnings_fraction": pct(paper_warning_rows, len(rows)),
        },
    }
    return summary


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = ["# Full-Text Dataset Census Summary", ""]
    lines.extend([
        f"- Papers: {summary['unique_papers']}",
        f"- Datasets/resources: {summary['datasets']}",
        f"- ACUs: {summary['total_acus']}",
        f"- Mean datasets per paper: {summary['mean_datasets_per_paper']:.3f}",
        f"- Mean ACUs per dataset: {summary['mean_acus_per_dataset']:.3f}",
        f"- Prior dataset mentions: {summary['total_prior_dataset_mentions']}",
        "",
    ])

    year_rows = []
    for year in sorted(summary["papers_by_year"]):
        year_rows.append([
            year,
            summary["papers_by_year"].get(year, 0),
            summary["datasets_by_year"].get(year, 0),
            (summary["synthetic_generation_by_year"].get(year) or {}).get("True", 0),
            (summary["artifact_by_year"].get(year) or {}).get("huggingface", 0),
            (summary["artifact_by_year"].get(year) or {}).get("github_or_code", 0),
        ])
    lines.extend(["## By Year", "", markdown_table(
        ["Year", "Papers", "Datasets", "LLM synthetic", "HuggingFace", "GitHub/code"],
        year_rows,
    ), ""])

    lines.extend(["## Paper Sector", "", markdown_table(
        ["Sector", "Papers"],
        [[key, value] for key, value in summary["paper_sector"].items()],
    ), ""])

    lines.extend(["## Dataset Roles", "", markdown_table(
        ["Role", "Datasets"],
        [[key, value] for key, value in list(summary["roles"].items())[:12]],
    ), ""])

    lines.extend(["## Release Status", "", markdown_table(
        ["Status", "Datasets"],
        [[key, value] for key, value in summary["release_status"].items()],
    ), ""])

    governance_rows = []
    for key, counts in summary["governance"].items():
        governance_rows.append([key, counts.get("yes", 0), counts.get("no", 0), counts.get("unclear", 0), counts.get("unknown", 0)])
    lines.extend(["## Governance Mentions", "", markdown_table(
        ["Field", "Yes", "No", "Unclear", "Unknown"],
        governance_rows,
    ), ""])

    lines.extend(["## Top Prior Dataset Names", "", markdown_table(
        ["Dataset", "Mentions"],
        [[key, value] for key, value in list(summary["top_prior_dataset_names"].items())[:20]],
    ), ""])

    lines.extend(["## Top ACU Types", "", markdown_table(
        ["ACU type", "Count"],
        [[key, value] for key, value in summary["acu_types"].items()],
    ), ""])
    return "\n".join(lines).rstrip() + "\n"


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv_tables(output_dir: str | Path, summary: dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def write_counter(name: str, counter: dict[str, int], key_name: str = "key") -> None:
        with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([key_name, "count"])
            for key, value in counter.items():
                writer.writerow([key, value])

    def write_nested(name: str, nested: dict[str, dict[str, int]], outer_name: str, inner_name: str) -> None:
        with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([outer_name, inner_name, "count"])
            for outer, inner_counts in nested.items():
                for inner, value in inner_counts.items():
                    writer.writerow([outer, inner, value])

    for name in [
        "papers_by_year",
        "datasets_by_year",
        "paper_sector",
        "roles",
        "resource_types",
        "release_status",
        "documentation_type",
        "maintenance_status",
        "prior_relationships",
        "acu_types",
        "top_tasks",
        "top_domains",
        "top_languages",
        "top_modalities",
        "top_prior_dataset_names",
        "top_source_dataset_names",
    ]:
        write_counter(name, summary[name])

    for name, outer, inner in [
        ("paper_sector_by_year", "year", "sector"),
        ("roles_by_year", "year", "role"),
        ("release_status_by_year", "year", "release_status"),
        ("artifact_by_year", "year", "artifact"),
        ("synthetic_generation_by_year", "year", "uses_llm"),
    ]:
        write_nested(name, summary[name], outer, inner)

    for key, counts in summary["governance"].items():
        write_counter(f"governance_{key}", counts, key_name="value")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready summaries from full-text dataset extraction JSONL.")
    parser.add_argument("jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--output-csv-dir", default=None)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    rows = read_jsonl(args.jsonl)
    summary = summarize(rows, top_k=args.top_k)
    write_json(args.output_json, summary)
    Path(args.output_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_markdown).write_text(build_markdown(summary), encoding="utf-8")
    if args.output_csv_dir:
        write_csv_tables(args.output_csv_dir, summary)
    print(json.dumps({
        "rows": summary["input_rows"],
        "unique_papers": summary["unique_papers"],
        "datasets": summary["datasets"],
        "total_acus": summary["total_acus"],
        "output_json": args.output_json,
        "output_markdown": args.output_markdown,
        "output_csv_dir": args.output_csv_dir,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
