#!/usr/bin/env python3
"""Audit enriched dataset metadata schema completeness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_OUTPUT_JSON = Path("artifacts/metadata_schema_audit_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/metadata_schema_audit_2020_2025.md")

REQUIRED_TOP_LEVEL = (
    "paper_identifiers",
    "paper_metrics",
    "paper_metadata_sources",
    "dataset_urls",
    "dataset_name_candidates",
    "hf_metadata",
    "github_metadata",
    "pwc_metadata",
    "resource_health",
    "metadata_enrichment",
)

REQUIRED_PAPER_IDENTIFIERS = (
    "doi",
    "arxiv_id",
    "acl_anthology_id",
    "openalex_work_id",
    "semantic_scholar_paper_id",
)

REQUIRED_PAPER_METRICS = (
    "citation_count",
    "influential_citation_count",
    "reference_count",
    "publication_year",
    "venue",
    "authors",
)

REQUIRED_DATASET_URL_GROUPS = (
    "all",
    "huggingface",
    "github",
    "paperswithcode",
    "project_pages",
    "downloads",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def has_key(mapping: Any, key: str) -> bool:
    return isinstance(mapping, Mapping) and key in mapping


def value_present(mapping: Any, key: str) -> bool:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return False
    value = mapping[key]
    return value is not None and value != ""


def list_present(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def audit_metadata_items(
    *,
    items: Any,
    prefix: str,
    existing_fields: Counter[str],
    present_values: Counter[str],
    missing_reason_counts: Counter[str],
) -> None:
    if not isinstance(items, list):
        missing_reason_counts[f"{prefix}_not_list"] += 1
        return
    for item in items:
        if not isinstance(item, Mapping):
            missing_reason_counts[f"{prefix}_item_not_object"] += 1
            continue
        for field in ("source", "match_method", "match_confidence", "match_confidence_score"):
            name = f"{prefix}.{field}"
            if field in item:
                existing_fields[name] += 1
            else:
                missing_reason_counts[f"missing_{name}"] += 1
            if item.get(field) is not None and item.get(field) != "":
                present_values[name] += 1


def audit_resource_health_items(
    *,
    items: Any,
    existing_fields: Counter[str],
    present_values: Counter[str],
    missing_reason_counts: Counter[str],
) -> None:
    if not isinstance(items, list):
        missing_reason_counts["resource_health_not_list"] += 1
        return
    for item in items:
        if not isinstance(item, Mapping):
            missing_reason_counts["resource_health_item_not_object"] += 1
            continue
        for field in ("url", "status", "resolved_url", "downloadable", "checked_at"):
            name = f"resource_health.{field}"
            if field in item:
                existing_fields[name] += 1
            else:
                missing_reason_counts[f"missing_{name}"] += 1
            if item.get(field) is not None and item.get(field) != "":
                present_values[name] += 1


def audit_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    missing_structure: Counter[str] = Counter()
    present_values: Counter[str] = Counter()
    existing_fields: Counter[str] = Counter()
    missing_reason_counts: Counter[str] = Counter()

    for row in rows:
        total_rows += 1
        metadata = row.get("public_metadata")
        if not isinstance(metadata, Mapping):
            missing_structure["public_metadata"] += 1
            missing_reason_counts["missing_public_metadata"] += 1
            continue

        for field in REQUIRED_TOP_LEVEL:
            if has_key(metadata, field):
                existing_fields[f"public_metadata.{field}"] += 1
            else:
                missing_structure[f"public_metadata.{field}"] += 1

        identifiers = metadata.get("paper_identifiers")
        for field in REQUIRED_PAPER_IDENTIFIERS:
            name = f"paper_identifiers.{field}"
            if has_key(identifiers, field):
                existing_fields[name] += 1
            else:
                missing_structure[name] += 1
            if value_present(identifiers, field):
                present_values[name] += 1

        metrics = metadata.get("paper_metrics")
        for field in REQUIRED_PAPER_METRICS:
            name = f"paper_metrics.{field}"
            if has_key(metrics, field):
                existing_fields[name] += 1
            else:
                missing_structure[name] += 1
            if value_present(metrics, field):
                present_values[name] += 1

        dataset_urls = metadata.get("dataset_urls")
        for field in REQUIRED_DATASET_URL_GROUPS:
            name = f"dataset_urls.{field}"
            if has_key(dataset_urls, field):
                existing_fields[name] += 1
            else:
                missing_structure[name] += 1
            if value_present(dataset_urls, field):
                present_values[name] += 1

        dataset_name_candidates = metadata.get("dataset_name_candidates")
        if list_present(dataset_name_candidates):
            present_values["public_metadata.dataset_name_candidates"] += 1

        audit_metadata_items(
            items=metadata.get("paper_metadata_sources"),
            prefix="paper_metadata_sources",
            existing_fields=existing_fields,
            present_values=present_values,
            missing_reason_counts=missing_reason_counts,
        )
        audit_metadata_items(
            items=metadata.get("hf_metadata"),
            prefix="hf_metadata",
            existing_fields=existing_fields,
            present_values=present_values,
            missing_reason_counts=missing_reason_counts,
        )
        audit_metadata_items(
            items=metadata.get("github_metadata"),
            prefix="github_metadata",
            existing_fields=existing_fields,
            present_values=present_values,
            missing_reason_counts=missing_reason_counts,
        )
        audit_metadata_items(
            items=metadata.get("pwc_metadata"),
            prefix="pwc_metadata",
            existing_fields=existing_fields,
            present_values=present_values,
            missing_reason_counts=missing_reason_counts,
        )
        audit_resource_health_items(
            items=metadata.get("resource_health"),
            existing_fields=existing_fields,
            present_values=present_values,
            missing_reason_counts=missing_reason_counts,
        )

        if isinstance(identifiers, Mapping):
            if not any(identifiers.get(key) for key in ("doi", "arxiv_id", "acl_anthology_id")):
                missing_reason_counts["no_exact_paper_identifier"] += 1
            if not identifiers.get("openalex_work_id"):
                missing_reason_counts["no_openalex_match"] += 1
            if not identifiers.get("semantic_scholar_paper_id"):
                missing_reason_counts["no_semantic_scholar_match"] += 1
        if isinstance(metrics, Mapping) and metrics.get("citation_count") is None:
            missing_reason_counts["no_citation_count"] += 1
        if isinstance(dataset_urls, Mapping):
            if not dataset_urls.get("huggingface"):
                missing_reason_counts["no_huggingface_url"] += 1
            if not dataset_urls.get("github"):
                missing_reason_counts["no_github_url"] += 1
        if not metadata.get("hf_metadata"):
            missing_reason_counts["no_huggingface_metadata"] += 1
        if not metadata.get("github_metadata"):
            missing_reason_counts["no_github_metadata"] += 1
        if not metadata.get("pwc_metadata"):
            missing_reason_counts["no_paperswithcode_metadata"] += 1
        if not metadata.get("resource_health"):
            missing_reason_counts["no_resource_health"] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": total_rows,
        "existing_fields": dict(sorted(existing_fields.items())),
        "present_values": dict(sorted(present_values.items())),
        "missing_structure": dict(sorted(missing_structure.items())),
        "missing_reason_counts": dict(sorted(missing_reason_counts.items())),
    }


def pct(count: int, rows: int) -> str:
    return f"{(100.0 * count / rows):.1f}%" if rows else "0.0%"


def render_markdown(audit: Mapping[str, Any], source: str) -> str:
    rows = int(audit.get("rows") or 0)
    lines = [
        "# Metadata Schema Audit",
        "",
        f"- Source: `{source}`",
        f"- Rows: `{rows}`",
        f"- Generated: {audit.get('generated_at')}",
        "",
        "## Field Presence",
        "",
        "| Field | Structure present | Non-empty value |",
        "| --- | ---: | ---: |",
    ]
    existing = audit.get("existing_fields") or {}
    present = audit.get("present_values") or {}
    field_names = sorted(set(existing) | set(present))
    for field in field_names:
        lines.append(
            f"| `{field}` | {existing.get(field, 0)} ({pct(int(existing.get(field, 0)), rows)}) | "
            f"{present.get(field, 0)} ({pct(int(present.get(field, 0)), rows)}) |"
        )

    missing_structure = audit.get("missing_structure") or {}
    if missing_structure:
        lines.extend(["", "## Missing Structure", "", "| Field | Rows missing |", "| --- | ---: |"])
        for field, count in sorted(missing_structure.items()):
            lines.append(f"| `{field}` | {count} |")

    missing_reasons = audit.get("missing_reason_counts") or {}
    if missing_reasons:
        lines.extend(["", "## Missing Metadata Reasons", "", "| Reason | Rows |", "| --- | ---: |"])
        for reason, count in sorted(missing_reasons.items()):
            lines.append(f"| `{reason}` | {count} |")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit enriched metadata schema completeness.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    audit = audit_rows(iter_jsonl(args.input_jsonl))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit, str(args.input_jsonl)), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
