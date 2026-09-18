#!/usr/bin/env python3
"""Build a small review sample from enriched public metadata rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def public_metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("public_metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def citation_count(row: Mapping[str, Any]) -> int | None:
    metrics = public_metadata(row).get("paper_metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get("citation_count")
    return int(value) if isinstance(value, int | float) else None


def dataset_urls(row: Mapping[str, Any]) -> Mapping[str, Any]:
    urls = public_metadata(row).get("dataset_urls")
    return urls if isinstance(urls, Mapping) else {}


def metadata_list(row: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = public_metadata(row).get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def successful_metadata_items(row: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    return [item for item in metadata_list(row, key) if not item.get("error")]


def title(row: Mapping[str, Any]) -> str:
    for key in ("paper_title", "title", "dataset_name", "canonical_name", "name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def stable_key(row: Mapping[str, Any]) -> str:
    metadata = public_metadata(row)
    enrichment = metadata.get("metadata_enrichment")
    if isinstance(enrichment, Mapping) and enrichment.get("record_id"):
        return str(enrichment["record_id"])
    for key in ("dataset_id", "paper_id", "source_id", "id"):
        if row.get(key):
            return str(row[key])
    return title(row)


def compact_row(row: Mapping[str, Any], bucket: str) -> dict[str, Any]:
    metadata = public_metadata(row)
    identifiers = metadata.get("paper_identifiers") if isinstance(metadata.get("paper_identifiers"), Mapping) else {}
    metrics = metadata.get("paper_metrics") if isinstance(metadata.get("paper_metrics"), Mapping) else {}
    urls = dataset_urls(row)
    hf_metadata = successful_metadata_items(row, "hf_metadata")
    github_metadata = successful_metadata_items(row, "github_metadata")
    pwc_metadata = successful_metadata_items(row, "pwc_metadata")
    health = metadata_list(row, "resource_health")
    return {
        "bucket": bucket,
        "record_id": stable_key(row),
        "title": title(row),
        "year": row.get("year") or row.get("publication_year") or metrics.get("publication_year"),
        "citation_count": metrics.get("citation_count"),
        "openalex_work_id": identifiers.get("openalex_work_id"),
        "semantic_scholar_paper_id": identifiers.get("semantic_scholar_paper_id"),
        "doi": identifiers.get("doi"),
        "hf_urls": urls.get("huggingface") or [],
        "github_urls": urls.get("github") or [],
        "all_urls": urls.get("all") or [],
        "hf_match_methods": [item.get("match_method") for item in hf_metadata if item.get("match_method")],
        "github_match_methods": [item.get("match_method") for item in github_metadata if item.get("match_method")],
        "pwc_match_methods": [item.get("match_method") for item in pwc_metadata if item.get("match_method")],
        "pwc_slugs": [item.get("slug") for item in pwc_metadata if item.get("slug")],
        "healthy_url_count": sum(1 for item in health if item.get("ok")),
        "downloadable_url_count": sum(1 for item in health if item.get("downloadable")),
    }


def unique_add(samples: list[dict[str, Any]], seen: set[str], row: Mapping[str, Any], bucket: str) -> None:
    key = f"{bucket}:{stable_key(row)}"
    if key in seen:
        return
    seen.add(key)
    samples.append(compact_row(row, bucket))


def build_sample(rows: list[Mapping[str, Any]], per_bucket: int = 5) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    with_citations = [row for row in rows if citation_count(row) is not None]
    high_citation = sorted(with_citations, key=lambda row: citation_count(row) or 0, reverse=True)[:per_bucket]
    low_citation = sorted(with_citations, key=lambda row: citation_count(row) or 0)[:per_bucket]
    no_citation = [row for row in rows if citation_count(row) is None][:per_bucket]
    hf_linked = [row for row in rows if dataset_urls(row).get("huggingface")][:per_bucket]
    github_linked = [row for row in rows if dataset_urls(row).get("github")][:per_bucket]
    hf_name_fallback = [
        row
        for row in rows
        if any(item.get("match_method") == "dataset_name_fuzzy" for item in successful_metadata_items(row, "hf_metadata"))
    ][:per_bucket]
    pwc_matched = [row for row in rows if successful_metadata_items(row, "pwc_metadata")][:per_bucket]
    healthy_url = [
        row for row in rows if any(item.get("ok") for item in metadata_list(row, "resource_health"))
    ][:per_bucket]
    downloadable_url = [
        row for row in rows if any(item.get("downloadable") for item in metadata_list(row, "resource_health"))
    ][:per_bucket]
    for bucket, bucket_rows in (
        ("high_citation", high_citation),
        ("low_citation", low_citation),
        ("no_citation", no_citation),
        ("huggingface_linked", hf_linked),
        ("github_linked", github_linked),
        ("huggingface_name_fallback", hf_name_fallback),
        ("paperswithcode_matched", pwc_matched),
        ("healthy_url", healthy_url),
        ("downloadable_url", downloadable_url),
    ):
        for row in bucket_rows:
            unique_add(samples, seen, row, bucket)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(rows),
        "per_bucket": per_bucket,
        "sample_count": len(samples),
        "samples": samples,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def render_markdown(sample: Mapping[str, Any], source: str) -> str:
    lines = [
        "# Metadata Review Sample",
        "",
        f"- Source: `{source}`",
        f"- Rows: `{sample.get('rows')}`",
        f"- Sample rows: `{sample.get('sample_count')}`",
        f"- Generated: {sample.get('generated_at')}",
        "",
        "| Bucket | Title | Year | Citations | HF URLs | GitHub URLs | PWC | Healthy URLs | Downloadable URLs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sample.get("samples") or []:
        if not isinstance(row, Mapping):
            continue
        title_text = str(row.get("title") or row.get("record_id") or "")[:80]
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("bucket"),
                title_text.replace("|", "\\|"),
                row.get("year", ""),
                row.get("citation_count", ""),
                len(row.get("hf_urls") or []),
                len(row.get("github_urls") or []),
                len(row.get("pwc_slugs") or []),
                row.get("healthy_url_count", 0),
                row.get("downloadable_url_count", 0),
            )
        )
    lines.extend([
        "",
        "## Manual Audit Use",
        "",
        "Inspect a few rows from each bucket to confirm identifier matches, citation counts, and dataset resource links are plausible.",
    ])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a metadata review sample from enriched JSONL rows.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = list(iter_jsonl(args.input_jsonl))
    sample = build_sample(rows, per_bucket=args.per_bucket)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(sample, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(args.output_jsonl, sample["samples"])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(sample, str(args.input_jsonl)), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
