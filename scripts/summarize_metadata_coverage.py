#!/usr/bin/env python3
"""Summarize public metadata enrichment coverage for reporting."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def pct(numerator: int, denominator: int) -> float:
    return round((100.0 * numerator / denominator), 1) if denominator else 0.0


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def summarize_enriched_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {
        "rows": 0,
        "paper_enrichment_rows": 0,
        "resource_enrichment_rows": 0,
        "paper_doi": 0,
        "paper_arxiv_id": 0,
        "paper_acl_id": 0,
        "paper_openalex_id": 0,
        "paper_semantic_scholar_id": 0,
        "paper_citation_count": 0,
        "paper_influential_citation_count": 0,
        "hf_dataset_links": 0,
        "hf_metadata_candidates": 0,
        "hf_download_counts": 0,
        "github_links": 0,
        "github_metadata_candidates": 0,
        "github_star_counts": 0,
        "pwc_links": 0,
        "pwc_metadata_candidates": 0,
        "healthy_urls": 0,
        "total_urls": 0,
        "paper_metadata_source_counts": {},
        "year_counts": {},
    }
    for row in rows:
        summary["rows"] += 1
        metadata = row.get("public_metadata")
        if not isinstance(metadata, Mapping):
            continue
        enrichment = metadata.get("metadata_enrichment")
        paper_only = bool(isinstance(enrichment, Mapping) and enrichment.get("paper_only"))
        resource_only = bool(isinstance(enrichment, Mapping) and enrichment.get("resource_only"))
        if not resource_only:
            summary["paper_enrichment_rows"] += 1
        if not paper_only:
            summary["resource_enrichment_rows"] += 1
        identifiers = metadata.get("paper_identifiers")
        if isinstance(identifiers, Mapping):
            summary["paper_doi"] += int(bool(identifiers.get("doi")))
            summary["paper_arxiv_id"] += int(bool(identifiers.get("arxiv_id")))
            summary["paper_acl_id"] += int(bool(identifiers.get("acl_anthology_id")))
            summary["paper_openalex_id"] += int(bool(identifiers.get("openalex_work_id")))
            summary["paper_semantic_scholar_id"] += int(bool(identifiers.get("semantic_scholar_paper_id")))
        metrics = metadata.get("paper_metrics")
        if isinstance(metrics, Mapping):
            summary["paper_citation_count"] += int(metrics.get("citation_count") is not None)
            summary["paper_influential_citation_count"] += int(metrics.get("influential_citation_count") is not None)
            year = metrics.get("publication_year")
            if year:
                year_counts = summary["year_counts"]
                year_counts[str(year)] = year_counts.get(str(year), 0) + 1
        for source in metadata.get("paper_metadata_sources") or []:
            if isinstance(source, Mapping) and source.get("source"):
                source_counts = summary["paper_metadata_source_counts"]
                source_name = str(source.get("source"))
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
        dataset_urls = metadata.get("dataset_urls")
        if isinstance(dataset_urls, Mapping):
            summary["hf_dataset_links"] += len(dataset_urls.get("huggingface") or [])
            summary["github_links"] += len(dataset_urls.get("github") or [])
            summary["pwc_links"] += len(dataset_urls.get("paperswithcode") or [])
            summary["total_urls"] += len(dataset_urls.get("all") or [])
        hf_items = [item for item in metadata.get("hf_metadata") or [] if isinstance(item, Mapping)]
        github_items = [item for item in metadata.get("github_metadata") or [] if isinstance(item, Mapping)]
        pwc_items = [item for item in metadata.get("pwc_metadata") or [] if isinstance(item, Mapping)]
        summary["hf_metadata_candidates"] += len(hf_items)
        summary["github_metadata_candidates"] += len(github_items)
        summary["pwc_metadata_candidates"] += len(pwc_items)
        for item in hf_items:
            if isinstance(item, Mapping) and item.get("downloads") is not None:
                summary["hf_download_counts"] += 1
        for item in github_items:
            if isinstance(item, Mapping) and item.get("stars") is not None:
                summary["github_star_counts"] += 1
        for item in metadata.get("resource_health") or []:
            if isinstance(item, Mapping) and item.get("ok"):
                summary["healthy_urls"] += 1

    rows_count = int(summary["rows"])
    paper_rows = int(summary["paper_enrichment_rows"])
    resource_rows = int(summary["resource_enrichment_rows"])
    hf_basis = max(int(summary["hf_dataset_links"]), int(summary["hf_metadata_candidates"]))
    github_basis = max(int(summary["github_links"]), int(summary["github_metadata_candidates"]))
    summary["coverage_rates"] = {
        "citation_count_pct": pct(int(summary["paper_citation_count"]), rows_count),
        "semantic_scholar_id_pct": pct(int(summary["paper_semantic_scholar_id"]), rows_count),
        "openalex_id_pct": pct(int(summary["paper_openalex_id"]), rows_count),
        "hf_download_count_per_hf_resource_pct": pct(int(summary["hf_download_counts"]), hf_basis),
        "hf_download_count_per_hf_link_pct": pct(int(summary["hf_download_counts"]), hf_basis),
        "github_star_count_per_github_resource_pct": pct(int(summary["github_star_counts"]), github_basis),
        "github_star_count_per_github_link_pct": pct(int(summary["github_star_counts"]), github_basis),
        "healthy_url_pct": pct(int(summary["healthy_urls"]), int(summary["total_urls"])),
    }
    summary["paper_metadata_coverage"] = {
        "attempted_rows": paper_rows,
        "citation_count": {
            "count": int(summary["paper_citation_count"]),
            "pct_of_attempted": pct(int(summary["paper_citation_count"]), paper_rows),
        },
        "semantic_scholar_id": {
            "count": int(summary["paper_semantic_scholar_id"]),
            "pct_of_attempted": pct(int(summary["paper_semantic_scholar_id"]), paper_rows),
        },
        "openalex_id": {
            "count": int(summary["paper_openalex_id"]),
            "pct_of_attempted": pct(int(summary["paper_openalex_id"]), paper_rows),
        },
        "source_counts": dict(summary["paper_metadata_source_counts"]),
    }
    summary["resource_metadata_coverage"] = {
        "attempted_rows": resource_rows,
        "hf": {
            "links": int(summary["hf_dataset_links"]),
            "metadata_candidates": int(summary["hf_metadata_candidates"]),
            "download_counts": int(summary["hf_download_counts"]),
            "pct_of_hf_resources": pct(int(summary["hf_download_counts"]), hf_basis),
        },
        "github": {
            "links": int(summary["github_links"]),
            "metadata_candidates": int(summary["github_metadata_candidates"]),
            "star_counts": int(summary["github_star_counts"]),
            "pct_of_github_resources": pct(int(summary["github_star_counts"]), github_basis),
        },
        "paperswithcode": {
            "links": int(summary["pwc_links"]),
            "metadata_candidates": int(summary["pwc_metadata_candidates"]),
        },
        "url_health": {
            "checked_or_known_urls": int(summary["total_urls"]),
            "healthy_urls": int(summary["healthy_urls"]),
            "pct_of_urls": pct(int(summary["healthy_urls"]), int(summary["total_urls"])),
        },
    }
    summary["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return summary


def render_markdown(summary: Mapping[str, Any], source: str) -> str:
    rates = summary.get("coverage_rates") or {}
    rows = int(summary.get("rows") or 0)
    paper_coverage = summary.get("paper_metadata_coverage") or {}
    resource_coverage = summary.get("resource_metadata_coverage") or {}
    paper_attempted = int(paper_coverage.get("attempted_rows") or summary.get("paper_enrichment_rows") or 0)
    resource_attempted = int(resource_coverage.get("attempted_rows") or summary.get("resource_enrichment_rows") or 0)
    hf = resource_coverage.get("hf") if isinstance(resource_coverage.get("hf"), Mapping) else {}
    github = resource_coverage.get("github") if isinstance(resource_coverage.get("github"), Mapping) else {}
    pwc = resource_coverage.get("paperswithcode") if isinstance(resource_coverage.get("paperswithcode"), Mapping) else {}
    url_health = resource_coverage.get("url_health") if isinstance(resource_coverage.get("url_health"), Mapping) else {}
    lines = [
        "# Public Metadata Coverage",
        "",
        f"- Source: `{source}`",
        f"- Rows: {rows}",
        f"- Paper enrichment attempted rows: {paper_attempted}",
        f"- Resource enrichment attempted rows: {resource_attempted}",
        f"- Generated: {summary.get('generated_at')}",
        "",
        "## Paper Metadata",
        "",
        "This section was not run for this file." if paper_attempted == 0 else f"Coverage denominator: {paper_attempted} attempted paper rows.",
        "",
        "| Field | Count | Coverage among attempted paper rows |",
        "| --- | ---: | ---: |",
        f"| DOI | {summary.get('paper_doi', 0)} | {pct(int(summary.get('paper_doi', 0)), rows)}% |",
        f"| arXiv ID | {summary.get('paper_arxiv_id', 0)} | {pct(int(summary.get('paper_arxiv_id', 0)), rows)}% |",
        f"| ACL ID | {summary.get('paper_acl_id', 0)} | {pct(int(summary.get('paper_acl_id', 0)), rows)}% |",
        f"| OpenAlex ID | {summary.get('paper_openalex_id', 0)} | {pct(int(summary.get('paper_openalex_id', 0)), paper_attempted)}% |",
        f"| Semantic Scholar ID | {summary.get('paper_semantic_scholar_id', 0)} | {pct(int(summary.get('paper_semantic_scholar_id', 0)), paper_attempted)}% |",
        f"| Citation count | {summary.get('paper_citation_count', 0)} | {pct(int(summary.get('paper_citation_count', 0)), paper_attempted)}% |",
        "",
        "## Dataset Resource Metadata",
        "",
        "This section was not run for this file." if resource_attempted == 0 else f"Coverage denominator: explicit HF/GitHub/Papers with Code resources found in {resource_attempted} attempted resource rows.",
        "",
        "| Field | Count | Coverage basis |",
        "| --- | ---: | --- |",
        f"| Hugging Face dataset links | {summary.get('hf_dataset_links', 0)} | rows may have multiple links |",
        f"| Hugging Face metadata candidates | {summary.get('hf_metadata_candidates', 0)} | explicit links plus name-fallback candidates |",
        f"| Hugging Face download counts | {summary.get('hf_download_counts', 0)} | {hf.get('pct_of_hf_resources', rates.get('hf_download_count_per_hf_resource_pct', rates.get('hf_download_count_per_hf_link_pct', 0)))}% of HF resources |",
        f"| GitHub links | {summary.get('github_links', 0)} | rows may have multiple links |",
        f"| GitHub metadata candidates | {summary.get('github_metadata_candidates', 0)} | explicit links plus metadata candidates |",
        f"| GitHub star counts | {summary.get('github_star_counts', 0)} | {github.get('pct_of_github_resources', rates.get('github_star_count_per_github_resource_pct', rates.get('github_star_count_per_github_link_pct', 0)))}% of GitHub resources |",
        f"| Papers with Code links | {summary.get('pwc_links', 0)} | rows may have multiple links |",
        f"| Papers with Code metadata candidates | {pwc.get('metadata_candidates', summary.get('pwc_metadata_candidates', 0))} | explicit links plus name-fallback candidates |",
        f"| Healthy URLs | {url_health.get('healthy_urls', summary.get('healthy_urls', 0))} | {url_health.get('pct_of_urls', rates.get('healthy_url_pct', 0))}% of all URLs |",
    ]
    year_counts = summary.get("year_counts") or {}
    if year_counts:
        lines.extend(["", "## Publication Years", "", "| Year | Rows |", "| --- | ---: |"])
        for year, count in sorted(year_counts.items()):
            lines.append(f"| {year} | {count} |")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize metadata enrichment coverage.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = summarize_enriched_rows(iter_jsonl(args.input_jsonl))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown = render_markdown(summary, str(args.input_jsonl))
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
