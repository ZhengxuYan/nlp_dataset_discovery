#!/usr/bin/env python3
"""Merge canonical paper-enrichment shards and produce a final coverage report."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SHARD_RE = re.compile(r"_(\d{6})_(\d{6})\.jsonl$")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def record_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("public_metadata")
    if isinstance(metadata, Mapping):
        enrichment = metadata.get("metadata_enrichment")
        if isinstance(enrichment, Mapping) and enrichment.get("record_id"):
            return str(enrichment["record_id"])
    for key in ("paper_id", "arxiv_id", "arxiv_base_id"):
        if row.get(key):
            return str(row[key])
    raise ValueError("row has no stable paper identifier")


def shard_bounds(path: Path) -> tuple[int, int]:
    match = SHARD_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse shard bounds from {path}")
    return int(match.group(1)), int(match.group(2))


def quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "p75": None, "p90": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)

    def nearest_rank(fraction: float) -> int:
        index = max(0, min(len(ordered) - 1, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p75": nearest_rank(0.75),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 2),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# Final arXiv Paper Metadata Enrichment Report",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Status: `{summary['status']}`",
        f"- Input and enriched rows: {overall['rows']:,}",
        f"- Semantic Scholar matches: {overall['semantic_scholar_matches']:,} ({overall['match_rate_pct']}%)",
        f"- Citation counts available: {overall['citation_counts']:,} ({overall['citation_coverage_pct']}%)",
        f"- Exact identifier matches: {overall['exact_matches']:,}",
        f"- Direct API cost: $0",
        "",
        "## Coverage by arXiv Year",
        "",
        "| Year | Papers | S2 matches | Match rate | Citation coverage | Median citations | Mean citations | P90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for year, values in sorted(summary["by_year"].items()):
        citation = values["citation_distribution"]
        lines.append(
            f"| {year} | {values['rows']:,} | {values['semantic_scholar_matches']:,} | "
            f"{values['match_rate_pct']}% | {values['citation_coverage_pct']}% | "
            f"{citation['median']} | {citation['mean']} | {citation['p90']} |"
        )
    lines.extend(["", "## Match Methods", "", "| Method | Rows |", "| --- | ---: |"]) 
    for method, count in sorted(summary["match_methods"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {method} | {count:,} |")
    lines.extend(["", "## Missing Metadata", "", "| Reason | Rows |", "| --- | ---: |"]) 
    for reason, count in sorted(summary["missing_reasons"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {reason} | {count:,} |")
    return "\n".join(lines) + "\n"


def finalize(
    input_path: Path,
    prefix_path: Path,
    shard_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    expected_start = sum(1 for _ in iter_jsonl(prefix_path))
    ordered_shards = sorted(
        (path for path in shard_paths if shard_bounds(path)[0] >= expected_start),
        key=shard_bounds,
    )
    if not ordered_shards:
        raise ValueError(f"no canonical shards begin at or after prefix offset {expected_start}")
    cursor = expected_start
    for path in ordered_shards:
        start, end = shard_bounds(path)
        if start != cursor:
            raise ValueError(f"non-contiguous shard sequence: expected {cursor}, found {start} in {path}")
        cursor = end

    original_rows = iter_jsonl(input_path)
    enriched_paths = [prefix_path, *ordered_shards]
    rows = matches = citations = exact_matches = 0
    match_methods: Counter[str] = Counter()
    missing_reasons: Counter[str] = Counter()
    by_year_counts: dict[str, Counter[str]] = defaultdict(Counter)
    citation_values: list[int] = []
    citations_by_year: dict[str, list[int]] = defaultdict(list)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for enriched_path in enriched_paths:
            for enriched in iter_jsonl(enriched_path):
                try:
                    original = next(original_rows)
                except StopIteration as exc:
                    raise ValueError("enriched output contains more rows than the input") from exc
                if record_id(original) != record_id(enriched):
                    raise ValueError(
                        f"record ID mismatch at row {rows}: {record_id(original)!r} != {record_id(enriched)!r}"
                    )
                rows += 1
                year = str(original.get("year") or str(original.get("published_date") or "")[:4] or "unknown")
                year_counter = by_year_counts[year]
                year_counter["rows"] += 1
                metadata = enriched.get("public_metadata")
                identifiers = metadata.get("paper_identifiers") if isinstance(metadata, Mapping) else None
                metrics = metadata.get("paper_metrics") if isinstance(metadata, Mapping) else None
                sources = metadata.get("paper_metadata_sources") if isinstance(metadata, Mapping) else None
                s2_id = identifiers.get("semantic_scholar_paper_id") if isinstance(identifiers, Mapping) else None
                if s2_id:
                    matches += 1
                    year_counter["matches"] += 1
                else:
                    missing_reasons["no_semantic_scholar_match"] += 1
                citation = metrics.get("citation_count") if isinstance(metrics, Mapping) else None
                if citation is not None:
                    citation_int = int(citation)
                    citations += 1
                    year_counter["citations"] += 1
                    citation_values.append(citation_int)
                    citations_by_year[year].append(citation_int)
                else:
                    missing_reasons["no_citation_count"] += 1
                if sources:
                    for source in sources:
                        if not isinstance(source, Mapping):
                            continue
                        method = str(source.get("match_method") or "unknown")
                        match_methods[method] += 1
                        if source.get("match_confidence") == "exact":
                            exact_matches += 1
                            year_counter["exact"] += 1
                            break
                output.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    try:
        next(original_rows)
    except StopIteration:
        pass
    else:
        raise ValueError("enriched output contains fewer rows than the input")

    def pct(value: int, denominator: int) -> float:
        return round(100 * value / denominator, 2) if denominator else 0.0

    by_year: dict[str, Any] = {}
    for year, counts in sorted(by_year_counts.items()):
        by_year[year] = {
            "rows": counts["rows"],
            "semantic_scholar_matches": counts["matches"],
            "match_rate_pct": pct(counts["matches"], counts["rows"]),
            "citation_counts": counts["citations"],
            "citation_coverage_pct": pct(counts["citations"], counts["rows"]),
            "exact_matches": counts["exact"],
            "citation_distribution": quantiles(citations_by_year[year]),
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "pass",
        "input": str(input_path),
        "output": str(output_path),
        "prefix": str(prefix_path),
        "shards": len(ordered_shards),
        "overall": {
            "rows": rows,
            "semantic_scholar_matches": matches,
            "match_rate_pct": pct(matches, rows),
            "citation_counts": citations,
            "citation_coverage_pct": pct(citations, rows),
            "exact_matches": exact_matches,
            "citation_distribution": quantiles(citation_values),
        },
        "by_year": by_year,
        "match_methods": dict(match_methods),
        "missing_reasons": dict(missing_reasons),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_paths = list(Path().glob(args.shard_glob))
    if not shard_paths:
        raise ValueError(f"no shards matched {args.shard_glob!r}")
    summary = finalize(args.input, args.prefix, shard_paths, args.output)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
