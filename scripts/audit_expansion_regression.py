#!/usr/bin/env python3
"""Audit whether expanded 2020-2025 artifacts preserve 2023-2025 coverage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cloud_placeholders import is_cloud_placeholder
from scripts.corpus_expansion import YearRange, extract_record_year, output_paths


DEFAULT_OUTPUT_JSON = Path("artifacts/expansion_regression_audit_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/expansion_regression_audit_2020_2025.md")


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if isinstance(value, Mapping):
                yield value


def iter_csv(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def iter_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    if path.suffix == ".jsonl":
        return iter_jsonl(path)
    if path.suffix == ".csv":
        return iter_csv(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return [value] if isinstance(value, Mapping) else []
    return []


def audit_file(path: Path, overlap: YearRange) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "status": "missing"}
    if is_cloud_placeholder(path):
        return {"path": str(path), "exists": True, "status": "cloud_placeholder"}
    if path.suffix not in {".jsonl", ".csv", ".json"}:
        return {"path": str(path), "exists": True, "status": "unsupported"}

    total_rows = 0
    missing_year_rows = 0
    overlap_rows = 0
    year_counts: Counter[str] = Counter()
    for row in iter_rows(path):
        total_rows += 1
        year = extract_record_year(row)
        if year is None:
            missing_year_rows += 1
            continue
        year_counts[str(year)] += 1
        if overlap.includes(year):
            overlap_rows += 1
    return {
        "path": str(path),
        "exists": True,
        "status": "counted",
        "total_rows": total_rows,
        "overlap_rows": overlap_rows,
        "missing_year_rows": missing_year_rows,
        "year_counts": dict(sorted(year_counts.items())),
    }


def artifact_pairs(root: Path, old_range: YearRange, new_range: YearRange) -> dict[str, tuple[Path, Path]]:
    old_paths = output_paths(root, old_range)
    new_paths = output_paths(root, new_range)
    pairs = {name: (old_paths[name], new_paths[name]) for name in sorted(old_paths)}
    for scope in ("all", "core"):
        name = f"acl_anthology_{scope}_jsonl"
        pairs[name] = (
            root / "data" / "census" / "acl_anthology" / f"acl_anthology_{old_range.label}_{scope}.jsonl",
            root / "data" / "census" / "acl_anthology" / f"acl_anthology_{new_range.label}_{scope}.jsonl",
        )
    return pairs


def compare_artifact(old_audit: Mapping[str, Any], new_audit: Mapping[str, Any]) -> dict[str, Any]:
    old_status = old_audit.get("status")
    new_status = new_audit.get("status")
    if old_status != "counted":
        status = "no_counted_baseline"
    elif new_status != "counted":
        status = "pending_new_artifact"
    elif int(new_audit.get("overlap_rows") or 0) >= int(old_audit.get("overlap_rows") or 0):
        status = "pass"
    else:
        status = "fail_shrunk_overlap"
    return {
        "status": status,
        "old_overlap_rows": old_audit.get("overlap_rows"),
        "new_overlap_rows": new_audit.get("overlap_rows"),
        "old_status": old_status,
        "new_status": new_status,
    }


def build_audit(root: Path, old_range: YearRange, new_range: YearRange) -> dict[str, Any]:
    overlap = YearRange(max(old_range.start_year, new_range.start_year), min(old_range.end_year, new_range.end_year))
    artifacts: dict[str, Any] = {}
    for name, (old_path, new_path) in artifact_pairs(root, old_range, new_range).items():
        old_audit = audit_file(old_path, overlap)
        new_audit = audit_file(new_path, overlap)
        artifacts[name] = {
            "old": old_audit,
            "new": new_audit,
            "comparison": compare_artifact(old_audit, new_audit),
        }
    summary = Counter(item["comparison"]["status"] for item in artifacts.values())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "old_range": {"start_year": old_range.start_year, "end_year": old_range.end_year, "label": old_range.label},
        "new_range": {"start_year": new_range.start_year, "end_year": new_range.end_year, "label": new_range.label},
        "overlap_range": {"start_year": overlap.start_year, "end_year": overlap.end_year, "label": overlap.label},
        "summary": dict(sorted(summary.items())),
        "artifacts": artifacts,
    }


def render_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Expansion Regression Audit: 2020-2025 vs 2023-2025",
        "",
        f"- Generated: {audit.get('generated_at')}",
        f"- Old range: `{audit.get('old_range', {}).get('label')}`",
        f"- New range: `{audit.get('new_range', {}).get('label')}`",
        f"- Overlap checked: `{audit.get('overlap_range', {}).get('label')}`",
        "",
        "## Summary",
        "",
        "| Status | Artifacts |",
        "| --- | ---: |",
    ]
    for status, count in (audit.get("summary") or {}).items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend([
        "",
        "## Artifact Comparisons",
        "",
        "| Artifact | Status | Old overlap rows | New overlap rows | Old file status | New file status |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ])
    for name, item in (audit.get("artifacts") or {}).items():
        comparison = item.get("comparison") or {}
        lines.append(
            "| `{}` | `{}` | {} | {} | `{}` | `{}` |".format(
                name,
                comparison.get("status"),
                comparison.get("old_overlap_rows", ""),
                comparison.get("new_overlap_rows", ""),
                comparison.get("old_status"),
                comparison.get("new_status"),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "`pass` means the expanded artifact has at least as many rows in the 2023-2025 overlap as the old 2023-2025 artifact.",
        "`pending_new_artifact` is expected before the full 2020-2025 run has produced final outputs.",
        "`fail_shrunk_overlap` should be investigated before using the expanded corpus for analysis.",
    ])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit old-vs-expanded corpus artifact counts.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--old-start-year", type=int, default=2023)
    parser.add_argument("--old-end-year", type=int, default=2025)
    parser.add_argument("--new-start-year", type=int, default=2020)
    parser.add_argument("--new-end-year", type=int, default=2025)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    audit = build_audit(
        args.root,
        YearRange(args.old_start_year, args.old_end_year),
        YearRange(args.new_start_year, args.new_end_year),
    )
    output_json = args.output_json if args.output_json.is_absolute() else args.root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else args.root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
