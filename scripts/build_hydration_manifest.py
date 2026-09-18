#!/usr/bin/env python3
"""Build a manifest of cloud placeholder files for hydration planning."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cloud_placeholders import is_cloud_placeholder


DEFAULT_PATHS = [Path("scripts"), Path("scv"), Path("scrapers"), Path("data")]
DEFAULT_OUTPUT_JSON = Path("artifacts/hydration_manifest_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/hydration_manifest_2020_2025.md")
DEFAULT_HIGH_PRIORITY_OUTPUT = Path("artifacts/high_priority_hydration_files_2020_2025.txt")

HIGH_PRIORITY_PREFIXES = (
    "scripts/",
    "scv/",
    "scrapers/arxiv_scraper/",
    "scrapers/conference_scraper/",
    "data/raw/",
    "data/processed/",
    "data/census/",
)

LOW_PRIORITY_HINTS = (
    "/.scrapy/",
    "/retrieval_cache/",
    "/extracted_papers/",
    "/visualizations/",
    "/pngs/",
    "/figures/",
    "/imgs/",
    "/pdfs/",
    "__pycache__",
)

EXCLUDED_PLACEHOLDER_SUFFIXES = (
    ".icloud-placeholder",
)


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (child for child in path.rglob("*") if child.is_file())


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def priority_for(rel_path: str) -> str:
    if any(hint in f"/{rel_path}" for hint in LOW_PRIORITY_HINTS):
        return "low"
    if any(rel_path.startswith(prefix) for prefix in HIGH_PRIORITY_PREFIXES):
        return "high"
    return "medium"


def should_exclude(rel_path: str) -> bool:
    return rel_path.endswith(EXCLUDED_PLACEHOLDER_SUFFIXES)


def build_manifest(root: Path, paths: list[Path], sample_limit: int = 20) -> dict[str, Any]:
    placeholders: list[tuple[str, int, str]] = []
    for path in iter_files(paths):
        if is_cloud_placeholder(path):
            rel = relative(path, root)
            if should_exclude(rel):
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = 0
            placeholders.append((rel, size, priority_for(rel)))

    by_root = Counter(rel.split("/", 1)[0] for rel, _, _ in placeholders)
    by_suffix = Counter((Path(rel).suffix or "<none>").lower() for rel, _, _ in placeholders)
    by_parent = Counter(str(Path(rel).parent) for rel, _, _ in placeholders)
    by_priority = Counter(priority for _, _, priority in placeholders)
    priority_samples: dict[str, list[str]] = defaultdict(list)
    priority_files: dict[str, list[str]] = defaultdict(list)
    for rel, _, priority in placeholders:
        priority_files[priority].append(rel)
        if len(priority_samples[priority]) < sample_limit:
            priority_samples[priority].append(rel)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "placeholder_count": len(placeholders),
        "placeholder_bytes": sum(size for _, size, _ in placeholders),
        "by_root": dict(sorted(by_root.items())),
        "by_suffix": dict(by_suffix.most_common()),
        "by_parent_top": dict(by_parent.most_common(40)),
        "by_priority": dict(sorted(by_priority.items())),
        "priority_files": {priority: sorted(files) for priority, files in priority_files.items()},
        "priority_samples": dict(priority_samples),
        "high_priority_prefixes": list(HIGH_PRIORITY_PREFIXES),
        "low_priority_hints": list(LOW_PRIORITY_HINTS),
        "excluded_placeholder_suffixes": list(EXCLUDED_PLACEHOLDER_SUFFIXES),
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    def table(counter: dict[str, Any], label: str, limit: int = 20) -> list[str]:
        lines = [f"## {label}", "", "| Item | Placeholder files |", "| --- | ---: |"]
        for key, value in list(counter.items())[:limit]:
            lines.append(f"| `{key}` | {value} |")
        return lines

    lines = [
        "# Hydration Manifest for 2020-2025 Expansion",
        "",
        f"- Generated: {manifest['generated_at']}",
        f"- Placeholder files: {manifest['placeholder_count']}",
        f"- Placeholder apparent bytes: {manifest['placeholder_bytes']}",
        "",
        "## Priority Meaning",
        "",
        "- `high`: scripts, SCV modules, scrapers, and current data roots needed for 2020-2025 execution.",
        "- `medium`: files that may matter but are outside known first-pass pipeline paths.",
        "- `low`: caches, extracted-paper images/PDFs, and visualization assets that are less likely to block first full pipeline execution.",
        "",
    ]
    lines.extend(table(manifest.get("by_priority", {}), "By Priority", 10))
    lines.extend([""])
    lines.extend(table(manifest.get("by_root", {}), "By Root", 20))
    lines.extend([""])
    lines.extend(table(manifest.get("by_suffix", {}), "By File Type", 20))
    lines.extend([""])
    lines.extend(table(manifest.get("by_parent_top", {}), "Top Placeholder Directories", 30))
    samples = manifest.get("priority_samples", {})
    for priority in ("high", "medium", "low"):
        values = samples.get(priority) or []
        if not values:
            continue
        lines.extend(["", f"## Sample `{priority}` Files", ""])
        lines.extend(f"- `{value}`" for value in values)
    lines.extend([
        "",
        "## Suggested Next Step",
        "",
        "Hydrate files listed in `artifacts/high_priority_hydration_files_2020_2025.txt` first, then rerun:",
        "",
        "```bash",
        "python scripts/check_cloud_placeholders.py scripts scv scrapers data --summary-only",
        "python scripts/build_hydration_manifest.py",
        "```",
    ])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a cloud-placeholder hydration manifest.")
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--high-priority-output", type=Path, default=DEFAULT_HIGH_PRIORITY_OUTPUT)
    parser.add_argument("--sample-limit", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = build_manifest(args.root, args.paths, sample_limit=args.sample_limit)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(manifest), encoding="utf-8")
    high_priority_files = manifest.get("priority_files", {}).get("high", [])
    args.high_priority_output.parent.mkdir(parents=True, exist_ok=True)
    args.high_priority_output.write_text("\n".join(high_priority_files) + ("\n" if high_priority_files else ""), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
