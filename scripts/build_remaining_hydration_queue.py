#!/usr/bin/env python3
"""Build a grouped queue of remaining high-priority cloud placeholders."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cloud_placeholders import is_cloud_placeholder, read_paths_file


DEFAULT_INPUT = Path("artifacts/high_priority_hydration_files_2020_2025.txt")
DEFAULT_OUTPUT_JSON = Path("artifacts/remaining_high_priority_hydration_queue_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/remaining_high_priority_hydration_queue_2020_2025.md")
DEFAULT_OUTPUT_TXT = Path("artifacts/remaining_high_priority_hydration_files_2020_2025.txt")


def build_queue(paths: list[Path], sample_limit: int = 12) -> dict[str, Any]:
    remaining = [path for path in paths if is_cloud_placeholder(path)]
    by_parent = Counter(str(path.parent) for path in remaining)
    samples: dict[str, list[str]] = defaultdict(list)
    for path in remaining:
        parent = str(path.parent)
        if len(samples[parent]) < sample_limit:
            samples[parent].append(path.as_posix())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "remaining_count": len(remaining),
        "by_parent": dict(by_parent.most_common()),
        "samples_by_parent": dict(samples),
        "remaining_files": [path.as_posix() for path in remaining],
    }


def render_markdown(queue: dict[str, Any], input_path: Path) -> str:
    lines = [
        "# Remaining High-Priority Hydration Queue",
        "",
        f"- Generated: {queue['generated_at']}",
        f"- Source list: `{input_path}`",
        f"- Remaining high-priority placeholders: `{queue['remaining_count']}`",
        "",
        "## By Directory",
        "",
        "| Directory | Remaining files |",
        "| --- | ---: |",
    ]
    for parent, count in queue.get("by_parent", {}).items():
        lines.append(f"| `{parent}` | {count} |")
    lines.extend(["", "## First Files By Directory", ""])
    for parent, samples in queue.get("samples_by_parent", {}).items():
        lines.extend([f"### `{parent}`", ""])
        lines.extend(f"- `{sample}`" for sample in samples)
        lines.append("")
    lines.extend([
        "## Verification",
        "",
        "```bash",
        "python scripts/check_cloud_placeholders.py --paths-file artifacts/remaining_high_priority_hydration_files_2020_2025.txt --summary-only",
        "python scripts/refresh_expansion_status_2020_2025.py",
        "python scripts/validate_expansion_readiness.py --allow-blocked",
        "```",
    ])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build remaining high-priority hydration queue artifacts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--sample-limit", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = read_paths_file(args.input) if args.input.exists() else []
    queue = build_queue(paths, sample_limit=args.sample_limit)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(queue, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(queue, args.input), encoding="utf-8")
    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    remaining_files = queue.get("remaining_files", [])
    args.output_txt.write_text("\n".join(remaining_files) + ("\n" if remaining_files else ""), encoding="utf-8")
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
