#!/usr/bin/env python3
"""Plan or trigger macOS cloud placeholder hydration for selected files.

The default mode is a dry run. Execution requires ``--execute`` and uses
``brctl download`` so the script does not read placeholder file contents.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cloud_placeholders import is_cloud_placeholder, read_paths_file


DEFAULT_PATHS_FILE = Path("artifacts/remaining_high_priority_hydration_files_2020_2025.txt")
DEFAULT_OUTPUT_JSON = Path("artifacts/hydration_helper_plan_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/hydration_helper_plan_2020_2025.md")


def unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen or not path.exists():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def find_downloader(explicit: str | None = None) -> list[str] | None:
    if explicit:
        return [explicit, "download"]
    brctl = shutil.which("brctl")
    if brctl:
        return [brctl, "download"]
    return None


def run_download(command_prefix: list[str], path: Path, timeout_seconds: int) -> dict[str, Any]:
    completed = subprocess.run(
        [*command_prefix, str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "path": path.as_posix(),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "still_placeholder": is_cloud_placeholder(path),
    }


def build_plan(
    paths: list[Path],
    *,
    execute: bool,
    limit: int | None = None,
    downloader: str | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    unique_paths = unique_existing_paths(paths)
    placeholders = [path for path in unique_paths if is_cloud_placeholder(path)]
    targets = placeholders[:limit] if limit is not None else placeholders
    command_prefix = find_downloader(downloader)
    results: list[dict[str, Any]] = []
    error: str | None = None
    if execute:
        if command_prefix is None:
            error = "brctl_not_found"
        else:
            for path in targets:
                results.append(run_download(command_prefix, path, timeout_seconds))
    remaining_after = [path for path in targets if is_cloud_placeholder(path)]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execute": execute,
        "input_count": len(paths),
        "existing_unique_count": len(unique_paths),
        "target_count": len(targets),
        "unlimited_placeholder_count": len(placeholders),
        "limit": limit,
        "downloader_command": command_prefix,
        "error": error,
        "target_files": [path.as_posix() for path in targets],
        "download_results": results,
        "remaining_target_placeholders": len(remaining_after),
    }


def render_markdown(plan: dict[str, Any], paths_file: Path) -> str:
    lines = [
        "# macOS Hydration Helper Plan",
        "",
        f"- Generated: {plan.get('generated_at')}",
        f"- Source list: `{paths_file}`",
        f"- Mode: `{'execute' if plan.get('execute') else 'dry-run'}`",
        f"- Input paths: `{plan.get('input_count')}`",
        f"- Existing unique paths: `{plan.get('existing_unique_count')}`",
        f"- Placeholder targets in scope: `{plan.get('target_count')}`",
        f"- Placeholder targets before limit: `{plan.get('unlimited_placeholder_count')}`",
        f"- Remaining target placeholders after run: `{plan.get('remaining_target_placeholders')}`",
        f"- Downloader command: `{plan.get('downloader_command')}`",
        f"- Error: `{plan.get('error')}`",
        "",
        "## Dry Run",
        "",
        "```bash",
        f"python scripts/hydrate_cloud_placeholders_macos.py --paths-file {paths_file}",
        "```",
        "",
        "## Execute",
        "",
        "```bash",
        f"python scripts/hydrate_cloud_placeholders_macos.py --paths-file {paths_file} --execute",
        "```",
        "",
        "## Verify",
        "",
        "```bash",
        f"python scripts/check_cloud_placeholders.py --paths-file {paths_file} --summary-only",
        "python scripts/refresh_expansion_status_2020_2025.py",
        "python scripts/validate_expansion_readiness.py --allow-blocked",
        "```",
        "",
        "## Target Files",
        "",
    ]
    for path in plan.get("target_files", [])[:100]:
        lines.append(f"- `{path}`")
    omitted = max(0, len(plan.get("target_files", [])) - 100)
    if omitted:
        lines.append(f"- ... `{omitted}` additional files omitted")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or trigger macOS placeholder hydration.")
    parser.add_argument("--paths-file", type=Path, default=DEFAULT_PATHS_FILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--downloader", help="Path to brctl-compatible downloader binary.")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = read_paths_file(args.paths_file) if args.paths_file.exists() else []
    plan = build_plan(
        paths,
        execute=args.execute,
        limit=args.limit,
        downloader=args.downloader,
        timeout_seconds=args.timeout_seconds,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(plan, args.paths_file), encoding="utf-8")
    print(args.output_md)
    return 2 if args.execute and plan.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
