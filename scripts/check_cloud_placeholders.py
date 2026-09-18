#!/usr/bin/env python3
"""Report macOS cloud placeholder files that should be hydrated before runs."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_PATHS = [Path("scripts"), Path("scv"), Path("scrapers"), Path("data")]


def is_cloud_placeholder(path: Path) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    return stat.st_size > 0 and getattr(stat, "st_blocks", 1) == 0


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*") if child.is_file())
    return files


def read_paths_file(path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            paths.append(Path(value))
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find cloud placeholder files that may block pipeline reads.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--paths-file",
        action="append",
        type=Path,
        default=[],
        help="Read additional paths from a newline-delimited file.",
    )
    parser.add_argument("--fail-on-placeholder", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="Only print the final placeholder count.")
    parser.add_argument("--limit", type=int, help="Print at most this many placeholder paths before the count.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = list(args.paths) if args.paths else ([] if args.paths_file else list(DEFAULT_PATHS))
    for paths_file in args.paths_file:
        paths.extend(read_paths_file(paths_file))
    placeholders = [path for path in iter_files(paths) if is_cloud_placeholder(path)]
    try:
        if not args.summary_only:
            shown = placeholders if args.limit is None else placeholders[: args.limit]
            for path in shown:
                print(path)
            if args.limit is not None and len(placeholders) > args.limit:
                print(f"... {len(placeholders) - args.limit} additional placeholders omitted")
        print(f"placeholder_count={len(placeholders)}")
    except BrokenPipeError:
        with contextlib.suppress(OSError):
            sys.stdout.close()
        os._exit(1)
    if placeholders and args.fail_on_placeholder:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
