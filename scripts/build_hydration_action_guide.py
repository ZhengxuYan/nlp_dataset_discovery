#!/usr/bin/env python3
"""Build an action guide for hydrating cloud placeholder files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MANIFEST = Path("artifacts/hydration_manifest_2020_2025.json")
DEFAULT_OUTPUT = Path("artifacts/hydration_action_guide_2020_2025.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render_guide(manifest: Mapping[str, Any]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    by_priority = manifest.get("by_priority") or {}
    placeholder_count = manifest.get("placeholder_count", "unknown")
    high = by_priority.get("high", 0)
    medium = by_priority.get("medium", 0)
    low = by_priority.get("low", 0)
    return f"""# Hydration Action Guide: 2020-2025 Expansion

Generated: {generated_at}

## Current Hydration Scope
- Total placeholder files: `{placeholder_count}`
- High-priority placeholders: `{high}`
- Medium-priority placeholders: `{medium}`
- Low-priority placeholders: `{low}`

## First Target
Hydrate the high-priority list first:

`artifacts/high_priority_hydration_files_2020_2025.txt`

These files cover the scripts, SCV modules, scrapers, and current data roots most likely to block the staged/full run.
For a grouped view of only the files still blocked, use:

`artifacts/remaining_high_priority_hydration_queue_2020_2025.md`

## Recommended Manual Steps
1. Open the repository folder in Finder.
2. Use the high-priority list to locate the listed files or parent folders.
3. For iCloud/Dropbox/OneDrive-style placeholders, choose the provider's local download action for those files or folders.
4. Re-run the high-priority placeholder check until it returns `placeholder_count=0`.
5. Refresh all local status artifacts and re-check readiness.

## Optional macOS Helper
Preview the exact remaining high-priority files that the helper would target:

```bash
python scripts/hydrate_cloud_placeholders_macos.py --paths-file artifacts/remaining_high_priority_hydration_files_2020_2025.txt
```

If the dry-run looks right, explicitly trigger macOS cloud download requests:

```bash
python scripts/hydrate_cloud_placeholders_macos.py --paths-file artifacts/remaining_high_priority_hydration_files_2020_2025.txt --execute
```

The helper uses `brctl download` when available and does not read placeholder file contents.

## Verification Commands
```bash
python scripts/check_cloud_placeholders.py --paths-file artifacts/high_priority_hydration_files_2020_2025.txt --summary-only
python scripts/check_cloud_placeholders.py --paths-file artifacts/remaining_high_priority_hydration_files_2020_2025.txt --summary-only
python scripts/refresh_expansion_status_2020_2025.py
python scripts/validate_expansion_readiness.py --allow-blocked
```

## After High-Priority Hydration
Run the staged 2020-2021 smoke plan first:

```bash
python scripts/run_corpus_expansion_2020_2025.py --start-year 2020 --end-year 2021 --mode smoke --plan-output artifacts/corpus_expansion_2020_2021_smoke_run_plan.json
```

Then, after the readiness gate is clear, run the full expansion:

```bash
python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps
```

## Notes
- Do not overwrite existing `2023_2025` outputs; the new pipeline writes `2020_2025` artifacts separately.
- If readiness still reports `full_placeholder_scan_not_clear` after high-priority hydration, inspect `artifacts/hydration_manifest_2020_2025.md` for medium-priority files that may still matter.
- Citation/download/star metrics are point-in-time values; refresh enrichment reports after the full run.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a hydration action guide.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    markdown = render_guide(load_json(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
