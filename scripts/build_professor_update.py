#!/usr/bin/env python3
"""Create a concise professor-facing update for the expansion work."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_OUTPUT = Path("artifacts/professor_update_2020_2025.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def render_update(run_plan: Mapping[str, Any], progress_json: Mapping[str, Any] | None = None) -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    placeholder_count = run_plan.get("placeholder_count", "unknown")
    safe = run_plan.get("safe_to_execute_full_pipeline", False)
    steps = run_plan.get("steps") or []
    step_lines = "\n".join(
        f"- {step.get('name')}: `{step.get('command_string')}`"
        for step in steps
        if isinstance(step, Mapping)
    )
    return f"""# Professor Update: Dataset Discovery Expansion

Generated: {generated_at}

## Short Status
I have converted the project from a mostly hardcoded 2023-2025 setup into a reproducible 2020-2025 expansion workflow. The actual ACL/arXiv/census entrypoints now accept year-range parameters, old 2023-2025 artifacts are preserved, and the new expected 2020-2025 outputs are explicitly named.

## Concrete Progress
- Added preflight checks for macOS cloud placeholder files, which were causing script/data reads to hang.
- Added a hydration manifest and high-priority file list that prioritize remaining placeholder files by execution relevance.
- Added a short hydration status artifact that combines the blocker count, local smoke evidence, and exact next commands.
- Parameterized ACL Anthology catalog generation, arXiv interval scraping, arXiv screening-catalog preparation, and dataset census defaults for 2020-2025.
- Added public metadata enrichment for paper identifiers/citations and dataset resource signals: OpenAlex, Semantic Scholar, Hugging Face downloads/likes, GitHub stars/forks, Papers with Code, and URL health.
- Added a safe orchestrator that writes the exact smoke/full run plan and refuses full execution while placeholder files remain.
- Ran a local end-to-end smoke path that builds an integrated dataset/ACU bank, enriches it from cached public metadata, and produces citation/download/star coverage output.
- Added metadata coverage reporting so the final enriched bank can report citation/download/star coverage rates.
- Added metadata schema auditing so the final enriched bank can report which planned metadata field groups are structurally present and why values are missing.
- Added a regression audit for comparing the old 2023-2025 artifacts against the new 2020-2025 artifacts on the 2023-2025 overlap.
- Added focused tests for year filtering, output naming, pipeline parameterization, enrichment parsing, coverage reporting, and run-plan generation.

## Current Blocker
The remaining blocker is operational: the workspace still has `{placeholder_count}` cloud placeholder files. Full execution should wait until that count is `0`; otherwise the old full-data scripts can block while reading cloud-only files.

Safe to execute full pipeline now: `{safe}`

## Run Plan
{step_lines}

## Next Step
Hydrate the remaining files locally, rerun `python scripts/check_cloud_placeholders.py scripts scv scrapers data --summary-only`, then run the orchestrator in full mode:

```bash
python scripts/build_hydration_manifest.py
python scripts/build_hydration_status_update.py
python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps
python scripts/audit_expansion_regression.py
```
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a professor-facing progress update.")
    parser.add_argument("--run-plan", type=Path, default=Path("artifacts/corpus_expansion_2020_2025_run_plan.json"))
    parser.add_argument("--progress-json", type=Path, default=Path("artifacts/progress_update_2020_2025.json"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_plan = load_json(args.run_plan)
    progress_json = load_json(args.progress_json)
    markdown = render_update(run_plan, progress_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
