#!/usr/bin/env python3
"""Validate whether the 2020-2025 expansion is ready for full execution."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_cloud_placeholders import DEFAULT_PATHS, is_cloud_placeholder, iter_files, read_paths_file


DEFAULT_RUN_PLAN = Path("artifacts/corpus_expansion_2020_2025_run_plan.json")
DEFAULT_HIGH_PRIORITY = Path("artifacts/high_priority_hydration_files_2020_2025.txt")
DEFAULT_LOCAL_MANIFEST = Path("artifacts/local_smoke_2020_2025/manifest.json")
DEFAULT_REFRESH_SUMMARY = Path("artifacts/refresh_expansion_status_2020_2025.json")
DEFAULT_OUTPUT_JSON = Path("artifacts/expansion_readiness_2020_2025.json")
DEFAULT_OUTPUT_MD = Path("artifacts/expansion_readiness_2020_2025.md")

REQUIRED_STEPS = (
    "preflight_placeholder_check",
    "acl_anthology_catalog_all",
    "arxiv_interval_scrape",
    "arxiv_screening_catalog",
    "fulltext_dataset_extraction",
    "integrated_fulltext_banks",
    "public_metadata_enrichment",
    "metadata_coverage_report",
    "metadata_schema_audit",
    "expansion_regression_audit",
)

REQUIRED_LOCAL_SMOKE_KEYS = (
    "dataset_bank",
    "acu_bank",
    "enriched_jsonl",
    "coverage_md",
    "schema_audit_md",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def count_placeholders(paths: list[Path]) -> int:
    return sum(1 for path in iter_files(paths) if is_cloud_placeholder(path))


def existing_manifest_paths(manifest: Mapping[str, Any]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for key in REQUIRED_LOCAL_SMOKE_KEYS:
        value = manifest.get(key)
        result[key] = bool(value and Path(str(value)).exists())
    return result


def validate(
    run_plan: Mapping[str, Any],
    high_priority_paths: list[Path],
    local_manifest: Mapping[str, Any],
    refresh_summary: Mapping[str, Any],
    full_paths: list[Path],
) -> dict[str, Any]:
    step_names = [step.get("name") for step in run_plan.get("steps") or [] if isinstance(step, Mapping)]
    missing_steps = [step for step in REQUIRED_STEPS if step not in step_names]
    high_priority_placeholders = count_placeholders(high_priority_paths)
    full_placeholders = count_placeholders(full_paths)
    local_smoke_paths = existing_manifest_paths(local_manifest)
    missing_local_smoke = [key for key, exists in local_smoke_paths.items() if not exists]
    refresh_ok = bool(refresh_summary.get("ok")) if refresh_summary else False
    blockers: list[str] = []
    if missing_steps:
        blockers.append("run_plan_missing_required_steps")
    if high_priority_placeholders:
        blockers.append("high_priority_placeholders_remaining")
    if full_placeholders:
        blockers.append("full_placeholder_scan_not_clear")
    if missing_local_smoke:
        blockers.append("local_smoke_artifacts_missing")
    if not refresh_ok:
        blockers.append("refresh_summary_missing_or_failed")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ready_for_full_pipeline": not blockers,
        "blockers": blockers,
        "run_plan_step_count": len(step_names),
        "missing_run_plan_steps": missing_steps,
        "high_priority_placeholder_count": high_priority_placeholders,
        "full_placeholder_count": full_placeholders,
        "local_smoke_artifacts": local_smoke_paths,
        "missing_local_smoke_artifacts": missing_local_smoke,
        "refresh_ok": refresh_ok,
    }


def render_markdown(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# Expansion Readiness: 2020-2025",
        "",
        f"- Generated: {readiness.get('generated_at')}",
        f"- Ready for full pipeline: `{readiness.get('ready_for_full_pipeline')}`",
        f"- Run plan steps: `{readiness.get('run_plan_step_count')}`",
        f"- High-priority placeholders: `{readiness.get('high_priority_placeholder_count')}`",
        f"- Full placeholder scan: `{readiness.get('full_placeholder_count')}`",
        f"- Refresh summary ok: `{readiness.get('refresh_ok')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = readiness.get("blockers") or []
    lines.extend(f"- `{blocker}`" for blocker in blockers) if blockers else lines.append("- None")
    lines.extend(["", "## Local Smoke Artifacts", "", "| Artifact | Exists |", "| --- | --- |"])
    for key, exists in (readiness.get("local_smoke_artifacts") or {}).items():
        lines.append(f"| `{key}` | `{exists}` |")
    lines.extend([
        "",
        "## Next Commands",
        "",
        "```bash",
        "python scripts/check_cloud_placeholders.py --paths-file artifacts/high_priority_hydration_files_2020_2025.txt --summary-only",
        "python scripts/refresh_expansion_status_2020_2025.py",
        "python scripts/validate_expansion_readiness.py",
        "python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps",
        "```",
    ])
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate readiness for the full 2020-2025 expansion run.")
    parser.add_argument("--run-plan", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--high-priority-paths", type=Path, default=DEFAULT_HIGH_PRIORITY)
    parser.add_argument("--local-manifest", type=Path, default=DEFAULT_LOCAL_MANIFEST)
    parser.add_argument("--refresh-summary", type=Path, default=DEFAULT_REFRESH_SUMMARY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--allow-blocked", action="store_true", help="Return 0 even when readiness blockers remain.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    high_priority_paths = read_paths_file(args.high_priority_paths) if args.high_priority_paths.exists() else []
    readiness = validate(
        load_json(args.run_plan),
        high_priority_paths,
        load_json(args.local_manifest),
        load_json(args.refresh_summary),
        list(DEFAULT_PATHS),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(readiness), encoding="utf-8")
    print(args.output_md)
    if readiness["ready_for_full_pipeline"] or args.allow_blocked:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
