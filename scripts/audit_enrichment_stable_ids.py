#!/usr/bin/env python3
"""Audit that metadata enrichment preserves stable record IDs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.enrich_dataset_metadata import stable_record_id


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield row


def enriched_record_id(row: Mapping[str, Any]) -> str:
    public_metadata = row.get("public_metadata")
    if isinstance(public_metadata, Mapping):
        enrichment = public_metadata.get("metadata_enrichment")
        if isinstance(enrichment, Mapping) and enrichment.get("record_id"):
            return str(enrichment["record_id"])
    stripped = dict(row)
    stripped.pop("public_metadata", None)
    return stable_record_id(stripped)


def duplicate_ids(ids: list[str]) -> dict[str, int]:
    return {record_id: count for record_id, count in Counter(ids).items() if count > 1}


def build_audit(input_jsonl: Path, enriched_jsonl: Path) -> dict[str, Any]:
    input_rows = list(iter_jsonl(input_jsonl))
    enriched_rows = list(iter_jsonl(enriched_jsonl))
    input_ids = [stable_record_id(row) for row in input_rows]
    enriched_ids = [enriched_record_id(row) for row in enriched_rows]
    input_set = set(input_ids)
    enriched_set = set(enriched_ids)
    input_duplicates = duplicate_ids(input_ids)
    enriched_duplicates = duplicate_ids(enriched_ids)
    missing = sorted(input_set - enriched_set)
    extra = sorted(enriched_set - input_set)
    status = "pass" if not input_duplicates and not enriched_duplicates and not missing and not extra else "fail"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_jsonl": str(input_jsonl),
        "enriched_jsonl": str(enriched_jsonl),
        "status": status,
        "input_rows": len(input_rows),
        "enriched_rows": len(enriched_rows),
        "unique_input_ids": len(input_set),
        "unique_enriched_ids": len(enriched_set),
        "matched_count": len(input_set & enriched_set),
        "input_duplicate_ids": input_duplicates,
        "enriched_duplicate_ids": enriched_duplicates,
        "missing_in_enriched": missing,
        "extra_in_enriched": extra,
        "uses_row_position": False,
        "matching_key": "public_metadata.metadata_enrichment.record_id",
    }


def render_markdown(audit: Mapping[str, Any]) -> str:
    status = audit.get("status")
    lines = [
        "# Enrichment Stable ID Audit",
        "",
        f"- Generated: {audit.get('generated_at')}",
        f"- Status: `{status}`",
        f"- Matching key: `{audit.get('matching_key')}`",
        f"- Uses row position: `{audit.get('uses_row_position')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Input rows | `{audit.get('input_rows')}` |",
        f"| Enriched rows | `{audit.get('enriched_rows')}` |",
        f"| Unique input IDs | `{audit.get('unique_input_ids')}` |",
        f"| Unique enriched IDs | `{audit.get('unique_enriched_ids')}` |",
        f"| Matched IDs | `{audit.get('matched_count')}` |",
        f"| Missing in enriched | `{len(audit.get('missing_in_enriched') or [])}` |",
        f"| Extra in enriched | `{len(audit.get('extra_in_enriched') or [])}` |",
        f"| Input duplicate IDs | `{len(audit.get('input_duplicate_ids') or {})}` |",
        f"| Enriched duplicate IDs | `{len(audit.get('enriched_duplicate_ids') or {})}` |",
        "",
    ]
    if status == "pass":
        lines.append("Enrichment preserved the original dataset-bank record IDs without relying on row order.")
    else:
        lines.extend(
            [
                "Enrichment ID preservation failed. Review duplicate, missing, or extra IDs before trusting merged metadata.",
                "",
                f"- Missing sample: `{(audit.get('missing_in_enriched') or [])[:10]}`",
                f"- Extra sample: `{(audit.get('extra_in_enriched') or [])[:10]}`",
            ]
        )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit stable ID preservation after public metadata enrichment.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--enriched-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    audit = build_audit(args.input_jsonl, args.enriched_jsonl)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(audit), encoding="utf-8")
    print(args.output_md)
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
