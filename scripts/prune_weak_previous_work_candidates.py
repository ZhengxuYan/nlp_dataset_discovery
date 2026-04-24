#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scv.benchmark_store import (
    DEFAULT_BENCHMARK_DRAFTS_PATH,
    DEFAULT_PREVIOUS_WORK_PATH,
    load_benchmark_drafts,
    load_previous_work_candidates,
    now_iso,
    write_jsonl,
)


def backup_path(path: Path) -> Path:
    stamp = now_iso().replace(":", "").replace("+", "Z")
    return path.with_suffix(path.suffix + f".bak.{stamp}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove weak previous-work candidates that are method-only loosely related items without dataset names.")
    parser.add_argument("--paper-ids", nargs="*", default=None)
    args = parser.parse_args()

    selected = set(args.paper_ids or [])
    candidates = load_previous_work_candidates()
    drafts = load_benchmark_drafts()
    previous_backup = backup_path(DEFAULT_PREVIOUS_WORK_PATH)
    draft_backup = backup_path(DEFAULT_BENCHMARK_DRAFTS_PATH)
    shutil.copy2(DEFAULT_PREVIOUS_WORK_PATH, previous_backup)
    shutil.copy2(DEFAULT_BENCHMARK_DRAFTS_PATH, draft_backup)

    removed_ids = set()
    kept = []
    for candidate in candidates:
        in_scope = not selected or candidate.query_paper_id in selected
        should_remove = (
            in_scope
            and not candidate.dataset_name
            and candidate.relationship_type == "loosely_related"
        )
        if should_remove:
            removed_ids.add(candidate.candidate_id)
            continue
        kept.append(candidate)
    write_jsonl(DEFAULT_PREVIOUS_WORK_PATH, kept)

    updated_drafts = []
    for draft in drafts:
        if selected and draft.query_paper_id not in selected:
            updated_drafts.append(draft)
            continue
        draft.candidate_ids = [candidate_id for candidate_id in draft.candidate_ids if candidate_id not in removed_ids]
        draft.hard_negative_ids = [candidate_id for candidate_id in draft.hard_negative_ids if candidate_id not in removed_ids]
        draft.soft_negative_ids = [candidate_id for candidate_id in draft.soft_negative_ids if candidate_id not in removed_ids]
        draft.suggested_hard_negative_ids = [candidate_id for candidate_id in draft.suggested_hard_negative_ids if candidate_id not in removed_ids]
        draft.suggested_soft_negative_ids = [candidate_id for candidate_id in draft.suggested_soft_negative_ids if candidate_id not in removed_ids]
        updated_drafts.append(draft)
    write_jsonl(DEFAULT_BENCHMARK_DRAFTS_PATH, updated_drafts)

    print(f"Removed {len(removed_ids)} weak previous-work candidates.")
    print(f"Backup written to {previous_backup}.")
    print(f"Backup written to {draft_backup}.")


if __name__ == "__main__":
    main()
