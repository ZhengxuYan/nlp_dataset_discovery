#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scv.benchmark_models import BenchmarkDraftRecord
from scv.benchmark_store import (
    DEFAULT_BENCHMARK_DRAFTS_PATH,
    DEFAULT_PREVIOUS_WORK_PATH,
    load_benchmark_drafts,
    now_iso,
    summarize_query_status,
    write_jsonl,
)


def backup_path(path: Path) -> Path:
    stamp = now_iso().replace(":", "").replace("+", "Z")
    return path.with_suffix(path.suffix + f".bak.{stamp}")


def clear_candidates(candidates_path: Path, drafts_path: Path) -> tuple[int, int, list[Path]]:
    backups: list[Path] = []
    candidate_count = 0
    if candidates_path.exists():
        candidate_count = sum(1 for line in candidates_path.read_text().splitlines() if line.strip())
        candidate_backup = backup_path(candidates_path)
        shutil.copy2(candidates_path, candidate_backup)
        backups.append(candidate_backup)
        candidates_path.write_text("")

    drafts = load_benchmark_drafts(drafts_path)
    if drafts:
        draft_backup = backup_path(drafts_path)
        shutil.copy2(drafts_path, draft_backup)
        backups.append(draft_backup)

    cleaned_drafts = []
    for draft in drafts:
        data = draft.model_dump()
        data.update(
            {
                "candidate_ids": [],
                "gold_prior_paper_ids": [],
                "gold_prior_dataset_names": [],
                "hard_negative_ids": [],
                "soft_negative_ids": [],
                "suggested_gold_prior_paper_ids": [],
                "suggested_hard_negative_ids": [],
                "suggested_soft_negative_ids": [],
                "gold_prior_support_acus": [],
                "gold_added_information_label": None,
                "annotation_notes": "",
                "linked_prior_papers": [],
                "draft_status": "needs_extraction",
            }
        )
        cleaned_drafts.append(BenchmarkDraftRecord(**data))

    if cleaned_drafts:
        write_jsonl(drafts_path, cleaned_drafts)
        derived_drafts = []
        for draft in load_benchmark_drafts(drafts_path):
            data = draft.model_dump()
            data["draft_status"] = summarize_query_status(draft.query_paper_id)
            derived_drafts.append(BenchmarkDraftRecord(**data))
        write_jsonl(drafts_path, derived_drafts)

    return candidate_count, len(cleaned_drafts), backups


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear all previous-work candidates and draft links/suggestions so candidates can be rebuilt cleanly.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_PREVIOUS_WORK_PATH)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_BENCHMARK_DRAFTS_PATH)
    args = parser.parse_args()

    candidate_count, draft_count, backups = clear_candidates(args.candidates, args.drafts)
    print(f"Cleared {candidate_count} previous-work candidates.")
    print(f"Reset candidate links/suggestions for {draft_count} benchmark drafts.")
    for backup in backups:
        print(f"Backup written to {backup}.")


if __name__ == "__main__":
    main()
