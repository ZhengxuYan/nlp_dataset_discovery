#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.strip() + "\n", encoding="utf-8")


def dcu_global_id(benchmark_id: str, local_id: str) -> str:
    local = str(local_id or "").strip()
    if "::" in local:
        return local
    return f"{benchmark_id}::{local}"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def index_old_labels(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for label in row.get("labels") or []:
        qid = str(label.get("query_acu_id") or "")
        for prior_id in label.get("selected_prior_acu_ids") or []:
            out[(qid, str(prior_id))] = label
    return out


def build_gold_rows(claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = 1
    for row in claim_rows:
        benchmark_id = str(row.get("benchmark_id") or "")
        old_labels = index_old_labels(row)
        query_acus = row.get("query_acus") or []
        prior_bank = []
        seen_prior_keys: set[tuple[str, str, str, str, str]] = set()
        for prior in row.get("prior_acu_bank") or []:
            prior_key = (
                str(prior.get("id") or ""),
                str(prior.get("prior_paper_id") or ""),
                str(prior.get("prior_dataset_id") or ""),
                str(prior.get("type") or ""),
                str(prior.get("text") or ""),
            )
            if prior_key in seen_prior_keys:
                continue
            seen_prior_keys.add(prior_key)
            prior_bank.append(prior)
        query_metadata = row.get("query_metadata") or {}
        for query in query_acus:
            qid = str(query.get("id") or "")
            for prior in prior_bank:
                pid = str(prior.get("id") or "")
                old_label = old_labels.get((qid, pid), {})
                out.append(
                    {
                        "annotation_id": f"gold_full_{idx:05d}",
                        "audit_type": "retrieval_gold_prior_full",
                        "benchmark_id": benchmark_id,
                        "query_paper_id": row.get("query_paper_id") or "",
                        "query_title": row.get("query_title") or "",
                        "query_dataset_name": row.get("query_dataset_name") or "",
                        "query_metadata": compact_json(query_metadata),
                        "query_dcu_id": dcu_global_id(benchmark_id, qid),
                        "query_dcu_local_id": qid,
                        "query_dcu_text": query.get("text") or "",
                        "query_dcu_type": query.get("type") or "",
                        "query_dcu_importance": query.get("importance") or "",
                        "query_dcu_evidence": query.get("evidence") or "",
                        "query_dcu_section": query.get("section") or "",
                        "prior_dcu_id": pid,
                        "prior_paper_id": prior.get("prior_paper_id") or "",
                        "prior_paper_title": prior.get("prior_paper_title") or "",
                        "prior_dataset_name": prior.get("prior_dataset_name") or "",
                        "prior_dataset_id": prior.get("prior_dataset_id") or "",
                        "prior_dcu_text": prior.get("text") or "",
                        "prior_dcu_type": prior.get("type") or "",
                        "prior_dcu_importance": prior.get("importance") or "",
                        "prior_dcu_evidence": prior.get("evidence") or "",
                        "prior_dcu_section": prior.get("section") or "",
                        "old_model_selected_as_gold": "yes" if old_label else "no",
                        "old_model_support_status": old_label.get("support_status") or "",
                        "old_model_delta_type": old_label.get("delta_type") or "",
                        "old_model_rationale": old_label.get("rationale") or "",
                        "human_gold_prior_decision": "",
                        "human_gold_prior_note": "",
                        "annotator_id": "",
                    }
                )
                idx += 1
    return out


def build_hard_negative_rows(hard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(hard_rows, start=1):
        hard = row.get("hard_negative_dcu") or {}
        out.append(
            {
                "annotation_id": f"hardneg_full_{idx:05d}",
                "audit_type": "retrieval_hard_negative_full",
                "benchmark_id": row.get("benchmark_id") or "",
                "query_title": row.get("query_title") or "",
                "query_dataset_name": row.get("query_dataset_name") or "",
                "query_dcu_id": row.get("query_dcu_id") or "",
                "query_dcu_text": row.get("query_text") or "",
                "query_dcu_type": row.get("query_type") or "",
                "query_delta_type": row.get("delta_type") or "",
                "gold_prior_dcu_ids": compact_json(row.get("gold_prior_dcu_ids") or []),
                "gold_prior_dcus": compact_json(row.get("gold_prior_dcus") or []),
                "best_gold_rank": row.get("best_gold_rank") or "",
                "hard_negative_dcu_id": row.get("hard_negative_dcu_id") or "",
                "hard_negative_rank": row.get("hard_negative_rank") or "",
                "hard_negative_paper": hard.get("paper") or "",
                "hard_negative_dataset": hard.get("dataset") or "",
                "hard_negative_type": hard.get("type") or "",
                "hard_negative_text": hard.get("text") or "",
                "hard_negative_types": compact_json(row.get("negative_types") or []),
                "negative_source": row.get("negative_source") or "",
                "ranked_before_gold": row.get("ranked_before_gold"),
                "features": compact_json(row.get("features") or {}),
                "human_hard_negative_decision": "",
                "human_hard_negative_note": "",
                "annotator_id": "",
            }
        )
    return out


CODEBOOK = """
# Retrieval Benchmark Relabeling Instructions

You are cleaning a claim-level prior evidence retrieval benchmark. You are not evaluating a retrieval model. Your job is to decide which prior DCUs should be rewarded as gold evidence, and which proposed hard negatives are valid distractors.

## Task A: Gold Prior DCU Relabeling

File: `retrieval_gold_prior_full_annotator_*.csv`

Each row contains one query DCU and one candidate prior DCU from cited prior dataset/benchmark papers. Fill:

- `human_gold_prior_decision`
- `human_gold_prior_note` when useful
- `annotator_id`

Allowed decisions:

- `covered`: the prior DCU states the same dataset contribution as the query DCU along the relevant dimension.
- `partially_covered`: the prior DCU genuinely overlaps with the query DCU, but differs in at least one important dimension such as task, domain, language, modality, source, annotation protocol, scale, release, or evaluation use.
- `not_evidence`: the prior DCU should not be rewarded as prior evidence for this query DCU.

Use high precision. If a prior DCU is only broadly related, generic, from the same paper but about a different contribution dimension, or too vague to compare, mark `not_evidence`.

The columns beginning with `old_model_` are only hints from the previous automatic labeling. Do not trust them by default.

## Task B: Hard Negative Validation

File: `retrieval_hard_negatives_full_annotator_*.csv`

Each row contains a query DCU, one gold prior DCU set, and one proposed hard negative. Fill:

- `human_hard_negative_decision`
- `human_hard_negative_note` when useful
- `annotator_id`

Allowed decisions:

- `valid_hard_negative`: related and plausible, but not actually evidence for the query DCU.
- `actually_evidence`: the supposed negative is actually `covered` or `partially_covered` evidence and should be in the gold set.
- `irrelevant_not_plausible`: too unrelated or obviously wrong to be a useful hard negative.

## Decision Tests

Gold prior: if a retrieval system returned this prior DCU for the query DCU, should we reward it? If yes, use `covered` or `partially_covered`; if no, use `not_evidence`.

Hard negative: would this be a plausible wrong retrieval result that should be ranked below gold evidence? If yes, use `valid_hard_negative`; if it should be rewarded, use `actually_evidence`; if it is too easy or unrelated, use `irrelevant_not_plausible`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full 100-paper retrieval benchmark relabeling CSVs.")
    parser.add_argument("--claim-level-jsonl", default="data/benchmark/claim_level_prior_acu_labels_pdf130.jsonl")
    parser.add_argument("--hard-negative-jsonl", default="data/benchmark/claim_level_dcu_hard_negatives_pdf130_same_gold_paper.jsonl")
    parser.add_argument("--output-dir", default="data/human_validation/retrieval_benchmark_relabeling_full")
    parser.add_argument("--annotators", nargs="+", default=["annotator_a", "annotator_b"])
    args = parser.parse_args()

    claim_rows = read_jsonl(args.claim_level_jsonl)
    hard_rows = read_jsonl(args.hard_negative_jsonl)
    gold_rows = build_gold_rows(claim_rows)
    hard_negative_rows = build_hard_negative_rows(hard_rows)

    gold_fields = list(gold_rows[0].keys()) if gold_rows else []
    hard_fields = list(hard_negative_rows[0].keys()) if hard_negative_rows else []
    output_dir = Path(args.output_dir)

    for annotator in args.annotators:
        write_csv(output_dir / f"retrieval_gold_prior_full_{annotator}.csv", gold_rows, gold_fields)
        write_csv(output_dir / f"retrieval_hard_negatives_full_{annotator}.csv", hard_negative_rows, hard_fields)

    write_markdown(output_dir / "README.md", CODEBOOK)
    summary = {
        "claim_level_jsonl": args.claim_level_jsonl,
        "hard_negative_jsonl": args.hard_negative_jsonl,
        "output_dir": str(output_dir),
        "annotators": args.annotators,
        "source_benchmark_records": len(claim_rows),
        "query_dcus": sum(len(row.get("query_acus") or []) for row in claim_rows),
        "prior_dcus": sum(len(row.get("prior_acu_bank") or []) for row in claim_rows),
        "gold_prior_annotation_rows_per_annotator": len(gold_rows),
        "hard_negative_annotation_rows_per_annotator": len(hard_negative_rows),
        "old_model_selected_gold_pairs": sum(1 for row in gold_rows if row["old_model_selected_as_gold"] == "yes"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
