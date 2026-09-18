#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VALID_GOLD = {"covered", "partially_covered"}
INVALID_GOLD = {"not_evidence"}
VALID_HARD_NEGATIVE = {"valid_hard_negative"}
INVALID_HARD_NEGATIVE = {"actually_evidence", "irrelevant_not_plausible"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def local_query_id(value: str) -> str:
    text = str(value or "").strip()
    if "::" in text:
        return text.rsplit("::", 1)[-1]
    return text


def split_validity(value: str) -> set[str]:
    text = norm(value)
    if not text:
        return set()
    parts = re.split(r"[;|,]", text)
    out = set()
    for part in parts:
        token = norm(part)
        if ":" in token:
            token = norm(token.rsplit(":", 1)[-1])
        token = token.replace(" ", "_")
        if token:
            out.add(token)
    return out


def read_resolved(path: str | Path) -> dict[str, dict[str, str]]:
    out = {}
    if not Path(path).exists():
        return out
    for row in read_csv(path):
        out[row.get("annotation_id") or ""] = row
    return out


def resolve_value(
    annotation_id: str,
    raw_value: str,
    double_value: str,
    resolved_a: dict[str, dict[str, str]],
    resolved_b: dict[str, dict[str, str]],
    field: str,
) -> tuple[str, str]:
    """Return (value, source_or_reason). Conflicts return ("", "conflict:...")."""
    if field == "gold":
        column = "adjudicated_final_label"
        fallback_a = raw_value
        fallback_b = double_value
    elif field == "hard_negative":
        column = "adjudicated_validity"
        fallback_a = raw_value
        fallback_b = double_value
    elif field == "empty":
        column = "adjudicated_final_label"
        fallback_a = raw_value
        fallback_b = double_value
    else:
        raise ValueError(field)

    resolved_values = []
    for resolved in (resolved_a, resolved_b):
        value = norm((resolved.get(annotation_id) or {}).get(column))
        if value:
            resolved_values.append(value)
    unique_resolved = sorted(set(resolved_values))
    if len(unique_resolved) == 1:
        return unique_resolved[0], "resolved"
    if len(unique_resolved) > 1:
        return "", "conflict:resolved_disagreement:" + " vs ".join(unique_resolved)

    a = norm(fallback_a)
    b = norm(fallback_b)
    if a and b and a != b:
        return "", f"conflict:annotator_disagreement:{a} vs {b}"
    return a or b, "single_or_agreed_annotation"


def audit_decisions(
    main_rows: list[dict[str, str]],
    double_rows: list[dict[str, str]],
    resolved_a: dict[str, dict[str, str]],
    resolved_b: dict[str, dict[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    double_by_id = {row.get("annotation_id") or "": row for row in double_rows}
    gold_decisions: dict[tuple[str, str], dict[str, Any]] = {}
    hard_decisions: dict[str, dict[str, Any]] = {}
    exclusions: list[dict[str, Any]] = []

    for row in main_rows:
        ann_id = row.get("annotation_id") or ""
        case = norm(row.get("benchmark_audit_case"))
        double = double_by_id.get(ann_id, {})
        benchmark_id = row.get("benchmark_id") or ""
        qid = local_query_id(row.get("query_dcu_id") or "")
        query_key = (benchmark_id, qid)

        if case in {"gold_prior", "hard_negative"}:
            value, source = resolve_value(
                ann_id,
                row.get("human_gold_prior_validity") or "",
                double.get("human_gold_prior_validity") or "",
                resolved_a,
                resolved_b,
                "gold",
            )
            if value in VALID_GOLD:
                gold_decisions[query_key] = {
                    "validity": value,
                    "annotation_id": ann_id,
                    "source": source,
                    "benchmark_audit_case": case,
                }
            else:
                exclusions.append({
                    "annotation_id": ann_id,
                    "benchmark_id": benchmark_id,
                    "query_acu_id": qid,
                    "case": case,
                    "component": "gold_prior",
                    "decision": value,
                    "reason": source if source.startswith("conflict:") else f"gold_not_valid:{value or 'blank'}",
                })

        if case == "hard_negative":
            value, source = resolve_value(
                ann_id,
                row.get("human_hard_negative_validity") or "",
                double.get("human_hard_negative_validity") or "",
                resolved_a,
                resolved_b,
                "hard_negative",
            )
            hard_id = row.get("hard_negative_dcu_id") or ""
            if value in VALID_HARD_NEGATIVE:
                hard_decisions[ann_id] = {
                    "validity": value,
                    "annotation_id": ann_id,
                    "source": source,
                    "hard_negative_dcu_id": hard_id,
                    "benchmark_id": benchmark_id,
                    "query_acu_id": qid,
                }
            else:
                exclusions.append({
                    "annotation_id": ann_id,
                    "benchmark_id": benchmark_id,
                    "query_acu_id": qid,
                    "hard_negative_dcu_id": hard_id,
                    "case": case,
                    "component": "hard_negative",
                    "decision": value,
                    "reason": source if source.startswith("conflict:") else f"hard_negative_not_valid:{value or 'blank'}",
                })

        if case == "empty_gold":
            value, source = resolve_value(
                ann_id,
                row.get("human_empty_gold_validity") or "",
                double.get("human_empty_gold_validity") or "",
                resolved_a,
                resolved_b,
                "empty",
            )
            if value != "valid_empty":
                exclusions.append({
                    "annotation_id": ann_id,
                    "benchmark_id": benchmark_id,
                    "query_acu_id": qid,
                    "case": case,
                    "component": "empty_gold",
                    "decision": value,
                    "reason": source if source.startswith("conflict:") else f"empty_not_valid:{value or 'blank'}",
                })

    return gold_decisions, hard_decisions, exclusions


def filter_claim_rows(
    claim_rows: list[dict[str, Any]],
    gold_decisions: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_rows = []
    exclusions = []
    for row in claim_rows:
        kept_labels = []
        for label in row.get("labels") or []:
            qid = str(label.get("query_acu_id") or "")
            key = (str(row.get("benchmark_id") or ""), qid)
            decision = gold_decisions.get(key)
            if decision:
                new_label = dict(label)
                new_label["evaluate"] = True
                new_label["validated_gold_decision"] = decision
                kept_labels.append(new_label)
            else:
                exclusions.append({
                    "benchmark_id": row.get("benchmark_id"),
                    "query_acu_id": qid,
                    "case": "claim_label",
                    "component": "gold_prior",
                    "reason": "not_in_validated_gold_subset",
                })
        if kept_labels:
            new_row = dict(row)
            new_row["labels"] = kept_labels
            new_row["evaluable_claim_labels"] = len(kept_labels)
            new_row["evaluable_query_acu_ids"] = [label.get("query_acu_id") for label in kept_labels]
            selected = []
            for label in kept_labels:
                selected.extend(label.get("selected_prior_acu_ids") or [])
            new_row["selected_prior_acu_ids"] = sorted(set(selected))
            kept_rows.append(new_row)
    return kept_rows, exclusions


def filter_hard_negatives(
    hard_rows: list[dict[str, Any]],
    main_audit_rows: list[dict[str, str]],
    gold_decisions: dict[tuple[str, str], dict[str, Any]],
    hard_decisions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_by_key = {}
    for row in main_audit_rows:
        if norm(row.get("benchmark_audit_case")) != "hard_negative":
            continue
        key = (
            row.get("benchmark_id") or "",
            local_query_id(row.get("query_dcu_id") or ""),
            row.get("hard_negative_dcu_id") or "",
        )
        audit_by_key[key] = row

    kept = []
    exclusions = []
    for row in hard_rows:
        key = (
            row.get("benchmark_id") or "",
            local_query_id(row.get("query_dcu_id") or ""),
            row.get("hard_negative_dcu_id") or "",
        )
        audit = audit_by_key.get(key)
        if not audit:
            exclusions.append({
                "benchmark_id": row.get("benchmark_id"),
                "query_acu_id": local_query_id(row.get("query_dcu_id") or ""),
                "hard_negative_dcu_id": row.get("hard_negative_dcu_id"),
                "case": "hard_negative",
                "component": "hard_negative",
                "reason": "not_human_audited",
            })
            continue
        ann_id = audit.get("annotation_id") or ""
        gold_key = (row.get("benchmark_id") or "", local_query_id(row.get("query_dcu_id") or ""))
        if gold_key not in gold_decisions:
            exclusions.append({
                "annotation_id": ann_id,
                "benchmark_id": row.get("benchmark_id"),
                "query_acu_id": local_query_id(row.get("query_dcu_id") or ""),
                "hard_negative_dcu_id": row.get("hard_negative_dcu_id"),
                "case": "hard_negative",
                "component": "hard_negative",
                "reason": "gold_prior_not_valid_for_hard_negative",
            })
            continue
        decision = hard_decisions.get(ann_id)
        if not decision:
            exclusions.append({
                "annotation_id": ann_id,
                "benchmark_id": row.get("benchmark_id"),
                "query_acu_id": local_query_id(row.get("query_dcu_id") or ""),
                "hard_negative_dcu_id": row.get("hard_negative_dcu_id"),
                "case": "hard_negative",
                "component": "hard_negative",
                "reason": "hard_negative_not_valid_or_conflict",
            })
            continue
        new_row = dict(row)
        new_row["validated_hard_negative_decision"] = decision
        new_row["validated_gold_decision"] = gold_decisions[gold_key]
        kept.append(new_row)
    return kept, exclusions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-validated retrieval benchmark subset.")
    parser.add_argument("--claim-level-jsonl", default="data/benchmark/claim_level_prior_acu_labels_pdf130.jsonl")
    parser.add_argument("--hard-negative-jsonl", default="data/benchmark/claim_level_dcu_hard_negatives_pdf130_same_gold_paper.jsonl")
    parser.add_argument("--main-retrieval-audit-csv", default="data/human_validation/retrieval_benchmark_label_validation_sample_labeled_jason_new.csv")
    parser.add_argument("--double-retrieval-audit-csv", default="data/human_validation/double_annotation_retrieval_label_sample_labeled_jiaxin_new.csv")
    parser.add_argument("--resolved-a-csv", default="data/human_validation/human_validation_adjudication_queue_resolved_jason.csv")
    parser.add_argument("--resolved-b-csv", default="data/human_validation/human_validation_adjudication_queue_resolved_jiaxin.csv")
    parser.add_argument("--output-claim-jsonl", default="data/benchmark/validated_claim_level_prior_dcu_labels.jsonl")
    parser.add_argument("--output-hard-negative-jsonl", default="data/benchmark/validated_claim_level_dcu_hard_negatives.jsonl")
    parser.add_argument("--output-exclusions-jsonl", default="data/benchmark/validated_retrieval_benchmark_exclusions.jsonl")
    parser.add_argument("--output-summary-json", default="data/benchmark/validated_retrieval_benchmark_summary.json")
    args = parser.parse_args()

    main_audit = read_csv(args.main_retrieval_audit_csv)
    double_audit = read_csv(args.double_retrieval_audit_csv)
    resolved_a = read_resolved(args.resolved_a_csv)
    resolved_b = read_resolved(args.resolved_b_csv)
    gold_decisions, hard_decisions, audit_exclusions = audit_decisions(main_audit, double_audit, resolved_a, resolved_b)

    claim_rows = read_jsonl(args.claim_level_jsonl)
    hard_rows = read_jsonl(args.hard_negative_jsonl)
    validated_claim_rows, claim_exclusions = filter_claim_rows(claim_rows, gold_decisions)
    validated_hard_rows, hard_exclusions = filter_hard_negatives(hard_rows, main_audit, gold_decisions, hard_decisions)
    exclusions = audit_exclusions + claim_exclusions + hard_exclusions

    write_jsonl(args.output_claim_jsonl, validated_claim_rows)
    write_jsonl(args.output_hard_negative_jsonl, validated_hard_rows)
    write_jsonl(args.output_exclusions_jsonl, exclusions)

    kept_labels = sum(len(row.get("labels") or []) for row in validated_claim_rows)
    summary = {
        "inputs": {
            "claim_level_jsonl": args.claim_level_jsonl,
            "hard_negative_jsonl": args.hard_negative_jsonl,
            "main_retrieval_audit_csv": args.main_retrieval_audit_csv,
            "double_retrieval_audit_csv": args.double_retrieval_audit_csv,
            "resolved_a_csv": args.resolved_a_csv,
            "resolved_b_csv": args.resolved_b_csv,
        },
        "validated_claim_rows": len(validated_claim_rows),
        "validated_claim_labels": kept_labels,
        "validated_hard_negative_rows": len(validated_hard_rows),
        "validated_gold_query_keys": len(gold_decisions),
        "validated_hard_negative_decisions": len(hard_decisions),
        "exclusions": len(exclusions),
        "exclusions_by_reason": dict(Counter(row.get("reason") for row in exclusions)),
        "outputs": {
            "claim_jsonl": args.output_claim_jsonl,
            "hard_negative_jsonl": args.output_hard_negative_jsonl,
            "exclusions_jsonl": args.output_exclusions_jsonl,
        },
    }
    write_json(args.output_summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
