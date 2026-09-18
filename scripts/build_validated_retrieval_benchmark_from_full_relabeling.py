#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_GOLD = {"covered", "partially_covered"}
VALID_HARD_NEGATIVE = {"valid_hard_negative"}


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


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def stable_acu_key(paper_id: str, dataset_id: str, acu_type: str, text: str) -> str:
    payload = "\t".join([clean_text(paper_id), clean_text(dataset_id), clean_text(acu_type), clean_text(text)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def query_local_id(query_dcu_id: str) -> str:
    text = str(query_dcu_id or "").strip()
    return text.rsplit("::", 1)[-1] if "::" in text else text


def final_decisions(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    rows_resolved: list[dict[str, str]],
    *,
    field: str,
    resolved_field: str,
) -> dict[str, tuple[str, dict[str, str], str]]:
    by_a = {row.get("annotation_id") or "": row for row in rows_a}
    by_b = {row.get("annotation_id") or "": row for row in rows_b}
    by_resolved = {row.get("annotation_id") or "": row for row in rows_resolved}
    out: dict[str, tuple[str, dict[str, str], str]] = {}
    for annotation_id in sorted(set(by_a) | set(by_b) | set(by_resolved)):
        if annotation_id in by_resolved and norm(by_resolved[annotation_id].get(resolved_field)):
            row = by_a.get(annotation_id) or by_b.get(annotation_id) or by_resolved[annotation_id]
            out[annotation_id] = (norm(by_resolved[annotation_id][resolved_field]), row, "adjudicated")
            continue
        value_a = norm((by_a.get(annotation_id) or {}).get(field))
        value_b = norm((by_b.get(annotation_id) or {}).get(field))
        row = by_a.get(annotation_id) or by_b.get(annotation_id)
        if value_a and value_b and value_a == value_b:
            out[annotation_id] = (value_a, row, "agreement")
        elif value_a and not value_b:
            out[annotation_id] = (value_a, row, "single_a")
        elif value_b and not value_a:
            out[annotation_id] = (value_b, row, "single_b")
        elif row is not None:
            out[annotation_id] = ("", row, "unresolved")
    return out


def local_prior_summary(prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": stable_acu_key(
            str(prior.get("prior_paper_id") or ""),
            str(prior.get("prior_dataset_id") or ""),
            str(prior.get("type") or ""),
            str(prior.get("text") or ""),
        ),
        "paper": prior.get("prior_paper_title") or "",
        "dataset": prior.get("prior_dataset_name") or "",
        "type": prior.get("type") or "",
        "text": prior.get("text") or "",
    }


def build_gold_index(gold_decisions: dict[str, tuple[str, dict[str, str], str]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for annotation_id, (decision, row, source) in gold_decisions.items():
        if decision not in VALID_GOLD:
            continue
        key = (row["benchmark_id"], row["query_dcu_local_id"])
        by_query[key].append({
            "annotation_id": annotation_id,
            "decision": decision,
            "source": source,
            "prior_dcu_id": row["prior_dcu_id"],
        })
    return by_query


def build_validated_claim_rows(
    claim_rows: list[dict[str, Any]],
    gold_by_query: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out = []
    exclusions = []
    for row in claim_rows:
        benchmark_id = str(row.get("benchmark_id") or "")
        validated_labels = []
        query_by_id = row.get("query_acu_by_id") or {acu.get("id"): acu for acu in row.get("query_acus") or []}
        prior_by_id = {acu.get("id"): acu for acu in row.get("prior_acu_bank") or []}
        for query_id in sorted(query_by_id):
            gold_items = gold_by_query.get((benchmark_id, str(query_id)), [])
            if not gold_items:
                exclusions.append({
                    "benchmark_id": benchmark_id,
                    "query_acu_id": query_id,
                    "component": "gold_prior",
                    "reason": "no_valid_gold_after_full_relabel",
                })
                continue
            selected = []
            decisions = []
            sources = []
            for item in gold_items:
                local_id = item["prior_dcu_id"]
                if local_id not in prior_by_id:
                    exclusions.append({
                        "benchmark_id": benchmark_id,
                        "query_acu_id": query_id,
                        "prior_dcu_id": local_id,
                        "component": "gold_prior",
                        "reason": "validated_prior_id_missing_from_original_bank",
                    })
                    continue
                selected.append(local_id)
                decisions.append(item["decision"])
                sources.append({
                    "annotation_id": item["annotation_id"],
                    "decision": item["decision"],
                    "source": item["source"],
                    "prior_dcu_id": local_id,
                })
            if not selected:
                continue
            support_status = "supported" if any(decision == "covered" for decision in decisions) else "partially_supported"
            validated_labels.append({
                "query_acu_id": query_id,
                "support_status": support_status,
                "selected_prior_acu_ids": sorted(set(selected)),
                "delta_type": query_by_id[query_id].get("type") or "",
                "rationale": "Human-validated full relabeling gold prior DCU set.",
                "evaluate": True,
                "validated_full_relabel_decisions": sources,
            })
        if validated_labels:
            new_row = dict(row)
            new_row["labels"] = validated_labels
            new_row["evaluable_claim_labels"] = validated_labels
            new_row["evaluable_query_acu_ids"] = [label["query_acu_id"] for label in validated_labels]
            selected_prior_ids = []
            for label in validated_labels:
                selected_prior_ids.extend(label["selected_prior_acu_ids"])
            new_row["selected_prior_acu_ids"] = sorted(set(selected_prior_ids))
            out.append(new_row)
    return out, exclusions


def build_validated_hard_negatives(
    hard_rows: list[dict[str, Any]],
    hard_decisions: dict[str, tuple[str, dict[str, str], str]],
    gold_by_query: dict[tuple[str, str], list[dict[str, Any]]],
    claim_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard_decision_by_id = hard_decisions
    original_hard_by_id = {f"hardneg_full_{idx:05d}": row for idx, row in enumerate(hard_rows, start=1)}
    claim_by_benchmark = {str(row.get("benchmark_id") or ""): row for row in claim_rows}
    out = []
    exclusions = []
    for annotation_id, original in original_hard_by_id.items():
        decision, _, source = hard_decision_by_id.get(annotation_id, ("", {}, "missing_decision"))
        if decision != "valid_hard_negative":
            exclusions.append({
                "annotation_id": annotation_id,
                "benchmark_id": original.get("benchmark_id"),
                "query_dcu_id": original.get("query_dcu_id"),
                "hard_negative_dcu_id": original.get("hard_negative_dcu_id"),
                "component": "hard_negative",
                "decision": decision,
                "reason": f"hard_negative_not_valid:{decision or source}",
            })
            continue
        benchmark_id = str(original.get("benchmark_id") or "")
        local_qid = query_local_id(str(original.get("query_dcu_id") or ""))
        gold_items = gold_by_query.get((benchmark_id, local_qid), [])
        if not gold_items:
            exclusions.append({
                "annotation_id": annotation_id,
                "benchmark_id": benchmark_id,
                "query_dcu_id": original.get("query_dcu_id"),
                "hard_negative_dcu_id": original.get("hard_negative_dcu_id"),
                "component": "hard_negative",
                "decision": decision,
                "reason": "valid_hard_negative_query_has_no_valid_gold",
            })
            continue
        claim_row = claim_by_benchmark.get(benchmark_id)
        prior_by_id = {acu.get("id"): acu for acu in (claim_row or {}).get("prior_acu_bank") or []}
        gold_global_ids = []
        gold_summaries = []
        for item in gold_items:
            prior = prior_by_id.get(item["prior_dcu_id"])
            if not prior:
                continue
            summary = local_prior_summary(prior)
            gold_global_ids.append(summary["id"])
            gold_summaries.append(summary)
        if not gold_global_ids:
            exclusions.append({
                "annotation_id": annotation_id,
                "benchmark_id": benchmark_id,
                "query_dcu_id": original.get("query_dcu_id"),
                "hard_negative_dcu_id": original.get("hard_negative_dcu_id"),
                "component": "hard_negative",
                "decision": decision,
                "reason": "valid_gold_ids_missing_for_hard_negative_query",
            })
            continue
        new_row = dict(original)
        new_row["gold_prior_dcu_ids"] = sorted(set(gold_global_ids))
        new_row["gold_prior_dcus"] = gold_summaries
        new_row["validated_hard_negative_decision"] = {
            "annotation_id": annotation_id,
            "decision": decision,
            "source": source,
        }
        out.append(new_row)
    return out, exclusions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated benchmark from the full 100-paper relabeling results.")
    parser.add_argument("--claim-level-jsonl", default="data/benchmark/claim_level_prior_acu_labels_pdf130.jsonl")
    parser.add_argument("--hard-negative-jsonl", default="data/benchmark/claim_level_dcu_hard_negatives_pdf130_same_gold_paper.jsonl")
    parser.add_argument("--relabel-dir", default="data/human_validation/retrieval_benchmark_relabeling_full")
    parser.add_argument("--output-claim-jsonl", default="data/benchmark/validated_full_relabel_claim_level_prior_dcu_labels.jsonl")
    parser.add_argument("--output-hard-negative-jsonl", default="data/benchmark/validated_full_relabel_dcu_hard_negatives.jsonl")
    parser.add_argument("--output-exclusions-jsonl", default="data/benchmark/validated_full_relabel_exclusions.jsonl")
    parser.add_argument("--output-summary-json", default="data/benchmark/validated_full_relabel_summary.json")
    args = parser.parse_args()

    relabel_dir = Path(args.relabel_dir)
    adj_dir = relabel_dir / "adjudication_queues"
    gold_decisions = final_decisions(
        read_csv(relabel_dir / "retrieval_gold_prior_full_annotator_a_labeled_jason.csv"),
        read_csv(relabel_dir / "retrieval_gold_prior_full_annotator_b_labeled_jiaxin.csv"),
        read_csv(adj_dir / "gold_prior_disagreement_adjudication_queue_resolved_resolved.csv"),
        field="human_gold_prior_decision",
        resolved_field="adjudicated_gold_prior_decision",
    )
    hard_decisions = final_decisions(
        read_csv(relabel_dir / "retrieval_hard_negatives_full_annotator_a_labeled_jason.csv"),
        read_csv(relabel_dir / "retrieval_hard_negatives_full_annotator_b_labeled_jiaxin.csv"),
        read_csv(adj_dir / "hard_negative_disagreement_adjudication_queue_resolved_resolved.csv"),
        field="human_hard_negative_decision",
        resolved_field="adjudicated_hard_negative_decision",
    )

    claim_rows = read_jsonl(args.claim_level_jsonl)
    hard_rows = read_jsonl(args.hard_negative_jsonl)
    gold_by_query = build_gold_index(gold_decisions)
    validated_claim_rows, claim_exclusions = build_validated_claim_rows(claim_rows, gold_by_query)
    validated_hard_rows, hard_exclusions = build_validated_hard_negatives(hard_rows, hard_decisions, gold_by_query, claim_rows)
    exclusions = claim_exclusions + hard_exclusions

    write_jsonl(args.output_claim_jsonl, validated_claim_rows)
    write_jsonl(args.output_hard_negative_jsonl, validated_hard_rows)
    write_jsonl(args.output_exclusions_jsonl, exclusions)

    gold_counts = Counter(decision for decision, _, _ in gold_decisions.values())
    hard_counts = Counter(decision for decision, _, _ in hard_decisions.values())
    validated_label_count = sum(len(row.get("labels") or []) for row in validated_claim_rows)
    validated_gold_pair_count = sum(len(label.get("selected_prior_acu_ids") or []) for row in validated_claim_rows for label in row.get("labels") or [])
    summary = {
        "inputs": {
            "claim_level_jsonl": args.claim_level_jsonl,
            "hard_negative_jsonl": args.hard_negative_jsonl,
            "relabel_dir": args.relabel_dir,
        },
        "full_relabel_gold_decision_counts": dict(gold_counts),
        "full_relabel_hard_negative_decision_counts": dict(hard_counts),
        "validated_claim_rows": len(validated_claim_rows),
        "validated_query_labels": validated_label_count,
        "validated_gold_prior_pairs": validated_gold_pair_count,
        "validated_hard_negative_rows": len(validated_hard_rows),
        "validated_hard_negative_query_count": len({row.get("query_dcu_id") for row in validated_hard_rows}),
        "validated_hard_negative_record_count": len({row.get("benchmark_id") for row in validated_hard_rows}),
        "exclusions": len(exclusions),
        "exclusions_by_reason": dict(Counter(row.get("reason") for row in exclusions)),
        "outputs": {
            "claim_jsonl": args.output_claim_jsonl,
            "hard_negative_jsonl": args.output_hard_negative_jsonl,
            "exclusions_jsonl": args.output_exclusions_jsonl,
        },
        "notes": [
            "The 19 hard-negative actually_evidence vs gold not_evidence cross-conflicts are excluded rather than converted into gold evidence.",
            "Hard-negative rows are retained only when the negative is valid and the query has at least one final validated gold prior DCU.",
        ],
    }
    write_json(args.output_summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
