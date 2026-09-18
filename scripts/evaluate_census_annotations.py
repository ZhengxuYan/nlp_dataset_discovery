#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def binary_metrics(pred: list[bool], gold: list[bool]) -> dict[str, Any]:
    assert len(pred) == len(gold)
    tp = sum(p and g for p, g in zip(pred, gold))
    tn = sum((not p) and (not g) for p, g in zip(pred, gold))
    fp = sum(p and (not g) for p, g in zip(pred, gold))
    fn = sum((not p) and g for p, g in zip(pred, gold))
    n = len(pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    neg_precision = tn / (tn + fn) if tn + fn else 0.0
    neg_recall = tn / (tn + fp) if tn + fp else 0.0
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if neg_precision + neg_recall else 0.0
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision_positive": precision,
        "recall_positive": recall,
        "f1_positive": f1,
        "macro_f1": (f1 + neg_f1) / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if not n:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    expected = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    return (observed - expected) / (1 - expected) if expected != 1 else 1.0


def align_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id = {}
    for row in rows:
        paper_id = row.get("paper_id")
        if paper_id:
            by_id[paper_id] = row
    return by_id


def evaluate_annotator(rows: list[dict[str, Any]], annotator_name: str) -> dict[str, Any]:
    usable = [
        row for row in rows
        if as_bool(row.get("gold_is_dataset_introducing")) is not None
    ]
    pred = [bool(row.get("pred_is_dataset_introducing")) for row in usable]
    gold = [bool(as_bool(row.get("gold_is_dataset_introducing"))) for row in usable]
    metrics = binary_metrics(pred, gold)
    metrics["annotator"] = annotator_name
    metrics["gold_positive"] = sum(gold)
    metrics["pred_positive"] = sum(pred)
    return metrics


def disagreement_row(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": row_a.get("paper_id"),
        "title": row_a.get("title"),
        "year": row_a.get("year"),
        "venue_prefix": row_a.get("venue_prefix"),
        "sample_bucket": row_a.get("sample_bucket"),
        "pred_is_dataset_introducing": row_a.get("pred_is_dataset_introducing"),
        "jason_gold": row_a.get("gold_is_dataset_introducing"),
        "jason_dataset_names": row_a.get("gold_dataset_names"),
        "jason_notes": row_a.get("notes"),
        "jiaxin_gold": row_b.get("gold_is_dataset_introducing"),
        "jiaxin_dataset_names": row_b.get("gold_dataset_names"),
        "jiaxin_notes": row_b.get("notes"),
        "pred_datasets": row_a.get("pred_datasets"),
        "pred_exclusion_reason": row_a.get("pred_exclusion_reason"),
        "abstract": row_a.get("abstract"),
        "url": row_a.get("url"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dataset census human annotations.")
    parser.add_argument("--jason-jsonl", required=True)
    parser.add_argument("--jiaxin-jsonl", required=True)
    parser.add_argument("--output-json", default="data/census/acl_validation_annotation_eval.json")
    parser.add_argument("--disagreements-jsonl", default="data/census/acl_validation_annotation_disagreements.jsonl")
    args = parser.parse_args()

    jason_rows = read_jsonl(args.jason_jsonl)
    jiaxin_rows = read_jsonl(args.jiaxin_jsonl)
    jason_by_id = align_rows(args.jason_jsonl)
    jiaxin_by_id = align_rows(args.jiaxin_jsonl)
    common_ids = sorted(set(jason_by_id) & set(jiaxin_by_id))
    common_pairs = [
        (jason_by_id[paper_id], jiaxin_by_id[paper_id])
        for paper_id in common_ids
        if as_bool(jason_by_id[paper_id].get("gold_is_dataset_introducing")) is not None
        and as_bool(jiaxin_by_id[paper_id].get("gold_is_dataset_introducing")) is not None
    ]
    jason_gold = [bool(as_bool(a.get("gold_is_dataset_introducing"))) for a, _ in common_pairs]
    jiaxin_gold = [bool(as_bool(b.get("gold_is_dataset_introducing"))) for _, b in common_pairs]
    pred = [bool(a.get("pred_is_dataset_introducing")) for a, _ in common_pairs]
    disagreements = [
        disagreement_row(a, b)
        for a, b in common_pairs
        if as_bool(a.get("gold_is_dataset_introducing")) != as_bool(b.get("gold_is_dataset_introducing"))
    ]
    model_vs_consensus_rows = []
    consensus_gold = []
    consensus_pred = []
    for a, b in common_pairs:
        ag = as_bool(a.get("gold_is_dataset_introducing"))
        bg = as_bool(b.get("gold_is_dataset_introducing"))
        if ag == bg:
            model_vs_consensus_rows.append(a)
            consensus_gold.append(bool(ag))
            consensus_pred.append(bool(a.get("pred_is_dataset_introducing")))
    report = {
        "row_counts": {
            "jason_rows": len(jason_rows),
            "jiaxin_rows": len(jiaxin_rows),
            "common_rows": len(common_ids),
            "common_labeled_rows": len(common_pairs),
            "annotator_disagreements": len(disagreements),
            "consensus_rows": len(model_vs_consensus_rows),
        },
        "model_vs_jason": evaluate_annotator(jason_rows, "jason"),
        "model_vs_jiaxin": evaluate_annotator(jiaxin_rows, "jiaxin"),
        "jason_vs_jiaxin": {
            **binary_metrics(jason_gold, jiaxin_gold),
            "cohen_kappa": cohen_kappa(jason_gold, jiaxin_gold),
            "jason_positive": sum(jason_gold),
            "jiaxin_positive": sum(jiaxin_gold),
        },
        "model_vs_consensus_agreed_labels": binary_metrics(consensus_pred, consensus_gold),
        "disagreement_by_bucket": dict(Counter(row.get("sample_bucket") for row in disagreements)),
    }
    write_json(args.output_json, report)
    write_jsonl(args.disagreements_jsonl, disagreements)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
