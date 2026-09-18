#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.suffix == ".jsonl":
        rows = []
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value: Any) -> str:
    return str(value or "").strip().lower()


def is_blankish(value: Any) -> bool:
    return norm(value) in {"", "blank", "none", "n/a", "na", "[]", "no", "null"}


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def yes_rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    vals = [norm(row.get(field)) for row in rows if norm(row.get(field)) in {"yes", "no"}]
    return {"n": len(vals), "rate": rate(sum(v == "yes" for v in vals), len(vals))}


def value_rate(rows: list[dict[str, Any]], field: str, positives: set[str]) -> dict[str, Any]:
    vals = [norm(row.get(field)) for row in rows if norm(row.get(field))]
    return {"n": len(vals), "rate": rate(sum(v in positives for v in vals), len(vals)), "counts": dict(Counter(vals))}


def value_rate_excluding(
    rows: list[dict[str, Any]],
    field: str,
    positives: set[str],
    exclude: set[str],
) -> dict[str, Any]:
    vals = [norm(row.get(field)) for row in rows if norm(row.get(field)) and norm(row.get(field)) not in exclude]
    return {"n": len(vals), "rate": rate(sum(v in positives for v in vals), len(vals)), "counts": dict(Counter(vals))}


def summarize_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "n_rows": len(rows),
        "coverage_label_accuracy": yes_rate(rows, "human_coverage_label_correct"),
        "selected_evidence_useful_precision": value_rate_excluding(
            rows, "human_selected_evidence_relevance", {"direct", "partial"}, {"na"}
        ),
        "selected_evidence_direct_precision": value_rate_excluding(
            rows, "human_selected_evidence_relevance", {"direct"}, {"na"}
        ),
        "rationale_groundedness": value_rate(rows, "human_rationale_groundedness", {"yes", "partial"}),
        "human_evidence_sufficient": value_rate(rows, "human_evidence_sufficient", {"yes"}),
        "human_missing_prior_high_risk_rate": value_rate(rows, "human_missing_prior_risk", {"high"}),
    }
    adequacy_pairs = []
    risk_pairs = []
    for row in rows:
        human_adequate = norm(row.get("human_evidence_sufficient"))
        if human_adequate in {"yes", "no"}:
            model_adequate = "yes" if norm(row.get("model_evidence_adequacy")) in {"high", "medium"} else "no"
            adequacy_pairs.append((model_adequate, human_adequate))
        human_risk = norm(row.get("human_missing_prior_risk"))
        if human_risk in {"high", "low"}:
            model_risk = "high" if norm(row.get("model_missing_prior_risk")) == "high" else "low"
            risk_pairs.append((model_risk, human_risk))
    out["evidence_adequacy_accuracy"] = {
        "n": len(adequacy_pairs),
        "rate": rate(sum(a == b for a, b in adequacy_pairs), len(adequacy_pairs)),
    }
    out["missing_prior_risk_accuracy"] = {
        "n": len(risk_pairs),
        "rate": rate(sum(a == b for a, b in risk_pairs), len(risk_pairs)),
    }
    out["by_model_label"] = {}
    for label, bucket in group_by(rows, "model_coverage_label").items():
        out["by_model_label"][label] = {
            "n": len(bucket),
            "coverage_label_accuracy": yes_rate(bucket, "human_coverage_label_correct"),
        }
    return out


def summarize_extraction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "record_validity": yes_rate(rows, "human_record_valid"),
        "role_accuracy": yes_rate(rows, "human_role_correct"),
        "dcu_groundedness": yes_rate(rows, "human_dcu_grounded"),
        "dcu_type_accuracy": yes_rate(rows, "human_dcu_type_correct"),
        "metadata_error_count": sum(1 for row in rows if not is_blankish(row.get("human_metadata_errors"))),
    }


def summarize_retrieval_label(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "gold_prior_validity": value_rate(rows, "human_gold_prior_validity", {"covered", "partially_covered"}),
        "hard_negative_validity": value_rate(rows, "human_hard_negative_validity", {"valid_hard_negative"}),
        "empty_gold_validity": value_rate(rows, "human_empty_gold_validity", {"valid_empty"}),
        "by_case": {
            case: {
                "n": len(bucket),
                "gold_prior_validity": value_rate(bucket, "human_gold_prior_validity", {"covered", "partially_covered"}),
                "hard_negative_validity": value_rate(bucket, "human_hard_negative_validity", {"valid_hard_negative"}),
                "empty_gold_validity": value_rate(bucket, "human_empty_gold_validity", {"valid_empty"}),
            }
            for case, bucket in group_by(rows, "benchmark_audit_case").items()
        },
    }


def summarize_missing_prior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "n_rows": len(rows),
        "external_search_done_rate": yes_rate(rows, "human_external_search_done"),
        "missed_prior_found_rate": yes_rate(rows, "human_found_missed_prior"),
        "false_not_covered_rate": yes_rate(rows, "human_final_false_not_covered"),
        "adequacy_correct": yes_rate(rows, "human_adequacy_correct"),
        "missing_prior_risk_correct": yes_rate(rows, "human_missing_prior_risk_correct"),
    }
    out["by_model_risk"] = {}
    for risk, bucket in group_by(rows, "model_missing_prior_risk").items():
        out["by_model_risk"][risk] = {
            "n": len(bucket),
            "false_not_covered_rate": yes_rate(bucket, "human_final_false_not_covered"),
        }
    return out


def group_by(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[norm(row.get(field)) or "unknown"].append(row)
    return dict(grouped)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_audit = group_by(rows, "audit_type")
    out = {"n_rows": len(rows), "by_audit": {}}
    for audit, bucket in by_audit.items():
        if audit == "attribution":
            out["by_audit"][audit] = summarize_attribution(bucket)
        elif audit == "extraction":
            out["by_audit"][audit] = summarize_extraction(bucket)
        elif audit == "retrieval_label":
            out["by_audit"][audit] = summarize_retrieval_label(bucket)
        elif audit == "missing_prior":
            out["by_audit"][audit] = summarize_missing_prior(bucket)
        else:
            out["by_audit"][audit] = {"n_rows": len(bucket)}
    return out


def agreement(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    by_id_a = {row.get("annotation_id"): row for row in rows_a}
    by_id_b = {row.get("annotation_id"): row for row in rows_b}
    shared = sorted(set(by_id_a) & set(by_id_b))
    out = {"shared_annotation_ids": len(shared), "fields": {}}
    for field in fields:
        pairs = []
        for ann_id in shared:
            a = norm(by_id_a[ann_id].get(field))
            b = norm(by_id_b[ann_id].get(field))
            if a and b:
                pairs.append((a, b))
        out["fields"][field] = {
            "n": len(pairs),
            "percent_agreement": rate(sum(a == b for a, b in pairs), len(pairs)),
            "disagreements": [
                {"annotation_id": ann_id, "a": norm(by_id_a[ann_id].get(field)), "b": norm(by_id_b[ann_id].get(field))}
                for ann_id in shared
                if norm(by_id_a[ann_id].get(field))
                and norm(by_id_b[ann_id].get(field))
                and norm(by_id_a[ann_id].get(field)) != norm(by_id_b[ann_id].get(field))
            ][:25],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed human validation annotations.")
    parser.add_argument("paths", nargs="+", help="Annotated CSV or JSONL files.")
    parser.add_argument("--annotator-b", nargs="*", default=[], help="Optional second annotator files for agreement.")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    rows_a = []
    for path in args.paths:
        rows_a.extend(read_rows(path))
    report = {"summary": summarize(rows_a)}
    if args.annotator_b:
        rows_b = []
        for path in args.annotator_b:
            rows_b.extend(read_rows(path))
        report["agreement"] = agreement(
            rows_a,
            rows_b,
            [
                "human_coverage_label_correct",
                "human_selected_evidence_relevance",
                "human_rationale_groundedness",
                "human_evidence_sufficient",
                "human_missing_prior_risk",
                "human_dcu_grounded",
                "human_dcu_type_correct",
                "human_gold_prior_validity",
                "human_hard_negative_validity",
                "human_empty_gold_validity",
            ],
        )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
