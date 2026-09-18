#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "annotation_id",
    "audit_type",
    "source_sample",
    "query_bank_id",
    "query_paper_id",
    "query_title",
    "query_year",
    "source_corpus",
    "query_dataset_id",
    "query_dataset_name",
    "query_dcu_id",
    "query_dcu_text",
    "query_dcu_type",
    "query_dcu_importance",
    "query_dcu_evidence",
    "query_dcu_section",
    "llm_label",
    "llm_raw_support_status",
    "llm_delta_type",
    "llm_evidence_adequacy",
    "llm_missing_prior_risk",
    "llm_selected_prior_dcu_ids",
    "llm_rationale",
    "prior_evidence_for_review",
    "record_analysis_inclusion",
    "record_exclusion_reason",
    "human_agrees_with_llm_label",
    "human_corrected_label",
    "human_label_note",
    "annotator_id",
]


PROMPT = """
# Attribution Label Agreement Audit

你现在要审核 LLM 对 dataset contribution claim 的 attribution label。每一行会给你一个 query DCU、LLM label、LLM selected prior evidence、LLM rationale，以及系统展示给 LLM 的 retrieved prior evidence。

你的任务不是重新检索 prior work，也不是评价整篇 paper。请只根据这一行展示的信息判断：

> 你是否同意 LLM 给出的 label？

## 要填写的列

- `human_agrees_with_llm_label`
  - `agree`
  - `disagree`
  - `uncertain`
- `human_corrected_label`
  - 如果 agree，可以留空；
  - 如果 disagree，请填：
    - `covered`
    - `partially_covered`
    - `not_covered`
    - `not_comparable`
    - `contradicted`
- `human_label_note`
  - 简短说明即可。
- `annotator_id`

## Label 含义

`covered`: retrieved prior evidence 已经覆盖 query DCU 的主要 dataset contribution。

`partially_covered`: retrieved prior evidence 和 query DCU 有真实重叠，但 query DCU 仍有重要差异，例如 task、domain、language、modality、source、annotation protocol、scale、release setting 或 evaluation use 不同。

`not_covered`: 在展示的 retrieved evidence 中，没有 prior DCU 覆盖或部分覆盖 query DCU。

`not_comparable`: query DCU 和 prior evidence 太 vague、结构不同，或不是同一类 contribution，无法可靠比较。

`contradicted`: retrieved prior evidence 和 query DCU 明确冲突。

## Decision Rules

选 `agree` 如果 LLM label 和展示的 evidence 关系基本正确。即使 rationale 写得不完美，只要 label 合理，也可以 agree。

选 `disagree` 如果 LLM 把只是 topic-related 的 prior evidence 当成 support，或者把真正覆盖 query 的 evidence 标成 not covered，或者 covered / partially_covered / not_covered 的边界明显错了。

选 `uncertain` 如果 query DCU 或 prior evidence 太 vague，展示信息不足，或者 label 边界确实模糊。

不要因为 prior paper 看起来相关就同意 label。判断单位是 query DCU vs prior DCU。

不要因为 LLM rationale 写得流畅就同意 label。主要判断 label 是否正确。

如果没有足够 evidence，请不要猜测外部 prior work。本任务只评估：

> Given the displayed evidence, is the LLM label correct?
"""


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_label(value: str) -> str:
    text = str(value or "").strip().lower()
    return {
        "supported": "covered",
        "partially_supported": "partially_covered",
        "unsupported": "not_covered",
    }.get(text, text)


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("query_bank_id", ""), row.get("query_dcu_id", ""), row.get("model_coverage_label", ""))


def convert_rows(paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        source = Path(path).stem
        for row in read_csv(path):
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "annotation_id": f"label_agree_{len(out) + 1:04d}",
                "audit_type": "attribution_label_agreement",
                "source_sample": source,
                "query_bank_id": row.get("query_bank_id", ""),
                "query_paper_id": row.get("query_paper_id", ""),
                "query_title": row.get("query_title", ""),
                "query_year": row.get("query_year", ""),
                "source_corpus": row.get("source_corpus", ""),
                "query_dataset_id": row.get("query_dataset_id", ""),
                "query_dataset_name": row.get("query_dataset_name", ""),
                "query_dcu_id": row.get("query_dcu_id", ""),
                "query_dcu_text": row.get("query_dcu_text", ""),
                "query_dcu_type": row.get("query_dcu_type", ""),
                "query_dcu_importance": row.get("query_dcu_importance", ""),
                "query_dcu_evidence": row.get("query_dcu_evidence", ""),
                "query_dcu_section": row.get("query_dcu_section", ""),
                "llm_label": norm_label(row.get("model_coverage_label", "")),
                "llm_raw_support_status": row.get("model_raw_support_status", ""),
                "llm_delta_type": row.get("model_delta_type", ""),
                "llm_evidence_adequacy": row.get("model_evidence_adequacy", ""),
                "llm_missing_prior_risk": row.get("model_missing_prior_risk", ""),
                "llm_selected_prior_dcu_ids": row.get("selected_prior_dcu_ids", ""),
                "llm_rationale": row.get("model_rationale", ""),
                "prior_evidence_for_review": row.get("prior_evidence_for_review", ""),
                "record_analysis_inclusion": row.get("record_analysis_inclusion", ""),
                "record_exclusion_reason": row.get("record_exclusion_reason", ""),
                "human_agrees_with_llm_label": "",
                "human_corrected_label": "",
                "human_label_note": "",
                "annotator_id": "",
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare agree/disagree attribution label audit packets.")
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        default=[
            "data/human_validation/attribution_natural_sample.csv",
            "data/human_validation/attribution_stratified_challenge_sample.csv",
        ],
    )
    parser.add_argument("--output-dir", default="data/human_validation/attribution_label_agreement_audit")
    parser.add_argument("--annotators", nargs="+", default=["annotator_a", "annotator_b"])
    args = parser.parse_args()

    rows = convert_rows(args.input_csvs)
    output_dir = Path(args.output_dir)
    for annotator in args.annotators:
        write_csv(output_dir / f"attribution_label_agreement_{annotator}.csv", rows, OUTPUT_FIELDS)
    (output_dir / "ANNOTATOR_PROMPT.md").write_text(PROMPT.strip() + "\n", encoding="utf-8")
    summary = {
        "input_csvs": args.input_csvs,
        "output_dir": args.output_dir,
        "annotators": args.annotators,
        "rows_per_annotator": len(rows),
        "label_counts": {},
    }
    for row in rows:
        summary["label_counts"][row["llm_label"]] = summary["label_counts"].get(row["llm_label"], 0) + 1
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
