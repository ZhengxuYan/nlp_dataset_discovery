#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100 * value:.1f}"


def tex_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100 * value:.1f}\\%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready human validation tables.")
    parser.add_argument("--validation-summary-json", default="data/human_validation/human_validation_summary_updated.json")
    parser.add_argument("--validated-benchmark-summary-json", default="data/benchmark/validated_retrieval_benchmark_summary.json")
    parser.add_argument("--output-tex", default="data/human_validation/paper_human_validation_table.tex")
    parser.add_argument("--output-md", default="data/human_validation/paper_human_validation_summary.md")
    parser.add_argument("--output-json", default="data/human_validation/paper_human_validation_summary.json")
    args = parser.parse_args()

    validation = read_json(args.validation_summary_json)
    benchmark = read_json(args.validated_benchmark_summary_json)
    audits = validation["summary"]["by_audit"]
    attribution = audits["attribution"]
    extraction = audits["extraction"]
    missing = audits["missing_prior"]

    rows = [
        {
            "component": "DCU construction",
            "metric": "DCU groundedness",
            "n": extraction["dcu_groundedness"]["n"],
            "value": extraction["dcu_groundedness"]["rate"],
            "interpretation": "strong",
        },
        {
            "component": "DCU construction",
            "metric": "DCU type accuracy",
            "n": extraction["dcu_type_accuracy"]["n"],
            "value": extraction["dcu_type_accuracy"]["rate"],
            "interpretation": "strong",
        },
        {
            "component": "Attribution",
            "metric": "coverage-label accuracy",
            "n": attribution["coverage_label_accuracy"]["n"],
            "value": attribution["coverage_label_accuracy"]["rate"],
            "interpretation": "moderate",
        },
        {
            "component": "Attribution",
            "metric": "selected-evidence useful precision",
            "n": attribution["selected_evidence_useful_precision"]["n"],
            "value": attribution["selected_evidence_useful_precision"]["rate"],
            "interpretation": "strong",
        },
        {
            "component": "Attribution",
            "metric": "rationale groundedness",
            "n": attribution["rationale_groundedness"]["n"],
            "value": attribution["rationale_groundedness"]["rate"],
            "interpretation": "strong",
        },
        {
            "component": "External prior audit",
            "metric": "false-not-covered rate",
            "n": missing["false_not_covered_rate"]["n"],
            "value": missing["false_not_covered_rate"]["rate"],
            "interpretation": "limitation",
        },
    ]
    summary = {
        "rows": rows,
        "validated_retrieval_benchmark": {
            "validated_claim_rows": benchmark["validated_claim_rows"],
            "validated_claim_labels": benchmark["validated_claim_labels"],
            "validated_hard_negative_rows": benchmark["validated_hard_negative_rows"],
            "exclusions": benchmark["exclusions"],
            "exclusions_by_reason": benchmark["exclusions_by_reason"],
        },
        "notes": [
            "Use DCU construction and selected-evidence/rationale metrics as the strongest validation evidence.",
            "Treat coverage-label accuracy as moderate.",
            "Treat false-not-covered as evidence of retrieval incompleteness, not as a success metric.",
            "Validated hard-negative N is too small for main-table claims.",
        ],
    }

    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Component & Metric & N & Value \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['component']} & {row['metric']} & {row['n']} & {tex_pct(row['value'])} \\\\"
        )
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Human validation summary. Coverage-label accuracy measures exact label correctness and is treated as moderate reliability. False-not-covered rate is measured by external prior audit and indicates retrieval incompleteness.}",
        "\\label{tab:human_audit}",
        "\\end{table}",
        "",
    ])

    md_lines = [
        "# Paper Human Validation Summary",
        "",
        "| Component | Metric | N | Value | Interpretation |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['component']} | {row['metric']} | {row['n']} | {pct(row['value'])}% | {row['interpretation']} |"
        )
    md_lines.extend([
        "",
        "## Validated Retrieval Benchmark",
        "",
        f"- Validated claim rows: {benchmark['validated_claim_rows']}",
        f"- Validated claim labels: {benchmark['validated_claim_labels']}",
        f"- Validated hard-negative rows: {benchmark['validated_hard_negative_rows']}",
        f"- Exclusions logged: {benchmark['exclusions']}",
        "",
        "The validated hard-negative subset is too small for a main-table hard-negative rejection claim.",
    ])

    Path(args.output_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_tex).write_text("\n".join(tex_lines), encoding="utf-8")
    Path(args.output_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_tex": args.output_tex,
        "output_md": args.output_md,
        "output_json": args.output_json,
        "validated_claim_labels": benchmark["validated_claim_labels"],
        "validated_hard_negative_rows": benchmark["validated_hard_negative_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
