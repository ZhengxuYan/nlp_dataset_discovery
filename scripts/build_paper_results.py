#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_RETRIEVAL_REPORTS = [
    "data/benchmark/retrieval_cache/latest_tier1_report.md",
    "data/benchmark/retrieval_cache/gpt_rerank_report.md",
]
DEFAULT_OUTPUT_DIR = "artifacts/paper_results"

METHOD_NAMES = {
    "bm25": "BM25",
    "lexical": "BM25",
    "minilm_dense": "MiniLM dense",
    "dense": "MiniLM dense",
    "minilm_fusion": "Dense + BM25 fusion",
    "fusion": "Dense + BM25 fusion",
    "gpt_5_4_listwise_rerank": "GPT-5.4 rerank",
    "gpt_5_4_oracle_tournament": "GPT-5.4 oracle rerank",
}
METHOD_ORDER = [
    "bm25",
    "lexical",
    "minilm_dense",
    "dense",
    "minilm_fusion",
    "fusion",
    "gpt_5_4_listwise_rerank",
    "gpt_5_4_oracle_tournament",
]
SUPPORT_ORDER = ["supported", "partially_supported", "unsupported", "contradicted", "not_comparable"]


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_retrieval_markdown(paths: Sequence[str]) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("|") or line.startswith("| ---") or "Method" in line:
                continue
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) < 6:
                continue
            method = parts[0]
            try:
                metrics[method] = {
                    "mrr": float(parts[1]),
                    "recall@1": float(parts[2]),
                    "recall@3": float(parts[3]),
                    "recall@5": float(parts[4]),
                    "recall@10": float(parts[5]),
                }
            except ValueError:
                continue
    return metrics


def ordered_methods(metrics: dict[str, dict[str, float]]) -> list[str]:
    seen = set()
    ordered = []
    for method in METHOD_ORDER:
        if method in metrics and method not in seen:
            ordered.append(method)
            seen.add(method)
    for method in sorted(metrics):
        if method not in seen:
            ordered.append(method)
    return ordered


def latex_table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, alignment: str) -> str:
    body = [
        "\\begin{tabular}{" + alignment + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    body.extend(" & ".join(row) + " \\\\" for row in rows)
    body.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(body)


def build_retrieval_table(metrics: dict[str, dict[str, float]]) -> str:
    rows = []
    for method in ordered_methods(metrics):
        row = metrics[method]
        rows.append([
            METHOD_NAMES.get(method, method),
            f"{row.get('mrr', 0.0):.3f}",
            f"{row.get('recall@1', 0.0):.3f}",
            f"{row.get('recall@3', 0.0):.3f}",
            f"{row.get('recall@5', 0.0):.3f}",
            f"{row.get('recall@10', 0.0):.3f}",
        ])
    return latex_table(["Method", "MRR", "R@1", "R@3", "R@5", "R@10"], rows, alignment="lccccc")


def summarize_attribution_rows(rows: Sequence[dict]) -> dict[str, dict]:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row.get("condition", "unknown")].append(row)

    summaries = {}
    for condition, condition_rows in by_condition.items():
        support_percentages: dict[str, list[float]] = defaultdict(list)
        unsupported_by_delta_type: Counter = Counter()
        for row in condition_rows:
            profile = row.get("profile") or {}
            for status in SUPPORT_ORDER:
                support_percentages[status].append(float((profile.get("support_percentages") or {}).get(status, 0.0)))
            unsupported_by_delta_type.update(profile.get("unsupported_by_delta_type") or {})
        summaries[condition] = {
            "n": len(condition_rows),
            "mean_added_information_score": mean([
                float((row.get("profile") or {}).get("added_information_score", 0.0))
                for row in condition_rows
            ]),
            "mean_support_percentages": {
                status: mean(values)
                for status, values in support_percentages.items()
            },
            "unsupported_by_delta_type": dict(unsupported_by_delta_type),
        }
    return summaries


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def build_attribution_table(summaries: dict[str, dict]) -> str:
    condition_order = ["oracle", "lexical", "dense", "fusion", "gpt_5_4_listwise_rerank"]
    rows = []
    for condition in condition_order + sorted(set(summaries) - set(condition_order)):
        if condition not in summaries:
            continue
        summary = summaries[condition]
        percentages = summary.get("mean_support_percentages") or {}
        rows.append([
            METHOD_NAMES.get(condition, condition),
            str(summary.get("n", 0)),
            f"{summary.get('mean_added_information_score', 0.0):.3f}",
            f"{percentages.get('supported', 0.0):.3f}",
            f"{percentages.get('partially_supported', 0.0):.3f}",
            f"{percentages.get('unsupported', 0.0):.3f}",
        ])
    return latex_table(
        ["Condition", "N", "Score", "Supported", "Partial", "Unsupported"],
        rows,
        alignment="lccccc",
    )


def best_prior_texts(row: dict, prior_ids: Sequence[str]) -> list[str]:
    prior_by_id = {prior["id"]: prior["text"] for prior in row.get("prior_acus", [])}
    return [prior_by_id.get(prior_id, "") for prior_id in prior_ids]


def select_case_studies(rows: Sequence[dict], condition: str = "oracle") -> list[dict]:
    candidates = [
        row for row in rows
        if row.get("condition") == condition and (row.get("profile") or {}).get("n_query_acus", 0) > 0
    ]
    if not candidates:
        candidates = list(rows)
    if not candidates:
        return []
    sorted_rows = sorted(candidates, key=lambda row: (row.get("profile") or {}).get("added_information_score", 0.0))
    targets = [
        ("mostly_supported", sorted_rows[0]),
        ("extension", sorted_rows[len(sorted_rows) // 2]),
        ("high_added_information", sorted_rows[-1]),
    ]
    case_studies = []
    used = set()
    for label, row in targets:
        key = (label, row.get("query_paper_id"), row.get("condition"))
        if key in used:
            continue
        used.add(key)
        selected_attributions = []
        for attribution in row.get("attributions", [])[:8]:
            selected_attributions.append({
                "query_acu": attribution.get("query_acu", ""),
                "support_status": attribution.get("support_status", ""),
                "best_prior_acus": best_prior_texts(row, attribution.get("best_prior_acu_ids", [])),
                "delta_type": attribution.get("delta_type", ""),
                "rationale": attribution.get("rationale", ""),
            })
        case_studies.append({
            "case_type": label,
            "query_paper_id": row.get("query_paper_id"),
            "query_dataset_name": row.get("query_dataset_name"),
            "condition": row.get("condition"),
            "added_information_score": (row.get("profile") or {}).get("added_information_score"),
            "attributions": selected_attributions,
        })
    return case_studies


def case_studies_markdown(case_studies: Sequence[dict]) -> str:
    lines = ["# Added-Information Case Studies", ""]
    for case in case_studies:
        lines.extend([
            f"## {case['case_type']}: {case.get('query_dataset_name')}",
            "",
            f"- Paper: `{case.get('query_paper_id')}`",
            f"- Condition: `{case.get('condition')}`",
            f"- Added-information score: {case.get('added_information_score', 0.0):.3f}",
            "",
            "| Query ACU | Status | Best Prior ACU | Delta Type | Rationale |",
            "| --- | --- | --- | --- | --- |",
        ])
        for attribution in case.get("attributions", []):
            prior = "; ".join(attribution.get("best_prior_acus") or [""])
            lines.append(
                "| {query} | {status} | {prior} | {delta} | {rationale} |".format(
                    query=escape_md(attribution.get("query_acu", "")),
                    status=attribution.get("support_status", ""),
                    prior=escape_md(prior),
                    delta=attribution.get("delta_type", ""),
                    rationale=escape_md(attribution.get("rationale", "")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def escape_md(text: str) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def sample_annotation_rows(path: str | Path, sample_size: int, seed: int) -> list[dict]:
    rows = read_jsonl(path)
    if len(rows) <= sample_size:
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


def build_evidence_quality_table(report: dict) -> str:
    evidence = ((report.get("added_information_attribution") or {}).get("evidence_quality") or {})
    if not evidence:
        return ""
    rows = [[
        str(evidence.get("n", 0)),
        f"{evidence.get('evidence_label_accuracy', 0.0):.3f}",
        f"{evidence.get('macro_f1', 0.0):.3f}",
        f"{evidence.get('evidence_precision', 0.0):.3f}",
        f"{evidence.get('rationale_groundedness_rate', 0.0):.3f}",
    ]]
    return latex_table(
        ["N", "Accuracy", "Macro-F1", "Evidence precision", "Grounded rationale"],
        rows,
        alignment="ccccc",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready result artifacts from retrieval and attribution outputs.")
    parser.add_argument("--retrieval-markdown", nargs="+", default=DEFAULT_RETRIEVAL_REPORTS)
    parser.add_argument("--attribution-jsonl", default=None)
    parser.add_argument("--human-eval-report-json", default=None)
    parser.add_argument("--annotation-template-jsonl", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--case-study-condition", default="oracle")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_metrics = parse_retrieval_markdown(args.retrieval_markdown)
    write_json(output_dir / "retrieval_metrics.json", retrieval_metrics)
    write_text(output_dir / "retrieval_table.tex", build_retrieval_table(retrieval_metrics))

    if args.attribution_jsonl and Path(args.attribution_jsonl).exists():
        attribution_rows = read_jsonl(args.attribution_jsonl)
        summaries = summarize_attribution_rows(attribution_rows)
        case_studies = select_case_studies(attribution_rows, condition=args.case_study_condition)
        write_json(output_dir / "attribution_summary.json", summaries)
        write_text(output_dir / "attribution_profile_table.tex", build_attribution_table(summaries))
        write_json(output_dir / "case_studies.json", case_studies)
        write_text(output_dir / "case_studies.md", case_studies_markdown(case_studies))

    if args.annotation_template_jsonl and Path(args.annotation_template_jsonl).exists():
        sample_rows = sample_annotation_rows(args.annotation_template_jsonl, args.sample_size, args.sample_seed)
        with (output_dir / "human_evidence_sample.jsonl").open("w", encoding="utf-8") as handle:
            for row in sample_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.human_eval_report_json and Path(args.human_eval_report_json).exists():
        report = json.loads(Path(args.human_eval_report_json).read_text(encoding="utf-8"))
        table = build_evidence_quality_table(report)
        if table:
            write_text(output_dir / "evidence_quality_table.tex", table)

    manifest = {
        "retrieval_sources": args.retrieval_markdown,
        "attribution_jsonl": args.attribution_jsonl,
        "annotation_template_jsonl": args.annotation_template_jsonl,
        "human_eval_report_json": args.human_eval_report_json,
        "outputs": sorted(str(path.relative_to(output_dir)) for path in output_dir.iterdir() if path.is_file()),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
