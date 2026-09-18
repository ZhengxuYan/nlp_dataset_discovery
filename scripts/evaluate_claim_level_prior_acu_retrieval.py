#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def list_tokens(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    output = set()
    for value in values:
        output.update(tokenize(str(value)))
    return output


def metadata_score(query_meta: dict[str, Any], prior_meta: dict[str, Any]) -> float:
    score = 0.0
    score += 0.22 * jaccard(list_tokens(query_meta.get("tasks")), list_tokens(prior_meta.get("tasks")))
    score += 0.15 * jaccard(list_tokens(query_meta.get("domains")), list_tokens(prior_meta.get("domains")))
    score += 0.12 * jaccard(list_tokens(query_meta.get("languages")), list_tokens(prior_meta.get("languages")))
    score += 0.12 * jaccard(list_tokens(query_meta.get("modalities")), list_tokens(prior_meta.get("modalities")))
    score += 0.18 * jaccard(list_tokens(query_meta.get("source_datasets")), list_tokens(prior_meta.get("source_datasets")))
    score += 0.10 * jaccard(tokenize(str(query_meta.get("source_data_origin") or "")), tokenize(str(prior_meta.get("source_data_origin") or "")))
    score += 0.06 * jaccard(tokenize(str(query_meta.get("annotation_protocol") or "")), tokenize(str(prior_meta.get("annotation_protocol") or "")))
    score += 0.05 * jaccard(tokenize(str(query_meta.get("collection_method") or "")), tokenize(str(prior_meta.get("collection_method") or "")))
    return score


def score_pair(query_acu: dict[str, Any], query_meta: dict[str, Any], prior_acu: dict[str, Any], method: str) -> float:
    q_tokens = tokenize(query_acu.get("text") or "")
    p_tokens = tokenize(prior_acu.get("text") or "")
    text_score = 0.65 * containment(q_tokens, p_tokens) + 0.35 * jaccard(q_tokens, p_tokens)
    type_bonus = 0.18 if query_acu.get("type") and query_acu.get("type") == prior_acu.get("type") else 0.0
    if method == "acu_text":
        return text_score
    if method == "acu_type_text":
        return text_score + type_bonus
    if method == "dcu_metadata":
        return 0.48 * text_score + type_bonus + 0.52 * metadata_score(query_meta, prior_acu.get("metadata") or {})
    raise ValueError(f"Unknown method: {method}")


def rank_prior_acus(query_acu: dict[str, Any], query_meta: dict[str, Any], prior_acus: list[dict[str, Any]], method: str) -> list[str]:
    scored = [
        (score_pair(query_acu, query_meta, prior_acu, method), prior_acu["id"])
        for prior_acu in prior_acus
    ]
    return [prior_id for _, prior_id in sorted(scored, key=lambda item: item[0], reverse=True)]


def evaluate(rows: list[dict[str, Any]], methods: list[str], top_ks: list[int]) -> dict[str, Any]:
    metric_rows = {method: [] for method in methods}
    by_delta = {method: defaultdict(list) for method in methods}
    examples = {method: [] for method in methods}
    total_labels = 0

    for row in rows:
        query_meta = row.get("query_metadata") or {}
        prior_acus = row.get("prior_acu_bank") or []
        prior_ids = {acu["id"] for acu in prior_acus}
        query_by_id = row.get("query_acu_by_id") or {acu["id"]: acu for acu in row.get("query_acus") or []}
        for label in row.get("labels") or []:
            if not label.get("evaluate"):
                continue
            gold = [prior_id for prior_id in label.get("selected_prior_acu_ids") or [] if prior_id in prior_ids]
            if not gold:
                continue
            query_acu = query_by_id.get(label["query_acu_id"])
            if not query_acu:
                continue
            total_labels += 1
            for method in methods:
                ranking = rank_prior_acus(query_acu, query_meta, prior_acus, method)
                first_rank = None
                for index, prior_id in enumerate(ranking, start=1):
                    if prior_id in gold:
                        first_rank = index
                        break
                metrics = {
                    "mrr": 1.0 / first_rank if first_rank else 0.0,
                    "gold_count": len(gold),
                }
                for k in top_ks:
                    metrics[f"recall@{k}"] = 1.0 if any(prior_id in gold for prior_id in ranking[:k]) else 0.0
                metric_rows[method].append(metrics)
                by_delta[method][label.get("delta_type") or "other"].append(metrics)
                if first_rank and first_rank > 3 and len(examples[method]) < 10:
                    examples[method].append({
                        "benchmark_id": row.get("benchmark_id"),
                        "query_dataset_name": row.get("query_dataset_name"),
                        "query_acu": query_acu,
                        "label": label,
                        "gold_prior_acu_ids": gold,
                        "first_gold_rank": first_rank,
                        "top5": ranking[:5],
                    })

    def summarize(items: list[dict[str, float]]) -> dict[str, float]:
        if not items:
            return {"n": 0}
        keys = sorted({key for item in items for key in item})
        return {"n": len(items), **{key: sum(float(item.get(key, 0.0)) for item in items) / len(items) for key in keys}}

    return {
        "claim_labels": total_labels,
        "by_method": {method: summarize(items) for method, items in metric_rows.items()},
        "by_delta_type": {
            method: {delta: summarize(items) for delta, items in delta_rows.items()}
            for method, delta_rows in by_delta.items()
        },
        "miss_examples": examples,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Claim-Level Prior ACU Retrieval",
        "",
        f"- Claim labels: {report['claim_labels']}",
        "",
        "| Method | N | MRR | R@1 | R@3 | R@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in report["by_method"].items():
        lines.append(
            f"| {method} | {row.get('n', 0)} | {row.get('mrr', 0.0):.3f} | {row.get('recall@1', 0.0):.3f} | {row.get('recall@3', 0.0):.3f} | {row.get('recall@5', 0.0):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local retrieval of claim-level selected prior ACUs.")
    parser.add_argument("--claim-level-jsonl", required=True)
    parser.add_argument("--methods", nargs="+", default=["acu_text", "acu_type_text", "dcu_metadata"])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.claim_level_jsonl)
    report = evaluate(rows, args.methods, args.top_ks)
    report.update({
        "claim_level_jsonl": args.claim_level_jsonl,
        "methods": args.methods,
        "top_ks": args.top_ks,
    })
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        output_markdown = Path(args.output_markdown)
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "output_json": str(output_json),
        "output_markdown": args.output_markdown,
        "claim_labels": report["claim_labels"],
        "by_method": report["by_method"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
