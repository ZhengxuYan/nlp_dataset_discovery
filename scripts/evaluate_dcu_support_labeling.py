#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import evaluate_global_claim_level_dcu_retrieval as ev


SUPPORT_STATUSES = {"supported", "partially_supported", "unsupported", "not_comparable", "contradicted"}
POSITIVE_STATUSES = {"supported", "partially_supported"}
PROMPT_VARIANTS = {
    "zero_shot_support_label",
    "fewshot_balanced_support_label",
    "fewshot_partial_sensitive",
    "two_stage_support_label",
}
PROMPT_VERSION = "dcu_support_labeling_v2"


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def gold_global_ids(row: dict[str, Any], label: dict[str, Any], candidate_ids: set[str]) -> list[str]:
    return [gid for gid in ev.local_gold_to_global_ids(row, label) if gid in candidate_ids]


def normalize_gold_status(label: dict[str, Any]) -> str:
    status = str(label.get("support_status") or "unsupported")
    if status not in SUPPORT_STATUSES:
        return "unsupported"
    return status


def row_prior_bank_global_ids(row: dict[str, Any], candidate_by_id: dict[str, dict[str, Any]]) -> list[str]:
    output = []
    seen = set()
    for acu in row.get("prior_acu_bank") or []:
        global_id = ev.stable_acu_key(
            str(acu.get("prior_paper_id") or ""),
            str(acu.get("prior_dataset_id") or ""),
            str(acu.get("type") or ""),
            str(acu.get("text") or ""),
        )
        if global_id in candidate_by_id and global_id not in seen:
            output.append(global_id)
            seen.add(global_id)
    return output


def candidate_card(candidate: dict[str, Any]) -> str:
    meta = candidate.get("metadata") or {}
    return (
        f"Prior DCU ID: {candidate['id']}\n"
        f"Paper: {candidate.get('prior_paper_title') or ''}\n"
        f"Dataset: {meta.get('dataset_name') or candidate.get('prior_dataset_name') or ''}\n"
        f"Type: {candidate.get('type') or ''}\n"
        f"Task/domain: {ev.format_list(meta.get('tasks'), 4)} / {ev.format_list(meta.get('domains'), 4)}\n"
        f"Language/modality: {ev.format_list(meta.get('languages'), 4)} / {ev.format_list(meta.get('modalities'), 4)}\n"
        f"Source/annotation: {meta.get('source_data_origin') or ''} {meta.get('annotation_protocol') or ''}\n"
        f"Scale: {meta.get('scale') or ''}\n"
        f"Claim: {candidate.get('text') or ''}"
    )


def fewshot_block(variant: str) -> str:
    if variant == "zero_shot_support_label":
        return ""
    balanced = """
Examples:
1. supported:
Query: "Dataset X contains human-written safety preference labels."
Prior: "Dataset Y contains human-written preference labels for safety alignment."
Label: supported, because the prior covers nearly the same contribution dimension.

2. partially_supported:
Query: "Dataset X contains 10,461 multilingual preference instances."
Prior: "Dataset Y contains 7,118 preference pairs."
Label: partially_supported, because Dataset Y is prior evidence for the same preference-data/scale dimension even though it does not mention Dataset X and the exact size/coverage differ.

3. unsupported:
Query: "Dataset X is released with a permissive commercial license."
Prior: "Dataset Y reports benchmark accuracy on summarization."
Label: unsupported, because the prior DCU is about evaluation, not availability/license.

4. not_comparable:
Query: "Dataset X has high-quality examples."
Prior: "Dataset Y was used to train a parser."
Label: not_comparable if the query claim is too vague or the candidate evidence cannot be compared.
"""
    if variant == "fewshot_balanced_support_label":
        return balanced
    partial = """
Partial-support calibration examples:
- same task but different scale => partially_supported, not unsupported.
- same source dataset but different annotation protocol => partially_supported.
- same language/domain but different benchmark format => partially_supported.
- same dataset family with narrower coverage => partially_supported.
- prior dataset does not mention the query dataset name => this is expected and must not by itself imply unsupported.
- exact same release/license/source/protocol claim => supported.
- broad NLP topic overlap without a comparable dataset contribution dimension => unsupported.

Prefer partially_supported over unsupported when a candidate prior DCU is useful evidence for the same dataset contribution dimension.
Do not mark supported unless the prior nearly covers the full query claim.
"""
    if variant == "fewshot_partial_sensitive":
        return balanced + "\n" + partial
    if variant == "two_stage_support_label":
        return balanced + """
For this variant, reason in two stages internally:
Stage 1: choose candidate prior DCUs that provide any direct or partial support.
Stage 2: assign the final support label from the selected evidence.
Return only the final JSON.
"""
    raise ValueError(f"Unknown prompt variant: {variant}")


def build_prompt(
    *,
    query_dcu: dict[str, Any],
    candidates: list[dict[str, Any]],
    variant: str,
) -> str:
    schema = (
        '{"support_status":"supported|partially_supported|unsupported|not_comparable|contradicted",'
        '"selected_prior_dcu_ids":["id"],"rationale":"brief evidence-grounded reason"}'
    )
    if variant == "two_stage_support_label":
        schema = (
            '{"candidate_evidence_ids":["id"],'
            '"support_status":"supported|partially_supported|unsupported|not_comparable|contradicted",'
            '"selected_prior_dcu_ids":["id"],"rationale":"brief evidence-grounded reason"}'
        )
    return f"""You are evaluating claim-level prior support for NLP dataset contributions.

Task:
Given one query Dataset Contribution Unit (DCU) and retrieved candidate prior DCUs, decide whether the prior candidates support the query claim.

Definitions:
- supported: selected prior DCUs already cover the query claim or nearly the same dataset contribution.
- partially_supported: selected prior DCUs cover the same contribution dimension or useful comparator evidence, but the query adds meaningful new information.
- unsupported: no candidate prior DCU provides meaningful support for this contribution dimension.
- contradicted: candidate prior DCUs directly conflict with the query claim.
- not_comparable: the query claim cannot be compared to the candidates.

Rules:
- Use only candidate prior DCU IDs listed below.
- Topical similarity alone is not enough.
- A valid prior DCU usually will not mention the query dataset name, because it is prior work. Do not require the prior to describe the query dataset itself.
- Judge whether the candidate is useful prior evidence for the same contribution dimension, not whether it exactly states the query claim.
- Partial support is important: same task/domain/source/annotation/scale/evaluation dimension can be partially_supported even when exact details differ.
- For source/data claims, a candidate about the source dataset or source family is partially_supported even if the query adds private data or new sampling details.
- For scale claims, a candidate with comparable size/count/coverage evidence is partially_supported even when numbers differ.
- For task/domain claims, a candidate with a closely related benchmark task/domain is partially_supported even when the query benchmark tests a new variant.
- Do not over-correct toward support: if candidates are broad topical neighbors but not useful prior evidence for the query contribution dimension, choose unsupported.
- If support_status is supported or partially_supported, selected_prior_dcu_ids must be non-empty.
- If no candidate provides evidence, return unsupported with an empty selected_prior_dcu_ids list.
{fewshot_block(variant)}

Return JSON only with this schema:
{schema}

Query DCU:
{ev.dcu_brief(query_dcu)}

Candidate prior DCUs:
{chr(10).join(candidate_card(candidate) for candidate in candidates)}
"""


def normalize_prediction(payload: dict[str, Any], candidate_ids: set[str]) -> dict[str, Any]:
    status = str(payload.get("support_status") or "unsupported")
    if status not in SUPPORT_STATUSES:
        status = "unsupported"
    selected = [
        str(candidate_id)
        for candidate_id in payload.get("selected_prior_dcu_ids") or payload.get("candidate_evidence_ids") or []
        if str(candidate_id) in candidate_ids
    ]
    if status in POSITIVE_STATUSES and not selected:
        status = "unsupported"
    if status not in POSITIVE_STATUSES:
        selected = selected if status == "contradicted" else []
    return {
        "support_status": status,
        "selected_prior_dcu_ids": selected,
        "rationale": str(payload.get("rationale") or ""),
    }


def status_match(gold: str, pred: str, collapse_positive: bool) -> bool:
    if collapse_positive and gold in POSITIVE_STATUSES and pred in POSITIVE_STATUSES:
        return True
    return gold == pred


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_rows(rows: list[dict[str, Any]], collapse_positive: bool) -> dict[str, Any]:
    n = len(rows)
    status_correct = sum(
        1 for row in rows
        if status_match(row["gold_status"], row["prediction"]["support_status"], collapse_positive)
    )
    evidence_tp = evidence_fp = evidence_fn = 0
    unsupported_fn = 0
    unsupported_fp = 0
    labels = (
        ["supported_or_partial", "unsupported", "not_comparable", "contradicted"]
        if collapse_positive
        else sorted(SUPPORT_STATUSES)
    )
    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = fp = fn = 0
        for row in rows:
            gold = row["gold_status"]
            pred = row["prediction"]["support_status"]
            if collapse_positive:
                gold = "supported_or_partial" if gold in POSITIVE_STATUSES else gold
                pred = "supported_or_partial" if pred in POSITIVE_STATUSES else pred
                current = label
            else:
                current = label
            if gold == current and pred == current:
                tp += 1
            elif gold != current and pred == current:
                fp += 1
            elif gold == current and pred != current:
                fn += 1
        per_label[label] = precision_recall_f1(tp, fp, fn)
    for row in rows:
        gold_ids = set(row["gold_prior_dcu_ids"])
        pred_ids = set(row["prediction"]["selected_prior_dcu_ids"])
        evidence_tp += len(gold_ids & pred_ids)
        evidence_fp += len(pred_ids - gold_ids)
        evidence_fn += len(gold_ids - pred_ids)
        gold_positive = row["gold_status"] in POSITIVE_STATUSES
        pred_unsupported = row["prediction"]["support_status"] == "unsupported"
        if gold_positive and pred_unsupported:
            unsupported_fn += 1
        if row["gold_status"] == "unsupported" and row["prediction"]["support_status"] in POSITIVE_STATUSES:
            unsupported_fp += 1
    macro_f1 = sum(item["f1"] for item in per_label.values()) / len(per_label) if per_label else 0.0
    return {
        "n": n,
        "label_accuracy": status_correct / n if n else 0.0,
        "macro_f1": macro_f1,
        "unsupported_false_negative_rate": unsupported_fn / n if n else 0.0,
        "unsupported_false_positive_rate": unsupported_fp / n if n else 0.0,
        "evidence": precision_recall_f1(evidence_tp, evidence_fp, evidence_fn),
        "gold_status_counts": dict(Counter(row["gold_status"] for row in rows)),
        "pred_status_counts": dict(Counter(row["prediction"]["support_status"] for row in rows)),
    }


def build_query_items(
    claim_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    include_unsupported: bool,
    limit: int | None,
    sample_size: int | None,
    sample_seed: int,
    sample_status_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], int]:
    candidate_ids = {candidate["id"] for candidate in candidates}
    items = []
    for row in claim_rows:
        query_by_id = row.get("query_acu_by_id") or {acu["id"]: acu for acu in row.get("query_acus") or []}
        for label in row.get("labels") or []:
            gold_status = normalize_gold_status(label)
            if gold_status == "unsupported" and not include_unsupported:
                continue
            query_acu = query_by_id.get(label.get("query_acu_id"))
            if not query_acu:
                continue
            gold_ids = gold_global_ids(row, label, candidate_ids)
            items.append({
                "row": row,
                "label": label,
                "query_dcu": ev.query_dcu_from_row(row, query_acu),
                "gold_status": gold_status,
                "gold_prior_dcu_ids": gold_ids,
            })
            if limit is not None and sample_size is None and not sample_status_counts and len(items) >= limit:
                return items, sum(
                    1 for item in items
                    if item["gold_status"] in POSITIVE_STATUSES and not item["gold_prior_dcu_ids"]
                )
    if sample_status_counts:
        rng = random.Random(sample_seed)
        by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_status[item["gold_status"]].append(item)
        sampled = []
        for status, count in sample_status_counts.items():
            bucket = list(by_status.get(status) or [])
            rng.shuffle(bucket)
            sampled.extend(bucket[:count])
        rng.shuffle(sampled)
        items = sampled
    elif sample_size is not None:
        rng = random.Random(sample_seed)
        items = list(items)
        rng.shuffle(items)
        items = items[:sample_size]
    missing_gold = sum(
        1 for item in items
        if item["gold_status"] in POSITIVE_STATUSES and not item["gold_prior_dcu_ids"]
    )
    return items, missing_gold


def parse_status_counts(values: list[str] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected STATUS=COUNT for --sample-status-counts, got: {value}")
        status, count_text = value.split("=", 1)
        status = status.strip()
        if status not in SUPPORT_STATUSES:
            raise ValueError(f"Unknown support status for --sample-status-counts: {status}")
        counts[status] = int(count_text)
    return counts


def evaluate(
    *,
    claim_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    prompt_variants: list[str],
    model: str,
    rerank_depth: int,
    include_unsupported: bool,
    limit: int | None,
    sample_size: int | None,
    sample_seed: int,
    sample_status_counts: dict[str, int],
    cache_dir: str,
    collapse_positive: bool,
    candidate_scope: str,
) -> dict[str, Any]:
    candidates, papers = ev.flatten_prior_acus(prior_rows)
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    candidate_ids = [candidate["id"] for candidate in candidates]
    indexes = {
        "dcu": ev.TextIndex(candidate_ids, [ev.serialize(candidate, "dcu") for candidate in candidates]),
    }
    items, missing_gold = build_query_items(
        claim_rows,
        candidates,
        include_unsupported=include_unsupported,
        limit=limit,
        sample_size=sample_size,
        sample_seed=sample_seed,
        sample_status_counts=sample_status_counts,
    )
    cache = ev.JsonCache(cache_dir)
    rows_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in prompt_variants}
    errors: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        query_dcu = item["query_dcu"]
        if candidate_scope == "gold_prior_bank":
            top_ids = row_prior_bank_global_ids(item["row"], candidate_by_id)
            if rerank_depth > 0:
                top_ids = top_ids[:rerank_depth]
        else:
            scored = ev.score_retrieval_method("hybrid_dcu", query_dcu, candidates, papers, indexes)
            ranking = ev.stable_rank(scored, candidate_by_id)
            top_ids = ranking[:rerank_depth]
        top_set = set(top_ids)
        candidate_cards = [candidate_by_id[candidate_id] for candidate_id in top_ids]
        answerable = (
            item["gold_status"] == "unsupported"
            or bool(set(item["gold_prior_dcu_ids"]) & top_set)
        )
        for variant in prompt_variants:
            key = {
                "schema": "dcu_support_labeling",
                "prompt_version": PROMPT_VERSION,
                "variant": variant,
                "model": model,
                "candidate_scope": candidate_scope,
                "query_id": query_dcu["id"],
                "query_text": query_dcu.get("text"),
                "candidate_ids": top_ids,
            }
            cached = cache.get("support_labeling", key)
            try:
                if cached is None:
                    prompt = build_prompt(query_dcu=query_dcu, candidates=candidate_cards, variant=variant)
                    payload, usage = ev.call_json_llm(prompt, model=model)
                    prediction = normalize_prediction(payload, set(top_ids))
                    cached = cache.set("support_labeling", key, {
                        "prediction": prediction,
                        "raw": payload,
                        "usage": usage,
                    })
                prediction = cached["prediction"]
                rows_by_variant[variant].append({
                    "benchmark_id": item["row"].get("benchmark_id"),
                    "query_dataset_name": item["row"].get("query_dataset_name"),
                    "query_dcu": query_dcu,
                    "gold_status": item["gold_status"],
                    "gold_prior_dcu_ids": item["gold_prior_dcu_ids"],
                    "gold_prior_in_topk": bool(set(item["gold_prior_dcu_ids"]) & top_set),
                    "answerable": answerable,
                    "prediction": prediction,
                    "top_candidate_ids": top_ids,
                })
            except Exception as exc:  # noqa: BLE001
                errors.append({
                    "variant": variant,
                    "query_id": query_dcu["id"],
                    "error": str(exc),
                })
        if idx % 25 == 0:
            print(json.dumps({"progress": idx, "total": len(items)}, ensure_ascii=False), flush=True)
    by_variant = {}
    for variant, rows in rows_by_variant.items():
        answerable_rows = [row for row in rows if row["answerable"]]
        retrieval_miss_rows = [row for row in rows if not row["answerable"]]
        by_variant[variant] = {
            "all": evaluate_rows(rows, collapse_positive),
            "answerable": evaluate_rows(answerable_rows, collapse_positive),
            "retrieval_miss": evaluate_rows(retrieval_miss_rows, collapse_positive),
            "examples": {
                "unsupported_false_negatives": [
                    row for row in answerable_rows
                    if row["gold_status"] in POSITIVE_STATUSES and row["prediction"]["support_status"] == "unsupported"
                ][:10],
                "unsupported_false_positives": [
                    row for row in answerable_rows
                    if row["gold_status"] == "unsupported" and row["prediction"]["support_status"] in POSITIVE_STATUSES
                ][:10],
            },
        }
    return {
        "claim_labels": len(items),
        "missing_positive_gold_global_ids": missing_gold,
        "global_prior_dcus": len(candidates),
        "rerank_depth": rerank_depth,
        "candidate_scope": candidate_scope,
        "model": model,
        "prompt_variants": prompt_variants,
        "include_unsupported": include_unsupported,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
        "sample_status_counts": sample_status_counts,
        "selected_gold_status_counts": dict(Counter(item["gold_status"] for item in items)),
        "collapse_positive": collapse_positive,
        "by_variant": by_variant,
        "errors": errors[:100],
        "error_count": len(errors),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# DCU Support Labeling Evaluation",
        "",
        f"- Claim labels: {report['claim_labels']}",
        f"- Model: {report['model']}",
        f"- Rerank depth: {report['rerank_depth']}",
        f"- Errors: {report['error_count']}",
        "",
        "| Variant | Subset | N | Acc | Macro F1 | Evidence P | Evidence R | Evidence F1 | Unsupported FN | Unsupported FP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, payload in report["by_variant"].items():
        for subset in ["all", "answerable", "retrieval_miss"]:
            row = payload[subset]
            evidence = row["evidence"]
            lines.append(
                f"| {variant} | {subset} | {row['n']} | "
                f"{row['label_accuracy']:.3f} | {row['macro_f1']:.3f} | "
                f"{evidence['precision']:.3f} | {evidence['recall']:.3f} | {evidence['f1']:.3f} | "
                f"{row['unsupported_false_negative_rate']:.3f} | {row['unsupported_false_positive_rate']:.3f} |"
            )
    lines.append("")
    lines.append("## Status Counts")
    for variant, payload in report["by_variant"].items():
        row = payload["answerable"]
        lines.append(f"### {variant}")
        lines.append(f"- Gold: `{json.dumps(row['gold_status_counts'], ensure_ascii=False)}`")
        lines.append(f"- Pred: `{json.dumps(row['pred_status_counts'], ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM support labeling from hybrid-DCU top-k prior DCUs.")
    parser.add_argument("--claim-level-jsonl", required=True)
    parser.add_argument("--prior-extractions-jsonl", required=True)
    parser.add_argument("--prompt-variants", nargs="+", choices=sorted(PROMPT_VARIANTS), default=["zero_shot_support_label"])
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument(
        "--candidate-scope",
        choices=["retrieved_topk", "gold_prior_bank"],
        default="retrieved_topk",
        help="Use hybrid_dcu global retrieved top-k candidates, or the original gold prior paper ACU bank.",
    )
    parser.add_argument("--include-unsupported", action="store_true")
    parser.add_argument("--strict-supported-vs-partial", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None, help="Randomly sample this many claim labels after filtering.")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument(
        "--sample-status-counts",
        nargs="*",
        default=None,
        help="Seeded stratified sample, e.g. unsupported=25 partially_supported=20 supported=5.",
    )
    parser.add_argument("--cache-dir", default="data/benchmark/retrieval_cache/dcu_support_labeling_cache")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    sample_status_counts = parse_status_counts(args.sample_status_counts)

    report = evaluate(
        claim_rows=ev.read_jsonl(args.claim_level_jsonl),
        prior_rows=ev.read_jsonl(args.prior_extractions_jsonl),
        prompt_variants=args.prompt_variants,
        model=args.model,
        rerank_depth=args.rerank_depth,
        include_unsupported=args.include_unsupported,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        sample_status_counts=sample_status_counts,
        cache_dir=args.cache_dir,
        collapse_positive=not args.strict_supported_vs_partial,
        candidate_scope=args.candidate_scope,
    )
    report.update({
        "claim_level_jsonl": args.claim_level_jsonl,
        "prior_extractions_jsonl": args.prior_extractions_jsonl,
        "cache_dir": args.cache_dir,
    })
    write_json(args.output_json, report)
    write_markdown(args.output_markdown, markdown_report(report))
    print(json.dumps({
        "output_json": args.output_json,
        "output_markdown": args.output_markdown,
        "claim_labels": report["claim_labels"],
        "missing_positive_gold_global_ids": report["missing_positive_gold_global_ids"],
        "global_prior_dcus": report["global_prior_dcus"],
        "by_variant": {
            variant: {
                subset: {
                    key: value
                    for key, value in payload[subset].items()
                    if key not in {"gold_status_counts", "pred_status_counts"}
                }
                for subset in ["all", "answerable", "retrieval_miss"]
            }
            for variant, payload in report["by_variant"].items()
        },
        "error_count": report["error_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
