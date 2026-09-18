#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return re.sub(r"\s+", " ", text)


def token_overlap(left: str, right: str) -> float:
    left_tokens = set(normalize(left).split())
    right_tokens = set(normalize(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def best_dataset_records(prior_extraction: dict[str, Any], candidate_names: Iterable[str]) -> list[dict[str, Any]]:
    datasets = prior_extraction.get("datasets") or []
    if not datasets:
        return []
    names = [name for name in candidate_names if name]
    if not names:
        return datasets
    scored = []
    for dataset in datasets:
        identity = dataset.get("dataset_identity") or {}
        dataset_names = [identity.get("canonical_name") or "", *(identity.get("aliases") or [])]
        score = max((token_overlap(a, b) for a in names for b in dataset_names), default=0.0)
        scored.append((score, dataset))
    best = max(score for score, _ in scored)
    if best >= 0.25:
        return [dataset for score, dataset in scored if score == best]
    return datasets


def build_extraction_index(prior_inputs: list[dict[str, Any]], prior_extractions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    input_by_paper_id = {row.get("paper_id"): row for row in prior_inputs}
    by_prior_ref_id = {}
    for extraction in prior_extractions:
        paper_id = extraction.get("paper_id")
        prior_input = input_by_paper_id.get(paper_id) or {}
        prior_ref_id = prior_input.get("prior_ref_id")
        if prior_ref_id:
            enriched = dict(extraction)
            enriched["_prior_input"] = prior_input
            by_prior_ref_id[prior_ref_id] = enriched
    return by_prior_ref_id


def build_mentions_index(reference_queue: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_benchmark_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in reference_queue:
        for mention in item.get("query_dataset_mentions") or []:
            benchmark_id = mention.get("benchmark_id")
            if benchmark_id:
                by_benchmark_id[benchmark_id].append({"queue_item": item, "mention": mention})
    return by_benchmark_id


def acu_rows_for_prior(
    prior_ref_id: str,
    prior_extraction: dict[str, Any],
    *,
    candidate_names: Iterable[str],
    prior_index: int,
) -> list[dict[str, Any]]:
    acus = []
    datasets = best_dataset_records(prior_extraction, candidate_names)
    for dataset_index, dataset in enumerate(datasets):
        identity = dataset.get("dataset_identity") or {}
        dataset_name = identity.get("canonical_name") or dataset.get("dataset_id") or ""
        for acu_index, acu in enumerate(dataset.get("acus") or []):
            acus.append({
                "id": f"p{prior_index}_d{dataset_index}_a{acu_index}",
                "text": acu.get("text") or "",
                "type": acu.get("type") or "other",
                "importance": acu.get("importance") or "medium",
                "evidence": acu.get("evidence") or "",
                "section": acu.get("section") or "",
                "prior_ref_id": prior_ref_id,
                "prior_paper_id": prior_extraction.get("paper_id"),
                "prior_paper_title": prior_extraction.get("title"),
                "prior_paper_year": prior_extraction.get("year"),
                "prior_dataset_name": dataset_name,
                "prior_dataset_id": dataset.get("dataset_id") or "",
            })
    return [acu for acu in acus if acu.get("text")]


def build_rows(
    query_rows: list[dict[str, Any]],
    reference_queue: list[dict[str, Any]],
    prior_inputs: list[dict[str, Any]],
    prior_extractions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mentions_by_benchmark = build_mentions_index(reference_queue)
    extraction_by_prior_ref = build_extraction_index(prior_inputs, prior_extractions)
    completed = []
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    for query in query_rows:
        benchmark_id = query.get("benchmark_id")
        mentions = mentions_by_benchmark.get(benchmark_id) or []
        prior_refs = []
        prior_acus = []
        seen_refs = set()
        for mention_item in mentions:
            queue_item = mention_item["queue_item"]
            mention = mention_item["mention"]
            prior_ref_id = queue_item.get("prior_ref_id")
            prior_extraction = extraction_by_prior_ref.get(prior_ref_id)
            if not prior_ref_id or not prior_extraction:
                continue
            if prior_ref_id not in seen_refs:
                seen_refs.add(prior_ref_id)
                prior_refs.append({
                    "prior_ref_id": prior_ref_id,
                    "prior_paper_id": prior_extraction.get("paper_id"),
                    "prior_paper_title": prior_extraction.get("title"),
                    "prior_paper_year": prior_extraction.get("year"),
                    "resolution": (prior_extraction.get("_prior_input") or {}).get("resolution") or {},
                    "candidate_dataset_name": mention.get("candidate_dataset_name"),
                    "candidate_paper_title": mention.get("candidate_paper_title"),
                    "relationship_type": mention.get("relationship_type"),
                    "reference_match_confidence": mention.get("reference_match_confidence"),
                })
            candidate_names = [
                mention.get("candidate_dataset_name") or "",
                *((queue_item.get("candidate_dataset_names") or {}).keys()),
            ]
            prior_acus.extend(
                acu_rows_for_prior(
                    prior_ref_id,
                    prior_extraction,
                    candidate_names=candidate_names,
                    prior_index=len(prior_refs) - 1,
                )
            )
        row = dict(query)
        row["gold_prior_support_refs"] = prior_refs
        row["gold_prior_paper_ids"] = [ref["prior_paper_id"] for ref in prior_refs if ref.get("prior_paper_id")]
        row["gold_prior_support_acus"] = prior_acus
        row["annotation_status"] = "complete" if row.get("query_acus") and prior_acus else "missing_prior_acus"
        row["benchmark_source"] = "acl_llm_citation_grounded_v1"
        row["built_at"] = now
        completed.append(row)
    return completed


def write_summary(path: str | Path, rows: list[dict[str, Any]]) -> None:
    statuses = collections.Counter(row.get("annotation_status") for row in rows)
    prior_counts = [len(row.get("gold_prior_support_refs") or []) for row in rows]
    acu_counts = [len(row.get("gold_prior_support_acus") or []) for row in rows]
    summary = {
        "rows": len(rows),
        "status_counts": dict(statuses),
        "complete_rows": statuses.get("complete", 0),
        "mean_prior_refs_per_row": sum(prior_counts) / len(prior_counts) if prior_counts else 0,
        "mean_prior_acus_per_row": sum(acu_counts) / len(acu_counts) if acu_counts else 0,
        "total_prior_refs_linked": sum(prior_counts),
        "total_prior_acus_linked": sum(acu_counts),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACL prior-support benchmark rows by joining query rows with extracted prior ACUs.")
    parser.add_argument("--query-jsonl", required=True)
    parser.add_argument("--reference-queue-jsonl", required=True)
    parser.add_argument("--prior-input-jsonl", required=True)
    parser.add_argument("--prior-extractions-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    rows = build_rows(
        read_jsonl(args.query_jsonl),
        read_jsonl(args.reference_queue_jsonl),
        read_jsonl(args.prior_input_jsonl),
        read_jsonl(args.prior_extractions_jsonl),
    )
    write_jsonl(args.output_jsonl, rows)
    if args.summary_json:
        write_summary(args.summary_json, rows)
    print(json.dumps({
        "output_jsonl": args.output_jsonl,
        "summary_json": args.summary_json,
        "rows": len(rows),
        "complete_rows": sum(row.get("annotation_status") == "complete" for row in rows),
        "total_prior_acus_linked": sum(len(row.get("gold_prior_support_acus") or []) for row in rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
