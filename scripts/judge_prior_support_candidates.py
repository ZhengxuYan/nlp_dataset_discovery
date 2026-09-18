#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


SUPPORT_LABELS = {"gold_prior_support", "hard_negative", "uncertain"}


SYSTEM_PROMPT = """You are judging candidate prior-support datasets/papers for an NLP dataset benchmark.

Goal:
For each candidate, decide whether it should be treated as a gold prior-support item, a hard negative, or uncertain.

Definitions:
- gold_prior_support: an earlier dataset/corpus/benchmark/source data resource that directly supports comparing what the query dataset adds. This includes direct source datasets, closest prior datasets for the same task/domain/language, and explicitly discussed comparison benchmarks.
- hard_negative: superficially related but not a true prior-support item. Examples: generic method paper, broad task family, unrelated benchmark, downstream model, vague topic without a concrete dataset/paper.
- uncertain: plausible but not enough evidence from the query record; needs human review.

Important:
- Be conservative. If the candidate is a vague family name and not a concrete dataset/paper, prefer uncertain or hard_negative.
- Direct source datasets should usually be gold_prior_support.
- Explicit comparison datasets in the query paper should usually be gold_prior_support.
- Low-confidence broad related resources should usually be uncertain or hard_negative.
- Do not require exact paper IDs; we are judging prior-support relevance, not resolving citations.

Return JSON only:
{
  "judgments": [
    {
      "candidate_index": 0,
      "support_label": "gold_prior_support|hard_negative|uncertain",
      "confidence": "low|medium|high",
      "rationale": "...",
      "requires_human_review": true
    }
  ],
  "summary": "..."
}
"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start : end + 1]
            decoder = json.JSONDecoder()
            try:
                payload, _ = decoder.raw_decode(candidate)
                return payload
            except json.JSONDecodeError:
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                payload, _ = decoder.raw_decode(repaired)
                return payload
        raise


def ensure_openai_client():
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for LLM judging.") from exc
    return OpenAI()


def existing_ids(path: str | Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {str(row.get("benchmark_id") or "") for row in read_jsonl(path)}


def compact_query(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": row.get("benchmark_id"),
        "query_paper_id": row.get("query_paper_id"),
        "query_title": row.get("query_title"),
        "query_year": row.get("query_year"),
        "query_dataset_name": row.get("query_dataset_name"),
        "query_dataset_role": row.get("query_dataset_role"),
        "query_acus": row.get("query_acus") or [],
        "construction": {
            "source_data_origin": (row.get("construction") or {}).get("source_data_origin"),
            "source_datasets": (row.get("construction") or {}).get("source_datasets") or [],
            "transformation_types": (row.get("construction") or {}).get("transformation_types") or [],
        },
        "prior_dataset_mentions": row.get("prior_dataset_mentions") or [],
        "candidate_priors": [
            {
                "candidate_index": idx,
                "dataset_name": candidate.get("dataset_name"),
                "paper_title": candidate.get("paper_title"),
                "relationship_type": candidate.get("relationship_type"),
                "why_relevant": candidate.get("why_relevant"),
                "evidence_from_query_paper": candidate.get("evidence_from_query_paper"),
                "confidence": candidate.get("confidence"),
                "search_query": candidate.get("search_query"),
            }
            for idx, candidate in enumerate(row.get("llm_candidate_priors") or [])
        ],
    }


def build_prompt(row: dict[str, Any]) -> str:
    return SYSTEM_PROMPT + "\n\nInput:\n" + json.dumps(compact_query(row), ensure_ascii=False, indent=2)


def normalize_judgments(row: dict[str, Any], payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    candidates = row.get("llm_candidate_priors") or []
    by_index = {}
    for item in payload.get("judgments") or []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("candidate_index"))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        label = item.get("support_label") or "uncertain"
        if label not in SUPPORT_LABELS:
            label = "uncertain"
        by_index[idx] = {
            "support_label": label,
            "confidence": item.get("confidence") or "medium",
            "rationale": item.get("rationale") or "",
            "requires_human_review": bool(item.get("requires_human_review", label == "uncertain")),
        }

    judgments = []
    for idx, candidate in enumerate(candidates):
        judgment = by_index.get(idx) or {
            "support_label": "uncertain",
            "confidence": "low",
            "rationale": "No valid judgment returned for this candidate.",
            "requires_human_review": True,
        }
        merged = dict(candidate)
        merged["candidate_index"] = idx
        merged["second_pass_judgment"] = judgment
        judgments.append(merged)
    return judgments, str(payload.get("summary") or "")


def judge_row(client, row: dict[str, Any], model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.responses.create(model=model.removeprefix("openai/"), input=build_prompt(row))
    payload = parse_json_object(response.output_text)
    judgments, summary = normalize_judgments(row, payload)
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage.dict() if hasattr(usage, "dict") else {})
    output = dict(row)
    output["llm_candidate_priors_judged"] = judgments
    output["second_pass_summary"] = summary
    output["second_pass_model"] = model
    output["second_pass_usage"] = usage_dict
    output["second_pass_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return output, usage_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Second-pass judge LLM-generated prior-support candidates.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="data/benchmark/acl_prior_support_benchmark_candidates_judged.jsonl")
    parser.add_argument("--error-jsonl", default=None)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not args.overwrite:
        done = existing_ids(args.output_jsonl)
        rows = [row for row in rows if str(row.get("benchmark_id") or "") not in done]

    print(json.dumps({
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "remaining_rows": len(rows),
        "model": args.model,
        "dry_run": args.dry_run,
    }, indent=2))

    if args.dry_run:
        for row in rows[:3]:
            print(json.dumps(compact_query(row), ensure_ascii=False, indent=2)[:6000])
        return

    client = ensure_openai_client()
    error_path = args.error_jsonl or str(Path(args.output_jsonl).with_suffix(".errors.jsonl"))
    processed = 0
    failed = 0
    total_input = 0
    total_output = 0
    for row in rows:
        started = time.monotonic()
        try:
            output, usage = judge_row(client, row, args.model)
            output["second_pass_runtime_seconds"] = round(time.monotonic() - started, 3)
            append_jsonl(args.output_jsonl, [output])
            processed += 1
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(error_path, [{
                "benchmark_id": row.get("benchmark_id"),
                "query_paper_id": row.get("query_paper_id"),
                "query_dataset_name": row.get("query_dataset_name"),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "failed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }])
            failed += 1
        print(json.dumps({
            "processed": processed,
            "failed": failed,
            "last": row.get("benchmark_id"),
            "avg_input_tokens": round(total_input / processed, 1) if processed else None,
            "avg_output_tokens": round(total_output / processed, 1) if processed else None,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
