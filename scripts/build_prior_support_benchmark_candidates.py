#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv


RELATIONSHIP_TYPES = {
    "source_dataset",
    "closest_prior_dataset",
    "baseline_benchmark",
    "comparison_dataset",
    "shared_task",
    "loosely_related",
}


SYSTEM_PROMPT = """You are helping construct a benchmark for prior-support retrieval in NLP dataset papers.

Task:
Given one query dataset and its ACUs, propose prior-support candidate datasets/papers that a retrieval system should find.

Definition:
A prior-support paper/dataset is an earlier dataset, corpus, benchmark, shared task, or source dataset that helps determine what information the query dataset adds.

Include:
- source datasets directly reused or transformed
- closest prior datasets for the same task/domain/language
- comparison benchmarks explicitly discussed
- shared tasks or benchmark suites that define the prior setting

Exclude:
- generic method papers
- model-only baselines unless they introduce a relevant dataset
- unrelated datasets only listed in broad experiments
- papers after the query paper year

Return JSON only:
{
  "candidate_priors": [
    {
      "dataset_name": "...",
      "paper_title": "...",
      "relationship_type": "source_dataset|closest_prior_dataset|baseline_benchmark|comparison_dataset|shared_task|loosely_related",
      "why_relevant": "...",
      "supporting_query_acu_ids": ["q0"],
      "evidence_from_query_paper": "...",
      "search_query": "...",
      "confidence": "low|medium|high"
    }
  ],
  "notes": "..."
}

Rules:
- Prefer 3-8 candidates.
- If the provided prior mentions are sparse, propose only candidates strongly implied by query ACUs and paper context.
- Do not fabricate exact paper titles. If the title is unknown, use an empty string and provide a good search_query.
- Evidence must come from the provided query dataset record.
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
            decoder = json.JSONDecoder()
            candidate = cleaned[start : end + 1]
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
        raise RuntimeError("The openai package is required for LLM expansion.") from exc
    return OpenAI()


def existing_ids(path: str | Path) -> set[str]:
    if not Path(path).exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        benchmark_id = str(row.get("benchmark_id") or "").strip()
        if benchmark_id:
            ids.add(benchmark_id)
    return ids


def acu_id(index: int, acu: dict[str, Any]) -> str:
    return str(acu.get("id") or f"q{index}")


def dataset_name(dataset: dict[str, Any]) -> str:
    return str((dataset.get("dataset_identity") or {}).get("canonical_name") or dataset.get("dataset_id") or "").strip()


def artifact_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    availability = dataset.get("availability") or {}
    artifacts = availability.get("artifacts") or {}
    return {
        "release_status": availability.get("release_status"),
        "license": availability.get("license"),
        "dataset_urls": artifacts.get("dataset_urls") or [],
        "github_repos": artifacts.get("github_repos") or [],
        "huggingface_ids": artifacts.get("huggingface_ids") or [],
        "project_page_urls": artifacts.get("project_page_urls") or [],
    }


def query_record_from_dataset(row: dict[str, Any], dataset: dict[str, Any], dataset_index: int) -> dict[str, Any]:
    acus = []
    for idx, acu in enumerate(dataset.get("acus") or []):
        acus.append({
            "id": acu_id(idx, acu),
            "text": acu.get("text") or "",
            "type": acu.get("type") or "other",
            "importance": acu.get("importance") or "medium",
            "evidence": acu.get("evidence") or "",
            "section": acu.get("section") or "",
        })

    name = dataset_name(dataset)
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")[:70] or f"dataset-{dataset_index}"
    return {
        "benchmark_id": f"{row.get('paper_id')}::dataset::{dataset_index}::{safe_name}",
        "query_paper_id": row.get("paper_id"),
        "query_acl_id": row.get("acl_id"),
        "query_title": row.get("title"),
        "query_year": row.get("year"),
        "query_venue_prefix": row.get("venue_prefix"),
        "query_dataset_id": dataset.get("dataset_id"),
        "query_dataset_name": name,
        "query_dataset_role": dataset.get("role"),
        "query_dataset_resource_type": dataset.get("resource_type"),
        "query_acus": acus,
        "coverage": dataset.get("coverage") or {},
        "construction": dataset.get("construction") or {},
        "availability": artifact_summary(dataset),
        "prior_dataset_mentions": dataset.get("prior_dataset_mentions") or [],
        "llm_candidate_priors": [],
        "manual_gold_prior_paper_ids": [],
        "manual_gold_prior_dataset_names": [],
        "manual_hard_negative_ids": [],
        "manual_notes": "",
        "annotation_status": "needs_review",
    }


def collect_dataset_queries(rows: list[dict[str, Any]], min_acus: int, require_prior_mentions: bool) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for row in rows:
        for dataset_index, dataset in enumerate(row.get("datasets") or []):
            if not dataset_name(dataset):
                continue
            if len(dataset.get("acus") or []) < min_acus:
                continue
            if require_prior_mentions and not dataset.get("prior_dataset_mentions"):
                continue
            queries.append(query_record_from_dataset(row, dataset, dataset_index))
    return queries


def build_llm_prompt(query: dict[str, Any]) -> str:
    payload = {
        "query": {
            "benchmark_id": query["benchmark_id"],
            "paper_id": query["query_paper_id"],
            "title": query["query_title"],
            "year": query["query_year"],
            "dataset_name": query["query_dataset_name"],
            "dataset_role": query["query_dataset_role"],
            "resource_type": query["query_dataset_resource_type"],
            "acus": query["query_acus"],
            "coverage": query["coverage"],
            "construction": {
                "source_data_origin": (query["construction"] or {}).get("source_data_origin"),
                "source_datasets": (query["construction"] or {}).get("source_datasets") or [],
                "transformation_types": (query["construction"] or {}).get("transformation_types") or [],
            },
            "availability": query["availability"],
            "prior_dataset_mentions": query["prior_dataset_mentions"],
        }
    }
    return SYSTEM_PROMPT + "\n\nInput:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_candidate_priors(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    candidates = payload.get("candidate_priors") or []
    if not isinstance(candidates, list):
        candidates = []
    normalized = []
    seen = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        dataset = str(item.get("dataset_name") or "").strip()
        title = str(item.get("paper_title") or "").strip()
        search_query = str(item.get("search_query") or "").strip()
        if not (dataset or title or search_query):
            continue
        key = (dataset.lower(), title.lower(), search_query.lower())
        if key in seen:
            continue
        seen.add(key)
        relationship = item.get("relationship_type") or "loosely_related"
        if relationship not in RELATIONSHIP_TYPES:
            relationship = "loosely_related"
        normalized.append({
            "dataset_name": dataset,
            "paper_title": title,
            "relationship_type": relationship,
            "why_relevant": item.get("why_relevant") or "",
            "supporting_query_acu_ids": item.get("supporting_query_acu_ids") if isinstance(item.get("supporting_query_acu_ids"), list) else [],
            "evidence_from_query_paper": item.get("evidence_from_query_paper") or "",
            "search_query": search_query,
            "confidence": item.get("confidence") or "medium",
            "resolved_paper_id": "",
            "resolution_status": "unresolved",
            "manual_is_gold": None,
            "manual_notes": "",
        })
    return normalized, str(payload.get("notes") or "")


def expand_with_openai(client, query: dict[str, Any], model: str) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    response = client.responses.create(model=model.removeprefix("openai/"), input=build_llm_prompt(query))
    payload = parse_json_object(response.output_text)
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage.dict() if hasattr(usage, "dict") else {})
    candidates, notes = normalize_candidate_priors(payload)
    return candidates, notes, usage_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LLM-assisted prior-support benchmark candidate rows from full-text dataset extractions.")
    parser.add_argument("--extractions-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="data/benchmark/acl_prior_support_benchmark_candidates.jsonl")
    parser.add_argument("--error-jsonl", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="random")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--min-acus", type=int, default=3)
    parser.add_argument("--require-prior-mentions", action="store_true")
    parser.add_argument("--llm-expand", action="store_true")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.extractions_jsonl)
    queries = collect_dataset_queries(rows, args.min_acus, args.require_prior_mentions)
    if args.sample_mode == "random":
        rng = random.Random(args.sample_seed)
        rng.shuffle(queries)
    if args.limit is not None:
        queries = queries[: args.limit]
    if not args.overwrite:
        done = existing_ids(args.output_jsonl)
        queries = [query for query in queries if query["benchmark_id"] not in done]

    status = {
        "extractions_jsonl": args.extractions_jsonl,
        "output_jsonl": args.output_jsonl,
        "available_query_datasets": len(collect_dataset_queries(rows, args.min_acus, args.require_prior_mentions)),
        "remaining_rows": len(queries),
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "min_acus": args.min_acus,
        "require_prior_mentions": args.require_prior_mentions,
        "llm_expand": args.llm_expand,
        "model": args.model if args.llm_expand else None,
        "dry_run": args.dry_run,
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))

    if args.dry_run:
        for query in queries[:3]:
            print(json.dumps(query, ensure_ascii=False, indent=2)[:5000])
        return

    client = ensure_openai_client() if args.llm_expand else None
    error_path = args.error_jsonl or str(Path(args.output_jsonl).with_suffix(".errors.jsonl"))
    processed = 0
    failed = 0
    for query in queries:
        started = time.monotonic()
        try:
            if client:
                candidates, notes, usage = expand_with_openai(client, query, args.model)
                query["llm_candidate_priors"] = candidates
                query["llm_candidate_notes"] = notes
                query["llm_usage"] = usage
            query["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            query["runtime_seconds"] = round(time.monotonic() - started, 3)
            append_jsonl(args.output_jsonl, [query])
            processed += 1
        except Exception as exc:  # noqa: BLE001
            append_jsonl(error_path, [{
                "benchmark_id": query.get("benchmark_id"),
                "query_paper_id": query.get("query_paper_id"),
                "query_dataset_name": query.get("query_dataset_name"),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "failed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }])
            failed += 1
        print(json.dumps({"processed": processed, "failed": failed, "last": query.get("benchmark_id")}, ensure_ascii=False))
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
