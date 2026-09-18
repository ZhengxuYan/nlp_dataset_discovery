#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv


DEFAULT_CATALOG = "data/census/pending_papers.jsonl"
DEFAULT_OUTPUT = "data/census/dataset_census_classification.jsonl"


SYSTEM_PROMPT = """You are building a census of NLP papers that introduce datasets.

Classify each paper using only the provided metadata and abstract. Be conservative and grounded.

A paper is dataset-introducing if it creates, releases, curates, annotates, translates, filters, augments, synthesizes, or substantially repackages a named dataset, corpus, benchmark, training set, evaluation set, test set, or data collection as a contribution of this paper.

Do not mark a paper as dataset-introducing if it only uses existing datasets for training/evaluation, only proposes a model/method, or only surveys prior datasets.

For each introduced dataset, extract short dataset-centric ACUs. These are claims about the dataset itself: source, task/domain, annotation/protocol, scale/coverage, evaluation/use, availability/quality. If the abstract does not provide enough detail, include only grounded ACUs and set confidence lower.

Return JSON only with this shape:
{
  "papers": [
    {
      "paper_id": "...",
      "is_nlp_paper": true,
      "is_dataset_mentioned": true,
      "is_dataset_introducing": true,
      "introduced_datasets": [
        {
          "name": "...",
          "role": "Main Contribution|Training Data|Evaluation Benchmark|Fine-tuning|Other",
          "source_dataset": "None or source dataset name",
          "transformation_type": "None|Filtering|Annotation|Translation|Augmentation|Synthetic generation|Mixture|Other",
          "usage_description": "...",
          "acus": ["..."],
          "confidence": "High|Medium|Low"
        }
      ],
      "exclusion_reason": "..."
    }
  ]
}
"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv_catalog(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    deduped = []
    for row in rows:
        paper_id = str(row.get("arXiv ID") or row.get("arxiv_id") or "").strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        deduped.append({
            "paper_id": paper_id,
            "title": row.get("Title") or "",
            "published_date": row.get("Publication Date") or "",
            "abstract": row.get("Abstract") or "",
            "categories": row.get("Categories") or "",
            "primary_category": row.get("Primary Category") or "",
            "comment": row.get("Comment") or "",
            "journal_reference": row.get("Journal Reference") or "",
            "arxiv_url": row.get("arXiv URL") or "",
            "pdf_url": row.get("PDF URL") or "",
        })
    return deduped


def load_catalog(path: str | Path) -> list[dict[str, Any]]:
    if str(path).endswith(".csv"):
        return read_csv_catalog(path)
    return read_jsonl(path)


def existing_ids(path: str | Path) -> set[str]:
    if not Path(path).exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        paper_id = str(row.get("paper_id") or row.get("arxiv_id") or "").strip()
        if paper_id:
            ids.add(paper_id)
    return ids


def arxiv_base_id(paper_id: str) -> str:
    return re.sub(r"v\d+$", "", paper_id.strip())


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
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                return json.loads(repaired)
        raise


def ensure_openai_client():
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for census classification.") from exc
    return OpenAI()


def ensure_gemini_client():
    load_dotenv()
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise RuntimeError("The google-genai package is required for Gemini census classification.") from exc
    return genai.Client(), genai_types


def compact_paper(row: dict[str, Any], abstract_char_limit: int) -> dict[str, Any]:
    return {
        "paper_id": row.get("paper_id") or row.get("arxiv_id"),
        "title": row.get("title") or "",
        "published_date": row.get("published_date") or "",
        "year": row.get("year") or "",
        "venue_prefix": row.get("venue_prefix") or "",
        "event": row.get("event") or "",
        "booktitle": row.get("booktitle") or "",
        "journal": row.get("journal") or "",
        "categories": row.get("categories") or "",
        "primary_category": row.get("primary_category") or "",
        "comment": row.get("comment") or "",
        "journal_reference": row.get("journal_reference") or "",
        "abstract": str(row.get("abstract") or "")[:abstract_char_limit],
    }


def build_prompt(rows: Sequence[dict[str, Any]], abstract_char_limit: int) -> str:
    papers = [compact_paper(row, abstract_char_limit) for row in rows]
    return SYSTEM_PROMPT + "\nPapers:\n" + json.dumps(papers, ensure_ascii=False, indent=2)


def classify_batch_openai(client, rows: Sequence[dict[str, Any]], *, model: str, abstract_char_limit: int) -> list[dict[str, Any]]:
    prompt = build_prompt(rows, abstract_char_limit)
    response = client.responses.create(model=model.removeprefix("openai/"), input=prompt)
    payload = parse_json_object(response.output_text)
    return normalize_batch_outputs(rows, payload)


def classify_batch_gemini(client, genai_types, rows: Sequence[dict[str, Any]], *, model: str, abstract_char_limit: int) -> list[dict[str, Any]]:
    prompt = build_prompt(rows, abstract_char_limit)
    response = client.models.generate_content(
        model=model.removeprefix("gemini/"),
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    payload = parse_json_object(response.text or "")
    return normalize_batch_outputs(rows, payload)


def normalize_batch_outputs(rows: Sequence[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = payload.get("papers")
    if not isinstance(outputs, list):
        raise ValueError("Model response missing papers list")
    by_id = {str(row.get("paper_id")): row for row in rows}
    by_arxiv_base_id = {arxiv_base_id(paper_id): paper_id for paper_id in by_id}
    normalized = []
    seen_ids: set[str] = set()
    for output in outputs:
        paper_id = str(output.get("paper_id") or "").strip()
        if paper_id not in by_id:
            paper_id = by_arxiv_base_id.get(arxiv_base_id(paper_id), paper_id)
        if paper_id not in by_id:
            raise ValueError(f"Model returned unknown paper_id: {paper_id}")
        if paper_id in seen_ids:
            raise ValueError(f"Model returned duplicate paper_id: {paper_id}")
        seen_ids.add(paper_id)
        source = by_id[paper_id]
        normalized.append({
            "paper_id": paper_id,
            "title": source.get("title") or "",
            "published_date": source.get("published_date") or "",
            "year": source.get("year") or parse_year(source.get("published_date")),
            "source_type": "abstract_census",
            "categories": source.get("categories") or "",
            "primary_category": source.get("primary_category") or "",
            "venue_prefix": source.get("venue_prefix") or "",
            "event": source.get("event") or "",
            "booktitle": source.get("booktitle") or "",
            "journal": source.get("journal") or "",
            "is_nlp_paper": bool(output.get("is_nlp_paper")),
            "is_dataset_mentioned": bool(output.get("is_dataset_mentioned")),
            "is_dataset_introducing": bool(output.get("is_dataset_introducing")),
            "datasets": normalize_output_datasets(output.get("introduced_datasets") or []),
            "exclusion_reason": output.get("exclusion_reason") or "",
            "classified_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })
    missing_ids = sorted(set(by_id) - seen_ids)
    if missing_ids:
        raise ValueError(f"Model omitted paper_id(s): {', '.join(missing_ids)}")
    return normalized


def classify_batch(
    client_bundle,
    rows: Sequence[dict[str, Any]],
    *,
    backend: str,
    model: str,
    abstract_char_limit: int,
) -> list[dict[str, Any]]:
    if backend == "openai":
        return classify_batch_openai(client_bundle, rows, model=model, abstract_char_limit=abstract_char_limit)
    if backend == "gemini":
        client, genai_types = client_bundle
        return classify_batch_gemini(client, genai_types, rows, model=model, abstract_char_limit=abstract_char_limit)
    raise ValueError(f"Unknown backend: {backend}")


def normalize_output_datasets(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    datasets = []
    for item in items:
        if not isinstance(item, dict):
            continue
        datasets.append({
            "name": item.get("name") or "",
            "role": item.get("role") or "",
            "is_introduced": True,
            "source_dataset": item.get("source_dataset") or "None",
            "transformation_type": item.get("transformation_type") or "None",
            "usage_description": item.get("usage_description") or "",
            "acus": item.get("acus") if isinstance(item.get("acus"), list) else [],
            "confidence": item.get("confidence") or "Medium",
        })
    return datasets


def parse_year(value: Any) -> int | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").year
    except ValueError:
        return None


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(rows: Sequence[dict[str, Any]], batch_size: int) -> list[tuple[int, list[dict[str, Any]]]]:
    return [
        (start, list(rows[start:start + batch_size]))
        for start in range(0, len(rows), batch_size)
    ]


def error_row(batch_index: int, batch: Sequence[dict[str, Any]], exc: Exception) -> dict[str, Any]:
    return {
        "batch_index": batch_index,
        "paper_ids": [row.get("paper_id") or row.get("arxiv_id") for row in batch],
        "error": str(exc),
        "error_type": type(exc).__name__,
        "failed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run abstract-level dataset-introduction census classification.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--error-jsonl", default=None)
    parser.add_argument("--backend", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--abstract-char-limit", type=int, default=2200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_catalog(args.catalog)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit is not None:
        rows = rows[: args.limit]

    if not args.overwrite:
        done = existing_ids(args.output_jsonl)
        rows = [row for row in rows if str(row.get("paper_id") or row.get("arxiv_id") or "") not in done]

    print(json.dumps({
        "catalog": args.catalog,
        "output_jsonl": args.output_jsonl,
        "error_jsonl": args.error_jsonl,
        "backend": args.backend,
        "model": args.model,
        "remaining_rows": len(rows),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "dry_run": args.dry_run,
    }, indent=2))

    if args.dry_run:
        for row in rows[: min(len(rows), args.batch_size)]:
            print(json.dumps(compact_paper(row, args.abstract_char_limit), ensure_ascii=False))
        return

    if args.backend == "openai":
        client_bundle = ensure_openai_client()
    else:
        client_bundle = ensure_gemini_client()

    error_path = args.error_jsonl or str(Path(args.output_jsonl).with_suffix(".errors.jsonl"))
    processed = 0
    failed = 0
    batches = batched(rows, args.batch_size)

    def run_one(batch_index: int, batch: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None]:
        last_exc: Exception | None = None
        for attempt in range(args.max_retries + 1):
            try:
                return batch_index, classify_batch(
                    client_bundle,
                    batch,
                    backend=args.backend,
                    model=args.model,
                    abstract_char_limit=args.abstract_char_limit,
                ), None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < args.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        assert last_exc is not None
        return batch_index, [], error_row(batch_index, batch, last_exc)

    if args.workers <= 1:
        for batch_index, batch in batches:
            _, outputs, error = run_one(batch_index, batch)
            if outputs:
                append_jsonl(args.output_jsonl, outputs)
                processed += len(outputs)
            if error:
                append_jsonl(error_path, [error])
                failed += len(batch)
            print(json.dumps({"processed": processed, "failed": failed, "last_batch": len(outputs)}, indent=2))
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_one, batch_index, batch)
                for batch_index, batch in batches
            ]
            for future in as_completed(futures):
                batch_index, outputs, error = future.result()
                if outputs:
                    append_jsonl(args.output_jsonl, outputs)
                    processed += len(outputs)
                if error:
                    append_jsonl(error_path, [error])
                    failed += len(error.get("paper_ids") or [])
                print(json.dumps({
                    "processed": processed,
                    "failed": failed,
                    "completed_batch_index": batch_index,
                    "last_batch": len(outputs),
                }, indent=2))
                if args.sleep:
                    time.sleep(args.sleep)


if __name__ == "__main__":
    main()
