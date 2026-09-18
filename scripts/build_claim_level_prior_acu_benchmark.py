#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


SUPPORT_STATUSES = {"supported", "partially_supported", "unsupported", "not_comparable"}
DELTA_TYPES = {
    "task/domain",
    "data/source",
    "annotation/protocol",
    "scale/coverage",
    "evaluation/use",
    "availability/quality",
    "other",
}


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:24]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")
    return json.loads(match.group(0))


def ensure_openai_client():
    from openai import OpenAI

    return OpenAI()


def acu_id(acu: dict[str, Any], prefix: str, index: int) -> str:
    return str(acu.get("id") or f"{prefix}{index}")


def list_strings(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        text = str(item or "").strip()
        if text:
            output.append(text)
    return output[:limit]


def dataset_name(dataset: dict[str, Any]) -> str:
    identity = dataset.get("dataset_identity") or {}
    return str(identity.get("canonical_name") or dataset.get("dataset_id") or dataset.get("name") or "").strip()


def prior_extraction_payloads(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payloads = {}
    for row in read_jsonl(path):
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            payloads[paper_id] = row
    return payloads


def source_dataset_names(dataset: dict[str, Any], limit: int = 8) -> list[str]:
    construction = dataset.get("construction") or {}
    output = []
    for source in construction.get("source_datasets") or []:
        if isinstance(source, dict) and source.get("name"):
            output.append(str(source["name"]))
    return output[:limit]


def scale_summary(dataset: dict[str, Any]) -> str:
    scale = dataset.get("scale") or {}
    parts = []
    if scale.get("size_text"):
        parts.append(str(scale["size_text"]))
    for key in ["num_instances", "num_tokens", "num_documents", "num_dialogues", "num_images", "num_audio_hours", "num_languages", "num_domains"]:
        value = scale.get(key)
        if value not in (None, "", 0):
            parts.append(f"{key}={value}")
    return "; ".join(parts[:8])


def compact_dataset_metadata(dataset: dict[str, Any]) -> dict[str, Any]:
    coverage = dataset.get("coverage") or {}
    construction = dataset.get("construction") or {}
    synthetic = construction.get("synthetic_generation") or {}
    evaluation = dataset.get("evaluation") or {}
    availability = dataset.get("availability") or {}
    return {
        "dataset_name": dataset_name(dataset),
        "role": str(dataset.get("role") or ""),
        "resource_type": str(dataset.get("resource_type") or ""),
        "primary_use": str(dataset.get("primary_use") or ""),
        "tasks": list_strings(coverage.get("tasks")),
        "domains": list_strings(coverage.get("domains")),
        "languages": list_strings(coverage.get("languages")),
        "modalities": list_strings(coverage.get("modality")),
        "source_data_origin": str(construction.get("source_data_origin") or ""),
        "collection_method": str(construction.get("collection_method") or ""),
        "source_datasets": source_dataset_names(dataset),
        "transformation_types": list_strings(construction.get("transformation_types")),
        "annotation_protocol": str(construction.get("annotation_protocol") or ""),
        "annotator_type": str(construction.get("annotator_type") or ""),
        "quality_control": str(construction.get("quality_control") or ""),
        "uses_llm": synthetic.get("uses_llm"),
        "llm_models": list_strings(synthetic.get("model_names")),
        "scale": scale_summary(dataset),
        "used_for_training": evaluation.get("used_for_training"),
        "used_for_evaluation": evaluation.get("used_for_evaluation"),
        "metrics": list_strings(evaluation.get("benchmark_metrics")),
        "release_status": str(availability.get("release_status") or ""),
        "license": str(availability.get("license") or ""),
    }


def query_metadata(row: dict[str, Any]) -> dict[str, Any]:
    coverage = row.get("coverage") or {}
    construction = row.get("construction") or {}
    synthetic = construction.get("synthetic_generation") or {}
    return {
        "paper_title": row.get("query_title") or row.get("query_paper_id"),
        "dataset_name": row.get("query_dataset_name"),
        "role": row.get("query_dataset_role") or "",
        "resource_type": row.get("query_dataset_resource_type") or "",
        "tasks": list_strings(coverage.get("tasks")),
        "domains": list_strings(coverage.get("domains")),
        "languages": list_strings(coverage.get("languages")),
        "modalities": list_strings(coverage.get("modality")),
        "source_data_origin": str(construction.get("source_data_origin") or ""),
        "collection_method": str(construction.get("collection_method") or ""),
        "source_datasets": source_dataset_names({"construction": construction}),
        "transformation_types": list_strings(construction.get("transformation_types")),
        "annotation_protocol": str(construction.get("annotation_protocol") or ""),
        "annotator_type": str(construction.get("annotator_type") or ""),
        "quality_control": str(construction.get("quality_control") or ""),
        "uses_llm": synthetic.get("uses_llm"),
        "llm_models": list_strings(synthetic.get("model_names")),
    }


def find_prior_dataset_metadata(acu: dict[str, Any], prior_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paper_id = str(acu.get("prior_paper_id") or "")
    payload = prior_payloads.get(paper_id) or {}
    datasets = payload.get("datasets") or []
    if not datasets:
        return {}
    prior_dataset_id = str(acu.get("prior_dataset_id") or "").lower()
    prior_dataset_name = str(acu.get("prior_dataset_name") or "").lower()
    best = None
    for dataset in datasets:
        name = dataset_name(dataset).lower()
        dataset_id = str(dataset.get("dataset_id") or "").lower()
        if prior_dataset_id and prior_dataset_id == dataset_id:
            return compact_dataset_metadata(dataset)
        if prior_dataset_name and (prior_dataset_name == name or prior_dataset_name in name or name in prior_dataset_name):
            best = dataset
    return compact_dataset_metadata(best or datasets[0])


def normalize_query_acus(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for index, acu in enumerate(row.get("query_acus") or []):
        if not isinstance(acu, dict):
            acu = {"text": str(acu)}
        text = str(acu.get("text") or "").strip()
        if not text:
            continue
        output.append({
            "id": acu_id(acu, "q", index),
            "text": text,
            "type": str(acu.get("type") or "other"),
            "importance": str(acu.get("importance") or "medium"),
            "evidence": str(acu.get("evidence") or ""),
            "section": str(acu.get("section") or ""),
        })
    return output


def normalize_prior_acus(row: dict[str, Any], prior_payloads: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    prior_payloads = prior_payloads or {}
    output = []
    for index, acu in enumerate(row.get("gold_prior_support_acus") or []):
        if not isinstance(acu, dict):
            acu = {"text": str(acu)}
        text = str(acu.get("text") or "").strip()
        if not text:
            continue
        prior = {
            "id": acu_id(acu, "p", index),
            "text": text,
            "type": str(acu.get("type") or "other"),
            "importance": str(acu.get("importance") or "medium"),
            "prior_paper_id": str(acu.get("prior_paper_id") or ""),
            "prior_paper_title": str(acu.get("prior_paper_title") or ""),
            "prior_dataset_name": str(acu.get("prior_dataset_name") or ""),
            "prior_dataset_id": str(acu.get("prior_dataset_id") or ""),
            "evidence": str(acu.get("evidence") or ""),
            "section": str(acu.get("section") or ""),
        }
        prior["metadata"] = find_prior_dataset_metadata(prior, prior_payloads)
        output.append(prior)
    return output


def metadata_lines(metadata: dict[str, Any], *, indent: str = "    ") -> list[str]:
    lines = []
    for key, value in metadata.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value_text = "; ".join(str(item) for item in value)
        else:
            value_text = str(value)
        lines.append(f"{indent}{key}: {value_text}")
    return lines


def format_acus(acus: list[dict[str, Any]], *, shared_metadata: dict[str, Any] | None = None) -> str:
    lines = []
    if shared_metadata:
        lines.append("Shared metadata:")
        lines.extend(metadata_lines(shared_metadata, indent="  "))
    for acu in acus:
        meta = []
        if acu.get("type"):
            meta.append(f"type={acu['type']}")
        if acu.get("prior_dataset_name"):
            meta.append(f"dataset={acu['prior_dataset_name']}")
        if acu.get("prior_paper_title"):
            meta.append(f"paper={acu['prior_paper_title']}")
        suffix = f" [{' ; '.join(meta)}]" if meta else ""
        lines.append(f"- {acu['id']}{suffix}: {acu['text']}")
        if acu.get("metadata"):
            lines.extend(metadata_lines(acu["metadata"]))
    return "\n".join(lines) or "- none"


def build_prompt(
    row: dict[str, Any],
    query_acus: list[dict[str, Any]],
    prior_acus: list[dict[str, Any]],
    *,
    query_meta: dict[str, Any],
) -> str:
    return f"""You are constructing claim-level gold evidence labels for an NLP dataset prior-support benchmark.

For each query ACU, choose the subset of prior ACUs that provide useful prior evidence for assessing that claim.
This is not strict textual entailment. A prior ACU can be selected when it is a meaningful comparator for the same dataset-contribution dimension, even if the query claim adds a different number, dataset name, language, source, annotation protocol, or evaluation setting.

Important rules:
- Select prior ACUs only from the supplied prior ACU bank.
- A selected prior ACU must be evidence that helps compare the query claim against prior dataset work, not merely a shared broad topic.
- For scale/coverage claims, prior ACUs giving comparable dataset size, language/domain coverage, or modality coverage should usually be selected as partially_supported, even if the exact numbers differ.
- For task/domain claims, prior ACUs describing the same or closely related benchmark task/domain should usually be selected as partially_supported, even if the query dataset introduces a new variant.
- For data/source claims, prior ACUs about the same source dataset, dataset family, data origin, or closely related construction source should usually be selected.
- For annotation/protocol claims, prior ACUs about comparable annotation, verification, labeling, or quality-control protocols should usually be selected.
- For evaluation/use claims, prior ACUs about comparable benchmark use, metrics, baselines, or evaluation setup should usually be selected.
- Return an empty selected_prior_acu_ids list only when the supplied prior ACU bank has no meaningful evidence for the query ACU's contribution dimension.
- Empty sets are allowed and often correct because some query ACUs are entirely new added information.
- Use support_status:
  - supported: the selected prior ACUs already cover the query claim or nearly the same claim.
  - partially_supported: selected prior ACUs cover the same contribution dimension or comparable prior evidence, but the query adds meaningful new information.
  - unsupported: no supplied prior ACU supports the claim.
  - not_comparable: the query claim cannot be compared to the prior ACU bank.
- If selected_prior_acu_ids is empty, support_status should usually be unsupported or not_comparable.
- Prefer partially_supported over unsupported when there is a same-dimension prior ACU that would be useful as evidence in a prior-work comparison.
- Keep rationales brief and evidence-grounded.

Return JSON only:
{{
  "labels": [
    {{
      "query_acu_id": "q0",
      "support_status": "supported|partially_supported|unsupported|not_comparable",
      "selected_prior_acu_ids": ["p0"],
      "delta_type": "task/domain|data/source|annotation/protocol|scale/coverage|evaluation/use|availability/quality|other",
      "rationale": "brief reason"
    }}
  ]
}}

Query paper: {row.get("query_title") or row.get("query_paper_id")}
Query dataset: {row.get("query_dataset_name")}

Query DCU cards:
{format_acus(query_acus, shared_metadata=query_meta)}

Gold prior DCU/ACU bank:
{format_acus(prior_acus)}
"""


def normalize_labels(payload: dict[str, Any], query_acus: list[dict[str, Any]], prior_acus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query_ids = {acu["id"] for acu in query_acus}
    prior_ids = {acu["id"] for acu in prior_acus}
    by_query = {str(label.get("query_acu_id") or ""): label for label in payload.get("labels") or [] if isinstance(label, dict)}
    labels = []
    for query_acu in query_acus:
        raw = by_query.get(query_acu["id"]) or {}
        selected = [
            str(prior_id)
            for prior_id in raw.get("selected_prior_acu_ids") or []
            if str(prior_id) in prior_ids
        ]
        status = str(raw.get("support_status") or "").strip()
        if status not in SUPPORT_STATUSES:
            status = "partially_supported" if selected else "unsupported"
        if not selected and status in {"supported", "partially_supported"}:
            status = "unsupported"
        delta_type = str(raw.get("delta_type") or query_acu.get("type") or "other")
        if delta_type not in DELTA_TYPES:
            delta_type = "other"
        labels.append({
            "query_acu_id": query_acu["id"],
            "support_status": status,
            "selected_prior_acu_ids": selected,
            "delta_type": delta_type,
            "rationale": str(raw.get("rationale") or "").strip(),
            "evaluate": bool(selected) and status in {"supported", "partially_supported"},
        })
    missing = query_ids - {label["query_acu_id"] for label in labels}
    if missing:
        raise ValueError(f"Missing normalized labels for query ACUs: {sorted(missing)}")
    return labels


def cache_path(cache_dir: str | Path, key: dict[str, Any]) -> Path:
    path = Path(cache_dir) / "claim_level_prior_acu_labels"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{stable_hash(key)}.json"


def load_completed_output(path: str | Path | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    completed = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            benchmark_id = row.get("benchmark_id")
            if benchmark_id:
                completed.add(str(benchmark_id))
    return completed


def label_row(
    row: dict[str, Any],
    *,
    model: str,
    cache_dir: str,
    overwrite_cache: bool,
    prior_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    benchmark_id = str(row.get("benchmark_id") or row.get("query_dataset_id") or row.get("query_paper_id"))
    query_acus = normalize_query_acus(row)
    prior_acus = normalize_prior_acus(row, prior_payloads)
    query_meta = query_metadata(row)
    if not query_acus or not prior_acus:
        raise ValueError("missing query_acus or gold_prior_support_acus")

    key = {
        "benchmark_id": benchmark_id,
        "query_acus": query_acus,
        "prior_acus": prior_acus,
        "query_metadata": query_meta,
        "model": model,
        "schema": "claim_level_prior_acu_selection_v3_dcu_metadata",
    }
    path = cache_path(cache_dir, key)
    if path.exists() and not overwrite_cache:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        client = ensure_openai_client()
        prompt = build_prompt(row, query_acus, prior_acus, query_meta=query_meta)
        response = client.responses.create(model=model, input=prompt)
        payload = parse_json_object(response.output_text)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    labels = normalize_labels(payload, query_acus, prior_acus)
    prior_by_id = {acu["id"]: acu for acu in prior_acus}
    query_by_id = {acu["id"]: acu for acu in query_acus}
    evaluable_labels = [label for label in labels if label["evaluate"]]
    return {
        "benchmark_id": benchmark_id,
        "query_paper_id": row.get("query_paper_id"),
        "query_title": row.get("query_title"),
        "query_dataset_name": row.get("query_dataset_name"),
        "gold_prior_paper_ids": row.get("gold_prior_paper_ids") or [],
        "query_metadata": query_meta,
        "query_acus": query_acus,
        "prior_acu_bank": prior_acus,
        "labels": labels,
        "evaluable_claim_labels": evaluable_labels,
        "evaluable_query_acu_ids": [label["query_acu_id"] for label in evaluable_labels],
        "selected_prior_acu_ids": sorted({
            prior_id
            for label in evaluable_labels
            for prior_id in label["selected_prior_acu_ids"]
        }),
        "query_acu_by_id": query_by_id,
        "prior_acu_by_id": prior_by_id,
        "model": model,
        "cache_path": str(path),
    }


def summarize(rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter()
    delta_counts = Counter()
    query_acus = 0
    evaluable_query_acus = 0
    selected_prior_counts = []
    for row in rows:
        for label in row["labels"]:
            query_acus += 1
            label_counts[label["support_status"]] += 1
            delta_counts[label["delta_type"]] += 1
            if label["evaluate"]:
                evaluable_query_acus += 1
                selected_prior_counts.append(len(label["selected_prior_acu_ids"]))
    return {
        "rows": len(rows),
        "errors": len(errors),
        "query_acus": query_acus,
        "evaluable_query_acus": evaluable_query_acus,
        "evaluable_query_acu_rate": evaluable_query_acus / query_acus if query_acus else 0.0,
        "support_status_counts": dict(label_counts),
        "delta_type_counts": dict(delta_counts),
        "mean_selected_prior_acus_per_evaluable_query_acu": (
            sum(selected_prior_counts) / len(selected_prior_counts) if selected_prior_counts else 0.0
        ),
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Claim-Level Prior ACU Benchmark",
        "",
        f"- Rows: {summary['rows']}",
        f"- Errors: {summary['errors']}",
        f"- Query ACUs: {summary['query_acus']}",
        f"- Evaluable query ACUs: {summary['evaluable_query_acus']} ({summary['evaluable_query_acu_rate']:.3f})",
        f"- Mean selected prior ACUs per evaluable query ACU: {summary['mean_selected_prior_acus_per_evaluable_query_acu']:.2f}",
        "",
        "## Support Status",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(summary["support_status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Delta Type", "", "| Type | Count |", "| --- | ---: |"])
    for delta_type, count in sorted(summary["delta_type_counts"].items()):
        lines.append(f"| {delta_type} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build claim-level gold prior-ACU evidence labels from citation-grounded gold prior ACU banks.")
    parser.add_argument("--acl-benchmark-jsonl", required=True)
    parser.add_argument("--prior-extractions-jsonl", default=None, help="Optional prior extraction JSONL used to enrich prior ACU cards with DCU metadata.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--evaluable-output-jsonl", default=None)
    parser.add_argument("--error-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-markdown", default=None)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--cache-dir", default="data/benchmark/retrieval_cache")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = [
        row for row in read_jsonl(args.acl_benchmark_jsonl)
        if normalize_query_acus(row) and normalize_prior_acus(row)
    ]
    if args.limit is not None:
        rows = rows[:args.limit]
    completed = set() if args.overwrite_output else load_completed_output(args.output_jsonl)
    rows = [
        row for row in rows
        if str(row.get("benchmark_id") or row.get("query_dataset_id") or row.get("query_paper_id")) not in completed
    ]

    if args.overwrite_output:
        for path in [args.output_jsonl, args.error_jsonl, args.evaluable_output_jsonl]:
            if path:
                Path(path).unlink(missing_ok=True)

    print(json.dumps({
        "input_rows_to_process": len(rows),
        "already_completed_rows": len(completed),
        "model": args.model,
        "prior_extractions_jsonl": args.prior_extractions_jsonl,
        "output_jsonl": args.output_jsonl,
        "evaluable_output_jsonl": args.evaluable_output_jsonl,
        "dry_run": args.dry_run,
    }, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.monotonic()
    prior_payloads = prior_extraction_payloads(args.prior_extractions_jsonl)

    def process(row: dict[str, Any]) -> dict[str, Any]:
        return label_row(
            row,
            model=args.model,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            prior_payloads=prior_payloads,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(process, row): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            benchmark_id = str(row.get("benchmark_id") or row.get("query_dataset_id") or row.get("query_paper_id"))
            try:
                result = future.result()
                result_rows.append(result)
                append_jsonl(args.output_jsonl, result)
                if args.evaluable_output_jsonl:
                    for label in result["evaluable_claim_labels"]:
                        append_jsonl(args.evaluable_output_jsonl, {
                            "benchmark_id": result["benchmark_id"],
                            "query_paper_id": result["query_paper_id"],
                            "query_dataset_name": result["query_dataset_name"],
                            "query_acu": result["query_acu_by_id"][label["query_acu_id"]],
                            "label": label,
                            "selected_prior_acus": [
                                result["prior_acu_by_id"][prior_id]
                                for prior_id in label["selected_prior_acu_ids"]
                                if prior_id in result["prior_acu_by_id"]
                            ],
                            "gold_prior_paper_ids": result["gold_prior_paper_ids"],
                        })
            except Exception as exc:
                error = {"benchmark_id": benchmark_id, "error": str(exc)}
                errors.append(error)
                append_jsonl(args.error_jsonl, error)
            elapsed = time.monotonic() - started
            print(json.dumps({
                "processed": index,
                "total": len(rows),
                "failed": len(errors),
                "elapsed_seconds": round(elapsed, 1),
                "avg_seconds_per_row": round(elapsed / index, 2),
            }, ensure_ascii=False), flush=True)

    all_output_rows = []
    if Path(args.output_jsonl).exists():
        all_output_rows = read_jsonl(args.output_jsonl)
    all_errors = []
    if Path(args.error_jsonl).exists():
        all_errors = read_jsonl(args.error_jsonl)
    summary = summarize(all_output_rows, all_errors)
    summary.update({
        "input_jsonl": args.acl_benchmark_jsonl,
        "prior_extractions_jsonl": args.prior_extractions_jsonl,
        "output_jsonl": args.output_jsonl,
        "evaluable_output_jsonl": args.evaluable_output_jsonl,
        "error_jsonl": args.error_jsonl,
        "model": args.model,
    })
    write_json(args.summary_json, summary)
    if args.summary_markdown:
        Path(args.summary_markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_markdown).write_text(markdown_summary(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
