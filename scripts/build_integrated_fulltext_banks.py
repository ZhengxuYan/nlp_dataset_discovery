#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text == "unclear" else text


def list_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def format_list(values: Any, max_items: int = 12) -> str:
    output = []
    for value in list_values(values):
        if isinstance(value, dict):
            value = value.get("name") or value.get("dataset_name") or value.get("relationship") or json.dumps(value, sort_keys=True)
        text = clean_text(value)
        if text:
            output.append(text)
    return " ".join(output[:max_items])


def scale_text(scale: Any) -> str:
    if not isinstance(scale, dict):
        return clean_text(scale)
    return " ".join(
        clean_text(value)
        for value in [
            scale.get("size_text"),
            scale.get("num_instances"),
            scale.get("num_documents"),
            scale.get("num_tokens"),
            scale.get("num_languages"),
            scale.get("num_domains"),
        ]
        if clean_text(value)
    )


def dataset_name(dataset: dict[str, Any]) -> str:
    identity = dataset.get("dataset_identity") or {}
    return clean_text(identity.get("canonical_name") or dataset.get("dataset_name") or dataset.get("dataset_id"))


def flatten_dataset(
    paper: dict[str, Any],
    dataset: dict[str, Any],
    *,
    dataset_index: int,
    source_corpus: str,
) -> dict[str, Any]:
    identity = dataset.get("dataset_identity") or {}
    coverage = dataset.get("coverage") or {}
    construction = dataset.get("construction") or {}
    synthetic = construction.get("synthetic_generation") or {}
    scale = dataset.get("scale") or {}
    availability = dataset.get("availability") or {}
    governance = dataset.get("governance") or {}
    evaluation = dataset.get("evaluation") or {}
    bank_id = f"{paper.get('paper_id')}::dataset::{dataset_index}"
    source_datasets = list_values(construction.get("source_datasets"))
    row = {
        "bank_id": bank_id,
        "source_corpus": source_corpus,
        "paper_id": paper.get("paper_id") or "",
        "acl_id": paper.get("acl_id") or "",
        "doi": paper.get("doi") or "",
        "arxiv_id": paper.get("arxiv_id") or "",
        "title": paper.get("title") or "",
        "year": paper.get("year"),
        "venue_prefix": paper.get("venue_prefix") or "",
        "event": paper.get("event") or "",
        "anthology_url": paper.get("anthology_url") or "",
        "arxiv_url": paper.get("arxiv_url") or "",
        "pdf_url": paper.get("pdf_url") or "",
        "dataset_index": dataset_index,
        "dataset_id": dataset.get("dataset_id") or identity.get("canonical_name") or str(dataset_index),
        "dataset_name": dataset_name(dataset),
        "aliases": list_values(identity.get("aliases")),
        "acronym": identity.get("acronym") or "",
        "is_new_dataset": identity.get("is_new_dataset"),
        "role": dataset.get("role") or "",
        "resource_type": dataset.get("resource_type") or "",
        "primary_use": dataset.get("primary_use") or "",
        "is_reusable_resource": dataset.get("is_reusable_resource"),
        "usage_description": dataset.get("usage_description") or "",
        "tasks": list_values(coverage.get("tasks")),
        "domains": list_values(coverage.get("domains")),
        "languages": list_values(coverage.get("languages")),
        "modalities": list_values(coverage.get("modality") or coverage.get("modalities")),
        "genres": list_values(coverage.get("genres")),
        "unit_of_analysis": coverage.get("unit_of_analysis") or "",
        "input_output_format": coverage.get("input_output_format") or "",
        "label_space": coverage.get("label_space") or "",
        "source_data_origin": construction.get("source_data_origin") or "",
        "source_datasets": source_datasets,
        "collection_method": construction.get("collection_method") or "",
        "transformation_types": list_values(construction.get("transformation_types")),
        "annotation_protocol": construction.get("annotation_protocol") or "",
        "annotator_type": construction.get("annotator_type") or "",
        "quality_control": construction.get("quality_control") or "",
        "uses_llm_synthetic_generation": synthetic.get("uses_llm"),
        "synthetic_model_names": list_values(synthetic.get("model_names")),
        "synthetic_human_verification": synthetic.get("human_verification") or "",
        "scale": scale,
        "release_status": availability.get("release_status") or "",
        "license": availability.get("license") or "",
        "access_restrictions": availability.get("access_restrictions") or "",
        "documentation_type": availability.get("documentation_type") or "",
        "maintenance_status": availability.get("maintenance_status") or "",
        "artifacts": list_values(availability.get("artifacts")),
        "governance": governance,
        "used_for_training": evaluation.get("used_for_training"),
        "used_for_evaluation": evaluation.get("used_for_evaluation"),
        "benchmark_metrics": list_values(evaluation.get("benchmark_metrics")),
        "added_information_summary": dataset.get("added_information_summary") or "",
        "prior_dataset_mentions": list_values(dataset.get("prior_dataset_mentions")),
        "acus": list_values(dataset.get("acus")),
        "n_acus": len(list_values(dataset.get("acus"))),
        "confidence": (paper.get("extraction_quality") or {}).get("confidence") or dataset.get("confidence"),
        "ambiguities": list_values((paper.get("extraction_quality") or {}).get("ambiguities") or dataset.get("ambiguities")),
        "missing_information": list_values((paper.get("extraction_quality") or {}).get("missing_information") or dataset.get("missing_information")),
    }
    row["search_text"] = "\n".join(
        part
        for part in [
            clean_text(row["title"]),
            clean_text(row["dataset_name"]),
            clean_text(row["role"]),
            clean_text(row["resource_type"]),
            format_list(row["tasks"]),
            format_list(row["domains"]),
            format_list(row["languages"]),
            format_list(row["modalities"]),
            clean_text(row["usage_description"]),
            clean_text(row["added_information_summary"]),
            clean_text(row["source_data_origin"]),
            format_list(row["source_datasets"]),
            clean_text(row["annotation_protocol"]),
            scale_text(row["scale"]),
        ]
        if part
    )
    return row


def flatten_acu(dataset_row: dict[str, Any], acu: dict[str, Any]) -> dict[str, Any]:
    acu_id = clean_text(acu.get("id") or acu.get("acu_id") or f"a{dataset_row.get('n_acus', 0)}")
    acu_text = clean_text(acu.get("text") or acu.get("acu_text"))
    evidence = clean_text(acu.get("evidence"))
    row = {
        "acu_global_id": f"{dataset_row['bank_id']}::acu::{acu_id}",
        "bank_id": dataset_row["bank_id"],
        "source_corpus": dataset_row.get("source_corpus") or "",
        "paper_id": dataset_row.get("paper_id") or "",
        "title": dataset_row.get("title") or "",
        "year": dataset_row.get("year"),
        "dataset_id": dataset_row.get("dataset_id") or "",
        "dataset_name": dataset_row.get("dataset_name") or "",
        "dataset_role": dataset_row.get("role") or "",
        "resource_type": dataset_row.get("resource_type") or "",
        "acu_id": acu_id,
        "acu_text": acu_text,
        "acu_type": clean_text(acu.get("type") or acu.get("acu_type")),
        "importance": clean_text(acu.get("importance")),
        "evidence": evidence,
        "section": clean_text(acu.get("section")),
    }
    row["search_text"] = "\n".join(
        part
        for part in [
            clean_text(dataset_row.get("title")),
            clean_text(dataset_row.get("dataset_name")),
            clean_text(dataset_row.get("role")),
            clean_text(dataset_row.get("resource_type")),
            format_list(dataset_row.get("tasks")),
            format_list(dataset_row.get("domains")),
            format_list(dataset_row.get("languages")),
            format_list(dataset_row.get("modalities")),
            clean_text(dataset_row.get("added_information_summary")),
            clean_text(dataset_row.get("source_data_origin")),
            format_list(dataset_row.get("source_datasets")),
            clean_text(dataset_row.get("annotation_protocol")),
            scale_text(dataset_row.get("scale")),
            row["acu_type"],
            acu_text,
            evidence,
        ]
        if part
    )
    return row


def infer_source_corpus(path: str | Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = Path(path).name.lower()
    if "arxiv" in name:
        return "arxiv"
    if "acl" in name or "pdf_all" in name:
        return "acl"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build integrated fulltext dataset and ACU banks from extraction JSONL files.")
    parser.add_argument("--input-jsonl", action="append", required=True, help="Extraction JSONL. Can be passed multiple times.")
    parser.add_argument("--source-corpus", action="append", default=None, help="Optional corpus label for each --input-jsonl.")
    parser.add_argument("--output-extractions-jsonl", required=True)
    parser.add_argument("--output-dataset-bank-jsonl", required=True)
    parser.add_argument("--output-acu-bank-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--skip-empty-datasets", action="store_true")
    args = parser.parse_args()

    if args.source_corpus and len(args.source_corpus) != len(args.input_jsonl):
        raise SystemExit("--source-corpus must be supplied once per --input-jsonl when used.")

    for path in [args.output_extractions_jsonl, args.output_dataset_bank_jsonl, args.output_acu_bank_jsonl]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    by_source = Counter()
    years = Counter()
    dataset_years = Counter()
    acu_years = Counter()
    seen_papers = set()
    seen_bank_ids = set()
    seen_acu_ids = set()

    with (
        Path(args.output_extractions_jsonl).open("w", encoding="utf-8") as extraction_out,
        Path(args.output_dataset_bank_jsonl).open("w", encoding="utf-8") as dataset_out,
        Path(args.output_acu_bank_jsonl).open("w", encoding="utf-8") as acu_out,
    ):
        for index, input_path in enumerate(args.input_jsonl):
            source_corpus = infer_source_corpus(
                input_path,
                args.source_corpus[index] if args.source_corpus else None,
            )
            for paper in read_jsonl(input_path):
                stats["input_rows"] += 1
                paper = dict(paper)
                paper["source_corpus"] = source_corpus
                paper_id = str(paper.get("paper_id") or "")
                if paper_id in seen_papers:
                    stats["duplicate_papers_skipped"] += 1
                    continue
                seen_papers.add(paper_id)
                datasets = list_values(paper.get("datasets"))
                if args.skip_empty_datasets and not datasets:
                    stats["empty_dataset_papers_skipped"] += 1
                    continue
                append_jsonl(extraction_out, paper)
                stats["papers"] += 1
                by_source[source_corpus] += 1
                if paper.get("year") is not None:
                    years[str(paper.get("year"))] += 1
                for dataset_index, dataset in enumerate(datasets):
                    dataset_row = flatten_dataset(paper, dataset, dataset_index=dataset_index, source_corpus=source_corpus)
                    bank_id = str(dataset_row.get("bank_id") or "")
                    if bank_id in seen_bank_ids:
                        stats["duplicate_datasets_skipped"] += 1
                        continue
                    seen_bank_ids.add(bank_id)
                    append_jsonl(dataset_out, dataset_row)
                    stats["datasets"] += 1
                    if dataset_row.get("year") is not None:
                        dataset_years[str(dataset_row.get("year"))] += 1
                    for acu in dataset_row.get("acus") or []:
                        acu_row = flatten_acu(dataset_row, acu)
                        acu_global_id = str(acu_row.get("acu_global_id") or "")
                        if acu_global_id in seen_acu_ids:
                            stats["duplicate_acus_skipped"] += 1
                            continue
                        seen_acu_ids.add(acu_global_id)
                        append_jsonl(acu_out, acu_row)
                        stats["acus"] += 1
                        if acu_row.get("year") is not None:
                            acu_years[str(acu_row.get("year"))] += 1

    summary = {
        **dict(stats),
        "inputs": args.input_jsonl,
        "source_corpora": args.source_corpus,
        "papers_by_source": dict(by_source),
        "papers_by_year": dict(sorted(years.items())),
        "datasets_by_year": dict(sorted(dataset_years.items())),
        "acus_by_year": dict(sorted(acu_years.items())),
        "output_extractions_jsonl": args.output_extractions_jsonl,
        "output_dataset_bank_jsonl": args.output_dataset_bank_jsonl,
        "output_acu_bank_jsonl": args.output_acu_bank_jsonl,
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
