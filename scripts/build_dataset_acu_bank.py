#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_id(*parts: Any) -> str:
    text = "\n".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def clean_list(values: Any) -> list[str]:
    cleaned = []
    for value in as_list(values):
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def dataset_name(dataset: dict[str, Any]) -> str:
    identity = as_dict(dataset.get("dataset_identity"))
    return (
        str(identity.get("canonical_name") or "").strip()
        or str(dataset.get("dataset_id") or "").strip()
        or "unknown_dataset"
    )


def artifact_links(dataset: dict[str, Any]) -> dict[str, list[str]]:
    availability = as_dict(dataset.get("availability"))
    artifacts = as_dict(availability.get("artifacts"))
    return {
        key: clean_list(artifacts.get(key))
        for key in [
            "dataset_urls",
            "project_page_urls",
            "code_urls",
            "huggingface_ids",
            "github_repos",
            "zenodo_urls",
            "osf_urls",
            "kaggle_urls",
            "paperswithcode_urls",
            "other_urls",
        ]
    }


def prior_mentions(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    mentions = []
    for mention in as_list(dataset.get("prior_dataset_mentions")):
        if not isinstance(mention, dict):
            continue
        mentions.append({
            "name": mention.get("name") or "",
            "relationship_type": mention.get("relationship_type") or "unknown",
            "cited_paper_title": mention.get("cited_paper_title") or "",
            "cited_paper_id": mention.get("cited_paper_id") or "",
            "evidence": mention.get("evidence") or "",
            "prior_support_acus": clean_list(mention.get("prior_support_acus")),
        })
    return mentions


def source_datasets(dataset: dict[str, Any]) -> list[dict[str, str]]:
    construction = as_dict(dataset.get("construction"))
    sources = []
    for source in as_list(construction.get("source_datasets")):
        if isinstance(source, dict):
            sources.append({
                "name": str(source.get("name") or "").strip(),
                "relationship": str(source.get("relationship") or "unknown").strip(),
                "evidence": str(source.get("evidence") or "").strip(),
            })
        else:
            sources.append({
                "name": str(source).strip(),
                "relationship": "unknown",
                "evidence": "",
            })
    return [source for source in sources if source["name"]]


def search_text_for_dataset(row: dict[str, Any], dataset: dict[str, Any]) -> str:
    coverage = as_dict(dataset.get("coverage"))
    construction = as_dict(dataset.get("construction"))
    scale = as_dict(dataset.get("scale"))
    availability = as_dict(dataset.get("availability"))
    governance = as_dict(dataset.get("governance"))
    evaluation = as_dict(dataset.get("evaluation"))
    identity = as_dict(dataset.get("dataset_identity"))
    synthetic = as_dict(construction.get("synthetic_generation"))
    parts = [
        row.get("title"),
        row.get("abstract"),
        dataset_name(dataset),
        " ".join(clean_list(identity.get("aliases"))),
        identity.get("acronym"),
        dataset.get("role"),
        dataset.get("resource_type"),
        dataset.get("primary_use"),
        dataset.get("usage_description"),
        " ".join(clean_list(coverage.get("tasks"))),
        " ".join(clean_list(coverage.get("domains"))),
        " ".join(clean_list(coverage.get("languages"))),
        " ".join(clean_list(coverage.get("modality"))),
        coverage.get("genre"),
        coverage.get("unit_of_analysis"),
        coverage.get("input_output_format"),
        coverage.get("label_space"),
        construction.get("source_data_origin"),
        construction.get("collection_method"),
        " ".join(clean_list(construction.get("transformation_types"))),
        construction.get("annotation_protocol"),
        construction.get("annotator_type"),
        construction.get("quality_control"),
        " ".join(clean_list(synthetic.get("model_names"))),
        synthetic.get("human_verification"),
        scale.get("size_text"),
        availability.get("release_status"),
        availability.get("license"),
        availability.get("access_restrictions"),
        availability.get("documentation_type"),
        availability.get("maintenance_status"),
        " ".join(governance.get("known_limitations") or []),
        " ".join(clean_list(evaluation.get("benchmark_metrics"))),
        " ".join(clean_list(evaluation.get("baseline_models"))),
        " ".join(clean_list(evaluation.get("compared_datasets"))),
        evaluation.get("reported_improvement"),
        evaluation.get("human_evaluation"),
        evaluation.get("ablation_or_data_study"),
        dataset.get("added_information_summary"),
        " ".join(acu.get("text", "") for acu in as_list(dataset.get("acus")) if isinstance(acu, dict)),
        " ".join(mention.get("name", "") for mention in prior_mentions(dataset)),
        " ".join(source.get("name", "") for source in source_datasets(dataset)),
    ]
    return "\n".join(str(part).strip() for part in parts if str(part or "").strip())


def dataset_record(row: dict[str, Any], dataset: dict[str, Any], dataset_index: int) -> dict[str, Any]:
    paper_id = row.get("paper_id") or ""
    name = dataset_name(dataset)
    dataset_id = dataset.get("dataset_id") or stable_id(paper_id, dataset_index, name)
    identity = as_dict(dataset.get("dataset_identity"))
    coverage = as_dict(dataset.get("coverage"))
    construction = as_dict(dataset.get("construction"))
    availability = as_dict(dataset.get("availability"))
    governance = as_dict(dataset.get("governance"))
    scale = as_dict(dataset.get("scale"))
    synthetic = as_dict(construction.get("synthetic_generation"))
    bank_id = f"{paper_id}::dataset::{dataset_index}"
    acus = [
        {
            "acu_id": acu.get("id") or f"q{acu_index}",
            "text": acu.get("text") or "",
            "type": acu.get("type") or "unknown",
            "importance": acu.get("importance") or "medium",
            "evidence": acu.get("evidence") or "",
            "section": acu.get("section") or "unknown",
        }
        for acu_index, acu in enumerate(as_list(dataset.get("acus")))
        if isinstance(acu, dict)
    ]
    return {
        "bank_id": bank_id,
        "paper_id": paper_id,
        "acl_id": row.get("acl_id") or "",
        "title": row.get("title") or "",
        "year": row.get("year"),
        "venue_prefix": row.get("venue_prefix") or "",
        "event": row.get("event") or "",
        "anthology_url": row.get("anthology_url") or "",
        "pdf_url": row.get("pdf_url") or "",
        "dataset_index": dataset_index,
        "dataset_id": dataset_id,
        "dataset_name": name,
        "aliases": clean_list(identity.get("aliases")),
        "acronym": identity.get("acronym") or "",
        "is_new_dataset": identity.get("is_new_dataset"),
        "role": dataset.get("role") or "unknown",
        "resource_type": dataset.get("resource_type") or "unknown",
        "primary_use": dataset.get("primary_use") or "unknown",
        "is_reusable_resource": dataset.get("is_reusable_resource"),
        "usage_description": dataset.get("usage_description") or "",
        "tasks": clean_list(coverage.get("tasks")),
        "domains": clean_list(coverage.get("domains")),
        "languages": clean_list(coverage.get("languages")),
        "modalities": clean_list(coverage.get("modality")),
        "genres": clean_list(coverage.get("genre")),
        "unit_of_analysis": coverage.get("unit_of_analysis") or "",
        "input_output_format": coverage.get("input_output_format") or "",
        "label_space": coverage.get("label_space") or "",
        "source_data_origin": construction.get("source_data_origin") or "",
        "source_datasets": source_datasets(dataset),
        "collection_method": construction.get("collection_method") or "",
        "transformation_types": clean_list(construction.get("transformation_types")),
        "annotation_protocol": construction.get("annotation_protocol") or "",
        "annotator_type": construction.get("annotator_type") or "",
        "quality_control": construction.get("quality_control") or "",
        "uses_llm_synthetic_generation": synthetic.get("uses_llm"),
        "synthetic_model_names": clean_list(synthetic.get("model_names")),
        "synthetic_human_verification": synthetic.get("human_verification") or "",
        "scale": {
            "size_text": scale.get("size_text") or "",
            "num_instances": scale.get("num_instances"),
            "num_tokens": scale.get("num_tokens"),
            "num_documents": scale.get("num_documents"),
            "num_dialogues": scale.get("num_dialogues"),
            "num_images": scale.get("num_images"),
            "num_audio_hours": scale.get("num_audio_hours"),
            "num_languages": scale.get("num_languages"),
            "num_domains": scale.get("num_domains"),
        },
        "release_status": availability.get("release_status") or "unknown",
        "license": availability.get("license") or "unknown",
        "access_restrictions": availability.get("access_restrictions") or "unknown",
        "documentation_type": availability.get("documentation_type") or "unknown",
        "maintenance_status": availability.get("maintenance_status") or "unknown",
        "artifacts": artifact_links(dataset),
        "governance": {
            "ethics_discussed": governance.get("ethics_discussed") or "unknown",
            "pii_discussed": governance.get("pii_discussed") or "unknown",
            "consent_discussed": governance.get("consent_discussed") or "unknown",
            "copyright_discussed": governance.get("copyright_discussed") or "unknown",
            "bias_or_fairness_discussed": governance.get("bias_or_fairness_discussed") or "unknown",
            "known_limitations": clean_list(governance.get("known_limitations")),
        },
        "added_information_summary": dataset.get("added_information_summary") or "",
        "prior_dataset_mentions": prior_mentions(dataset),
        "acus": acus,
        "n_acus": len(acus),
        "confidence": dataset.get("confidence") or "unknown",
        "ambiguities": clean_list(dataset.get("ambiguities")),
        "missing_information": clean_list(dataset.get("missing_information")),
        "search_text": search_text_for_dataset(row, dataset),
    }


def acu_records(dataset_row: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for index, acu in enumerate(dataset_row["acus"]):
        acu_global_id = f"{dataset_row['bank_id']}::acu::{acu['acu_id'] or index}"
        search_text = "\n".join(part for part in [
            dataset_row["title"],
            dataset_row["dataset_name"],
            dataset_row["role"],
            dataset_row["resource_type"],
            " ".join(dataset_row["tasks"]),
            " ".join(dataset_row["domains"]),
            " ".join(dataset_row["languages"]),
            " ".join(dataset_row["modalities"]),
            dataset_row["added_information_summary"],
            acu["text"],
            acu["evidence"],
        ] if part)
        records.append({
            "acu_global_id": acu_global_id,
            "bank_id": dataset_row["bank_id"],
            "paper_id": dataset_row["paper_id"],
            "title": dataset_row["title"],
            "year": dataset_row["year"],
            "dataset_id": dataset_row["dataset_id"],
            "dataset_name": dataset_row["dataset_name"],
            "dataset_role": dataset_row["role"],
            "resource_type": dataset_row["resource_type"],
            "acu_id": acu["acu_id"],
            "acu_text": acu["text"],
            "acu_type": acu["type"],
            "importance": acu["importance"],
            "evidence": acu["evidence"],
            "section": acu["section"],
            "search_text": search_text,
        })
    return records


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized dataset and ACU bank from full-text extraction JSONL.")
    parser.add_argument("jsonl")
    parser.add_argument("--output-jsonl", required=True, help="Dataset-level bank JSONL.")
    parser.add_argument("--acu-output-jsonl", required=True, help="ACU-level bank JSONL.")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--min-acus", type=int, default=0)
    parser.add_argument("--exclude-false-positive", action="store_true")
    args = parser.parse_args()

    dataset_rows: list[dict[str, Any]] = []
    acu_rows: list[dict[str, Any]] = []
    input_papers = 0
    skipped_datasets = 0
    for row in read_jsonl(args.jsonl):
        input_papers += 1
        quality = as_dict(row.get("extraction_quality"))
        if args.exclude_false_positive and quality.get("possible_false_positive_dataset_paper"):
            continue
        for dataset_index, dataset in enumerate(as_list(row.get("datasets"))):
            if not isinstance(dataset, dict):
                continue
            record = dataset_record(row, dataset, dataset_index)
            if record["n_acus"] < args.min_acus:
                skipped_datasets += 1
                continue
            dataset_rows.append(record)
            acu_rows.extend(acu_records(record))

    dataset_rows.sort(key=lambda row: (str(row.get("year") or ""), row.get("paper_id") or "", row.get("dataset_index") or 0))
    acu_rows.sort(key=lambda row: (str(row.get("year") or ""), row.get("paper_id") or "", row.get("acu_global_id") or ""))
    write_jsonl(args.output_jsonl, dataset_rows)
    write_jsonl(args.acu_output_jsonl, acu_rows)
    summary = {
        "input_papers": input_papers,
        "dataset_rows": len(dataset_rows),
        "acu_rows": len(acu_rows),
        "skipped_datasets": skipped_datasets,
        "min_acus": args.min_acus,
        "output_jsonl": args.output_jsonl,
        "acu_output_jsonl": args.acu_output_jsonl,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
