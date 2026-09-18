#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ATTRIBUTION_JSONL = [
    "data/census/integrated_dcu_native_attribution_2024_gemini31_flashlite_top50_datasetcompact_v5.jsonl",
    "data/census/integrated_dcu_native_attribution_2025_gemini31_flashlite_top50_datasetcompact_v5.jsonl",
]
DEFAULT_DATASET_BANK = "data/census/integrated_fulltext_dataset_bank_2023_2025.jsonl"
DEFAULT_ACU_BANK = "data/census/integrated_fulltext_acu_bank_2023_2025.jsonl"
DEFAULT_BENCHMARK_LABELS = "data/benchmark/claim_level_prior_acu_labels_pdf130.jsonl"
DEFAULT_HARD_NEGATIVES = "data/benchmark/claim_level_dcu_hard_negatives_pdf130_same_gold_paper.jsonl"
DEFAULT_OUTPUT_DIR = "data/human_validation"


ANNOTATION_FIELDS = {
    "attribution": [
        "human_coverage_label_correct",
        "human_corrected_coverage_label",
        "human_selected_evidence_relevance",
        "human_rationale_groundedness",
        "human_evidence_sufficient",
        "human_missing_prior_risk",
        "human_notes",
        "annotator_id",
    ],
    "extraction": [
        "human_record_valid",
        "human_role_correct",
        "human_dcu_grounded",
        "human_dcu_type_correct",
        "human_metadata_errors",
        "human_notes",
        "annotator_id",
    ],
    "retrieval_label": [
        "human_gold_prior_validity",
        "human_hard_negative_validity",
        "human_empty_gold_validity",
        "human_corrected_label",
        "human_notes",
        "annotator_id",
    ],
    "missing_prior": [
        "human_external_search_done",
        "human_found_missed_prior",
        "human_missed_prior_title_or_url",
        "human_missed_prior_evidence",
        "human_final_false_not_covered",
        "human_adequacy_correct",
        "human_missing_prior_risk_correct",
        "human_notes",
        "annotator_id",
    ],
}


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten_for_csv(row.get(key, "")) for key in keys})


def normalize_label(label: str) -> str:
    mapping = {
        "supported": "covered",
        "partially_supported": "partially_covered",
        "unsupported": "not_covered",
    }
    return mapping.get(str(label or "").strip().lower(), str(label or "").strip().lower())


def compact_prior_dcu(dcu: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": dcu.get("acu_global_id") or dcu.get("id") or dcu.get("prior_dcu_id"),
        "paper_id": dcu.get("paper_id") or dcu.get("prior_paper_id"),
        "paper_title": dcu.get("title") or dcu.get("paper") or dcu.get("prior_paper_title"),
        "year": dcu.get("year"),
        "dataset_name": dcu.get("dataset_name") or dcu.get("dataset") or dcu.get("prior_dataset_name"),
        "type": dcu.get("acu_type") or dcu.get("type"),
        "text": dcu.get("acu_text") or dcu.get("text"),
        "evidence": dcu.get("evidence"),
        "section": dcu.get("section"),
        "retrieval_score": dcu.get("retrieval_score"),
    }


def query_context(row: dict[str, Any], query_acu: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_bank_id": row.get("query_bank_id"),
        "query_paper_id": row.get("query_paper_id"),
        "query_title": row.get("query_title"),
        "query_year": row.get("query_year"),
        "source_corpus": row.get("source_corpus"),
        "query_dataset_id": row.get("query_dataset_id"),
        "query_dataset_name": row.get("query_dataset_name"),
        "query_dcu_id": query_acu.get("id"),
        "query_dcu_text": query_acu.get("text"),
        "query_dcu_type": query_acu.get("type"),
        "query_dcu_importance": query_acu.get("importance"),
        "query_dcu_evidence": query_acu.get("evidence"),
        "query_dcu_section": query_acu.get("section"),
    }


def flatten_attribution_decisions(paths: list[str], top_k_evidence: int) -> Iterable[dict[str, Any]]:
    for path in paths:
        for row in iter_jsonl(path):
            query_by_id = {str(acu.get("id")): acu for acu in row.get("query_acus") or []}
            union_by_id = {
                str(dcu.get("acu_global_id") or dcu.get("id")): dcu
                for dcu in row.get("union_prior_dcus") or []
            }
            retrievals = row.get("query_acu_retrievals") or {}
            profile = row.get("profile") or {}
            for attr in row.get("attributions") or []:
                qid = str(attr.get("query_acu_id") or "")
                query_acu = query_by_id.get(qid, {"id": qid})
                selected_ids = [
                    str(x)
                    for x in (attr.get("best_prior_acu_ids") or attr.get("selected_prior_dcu_ids") or [])
                ]
                selected = [
                    compact_prior_dcu(union_by_id[pid])
                    for pid in selected_ids
                    if pid in union_by_id
                ]
                retrieved = []
                q_retrieval = retrievals.get(qid) or {}
                for prior in (q_retrieval.get("prior_dcus") or [])[:top_k_evidence]:
                    retrieved.append(compact_prior_dcu(prior))
                seen = {str(item.get("id")) for item in selected}
                for item in retrieved:
                    if str(item.get("id")) not in seen:
                        selected.append(item)
                        seen.add(str(item.get("id")))
                    if len(selected) >= top_k_evidence:
                        break
                yield {
                    "source_attribution_file": path,
                    **query_context(row, query_acu),
                    "model_coverage_label": normalize_label(attr.get("support_status") or ""),
                    "model_raw_support_status": attr.get("support_status"),
                    "model_delta_type": attr.get("delta_type"),
                    "model_importance": attr.get("importance"),
                    "model_evidence_adequacy": attr.get("evidence_adequacy"),
                    "model_missing_prior_risk": attr.get("missing_prior_risk"),
                    "model_rationale": attr.get("rationale"),
                    "selected_prior_dcu_ids": selected_ids,
                    "prior_evidence_for_review": selected,
                    "record_analysis_inclusion": profile.get("analysis_inclusion"),
                    "record_exclusion_reason": profile.get("analysis_exclusion_reason"),
                }


def sample_attribution_audits(
    paths: list[str],
    rng: random.Random,
    *,
    natural_n: int,
    stratified_n: int,
    missing_prior_n: int,
    top_k_evidence: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    natural: list[dict[str, Any]] = []
    seen_count = 0
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in flatten_attribution_decisions(paths, top_k_evidence):
        seen_count += 1
        if len(natural) < natural_n:
            natural.append(item)
        else:
            j = rng.randrange(seen_count)
            if j < natural_n:
                natural[j] = item

        adequacy_group = "adequate" if item.get("model_evidence_adequacy") in {"high", "medium"} else "low_adequacy"
        risk_group = "high_risk" if item.get("model_missing_prior_risk") == "high" else "low_or_medium_risk"
        key = (
            str(item.get("model_coverage_label") or "unknown"),
            str(item.get("query_dcu_type") or item.get("model_delta_type") or "unknown"),
            adequacy_group,
            risk_group,
        )
        bucket = buckets[key]
        if len(bucket) < 5:
            bucket.append(item)
        else:
            j = rng.randrange(seen_count)
            if j % 5 == 0:
                bucket[rng.randrange(len(bucket))] = item

        label = item.get("model_coverage_label")
        if label == "not_covered" and adequacy_group == "adequate" and risk_group != "high_risk":
            missing_key = "not_covered_adequate"
        elif label == "not_covered" and risk_group == "high_risk":
            missing_key = "not_covered_high_risk"
        elif label in {"covered", "partially_covered"}:
            missing_key = "positive_sanity"
        else:
            missing_key = "other"
        mb = missing_buckets[missing_key]
        if len(mb) < 200:
            mb.append(item)
        else:
            j = rng.randrange(seen_count)
            if j % 200 == 0:
                mb[rng.randrange(len(mb))] = item

    stratified_pool = []
    for key in sorted(buckets):
        bucket = buckets[key]
        rng.shuffle(bucket)
        stratified_pool.extend(bucket[:2])
    if len(stratified_pool) < stratified_n:
        extras = [item for bucket in buckets.values() for item in bucket]
        rng.shuffle(extras)
        existing = {id(item) for item in stratified_pool}
        stratified_pool.extend([item for item in extras if id(item) not in existing])
    stratified = stratified_pool[:stratified_n]

    missing_prior = []
    plan = [
        ("not_covered_adequate", 50),
        ("not_covered_high_risk", 25),
        ("positive_sanity", max(0, missing_prior_n - 75)),
    ]
    for key, count in plan:
        bucket = missing_buckets.get(key, [])
        rng.shuffle(bucket)
        missing_prior.extend(bucket[:count])
    if len(missing_prior) < missing_prior_n:
        extras = [item for bucket in missing_buckets.values() for item in bucket]
        rng.shuffle(extras)
        existing_ids = {id(item) for item in missing_prior}
        missing_prior.extend([item for item in extras if id(item) not in existing_ids][: missing_prior_n - len(missing_prior)])

    return natural, stratified[:stratified_n], missing_prior[:missing_prior_n]


def add_annotation_fields(rows: list[dict[str, Any]], audit_type: str, prefix: str) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows, start=1):
        enriched = {"annotation_id": f"{prefix}_{idx:04d}", "audit_type": audit_type, **row}
        for field in ANNOTATION_FIELDS[audit_type]:
            enriched.setdefault(field, "")
        out.append(enriched)
    return out


def sample_extraction_rows(
    dataset_bank: str,
    acu_bank: str,
    rng: random.Random,
    *,
    records_n: int,
    dcus_per_record: int,
) -> list[dict[str, Any]]:
    candidates = []
    seen = 0
    for row in iter_jsonl(dataset_bank):
        if int(row.get("n_acus") or len(row.get("acus") or []) or 0) <= 0:
            continue
        seen += 1
        if len(candidates) < records_n:
            candidates.append(row)
        else:
            j = rng.randrange(seen)
            if j < records_n:
                candidates[j] = row
    bank_ids = {str(row.get("bank_id")) for row in candidates}
    acus_by_bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for acu in iter_jsonl(acu_bank):
        bank_id = str(acu.get("bank_id") or "")
        if bank_id in bank_ids:
            acus_by_bank[bank_id].append(acu)

    rows = []
    for record in candidates:
        bank_id = str(record.get("bank_id"))
        acus = acus_by_bank.get(bank_id) or record.get("acus") or []
        rng.shuffle(acus)
        for acu in acus[:dcus_per_record]:
            rows.append({
                "bank_id": bank_id,
                "source_corpus": record.get("source_corpus"),
                "paper_id": record.get("paper_id"),
                "paper_title": record.get("title"),
                "year": record.get("year"),
                "dataset_id": record.get("dataset_id"),
                "dataset_name": record.get("dataset_name"),
                "role": record.get("role"),
                "resource_type": record.get("resource_type"),
                "tasks": record.get("tasks"),
                "domains": record.get("domains"),
                "languages": record.get("languages"),
                "modalities": record.get("modalities"),
                "source_data_origin": record.get("source_data_origin"),
                "source_datasets": record.get("source_datasets"),
                "collection_method": record.get("collection_method"),
                "annotation_protocol": record.get("annotation_protocol"),
                "quality_control": record.get("quality_control"),
                "scale": record.get("scale"),
                "release_status": record.get("release_status"),
                "license": record.get("license"),
                "paper_url": record.get("anthology_url") or record.get("pdf_url"),
                "dcu_id": acu.get("acu_global_id") or acu.get("acu_id") or acu.get("id"),
                "dcu_text": acu.get("acu_text") or acu.get("text"),
                "dcu_type": acu.get("acu_type") or acu.get("type"),
                "dcu_importance": acu.get("importance"),
                "dcu_evidence": acu.get("evidence"),
                "dcu_section": acu.get("section"),
            })
    return rows


def sample_retrieval_label_rows(
    benchmark_labels: str,
    hard_negatives: str,
    rng: random.Random,
    *,
    total_n: int,
) -> list[dict[str, Any]]:
    positives = []
    empties = []
    for row in iter_jsonl(benchmark_labels):
        query_by_id = row.get("query_acu_by_id") or {acu.get("id"): acu for acu in row.get("query_acus") or []}
        prior_by_id = row.get("prior_acu_by_id") or {acu.get("id"): acu for acu in row.get("prior_acu_bank") or []}
        for label in row.get("labels") or []:
            query = query_by_id.get(label.get("query_acu_id")) or {}
            selected_ids = label.get("selected_prior_acu_ids") or []
            base = {
                "benchmark_id": row.get("benchmark_id"),
                "query_paper_id": row.get("query_paper_id"),
                "query_title": row.get("query_title"),
                "query_dataset_name": row.get("query_dataset_name"),
                "query_dcu_id": label.get("query_acu_id"),
                "query_dcu_text": query.get("text"),
                "query_dcu_type": query.get("type"),
                "query_dcu_evidence": query.get("evidence"),
                "label_support_status": normalize_label(label.get("support_status") or ""),
                "label_rationale": label.get("rationale"),
                "gold_prior_dcus": [compact_prior_dcu(prior_by_id[pid]) for pid in selected_ids if pid in prior_by_id],
                "gold_prior_dcu_ids": selected_ids,
                "evaluate_in_retrieval_metrics": label.get("evaluate"),
            }
            if selected_ids:
                positives.append({"benchmark_audit_case": "gold_prior", **base})
            else:
                empties.append({"benchmark_audit_case": "empty_gold", **base})

    hard_rows = []
    for row in iter_jsonl(hard_negatives):
        hard_rows.append({
            "benchmark_audit_case": "hard_negative",
            "benchmark_id": row.get("benchmark_id"),
            "query_title": row.get("query_title"),
            "query_dataset_name": row.get("query_dataset_name"),
            "query_dcu_id": row.get("query_dcu_id"),
            "query_dcu_text": row.get("query_text"),
            "query_dcu_type": row.get("query_type"),
            "query_delta_type": row.get("delta_type"),
            "label_support_status": "gold_prior_with_hard_negative",
            "gold_prior_dcu_ids": row.get("gold_prior_dcu_ids"),
            "gold_prior_dcus": row.get("gold_prior_dcus"),
            "hard_negative_dcu_id": row.get("hard_negative_dcu_id"),
            "hard_negative_dcu": row.get("hard_negative_dcu"),
            "hard_negative_types": row.get("negative_types"),
            "negative_source": row.get("negative_source"),
        })

    rng.shuffle(positives)
    rng.shuffle(empties)
    rng.shuffle(hard_rows)
    n_gold = min(40, total_n // 2, len(positives))
    n_hard = min(40, total_n - n_gold, len(hard_rows))
    n_empty = min(total_n - n_gold - n_hard, len(empties))
    rows = positives[:n_gold] + hard_rows[:n_hard] + empties[:n_empty]
    if len(rows) < total_n:
        extras = positives[n_gold:] + hard_rows[n_hard:] + empties[n_empty:]
        rng.shuffle(extras)
        rows.extend(extras[: total_n - len(rows)])
    rng.shuffle(rows)
    return rows[:total_n]


def write_codebook(path: str | Path) -> None:
    text = """# Human Validation Codebook

This packet supports four audits: attribution decisions, extraction/DCU construction, retrieval-benchmark labels, and external missing-prior checks.

## Coverage Labels

- `covered`: Prior evidence already states the same dataset contribution along the relevant dimension.
- `partially_covered`: Prior evidence overlaps with the query claim but differs in an important dimension, such as domain, language, source, annotation protocol, scale, release setting, or evaluation use.
- `not_covered`: No displayed prior DCU covers or partially covers the query claim.
- `contradicted`: Displayed prior evidence conflicts with the query claim.
- `not_comparable`: The query and prior claims are too vague, generic, or structurally different to compare.

When judging a model label, use only the evidence shown in the annotation item unless the audit explicitly asks for external search.

## Selected Evidence Relevance

Use:

- `direct`: The selected prior DCU directly supports `covered` or is a strong comparator for `partially_covered`.
- `partial`: The selected prior DCU is related and useful but incomplete.
- `none`: The selected prior DCU is irrelevant, too generic, or from the wrong dataset/task/source.
- `na`: No selected prior evidence is present.

## Rationale Groundedness

Use:

- `yes`: The rationale uses only the query claim and displayed prior evidence.
- `partial`: Mostly grounded, with a minor unsupported inference.
- `no`: Ungrounded, unsupported, or contradicted by the displayed evidence.

## Evidence Adequacy

For attribution audits, use these values:

- `human_coverage_label_correct`: `yes`, `no`, or `uncertain`.
- `human_corrected_coverage_label`: one of `covered`, `partially_covered`, `not_covered`, `contradicted`, `not_comparable`, or blank when the model label is correct.
- `human_selected_evidence_relevance`: `direct`, `partial`, `none`, or `na`.
- `human_rationale_groundedness`: `yes`, `partial`, or `no`.

Mark `human_evidence_sufficient` as:

- `yes`: The displayed prior evidence is sufficient to interpret the coverage label.
- `no`: The evidence set is sparse, generic, wrong-domain, or missing obvious prior families.
- `uncertain`: You cannot tell from the item.

## Missing-Prior Risk

Mark `human_missing_prior_risk` as:

- `high`: There are signs that relevant prior work is missing and could change the label.
- `low`: The displayed evidence is broad/on-topic enough that omitted priors are unlikely to change the label.
- `uncertain`: Not enough context.

## External Missing-Prior Audit

For `missing_prior_external_audit`, do a lightweight search beyond the displayed evidence. Search places such as the paper bibliography, exact dataset/source names, ACL Anthology, arXiv, Hugging Face, GitHub, or a web search. Record any missed prior in `human_missed_prior_title_or_url` and quote or summarize the relevant evidence in `human_missed_prior_evidence`.

Use `human_final_false_not_covered=yes` only if you find a prior dataset claim that should have made the query `covered` or `partially_covered`.

## Extraction/DCU Audit

- `human_record_valid`: Is this a real dataset or benchmark record, introduced or substantially modified in the paper?
- `human_role_correct`: Is the role/resource type plausible from the shown metadata and evidence?
- `human_dcu_grounded`: Is the DCU claim faithful to the shown evidence span?
- `human_dcu_type_correct`: Is the DCU type correct?
- `human_metadata_errors`: List any incorrect metadata fields, or leave blank.

Use `yes`, `no`, or `uncertain` for the binary fields.

## Retrieval Benchmark Label Audit

- For `gold_prior` cases, judge whether the gold prior DCU truly covers or partially covers the query DCU.
- For `hard_negative` cases, judge whether the hard negative is plausible but does not cover the query DCU.
- For `empty_gold` cases, judge whether it is plausible that no cited-prior DCU covers or partially covers the query DCU.

Use:

- `human_gold_prior_validity`: `covered`, `partially_covered`, `not_evidence`, or `uncertain`.
- `human_hard_negative_validity`: `valid_hard_negative`, `actually_evidence`, `irrelevant_not_plausible`, or `uncertain`.
- `human_empty_gold_validity`: `valid_empty`, `missed_cited_prior`, or `uncertain`.
"""
    Path(path).write_text(text, encoding="utf-8")


def write_readme(path: str | Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Human Validation Packet",
        "",
        "Generated files:",
        "",
        "- `human_validation_codebook.md`: annotation guidelines.",
        "- `attribution_natural_sample.*`: natural-prevalence attribution audit.",
        "- `attribution_stratified_challenge_sample.*`: stratified challenge attribution audit.",
        "- `extraction_dcu_validation_sample.*`: dataset-record and DCU construction audit.",
        "- `retrieval_benchmark_label_validation_sample.*`: gold-prior, hard-negative, and empty-gold benchmark label audit.",
        "- `missing_prior_external_audit_sample.*`: external search audit for not-covered and sanity cases.",
        "- `double_annotation_*`: smaller overlapping subsets for estimating human-human agreement.",
        "",
        "Counts:",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("Use the CSV files for spreadsheet annotation and keep the JSONL files as canonical context-preserving records.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pair(out_dir: Path, stem: str, rows: list[dict[str, Any]]) -> None:
    write_jsonl(out_dir / f"{stem}.jsonl", rows)
    write_csv(out_dir / f"{stem}.csv", rows)


def mark_double_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "double_annotation": True} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create human validation annotation packets.")
    parser.add_argument("--attribution-jsonl", nargs="*", default=DEFAULT_ATTRIBUTION_JSONL)
    parser.add_argument("--dataset-bank-jsonl", default=DEFAULT_DATASET_BANK)
    parser.add_argument("--acu-bank-jsonl", default=DEFAULT_ACU_BANK)
    parser.add_argument("--benchmark-labels-jsonl", default=DEFAULT_BENCHMARK_LABELS)
    parser.add_argument("--hard-negatives-jsonl", default=DEFAULT_HARD_NEGATIVES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--natural-n", type=int, default=150)
    parser.add_argument("--stratified-n", type=int, default=200)
    parser.add_argument("--missing-prior-n", type=int, default=100)
    parser.add_argument("--extraction-records-n", type=int, default=100)
    parser.add_argument("--dcus-per-record", type=int, default=3)
    parser.add_argument("--retrieval-label-n", type=int, default=100)
    parser.add_argument("--top-k-evidence", type=int, default=10)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    natural, stratified, missing = sample_attribution_audits(
        args.attribution_jsonl,
        rng,
        natural_n=args.natural_n,
        stratified_n=args.stratified_n,
        missing_prior_n=args.missing_prior_n,
        top_k_evidence=args.top_k_evidence,
    )
    natural = add_annotation_fields(natural, "attribution", "attr_nat")
    stratified = add_annotation_fields(stratified, "attribution", "attr_strat")
    missing = add_annotation_fields(missing, "missing_prior", "miss_prior")

    extraction = sample_extraction_rows(
        args.dataset_bank_jsonl,
        args.acu_bank_jsonl,
        rng,
        records_n=args.extraction_records_n,
        dcus_per_record=args.dcus_per_record,
    )
    extraction = add_annotation_fields(extraction, "extraction", "extract")

    retrieval = sample_retrieval_label_rows(
        args.benchmark_labels_jsonl,
        args.hard_negatives_jsonl,
        rng,
        total_n=args.retrieval_label_n,
    )
    retrieval = add_annotation_fields(retrieval, "retrieval_label", "retrieval")

    write_pair(out_dir, "attribution_natural_sample", natural)
    write_pair(out_dir, "attribution_stratified_challenge_sample", stratified)
    write_pair(out_dir, "missing_prior_external_audit_sample", missing)
    write_pair(out_dir, "extraction_dcu_validation_sample", extraction)
    write_pair(out_dir, "retrieval_benchmark_label_validation_sample", retrieval)
    double_attribution = mark_double_rows(natural[:50] + stratified[:50])
    double_extraction = mark_double_rows(extraction[:40])
    double_retrieval = mark_double_rows(retrieval[:40])
    double_missing = mark_double_rows(missing[:40])
    write_pair(out_dir, "double_annotation_attribution_sample", double_attribution)
    write_pair(out_dir, "double_annotation_extraction_sample", double_extraction)
    write_pair(out_dir, "double_annotation_retrieval_label_sample", double_retrieval)
    write_pair(out_dir, "double_annotation_missing_prior_sample", double_missing)
    write_codebook(out_dir / "human_validation_codebook.md")

    summary = {
        "attribution_natural_sample": len(natural),
        "attribution_stratified_challenge_sample": len(stratified),
        "missing_prior_external_audit_sample": len(missing),
        "extraction_dcu_validation_sample": len(extraction),
        "retrieval_benchmark_label_validation_sample": len(retrieval),
        "double_annotation_attribution_sample": len(double_attribution),
        "double_annotation_extraction_sample": len(double_extraction),
        "double_annotation_retrieval_label_sample": len(double_retrieval),
        "double_annotation_missing_prior_sample": len(double_missing),
        "seed": args.seed,
        "top_k_evidence": args.top_k_evidence,
    }
    write_readme(out_dir / "README.md", summary)
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
