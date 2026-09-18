#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_fulltext_prior_candidate_queue as queue_builder  # noqa: E402
import evaluate_global_claim_level_dcu_retrieval as llm_utils  # noqa: E402


SUPPORT_STATUSES = {"supported", "partially_supported", "unsupported", "contradicted", "not_comparable"}
POSITIVE_STATUSES = {"supported", "partially_supported"}
DELTA_TYPES = {
    "task/domain",
    "data/source",
    "annotation/protocol",
    "scale/coverage",
    "evaluation/use",
    "availability/quality",
    "governance/ethics",
    "other",
}
IMPORTANCE_LEVELS = {"low", "medium", "high"}
ADEQUACY_LEVELS = {"low", "medium", "high"}
RISK_LEVELS = {"low", "medium", "high"}
SUPPORT_DELTA_VALUES = {"supported": 0.0, "partially_supported": 0.5, "unsupported": 1.0}
EXCLUDED_SUPPORT_STATUSES = {"contradicted", "not_comparable"}
IMPORTANCE_WEIGHTS = {"low": 0.5, "medium": 1.0, "high": 1.5}


PROMPT_VERSION = "fulltext_dcu_native_attribution_v3_strict_support_adequacy"
BATCH_PROMPT_VERSION = "fulltext_dcu_native_attribution_v5_dataset_compact_calibrated_risk"
DENSE_SCORE_LOCK = threading.Lock()
PROMPT = """You are an expert NLP researcher doing claim-level prior evidence attribution.

Task:
Given one query Dataset Contribution Unit (query DCU) and the top retrieved prior DCUs, decide whether the prior DCUs support the query claim.

Support statuses:
- supported: selected prior DCUs directly state the same factual contribution. Key specifics should match, such as dataset/source family, language/domain, construction source, annotation protocol, metric, release/access claim, or measured scale.
- partially_supported: selected prior DCUs cover the same contribution dimension or useful comparator evidence, but key specifics differ or the query adds meaningful new language/domain/source/protocol/scale/evaluation details.
- unsupported: none of the supplied prior DCUs provide meaningful evidence for this contribution dimension.
- contradicted: supplied prior DCUs directly conflict with the query claim.
- not_comparable: the query claim cannot be compared to the supplied prior DCUs.

Rules:
- Use only prior_dcu_ids from the supplied candidate list.
- A prior DCU usually will not mention the query dataset name; do not require it to describe the query dataset itself.
- Topical similarity alone is not enough. Select evidence only when it is useful prior evidence for the same dataset-contribution dimension.
- If the prior is merely another dataset of the same broad type, choose partially_supported, not supported.
- For scale/coverage claims, different numbers or different coverage are usually partially_supported, not supported, unless the prior describes the same dataset family or the same measured scale.
- For task/domain claims, closely related benchmark task/domain evidence is usually partially_supported when the query introduces a new variant, new setting, new language, or new modality.
- For data/source claims, the same source dataset/family/data origin can be supported; merely related construction sources are partially_supported.
- For annotation/protocol claims, the same protocol or same label schema can be supported; comparable annotation, verification, labeling, or quality-control protocols are partially_supported.
- For evaluation/use claims, the same metric/evaluation setup can be supported; comparable benchmark use, metrics, baselines, or evaluation setup are partially_supported.
- Use unsupported when candidates are only broad topical neighbors or generic dataset examples without comparable contribution evidence.
- If support_status is supported or partially_supported, selected_prior_dcu_ids must be non-empty.
- If no supplied candidate provides evidence, return unsupported with an empty selected_prior_dcu_ids list.
- Separately judge whether the supplied top retrieved prior DCUs are adequate for this query DCU.
- evidence_adequacy:
  - high: retrieved prior DCUs include direct source/comparison datasets, close task/domain matches, or strong antecedent evidence.
  - medium: retrieved prior DCUs are related and useful but may miss stronger direct antecedents.
  - low: retrieved prior DCUs are mostly weak topical neighbors, sparse, or unlikely to include the strongest prior work.
- missing_prior_risk:
  - low: unlikely that omitted priors would substantially change this support label.
  - medium: plausible that omitted priors could change the label.
  - high: likely that important prior work is missing, so this attribution should be treated as low-confidence.
- Keep the rationale short and evidence-grounded.

Return JSON only:
{{"support_status":"supported|partially_supported|unsupported|contradicted|not_comparable","selected_prior_dcu_ids":["prior_dcu_id"],"delta_type":"task/domain|data/source|annotation/protocol|scale/coverage|evaluation/use|availability/quality|governance/ethics|other","importance":"low|medium|high","evidence_adequacy":"high|medium|low","missing_prior_risk":"low|medium|high","rationale":"brief reason"}}

Query DCU:
{query_dcu}

Retrieved prior DCUs:
{prior_dcus}
"""

BATCH_PROMPT = """You are an expert NLP researcher doing claim-level prior evidence attribution.

Task:
Given one query dataset with multiple query Dataset Contribution Units (query DCUs), and a compact union of retrieved prior DCUs, assign one prior-evidence label to every query DCU.

Support statuses:
- supported: selected prior DCUs directly state the same factual contribution. Key specifics should match, such as dataset/source family, language/domain, construction source, annotation protocol, metric, release/access claim, or measured scale.
- partially_supported: selected prior DCUs cover the same contribution dimension or useful comparator evidence, but key specifics differ or the query adds meaningful new language/domain/source/protocol/scale/evaluation details.
- unsupported: none of the supplied prior DCUs provide meaningful evidence for this contribution dimension.
- contradicted: supplied prior DCUs directly conflict with the query claim.
- not_comparable: the query claim cannot be compared to the supplied prior DCUs.

Rules:
- Return exactly one attribution object for each query_acu_id.
- Use only prior_dcu_ids from the supplied candidate list.
- Topical similarity alone is not enough. Select evidence only when it is useful prior evidence for the same dataset-contribution dimension.
- If the prior is merely another dataset of the same broad type, choose partially_supported, not supported.
- For scale/coverage claims, different numbers or different coverage are usually partially_supported, not supported, unless the prior describes the same dataset family or the same measured scale.
- For task/domain claims, closely related benchmark task/domain evidence is usually partially_supported when the query introduces a new variant, new setting, new language, or new modality.
- For data/source claims, the same source dataset/family/data origin can be supported; merely related construction sources are partially_supported.
- For annotation/protocol claims, the same protocol or same label schema can be supported; comparable annotation, verification, labeling, or quality-control protocols are partially_supported.
- For evaluation/use claims, the same metric/evaluation setup can be supported; comparable benchmark use, metrics, baselines, or evaluation setup are partially_supported.
- If support_status is supported or partially_supported, selected_prior_dcu_ids must be non-empty.
- If no supplied candidate provides evidence, return unsupported with an empty selected_prior_dcu_ids list.
- Judge evidence_adequacy and missing_prior_risk separately for each query DCU.
- Do not mark evidence_adequacy as low merely because the evidence is not exhaustive or not from the exact same dataset. If the retrieved DCUs include same-type, same-task/domain, same-source-family, same-annotation, same-scale, or otherwise comparable contribution evidence, evidence_adequacy should usually be medium. Use high when the evidence includes direct source/family/prior benchmark evidence. Use low only when candidates are mostly broad topical neighbors or too sparse to support a meaningful comparison.
- Do not mark missing_prior_risk as high merely because stronger prior work may exist outside the top retrieved candidates. Top-50 retrieval is expected to be non-exhaustive. Use high only when the supplied candidates are inadequate to judge the query DCU, such as when the query clearly points to a source dataset/family/protocol but no comparable candidate appears, or when the candidates are generic topical neighbors. Use medium when omitted priors could refine the label but the supplied candidates are enough for a supported/partially_supported/unsupported decision. Use low when direct source/family/prior benchmark evidence is present or omitted priors are unlikely to change the label.
- If you assign supported or partially_supported with selected evidence, evidence_adequacy should usually be medium or high, and missing_prior_risk should usually be low or medium. Only assign high missing_prior_risk with a positive support label when the selected evidence is weak and likely incomplete.
- Keep each rationale short and evidence-grounded.

Return JSON only:
{{"attributions":[{{"query_acu_id":"q0","support_status":"supported|partially_supported|unsupported|contradicted|not_comparable","selected_prior_dcu_ids":["prior_dcu_id"],"delta_type":"task/domain|data/source|annotation/protocol|scale/coverage|evaluation/use|availability/quality|governance/ethics|other","importance":"low|medium|high","evidence_adequacy":"high|medium|low","missing_prior_risk":"low|medium|high","rationale":"brief reason"}}]}}

Query dataset:
{query_dataset}

Query DCUs:
{query_dcus}

Retrieved prior DCU candidates:
{prior_dcus}
"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    target = Path(path)
    if not target.exists():
        return rows
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_query_rows(
    path: str | Path,
    *,
    query_year: int | None,
    limit: int | None,
    sample_mode: str,
) -> list[dict[str, Any]]:
    if limit is None or sample_mode != "first":
        rows = read_jsonl(path)
        if query_year is not None:
            rows = [row for row in rows if row.get("year") == query_year]
        return [row for row in rows if query_acus(row)]
    rows = []
    for row in iter_jsonl(path):
        if query_year is not None and row.get("year") != query_year:
            continue
        if not query_acus(row):
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def existing_ids(path: str | Path) -> set[str]:
    return {
        str(row.get("query_bank_id") or "")
        for row in read_jsonl(path)
        if row.get("query_bank_id")
    }


def row_bank_id(row: dict[str, Any]) -> str:
    return str(row.get("query_bank_id") or row.get("bank_id") or "")


def select_rows(rows: list[dict[str, Any]], *, limit: int | None, sample_mode: str, sample_seed: int) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(sample_seed)
    if sample_mode == "first":
        return rows[:limit]
    if sample_mode == "random":
        return rng.sample(rows, limit)
    if sample_mode == "stratified_year":
        by_year: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_year[str(row.get("year") or "unknown")].append(row)
        selected = []
        remaining = limit
        years = sorted(by_year)
        for idx, year in enumerate(years):
            bucket = by_year[year]
            take = remaining if idx == len(years) - 1 else round(limit * len(bucket) / len(rows))
            take = min(take, remaining, len(bucket))
            selected.extend(rng.sample(bucket, take))
            remaining = limit - len(selected)
        if len(selected) < limit:
            selected_ids = {row.get("bank_id") for row in selected}
            rest = [row for row in rows if row.get("bank_id") not in selected_ids]
            selected.extend(rng.sample(rest, min(limit - len(selected), len(rest))))
        return selected[:limit]
    raise ValueError(f"Unknown sample mode: {sample_mode}")


def query_acus(row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("query_acus") and not row.get("acus"):
        output = []
        for index, acu in enumerate(row.get("query_acus") or []):
            text = str(acu.get("text") or acu.get("query_acu") or acu.get("acu_text") or "").strip()
            if not text:
                continue
            output.append({
                "id": str(acu.get("id") or acu.get("query_acu_id") or f"q{index}"),
                "text": text,
                "type": str(acu.get("type") or acu.get("query_acu_type") or acu.get("acu_type") or ""),
                "importance": str(acu.get("importance") or "medium"),
                "evidence": str(acu.get("evidence") or ""),
                "section": str(acu.get("section") or ""),
            })
        return output
    output = []
    for index, acu in enumerate(row.get("acus") or []):
        text = str(acu.get("text") or acu.get("acu_text") or "").strip()
        if not text:
            continue
        output.append({
            "id": str(acu.get("id") or acu.get("acu_id") or f"q{index}"),
            "text": text,
            "type": str(acu.get("type") or acu.get("acu_type") or ""),
            "importance": str(acu.get("importance") or "medium"),
            "evidence": str(acu.get("evidence") or ""),
            "section": str(acu.get("section") or ""),
        })
    return output


def query_dcu_card(row: dict[str, Any], acu: dict[str, Any]) -> str:
    if not row.get("dataset_name") and row.get("query_dataset_name"):
        row = {
            **row,
            "dataset_name": row.get("query_dataset_name"),
            "title": row.get("query_title"),
            "year": row.get("query_year"),
        }
    return queue_builder.query_dcu_text(row, acu)


def prior_dcu_card(acu: dict[str, Any], dataset: dict[str, Any] | None) -> str:
    return (
        f"Prior DCU ID: {acu.get('acu_global_id') or ''}\n"
        f"{queue_builder.prior_dcu_text(acu, dataset)}"
    )


def truncate_text(text: Any, max_chars: int = 220) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def compact_query_dataset_card(row: dict[str, Any]) -> str:
    parts = [
        f"dataset: {row.get('dataset_name') or row.get('query_dataset_name') or ''}",
        f"paper: {row.get('title') or row.get('query_title') or ''}",
        f"year: {row.get('year') or row.get('query_year') or ''}",
        f"tasks: {queue_builder.format_list(row.get('tasks'))}",
        f"domains: {queue_builder.format_list(row.get('domains'))}",
        f"languages: {queue_builder.format_list(row.get('languages'))}",
        f"modalities: {queue_builder.format_list(row.get('modalities'))}",
        f"source: {truncate_text(row.get('source_data_origin'), 160)}",
        f"annotation: {truncate_text(row.get('annotation_protocol'), 160)}",
        f"scale: {queue_builder.scale_text(row)}",
    ]
    return "\n".join(part for part in parts if not part.endswith(": "))


def compact_query_dcu_card(acu: dict[str, Any]) -> str:
    parts = [
        f"id: {acu.get('id') or ''}",
        f"type: {acu.get('type') or acu.get('acu_type') or ''}",
        f"importance: {acu.get('importance') or 'medium'}",
        f"claim: {truncate_text(acu.get('text') or acu.get('acu_text'), 360)}",
    ]
    return " | ".join(part for part in parts if not part.endswith(": "))


def compact_prior_dcu_prompt_card(candidate: dict[str, Any], dataset: dict[str, Any] | None) -> str:
    dataset = dataset or {}
    metadata = [
        f"task={queue_builder.format_list(dataset.get('tasks'))}",
        f"domain={queue_builder.format_list(dataset.get('domains'))}",
        f"lang={queue_builder.format_list(dataset.get('languages'))}",
        f"modality={queue_builder.format_list(dataset.get('modalities'))}",
        f"source={truncate_text(dataset.get('source_data_origin'), 120)}",
        f"annotation={truncate_text(dataset.get('annotation_protocol'), 120)}",
        f"scale={queue_builder.scale_text(dataset)}",
    ]
    metadata = [item for item in metadata if not item.endswith("=")]
    paper_title = dataset.get("title") or candidate.get("title") or candidate.get("prior_title") or ""
    dataset_name = dataset.get("dataset_name") or candidate.get("dataset_name") or candidate.get("prior_dataset_name") or ""
    year = dataset.get("year") or candidate.get("year") or candidate.get("prior_year") or ""
    lines = [
        f"id: {candidate.get('acu_global_id') or ''}",
        f"type: {candidate.get('acu_type') or ''}",
        f"dataset: {dataset_name}",
        f"paper: {year} | {truncate_text(paper_title, 180)}",
        f"claim: {truncate_text(candidate.get('acu_text'), 360)}",
        f"metadata: {'; '.join(metadata)}" if metadata else "",
    ]
    evidence = truncate_text(candidate.get("evidence"), 180)
    if evidence:
        lines.append(f"evidence excerpt: {evidence}")
    return "\n".join(line for line in lines if line)


def retrieve_prior_dcus(
    *,
    query: dict[str, Any],
    query_acu: dict[str, Any],
    acu_rows: list[dict[str, Any]],
    dataset_by_bank_id: dict[str, dict[str, Any]],
    hybrid_index: queue_builder.HybridDcuIndex,
    year_mode: str,
    candidate_depth: int,
    min_score: float,
    fast_dense_pool_size: int = 0,
    fast_sparse_pool_size: int = 0,
) -> list[dict[str, Any]]:
    # SentenceTransformer query encoding is not reliably thread-safe on every
    # local torch/MPS setup; keep retrieval scoring serialized while allowing
    # API calls and row processing to remain concurrent.
    query_text = query_dcu_card(query, query_acu)
    with DENSE_SCORE_LOCK:
        if fast_dense_pool_size > 0:
            ranked_scores = hybrid_index.fast_hybrid_scores(
                query_text,
                dense_pool_size=fast_dense_pool_size,
                sparse_pool_size=fast_sparse_pool_size,
            )
            score_lookup = None
        else:
            scores = hybrid_index.scores(query_text)
            ranked_scores = [
                (int(index), float(scores[int(index)]))
                for index in queue_builder.top_indices(scores, min(candidate_depth * 10, len(acu_rows)))
            ]
            score_lookup = scores
    candidates = []
    for acu_index, ranked_score in ranked_scores:
        score = float(ranked_score if score_lookup is None else score_lookup[int(acu_index)])
        if score < min_score:
            continue
        acu = acu_rows[int(acu_index)]
        if not queue_builder.is_allowed_prior(query, acu, year_mode):
            continue
        dataset = dataset_by_bank_id.get(str(acu.get("bank_id") or ""))
        candidates.append({
            **acu,
            "retrieval_score": score,
            "prior_dataset_name": (dataset or {}).get("dataset_name") or acu.get("dataset_name") or "",
            "prior_title": (dataset or {}).get("title") or acu.get("title") or "",
            "prior_year": (dataset or {}).get("year") or acu.get("year"),
        })
        if len(candidates) >= candidate_depth:
            break
    return candidates


def build_prompt(query: dict[str, Any], query_acu: dict[str, Any], candidates: list[dict[str, Any]], dataset_by_bank_id: dict[str, dict[str, Any]]) -> str:
    return PROMPT.format(
        query_dcu=query_dcu_card(query, query_acu),
        prior_dcus="\n\n".join(
            prior_dcu_card(candidate, dataset_by_bank_id.get(str(candidate.get("bank_id") or "")))
            for candidate in candidates
        ) or "None",
    )


def coerce_json_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            return payload[0]
        if all(isinstance(item, dict) for item in payload):
            return {"attributions": payload}
    return {}


def normalize_prediction(payload: dict[str, Any], query_acu: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    payload = coerce_json_payload(payload)
    candidate_ids = {str(candidate.get("acu_global_id") or "") for candidate in candidates}
    status = str(payload.get("support_status") or "unsupported")
    if status not in SUPPORT_STATUSES:
        status = "unsupported"
    selected = [
        str(candidate_id)
        for candidate_id in payload.get("selected_prior_dcu_ids") or payload.get("best_prior_acu_ids") or []
        if str(candidate_id) in candidate_ids
    ]
    if status in POSITIVE_STATUSES and not selected:
        status = "unsupported"
    if status not in POSITIVE_STATUSES and status != "contradicted":
        selected = []
    delta_type = str(payload.get("delta_type") or query_acu.get("type") or "other")
    if delta_type not in DELTA_TYPES:
        delta_type = "other"
    importance = str(payload.get("importance") or query_acu.get("importance") or "medium")
    if importance not in IMPORTANCE_LEVELS:
        importance = "medium"
    evidence_adequacy = str(payload.get("evidence_adequacy") or "medium").lower()
    if evidence_adequacy not in ADEQUACY_LEVELS:
        evidence_adequacy = "medium"
    missing_prior_risk = str(payload.get("missing_prior_risk") or "medium").lower()
    if missing_prior_risk not in RISK_LEVELS:
        missing_prior_risk = "medium"
    return {
        "query_acu_id": query_acu["id"],
        "query_acu": query_acu["text"],
        "query_acu_type": query_acu.get("type") or "",
        "support_status": status,
        "best_prior_acu_ids": selected,
        "delta_type": delta_type,
        "importance": importance,
        "evidence_adequacy": evidence_adequacy,
        "missing_prior_risk": missing_prior_risk,
        "rationale": str(payload.get("rationale") or ""),
    }


def collect_candidates_for_query_acu(
    row: dict[str, Any],
    query_acu: dict[str, Any],
    *,
    acu_rows: list[dict[str, Any]] | None,
    dataset_by_bank_id: dict[str, dict[str, Any]],
    hybrid_index: queue_builder.HybridDcuIndex | None,
    year_mode: str,
    candidate_depth: int,
    min_score: float,
) -> list[dict[str, Any]]:
    precomputed = (row.get("query_acu_retrievals") or {}).get(query_acu["id"])
    if precomputed is not None:
        return list(precomputed.get("prior_dcus") or [])[:candidate_depth]
    if acu_rows is None or hybrid_index is None:
        raise ValueError("Missing retrieval candidates and no hybrid index was initialized.")
    return retrieve_prior_dcus(
        query=row,
        query_acu=query_acu,
        acu_rows=acu_rows,
        dataset_by_bank_id=dataset_by_bank_id,
        hybrid_index=hybrid_index,
        year_mode=year_mode,
        candidate_depth=candidate_depth,
        min_score=min_score,
        fast_dense_pool_size=0,
        fast_sparse_pool_size=0,
    )


def build_dataset_compact_prompt(
    query: dict[str, Any],
    q_acus: list[dict[str, Any]],
    union_candidates: list[dict[str, Any]],
    dataset_by_bank_id: dict[str, dict[str, Any]],
) -> str:
    return BATCH_PROMPT.format(
        query_dataset=compact_query_dataset_card(query),
        query_dcus="\n".join(compact_query_dcu_card(acu) for acu in q_acus) or "None",
        prior_dcus="\n\n".join(
            compact_prior_dcu_prompt_card(candidate, dataset_by_bank_id.get(str(candidate.get("bank_id") or "")))
            for candidate in union_candidates
        ) or "None",
    )


def normalize_batch_predictions(
    payload: Any,
    q_acus: list[dict[str, Any]],
    union_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = coerce_json_payload(payload)
    raw_attributions = payload.get("attributions") or payload.get("results") or payload.get("labels") or []
    if isinstance(raw_attributions, dict):
        raw_attributions = list(raw_attributions.values())
    if not isinstance(raw_attributions, list):
        raw_attributions = []
    by_query_id = {
        str(item.get("query_acu_id") or item.get("id") or ""): item
        for item in raw_attributions
        if isinstance(item, dict)
    }
    predictions = []
    for query_acu in q_acus:
        raw = by_query_id.get(str(query_acu["id"])) or {}
        predictions.append(normalize_prediction(raw, query_acu, union_candidates))
    return predictions


def compact_prior_dcus(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "acu_global_id": candidate.get("acu_global_id") or "",
            "bank_id": candidate.get("bank_id") or "",
            "paper_id": candidate.get("paper_id") or "",
            "title": candidate.get("title") or candidate.get("prior_title") or "",
            "year": candidate.get("year") or candidate.get("prior_year"),
            "dataset_name": candidate.get("dataset_name") or candidate.get("prior_dataset_name") or "",
            "acu_text": candidate.get("acu_text") or "",
            "acu_type": candidate.get("acu_type") or "",
            "evidence": candidate.get("evidence") or "",
            "section": candidate.get("section") or "",
            "retrieval_score": candidate.get("retrieval_score"),
        }
        for candidate in candidates
    ]


def added_information_profile(
    attributions: list[dict[str, Any]],
    *,
    max_low_adequacy_rate: float = 0.5,
    max_high_risk_rate: float = 0.34,
    min_adequate_rate: float = 0.5,
) -> dict[str, Any]:
    counts = Counter(row["support_status"] for row in attributions)
    adequacy_counts = Counter(row.get("evidence_adequacy") or "medium" for row in attributions)
    risk_counts = Counter(row.get("missing_prior_risk") or "medium" for row in attributions)
    unsupported_by_delta_type = Counter(
        row["delta_type"]
        for row in attributions
        if row["support_status"] == "unsupported"
    )
    numerator = 0.0
    denominator = 0.0
    for row in attributions:
        status = row["support_status"]
        if status in EXCLUDED_SUPPORT_STATUSES:
            continue
        weight = IMPORTANCE_WEIGHTS[row["importance"]]
        numerator += SUPPORT_DELTA_VALUES[status] * weight
        denominator += weight
    total = len(attributions)
    low_adequacy_count = adequacy_counts.get("low", 0)
    high_risk_count = risk_counts.get("high", 0)
    adequate_attribution_count = sum(
        1 for row in attributions
        if row.get("evidence_adequacy") in {"high", "medium"}
        and row.get("missing_prior_risk") in {"low", "medium"}
    )
    low_adequacy_rate = low_adequacy_count / total if total else 0.0
    high_risk_rate = high_risk_count / total if total else 0.0
    adequate_attribution_rate = adequate_attribution_count / total if total else 0.0
    analysis_inclusion = (
        total > 0
        and low_adequacy_rate <= max_low_adequacy_rate
        and high_risk_rate <= max_high_risk_rate
        and adequate_attribution_rate >= min_adequate_rate
    )
    return {
        "n_query_acus": total,
        "added_information_score": numerator / denominator if denominator else None,
        "support_counts": dict(counts),
        "evidence_adequacy_counts": dict(adequacy_counts),
        "missing_prior_risk_counts": dict(risk_counts),
        "analysis_inclusion": analysis_inclusion,
        "analysis_exclusion_reason": ""
        if analysis_inclusion
        else "low_prior_adequacy_or_high_missing_prior_risk",
        "low_adequacy_query_acu_count": low_adequacy_count,
        "high_missing_prior_risk_query_acu_count": high_risk_count,
        "adequate_query_acu_count": adequate_attribution_count,
        "low_adequacy_rate": low_adequacy_rate,
        "high_missing_prior_risk_rate": high_risk_rate,
        "adequate_query_acu_rate": adequate_attribution_rate,
        "analysis_inclusion_thresholds": {
            "max_low_adequacy_rate": max_low_adequacy_rate,
            "max_high_risk_rate": max_high_risk_rate,
            "min_adequate_rate": min_adequate_rate,
        },
        "support_percentages": {
            status: counts.get(status, 0) / total if total else 0.0
            for status in ["supported", "partially_supported", "unsupported", "contradicted", "not_comparable"]
        },
        "unsupported_by_delta_type": dict(unsupported_by_delta_type),
        "excluded_from_score_count": sum(counts.get(status, 0) for status in EXCLUDED_SUPPORT_STATUSES),
    }


def usage_tokens(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_token_count") or usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or usage.get("candidates_token_count") or usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or usage.get("total_token_count") or 0),
    }


def run_one(
    row: dict[str, Any],
    *,
    model: str,
    acu_rows: list[dict[str, Any]] | None,
    dataset_by_bank_id: dict[str, dict[str, Any]],
    hybrid_index: queue_builder.HybridDcuIndex | None,
    year_mode: str,
    candidate_depth: int,
    min_score: float,
    max_low_adequacy_rate: float = 0.5,
    max_high_risk_rate: float = 0.34,
    min_adequate_rate: float = 0.5,
) -> dict[str, Any]:
    start = time.time()
    attributions = []
    retrievals = {}
    usage_total = Counter()
    q_acus = query_acus(row)
    if not q_acus:
        raise ValueError("No query ACUs.")
    for query_acu in q_acus:
        candidates = collect_candidates_for_query_acu(
            row,
            query_acu,
            acu_rows=acu_rows,
            dataset_by_bank_id=dataset_by_bank_id,
            hybrid_index=hybrid_index,
            year_mode=year_mode,
            candidate_depth=candidate_depth,
            min_score=min_score,
        )
        if not candidates:
            prediction = {
                "query_acu_id": query_acu["id"],
                "query_acu": query_acu["text"],
                "query_acu_type": query_acu.get("type") or "",
                "support_status": "unsupported",
                "best_prior_acu_ids": [],
                "delta_type": query_acu.get("type") or "other",
                "importance": query_acu.get("importance") or "medium",
                "evidence_adequacy": "low",
                "missing_prior_risk": "high",
                "rationale": "No prior DCU candidates were retrieved.",
            }
        else:
            prompt = build_prompt(row, query_acu, candidates, dataset_by_bank_id)
            payload, usage = llm_utils.call_json_llm(prompt, model=model)
            prediction = normalize_prediction(payload, query_acu, candidates)
            usage_total.update(usage_tokens(usage))
        attributions.append(prediction)
        retrievals[query_acu["id"]] = {
            "method": "hybrid_dcu",
            "candidate_depth": candidate_depth,
            "prior_dcus": compact_prior_dcus(candidates),
        }
    return {
        "query_bank_id": row_bank_id(row),
        "query_paper_id": row.get("query_paper_id") or row.get("paper_id") or "",
        "query_dataset_id": row.get("query_dataset_id") or row.get("dataset_id") or "",
        "query_dataset_name": row.get("query_dataset_name") or row.get("dataset_name") or "",
        "query_title": row.get("query_title") or row.get("title") or "",
        "query_year": row.get("query_year") or row.get("year"),
        "source_corpus": row.get("source_corpus") or "",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "attribution_mode": "per_acu",
        "retrieval": {
            "method": "hybrid_dcu",
            "year_mode": year_mode,
            "candidate_depth": candidate_depth,
            "min_score": min_score,
        },
        "query_acus": q_acus,
        "query_acu_retrievals": retrievals,
        "attributions": attributions,
        "profile": added_information_profile(
            attributions,
            max_low_adequacy_rate=max_low_adequacy_rate,
            max_high_risk_rate=max_high_risk_rate,
            min_adequate_rate=min_adequate_rate,
        ),
        "llm_usage": dict(usage_total),
        "runtime_seconds": round(time.time() - start, 3),
    }


def run_one_dataset_compact(
    row: dict[str, Any],
    *,
    model: str,
    acu_rows: list[dict[str, Any]] | None,
    dataset_by_bank_id: dict[str, dict[str, Any]],
    hybrid_index: queue_builder.HybridDcuIndex | None,
    year_mode: str,
    candidate_depth: int,
    min_score: float,
    max_low_adequacy_rate: float = 0.5,
    max_high_risk_rate: float = 0.34,
    min_adequate_rate: float = 0.5,
) -> dict[str, Any]:
    start = time.time()
    q_acus = query_acus(row)
    if not q_acus:
        raise ValueError("No query ACUs.")

    retrievals = {}
    union_by_id: dict[str, dict[str, Any]] = {}
    first_seen_order: dict[str, int] = {}
    for query_acu in q_acus:
        candidates = collect_candidates_for_query_acu(
            row,
            query_acu,
            acu_rows=acu_rows,
            dataset_by_bank_id=dataset_by_bank_id,
            hybrid_index=hybrid_index,
            year_mode=year_mode,
            candidate_depth=candidate_depth,
            min_score=min_score,
        )
        retrievals[query_acu["id"]] = {
            "method": "hybrid_dcu",
            "candidate_depth": candidate_depth,
            "prior_dcus": compact_prior_dcus(candidates),
        }
        for rank, candidate in enumerate(candidates):
            candidate_id = str(candidate.get("acu_global_id") or "")
            if not candidate_id:
                continue
            score = float(candidate.get("retrieval_score") or 0.0)
            if candidate_id not in union_by_id:
                first_seen_order[candidate_id] = len(first_seen_order)
                union_by_id[candidate_id] = {
                    **candidate,
                    "retrieval_score": score,
                    "retrieved_for_query_acu_ids": [query_acu["id"]],
                    "best_query_rank": rank + 1,
                }
            else:
                existing = union_by_id[candidate_id]
                existing["retrieval_score"] = max(float(existing.get("retrieval_score") or 0.0), score)
                existing["best_query_rank"] = min(int(existing.get("best_query_rank") or rank + 1), rank + 1)
                ids = existing.setdefault("retrieved_for_query_acu_ids", [])
                if query_acu["id"] not in ids:
                    ids.append(query_acu["id"])

    union_candidates = sorted(
        union_by_id.values(),
        key=lambda candidate: (
            -float(candidate.get("retrieval_score") or 0.0),
            int(candidate.get("best_query_rank") or 10**9),
            first_seen_order.get(str(candidate.get("acu_global_id") or ""), 10**9),
        ),
    )

    usage_total = Counter()
    if union_candidates:
        prompt = build_dataset_compact_prompt(row, q_acus, union_candidates, dataset_by_bank_id)
        payload, usage = llm_utils.call_json_llm(prompt, model=model)
        attributions = normalize_batch_predictions(payload, q_acus, union_candidates)
        usage_total.update(usage_tokens(usage))
    else:
        attributions = [
            {
                "query_acu_id": query_acu["id"],
                "query_acu": query_acu["text"],
                "query_acu_type": query_acu.get("type") or "",
                "support_status": "unsupported",
                "best_prior_acu_ids": [],
                "delta_type": query_acu.get("type") or "other",
                "importance": query_acu.get("importance") or "medium",
                "evidence_adequacy": "low",
                "missing_prior_risk": "high",
                "rationale": "No prior DCU candidates were retrieved.",
            }
            for query_acu in q_acus
        ]

    return {
        "query_bank_id": row_bank_id(row),
        "query_paper_id": row.get("query_paper_id") or row.get("paper_id") or "",
        "query_dataset_id": row.get("query_dataset_id") or row.get("dataset_id") or "",
        "query_dataset_name": row.get("query_dataset_name") or row.get("dataset_name") or "",
        "query_title": row.get("query_title") or row.get("title") or "",
        "query_year": row.get("query_year") or row.get("year"),
        "source_corpus": row.get("source_corpus") or "",
        "model": model,
        "prompt_version": BATCH_PROMPT_VERSION,
        "attribution_mode": "dataset_compact",
        "retrieval": {
            "method": "hybrid_dcu",
            "year_mode": year_mode,
            "candidate_depth": candidate_depth,
            "union_prior_dcu_count": len(union_candidates),
            "min_score": min_score,
        },
        "query_acus": q_acus,
        "query_acu_retrievals": retrievals,
        "union_prior_dcus": compact_prior_dcus(union_candidates),
        "attributions": attributions,
        "profile": added_information_profile(
            attributions,
            max_low_adequacy_rate=max_low_adequacy_rate,
            max_high_risk_rate=max_high_risk_rate,
            min_adequate_rate=min_adequate_rate,
        ),
        "llm_usage": dict(usage_total),
        "runtime_seconds": round(time.time() - start, 3),
    }


def summarize(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    input_price: float,
    output_price: float,
    max_low_adequacy_rate: float = 0.5,
    max_high_risk_rate: float = 0.34,
    min_adequate_rate: float = 0.5,
) -> dict[str, Any]:
    for row in rows:
        if row.get("attributions"):
            row["profile"] = added_information_profile(
                row.get("attributions") or [],
                max_low_adequacy_rate=max_low_adequacy_rate,
                max_high_risk_rate=max_high_risk_rate,
                min_adequate_rate=min_adequate_rate,
            )
    scores = [
        float((row.get("profile") or {}).get("added_information_score"))
        for row in rows
        if (row.get("profile") or {}).get("added_information_score") is not None
    ]
    support = Counter()
    included_scores = [
        float((row.get("profile") or {}).get("added_information_score"))
        for row in rows
        if (row.get("profile") or {}).get("analysis_inclusion")
        and (row.get("profile") or {}).get("added_information_score") is not None
    ]
    excluded_scores = [
        float((row.get("profile") or {}).get("added_information_score"))
        for row in rows
        if not (row.get("profile") or {}).get("analysis_inclusion")
        and (row.get("profile") or {}).get("added_information_score") is not None
    ]
    adequacy = Counter()
    risk = Counter()
    inclusion = Counter()
    delta = Counter()
    source = Counter()
    input_tokens = output_tokens = 0
    runtimes = []
    n_attributions = 0
    for row in rows:
        source[row.get("source_corpus") or "unknown"] += 1
        profile = row.get("profile") or {}
        inclusion["included" if profile.get("analysis_inclusion") else "excluded"] += 1
        support.update(profile.get("support_counts") or {})
        adequacy.update(profile.get("evidence_adequacy_counts") or {})
        risk.update(profile.get("missing_prior_risk_counts") or {})
        delta.update(profile.get("unsupported_by_delta_type") or {})
        n_attributions += int(profile.get("n_query_acus") or 0)
        usage = row.get("llm_usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        if row.get("runtime_seconds") is not None:
            runtimes.append(float(row["runtime_seconds"]))
    return {
        "rows": len(rows),
        "errors": len(errors),
        "attributions": n_attributions,
        "rows_by_source_corpus": dict(source),
        "mean_added_information_score": statistics.mean(scores) if scores else None,
        "median_added_information_score": statistics.median(scores) if scores else None,
        "included_rows": inclusion.get("included", 0),
        "excluded_low_prior_rows": inclusion.get("excluded", 0),
        "mean_added_information_score_included": statistics.mean(included_scores) if included_scores else None,
        "median_added_information_score_included": statistics.median(included_scores) if included_scores else None,
        "mean_added_information_score_excluded": statistics.mean(excluded_scores) if excluded_scores else None,
        "support_counts": dict(support),
        "evidence_adequacy_counts": dict(adequacy),
        "missing_prior_risk_counts": dict(risk),
        "support_percentages": {
            key: value / n_attributions if n_attributions else 0.0
            for key, value in sorted(support.items())
        },
        "unsupported_by_delta_type": dict(delta),
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "estimated_input_cost": input_tokens / 1_000_000 * input_price,
        "estimated_output_cost": output_tokens / 1_000_000 * output_price,
        "estimated_total_cost": input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price,
        "runtime_seconds_mean": statistics.mean(runtimes) if runtimes else None,
        "runtime_seconds_total_observed": sum(runtimes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DCU-native fulltext added-information attribution.")
    parser.add_argument("--dataset-bank-jsonl", required=True)
    parser.add_argument("--acu-bank-jsonl", required=True)
    parser.add_argument("--precomputed-retrieval-jsonl", default=None)
    parser.add_argument("--query-year", type=int, default=None)
    parser.add_argument("--year-mode", choices=["earlier", "earlier_or_same", "any"], default="earlier")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["first", "random", "stratified_year"], default="stratified_year")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--attribution-mode", choices=["per_acu", "dataset_compact"], default="per_acu")
    parser.add_argument("--max-low-adequacy-rate", type=float, default=0.5)
    parser.add_argument("--max-high-risk-rate", type=float, default=0.34)
    parser.add_argument("--min-adequate-rate", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--hybrid-dcu-embedding-cache", default="data/census/integrated_fulltext_acu_bank_2023_2025_hybrid_dcu_minilm_embeddings.npz")
    parser.add_argument("--hybrid-dcu-dense-batch-size", type=int, default=128)
    parser.add_argument("--hybrid-dcu-max-text-chars", type=int, default=1600)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--error-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--input-price-per-1m", type=float, default=0.75)
    parser.add_argument("--output-price-per-1m", type=float, default=4.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite:
        for path in [args.output_jsonl, args.error_jsonl, args.summary_json]:
            Path(path).unlink(missing_ok=True)

    if args.precomputed_retrieval_jsonl:
        retrieval_rows = read_jsonl(args.precomputed_retrieval_jsonl)
        if args.query_year is not None:
            retrieval_rows = [row for row in retrieval_rows if row.get("query_year") == args.query_year]
        rows = select_rows(retrieval_rows, limit=args.limit, sample_mode=args.sample_mode, sample_seed=args.sample_seed)
        acu_rows = None
    else:
        dataset_rows = load_query_rows(
            args.dataset_bank_jsonl,
            query_year=args.query_year,
            limit=args.limit,
            sample_mode=args.sample_mode,
        )
        acu_rows = read_jsonl(args.acu_bank_jsonl)
        rows = select_rows(dataset_rows, limit=args.limit, sample_mode=args.sample_mode, sample_seed=args.sample_seed)
    done = existing_ids(args.output_jsonl)
    rows = [row for row in rows if row_bank_id(row) not in done]

    dataset_by_bank_id = {
        str(row.get("bank_id") or ""): row
        for row in read_jsonl(args.dataset_bank_jsonl)
        if row.get("bank_id")
    }
    hybrid_index = None
    if acu_rows is not None:
        hybrid_index = queue_builder.HybridDcuIndex(
            acu_rows,
            dataset_by_bank_id,
            embedding_cache=args.hybrid_dcu_embedding_cache,
            dense_batch_size=args.hybrid_dcu_dense_batch_size,
            max_text_chars=args.hybrid_dcu_max_text_chars,
        )

    print(json.dumps({
        "dataset_bank_jsonl": args.dataset_bank_jsonl,
        "acu_bank_jsonl": args.acu_bank_jsonl,
        "precomputed_retrieval_jsonl": args.precomputed_retrieval_jsonl,
        "remaining_rows": len(rows),
        "acu_corpus_rows": len(acu_rows) if acu_rows is not None else None,
        "model": args.model,
        "candidate_depth": args.candidate_depth,
        "attribution_mode": args.attribution_mode,
        "max_low_adequacy_rate": args.max_low_adequacy_rate,
        "max_high_risk_rate": args.max_high_risk_rate,
        "min_adequate_rate": args.min_adequate_rate,
        "year_mode": args.year_mode,
        "workers": args.workers,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
    }, ensure_ascii=False, indent=2), flush=True)

    output_rows = read_jsonl(args.output_jsonl)
    error_rows = read_jsonl(args.error_jsonl)
    processed = 0
    failed = 0

    def run_with_retries(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        last_error = ""
        for _attempt in range(args.max_retries + 1):
            try:
                if args.attribution_mode == "dataset_compact":
                    return True, run_one_dataset_compact(
                        row,
                        model=args.model,
                        acu_rows=acu_rows,
                        dataset_by_bank_id=dataset_by_bank_id,
                        hybrid_index=hybrid_index,
                        year_mode=args.year_mode,
                        candidate_depth=args.candidate_depth,
                        min_score=args.min_score,
                        max_low_adequacy_rate=args.max_low_adequacy_rate,
                        max_high_risk_rate=args.max_high_risk_rate,
                        min_adequate_rate=args.min_adequate_rate,
                    )
                return True, run_one(
                    row,
                    model=args.model,
                    acu_rows=acu_rows,
                    dataset_by_bank_id=dataset_by_bank_id,
                    hybrid_index=hybrid_index,
                    year_mode=args.year_mode,
                    candidate_depth=args.candidate_depth,
                    min_score=args.min_score,
                    max_low_adequacy_rate=args.max_low_adequacy_rate,
                    max_high_risk_rate=args.max_high_risk_rate,
                    min_adequate_rate=args.min_adequate_rate,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                time.sleep(1.0)
        return False, {
            "query_bank_id": row_bank_id(row),
            "query_paper_id": row.get("query_paper_id") or row.get("paper_id") or "",
            "query_dataset_name": row.get("query_dataset_name") or row.get("dataset_name") or "",
            "error": last_error,
        }

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_with_retries, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            ok, payload = future.result()
            processed += 1
            if ok:
                output_rows.append(payload)
                append_jsonl(args.output_jsonl, payload)
            else:
                failed += 1
                error_rows.append(payload)
                append_jsonl(args.error_jsonl, payload)
            elapsed = time.monotonic() - started
            print(json.dumps({
                "processed": processed,
                "total": len(rows),
                "failed": failed,
                "elapsed_seconds": round(elapsed, 1),
                "avg_seconds_per_row": round(elapsed / processed, 2),
            }, ensure_ascii=False), flush=True)

    summary = summarize(
        output_rows,
        error_rows,
        input_price=args.input_price_per_1m,
        output_price=args.output_price_per_1m,
        max_low_adequacy_rate=args.max_low_adequacy_rate,
        max_high_risk_rate=args.max_high_risk_rate,
        min_adequate_rate=args.min_adequate_rate,
    )
    summary.update({
        "dataset_bank_jsonl": args.dataset_bank_jsonl,
        "acu_bank_jsonl": args.acu_bank_jsonl,
        "query_year": args.query_year,
        "year_mode": args.year_mode,
        "model": args.model,
        "candidate_depth": args.candidate_depth,
        "attribution_mode": args.attribution_mode,
        "max_low_adequacy_rate": args.max_low_adequacy_rate,
        "max_high_risk_rate": args.max_high_risk_rate,
        "min_adequate_rate": args.min_adequate_rate,
        "hybrid_dcu_embedding_cache": args.hybrid_dcu_embedding_cache,
        "output_jsonl": args.output_jsonl,
        "error_jsonl": args.error_jsonl,
    })
    write_json(args.summary_json, summary)
    print(json.dumps({
        "output_jsonl": args.output_jsonl,
        "error_jsonl": args.error_jsonl,
        "summary_json": args.summary_json,
        **summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
