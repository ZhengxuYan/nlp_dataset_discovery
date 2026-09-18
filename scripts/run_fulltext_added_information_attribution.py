#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")


SUPPORT_STATUSES = {"supported", "partially_supported", "unsupported", "contradicted", "not_comparable"}
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


PROMPT = """You are an expert NLP researcher auditing dataset added information.

Your role is evidence attribution, not novelty labeling. For each query ACU, decide whether the prior-support ACUs already support the claim.

Support statuses:
- supported: the query ACU is directly supported by one or more prior ACUs.
- partially_supported: the prior ACUs support part of the claim, but the query ACU adds a meaningful detail, extension, or change.
- unsupported: the query ACU is not supported by the prior ACUs.
- contradicted: the prior ACUs directly conflict with the query ACU.
- not_comparable: the query ACU cannot be compared to the supplied prior ACUs.

Delta types:
- task/domain
- data/source
- annotation/protocol
- scale/coverage
- evaluation/use
- availability/quality
- governance/ethics
- other

Importance:
- low: incidental metadata or minor implementation detail.
- medium: useful dataset detail.
- high: central contribution claim.

Rules:
- Return exactly one attribution for each query ACU ID.
- Use only prior ACU IDs from the supplied prior list.
- If no prior ACU supports the query ACU, use unsupported and leave best_prior_acu_ids empty.
- The rationale must explicitly mention the query claim and, when applicable, the selected prior ACU content.
- Separately judge whether the supplied prior-support set is adequate for this query. Do not lower an ACU support label only because prior work may be missing.
- Do not output an overall novelty label.

Prior-set adequacy:
- high: supplied priors include explicit source/comparison datasets, close task/domain matches, or strong direct antecedents.
- medium: supplied priors are related and useful but may miss some direct antecedents.
- low: supplied priors are mostly weak topical neighbors, sparse, or unlikely to include the strongest prior work.

Missing-prior risk:
- low: unlikely that an omitted prior would substantially change many support labels.
- medium: plausible that omitted priors would change some support labels.
- high: likely that important prior work is missing, so the added-information score should be treated as low-confidence.

Query dataset: {query_dataset_name}
Query paper: {query_title}

Query ACUs:
{query_acus}

Prior-support ACUs:
{prior_acus}

Return exactly this JSON shape with no markdown fences:
{{"prior_set_adequacy":"high|medium|low","missing_prior_risk":"low|medium|high","prior_set_rationale":"...","attributions":[{{"query_acu_id":"q0","query_acu":"...","support_status":"supported|partially_supported|unsupported|contradicted|not_comparable","best_prior_acu_ids":["p0"],"delta_type":"task/domain|data/source|annotation/protocol|scale/coverage|evaluation/use|availability/quality|governance/ethics|other","importance":"low|medium|high","rationale":"..."}}]}}
"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
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
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(candidate)
        raise


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        payload = usage.model_dump()
    elif isinstance(usage, dict):
        payload = usage
    else:
        payload = {
            key: getattr(usage, key)
            for key in ["input_tokens", "output_tokens", "total_tokens"]
            if hasattr(usage, key)
        }
    return {
        "input_tokens": int(payload.get("input_tokens") or payload.get("prompt_tokens") or 0),
        "output_tokens": int(payload.get("output_tokens") or payload.get("completion_tokens") or 0),
        "total_tokens": int(payload.get("total_tokens") or 0),
    }


def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required.") from exc
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def query_acus(row: dict[str, Any]) -> list[dict[str, str]]:
    output = []
    for idx, acu in enumerate(row.get("query_acus") or []):
        if not isinstance(acu, dict):
            continue
        text = str(acu.get("text") or acu.get("acu_text") or "").strip()
        if text:
            output.append({"id": f"q{idx}", "text": text})
    return output


def prior_acus(row: dict[str, Any], *, top_prior_candidates: int, max_prior_acus: int) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for candidate in (row.get("prior_candidates") or [])[:top_prior_candidates]:
        for acu in candidate.get("matched_acus") or []:
            text = str(acu.get("acu_text") or "").strip()
            if not text:
                continue
            key = acu.get("acu_global_id") or text
            if key in seen:
                continue
            seen.add(key)
            output.append({
                "id": f"p{len(output)}",
                "text": text,
                "acu_global_id": acu.get("acu_global_id") or "",
                "prior_bank_id": candidate.get("prior_bank_id") or "",
                "prior_dataset_name": candidate.get("prior_dataset_name") or "",
                "prior_title": candidate.get("prior_title") or "",
                "prior_year": candidate.get("prior_year"),
                "retrieval_score": acu.get("score"),
            })
            if len(output) >= max_prior_acus:
                return output
    return output


def build_prompt(row: dict[str, Any], q_acus: list[dict[str, str]], p_acus: list[dict[str, Any]]) -> str:
    return PROMPT.format(
        query_dataset_name=row.get("query_dataset_name") or "",
        query_title=row.get("query_title") or "",
        query_acus="\n".join(f"- {acu['id']}: {acu['text']}" for acu in q_acus) or "- None",
        prior_acus="\n".join(
            f"- {acu['id']}: {acu['text']} "
            f"[dataset={acu.get('prior_dataset_name')}; year={acu.get('prior_year')}]"
            for acu in p_acus
        ) or "- None",
    )


def normalize_attributions(payload: dict[str, Any], q_acus: list[dict[str, str]], p_acus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {acu["id"]: acu["text"] for acu in q_acus}
    allowed_prior = {acu["id"] for acu in p_acus}
    rows = payload.get("attributions")
    if not isinstance(rows, list):
        raise ValueError("Missing attributions list.")
    normalized = []
    seen = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("query_acu_id") or "")
        if qid not in expected:
            raise ValueError(f"Unknown query_acu_id: {qid}")
        if qid in seen:
            raise ValueError(f"Duplicate query_acu_id: {qid}")
        seen.add(qid)
        support_status = str(item.get("support_status") or "unsupported")
        if support_status not in SUPPORT_STATUSES:
            support_status = "unsupported"
        delta_type = str(item.get("delta_type") or "other")
        if delta_type not in DELTA_TYPES:
            delta_type = "other"
        importance = str(item.get("importance") or "medium")
        if importance not in IMPORTANCE_LEVELS:
            importance = "medium"
        best_prior_ids = [
            str(pid)
            for pid in item.get("best_prior_acu_ids") or []
            if str(pid) in allowed_prior
        ]
        normalized.append({
            "query_acu_id": qid,
            "query_acu": expected[qid],
            "support_status": support_status,
            "best_prior_acu_ids": best_prior_ids,
            "delta_type": delta_type,
            "importance": importance,
            "rationale": str(item.get("rationale") or ""),
        })
    missing = [qid for qid in expected if qid not in seen]
    if missing:
        raise ValueError(f"Missing attribution for query ACUs: {missing}")
    return normalized


def normalize_prior_set_assessment(payload: dict[str, Any]) -> dict[str, str]:
    adequacy = str(payload.get("prior_set_adequacy") or "medium").lower()
    if adequacy not in ADEQUACY_LEVELS:
        adequacy = "medium"
    risk = str(payload.get("missing_prior_risk") or "medium").lower()
    if risk not in RISK_LEVELS:
        risk = "medium"
    return {
        "prior_set_adequacy": adequacy,
        "missing_prior_risk": risk,
        "prior_set_rationale": str(payload.get("prior_set_rationale") or ""),
    }


def added_information_profile(attributions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["support_status"] for row in attributions)
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
    return {
        "n_query_acus": total,
        "added_information_score": numerator / denominator if denominator else None,
        "support_counts": dict(counts),
        "support_percentages": {
            status: counts.get(status, 0) / total if total else 0.0
            for status in ["supported", "partially_supported", "unsupported", "contradicted", "not_comparable"]
        },
        "unsupported_by_delta_type": dict(unsupported_by_delta_type),
        "excluded_from_score_count": sum(counts.get(status, 0) for status in EXCLUDED_SUPPORT_STATUSES),
    }


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
            by_year[str(row.get("query_year") or "unknown")].append(row)
        selected = []
        remaining = limit
        years = sorted(by_year)
        for idx, year in enumerate(years):
            bucket = by_year[year]
            if idx == len(years) - 1:
                take = remaining
            else:
                take = round(limit * len(bucket) / len(rows))
                take = min(take, remaining)
            selected.extend(rng.sample(bucket, min(take, len(bucket))))
            remaining = limit - len(selected)
        if len(selected) < limit:
            already = {row["query_bank_id"] for row in selected}
            rest = [row for row in rows if row.get("query_bank_id") not in already]
            selected.extend(rng.sample(rest, min(limit - len(selected), len(rest))))
        return selected[:limit]
    raise ValueError(f"Unknown sample mode: {sample_mode}")


def existing_ids(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    ids = set()
    query_bank_id_pattern = re.compile(r'"query_bank_id"\s*:\s*"([^"]+)"')
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                match = query_bank_id_pattern.search(line)
                if match:
                    ids.add(match.group(1))
                    continue
                try:
                    ids.add(json.loads(line).get("query_bank_id"))
                except Exception:
                    continue
    return {str(value) for value in ids if value}


def run_one(row: dict[str, Any], *, model: str, top_prior_candidates: int, max_prior_acus: int) -> dict[str, Any]:
    start = time.time()
    q_acus = query_acus(row)
    p_acus = prior_acus(row, top_prior_candidates=top_prior_candidates, max_prior_acus=max_prior_acus)
    if not q_acus:
        raise ValueError("No query ACUs.")
    if not p_acus:
        raise ValueError("No prior ACUs.")
    prompt = build_prompt(row, q_acus, p_acus)
    client = get_openai_client()
    response = client.responses.create(model=model.removeprefix("openai/"), input=prompt)
    parsed = parse_json_object(response.output_text)
    attributions = normalize_attributions(parsed, q_acus, p_acus)
    prior_set_assessment = normalize_prior_set_assessment(parsed)
    return {
        "query_bank_id": row.get("query_bank_id") or "",
        "query_paper_id": row.get("query_paper_id") or "",
        "query_dataset_id": row.get("query_dataset_id") or "",
        "query_dataset_name": row.get("query_dataset_name") or "",
        "query_title": row.get("query_title") or "",
        "query_year": row.get("query_year"),
        "model": model,
        "query_acus": q_acus,
        "prior_acus": p_acus,
        "prior_set_assessment": prior_set_assessment,
        "attributions": attributions,
        "profile": added_information_profile(attributions),
        "llm_usage": usage_dict(response),
        "runtime_seconds": round(time.time() - start, 3),
    }


def summarize_output(rows: list[dict[str, Any]], errors: list[dict[str, Any]], *, input_price: float, output_price: float) -> dict[str, Any]:
    scores = [
        float((row.get("profile") or {}).get("added_information_score"))
        for row in rows
        if (row.get("profile") or {}).get("added_information_score") is not None
    ]
    support = defaultdict(list)
    unsupported_by_delta_type: Counter = Counter()
    input_tokens = 0
    output_tokens = 0
    runtimes = []
    adequacy_counts: Counter = Counter()
    risk_counts: Counter = Counter()
    scores_by_adequacy: dict[str, list[float]] = defaultdict(list)
    scores_by_risk: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        profile = row.get("profile") or {}
        score = profile.get("added_information_score")
        assessment = row.get("prior_set_assessment") or {}
        adequacy = assessment.get("prior_set_adequacy") or "unknown"
        risk = assessment.get("missing_prior_risk") or "unknown"
        adequacy_counts[adequacy] += 1
        risk_counts[risk] += 1
        if score is not None:
            scores_by_adequacy[adequacy].append(float(score))
            scores_by_risk[risk].append(float(score))
        for status, value in (profile.get("support_percentages") or {}).items():
            support[status].append(float(value))
        unsupported_by_delta_type.update(profile.get("unsupported_by_delta_type") or {})
        usage = row.get("llm_usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        if row.get("runtime_seconds"):
            runtimes.append(float(row["runtime_seconds"]))
    return {
        "rows": len(rows),
        "errors": len(errors),
        "mean_added_information_score": statistics.mean(scores) if scores else None,
        "median_added_information_score": statistics.median(scores) if scores else None,
        "mean_support_percentages": {
            status: statistics.mean(values) if values else 0.0
            for status, values in sorted(support.items())
        },
        "prior_set_adequacy_counts": dict(adequacy_counts),
        "missing_prior_risk_counts": dict(risk_counts),
        "mean_score_by_prior_set_adequacy": {
            key: statistics.mean(values) if values else None
            for key, values in sorted(scores_by_adequacy.items())
        },
        "mean_score_by_missing_prior_risk": {
            key: statistics.mean(values) if values else None
            for key, values in sorted(scores_by_risk.items())
        },
        "unsupported_by_delta_type": dict(unsupported_by_delta_type),
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "estimated_input_cost": input_tokens / 1_000_000 * input_price,
        "estimated_output_cost": output_tokens / 1_000_000 * output_price,
        "estimated_total_cost": input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price,
        "runtime_seconds_mean": statistics.mean(runtimes) if runtimes else None,
        "runtime_seconds_total_observed": sum(runtimes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM added-information attribution from a full-text prior candidate queue.")
    parser.add_argument("--queue-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["first", "random", "stratified_year"], default="stratified_year")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--query-year", type=int, default=None)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-prior-candidates", type=int, default=10)
    parser.add_argument("--max-prior-acus", type=int, default=40)
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

    queue_rows = read_jsonl(args.queue_jsonl)
    if args.query_year is not None:
        queue_rows = [row for row in queue_rows if row.get("query_year") == args.query_year]
    rows = select_rows(
        queue_rows,
        limit=args.limit,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
    )
    done = existing_ids(args.output_jsonl)
    rows = [row for row in rows if str(row.get("query_bank_id") or "") not in done]
    print(json.dumps({
        "queue_jsonl": args.queue_jsonl,
        "remaining_rows": len(rows),
        "model": args.model,
        "workers": args.workers,
        "top_prior_candidates": args.top_prior_candidates,
        "max_prior_acus": args.max_prior_acus,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
    }, ensure_ascii=False, indent=2), flush=True)

    processed = 0
    failed = 0
    output_rows = read_jsonl(args.output_jsonl) if Path(args.output_jsonl).exists() else []
    error_rows = read_jsonl(args.error_jsonl) if Path(args.error_jsonl).exists() else []

    def run_with_retries(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        last_error = ""
        for attempt in range(args.max_retries + 1):
            try:
                return True, run_one(
                    row,
                    model=args.model,
                    top_prior_candidates=args.top_prior_candidates,
                    max_prior_acus=args.max_prior_acus,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt < args.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        return False, {
            "query_bank_id": row.get("query_bank_id") or "",
            "query_dataset_name": row.get("query_dataset_name") or "",
            "error": last_error,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_with_retries, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            ok, payload = future.result()
            if ok:
                processed += 1
                output_rows.append(payload)
                append_jsonl(args.output_jsonl, payload)
            else:
                failed += 1
                error_rows.append(payload)
                append_jsonl(args.error_jsonl, payload)
            summary = summarize_output(
                output_rows,
                error_rows,
                input_price=args.input_price_per_1m,
                output_price=args.output_price_per_1m,
            )
            write_json(args.summary_json, summary)
            print(json.dumps({
                "processed": processed,
                "failed": failed,
                "total_output_rows": len(output_rows),
                "avg_runtime_seconds": round(summary["runtime_seconds_mean"] or 0, 3),
                "estimated_total_cost": round(summary["estimated_total_cost"], 4),
            }, ensure_ascii=False), flush=True)

    final_summary = summarize_output(
        output_rows,
        error_rows,
        input_price=args.input_price_per_1m,
        output_price=args.output_price_per_1m,
    )
    write_json(args.summary_json, final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
