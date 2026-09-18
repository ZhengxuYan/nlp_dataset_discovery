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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "--allow-model-download" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from scv.benchmark_models import BenchmarkDraftRecord
from scv.benchmark_store import (
    load_benchmark_drafts,
    load_previous_work_candidates,
    load_processed_bank,
)


try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

SentenceTransformer = None  # type: ignore
torch = None  # type: ignore
F = None  # type: ignore
AutoModel = None  # type: ignore
AutoModelForCausalLM = None  # type: ignore
AutoTokenizer = None  # type: ignore
OpenAI = None  # type: ignore
genai = None  # type: ignore
genai_types = None  # type: ignore

try:
    from bespokelabs import curator
except ImportError:
    curator = None  # type: ignore


DEFAULT_DRAFTS = "data/benchmark/benchmark_drafts.jsonl"
DEFAULT_PROCESSED_BANK = "data/benchmark/processed_bank.jsonl"
DEFAULT_PREVIOUS_WORK = "data/benchmark/previous_work_candidates.jsonl"
DEFAULT_RETRIEVAL_CACHE = "data/benchmark/retrieval_cache"


ORDINAL_LABELS = ["repackaging", "incremental", "substantial"]
ORDINAL_TO_INT = {label: idx for idx, label in enumerate(ORDINAL_LABELS)}


@dataclass
class CandidateRecord:
    candidate_id: str
    name: str
    acus: List[str]
    summary_text: str
    domain: str = ""
    role: str = ""
    source_dataset: str = ""
    is_cited: bool = False


@dataclass
class DCURecord:
    dcu_id: str
    paper_id: str
    dataset_name: str
    paper_title: str
    text: str
    acu_type: str = ""
    role: str = ""
    resource_type: str = ""
    tasks: Tuple[str, ...] = ()
    domains: Tuple[str, ...] = ()
    languages: Tuple[str, ...] = ()
    modalities: Tuple[str, ...] = ()
    source_datasets: Tuple[str, ...] = ()
    year: Optional[int] = None

    def serialized(self) -> str:
        parts = [
            f"dataset: {self.dataset_name}",
            f"title: {self.paper_title}",
            f"type: {self.acu_type}",
            f"role: {self.role}",
            f"resource_type: {self.resource_type}",
            f"tasks: {'; '.join(self.tasks)}",
            f"domains: {'; '.join(self.domains)}",
            f"languages: {'; '.join(self.languages)}",
            f"modalities: {'; '.join(self.modalities)}",
            f"sources: {'; '.join(self.source_datasets)}",
            f"claim: {self.text}",
        ]
        return "\n".join(part for part in parts if part.split(": ", 1)[-1])


def tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def reciprocal_rank(rank: Optional[int]) -> float:
    if not rank or rank <= 0:
        return 0.0
    return 1.0 / rank


def ordinal_from_score(score: float) -> str:
    if score < 0.2:
        return "repackaging"
    if score < 0.55:
        return "incremental"
    return "substantial"


def evaluate_retrieval_run(ranked_candidate_ids: Sequence[str], gold_support_ids: Sequence[str]) -> Dict[str, float]:
    gold = list(dict.fromkeys(gold_support_ids))
    ranked = list(ranked_candidate_ids)
    if not gold:
        return {"mrr": 0.0, "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0}
    first_hit_rank = None
    for rank, candidate_id in enumerate(ranked, start=1):
        if candidate_id in gold:
            first_hit_rank = rank
            break
    return {
        "mrr": reciprocal_rank(first_hit_rank),
        "recall@1": 1.0 if any(candidate_id in gold for candidate_id in ranked[:1]) else 0.0,
        "recall@3": 1.0 if any(candidate_id in gold for candidate_id in ranked[:3]) else 0.0,
        "recall@5": 1.0 if any(candidate_id in gold for candidate_id in ranked[:5]) else 0.0,
    }


def summarize_metric_runs(metric_rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(metric_rows)
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: safe_mean([row.get(key, 0.0) for row in rows]) for key in keys}


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    import math
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = safe_mean(left_ranks)
    right_mean = safe_mean(right_ranks)
    num = sum((l - left_mean) * (r - right_mean) for l, r in zip(left_ranks, right_ranks))
    den_left = math.sqrt(sum((l - left_mean) ** 2 for l in left_ranks))
    den_right = math.sqrt(sum((r - right_mean) ** 2 for r in right_ranks))
    denom = den_left * den_right
    return float(num / denom) if denom else 0.0


class AddedInformationLabelOutput(BaseModel):
    label: Literal["repackaging", "incremental", "substantial"] = Field(
        description="Ordinal added-information label."
    )
    score: float = Field(
        description="Continuous score from 0.0 fully supported/repackaging to 1.0 substantial added information."
    )
    rationale: str = Field(description="Brief rationale grounded in the supplied ACUs.")


SupportStatus = Literal["supported", "partially_supported", "unsupported", "contradicted", "not_comparable"]
DeltaType = Literal[
    "task/domain",
    "data/source",
    "annotation/protocol",
    "scale/coverage",
    "evaluation/use",
    "availability/quality",
    "other",
]
Importance = Literal["low", "medium", "high"]

SUPPORT_DELTA_VALUES = {
    "supported": 0.0,
    "partially_supported": 0.5,
    "unsupported": 1.0,
}
EXCLUDED_SUPPORT_STATUSES = {"contradicted", "not_comparable"}
IMPORTANCE_WEIGHTS = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5,
}


class ClaimAttributionOutput(BaseModel):
    query_acu_id: str = Field(description="ID of the query ACU, for example q0.")
    query_acu: str = Field(description="The exact query ACU being evaluated.")
    support_status: SupportStatus = Field(description="How well prior ACUs support this query ACU.")
    best_prior_acu_ids: List[str] = Field(
        default_factory=list,
        description="IDs of the most relevant prior ACUs, for example p0 and p3.",
    )
    delta_type: DeltaType = Field(description="Primary type of unsupported or partially supported delta.")
    importance: Importance = Field(description="Importance of this ACU to the dataset contribution.")
    rationale: str = Field(description="Grounded rationale comparing the query ACU to selected prior ACUs.")


class AddedInformationAttributionOutput(BaseModel):
    attributions: List[ClaimAttributionOutput] = Field(
        description="One attribution decision for each query ACU."
    )


class PairwiseSupportScoreOutput(BaseModel):
    support_level: Literal["strong", "partial", "weak", "none"] = Field(
        description="How strongly the candidate prior paper supports the query dataset contribution claims."
    )
    supported_query_dcu_ids: List[str] = Field(
        default_factory=list,
        description="Query DCU IDs directly or partially supported by the candidate paper.",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence from 0.0 to 1.0 that this candidate is true prior-support evidence.",
    )
    rationale: str = Field(description="Brief evidence-grounded rationale.")


ATTRIBUTION_PROMPT = """You are an expert NLP researcher auditing dataset added information.

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
- Do not output an overall novelty label.

Query dataset: {query_dataset_name}

Query ACUs:
{query_acus}

Prior-support ACUs:
{prior_acus}

Return structured output only.
"""


ADDED_INFORMATION_PROMPT = """You are evaluating how much added information a query dataset contributes relative to prior-support ACUs.

Labels:
- repackaging: mostly reformats, filters, or lightly repackages the prior support.
- incremental: adds a modest but real extension, annotation layer, scale/domain/language/task variation, or combination.
- substantial: introduces a clearly new construction, capability, task setting, modality, annotation structure, or large new resource.

Query dataset: {query_dataset_name}

Query ACUs:
{query_acus}

Prior-support ACUs:
{prior_acus}

Return structured output only.
"""


if curator is not None:
    class AddedInformationLabeler(curator.LLM):
        response_format = AddedInformationLabelOutput

        def prompt(self, input: dict) -> str:
            return ADDED_INFORMATION_PROMPT.format(
                query_dataset_name=input["query_dataset_name"],
                query_acus="\n".join(f"- {acu}" for acu in input.get("query_acus", [])) or "- None",
                prior_acus="\n".join(f"- {acu}" for acu in input.get("prior_acus", [])) or "- None",
            )


    class AddedInformationAttributor(curator.LLM):
        response_format = AddedInformationAttributionOutput

        def prompt(self, input: dict) -> str:
            return ATTRIBUTION_PROMPT.format(
                query_dataset_name=input["query_dataset_name"],
                query_acus="\n".join(
                    f"- {acu['id']}: {acu['text']}"
                    for acu in input.get("query_acus_with_ids", [])
                ) or "- None",
                prior_acus="\n".join(
                    f"- {acu['id']}: {acu['text']}"
                    for acu in input.get("prior_acus_with_ids", [])
                ) or "- None",
            )
else:
    AddedInformationLabeler = None  # type: ignore
    AddedInformationAttributor = None  # type: ignore


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def stable_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:24]


def progress_log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), file=sys.stderr, flush=True)


class JsonCache:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hits: Counter[str] = Counter()
        self.misses: Counter[str] = Counter()
        self.writes: Counter[str] = Counter()

    def path(self, namespace: str, key_payload) -> Path:
        path = self.base_dir / namespace
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{stable_hash(key_payload)}.json"

    def get(self, namespace: str, key_payload):
        path = self.path(namespace, key_payload)
        if not path.exists():
            self.misses[namespace] += 1
            return None
        self.hits[namespace] += 1
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, namespace: str, key_payload, value):
        path = self.path(namespace, key_payload)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.writes[namespace] += 1
        return value

    def stats(self) -> dict:
        return {
            "hits": dict(self.hits),
            "misses": dict(self.misses),
            "writes": dict(self.writes),
        }


def default_rank_checkpoint_path(cache_dir: str | Path, run_key: str) -> Path:
    return Path(cache_dir) / "retrieval_run_checkpoints" / f"{run_key}.jsonl"


def load_rank_checkpoints(path: str | Path | None, run_key: str) -> Dict[tuple[str, str], dict]:
    if not path:
        return {}
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return {}
    rows: Dict[tuple[str, str], dict] = {}
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_key") != run_key:
                continue
            method = str(row.get("method") or "")
            query_paper_id = str(row.get("query_paper_id") or "")
            if method and query_paper_id:
                rows[(method, query_paper_id)] = row
    return rows


def append_rank_checkpoint(path: str | Path | None, row: dict) -> None:
    if not path:
        return
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def seconds_to_hms(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def complete_drafts(drafts_path: str) -> List[BenchmarkDraftRecord]:
    return [
        draft for draft in load_benchmark_drafts(Path(drafts_path))
        if draft.draft_status == "complete"
        and draft.gold_prior_paper_ids
    ]


def read_jsonl_dicts(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def acu_text(acu) -> str:
    if isinstance(acu, dict):
        return str(acu.get("text") or "")
    return str(acu or "")


def acu_type(acu) -> str:
    if isinstance(acu, dict):
        return str(acu.get("type") or acu.get("acu_type") or "")
    return ""


def dataset_record_name(dataset: dict) -> str:
    identity = dataset.get("dataset_identity") or {}
    return str(dataset.get("name") or identity.get("canonical_name") or dataset.get("dataset_id") or "")


def list_strings(value: object) -> Tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item or "").strip())


def source_dataset_names(dataset: dict) -> Tuple[str, ...]:
    construction = dataset.get("construction") or {}
    names: List[str] = []
    for source in construction.get("source_datasets") or []:
        if isinstance(source, dict) and source.get("name"):
            names.append(str(source["name"]))
    if dataset.get("source_dataset"):
        names.append(str(dataset["source_dataset"]))
    return tuple(unique(names))


def infer_acu_type(text: str) -> str:
    """Best-effort type recovery for benchmark rows that only preserve ACU text."""
    tokens = set(tokenize(text))
    lowered = (text or "").lower()
    rules = [
        (
            "scale/coverage",
            {
                "span", "spans", "cover", "covers", "coverage", "languages", "language",
                "instances", "examples", "samples", "tokens", "documents", "scripts",
                "domains", "large", "scale", "million", "billion", "k", "hours",
            },
        ),
        (
            "annotation/protocol",
            {
                "annotation", "annotated", "annotator", "annotators", "label", "labels",
                "labeled", "labelling", "guidelines", "protocol", "quality", "agreement",
                "verified", "verification", "human", "expert", "crowd", "crowdsourced",
            },
        ),
        (
            "data/source",
            {
                "source", "sources", "collected", "collection", "crawl", "crawled",
                "generated", "synthetic", "translated", "derived", "extracted", "mined",
                "from", "web", "social", "llm", "model", "template", "templates",
            },
        ),
        (
            "evaluation/use",
            {
                "benchmark", "evaluate", "evaluation", "test", "testing", "metric",
                "metrics", "baseline", "baselines", "leaderboard", "assess", "measure",
            },
        ),
        (
            "availability/quality",
            {
                "released", "available", "license", "licensed", "github", "huggingface",
                "quality", "filtering", "filtered", "deduplicated", "cleaned",
            },
        ),
        (
            "governance/ethics",
            {
                "ethics", "ethical", "privacy", "pii", "consent", "copyright", "bias",
                "fairness", "safety", "harm", "toxic", "toxicity",
            },
        ),
    ]
    if re.search(r"\b\d+([.,]\d+)?\s*(k|m|b|million|billion|languages?|scripts?|tokens?|instances?|examples?|documents?|hours?)\b", lowered):
        return "scale/coverage"
    best_type = ""
    best_hits = 0
    for acu_type_name, keywords in rules:
        hits = len(tokens & keywords)
        if hits > best_hits:
            best_type = acu_type_name
            best_hits = hits
    if best_type:
        return best_type
    return "task/domain"


def payload_year(payload: dict, paper_id: str = "") -> Optional[int]:
    value = payload.get("year")
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?:ACL:)?(20\d{2})[.:-]", paper_id or "")
    if match:
        return int(match.group(1))
    return None


def draft_year(draft: BenchmarkDraftRecord) -> Optional[int]:
    match = re.search(r"(?:ACL:)?(20\d{2})[.:-]", draft.query_paper_id or "")
    return int(match.group(1)) if match else None


def complete_acl_drafts(path: str) -> List[BenchmarkDraftRecord]:
    drafts: List[BenchmarkDraftRecord] = []
    for row in read_jsonl_dicts(path):
        if row.get("annotation_status") != "complete":
            continue
        query_acus = [acu_text(acu) for acu in row.get("query_acus") or [] if acu_text(acu)]
        prior_acus = [acu_text(acu) for acu in row.get("gold_prior_support_acus") or [] if acu_text(acu)]
        gold_ids = [str(pid) for pid in row.get("gold_prior_paper_ids") or [] if pid]
        if not query_acus or not prior_acus or not gold_ids:
            continue
        drafts.append(BenchmarkDraftRecord(
            query_paper_id=str(row.get("benchmark_id") or row.get("query_dataset_id") or row.get("query_paper_id")),
            query_dataset_name=str(row.get("query_dataset_name") or ""),
            query_acus=query_acus,
            gold_prior_paper_ids=gold_ids,
            gold_prior_dataset_names=[
                str(ref.get("candidate_dataset_name") or "")
                for ref in row.get("gold_prior_support_refs") or []
                if ref.get("candidate_dataset_name")
            ],
            gold_prior_support_acus=prior_acus,
            draft_status="complete",
            annotation_notes=json.dumps({
                "source": "ACL citation-grounded benchmark row.",
                "query_title": row.get("query_title") or "",
                "query_year": row.get("query_year"),
                "query_dataset_role": row.get("query_dataset_role") or "",
                "query_dataset_resource_type": row.get("query_dataset_resource_type") or "",
                "coverage": row.get("coverage") or {},
                "construction": row.get("construction") or {},
            }, ensure_ascii=False),
        ))
    return drafts


def processed_payloads(processed_bank_path: str) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for row in load_processed_bank(Path(processed_bank_path)):
        if row.processing_status != "processed":
            continue
        path = Path(row.processed_json_path)
        if not path.exists():
            continue
        try:
            payloads[row.paper_id] = load_json(path)
        except Exception:
            continue
    return payloads


def fulltext_extraction_payloads(path: str) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for row in read_jsonl_dicts(path):
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            payloads[paper_id] = row
    return payloads


def processed_to_candidate(paper_id: str, payload: dict, *, is_cited: bool = False) -> CandidateRecord:
    dataset_names: List[str] = []
    acus: List[str] = []
    summary_parts: List[str] = []
    domains: List[str] = []
    roles: List[str] = []
    sources: List[str] = []

    for dataset in payload.get("datasets") or []:
        name = dataset_record_name(dataset)
        if name:
            dataset_names.append(name)
        acus.extend([acu_text(acu) for acu in dataset.get("acus") or [] if acu_text(acu)])
        for key in ("usage_description", "added_information_summary", "novelty_summary"):
            if dataset.get(key):
                summary_parts.append(str(dataset[key]))
        if dataset.get("domain"):
            domains.append(str(dataset["domain"]))
        if dataset.get("role"):
            roles.append(str(dataset["role"]))
        if dataset.get("source_dataset"):
            sources.append(str(dataset["source_dataset"]))

    title = payload.get("title") or paper_id
    summary_text = " ".join([
        title,
        payload.get("contribution_summary") or payload.get("paper_contribution_summary") or "",
        " ".join(dataset_names),
        " ".join(acus),
        " ".join(summary_parts),
    ])
    return CandidateRecord(
        candidate_id=paper_id,
        name="; ".join(dataset_names) or title,
        acus=unique(acus),
        summary_text=summary_text,
        domain=domains[0] if domains else "",
        role=roles[0] if roles else "",
        source_dataset=sources[0] if sources else "",
        is_cited=is_cited,
    )


def query_text_for_draft(draft: BenchmarkDraftRecord) -> str:
    return " ".join([draft.query_dataset_name, *draft.query_acus])


def draft_metadata(draft: BenchmarkDraftRecord) -> dict:
    try:
        data = json.loads(draft.annotation_notes or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def query_paper_summary_for_draft(draft: BenchmarkDraftRecord) -> str:
    metadata = draft_metadata(draft)
    title = str(metadata.get("query_title") or draft.query_paper_id)
    coverage = metadata.get("coverage") or {}
    construction = metadata.get("construction") or {}
    parts = [
        f"Paper ID: {draft.query_paper_id}",
        f"Paper title: {title}",
        f"Dataset: {draft.query_dataset_name}",
    ]
    if metadata.get("query_dataset_role"):
        parts.append(f"Dataset role: {metadata['query_dataset_role']}")
    if metadata.get("query_dataset_resource_type"):
        parts.append(f"Resource type: {metadata['query_dataset_resource_type']}")
    for label, key in [
        ("Tasks", "tasks"),
        ("Domains", "domains"),
        ("Languages", "languages"),
        ("Modalities", "modality"),
    ]:
        values = coverage.get(key) or []
        if values:
            parts.append(f"{label}: {'; '.join(str(value) for value in values[:12])}")
    if construction.get("source_data_origin"):
        parts.append(f"Source origin: {construction.get('source_data_origin')}")
    if construction.get("collection_method"):
        parts.append(f"Collection: {construction.get('collection_method')}")
    return "\n".join(parts)


def query_representation_for_draft(draft: BenchmarkDraftRecord, representation_level: str) -> str:
    query_dcus = query_dcus_for_draft(draft)
    if representation_level == "paper_only":
        return query_paper_summary_for_draft(draft)
    if representation_level == "acu_query":
        return (
            f"Query dataset: {draft.query_dataset_name}\n"
            "Atomic contribution claims:\n"
            + "\n".join(f"- q{index}: {acu}" for index, acu in enumerate(draft.query_acus))
        )
    if representation_level in {"dcu_query", "dcu_evidence"}:
        lines = [f"Query dataset: {draft.query_dataset_name}", "Dataset Contribution Units:"]
        for dcu in query_dcus:
            fields = [f"type={dcu.acu_type or 'other'}"]
            if dcu.dataset_name:
                fields.append(f"dataset={dcu.dataset_name}")
            if dcu.tasks:
                fields.append(f"tasks={'; '.join(dcu.tasks[:3])}")
            if dcu.domains:
                fields.append(f"domains={'; '.join(dcu.domains[:3])}")
            if dcu.languages:
                fields.append(f"languages={'; '.join(dcu.languages[:5])}")
            if dcu.modalities:
                fields.append(f"modalities={'; '.join(dcu.modalities[:3])}")
            lines.append(f"- {dcu.dcu_id} [{', '.join(fields)}]: {dcu.text}")
        return "\n".join(lines)
    raise ValueError(f"Unknown representation level: {representation_level}")


def candidate_brief(candidate: CandidateRecord, max_chars: int = 1400) -> str:
    text = f"Paper ID: {candidate.candidate_id}\nDataset/Paper: {candidate.name}\nSummary: {candidate.summary_text}"
    return text[:max_chars]


def candidate_dcu_brief(
    candidate: CandidateRecord,
    payload: dict,
    query_dcus: Sequence[DCURecord],
    *,
    max_dcus: int = 5,
    max_chars: int = 1800,
) -> str:
    prior_dcus = prior_dcus_for_paper(candidate.candidate_id, payload)
    scored: List[tuple[float, DCURecord]] = []
    for prior in prior_dcus:
        best_score = max(
            (routed_dcu_pair_score(query, prior) for query in query_dcus),
            default=0.0,
        )
        scored.append((best_score, prior))
    top_dcus = [
        dcu for _, dcu in sorted(scored, key=lambda item: item[0], reverse=True)[:max_dcus]
    ]
    lines = [
        f"Paper ID: {candidate.candidate_id}",
        f"Dataset/Paper: {candidate.name}",
        f"Summary: {candidate.summary_text[:500]}",
    ]
    if top_dcus:
        lines.append("Candidate DCUs:")
        for dcu in top_dcus:
            fields = [f"type={dcu.acu_type or 'other'}"]
            if dcu.dataset_name:
                fields.append(f"dataset={dcu.dataset_name}")
            if dcu.tasks:
                fields.append(f"tasks={'; '.join(dcu.tasks[:3])}")
            if dcu.languages:
                fields.append(f"languages={'; '.join(dcu.languages[:5])}")
            lines.append(f"- {dcu.dcu_id} [{', '.join(fields)}]: {dcu.text}")
    else:
        lines.append("Candidate DCUs: none extracted")
    return "\n".join(lines)[:max_chars]


def collect_acus_for_papers(paper_ids: Sequence[str], payloads: Dict[str, dict], max_acus: int = 40) -> List[str]:
    acus: List[str] = []
    for paper_id in paper_ids:
        payload = payloads.get(paper_id) or {}
        for dataset in payload.get("datasets") or []:
            acus.extend([acu_text(acu) for acu in dataset.get("acus") or [] if acu_text(acu)])
    return unique(acus)[:max_acus]


def query_dcus_for_draft(draft: BenchmarkDraftRecord) -> List[DCURecord]:
    metadata = draft_metadata(draft)
    coverage = metadata.get("coverage") or {}
    construction = metadata.get("construction") or {}
    source_names = []
    for source in construction.get("source_datasets") or []:
        if isinstance(source, dict) and source.get("name"):
            source_names.append(str(source["name"]))
    return [
        DCURecord(
            dcu_id=f"q{index}",
            paper_id=draft.query_paper_id,
            dataset_name=draft.query_dataset_name,
            paper_title=str(metadata.get("query_title") or draft.query_paper_id),
            text=acu,
            acu_type=infer_acu_type(acu),
            role=str(metadata.get("query_dataset_role") or ""),
            resource_type=str(metadata.get("query_dataset_resource_type") or ""),
            tasks=list_strings(coverage.get("tasks")),
            domains=list_strings(coverage.get("domains")),
            languages=list_strings(coverage.get("languages")),
            modalities=list_strings(coverage.get("modality")),
            source_datasets=tuple(source_names),
            year=draft_year(draft),
        )
        for index, acu in enumerate(draft.query_acus)
        if acu
    ]


def prior_dcus_for_paper(paper_id: str, payload: dict) -> List[DCURecord]:
    records: List[DCURecord] = []
    title = str(payload.get("title") or paper_id)
    year = payload_year(payload, paper_id)
    for dataset_index, dataset in enumerate(payload.get("datasets") or []):
        coverage = dataset.get("coverage") or {}
        dataset_name = dataset_record_name(dataset)
        for acu_index, acu in enumerate(dataset.get("acus") or []):
            text = acu_text(acu)
            if not text:
                continue
            records.append(DCURecord(
                dcu_id=f"{paper_id}::d{dataset_index}::a{acu_index}",
                paper_id=paper_id,
                dataset_name=dataset_name,
                paper_title=title,
                text=text,
                acu_type=acu_type(acu) or infer_acu_type(text),
                role=str(dataset.get("role") or ""),
                resource_type=str(dataset.get("resource_type") or ""),
                tasks=list_strings(coverage.get("tasks")),
                domains=list_strings(coverage.get("domains")),
                languages=list_strings(coverage.get("languages")),
                modalities=list_strings(coverage.get("modality")),
                source_datasets=source_dataset_names(dataset),
                year=year,
            ))
    return records


def dcu_field_tokens(values: Sequence[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(tokenize(value))
    return tokens


def dcu_genericness_penalty(dcu: DCURecord) -> float:
    text_tokens = set(tokenize(dcu.text))
    generic_terms = {
        "dataset", "data", "benchmark", "corpus", "paper", "introduces",
        "proposes", "resource", "resources", "nlp", "language", "model",
    }
    if len(text_tokens) <= 5:
        return 0.12
    if text_tokens and len(text_tokens - generic_terms) / len(text_tokens) < 0.45:
        return 0.10
    return 0.0


def dcu_support_score(query: DCURecord, prior: DCURecord) -> float:
    q_tokens = set(tokenize(query.serialized()))
    p_tokens = set(tokenize(prior.serialized()))
    if not q_tokens or not p_tokens:
        return 0.0
    overlap = q_tokens & p_tokens
    jaccard = len(overlap) / len(q_tokens | p_tokens)
    containment = len(overlap) / min(len(q_tokens), len(p_tokens))
    score = (0.45 * jaccard) + (0.30 * containment)

    if query.acu_type and prior.acu_type and query.acu_type == prior.acu_type:
        score += 0.15

    query_name_tokens = set(tokenize(query.dataset_name))
    prior_name_tokens = set(tokenize(" ".join([prior.dataset_name, *prior.source_datasets])))
    if query_name_tokens and prior_name_tokens:
        name_overlap = len(query_name_tokens & prior_name_tokens) / len(query_name_tokens | prior_name_tokens)
        score += min(0.18, 0.28 * name_overlap)

    for q_values, p_values, weight in [
        (query.tasks, prior.tasks, 0.12),
        (query.domains, prior.domains, 0.10),
        (query.languages, prior.languages, 0.08),
        (query.modalities, prior.modalities, 0.06),
    ]:
        q_field = dcu_field_tokens(q_values)
        p_field = dcu_field_tokens(p_values)
        if q_field and p_field:
            score += weight * (len(q_field & p_field) / len(q_field | p_field))

    score -= dcu_genericness_penalty(prior)
    return max(0.0, min(1.0, score))


def candidate_papers_by_query(previous_work_path: str) -> Dict[str, set]:
    by_query: Dict[str, set] = defaultdict(set)
    for candidate in load_previous_work_candidates(Path(previous_work_path)):
        if candidate.resolved_paper_id:
            by_query[candidate.query_paper_id].add(candidate.resolved_paper_id)
    return by_query


def lexical_rank(
    query_name: str,
    query_acus: Sequence[str],
    candidates: Sequence[CandidateRecord],
    top_k: int,
) -> List[str]:
    if BM25Okapi is None:
        return []
    corpus_tokens = [tokenize(candidate.summary_text) for candidate in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    query_text = " ".join([query_name, *query_acus])
    scores = bm25.get_scores(tokenize(query_text))
    return [
        candidate.candidate_id
        for candidate, _ in sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[:top_k]
    ]


class SimpleDenseIndex:
    _model = None

    def __init__(self, candidates: Sequence[CandidateRecord], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if np is None:
            raise RuntimeError("sentence-transformers or numpy is unavailable.")
        self.candidates = list(candidates)
        self.model = self._get_model(model_name)
        self.doc_texts = [candidate.summary_text for candidate in self.candidates]
        self.doc_embeddings = self.model.encode(
            self.doc_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.tokenized_corpus = [tokenize(candidate.summary_text) for candidate in self.candidates]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if BM25Okapi is not None else None

    @classmethod
    def _get_model(cls, model_name: str):
        if cls._model is None:
            sentence_transformer = ensure_sentence_transformer()
            local = os.environ.get("BENCHMARK_DENSE_MODEL_PATH") or find_local_minilm_snapshot()
            cls._model = sentence_transformer(local or model_name)
        return cls._model

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_v = float(scores.min())
        max_v = float(scores.max())
        if max_v - min_v < 1e-8:
            return np.ones_like(scores, dtype=float)
        return (scores - min_v) / (max_v - min_v)

    def rank(self, query_name: str, query_acus: Sequence[str], top_k: int, method: str) -> List[str]:
        query_text = " ".join([query_name, *query_acus])
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        dense_scores = self.doc_embeddings @ query_embedding
        if method == "dense":
            final_scores = dense_scores
        elif method == "fusion":
            lexical_scores = np.zeros(len(self.candidates), dtype=float)
            if self.bm25 is not None:
                lexical_scores = np.asarray(self.bm25.get_scores(tokenize(query_text)), dtype=float)
            final_scores = (0.6 * self._normalize(dense_scores)) + (0.4 * self._normalize(lexical_scores))
        else:
            raise ValueError(f"SimpleDenseIndex does not support method {method}")
        ranked_indices = np.argsort(-final_scores)[:top_k]
        return [self.candidates[index].candidate_id for index in ranked_indices]


class CorpusDenseIndex:
    _model = None

    def __init__(self, candidates: Sequence[CandidateRecord]):
        if np is None:
            raise RuntimeError("sentence-transformers or numpy is unavailable.")
        self.candidates = list(candidates)
        self.model = self._get_model()
        self.doc_texts = [candidate.summary_text for candidate in self.candidates]
        self.doc_embeddings = self.model.encode(
            self.doc_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.tokenized_corpus = [tokenize(candidate.summary_text) for candidate in self.candidates]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if BM25Okapi is not None else None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            sentence_transformer = ensure_sentence_transformer()
            local = os.environ.get("BENCHMARK_DENSE_MODEL_PATH") or find_local_minilm_snapshot()
            cls._model = sentence_transformer(local or "sentence-transformers/all-MiniLM-L6-v2")
        return cls._model

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_v = float(scores.min())
        max_v = float(scores.max())
        if max_v - min_v < 1e-8:
            return np.ones_like(scores, dtype=float)
        return (scores - min_v) / (max_v - min_v)

    def rank(self, draft: BenchmarkDraftRecord, top_k: int, method: str) -> List[str]:
        query_text = query_text_for_draft(draft)
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        dense_scores = self.doc_embeddings @ query_embedding
        if method == "dense":
            final_scores = dense_scores
        elif method == "fusion":
            lexical_scores = np.zeros(len(self.candidates), dtype=float)
            if self.bm25 is not None:
                lexical_scores = np.asarray(self.bm25.get_scores(tokenize(query_text)), dtype=float)
            final_scores = (0.6 * self._normalize(dense_scores)) + (0.4 * self._normalize(lexical_scores))
        else:
            raise ValueError(f"CorpusDenseIndex does not support method {method}")
        ranked_indices = np.argsort(-final_scores)
        ranked = [self.candidates[index].candidate_id for index in ranked_indices if self.candidates[index].candidate_id != draft.query_paper_id]
        return ranked[:top_k]


def find_local_minilm_snapshot() -> Optional[str]:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    if not cache_dir.exists():
        return None
    for snapshot in cache_dir.iterdir():
        if (snapshot / "modules.json").exists() and (snapshot / "config.json").exists():
            return str(snapshot)
    return None


def has_local_hf_model(repo_id: str) -> bool:
    cache_name = "models--" + repo_id.replace("/", "--")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def ensure_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for dense retrieval methods.") from exc
        SentenceTransformer = _SentenceTransformer
    return SentenceTransformer


def ensure_torch_transformers(*, causal: bool = False):
    global torch, F, AutoModel, AutoModelForCausalLM, AutoTokenizer
    if torch is None or F is None:
        try:
            import torch as _torch
            import torch.nn.functional as _F
        except ImportError as exc:
            raise RuntimeError("torch is required for Qwen methods.") from exc
        torch = _torch
        F = _F
    if AutoTokenizer is None or AutoModel is None or (causal and AutoModelForCausalLM is None):
        try:
            from transformers import AutoModel as _AutoModel
            from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
            from transformers import AutoTokenizer as _AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for Qwen methods.") from exc
        AutoModel = _AutoModel
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
    return torch, F, AutoTokenizer, AutoModel, AutoModelForCausalLM


def ensure_openai_client():
    global OpenAI
    if OpenAI is None:
        try:
            from openai import OpenAI as _OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for GPT retrieval methods.") from exc
        OpenAI = _OpenAI
    return OpenAI()


def ensure_google_genai():
    global genai, genai_types
    if genai is None:
        try:
            from google import genai as _genai
            from google.genai import types as _genai_types
        except ImportError as exc:
            raise RuntimeError("google-genai is required for Gemini embeddings.") from exc
        genai = _genai
        genai_types = _genai_types
    return genai, genai_types


@dataclass
class RetrievalContext:
    drafts: Sequence[BenchmarkDraftRecord]
    payloads: Dict[str, dict]
    cited_by_query: Dict[str, set]
    cache: JsonCache
    top_k: int
    allow_model_download: bool
    allow_fallback_methods: bool
    rerank_depth: int
    gpt_model: str

    @property
    def corpus_ids(self) -> List[str]:
        return list(self.payloads.keys())

    @property
    def corpus_candidates(self) -> List[CandidateRecord]:
        return [processed_to_candidate(paper_id, payload) for paper_id, payload in self.payloads.items()]


class RetrieverMethod:
    method_type = "unknown"
    cost = "unknown"

    def __init__(self, ctx: RetrievalContext):
        self.ctx = ctx

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        raise NotImplementedError

    def metrics_for_last_rank(self, draft: BenchmarkDraftRecord, ranked_ids: Sequence[str]) -> Dict[str, float]:
        return {}

    def _candidates_for_query(self, draft: BenchmarkDraftRecord) -> List[CandidateRecord]:
        return [
            processed_to_candidate(
                paper_id,
                payload,
                is_cited=paper_id in self.ctx.cited_by_query.get(draft.query_paper_id, set()),
            )
            for paper_id, payload in self.ctx.payloads.items()
            if paper_id != draft.query_paper_id
        ]


class BM25Retriever(RetrieverMethod):
    method_type = "lexical"
    cost = "local"

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        return lexical_rank(draft.query_dataset_name, draft.query_acus, self._candidates_for_query(draft), top_k)


class MiniLMDenseRetriever(RetrieverMethod):
    method_type = "dense"
    cost = "local"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        self.index = CorpusDenseIndex(ctx.corpus_candidates)

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        return self.index.rank(draft, top_k, "dense")


class MiniLMFusionRetriever(RetrieverMethod):
    method_type = "hybrid"
    cost = "local"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        self.index = CorpusDenseIndex(ctx.corpus_candidates)

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        return self.index.rank(draft, top_k, "fusion")


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def weighted_rrf_merge(
    rankings: Sequence[tuple[Sequence[str], float]],
    *,
    top_k: int,
    k: int = 60,
) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    first_seen: Dict[str, int] = {}
    for ranking_index, (ranking, weight) in enumerate(rankings):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += weight * rrf_score(rank, k=k)
            first_seen.setdefault(item_id, ranking_index * 100000 + rank)
    return [
        item_id
        for item_id, _ in sorted(
            scores.items(),
            key=lambda item: (item[1], -first_seen.get(item[0], 0)),
            reverse=True,
        )
    ][:top_k]


def dcu_claim_tokens(dcu: DCURecord) -> set[str]:
    return set(tokenize(" ".join([
        dcu.dataset_name,
        dcu.paper_title,
        dcu.acu_type,
        dcu.text,
    ])))


def dcu_metadata_tokens(dcu: DCURecord) -> set[str]:
    return dcu_field_tokens([
        dcu.dataset_name,
        dcu.paper_title,
        dcu.role,
        dcu.resource_type,
        *dcu.tasks,
        *dcu.domains,
        *dcu.languages,
        *dcu.modalities,
        *dcu.source_datasets,
    ])


def token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def token_containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:k|m|b|million|billion|languages?|scripts?|tokens?|instances?|examples?|documents?|hours?)?\b", (text or "").lower()))


TYPE_ROUTE_WEIGHTS = {
    "task/domain": {
        "claim": 0.42,
        "metadata": 0.18,
        "name": 0.12,
        "source": 0.04,
        "numeric": 0.02,
        "type": 0.12,
        "base": 0.10,
    },
    "data/source": {
        "claim": 0.24,
        "metadata": 0.18,
        "name": 0.22,
        "source": 0.20,
        "numeric": 0.02,
        "type": 0.10,
        "base": 0.04,
    },
    "annotation/protocol": {
        "claim": 0.40,
        "metadata": 0.10,
        "name": 0.08,
        "source": 0.04,
        "numeric": 0.02,
        "type": 0.22,
        "base": 0.14,
    },
    "scale/coverage": {
        "claim": 0.22,
        "metadata": 0.24,
        "name": 0.08,
        "source": 0.02,
        "numeric": 0.26,
        "type": 0.12,
        "base": 0.06,
    },
    "evaluation/use": {
        "claim": 0.34,
        "metadata": 0.14,
        "name": 0.16,
        "source": 0.04,
        "numeric": 0.02,
        "type": 0.16,
        "base": 0.14,
    },
    "availability/quality": {
        "claim": 0.28,
        "metadata": 0.24,
        "name": 0.08,
        "source": 0.04,
        "numeric": 0.02,
        "type": 0.22,
        "base": 0.12,
    },
    "governance/ethics": {
        "claim": 0.34,
        "metadata": 0.12,
        "name": 0.06,
        "source": 0.04,
        "numeric": 0.02,
        "type": 0.28,
        "base": 0.14,
    },
    "other": {
        "claim": 0.42,
        "metadata": 0.18,
        "name": 0.12,
        "source": 0.04,
        "numeric": 0.02,
        "type": 0.08,
        "base": 0.14,
    },
}


def routed_dcu_pair_score(query: DCURecord, prior: DCURecord, *, base_score: float = 0.0) -> float:
    weights = TYPE_ROUTE_WEIGHTS.get(query.acu_type or "other", TYPE_ROUTE_WEIGHTS["other"])
    q_claim = dcu_claim_tokens(query)
    p_claim = dcu_claim_tokens(prior)
    q_meta = dcu_metadata_tokens(query)
    p_meta = dcu_metadata_tokens(prior)
    q_name = set(tokenize(query.dataset_name))
    p_name = set(tokenize(" ".join([prior.dataset_name, *prior.source_datasets])))
    q_source = dcu_field_tokens(query.source_datasets)
    p_source = dcu_field_tokens(prior.source_datasets)
    q_numeric = numeric_tokens(query.text)
    p_numeric = numeric_tokens(prior.text)
    score = 0.0
    score += weights["claim"] * ((0.65 * token_containment(q_claim, p_claim)) + (0.35 * token_jaccard(q_claim, p_claim)))
    score += weights["metadata"] * token_jaccard(q_meta, p_meta)
    score += weights["name"] * token_containment(q_name, p_name)
    score += weights["source"] * token_containment(q_source, p_source)
    score += weights["numeric"] * token_containment(q_numeric, p_numeric)
    if query.acu_type and prior.acu_type and query.acu_type == prior.acu_type:
        score += weights["type"]
    score += weights["base"] * base_score
    score -= dcu_genericness_penalty(prior)
    return max(0.0, min(1.0, score))


class DCURetrievalMixin:
    def _init_dcu_pooler(self) -> None:
        try:
            self.pooler = MiniLMFusionRetriever(self.ctx)
        except Exception:
            self.pooler = None

    def _pool_ids(self, draft: BenchmarkDraftRecord, pool_size: int) -> List[str]:
        pooler = getattr(self, "pooler", None)
        if pooler is not None:
            try:
                return pooler.rank(draft, pool_size)
            except Exception:
                pass
        try:
            ranked = lexical_rank(draft.query_dataset_name, draft.query_acus, self._candidates_for_query(draft), pool_size)
            if ranked:
                return ranked
        except Exception:
            pass
        return [candidate.candidate_id for candidate in self._candidates_for_query(draft)[:pool_size]]

    def _candidate_dcus(self, draft: BenchmarkDraftRecord, pool_ids: Sequence[str]) -> List[DCURecord]:
        q_year = draft_year(draft)
        output: List[DCURecord] = []
        for paper_id in pool_ids:
            if paper_id == draft.query_paper_id:
                continue
            for dcu in prior_dcus_for_paper(paper_id, self.ctx.payloads.get(paper_id) or {}):
                if q_year is not None and dcu.year is not None and dcu.year > q_year:
                    continue
                output.append(dcu)
        return output

    def _rank_from_pair_scores(
        self,
        query_dcus: Sequence[DCURecord],
        prior_dcus: Sequence[DCURecord],
        pair_scores: Dict[tuple[str, str], float],
        pool_ids: Sequence[str],
        top_k: int,
    ) -> List[str]:
        paper_scores: Dict[str, float] = defaultdict(float)
        paper_best: Dict[str, float] = defaultdict(float)
        for prior in prior_dcus:
            scores = [pair_scores.get((query.dcu_id, prior.dcu_id), 0.0) for query in query_dcus]
            if not scores:
                continue
            best = max(scores)
            mean_top = safe_mean(sorted(scores, reverse=True)[: min(3, len(scores))])
            paper_scores[prior.paper_id] += mean_top
            paper_best[prior.paper_id] = max(paper_best[prior.paper_id], best)
        pool_rank_bonus = {paper_id: rrf_score(rank + 1) for rank, paper_id in enumerate(pool_ids)}
        ranked = [
            paper_id
            for paper_id, _ in sorted(
                paper_scores.items(),
                key=lambda item: (
                    paper_best[item[0]],
                    item[1],
                    pool_rank_bonus.get(item[0], 0.0),
                ),
                reverse=True,
            )
        ]
        return unique(ranked + list(pool_ids))[:top_k]


class ACURetriever(DCURetrievalMixin, RetrieverMethod):
    method_type = "acu_retrieval"
    cost = "local"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        self._init_dcu_pooler()

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        pool_size = max(top_k * 5, self.ctx.rerank_depth)
        pool_ids = self._pool_ids(draft, pool_size)
        query_dcus = query_dcus_for_draft(draft)
        prior_dcus = self._candidate_dcus(draft, pool_ids)
        pair_scores: Dict[tuple[str, str], float] = {}
        for query in query_dcus:
            q_tokens = set(tokenize(query.text))
            for prior in prior_dcus:
                p_tokens = set(tokenize(prior.text))
                pair_scores[(query.dcu_id, prior.dcu_id)] = (
                    0.65 * token_containment(q_tokens, p_tokens)
                    + 0.35 * token_jaccard(q_tokens, p_tokens)
                )
        return self._rank_from_pair_scores(query_dcus, prior_dcus, pair_scores, pool_ids, top_k)


class DCURetriever(DCURetrievalMixin, RetrieverMethod):
    method_type = "dcu_retrieval"
    cost = "local"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        self._init_dcu_pooler()

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        pool_size = max(top_k * 5, self.ctx.rerank_depth)
        pool_ids = self._pool_ids(draft, pool_size)
        query_dcus = query_dcus_for_draft(draft)
        prior_dcus = self._candidate_dcus(draft, pool_ids)
        pair_scores: Dict[tuple[str, str], float] = {}
        for query in query_dcus:
            for prior in prior_dcus:
                pair_scores[(query.dcu_id, prior.dcu_id)] = dcu_support_score(query, prior)
        return self._rank_from_pair_scores(query_dcus, prior_dcus, pair_scores, pool_ids, top_k)


class TypeRoutedDCURetriever(DCURetrievalMixin, RetrieverMethod):
    method_type = "type_routed_dcu"
    cost = "local"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        self._init_dcu_pooler()

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        pool_size = max(top_k * 5, self.ctx.rerank_depth)
        pool_ids = self._pool_ids(draft, pool_size)
        pool_rank_base = {paper_id: 1.0 - (rank / max(len(pool_ids), 1)) for rank, paper_id in enumerate(pool_ids)}
        query_dcus = query_dcus_for_draft(draft)
        prior_dcus = self._candidate_dcus(draft, pool_ids)
        pair_scores: Dict[tuple[str, str], float] = {}
        for query in query_dcus:
            for prior in prior_dcus:
                pair_scores[(query.dcu_id, prior.dcu_id)] = routed_dcu_pair_score(
                    query,
                    prior,
                    base_score=pool_rank_base.get(prior.paper_id, 0.0),
                )
        typed_rank = self._rank_from_pair_scores(query_dcus, prior_dcus, pair_scores, pool_ids, len(pool_ids))
        return weighted_rrf_merge(
            [
                (pool_ids, 0.70),
                (typed_rank, 0.30),
            ],
            top_k=top_k,
        )


class TypeRoutedDCUSupportReranker(TypeRoutedDCURetriever):
    method_type = "type_routed_dcu_support_rerank"
    cost = "api"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        self.reranker = GPTListwiseReranker(ctx, oracle=False, support_aware=True, pooler=TypeRoutedDCURetriever(ctx))

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        return self.reranker.rank(draft, top_k)


class SupportAwareDCUReranker(RetrieverMethod):
    method_type = "support_aware_dcu_rerank"
    cost = "api"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        self.reranker = GPTListwiseReranker(
            ctx,
            oracle=False,
            support_aware=True,
            include_candidate_dcus=True,
            pooler=MiniLMFusionRetriever(ctx),
        )

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        return self.reranker.rank(draft, top_k)


class TypeRoutedDCUSupportEvidenceReranker(TypeRoutedDCURetriever):
    method_type = "type_routed_dcu_support_evidence_rerank"
    cost = "api"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        self.reranker = GPTListwiseReranker(
            ctx,
            oracle=False,
            support_aware=True,
            include_candidate_dcus=True,
            pooler=TypeRoutedDCURetriever(ctx),
        )

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        return self.reranker.rank(draft, top_k)


class HybridExactRetriever(RetrieverMethod):
    method_type = "exact_existing"
    cost = "local_model"

    def __init__(self, ctx: RetrievalContext, method: str):
        super().__init__(ctx)
        self.method = method
        splade_required = method in {"splade", "rank_fusion", "hybrid_rerank"}
        colbert_required = method in {"colbert", "rank_fusion", "hybrid_rerank"}
        if splade_required and not ctx.allow_fallback_methods and not ctx.allow_model_download:
            if not has_local_hf_model("naver/splade-cocondenser-ensembledistil"):
                raise RuntimeError(
                    "Exact SPLADE-dependent method requested, but naver/splade-cocondenser-ensembledistil "
                    "is not available in the local HuggingFace cache. Re-run with --allow-model-download "
                    "to download it, or use --allow-fallback-methods to explicitly permit approximations."
                )
        if colbert_required and not ctx.allow_fallback_methods:
            raise RuntimeError(
                "Strict ColBERT-dependent method requested, but this repo only has a MiniLM token-MaxSim "
                "ColBERT-style approximation, not an exact ColBERT/LateOn implementation. Use the `lateon` "
                "method after installing PyLate for an exact late-interaction baseline, or pass "
                "--allow-fallback-methods to explicitly run the legacy approximation."
            )
        from scv.benchmarking import HybridSupportRetriever as _HybridSupportRetriever
        _HybridSupportRetriever.STRICT_EXACT_METHODS = not ctx.allow_fallback_methods
        self.retriever_cls = _HybridSupportRetriever

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        retriever = self.retriever_cls(self._candidates_for_query(draft))
        return [
            candidate_id
            for candidate_id, _, _ in retriever.rank(
                draft.query_dataset_name,
                draft.query_acus,
                top_k=top_k,
                method=self.method,
            )
        ]


def _last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _load_transformer_embedding_model(model_name: str):
    ensure_torch_transformers(causal=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()
    return tokenizer, model


class TransformerEmbeddingRetriever(RetrieverMethod):
    method_type = "dense"
    cost = "local_model"
    _model_cache: Dict[str, Tuple[object, object]] = {}

    def __init__(self, ctx: RetrievalContext, model_name: str, instruction: str):
        super().__init__(ctx)
        self.model_name = model_name
        self.instruction = instruction
        self.namespace = f"embeddings_{model_name.replace('/', '_')}"
        if not ctx.allow_model_download and not has_local_hf_model(model_name):
            raise RuntimeError(
                f"{model_name} is not available in the local HuggingFace cache. "
                "Re-run with --allow-model-download to download it."
            )

    def _embed(self, texts: Sequence[str], *, is_query: bool) -> List[List[float]]:
        input_texts = [
            f"Instruct: {self.instruction}\nQuery: {text}" if is_query and self.instruction else text
            for text in texts
        ]
        key = {"model": self.model_name, "texts": input_texts}
        cached = self.ctx.cache.get(self.namespace, key)
        if cached is not None:
            return cached
        if self.model_name not in self._model_cache:
            self._model_cache[self.model_name] = _load_transformer_embedding_model(self.model_name)
        tokenizer, model = self._model_cache[self.model_name]
        all_embeddings: List[List[float]] = []
        batch_size = int(os.environ.get("QWEN_EMBED_BATCH_SIZE", "2"))
        max_length = int(os.environ.get("QWEN_EMBED_MAX_LENGTH", "2048"))
        device = next(model.parameters()).device
        for start in range(0, len(input_texts), batch_size):
            batch = input_texts[start:start + batch_size]
            batch_dict = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = _last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().numpy().tolist())
        return self.ctx.cache.set(self.namespace, key, all_embeddings)

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        candidates = self._candidates_for_query(draft)
        doc_embeddings = np.asarray(self._embed([candidate.summary_text for candidate in candidates], is_query=False), dtype=float)
        query_embedding = np.asarray(self._embed([query_text_for_draft(draft)], is_query=True)[0], dtype=float)
        scores = doc_embeddings @ query_embedding
        ranked_indices = np.argsort(-scores)[:top_k]
        return [candidates[index].candidate_id for index in ranked_indices]


class GeminiEmbeddingRetriever(RetrieverMethod):
    method_type = "dense_api"
    cost = "api"

    def __init__(self, ctx: RetrievalContext, model_name: str = "gemini-embedding-001"):
        super().__init__(ctx)
        self.model_name = model_name
        self.namespace = f"gemini_embeddings_{model_name}"
        google_genai, _ = ensure_google_genai()
        self.client = google_genai.Client()

    def _embed_one(self, text: str, task_type: str) -> List[float]:
        key = {"model": self.model_name, "task_type": task_type, "text": text}
        cached = self.ctx.cache.get(self.namespace, key)
        if cached is not None:
            return cached
        kwargs = {"model": self.model_name, "contents": text}
        if genai_types is not None:
            kwargs["config"] = genai_types.EmbedContentConfig(task_type=task_type)
        result = self.client.models.embed_content(**kwargs)
        embedding = result.embeddings[0].values if hasattr(result.embeddings[0], "values") else result.embeddings[0]
        return self.ctx.cache.set(self.namespace, key, list(embedding))

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        candidates = self._candidates_for_query(draft)
        doc_embeddings = np.asarray([self._embed_one(candidate.summary_text, "RETRIEVAL_DOCUMENT") for candidate in candidates], dtype=float)
        query_embedding = np.asarray(self._embed_one(query_text_for_draft(draft), "RETRIEVAL_QUERY"), dtype=float)
        doc_embeddings = doc_embeddings / np.maximum(np.linalg.norm(doc_embeddings, axis=1, keepdims=True), 1e-12)
        query_embedding = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
        scores = doc_embeddings @ query_embedding
        ranked_indices = np.argsort(-scores)[:top_k]
        return [candidates[index].candidate_id for index in ranked_indices]


class GPTQueryRewriteBM25Retriever(RetrieverMethod):
    method_type = "llm_query_rewrite"
    cost = "api"

    def __init__(self, ctx: RetrievalContext, base: str):
        super().__init__(ctx)
        self.client = ensure_openai_client()
        self.base = base
        self.dense = MiniLMDenseRetriever(ctx) if base == "dense" else None

    def rewrite(self, draft: BenchmarkDraftRecord) -> str:
        key = {"model": self.ctx.gpt_model, "query": query_text_for_draft(draft)}
        cached = self.ctx.cache.get("gpt_query_rewrites", key)
        if cached is not None:
            return cached["query"]
        prompt = (
            "Rewrite this dataset-prior retrieval query into a concise search query. "
            "The goal is to retrieve earlier dataset papers, source datasets, or benchmark papers that support comparison. "
            "Return only the rewritten query.\n\n"
            f"Dataset: {draft.query_dataset_name}\n"
            f"ACUs:\n" + "\n".join(f"- {acu}" for acu in draft.query_acus)
        )
        response = self.client.responses.create(model=self.ctx.gpt_model, input=prompt)
        rewritten = response.output_text.strip()
        return self.ctx.cache.set("gpt_query_rewrites", key, {"query": rewritten})["query"]

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        rewritten = self.rewrite(draft)
        if self.base == "bm25":
            return lexical_rank(rewritten, [], self._candidates_for_query(draft), top_k)
        if self.base == "dense":
            synthetic = BenchmarkDraftRecord(
                query_paper_id=draft.query_paper_id,
                query_dataset_name=rewritten,
                query_acus=[],
                gold_prior_paper_ids=draft.gold_prior_paper_ids,
                gold_added_information_label=draft.gold_added_information_label,
            )
            return self.dense.rank(synthetic, top_k)
        raise ValueError(f"Unknown GPT rewrite base: {self.base}")


class GPTListwiseReranker(RetrieverMethod):
    method_type = "llm_reranker"
    cost = "api"

    def __init__(
        self,
        ctx: RetrievalContext,
        oracle: bool = False,
        *,
        support_aware: bool = False,
        include_candidate_dcus: bool = False,
        representation_level: str = "acu_query",
        pooler: RetrieverMethod | None = None,
    ):
        super().__init__(ctx)
        self.client = ensure_openai_client()
        self.oracle = oracle
        self.support_aware = support_aware
        self.include_candidate_dcus = include_candidate_dcus
        self.representation_level = representation_level
        self.pooler = pooler or MiniLMFusionRetriever(ctx)

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        pool = self.pooler.rank(draft, max(top_k, self.ctx.rerank_depth))
        if self.oracle:
            pool = unique(list(draft.gold_prior_paper_ids) + pool)
        candidates_by_id = {candidate.candidate_id: candidate for candidate in self._candidates_for_query(draft)}
        pool = [paper_id for paper_id in pool if paper_id in candidates_by_id][:self.ctx.rerank_depth]
        key = {
            "model": self.ctx.gpt_model,
            "oracle": self.oracle,
            "support_aware": self.support_aware,
            "include_candidate_dcus": self.include_candidate_dcus,
            "representation_level": self.representation_level,
            "pooler": getattr(self.pooler, "method_type", "unknown"),
            "query": query_text_for_draft(draft),
            "pool": pool,
        }
        cached = self.ctx.cache.get("gpt_listwise_rerank", key)
        if cached is not None:
            return cached["ranking"][:top_k]
        query_dcus = query_dcus_for_draft(draft)
        query_block = query_representation_for_draft(draft, self.representation_level)
        if self.include_candidate_dcus:
            candidate_block = "\n\n".join(
                candidate_dcu_brief(
                    candidates_by_id[paper_id],
                    self.ctx.payloads.get(paper_id) or {},
                    query_dcus,
                )
                for paper_id in pool
            )
        else:
            candidate_block = "\n\n".join(candidate_brief(candidates_by_id[paper_id]) for paper_id in pool)
        if self.support_aware:
            prompt = (
                "Rank candidate prior-support papers for the query dataset. "
                "This is not ordinary topical relevance. Rank papers by whether they contain dataset claims, source datasets, benchmarks, or construction details that support, partially support, or make the query contribution claims less novel. "
                "Surface-similar papers that do not support the typed dataset contribution claims should be ranked lower. "
                "When candidate DCUs are supplied, use them as the primary evidence and use the summary only as context. "
                "Return JSON only with this schema: {\"ranking\": [\"paper_id\", ...]}.\n\n"
                f"Query representation ({self.representation_level}):\n{query_block}\n\n"
                f"Candidates:\n{candidate_block}"
            )
        else:
            prompt = (
                "Rank candidate prior-support papers for the query dataset. "
                "A good prior-support paper is an earlier dataset, corpus, benchmark, or source dataset useful for comparing what the query dataset adds. "
                "Return JSON only with this schema: {\"ranking\": [\"paper_id\", ...]}.\n\n"
                f"Query representation ({self.representation_level}):\n{query_block}\n\n"
                f"Candidates:\n{candidate_block}"
            )
        response = self.client.responses.create(model=self.ctx.gpt_model, input=prompt)
        text = response.output_text.strip()
        try:
            parsed = json.loads(text)
            ranking = [paper_id for paper_id in parsed.get("ranking", []) if paper_id in pool]
        except Exception:
            ranking = []
        ranking = unique(ranking + pool)
        return self.ctx.cache.set("gpt_listwise_rerank", key, {"ranking": ranking})["ranking"][:top_k]


class PairwiseSupportReranker(RetrieverMethod):
    method_type = "pairwise_support_rerank"
    cost = "api"

    def __init__(
        self,
        ctx: RetrievalContext,
        *,
        base_method: str = "gpt_listwise_rerank",
        pairwise_depth: int = 15,
    ):
        super().__init__(ctx)
        self.client = ensure_openai_client()
        self.base_method = base_method
        self.pairwise_depth = pairwise_depth
        if base_method == "fusion":
            self.base = MiniLMFusionRetriever(ctx)
        elif base_method == "type_routed_dcu":
            self.base = TypeRoutedDCURetriever(ctx)
        elif base_method == "gpt_listwise_rerank":
            self.base = GPTListwiseReranker(ctx, oracle=False)
        else:
            raise ValueError(f"Unknown pairwise base method: {base_method}")

    def _score_candidate(
        self,
        draft: BenchmarkDraftRecord,
        paper_id: str,
        candidate: CandidateRecord,
        base_rank: int,
    ) -> dict:
        query_dcus = query_dcus_for_draft(draft)
        payload = self.ctx.payloads.get(paper_id) or {}
        candidate_text = candidate_dcu_brief(
            candidate,
            payload,
            query_dcus,
            max_dcus=8,
            max_chars=2500,
        )
        key = {
            "model": self.ctx.gpt_model,
            "query_paper_id": draft.query_paper_id,
            "query_dataset_name": draft.query_dataset_name,
            "query_dcus": [(dcu.dcu_id, dcu.acu_type, dcu.text) for dcu in query_dcus],
            "candidate_paper_id": paper_id,
            "candidate_text": candidate_text,
            "schema": "pairwise_support_score_v1",
        }
        cached = self.ctx.cache.get("pairwise_support_scores", key)
        if cached is not None:
            return cached
        query_block = "\n".join(
            f"- {dcu.dcu_id} [{dcu.acu_type or 'other'}]: {dcu.text}"
            for dcu in query_dcus
        )
        prompt = (
            "You are judging whether one candidate prior paper provides prior-support evidence for a query dataset paper. "
            "Do not judge topical similarity alone. A useful prior-support paper should directly or partially support at least one typed query DCU, such as task/domain, data/source, annotation/protocol, scale/coverage, or evaluation/use. "
            "Return JSON only with this schema: {\"support_level\":\"strong|partial|weak|none\", \"supported_query_dcu_ids\":[\"q0\"], \"confidence\":0.0, \"rationale\":\"...\"}.\n\n"
            f"Query dataset: {draft.query_dataset_name}\n"
            f"Query DCUs:\n{query_block}\n\n"
            f"Candidate prior paper:\n{candidate_text}"
        )
        response = self.client.responses.create(model=self.ctx.gpt_model, input=prompt)
        text = response.output_text.strip()
        try:
            parsed = json.loads(text)
            output = PairwiseSupportScoreOutput(**parsed)
        except Exception:
            output = PairwiseSupportScoreOutput(
                support_level="none",
                supported_query_dcu_ids=[],
                confidence=0.0,
                rationale=f"Could not parse model output: {text[:200]}",
            )
        level_score = {
            "strong": 3.0,
            "partial": 2.0,
            "weak": 1.0,
            "none": 0.0,
        }[output.support_level]
        valid_ids = {dcu.dcu_id for dcu in query_dcus}
        covered_ids = [qid for qid in output.supported_query_dcu_ids if qid in valid_ids]
        result = {
            "candidate_paper_id": paper_id,
            "base_rank": base_rank,
            "support_level": output.support_level,
            "supported_query_dcu_ids": covered_ids,
            "confidence": max(0.0, min(1.0, float(output.confidence))),
            "rationale": output.rationale,
            "score": level_score + 0.25 * len(covered_ids) + 0.5 * max(0.0, min(1.0, float(output.confidence))),
        }
        return self.ctx.cache.set("pairwise_support_scores", key, result)

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        base_pool_size = max(top_k, self.ctx.rerank_depth)
        base_ranking = self.base.rank(draft, base_pool_size)
        candidates_by_id = {candidate.candidate_id: candidate for candidate in self._candidates_for_query(draft)}
        rerank_pool = [paper_id for paper_id in base_ranking if paper_id in candidates_by_id][: self.pairwise_depth]
        scored = []
        for base_rank, paper_id in enumerate(rerank_pool, start=1):
            score = self._score_candidate(draft, paper_id, candidates_by_id[paper_id], base_rank)
            scored.append((paper_id, score))
        reranked = [
            paper_id
            for paper_id, score in sorted(
                scored,
                key=lambda item: (
                    item[1].get("score", 0.0),
                    -item[1].get("base_rank", 10**6),
                ),
                reverse=True,
            )
        ]
        return unique(reranked + base_ranking)[:top_k]


class QwenReranker(RetrieverMethod):
    method_type = "local_reranker"
    cost = "local_model"

    def __init__(self, ctx: RetrievalContext, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        super().__init__(ctx)
        self.model_name = model_name
        if not ctx.allow_model_download and not has_local_hf_model(model_name):
            raise RuntimeError(
                f"{model_name} is not available in the local HuggingFace cache. "
                "Re-run with --allow-model-download to download it."
            )
        self.pooler = MiniLMFusionRetriever(ctx)
        ensure_torch_transformers(causal=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.true_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]

    def _score_pair(self, query: str, document: str) -> float:
        key = {"model": self.model_name, "query": query, "document": document[:3000]}
        cached = self.ctx.cache.get("qwen_reranker_scores", key)
        if cached is not None:
            return float(cached["score"])
        prompt = (
            "Given a query and a document, determine whether the document is a relevant prior-support paper. "
            "Answer yes or no.\n\n"
            f"Query: {query}\n\nDocument: {document[:3000]}\n\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, [self.true_id, self.false_id]]
            probs = torch.softmax(logits, dim=0)
            score = float(probs[0].cpu())
        return float(self.ctx.cache.set("qwen_reranker_scores", key, {"score": score})["score"])

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        pool = self.pooler.rank(draft, max(top_k, self.ctx.rerank_depth))
        candidates = {candidate.candidate_id: candidate for candidate in self._candidates_for_query(draft)}
        query = query_text_for_draft(draft)
        scored = [(paper_id, self._score_pair(query, candidates[paper_id].summary_text)) for paper_id in pool if paper_id in candidates]
        return [paper_id for paper_id, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]]


class LateOnRetriever(RetrieverMethod):
    method_type = "late_interaction"
    cost = "local_model"

    def __init__(self, ctx: RetrievalContext):
        super().__init__(ctx)
        try:
            import pylate  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("PyLate is not installed. Install pylate to run exact LateOn late-interaction retrieval.") from exc

    def rank(self, draft: BenchmarkDraftRecord, top_k: int) -> List[str]:
        raise RuntimeError("LateOn exact retrieval is not wired yet; PyLate is required and unavailable in this environment.")


def build_retriever(method: str, ctx: RetrievalContext) -> RetrieverMethod:
    aliases = {
        "bm25": "lexical",
        "minilm_dense": "dense",
        "minilm_fusion": "fusion",
        "tr_dcu": "type_routed_dcu",
        "tr_dcu_support_rerank": "type_routed_dcu_support_rerank",
        "support_aware_dcu": "support_aware_dcu_rerank",
        "tr_dcu_support_evidence_rerank": "type_routed_dcu_support_evidence_rerank",
        "pairwise_support": "pairwise_support_rerank",
        "pairwise_support_fusion": "pairwise_support_rerank_fusion",
        "pairwise_support_tr_dcu": "pairwise_support_rerank_type_routed_dcu",
        "vanilla_listwise_rerank": "paper_listwise_rerank",
        "paper_rerank": "paper_listwise_rerank",
        "acu_rerank": "acu_query_listwise_rerank",
        "dcu_query_rerank": "dcu_query_listwise_rerank",
        "dcu_evidence_rerank": "dcu_evidence_listwise_rerank",
        "gpt_5_4_query_rewrite_bm25": "gpt_query_rewrite_bm25",
        "gpt_5_4_query_rewrite_dense": "gpt_query_rewrite_dense",
        "gpt_5_4_listwise_rerank": "gpt_listwise_rerank",
        "gpt_5_4_oracle_tournament": "gpt_oracle_tournament",
    }
    method = aliases.get(method, method)
    if method == "lexical":
        return BM25Retriever(ctx)
    if method == "dense":
        return MiniLMDenseRetriever(ctx)
    if method == "fusion":
        return MiniLMFusionRetriever(ctx)
    if method == "acu_retrieval":
        return ACURetriever(ctx)
    if method == "dcu_retrieval":
        return DCURetriever(ctx)
    if method == "type_routed_dcu":
        return TypeRoutedDCURetriever(ctx)
    if method == "type_routed_dcu_support_rerank":
        return TypeRoutedDCUSupportReranker(ctx)
    if method == "support_aware_dcu_rerank":
        return SupportAwareDCUReranker(ctx)
    if method == "type_routed_dcu_support_evidence_rerank":
        return TypeRoutedDCUSupportEvidenceReranker(ctx)
    if method in {"splade", "colbert", "rank_fusion", "hybrid_rerank"}:
        return HybridExactRetriever(ctx, method)
    if method == "qwen3_embedding_0_6b":
        return TransformerEmbeddingRetriever(ctx, "Qwen/Qwen3-Embedding-0.6B", "Given a dataset paper description, retrieve prior dataset, corpus, benchmark, or source dataset papers that support comparison.")
    if method == "qwen3_embedding_4b":
        return TransformerEmbeddingRetriever(ctx, "Qwen/Qwen3-Embedding-4B", "Given a dataset paper description, retrieve prior dataset, corpus, benchmark, or source dataset papers that support comparison.")
    if method == "qwen3_reranker_0_6b":
        return QwenReranker(ctx, "Qwen/Qwen3-Reranker-0.6B")
    if method == "gemini_embedding_001":
        return GeminiEmbeddingRetriever(ctx)
    if method == "gpt_query_rewrite_bm25":
        return GPTQueryRewriteBM25Retriever(ctx, "bm25")
    if method == "gpt_query_rewrite_dense":
        return GPTQueryRewriteBM25Retriever(ctx, "dense")
    if method == "gpt_listwise_rerank":
        return GPTListwiseReranker(ctx, oracle=False)
    if method == "paper_listwise_rerank":
        return GPTListwiseReranker(
            ctx,
            oracle=False,
            support_aware=False,
            representation_level="paper_only",
            pooler=MiniLMFusionRetriever(ctx),
        )
    if method == "acu_query_listwise_rerank":
        return GPTListwiseReranker(
            ctx,
            oracle=False,
            support_aware=False,
            representation_level="acu_query",
            pooler=MiniLMFusionRetriever(ctx),
        )
    if method == "dcu_query_listwise_rerank":
        return GPTListwiseReranker(
            ctx,
            oracle=False,
            support_aware=True,
            representation_level="dcu_query",
            pooler=MiniLMFusionRetriever(ctx),
        )
    if method == "dcu_evidence_listwise_rerank":
        return GPTListwiseReranker(
            ctx,
            oracle=False,
            support_aware=True,
            include_candidate_dcus=True,
            representation_level="dcu_evidence",
            pooler=MiniLMFusionRetriever(ctx),
        )
    if method == "support_aware_rerank":
        return GPTListwiseReranker(ctx, oracle=False, support_aware=True)
    if method == "pairwise_support_rerank":
        return PairwiseSupportReranker(ctx, base_method="gpt_listwise_rerank")
    if method == "pairwise_support_rerank_fusion":
        return PairwiseSupportReranker(ctx, base_method="fusion")
    if method == "pairwise_support_rerank_type_routed_dcu":
        return PairwiseSupportReranker(ctx, base_method="type_routed_dcu")
    if method == "gpt_oracle_tournament":
        return GPTListwiseReranker(ctx, oracle=True)
    if method == "lateon":
        return LateOnRetriever(ctx)
    raise ValueError(f"Unknown retrieval method: {method}")


def hybrid_debug_counters() -> Dict[str, int]:
    try:
        from scv.benchmarking import HybridSupportRetriever as _HybridSupportRetriever
    except Exception:
        return {}
    return _HybridSupportRetriever.get_debug_counters()


def format_markdown_report(report: dict) -> str:
    retrieval = report.get("retrieval") or {}
    lines = [
        "# Completed Benchmark Retrieval Evaluation",
        "",
        f"- Complete rows: {report.get('row_counts', {}).get('complete', 0)}",
        f"- Processed corpus papers: {report.get('row_counts', {}).get('processed_corpus', 0)}",
        "",
        "| Method | MRR | R@1 | R@3 | R@5 | R@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in retrieval.items():
        lines.append(
            "| {method} | {mrr:.3f} | {r1:.3f} | {r3:.3f} | {r5:.3f} | {r10:.3f} |".format(
                method=method,
                mrr=metrics.get("mrr", 0.0),
                r1=metrics.get("recall@1", 0.0),
                r3=metrics.get("recall@3", 0.0),
                r5=metrics.get("recall@5", 0.0),
                r10=metrics.get("recall@10", 0.0),
            )
        )
    skipped = report.get("skipped") or {}
    if skipped:
        lines.extend(["", "## Skipped / Failed Methods"])
        for key, values in skipped.items():
            lines.append(f"- `{key}`: {len(values)} issue(s). First: {values[0] if values else ''}")
    attribution = report.get("added_information_attribution") or {}
    if attribution:
        lines.extend([
            "",
            "## Added-Information Attribution",
            "",
            "| Condition | N | Mean Score | Supported | Partial | Unsupported |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for condition, summary in (attribution.get("conditions") or {}).items():
            percentages = summary.get("mean_support_percentages") or {}
            lines.append(
                "| {condition} | {n} | {score:.3f} | {supported:.3f} | {partial:.3f} | {unsupported:.3f} |".format(
                    condition=condition,
                    n=summary.get("n", 0),
                    score=summary.get("mean_added_information_score", 0.0),
                    supported=percentages.get("supported", 0.0),
                    partial=percentages.get("partially_supported", 0.0),
                    unsupported=percentages.get("unsupported", 0.0),
                )
            )
        evidence_eval = attribution.get("evidence_quality")
        if evidence_eval:
            lines.extend([
                "",
                "## Evidence-Quality Evaluation",
                "",
                f"- N: {evidence_eval.get('n', 0)}",
                f"- Evidence-label accuracy: {evidence_eval.get('evidence_label_accuracy', 0.0):.3f}",
                f"- Macro-F1: {evidence_eval.get('macro_f1', 0.0):.3f}",
                f"- Evidence precision: {evidence_eval.get('evidence_precision', 0.0):.3f}",
                f"- Rationale groundedness: {evidence_eval.get('rationale_groundedness_rate', 0.0):.3f}",
            ])
    return "\n".join(lines) + "\n"


def run_retrieval_eval(
    drafts: Sequence[BenchmarkDraftRecord],
    payloads: Dict[str, dict],
    cited_by_query: Dict[str, set],
    methods: Sequence[str],
    top_k: int,
    *,
    cache_dir: str,
    allow_model_download: bool,
    allow_fallback_methods: bool,
    rerank_depth: int,
    gpt_model: str,
    progress_every: int,
    rank_checkpoint_jsonl: str | None,
    resume_rank_checkpoint: bool,
) -> tuple[Dict[str, dict], Dict[str, List[dict]], Dict[str, List[str]], dict]:
    metric_rows: Dict[str, List[Dict[str, float]]] = {method: [] for method in methods}
    misses: Dict[str, List[dict]] = {method: [] for method in methods}
    skipped: Dict[str, List[str]] = defaultdict(list)
    method_objects: Dict[str, RetrieverMethod] = {}
    cache = JsonCache(cache_dir)
    run_key = stable_hash({
        "draft_ids": [draft.query_paper_id for draft in drafts],
        "payload_ids": sorted(payloads.keys()),
        "top_k": top_k,
        "rerank_depth": rerank_depth,
        "model": gpt_model,
    })
    checkpoint_path = Path(rank_checkpoint_jsonl) if rank_checkpoint_jsonl else default_rank_checkpoint_path(cache_dir, run_key)
    checkpoint_rows = load_rank_checkpoints(checkpoint_path, run_key) if resume_rank_checkpoint else {}
    ctx = RetrievalContext(
        drafts=drafts,
        payloads=payloads,
        cited_by_query=cited_by_query,
        cache=cache,
        top_k=top_k,
        allow_model_download=allow_model_download,
        allow_fallback_methods=allow_fallback_methods,
        rerank_depth=rerank_depth,
        gpt_model=gpt_model,
    )
    progress_log(
        "retrieval_start",
        rows=len(drafts),
        processed_corpus=len(payloads),
        methods=list(methods),
        top_k=top_k,
        rerank_depth=rerank_depth,
        model=gpt_model,
        run_key=run_key,
        rank_checkpoint_jsonl=str(checkpoint_path),
        resume_rank_checkpoint=resume_rank_checkpoint,
        checkpoint_rows_loaded=len(checkpoint_rows),
    )
    for method in methods:
        try:
            method_objects[method] = build_retriever(method, ctx)
            progress_log("retriever_ready", method=method)
        except Exception as exc:
            skipped[f"{method}_init_failed"].append(str(exc))
            progress_log("retriever_init_failed", method=method, error=str(exc))

    valid_drafts: List[tuple[BenchmarkDraftRecord, List[str]]] = []
    for draft in drafts:
        gold = [paper_id for paper_id in draft.gold_prior_paper_ids if paper_id in payloads]
        if not gold:
            skipped["gold_not_processed"].append(draft.query_paper_id)
            continue
        valid_drafts.append((draft, gold))

    total_work = len(valid_drafts) * len(method_objects)
    completed_work = 0
    resumed_work = 0
    started_at = time.monotonic()
    progress_log(
        "retrieval_eval_ready",
        valid_rows=len(valid_drafts),
        skipped_gold_not_processed=len(skipped.get("gold_not_processed", [])),
        active_methods=list(method_objects.keys()),
        total_rank_calls=total_work,
    )

    for draft_index, (draft, gold) in enumerate(valid_drafts, start=1):
        for method in methods:
            if method not in method_objects:
                continue
            checkpoint_row = checkpoint_rows.get((method, draft.query_paper_id))
            if checkpoint_row and not checkpoint_row.get("error"):
                ranked_ids = [str(candidate_id) for candidate_id in checkpoint_row.get("ranked_ids") or []]
                metrics = {
                    key: float(value)
                    for key, value in (checkpoint_row.get("metrics") or {}).items()
                }
                metric_rows[method].append(metrics)
                if metrics.get("recall@5", 0.0) == 0.0 and len(misses[method]) < 10:
                    misses[method].append({
                        "query_paper_id": draft.query_paper_id,
                        "query_dataset_name": draft.query_dataset_name,
                        "gold_prior_paper_ids": gold,
                        "top_retrieved": ranked_ids[:5],
                    })
                completed_work += 1
                resumed_work += 1
                should_log = (
                    progress_every > 0
                    and (
                        completed_work == 1
                        or completed_work % progress_every == 0
                        or completed_work == total_work
                    )
                )
                if should_log:
                    elapsed = time.monotonic() - started_at
                    remaining_calls = total_work - completed_work
                    new_calls = max(1, completed_work - resumed_work)
                    estimated_remaining = 0.0 if completed_work == resumed_work else (elapsed / new_calls) * remaining_calls
                    progress_log(
                        "retrieval_progress",
                        completed_rank_calls=completed_work,
                        resumed_rank_calls=resumed_work,
                        total_rank_calls=total_work,
                        remaining_rank_calls=remaining_calls,
                        draft_index=draft_index,
                        total_drafts=len(valid_drafts),
                        method=method,
                        query_paper_id=draft.query_paper_id,
                        source="rank_checkpoint",
                        elapsed_seconds=round(elapsed, 1),
                        estimated_remaining_seconds=round(estimated_remaining, 1),
                        estimated_remaining=seconds_to_hms(estimated_remaining),
                        cache=cache.stats(),
                    )
                continue
            try:
                ranked_ids = method_objects[method].rank(draft, top_k)
                metrics = evaluate_retrieval_run(ranked_ids, gold)
                if top_k >= 10:
                    metrics["recall@10"] = 1.0 if any(candidate_id in gold for candidate_id in ranked_ids[:10]) else 0.0
                metrics.update(method_objects[method].metrics_for_last_rank(draft, ranked_ids))
                metric_rows[method].append(metrics)
                append_rank_checkpoint(checkpoint_path, {
                    "run_key": run_key,
                    "method": method,
                    "query_paper_id": draft.query_paper_id,
                    "query_dataset_name": draft.query_dataset_name,
                    "gold_prior_paper_ids": gold,
                    "ranked_ids": ranked_ids,
                    "metrics": metrics,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
                if metrics.get("recall@5", 0.0) == 0.0 and len(misses[method]) < 10:
                    misses[method].append({
                        "query_paper_id": draft.query_paper_id,
                        "query_dataset_name": draft.query_dataset_name,
                        "gold_prior_paper_ids": gold,
                        "top_retrieved": ranked_ids[:5],
                    })
            except Exception as exc:
                skipped[f"{method}_failed"].append(f"{draft.query_paper_id}: {exc}")
                append_rank_checkpoint(checkpoint_path, {
                    "run_key": run_key,
                    "method": method,
                    "query_paper_id": draft.query_paper_id,
                    "query_dataset_name": draft.query_dataset_name,
                    "gold_prior_paper_ids": gold,
                    "error": str(exc),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
            finally:
                completed_work += 1
                should_log = (
                    progress_every > 0
                    and (
                        completed_work == 1
                        or completed_work % progress_every == 0
                        or completed_work == total_work
                    )
                )
                if should_log:
                    elapsed = time.monotonic() - started_at
                    remaining_calls = total_work - completed_work
                    new_calls = max(1, completed_work - resumed_work)
                    estimated_remaining = (elapsed / new_calls) * remaining_calls
                    progress_log(
                        "retrieval_progress",
                        completed_rank_calls=completed_work,
                        resumed_rank_calls=resumed_work,
                        total_rank_calls=total_work,
                        remaining_rank_calls=remaining_calls,
                        draft_index=draft_index,
                        total_drafts=len(valid_drafts),
                        method=method,
                        query_paper_id=draft.query_paper_id,
                        source="computed",
                        elapsed_seconds=round(elapsed, 1),
                        avg_seconds_per_new_rank_call=round(elapsed / new_calls, 3),
                        estimated_remaining_seconds=round(estimated_remaining, 1),
                        estimated_remaining=seconds_to_hms(estimated_remaining),
                        cache=cache.stats(),
                    )
                continue

    summaries = {
        method: summarize_metric_runs(rows)
        for method, rows in metric_rows.items()
        if rows
    }
    progress_log(
        "retrieval_done",
        completed_rank_calls=completed_work,
        resumed_rank_calls=resumed_work,
        total_rank_calls=total_work,
        elapsed_seconds=round(time.monotonic() - started_at, 1),
        rank_checkpoint_jsonl=str(checkpoint_path),
        cache=cache.stats(),
        methods_with_results=list(summaries.keys()),
    )
    return summaries, misses, skipped, {
        **cache.stats(),
        "rank_checkpoint_jsonl": str(checkpoint_path),
        "rank_checkpoint_run_key": run_key,
        "rank_checkpoint_rows_loaded": len(checkpoint_rows),
        "rank_checkpoint_rows_resumed": resumed_work,
    }


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    f1s = []
    for label in ORDINAL_LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return safe_mean(f1s)


def added_information_summary(gold: Sequence[str], pred: Sequence[str]) -> dict:
    gold_int = [ORDINAL_TO_INT[label] for label in gold]
    pred_int = [ORDINAL_TO_INT[label] for label in pred]
    confusion = {
        gold_label: {pred_label: 0 for pred_label in ORDINAL_LABELS}
        for gold_label in ORDINAL_LABELS
    }
    for gold_label, pred_label in zip(gold, pred):
        confusion[gold_label][pred_label] += 1
    return {
        "n": len(gold),
        "accuracy": safe_mean([1.0 if g == p else 0.0 for g, p in zip(gold, pred)]),
        "macro_f1": macro_f1(gold, pred),
        "spearman": spearman_correlation(gold_int, pred_int),
        "confusion": confusion,
    }


def heuristic_added_information_label(query_acus: Sequence[str], prior_acus: Sequence[str]) -> str:
    if not query_acus:
        return "repackaging"
    prior_tokens = set(token for acu in prior_acus for token in tokenize(acu))
    if not prior_tokens:
        return "substantial"
    unsupported = []
    for acu in query_acus:
        tokens = set(tokenize(acu))
        overlap = len(tokens & prior_tokens) / max(len(tokens), 1)
        unsupported.append(1.0 - overlap)
    return ordinal_from_score(safe_mean(unsupported))


def parse_curator_row(row) -> AddedInformationLabelOutput:
    if hasattr(row, "label"):
        return row
    if isinstance(row, dict) and "parsed_response_message" in row:
        parsed = row["parsed_response_message"]
        if isinstance(parsed, AddedInformationLabelOutput):
            return parsed
        return AddedInformationLabelOutput(**parsed)
    if isinstance(row, dict):
        return AddedInformationLabelOutput(**row)
    raise TypeError(f"Unsupported LLM response row: {type(row)}")


def model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_curator_attribution_row(row) -> AddedInformationAttributionOutput:
    if isinstance(row, AddedInformationAttributionOutput):
        return row
    if hasattr(row, "attributions"):
        return row
    if isinstance(row, dict) and "parsed_response_message" in row:
        parsed = row["parsed_response_message"]
        if isinstance(parsed, AddedInformationAttributionOutput):
            return parsed
        return AddedInformationAttributionOutput(**parsed)
    if isinstance(row, dict):
        return AddedInformationAttributionOutput(**row)
    raise TypeError(f"Unsupported LLM attribution response row: {type(row)}")


def parse_json_object(text: str) -> dict:
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


def normalize_delta_type_value(value: str) -> str:
    normalized = (value or "").strip().lower().replace("_", " ").replace("-", " ")
    if normalized in {
        "task/domain", "task", "domain", "topic", "use case", "application",
    }:
        return "task/domain"
    if normalized in {
        "data/source", "data source", "source", "sources", "collection", "origin",
    }:
        return "data/source"
    if normalized in {
        "annotation/protocol", "annotation", "protocol", "method", "methods",
        "methodology", "procedure", "construction method", "labeling", "labelling",
    }:
        return "annotation/protocol"
    if normalized in {
        "scale/coverage", "scale", "coverage", "size", "language coverage",
        "domain coverage",
    }:
        return "scale/coverage"
    if normalized in {
        "evaluation/use", "evaluation", "use", "benchmark", "metric", "metrics",
    }:
        return "evaluation/use"
    if normalized in {
        "availability/quality", "availability", "quality", "release", "license",
    }:
        return "availability/quality"
    if normalized in {"other", "unclear", "unknown"}:
        return "other"
    return "other"


def repair_attribution_payload(payload: dict) -> dict:
    for attribution in payload.get("attributions") or []:
        if isinstance(attribution, dict):
            attribution["delta_type"] = normalize_delta_type_value(str(attribution.get("delta_type") or "other"))
    return payload


def direct_openai_attribution(
    query_dataset_name: str,
    query_acus: Sequence[str],
    prior_acus: Sequence[str],
    *,
    model_name: str,
) -> AddedInformationAttributionOutput:
    client = ensure_openai_client()
    openai_model = model_name.removeprefix("openai/")
    prompt = ATTRIBUTION_PROMPT.format(
        query_dataset_name=query_dataset_name,
        query_acus="\n".join(
            f"- {acu['id']}: {acu['text']}"
            for acu in with_acu_ids(query_acus, "q")
        ) or "- None",
        prior_acus="\n".join(
            f"- {acu['id']}: {acu['text']}"
            for acu in with_acu_ids(prior_acus, "p")
        ) or "- None",
    )
    prompt += (
        "\nReturn exactly this JSON shape with no markdown fences:\n"
        '{"attributions":[{"query_acu_id":"q0","query_acu":"...",'
        '"support_status":"supported|partially_supported|unsupported|contradicted|not_comparable",'
        '"best_prior_acu_ids":["p0"],"delta_type":"task/domain|data/source|annotation/protocol|scale/coverage|evaluation/use|availability/quality|other",'
        '"importance":"low|medium|high","rationale":"..."}]}\n'
    )
    response = client.responses.create(model=openai_model, input=prompt)
    return AddedInformationAttributionOutput(**repair_attribution_payload(parse_json_object(response.output_text)))


def with_acu_ids(acus: Sequence[str], prefix: str) -> List[dict]:
    return [
        {"id": f"{prefix}{idx}", "text": acu}
        for idx, acu in enumerate(acus)
        if acu
    ]


def normalize_attributions(
    output: AddedInformationAttributionOutput,
    query_acus: Sequence[str],
    prior_acus: Sequence[str],
) -> List[ClaimAttributionOutput]:
    expected_query = {f"q{idx}": acu for idx, acu in enumerate(query_acus) if acu}
    allowed_prior_ids = {f"p{idx}" for idx, acu in enumerate(prior_acus) if acu}
    seen = set()
    normalized: List[ClaimAttributionOutput] = []

    for attribution in output.attributions:
        if attribution.query_acu_id not in expected_query:
            raise ValueError(f"Unknown query_acu_id: {attribution.query_acu_id}")
        if attribution.query_acu_id in seen:
            raise ValueError(f"Duplicate attribution for query_acu_id: {attribution.query_acu_id}")
        unknown_prior_ids = [prior_id for prior_id in attribution.best_prior_acu_ids if prior_id not in allowed_prior_ids]
        if unknown_prior_ids:
            raise ValueError(f"Unknown best_prior_acu_ids: {unknown_prior_ids}")
        if attribution.query_acu != expected_query[attribution.query_acu_id]:
            attribution.query_acu = expected_query[attribution.query_acu_id]
        normalized.append(attribution)
        seen.add(attribution.query_acu_id)

    missing = [query_id for query_id in expected_query if query_id not in seen]
    if missing:
        raise ValueError(f"Missing attribution for query_acu_id(s): {missing}")
    return normalized


def heuristic_claim_attributions(
    query_acus: Sequence[str],
    prior_acus: Sequence[str],
) -> List[ClaimAttributionOutput]:
    prior_tokens = [(f"p{idx}", acu, set(tokenize(acu))) for idx, acu in enumerate(prior_acus) if acu]
    output = []
    for idx, query_acu in enumerate(query_acus):
        query_tokens = set(tokenize(query_acu))
        best_prior_id = ""
        best_overlap = 0.0
        for prior_id, _, tokens in prior_tokens:
            overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_prior_id = prior_id
        if not prior_tokens:
            status = "unsupported"
            best_prior_ids: List[str] = []
        elif best_overlap >= 0.75:
            status = "supported"
            best_prior_ids = [best_prior_id]
        elif best_overlap >= 0.25:
            status = "partially_supported"
            best_prior_ids = [best_prior_id]
        else:
            status = "unsupported"
            best_prior_ids = []
        output.append(ClaimAttributionOutput(
            query_acu_id=f"q{idx}",
            query_acu=query_acu,
            support_status=status,  # type: ignore[arg-type]
            best_prior_acu_ids=best_prior_ids,
            delta_type="other",
            importance="medium",
            rationale=(
                f"Heuristic token-overlap attribution for query ACU '{query_acu}'."
                if best_prior_ids
                else f"No prior ACU had enough token overlap to support query ACU '{query_acu}'."
            ),
        ))
    return output


def added_information_profile(attributions: Sequence[ClaimAttributionOutput]) -> dict:
    counts = Counter(attribution.support_status for attribution in attributions)
    total = len(attributions)
    unsupported_by_delta_type = Counter(
        attribution.delta_type
        for attribution in attributions
        if attribution.support_status == "unsupported"
    )
    score_numerator = 0.0
    score_denominator = 0.0
    for attribution in attributions:
        if attribution.support_status in EXCLUDED_SUPPORT_STATUSES:
            continue
        delta_value = SUPPORT_DELTA_VALUES[attribution.support_status]
        weight = IMPORTANCE_WEIGHTS[attribution.importance]
        score_numerator += delta_value * weight
        score_denominator += weight
    score = score_numerator / score_denominator if score_denominator else 0.0
    return {
        "n_query_acus": total,
        "added_information_score": score,
        "support_counts": dict(counts),
        "support_percentages": {
            status: counts.get(status, 0) / total if total else 0.0
            for status in ["supported", "partially_supported", "unsupported", "contradicted", "not_comparable"]
        },
        "unsupported_by_delta_type": dict(unsupported_by_delta_type),
        "excluded_from_score_count": sum(counts.get(status, 0) for status in EXCLUDED_SUPPORT_STATUSES),
    }


def summarize_attribution_profiles(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"n": 0}
    percentages_by_status: Dict[str, List[float]] = defaultdict(list)
    unsupported_by_delta_type: Counter = Counter()
    for row in rows:
        profile = row["profile"]
        for status, value in profile.get("support_percentages", {}).items():
            percentages_by_status[status].append(float(value))
        unsupported_by_delta_type.update(profile.get("unsupported_by_delta_type", {}))
    return {
        "n": len(rows),
        "mean_added_information_score": safe_mean([row["profile"]["added_information_score"] for row in rows]),
        "mean_support_percentages": {
            status: safe_mean(values)
            for status, values in sorted(percentages_by_status.items())
        },
        "unsupported_by_delta_type": dict(unsupported_by_delta_type),
    }


def attribution_cache_key(
    draft: BenchmarkDraftRecord,
    prior_acus: Sequence[str],
    *,
    condition: str,
    method: str,
    model_name: str,
) -> dict:
    return {
        "query_paper_id": draft.query_paper_id,
        "query_dataset_name": draft.query_dataset_name,
        "query_acus": list(draft.query_acus),
        "prior_acus": list(prior_acus),
        "condition": condition,
        "method": method,
        "model_name": model_name,
        "schema": "claim_attribution_v1",
    }


def run_attributor_for_example(
    draft: BenchmarkDraftRecord,
    prior_acus: Sequence[str],
    *,
    condition: str,
    method: str,
    model_name: str,
    cache: JsonCache,
) -> dict:
    key = attribution_cache_key(
        draft,
        prior_acus,
        condition=condition,
        method=method,
        model_name=model_name,
    )
    cached = cache.get("added_information_attribution", key)
    if cached is not None:
        return cached

    query_acus = list(draft.query_acus)
    if method == "heuristic":
        attributions = heuristic_claim_attributions(query_acus, prior_acus)
    elif method == "llm":
        output = direct_openai_attribution(
            draft.query_dataset_name,
            query_acus,
            prior_acus,
            model_name=model_name,
        )
        attributions = normalize_attributions(output, query_acus, prior_acus)
    else:
        raise ValueError(f"Unknown attribution method: {method}")

    profile = added_information_profile(attributions)
    result = {
        "query_paper_id": draft.query_paper_id,
        "query_dataset_name": draft.query_dataset_name,
        "condition": condition,
        "attribution_method": method,
        "model_name": model_name,
        "query_acus": with_acu_ids(query_acus, "q"),
        "prior_acus": with_acu_ids(prior_acus, "p"),
        "attributions": [model_to_dict(attribution) for attribution in attributions],
        "profile": profile,
    }
    return cache.set("added_information_attribution", key, result)


def run_added_information_attribution(
    drafts: Sequence[BenchmarkDraftRecord],
    payloads: Dict[str, dict],
    rankings_by_condition: Dict[str, Dict[str, List[str]]],
    *,
    conditions: Sequence[str],
    method: str,
    model_name: str,
    top_k: int,
    cache_dir: str,
) -> dict:
    cache = JsonCache(cache_dir)
    rows = []
    errors = []
    for condition in conditions:
        for draft in drafts:
            if condition == "oracle":
                prior_acus = list(draft.gold_prior_support_acus)
            else:
                ranked_ids = rankings_by_condition.get(condition, {}).get(draft.query_paper_id, [])
                prior_acus = collect_acus_for_papers(ranked_ids[:top_k], payloads)
            try:
                rows.append(run_attributor_for_example(
                    draft,
                    prior_acus,
                    condition=condition,
                    method=method,
                    model_name=model_name,
                    cache=cache,
                ))
            except Exception as exc:
                errors.append({
                    "query_paper_id": draft.query_paper_id,
                    "condition": condition,
                    "error": str(exc),
                })
    return {
        "conditions": {
            condition: summarize_attribution_profiles([
                row for row in rows if row["condition"] == condition
            ])
            for condition in conditions
        },
        "rows": rows,
        "errors": errors,
    }


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def attribution_annotation_rows(attribution_rows: Sequence[dict]) -> List[dict]:
    rows = []
    for row in attribution_rows:
        prior_by_id = {prior["id"]: prior["text"] for prior in row.get("prior_acus", [])}
        for attribution in row.get("attributions", []):
            rows.append({
                "query_paper_id": row["query_paper_id"],
                "query_dataset_name": row["query_dataset_name"],
                "condition": row["condition"],
                "query_acu_id": attribution["query_acu_id"],
                "query_acu": attribution["query_acu"],
                "pred_support_status": attribution["support_status"],
                "pred_best_prior_acus": [
                    {"id": prior_id, "text": prior_by_id.get(prior_id, "")}
                    for prior_id in attribution.get("best_prior_acu_ids", [])
                ],
                "pred_delta_type": attribution["delta_type"],
                "pred_importance": attribution["importance"],
                "pred_rationale": attribution["rationale"],
                "gold_support_status": None,
                "selected_evidence_relevant": None,
                "rationale_grounded": None,
            })
    return rows


def support_macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    labels = ["supported", "partially_supported", "unsupported"]
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return safe_mean(f1s)


def evaluate_attribution_human_labels(path: str | Path) -> dict:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    labeled = [
        row for row in rows
        if row.get("gold_support_status") in {"supported", "partially_supported", "unsupported"}
    ]
    if not labeled:
        return {"n": 0, "error": "No rows with gold_support_status."}
    gold = [row["gold_support_status"] for row in labeled]
    pred = [row["pred_support_status"] for row in labeled]
    evidence_rows = [row for row in labeled if row.get("selected_evidence_relevant") is not None]
    rationale_rows = [row for row in labeled if row.get("rationale_grounded") is not None]
    return {
        "n": len(labeled),
        "evidence_label_accuracy": safe_mean([1.0 if g == p else 0.0 for g, p in zip(gold, pred)]),
        "macro_f1": support_macro_f1(gold, pred),
        "evidence_precision": safe_mean([
            1.0 if row.get("selected_evidence_relevant") else 0.0
            for row in evidence_rows
        ]),
        "rationale_groundedness_rate": safe_mean([
            1.0 if row.get("rationale_grounded") else 0.0
            for row in rationale_rows
        ]),
    }


def run_added_information_eval(
    drafts: Sequence[BenchmarkDraftRecord],
    payloads: Dict[str, dict],
    retrieval_rankings: Dict[str, List[str]],
    *,
    mode: str,
    method: str,
    model_name: str,
    top_k: int,
) -> dict:
    examples = []
    gold_labels = []
    pred_labels = []

    for draft in drafts:
        if mode == "oracle":
            prior_acus = list(draft.gold_prior_support_acus)
        else:
            prior_acus = collect_acus_for_papers(retrieval_rankings.get(draft.query_paper_id, [])[:top_k], payloads)
        if not prior_acus:
            continue
        examples.append({
            "query_paper_id": draft.query_paper_id,
            "query_dataset_name": draft.query_dataset_name,
            "query_acus": list(draft.query_acus),
            "prior_acus": prior_acus,
            "gold_label": draft.gold_added_information_label,
        })

    if not examples:
        return {"n": 0, "error": "No examples with prior ACUs."}

    gold_labels = [example["gold_label"] for example in examples]
    if method == "heuristic":
        pred_labels = [
            heuristic_added_information_label(example["query_acus"], example["prior_acus"])
            for example in examples
        ]
    elif method == "llm":
        if curator is None or AddedInformationLabeler is None:
            return {"n": len(examples), "error": "bespokelabs.curator is unavailable."}
        labeler = AddedInformationLabeler(model_name=model_name)
        responses = labeler(examples)
        pred_labels = [parse_curator_row(row).label for row in responses.dataset]
    else:
        raise ValueError(f"Unknown added-information method: {method}")

    result = added_information_summary(gold_labels, pred_labels)
    result["examples"] = [
        {
            "query_paper_id": example["query_paper_id"],
            "gold": gold,
            "pred": pred,
        }
        for example, gold, pred in zip(examples[:10], gold_labels[:10], pred_labels[:10])
    ]
    return result


def build_rankings_for_method(
    drafts: Sequence[BenchmarkDraftRecord],
    payloads: Dict[str, dict],
    cited_by_query: Dict[str, set],
    method: str,
    top_k: int,
    *,
    cache_dir: str,
    allow_model_download: bool,
    allow_fallback_methods: bool,
    rerank_depth: int,
    gpt_model: str,
) -> Dict[str, List[str]]:
    ctx = RetrievalContext(
        drafts=drafts,
        payloads=payloads,
        cited_by_query=cited_by_query,
        cache=JsonCache(cache_dir),
        top_k=top_k,
        allow_model_download=allow_model_download,
        allow_fallback_methods=allow_fallback_methods,
        rerank_depth=rerank_depth,
        gpt_model=gpt_model,
    )
    retriever = build_retriever(method, ctx)
    rankings = {}
    for draft in drafts:
        rankings[draft.query_paper_id] = retriever.rank(draft, top_k)
    return rankings


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate complete benchmark rows for retrieval and added-information estimation.")
    parser.add_argument("--drafts", default=DEFAULT_DRAFTS)
    parser.add_argument("--processed-bank", default=DEFAULT_PROCESSED_BANK)
    parser.add_argument("--acl-benchmark-jsonl", default=None, help="Use ACL citation-grounded benchmark rows instead of legacy benchmark drafts.")
    parser.add_argument("--prior-extractions-jsonl", default=None, help="Use full-text prior extraction JSONL as the retrieval corpus.")
    parser.add_argument("--previous-work", default=DEFAULT_PREVIOUS_WORK)
    parser.add_argument("--model", default=os.environ.get("BENCHMARK_BUILDER_MODEL", "gpt-5.4"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N complete rows for smoke tests.")
    parser.add_argument("--methods", nargs="+", default=["lexical", "dense", "fusion", "rank_fusion", "hybrid_rerank"])
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--added-info-method", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--end-to-end-retrieval-method", default="hybrid_rerank")
    parser.add_argument("--run-attribution", action="store_true", help="Run claim-level added-information evidence attribution.")
    parser.add_argument("--attribution-method", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument(
        "--attribution-conditions",
        nargs="+",
        default=["oracle", "fusion"],
        help="Attribution support conditions. Use `oracle` or any retrieval method name.",
    )
    parser.add_argument("--attribution-output-jsonl", default=None, help="Write claim-level attribution rows as JSONL.")
    parser.add_argument("--attribution-annotation-jsonl", default=None, help="Write an ACU-level human annotation template JSONL.")
    parser.add_argument("--attribution-human-eval-jsonl", default=None, help="Evaluate completed human labels for attribution evidence quality.")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow sentence-transformers/HuggingFace to download missing dense models.")
    parser.add_argument("--allow-fallback-methods", action="store_true", help="Allow SPLADE/ColBERT fallback behavior from HybridSupportRetriever.")
    parser.add_argument("--cache-dir", default=DEFAULT_RETRIEVAL_CACHE)
    parser.add_argument("--rerank-depth", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10, help="Print retrieval progress to stderr every N rank calls. Use 0 to disable.")
    parser.add_argument("--rank-checkpoint-jsonl", default=None, help="Incrementally store per-query ranking outputs in this JSONL. Defaults to a run-keyed file under --cache-dir.")
    parser.add_argument("--no-resume-rank-checkpoint", action="store_true", help="Do not reuse completed per-query rankings from the run checkpoint.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    if not args.allow_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    drafts = complete_acl_drafts(args.acl_benchmark_jsonl) if args.acl_benchmark_jsonl else complete_drafts(args.drafts)
    if args.limit is not None:
        drafts = drafts[:args.limit]
    if not drafts:
        raise SystemExit("No complete benchmark rows found.")
    payloads = fulltext_extraction_payloads(args.prior_extractions_jsonl) if args.prior_extractions_jsonl else processed_payloads(args.processed_bank)
    cited_by_query = {} if args.acl_benchmark_jsonl else candidate_papers_by_query(args.previous_work)

    retrieval, misses, skipped, cache_stats = run_retrieval_eval(
        drafts,
        payloads,
        cited_by_query,
        methods=args.methods,
        top_k=args.top_k,
        cache_dir=args.cache_dir,
        allow_model_download=args.allow_model_download,
        allow_fallback_methods=args.allow_fallback_methods,
        rerank_depth=args.rerank_depth,
        gpt_model=args.model,
        progress_every=args.progress_every,
        rank_checkpoint_jsonl=args.rank_checkpoint_jsonl,
        resume_rank_checkpoint=not args.no_resume_rank_checkpoint,
    )

    report = {
        "row_counts": {
            "complete": len(drafts),
            "processed_corpus": len(payloads),
        },
        "legacy_label_distribution": dict(Counter(draft.gold_added_information_label for draft in drafts if draft.gold_added_information_label)),
        "retrieval": retrieval,
        "miss_examples": misses,
        "skipped": dict(skipped),
        "cache_stats": cache_stats,
        "debug_counters": hybrid_debug_counters(),
    }

    if not args.retrieval_only:
        report["added_information"] = {}
        report["added_information"]["oracle"] = run_added_information_eval(
            drafts,
            payloads,
            {},
            mode="oracle",
            method=args.added_info_method,
            model_name=args.model,
            top_k=args.top_k,
        )
        try:
            rankings = build_rankings_for_method(
                drafts,
                payloads,
                cited_by_query,
                args.end_to_end_retrieval_method,
                args.top_k,
                cache_dir=args.cache_dir,
                allow_model_download=args.allow_model_download,
                allow_fallback_methods=args.allow_fallback_methods,
                rerank_depth=args.rerank_depth,
                gpt_model=args.model,
            )
        except Exception as exc:
            if args.end_to_end_retrieval_method != "lexical":
                try:
                    rankings = build_rankings_for_method(
                        drafts,
                        payloads,
                        cited_by_query,
                        "lexical",
                        args.top_k,
                        cache_dir=args.cache_dir,
                        allow_model_download=args.allow_model_download,
                        allow_fallback_methods=args.allow_fallback_methods,
                        rerank_depth=args.rerank_depth,
                        gpt_model=args.model,
                    )
                except Exception as lexical_exc:
                    report["added_information"]["end_to_end"] = {
                        "error": str(exc),
                        "lexical_fallback_error": str(lexical_exc),
                    }
                else:
                    report["added_information"]["end_to_end"] = run_added_information_eval(
                        drafts,
                        payloads,
                        rankings,
                        mode="end_to_end",
                        method=args.added_info_method,
                        model_name=args.model,
                        top_k=args.top_k,
                    )
                    report["added_information"]["end_to_end"]["retrieval_method"] = "lexical_fallback"
                    report["added_information"]["end_to_end"]["fallback_reason"] = str(exc)
            else:
                report["added_information"]["end_to_end"] = {"error": str(exc)}
        else:
            report["added_information"]["end_to_end"] = run_added_information_eval(
                drafts,
                payloads,
                rankings,
                mode="end_to_end",
                method=args.added_info_method,
                model_name=args.model,
                top_k=args.top_k,
            )

    if args.run_attribution:
        attribution_conditions = unique(args.attribution_conditions)
        rankings_by_condition: Dict[str, Dict[str, List[str]]] = {}
        attribution_skipped: Dict[str, List[str]] = defaultdict(list)
        for condition in attribution_conditions:
            if condition == "oracle":
                continue
            try:
                rankings_by_condition[condition] = build_rankings_for_method(
                    drafts,
                    payloads,
                    cited_by_query,
                    condition,
                    args.top_k,
                    cache_dir=args.cache_dir,
                    allow_model_download=args.allow_model_download,
                    allow_fallback_methods=args.allow_fallback_methods,
                    rerank_depth=args.rerank_depth,
                    gpt_model=args.model,
                )
            except Exception as exc:
                attribution_skipped[f"{condition}_ranking_failed"].append(str(exc))
                rankings_by_condition[condition] = {}

        attribution_report = run_added_information_attribution(
            drafts,
            payloads,
            rankings_by_condition,
            conditions=attribution_conditions,
            method=args.attribution_method,
            model_name=args.model,
            top_k=args.top_k,
            cache_dir=args.cache_dir,
        )
        if attribution_skipped:
            attribution_report["skipped"] = dict(attribution_skipped)
        if args.attribution_output_jsonl:
            write_jsonl(args.attribution_output_jsonl, attribution_report["rows"])
            attribution_report["output_jsonl"] = args.attribution_output_jsonl
        if args.attribution_annotation_jsonl:
            annotation_rows = attribution_annotation_rows(attribution_report["rows"])
            write_jsonl(args.attribution_annotation_jsonl, annotation_rows)
            attribution_report["annotation_jsonl"] = args.attribution_annotation_jsonl
            attribution_report["annotation_rows"] = len(annotation_rows)
        if args.attribution_human_eval_jsonl:
            attribution_report["evidence_quality"] = evaluate_attribution_human_labels(args.attribution_human_eval_jsonl)
        report["added_information_attribution"] = {
            key: value
            for key, value in attribution_report.items()
            if key != "rows"
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_markdown:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_markdown_report(report), encoding="utf-8")


if __name__ == "__main__":
    main()
