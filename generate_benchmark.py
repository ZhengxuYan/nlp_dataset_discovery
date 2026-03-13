import os
import re
import json
import argparse
from typing import List, Dict, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from bespokelabs import curator
from dotenv import load_dotenv

class Dataset(BaseModel):
    name: str = Field(description="Name of the dataset")
    acus: List[str] = Field(description="List of 3-5 short Atomic Content Units describing the dataset")

class ScoredDataset(Dataset):
    ground_truth_novelty: float = Field(description="A ground truth novelty score from 0.0 to 1.0 (relative to the true_ancestor).")

class LineageCluster(BaseModel):
    anchor_id: str = Field(description="Stable identifier for the real-world anchor paper and introduced dataset.")
    domain: str = Field(description="The NLP domain or task for this cluster, e.g., 'Biomedical Named Entity Recognition'")
    true_ancestor: Dataset = Field(description="The foundational semantic ancestor dataset.")
    
    # Phase 1: Retrieval Eval 
    query_dataset: Dataset = Field(description="The descendant dataset we will use as the retrieval query.")
    hard_negative: Dataset = Field(description="A distractor dataset that has high lexical/word overlap with the query but is semantically unrelated (different domain/task).")
    soft_negative: Dataset = Field(description="A dataset in a similar domain but fundamentally a different lineage or task.")
    
    # Phase 2: Novelty Scoring Eval (Relative to true_ancestor)
    incremental_descendant: ScoredDataset = Field(description="Scenario A: An incremental descendant holding a minor new contribution. Expected novelty: 0.3 - 0.5")
    breakthrough_dataset: ScoredDataset = Field(description="Scenario B: A completely new paradigm or massively scaled dataset in the same domain. Expected novelty: 0.8 - 1.0")
    reproduction_dataset: ScoredDataset = Field(description="Scenario C: A pure reproduction/translation that offers zero new methods/tasks. Expected novelty: 0.0 - 0.1")


class BenchmarkGeneration(BaseModel):
    clusters: List[LineageCluster] = Field(description="A list of generated dataset lineage clusters")

class NegativeValidationResult(BaseModel):
    accept: bool = Field(description="Whether the candidate satisfies the requested negative type.")
    reason: str = Field(description="Short justification.")


def normalize_name(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def safe_join(parts: List[Optional[str]]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


BAD_NAME_SUBSTRINGS = {
    "gpt",
    "llama",
    "gemini",
    "claude",
    "olmo",
    "bert",
    "roberta",
    "t5",
    "model",
    "checkpoint",
    "backbone",
    "weights",
    "generated",
    "training data",
    "synthetic dataset",
    "unnamed",
}

BAD_ROLE_SUBSTRINGS = {
    "model",
    "baseline",
    "backbone",
    "method",
    "framework",
    "algorithm",
    "instruction-tuning",
    "fine-tuning",
}

GOOD_ROLE_SUBSTRINGS = {
    "dataset",
    "benchmark",
    "corpus",
    "evaluation",
    "test set",
}

NEGATIVE_BAD_NAME_SUBSTRINGS = {
    "instruction",
    "instruct",
    "graph",
    "knowledge graph",
    "preference",
    "olmo",
    "bert",
    "roberta",
    "t5",
}

NEGATIVE_BAD_USAGE_SUBSTRINGS = {
    "instruction-tuning",
    "fine-tuning",
    "used to train",
    "used for training",
    "train the",
    "knowledge graph",
    "synthetic data generation",
    "seed corpus",
    "service llm",
    "coarse-tuning",
    "query-document pair prediction",
    "qdpp",
    "mlm",
}

NEGATIVE_GOOD_USAGE_SUBSTRINGS = {
    "used as a benchmark",
    "evaluation benchmark",
    "evaluate",
    "benchmark",
    "test",
    "corpus",
    "dataset",
}

STOPWORDS = {
    "the", "and", "for", "with", "from", "used", "using", "into", "that", "this",
    "data", "dataset", "datasets", "benchmark", "benchmarks", "corpus", "task", "tasks",
    "model", "models", "evaluation", "train", "training", "test", "set", "new", "based",
    "primary", "used", "study", "resource", "system", "systems",
}

TASK_FAMILY_KEYWORDS = {
    "retrieval": {"retrieval", "search", "ranking", "rank", "rag", "query", "document", "documents", "passage"},
    "dialogue": {"dialogue", "dialog", "chat", "conversation", "conversational", "assistant", "multiwoz"},
    "translation": {"translation", "translate", "translated", "cross-lingual", "multilingual", "bilingual"},
    "classification": {"classification", "classify", "classifier", "label", "labeled", "sentiment"},
    "summarization": {"summarization", "summary", "summarize", "headline", "meeting"},
    "reasoning": {"reasoning", "math", "logic", "theorem", "proof", "problem", "qa", "question", "questions"},
    "evaluation": {"evaluate", "evaluation", "benchmark", "judge", "judgment", "leaderboard", "metric", "metrics"},
    "emotion": {"emotion", "affect", "sentiment", "vad", "arousal", "valence"},
    "biomedical": {"medical", "biomedical", "clinical", "health", "patient", "medicine"},
    "safety": {"safety", "toxicity", "harm", "bias", "fairness", "stereotype"},
    "coding": {"code", "coding", "programming", "apps", "repository"},
    "speech": {"speech", "audio", "asr", "spoken", "voice"},
    "table": {"table", "tabular", "spreadsheet"},
}

GENERIC_TASK_FAMILIES = {"evaluation", "classification"}
_SEMANTIC_MODEL = None


def build_dataset_description(dataset: Dict) -> str:
    return safe_join([
        f"Dataset name: {dataset['name']}.",
        f"Paper title: {dataset.get('paper_title', '')}.",
        f"Domain: {dataset.get('domain', '')}.",
        f"Usage: {dataset.get('usage_description', '')}.",
        f"Role: {dataset.get('role', '')}.",
        f"Creators: {', '.join(dataset.get('creators', [])) if dataset.get('creators') else ''}.",
        f"Source dataset: {dataset.get('source_dataset', '')}.",
        f"Transformation type: {dataset.get('transformation_type', '')}.",
        f"Abstract: {dataset.get('abstract', '')}.",
    ])


def get_semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SEMANTIC_MODEL


def lexical_overlap_score(left: Dict, right: Dict) -> float:
    left_tokens = tokenize(left.get("selection_text", ""))
    right_tokens = tokenize(right.get("selection_text", ""))
    if not left_tokens or not right_tokens:
        return 0.0
    shared = left_tokens & right_tokens
    return len(shared) / max(min(len(left_tokens), len(right_tokens)), 1)


def extract_task_tokens(record: Dict) -> set[str]:
    text = safe_join([
        record.get("name", ""),
        record.get("role", ""),
        record.get("usage_description", ""),
        record.get("domain", ""),
        record.get("source_dataset", ""),
    ])
    tokens = tokenize(text)
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def infer_task_families(record: Dict) -> set[str]:
    tokens = extract_task_tokens(record)
    families = set()
    for family, keywords in TASK_FAMILY_KEYWORDS.items():
        if tokens & keywords:
            families.add(family)
    return families


def semantic_similarity(left: Dict, right: Dict) -> float:
    left_emb = left.get("semantic_embedding")
    right_emb = right.get("semantic_embedding")
    if left_emb is None or right_emb is None:
        return 0.0
    denom = (np.linalg.norm(left_emb) * np.linalg.norm(right_emb))
    if denom == 0:
        return 0.0
    return float(np.dot(left_emb, right_emb) / denom)


def format_candidate_summary(record: Dict) -> str:
    return safe_join([
        f"Name: {record.get('name', '')}.",
        f"Domain: {record.get('domain', '')}.",
        f"Role: {record.get('role', '')}.",
        f"Usage: {record.get('usage_description', '')}.",
        f"Source dataset: {record.get('source_dataset', '')}.",
    ])


def is_dataset_like(record: Dict) -> bool:
    name = normalize_name(record.get("name", ""))
    role = normalize_name(record.get("role", ""))
    usage = normalize_name(record.get("usage_description", ""))
    source_dataset = normalize_name(record.get("source_dataset", ""))

    if not name:
        return False

    if any(token in name for token in BAD_NAME_SUBSTRINGS):
        return False

    if any(token in role for token in BAD_ROLE_SUBSTRINGS):
        return False

    if "used for additional experimental results" in usage:
        return False

    if "decoding throughput" in usage or "inference" in usage:
        return False

    positive_signals = 0
    if any(token in role for token in GOOD_ROLE_SUBSTRINGS):
        positive_signals += 1
    if "dataset" in usage or "benchmark" in usage or "corpus" in usage:
        positive_signals += 1
    if source_dataset not in {"", "none", "unknown"}:
        positive_signals += 1
    if record.get("is_introduced"):
        positive_signals += 1

    return positive_signals >= 1


def is_valid_anchor(record: Dict) -> bool:
    name = normalize_name(record.get("name", ""))
    if not is_dataset_like(record):
        return False
    if "unnamed" in name:
        return False
    if len(name) < 3:
        return False
    return True


def is_benchmark_candidate(record: Dict) -> bool:
    if not is_dataset_like(record):
        return False

    name = normalize_name(record.get("name", ""))
    role = normalize_name(record.get("role", ""))
    usage = normalize_name(record.get("usage_description", ""))

    if any(token in name for token in NEGATIVE_BAD_NAME_SUBSTRINGS):
        return False
    if any(token in usage for token in NEGATIVE_BAD_USAGE_SUBSTRINGS):
        return False
    if "instruction" in role or "fine-tuning" in role:
        return False

    positive_signals = 0
    if "benchmark" in role or "evaluation" in role or "corpus" in role or "test set" in role:
        positive_signals += 1
    if any(token in usage for token in NEGATIVE_GOOD_USAGE_SUBSTRINGS):
        positive_signals += 1
    if record.get("is_introduced"):
        positive_signals += 1

    return positive_signals >= 1


def rank_negative_candidates(anchor: Dict, candidates: List[Dict], same_domain: bool) -> List[Tuple[float, Dict]]:
    ranked = []
    anchor_domain = normalize_name(anchor.get("domain", ""))
    anchor_role = normalize_name(anchor.get("role", ""))
    anchor_task_tokens = extract_task_tokens(anchor)
    anchor_families = infer_task_families(anchor)
    anchor_specific_families = anchor_families - GENERIC_TASK_FAMILIES

    for candidate in candidates:
        candidate_domain = normalize_name(candidate.get("domain", ""))
        overlap = lexical_overlap_score(anchor, candidate)
        role_match = 1.0 if anchor_role and anchor_role == normalize_name(candidate.get("role", "")) else 0.0
        domain_match = 1.0 if anchor_domain and anchor_domain == candidate_domain else 0.0
        candidate_task_tokens = extract_task_tokens(candidate)
        shared_task_tokens = anchor_task_tokens & candidate_task_tokens
        task_overlap = len(shared_task_tokens) / max(min(len(anchor_task_tokens), len(candidate_task_tokens)), 1) if anchor_task_tokens and candidate_task_tokens else 0.0
        exact_name_penalty = 1.0 if normalize_name(anchor.get("name", "")) == normalize_name(candidate.get("name", "")) else 0.0
        candidate_families = infer_task_families(candidate)
        shared_families = anchor_families & candidate_families
        shared_specific_families = anchor_specific_families & candidate_families
        generic_only_overlap = bool(shared_families) and not bool(shared_specific_families)
        semantic_score = semantic_similarity(anchor, candidate)

        if same_domain:
            if anchor_domain and candidate_domain and anchor_domain != candidate_domain:
                continue
            if not shared_families:
                continue
            if generic_only_overlap and semantic_score < 0.35 and task_overlap < 0.12:
                continue
            if semantic_score < 0.28:
                continue
            score = (
                (4.0 * domain_match)
                + (3.5 * len(shared_specific_families))
                + (1.5 * len(shared_families))
                + (4.0 * semantic_score)
                + (2.5 * task_overlap)
                + (2.0 * role_match)
                + (1.0 * overlap)
                - (2.0 if generic_only_overlap else 0.0)
                - (3.0 * exact_name_penalty)
            )
        else:
            if anchor_domain and candidate_domain and anchor_domain == candidate_domain:
                continue
            if shared_specific_families:
                continue
            if task_overlap > 0.15 or (shared_families and not generic_only_overlap):
                continue
            if semantic_score > 0.24:
                continue
            score = (
                (3.0 * overlap)
                + (1.0 * (1.0 - domain_match))
                + (0.5 * (1.0 - role_match))
                + (1.5 * (1.0 - semantic_score))
                - (3.0 * task_overlap)
                - (1.5 * len(shared_families))
            )

        ranked.append((score, candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def shortlist_negatives(
    anchor: Dict,
    candidates: List[Dict],
    same_domain: bool,
    used_anchor_ids: Optional[set[str]] = None,
    limit: int = 5,
) -> List[Dict]:
    ranked = rank_negative_candidates(anchor, candidates, same_domain=same_domain)
    if ranked:
        if used_anchor_ids:
            ranked.sort(key=lambda item: (item[1]["anchor_id"] in used_anchor_ids, -item[0]))
        return [item[1] for item in ranked[:limit]]

    fallback = sorted(
        candidates,
        key=lambda candidate: lexical_overlap_score(anchor, candidate),
        reverse=True,
    )
    if not fallback:
        raise ValueError(f"No eligible negative candidates found for anchor {anchor['anchor_id']}")
    return fallback[:limit]


class NegativeValidator(curator.LLM):
    response_format = NegativeValidationResult

    def __init__(self, *args, negative_type: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.negative_type = negative_type

    def prompt(self, input: dict) -> str:
        negative_type = input["negative_type"]
        anchor = input["anchor"]
        candidate = input["candidate"]

        if negative_type == "soft":
            criteria = (
                "Accept only if the candidate is in the same task/domain neighborhood as the anchor, "
                "with genuinely comparable evaluation or benchmark purpose, "
                "but clearly not the same dataset lineage or a direct extension/reproduction. "
                "Reject candidates that are only loosely related or share generic LLM/evaluation language."
            )
        else:
            criteria = (
                "Accept only if the candidate is from a different task/domain neighborhood than the anchor, "
                "while still being a real dataset/benchmark resource. Reject model names, checkpoints, or training artifacts."
            )

        return f"""You are validating a benchmark negative candidate for NLP dataset lineage evaluation.

Negative type: {negative_type}
Decision rule: {criteria}

Anchor dataset:
{format_candidate_summary(anchor)}

Candidate negative:
{format_candidate_summary(candidate)}

Return whether to accept the candidate and a short reason.
"""


def validate_negative_candidates(
    anchor: Dict,
    candidates: List[Dict],
    negative_type: str,
    model_name: str,
    backend: Optional[str],
    backend_params: Optional[Dict],
) -> Dict:
    validator = NegativeValidator(
        model_name=model_name,
        backend=backend,
        backend_params=backend_params,
        negative_type=negative_type,
    )

    requests = [
        {
            "negative_type": negative_type,
            "anchor": anchor,
            "candidate": candidate,
        }
        for candidate in candidates
    ]
    results = validator(requests)
    dataset_results = getattr(results, "dataset", results)
    if hasattr(dataset_results, "__iter__") and not isinstance(dataset_results, (list, dict, str)):
        dataset_results = list(dataset_results)

    accepted: List[Dict] = []

    for candidate, verdict in zip(candidates, dataset_results):
        if hasattr(verdict, "model_dump"):
            verdict = verdict.model_dump()
        elif hasattr(verdict, "dict"):
            verdict = verdict.dict()
        elif isinstance(verdict, dict) and "parsed_response_message" in verdict:
            parsed = verdict["parsed_response_message"]
            if isinstance(parsed, list) and parsed:
                verdict = parsed[0]
            else:
                verdict = parsed

        if isinstance(verdict, dict) and verdict.get("accept"):
            accepted.append(candidate)

    if accepted:
        accepted.sort(key=lambda candidate: semantic_similarity(anchor, candidate), reverse=True)
        return accepted[0]

    return candidates[0]


def build_catalog(input_csv: str, input_jsonl: str) -> Tuple[List[Dict], List[Dict]]:
    import pandas as pd

    print("Loading data for grounding...")
    df_csv = pd.read_csv(input_csv)
    id_to_abstract = dict(zip(df_csv["arXiv ID"].astype(str), df_csv["Abstract"].fillna("")))
    id_to_title = dict(zip(df_csv["arXiv ID"].astype(str), df_csv["Title"].fillna("")))

    introduced_datasets: List[Dict] = []
    all_datasets: List[Dict] = []
    seen_anchor_keys = set()

    with open(input_jsonl, "r") as f:
        for line in f:
            item = json.loads(line)
            arxiv_id = str(item.get("arxiv_id", ""))
            paper_title = item.get("title", id_to_title.get(arxiv_id, "Unknown Title"))
            paper_abstract = item.get("abstract", id_to_abstract.get(arxiv_id, ""))
            if not paper_abstract:
                print(f"Warning: Abstract missing for {arxiv_id}, it may impact generation.")

            raw_domain = (
                item.get("nlp_domain")
                or item.get("primary_category")
                or item.get("categories")
                or "General NLP"
            )
            paper_meta = {
                "arxiv_id": arxiv_id,
                "paper_title": paper_title,
                "abstract": paper_abstract,
                "domain": raw_domain,
                "publication_venue": item.get("publication_venue", ""),
                "published_date": item.get("published_date", ""),
                "source_type": item.get("source_type", ""),
            }

            raw_datasets = item.get("datasets", {})
            if isinstance(raw_datasets, dict):
                dataset_entries = [(ds_name, ds_info or {}) for ds_name, ds_info in raw_datasets.items()]
            elif isinstance(raw_datasets, list):
                dataset_entries = []
                for ds_info in raw_datasets:
                    if not isinstance(ds_info, dict):
                        continue
                    ds_name = ds_info.get("name")
                    if not ds_name:
                        continue
                    dataset_entries.append((ds_name, ds_info))
            else:
                dataset_entries = []

            for ds_name, ds_info in dataset_entries:
                creators = ds_info.get("creators", []) or []
                if isinstance(creators, str):
                    creators = [creators]
                record = {
                    **paper_meta,
                    "name": ds_name,
                    "role": ds_info.get("role", ""),
                    "usage_description": ds_info.get("usage_description", ""),
                    "source_dataset": ds_info.get("source_dataset", ""),
                    "transformation_type": ds_info.get("transformation_type", ""),
                    "creators": creators,
                    "confidence": ds_info.get("confidence"),
                    "is_introduced": bool(ds_info.get("is_introduced", False)),
                }
                record["selection_text"] = build_dataset_description(record)
                record["anchor_id"] = f"{arxiv_id}::{normalize_name(ds_name)}"
                if is_dataset_like(record):
                    all_datasets.append(record)

                if record["is_introduced"] and is_valid_anchor(record):
                    dedupe_key = (arxiv_id, normalize_name(ds_name))
                    if dedupe_key in seen_anchor_keys:
                        continue
                    seen_anchor_keys.add(dedupe_key)
                    introduced_datasets.append(record)

    if all_datasets:
        print(f"Computing semantic embeddings for {len(all_datasets)} dataset candidates...")
        model = get_semantic_model()
        texts = [dataset["selection_text"] for dataset in all_datasets]
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        for dataset, emb in zip(all_datasets, embeddings):
            dataset["semantic_embedding"] = emb
        anchor_lookup = {dataset["anchor_id"]: dataset["semantic_embedding"] for dataset in all_datasets}
        for dataset in introduced_datasets:
            dataset["semantic_embedding"] = anchor_lookup.get(dataset["anchor_id"])

    return introduced_datasets, all_datasets

def attach_real_negatives(
    anchor: Dict,
    all_datasets: List[Dict],
    model_name: str,
    backend: Optional[str],
    backend_params: Optional[Dict],
) -> Dict:
    candidates = [
        dataset for dataset in all_datasets
        if dataset["anchor_id"] != anchor["anchor_id"]
        and dataset["arxiv_id"] != anchor["arxiv_id"]
        and is_benchmark_candidate(dataset)
    ]

    hard_candidates = shortlist_negatives(anchor, candidates, same_domain=False)
    hard_negative = validate_negative_candidates(
        anchor,
        hard_candidates,
        negative_type="hard",
        model_name=model_name,
        backend=backend,
        backend_params=backend_params,
    )
    soft_pool = [candidate for candidate in candidates if candidate["anchor_id"] != hard_negative["anchor_id"]]
    soft_candidates = shortlist_negatives(
        anchor,
        soft_pool,
        same_domain=True,
        used_anchor_ids={hard_negative["anchor_id"]},
    )
    soft_negative = validate_negative_candidates(
        anchor,
        soft_candidates,
        negative_type="soft",
        model_name=model_name,
        backend=backend,
        backend_params=backend_params,
    )

    request = dict(anchor)
    request["introduced_dataset"] = anchor["name"]
    request["title"] = anchor["paper_title"]
    request["hard_negative_real"] = hard_negative
    request["soft_negative_real"] = soft_negative
    return request


def build_generation_request(anchor: Dict) -> Dict:
    request = dict(anchor)
    request["introduced_dataset"] = anchor["name"]
    request["title"] = anchor["paper_title"]
    return request


def extract_single_cluster(result) -> Optional[Dict]:
    dataset_results = getattr(result, "dataset", result)
    if hasattr(dataset_results, "__iter__") and not isinstance(dataset_results, (list, dict, str)):
        dataset_results = list(dataset_results)
    elif not isinstance(dataset_results, list):
        dataset_results = [dataset_results]

    def coerce_cluster(candidate) -> Optional[Dict]:
        if not candidate:
            return None

        if hasattr(candidate, "model_dump"):
            candidate = candidate.model_dump()
        elif hasattr(candidate, "dict"):
            candidate = candidate.dict()

        if isinstance(candidate, list):
            if not candidate:
                return None
            return coerce_cluster(candidate[0])

        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                return None

        if not isinstance(candidate, dict):
            return None

        if "clusters" in candidate and candidate["clusters"]:
            return coerce_cluster(candidate["clusters"][0])

        if "true_ancestor" in candidate and "query_dataset" in candidate:
            return candidate

        return None

    for res in dataset_results:
        if hasattr(res, "model_dump"):
            candidate = res.model_dump()
        elif hasattr(res, "dict"):
            candidate = res.dict()
        else:
            candidate = res

        parsed = coerce_cluster(candidate)
        if parsed:
            return parsed

        if isinstance(candidate, dict):
            parsed = coerce_cluster(candidate.get("parsed_response_message"))
            if parsed:
                return parsed
            parsed = coerce_cluster(candidate.get("response_message"))
            if parsed:
                return parsed

    return None


class BenchmarkGenerator(curator.LLM):
    response_format = LineageCluster
    
    def prompt(self, input: dict) -> str:
        domain = input.get("domain", "General NLP")
        title = input.get("title", input.get("paper_title", ""))
        abstract = input.get("abstract", "")
        introduced_dataset = input.get("introduced_dataset", input.get("name", ""))
        anchor_id = input.get("anchor_id", "")
        anchor_usage = input.get("usage_description", "")
        anchor_role = input.get("role", "")
        anchor_source_dataset = input.get("source_dataset", "")
        anchor_transformation_type = input.get("transformation_type", "")
        
        return f"""You are an expert NLP researcher. Your task is to generate 1 highly realistic benchmark 'Dataset Lineage Cluster' anchored on a REAL paper.

You are grounding this cluster around a REAL peer-reviewed paper that introduced a dataset. Keep `anchor_id` EXACTLY as provided.
--- 
Anchor ID: {anchor_id}
NLP Domain/Topic: {domain}
Paper Title: {title}
Paper Abstract: {abstract}
Introduced Dataset: {introduced_dataset}
Introduced Dataset Usage: {anchor_usage}
Introduced Dataset Role: {anchor_role}
Introduced Dataset Source Dataset: {anchor_source_dataset}
Introduced Dataset Transformation Type: {anchor_transformation_type}
---

For this cluster, create outputs with these constraints (use 3-5 ACUs per dataset):
1. `true_ancestor`: Must represent the actual '{introduced_dataset}' from the paper above. Use the paper abstract and metadata to form accurate ACUs.

--- RETRIEVAL EVALUATION COMPONENT ---
2. `query_dataset`: A plausible synthetic descendant dataset that builds logically on the ancestor and stays realistic for the same research area.
3. `hard_negative`: A synthetic distractor dataset. It MUST have high lexical overlap with the `query_dataset` ACUs, but belong to a clearly different task/domain. It should be realistic enough to fool keyword-based retrieval.
4. `soft_negative`: A synthetic dataset in the SAME research task/domain neighborhood as the `query_dataset`, but from a clearly different lineage. It should feel like a plausible competing benchmark or parallel effort, not an ancestor or descendant.

--- NOVELTY SCORING EVALUATION COMPONENT ---
You must generate three relative datasets compared to the `true_ancestor` and assign them a `ground_truth_novelty` score (0.0=Identical, 1.0=Groundbreaking):
5. `incremental_descendant`: Scenario A. A minor iteration (e.g., +10% more data or adding one language). Ground Truth: ~0.3 - 0.5
6. `breakthrough_dataset`: Scenario B. A massive leap in scale or an entirely novel task paradigm. Ground Truth: ~0.8 - 1.0
7. `reproduction_dataset`: Scenario C. An exact re-annotation or direct translation without new methods. Ground Truth: ~0.0 - 0.1

ACU formatting: Each ACU must be a single, short, fully independent sentence making ONE claim.
Name constraints:
- `anchor_id` must be "{anchor_id}" exactly.
- `true_ancestor.name` must be "{introduced_dataset}" exactly.
- The descendant and novelty datasets should be synthetic but realistic.
Quality constraints:
- Do not use model names, checkpoints, or vague placeholders as dataset names.
- The `soft_negative` must share task/domain intent with the query, not just generic evaluation language.
- The `hard_negative` must be a real-looking dataset concept, but clearly outside the query's task/domain.
"""

def generate_benchmark(
    output_file: str,
    input_csv: str,
    input_jsonl: str,
    num_clusters: int,
    max_retries: int = 3,
    model_name: str = "gpt-5-mini",
    backend: Optional[str] = None,
    backend_params: Optional[Dict] = None
):
    import random

    introduced_datasets, all_datasets = build_catalog(input_csv, input_jsonl)
    print(f"Found {len(introduced_datasets)} deduplicated introduced dataset anchors.")
    
    print(f"Using model: {model_name} with backend: {backend or 'default'}")
    generator = BenchmarkGenerator(
        model_name=model_name,
        backend=backend,
        backend_params=backend_params
    )
    
    if num_clusters > len(introduced_datasets):
        print(f"Warning: Requested {num_clusters} clusters but only found {len(introduced_datasets)} anchors. Sampling with replacement.")
        sampled_anchors = random.choices(introduced_datasets, k=num_clusters)
    else:
        sampled_anchors = random.sample(introduced_datasets, num_clusters)
        
    print(f"Generating benchmark with {num_clusters} clusters...")

    requests = [build_generation_request(anchor) for anchor in sampled_anchors]

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        pass

    total_generated = 0

    for idx, request in enumerate(requests, start=1):
        print(f"Processing anchor {idx}/{len(requests)}: {request['anchor_id']}")
        cluster = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                result = generator([request])
                cluster = extract_single_cluster(result)
                if cluster:
                    break
                last_error = RuntimeError("No valid cluster parsed from model response.")
            except Exception as exc:
                last_error = exc
            print(f"  Attempt {attempt}/{max_retries} failed for {request['anchor_id']}: {last_error}")

        if not cluster:
            print(f"  -> Skipping {request['anchor_id']} after {max_retries} failed attempts.")
            continue

        cluster["anchor_id"] = request["anchor_id"]
        cluster["provenance"] = {
            "anchor_paper": {
                "arxiv_id": request["arxiv_id"],
                "title": request["paper_title"],
                "publication_venue": request.get("publication_venue", ""),
                "published_date": request.get("published_date", ""),
                "source_type": request.get("source_type", ""),
                "domain": request.get("domain", ""),
            },
            "true_ancestor_metadata": {
                "usage_description": request.get("usage_description", ""),
                "role": request.get("role", ""),
                "source_dataset": request.get("source_dataset", ""),
                "transformation_type": request.get("transformation_type", ""),
                "creators": request.get("creators", []),
                "confidence": request.get("confidence"),
            },
            "negative_generation": {
                "type": "synthetic",
                "uses_real_negative_pool": False,
            },
        }

        with open(output_file, "a") as f:
            f.write(json.dumps(cluster) + "\n")

        total_generated += 1
        print(f"  -> Wrote cluster {total_generated}/{len(requests)}")

    print(f"Successfully generated {total_generated} clusters and saved to {output_file}")
    if total_generated != len(requests):
        print(f"Warning: Requested {len(requests)} clusters but only wrote {total_generated}.")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/benchmark/retrieval_benchmark.jsonl")
    parser.add_argument("--input_csv", type=str, default="data/processed/arxiv_nlp_conf_papers_2023_2025.csv")
    parser.add_argument("--input_jsonl", type=str, default="data/processed/arxiv_nlp_conf_papers_2023_2025_dataset_analysis(gpt-5-mini).jsonl")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters to generate")
    parser.add_argument("--max_retries", type=int, default=3, help="Per-anchor generation retries before giving up")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model name for benchmark generation.")
    parser.add_argument("--backend", type=str, default=None, help="Curator backend, for example `openai` or `litellm`.")
    parser.add_argument("--backend-params", type=str, default=None, help="JSON string for backend parameters.")
    args = parser.parse_args()

    backend = args.backend if args.backend else os.environ.get("BACKEND")
    backend_params = None
    raw_backend_params = args.backend_params if args.backend_params else os.environ.get("BACKEND_PARAMS")
    if raw_backend_params:
        try:
            backend_params = json.loads(raw_backend_params)
        except json.JSONDecodeError:
            raise SystemExit("Error parsing backend params JSON.")
    
    generate_benchmark(
        args.output,
        args.input_csv,
        args.input_jsonl,
        args.num_clusters,
        args.max_retries,
        model_name=args.model,
        backend=backend,
        backend_params=backend_params
    )
