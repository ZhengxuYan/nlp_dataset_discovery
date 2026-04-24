from __future__ import annotations

import json
import gzip
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from pydantic import BaseModel, Field

from .benchmark_models import (
    BenchmarkDraftRecord,
    PaperBankRecord,
    PreviousWorkCandidateRecord,
    PriorPaperSummary,
    ProcessedBankRecord,
)
from .benchmark_store import (
    DEFAULT_BENCHMARK_DRAFTS_PATH,
    DEFAULT_PAPER_BANK_PATH,
    DEFAULT_PDF_DIR,
    DEFAULT_PREVIOUS_WORK_PATH,
    DEFAULT_PROCESSED_BANK_PATH,
    DEFAULT_PROCESSED_DIR,
    canonical_paper_id,
    candidate_id,
    ensure_benchmark_dirs,
    load_benchmark_drafts,
    load_paper_bank,
    load_previous_work_candidates,
    load_processed_bank,
    load_processed_query_papers,
    is_human_complete_draft,
    now_iso,
    patch_record,
    slugify,
    summarize_query_status,
    replace_benchmark_drafts_for_queries,
    replace_previous_work_candidates_for_queries,
    upsert_benchmark_drafts,
    upsert_paper_bank,
    upsert_previous_work_candidates,
    upsert_processed_bank,
    write_jsonl,
)
from .models import ScvPaperAnalysis
from .utils import download_file, get_text_from_pdf
from .utils import extract_tar_gz, get_text_from_latex

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_SRC_URL = "https://arxiv.org/src/{arxiv_id}"
MAX_REFERENCE_FILE_BYTES = 5_000_000


class PreviousWorkCandidate(BaseModel):
    dataset_name: Optional[str] = Field(default=None)
    paper_title: Optional[str] = Field(default=None)
    citation_key: Optional[str] = Field(default=None)
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(default=None)
    citation_marker: Optional[str] = Field(default=None)
    relationship_type: str = Field(default="unknown")
    description: str
    confidence: str = Field(default="medium")
    evidence_text: str


class PreviousWorkExtractionResult(BaseModel):
    candidates: List[PreviousWorkCandidate] = Field(default_factory=list)


PREVIOUS_WORK_PROMPT = """You are extracting previous-work candidates from a dataset paper.

Return only previous work that is plausibly relevant for comparing the query dataset against earlier datasets, benchmarks, corpora, or source datasets.
Do NOT extract generic method citations unless they are tied to a dataset or benchmark comparison.
Do NOT invent paper titles. If the exact paper title is not visible in the provided text or reference list, leave paper_title empty.
Prefer citation_key when the paper text uses LaTeX citations such as \\citep{{key}} or \\cite{{key1,key2}}.

For each candidate, extract:
- dataset_name: if explicitly mentioned
- paper_title: only if explicitly mentioned in paper text or the reference list
- citation_key: if a LaTeX citation key is visible
- authors: if explicitly mentioned
- year: if explicitly mentioned
- citation_marker: if a citation marker like [12] or (Smith, 2020) is visible
- relationship_type: one of
  - closest_prior_dataset
  - source_dataset
  - parallel_benchmark
  - evaluation_baseline
  - loosely_related
  - unknown
- description: one short explanation of why this previous work matters
- confidence: low / medium / high
- evidence_text: short supporting quote or paraphrase grounded in the paper text

Query paper title: {title}
Query abstract: {abstract}

Known introduced datasets and ACUs:
{dataset_context}

Reference list excerpt:
{references}

Paper text / LaTeX excerpt:
{text}
"""


@dataclass
class ReferenceEntry:
    key: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    raw: str = ""


@dataclass
class QueryPaperContext:
    text: str
    source_type: str
    source_dir: Optional[Path]
    references: Dict[str, ReferenceEntry]


def _condense_text(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text
    head_chars = int(max_chars * 0.7)
    tail_chars = max_chars - head_chars
    return f"{text[:head_chars]}\n\n[... truncated ...]\n\n{text[-tail_chars:]}"


def _normalize_braced_text(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text or "")
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_year(text: str) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", text or "")
    return int(match.group(0)) if match else None


def _extract_arxiv_id(text: str) -> Optional[str]:
    patterns = [
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"arXiv[:\s]+([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"\beprint\s*=\s*[{\"]([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)[}\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _split_authors(author_text: str) -> List[str]:
    if not author_text:
        return []
    author_text = _normalize_braced_text(author_text)
    if " and " in author_text:
        parts = author_text.split(" and ")
    else:
        parts = re.split(r",\s*(?=[A-Z][A-Za-z-]+(?:\s|$))", author_text)
    return [part.strip() for part in parts if part.strip()]


def _bibtex_entries_from_file(path: Path) -> Dict[str, ReferenceEntry]:
    try:
        import bibtexparser
    except Exception:
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            database = bibtexparser.load(handle)
    except Exception:
        return {}

    entries: Dict[str, ReferenceEntry] = {}
    for entry in database.entries:
        key = entry.get("ID") or entry.get("id")
        if not key:
            continue
        raw = " ".join(str(value) for value in entry.values())
        title = _normalize_braced_text(entry.get("title", ""))
        url = entry.get("url") or entry.get("link")
        doi = entry.get("doi")
        arxiv_id = _extract_arxiv_id(raw)
        entries[key] = ReferenceEntry(
            key=key,
            title=title,
            authors=_split_authors(entry.get("author", "")),
            year=_extract_year(entry.get("year", "") or raw),
            url=url,
            doi=doi,
            arxiv_id=arxiv_id,
            raw=raw,
        )
    return entries


def _bbl_entries_from_file(path: Path) -> Dict[str, ReferenceEntry]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    entries: Dict[str, ReferenceEntry] = {}
    chunks = re.split(r"(?=\\bibitem)", text)
    for chunk in chunks:
        if "\\bibitem" not in chunk:
            continue
        key_match = re.search(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", chunk)
        if not key_match:
            continue
        key = key_match.group(1).strip()
        raw = _normalize_braced_text(chunk)
        title = ""
        href_title = re.search(r"\\newblock\s+\\href\s*\{[^{}]+\}\s*\{(.+?)\}\s*\.\s*\\newblock", chunk, re.DOTALL)
        if href_title:
            title = _normalize_braced_text(href_title.group(1))
        quoted_title = re.search(r"``([^`]{8,250})''", chunk) or re.search(r'"([^"]{8,250})"', chunk)
        if not title and quoted_title:
            title = _normalize_braced_text(quoted_title.group(1))
        else:
            emph_title = re.search(r"\\(?:emph|textit|textbf)\{([^{}]{8,250})\}", chunk)
            if not title and emph_title:
                title = _normalize_braced_text(emph_title.group(1))
        url_match = re.search(r"https?://[^\s}]+", chunk)
        doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", chunk)
        entries[key] = ReferenceEntry(
            key=key,
            title=title,
            year=_extract_year(raw),
            url=url_match.group(0) if url_match else None,
            doi=doi_match.group(0) if doi_match else None,
            arxiv_id=_extract_arxiv_id(chunk),
            raw=raw[:1000],
        )
    return entries


def parse_reference_map(source_dir: Path) -> Dict[str, ReferenceEntry]:
    references: Dict[str, ReferenceEntry] = {}
    for bib_path in source_dir.glob("**/*.bib"):
        if bib_path.stat().st_size > MAX_REFERENCE_FILE_BYTES:
            continue
        references.update(_bibtex_entries_from_file(bib_path))
    for bbl_path in source_dir.glob("**/*.bbl"):
        if bbl_path.stat().st_size > MAX_REFERENCE_FILE_BYTES:
            continue
        for key, entry in _bbl_entries_from_file(bbl_path).items():
            existing = references.get(key)
            if existing and existing.title:
                continue
            references[key] = entry
    return references


def _extract_arxiv_source_archive(source_archive: Path, source_dir: Path) -> None:
    if extract_tar_gz(str(source_archive), str(source_dir)):
        return
    try:
        with gzip.open(source_archive, "rb") as src:
            content = src.read()
        target = source_dir / "source.tex"
        target.write_bytes(content)
    except Exception:
        return


def _format_reference_list(references: Dict[str, ReferenceEntry], max_entries: int = 120) -> str:
    lines = []
    for entry in list(references.values())[:max_entries]:
        title = entry.title or "(title unavailable)"
        authors = ", ".join(entry.authors[:3])
        extras = []
        if entry.year:
            extras.append(str(entry.year))
        if entry.arxiv_id:
            extras.append(f"arXiv:{entry.arxiv_id}")
        if entry.url:
            extras.append(entry.url)
        meta = "; ".join(part for part in [authors, *extras] if part)
        lines.append(f"- {entry.key}: {title}" + (f" ({meta})" if meta else ""))
    return "\n".join(lines) or "- No parsed references available."


def _find_reference_for_candidate(candidate: PreviousWorkCandidate, references: Dict[str, ReferenceEntry]) -> Optional[ReferenceEntry]:
    if candidate.citation_key:
        for key in re.split(r"[,;\s]+", candidate.citation_key):
            key = key.strip()
            if key in references:
                return references[key]
    if candidate.citation_marker:
        marker = candidate.citation_marker.lower()
        marker_year = _extract_year(marker)
        marker_author = None
        author_match = re.search(r"([a-z][a-z-]+)\s+(?:et\s+al\.?|and|&|,|\()", marker)
        if author_match:
            marker_author = author_match.group(1)
        for entry in references.values():
            haystack = " ".join([entry.key, entry.title, " ".join(entry.authors), str(entry.year or ""), entry.raw]).lower()
            if marker and marker in haystack:
                return entry
            if marker_year and entry.year == marker_year and marker_author:
                author_haystack = " ".join(entry.authors).lower()
                if marker_author in author_haystack or marker_author in entry.key.lower():
                    return entry
    if candidate.paper_title:
        best_entry = None
        best_score = 0.0
        for entry in references.values():
            score = _title_similarity(candidate.paper_title, entry.title)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry and best_score >= 85:
            return best_entry
    return None


def _candidate_mentions_dataset(candidate: PreviousWorkCandidate) -> bool:
    if candidate.dataset_name:
        return True
    evidence = f"{candidate.description} {candidate.evidence_text}".lower()
    return any(term in evidence for term in ["dataset", "corpus", "benchmark", "shared task", "shared-task", "repository", "collection"])


def _should_keep_candidate(candidate: PreviousWorkCandidate) -> bool:
    relationship = candidate.relationship_type if candidate.relationship_type in {
        "closest_prior_dataset",
        "source_dataset",
        "parallel_benchmark",
        "evaluation_baseline",
        "loosely_related",
        "unknown",
    } else "unknown"
    if relationship in {"closest_prior_dataset", "source_dataset"}:
        return _candidate_mentions_dataset(candidate)
    if relationship in {"parallel_benchmark", "evaluation_baseline"}:
        return _candidate_mentions_dataset(candidate)
    if relationship == "loosely_related":
        return bool(candidate.dataset_name) and _candidate_mentions_dataset(candidate)
    return _candidate_mentions_dataset(candidate)


def prepare_query_paper_context(record: Dict, fetch_source: bool = True) -> QueryPaperContext:
    arxiv_id = record.get("arxiv_id")
    if arxiv_id:
        source_dir = Path("data/extracted_papers") / arxiv_id
        source_archive = source_dir / "source.tar.gz"
        if fetch_source and not any(source_dir.glob("**/*.tex")):
            source_dir.mkdir(parents=True, exist_ok=True)
            download_file(ARXIV_SRC_URL.format(arxiv_id=arxiv_id), str(source_archive))
            _extract_arxiv_source_archive(source_archive, source_dir)
        if source_dir.exists() and any(source_dir.glob("**/*.tex")):
            latex_text = get_text_from_latex(str(source_dir))
            if latex_text:
                references = parse_reference_map(source_dir)
                return QueryPaperContext(text=latex_text, source_type="latex", source_dir=source_dir, references=references)

    pdf_text = read_query_paper_text(record)
    return QueryPaperContext(text=pdf_text, source_type="pdf" if pdf_text else "abstract", source_dir=None, references={})


def _build_previous_work_extractor(model_name: str, backend: Optional[str], backend_params: Optional[Dict]):
    from bespokelabs import curator

    class PreviousWorkExtractor(curator.LLM):
        response_format = PreviousWorkExtractionResult

        def __init__(self, *args, text_char_limit: int = 18000, **kwargs):
            super().__init__(*args, **kwargs)
            self.text_char_limit = text_char_limit

        def prompt(self, input: dict) -> str:
            dataset_context_lines = []
            for dataset in input.get("datasets", []):
                dataset_context_lines.append(f"- Dataset: {dataset.get('name', 'Unknown')}")
                for acu in dataset.get("acus", [])[:5]:
                    dataset_context_lines.append(f"  - ACU: {acu}")
                for acu in dataset.get("previous_work_acus", [])[:5]:
                    dataset_context_lines.append(f"  - Mentioned previous work fact: {acu}")
            return PREVIOUS_WORK_PROMPT.format(
                title=input.get("title", ""),
                abstract=input.get("abstract", ""),
                dataset_context="\n".join(dataset_context_lines) or "- None",
                references=_condense_text(input.get("references", ""), 12000),
                text=_condense_text(input.get("text", ""), self.text_char_limit),
            )

    return PreviousWorkExtractor(
        model_name=model_name,
        backend=backend,
        backend_params=backend_params,
    )


def read_query_paper_text(record: Dict) -> str:
    arxiv_id = record.get("arxiv_id")
    possible_paths = []
    if arxiv_id:
        possible_paths.append(Path("data/temp_scv") / arxiv_id / "paper.pdf")
        possible_paths.append(Path("data/benchmark/pdfs") / f"arxiv:{arxiv_id}" / "paper.pdf")
        possible_paths.append(Path("data/extracted_papers") / arxiv_id / "paper.pdf")
    for path in possible_paths:
        if path.exists():
            text = get_text_from_pdf(str(path))
            if text:
                return text
    return record.get("abstract", "")


def build_query_paper_bank_records(query_rows: Sequence[Dict], source_path: str) -> List[PaperBankRecord]:
    records: List[PaperBankRecord] = []
    for row in query_rows:
        paper_id = canonical_paper_id(row.get("arxiv_id"), row.get("title", ""))
        authors = [author.get("name", "") for author in row.get("metadata", {}).get("authors", row.get("authors", [])) if author.get("name")]
        year = None
        published_date = row.get("published_date") or row.get("metadata", {}).get("date")
        if published_date:
            try:
                year = int(str(published_date)[:4])
            except ValueError:
                year = None
        url = f"https://arxiv.org/abs/{row['arxiv_id']}" if row.get("arxiv_id") else None
        pdf_path = None
        if row.get("arxiv_id"):
            local_pdf = Path("data/temp_scv") / row["arxiv_id"] / "paper.pdf"
            if local_pdf.exists():
                pdf_path = str(local_pdf)
        records.append(
            PaperBankRecord(
                paper_id=paper_id,
                arxiv_id=row.get("arxiv_id"),
                title=row.get("title", ""),
                url=url,
                pdf_path=pdf_path,
                source="query_seed",
                year=year,
                authors=authors,
                status="query_seed",
                is_query_seed=True,
                query_source_path=source_path,
            )
        )
    return records


def initialize_query_bank(query_jsonl_path: str) -> List[PaperBankRecord]:
    query_rows = load_processed_query_papers(query_jsonl_path)
    records = build_query_paper_bank_records(query_rows, query_jsonl_path)
    return upsert_paper_bank(records)


def extract_previous_work_candidates(
    query_jsonl_path: str,
    paper_ids: Optional[Sequence[str]] = None,
    model_name: str = "gpt-5.4",
    backend: Optional[str] = None,
    backend_params: Optional[Dict] = None,
) -> List[PreviousWorkCandidateRecord]:
    query_rows = load_processed_query_papers(query_jsonl_path)
    initialize_query_bank(query_jsonl_path)
    filtered_rows = []
    for row in query_rows:
        paper_id = canonical_paper_id(row.get("arxiv_id"), row.get("title", ""))
        if paper_ids and paper_id not in paper_ids and row.get("arxiv_id") not in paper_ids:
            continue
        filtered_rows.append(row)

    extractor = _build_previous_work_extractor(model_name, backend, backend_params)
    to_extract = []
    metadata = []
    contexts: List[QueryPaperContext] = []
    for row in filtered_rows:
        datasets = []
        for dataset in row.get("datasets", []):
            info = dataset.get("info", dataset)
            if info.get("is_introduced"):
                datasets.append(info)
        context = prepare_query_paper_context(row)
        to_extract.append(
            {
                "title": row.get("metadata", {}).get("title", row.get("title", "")),
                "abstract": row.get("abstract", ""),
                "datasets": datasets,
                "text": context.text,
                "references": _format_reference_list(context.references),
                "source_type": context.source_type,
            }
        )
        metadata.append(row)
        contexts.append(context)

    if not to_extract:
        return []

    raw_results = extractor(to_extract)
    candidate_rows: List[PreviousWorkCandidateRecord] = []
    for row, context, extraction in zip(metadata, contexts, raw_results.dataset):
        paper_id = canonical_paper_id(row.get("arxiv_id"), row.get("title", ""))
        candidates = extraction["candidates"] if isinstance(extraction, dict) else extraction.candidates
        kept_index = 0
        for candidate in candidates:
            if isinstance(candidate, dict):
                candidate = PreviousWorkCandidate(**candidate)
            if not _should_keep_candidate(candidate):
                continue
            reference = _find_reference_for_candidate(candidate, context.references)
            candidate_rows.append(
                PreviousWorkCandidateRecord(
                    query_paper_id=paper_id,
                    candidate_id=candidate_id(paper_id, kept_index),
                    dataset_name=candidate.dataset_name,
                    paper_title=candidate.paper_title,
                    citation_key=candidate.citation_key,
                    authors=candidate.authors,
                    year=candidate.year,
                    citation_marker=candidate.citation_marker,
                    reference_title=reference.title if reference else None,
                    reference_authors=reference.authors if reference else [],
                    reference_year=reference.year if reference else None,
                    reference_url=reference.url if reference else None,
                    reference_doi=reference.doi if reference else None,
                    reference_arxiv_id=reference.arxiv_id if reference else None,
                    resolution_source="reference_arxiv" if reference and reference.arxiv_id else "reference_title" if reference and reference.title else None,
                    relationship_type=candidate.relationship_type if candidate.relationship_type in {
                        "closest_prior_dataset",
                        "source_dataset",
                        "parallel_benchmark",
                        "evaluation_baseline",
                        "loosely_related",
                        "unknown",
                    } else "unknown",
                    description=candidate.description,
                    confidence=candidate.confidence.lower(),
                    evidence_text=candidate.evidence_text,
                )
            )
            kept_index += 1
    query_ids = {row.query_paper_id for row in candidate_rows}
    return replace_previous_work_candidates_for_queries(query_ids, candidate_rows) if query_ids else []


def _title_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    left_norm = slugify(left).replace("-", " ")
    right_norm = slugify(right).replace("-", " ")
    if left_norm == right_norm:
        return 100.0
    if fuzz is not None:
        return float(fuzz.token_sort_ratio(left_norm, right_norm))
    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return 100.0 * len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def search_arxiv_by_title(title: str, max_results: int = 3) -> List[Dict]:
    if not title:
        return []
    params = {
        "search_query": f'ti:"{title}"',
        "start": 0,
        "max_results": max_results,
    }
    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=20)
        response.raise_for_status()
    except Exception:
        return []

    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    hits = []
    for entry in root.findall("atom:entry", namespace):
        entry_id = entry.findtext("atom:id", default="", namespaces=namespace)
        title_text = entry.findtext("atom:title", default="", namespaces=namespace).strip()
        published = entry.findtext("atom:published", default="", namespaces=namespace)
        authors = [author.findtext("atom:name", default="", namespaces=namespace) for author in entry.findall("atom:author", namespace)]
        arxiv_id = entry_id.rsplit("/", 1)[-1]
        hits.append(
            {
                "arxiv_id": arxiv_id,
                "title": title_text,
                "url": entry_id,
                "authors": authors,
                "published": published,
            }
        )
    return hits


def resolve_previous_work_candidates(candidate_ids: Optional[Sequence[str]] = None) -> List[PreviousWorkCandidateRecord]:
    candidates = load_previous_work_candidates()
    paper_bank = load_paper_bank()
    title_index = {paper.title.lower(): paper for paper in paper_bank if paper.title}
    updated: List[PreviousWorkCandidateRecord] = []
    new_papers: List[PaperBankRecord] = []

    for candidate in candidates:
        if candidate_ids and candidate.candidate_id not in candidate_ids:
            updated.append(candidate)
            continue
        if candidate.resolution_status not in {"needs_resolution", "ambiguous", "unresolved"}:
            updated.append(candidate)
            continue

        matched_paper: Optional[PaperBankRecord] = None
        resolution_candidates: List[str] = []
        title_for_resolution = candidate.reference_title or candidate.paper_title

        if candidate.reference_arxiv_id:
            paper_id = canonical_paper_id(candidate.reference_arxiv_id, candidate.reference_title or candidate.paper_title or candidate.dataset_name or "")
            existing_paper = next((paper for paper in paper_bank if paper.paper_id == paper_id), None)
            if existing_paper:
                matched_paper = existing_paper
            else:
                matched_paper = PaperBankRecord(
                    paper_id=paper_id,
                    arxiv_id=candidate.reference_arxiv_id,
                    title=candidate.reference_title or candidate.paper_title or candidate.dataset_name or candidate.reference_arxiv_id,
                    url=candidate.reference_url or f"https://arxiv.org/abs/{candidate.reference_arxiv_id}",
                    source="reference_arxiv",
                    year=candidate.reference_year or candidate.year,
                    authors=candidate.reference_authors or candidate.authors,
                    status="resolved_metadata",
                )
                new_papers.append(matched_paper)
            candidate.resolution_status = "resolved_arxiv"
            candidate.resolved_paper_id = matched_paper.paper_id
            candidate.resolved_arxiv_id = matched_paper.arxiv_id
            candidate.resolved_url = matched_paper.url
            candidate.resolution_source = "reference_arxiv"
            updated.append(candidate)
            continue

        if title_for_resolution:
            exact = title_index.get(title_for_resolution.lower())
            if exact:
                matched_paper = exact
                candidate.resolution_status = "resolved_in_db"
                candidate.resolved_paper_id = exact.paper_id
                candidate.resolved_arxiv_id = exact.arxiv_id
                candidate.resolved_url = exact.url
                candidate.resolution_source = "reference_title" if candidate.reference_title else "paper_title"
            else:
                best_score = 0.0
                best_paper = None
                for paper in paper_bank:
                    score = _title_similarity(title_for_resolution, paper.title)
                    if score >= 92 and score > best_score:
                        best_score = score
                        best_paper = paper
                if best_paper:
                    matched_paper = best_paper
                    candidate.resolution_status = "resolved_in_db"
                    candidate.resolved_paper_id = best_paper.paper_id
                    candidate.resolved_arxiv_id = best_paper.arxiv_id
                    candidate.resolved_url = best_paper.url
                    candidate.resolution_source = "reference_title" if candidate.reference_title else "paper_title"

        if not matched_paper and title_for_resolution:
            arxiv_hits = search_arxiv_by_title(title_for_resolution)
            for hit in arxiv_hits:
                resolution_candidates.append(f"{hit['arxiv_id']}::{hit['title']}")
            if len(arxiv_hits) == 1 and _title_similarity(title_for_resolution, arxiv_hits[0]["title"]) >= 85:
                hit = arxiv_hits[0]
                paper_id = canonical_paper_id(hit["arxiv_id"], hit["title"])
                paper = PaperBankRecord(
                    paper_id=paper_id,
                    arxiv_id=hit["arxiv_id"],
                    title=hit["title"],
                    url=hit["url"],
                    source="arxiv_search",
                    year=int(hit["published"][:4]) if hit["published"] else None,
                    authors=hit["authors"],
                    status="resolved_metadata",
                )
                new_papers.append(paper)
                candidate.resolution_status = "resolved_arxiv"
                candidate.resolved_paper_id = paper_id
                candidate.resolved_arxiv_id = hit["arxiv_id"]
                candidate.resolved_url = hit["url"]
                candidate.resolution_source = "arxiv_search"
            elif len(arxiv_hits) > 1:
                candidate.resolution_status = "ambiguous"
                candidate.resolution_candidates = resolution_candidates
                candidate.resolution_source = "unresolved"
            else:
                candidate.resolution_status = "unresolved"
                candidate.resolution_candidates = resolution_candidates
                candidate.resolution_source = "unresolved"
        elif not matched_paper and not title_for_resolution:
            candidate.resolution_status = "unresolved"
            candidate.resolution_source = "unresolved"

        updated.append(candidate)

    if new_papers:
        upsert_paper_bank(new_papers)
    return upsert_previous_work_candidates(updated)


def fetch_resolved_prior_papers(
    statuses: Sequence[str] = ("resolved_arxiv",),
    paper_ids: Optional[Sequence[str]] = None,
) -> List[PaperBankRecord]:
    ensure_benchmark_dirs()
    papers = load_paper_bank()
    updated = []
    for paper in papers:
        if paper_ids and paper.paper_id not in paper_ids:
            updated.append(paper)
            continue
        if paper.status not in {"resolved_metadata", "fetch_failed"} and paper.status not in statuses:
            updated.append(paper)
            continue
        if not paper.arxiv_id:
            updated.append(paper)
            continue
        pdf_dir = DEFAULT_PDF_DIR / paper.paper_id
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / "paper.pdf"
        if not pdf_path.exists():
            ok = download_file(ARXIV_PDF_URL.format(arxiv_id=paper.arxiv_id), str(pdf_path))
            if not ok:
                paper.status = "fetch_failed"
                updated.append(paper)
                continue
        paper.pdf_path = str(pdf_path)
        paper.status = "fetched"
        updated.append(paper)
    return upsert_paper_bank(updated)


def process_prior_papers(
    model_name: str = "gpt-5.4",
    backend: Optional[str] = None,
    backend_params: Optional[Dict] = None,
    paper_ids: Optional[Sequence[str]] = None,
) -> List[ProcessedBankRecord]:
    papers = load_paper_bank()
    processed_bank = {row.paper_id: row for row in load_processed_bank()}
    from .extraction import ScvExtractor

    extractor = ScvExtractor(model_name=model_name, backend=backend, backend_params=backend_params)
    new_processed_rows: List[ProcessedBankRecord] = []
    updated_papers: List[PaperBankRecord] = []

    for paper in papers:
        if paper_ids and paper.paper_id not in paper_ids:
            updated_papers.append(paper)
            continue
        if paper.paper_id in processed_bank:
            updated_papers.append(paper)
            continue
        if paper.status != "fetched" or not paper.pdf_path or not os.path.exists(paper.pdf_path):
            updated_papers.append(paper)
            continue
        text = get_text_from_pdf(paper.pdf_path)
        if not text:
            paper.status = "processing_failed"
            updated_papers.append(paper)
            new_processed_rows.append(
                ProcessedBankRecord(
                    paper_id=paper.paper_id,
                    processed_json_path="",
                    model_name=model_name,
                    prompt_version="prior-work-v1",
                    processed_at=now_iso(),
                    processing_status="processing_failed",
                    error_message="Failed to extract text from PDF",
                )
            )
            continue
        try:
            result = extractor([{
                "title": paper.title,
                "abstract": "",
                "text": text,
            }])
            analysis = result.dataset[0]
            analysis_obj = ScvPaperAnalysis(**analysis) if isinstance(analysis, dict) else analysis
            output_path = DEFAULT_PROCESSED_DIR / f"{slugify(paper.paper_id)}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "paper_id": paper.paper_id,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": [author.model_dump() for author in analysis_obj.authors],
                "is_nlp_paper": analysis_obj.is_nlp_paper,
                "nlp_relevance_explanation": analysis_obj.nlp_relevance_explanation,
                "contribution_summary": analysis_obj.paper_contribution_summary,
                "datasets": [dataset.model_dump() for dataset in analysis_obj.datasets],
            }
            with open(output_path, "w") as handle:
                json.dump(payload, handle)
            new_processed_rows.append(
                ProcessedBankRecord(
                    paper_id=paper.paper_id,
                    processed_json_path=str(output_path),
                    model_name=model_name,
                    prompt_version="prior-work-v1",
                    processed_at=now_iso(),
                    processing_status="processed",
                )
            )
            paper.status = "processed"
        except Exception as exc:
            new_processed_rows.append(
                ProcessedBankRecord(
                    paper_id=paper.paper_id,
                    processed_json_path="",
                    model_name=model_name,
                    prompt_version="prior-work-v1",
                    processed_at=now_iso(),
                    processing_status="processing_failed",
                    error_message=str(exc),
                )
            )
            paper.status = "processing_failed"
        updated_papers.append(paper)

    if new_processed_rows:
        upsert_processed_bank(new_processed_rows)
    upsert_paper_bank(updated_papers)
    return load_processed_bank()


def _load_processed_payload(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def build_benchmark_drafts(query_jsonl_path: str, paper_ids: Optional[Sequence[str]] = None) -> List[BenchmarkDraftRecord]:
    query_rows = load_processed_query_papers(query_jsonl_path)
    candidates = load_previous_work_candidates()
    paper_bank = {row.paper_id: row for row in load_paper_bank()}
    processed_bank = {row.paper_id: row for row in load_processed_bank()}
    existing_drafts = {row.query_paper_id: row for row in load_benchmark_drafts()}
    drafts: List[BenchmarkDraftRecord] = []

    for row in query_rows:
        query_paper_id = canonical_paper_id(row.get("arxiv_id"), row.get("title", ""))
        if paper_ids and query_paper_id not in paper_ids and row.get("arxiv_id") not in paper_ids:
            if query_paper_id in existing_drafts:
                drafts.append(existing_drafts[query_paper_id])
            continue
        introduced = [dataset.get("info", dataset) for dataset in row.get("datasets", []) if dataset.get("info", dataset).get("is_introduced")]
        if not introduced:
            continue
        query_dataset = introduced[0]
        query_candidates = [candidate for candidate in candidates if candidate.query_paper_id == query_paper_id]
        sorted_candidates = sorted(
            query_candidates,
            key=lambda candidate: (
                0 if candidate.resolution_status in {"resolved_in_db", "resolved_arxiv"} else 1,
                0 if candidate.relationship_type == "closest_prior_dataset" else 1 if candidate.relationship_type == "source_dataset" else 2,
                0 if candidate.confidence == "high" else 1 if candidate.confidence == "medium" else 2,
                candidate.paper_title or candidate.dataset_name or candidate.candidate_id,
            ),
        )
        linked_prior_papers: List[PriorPaperSummary] = []
        for candidate in sorted_candidates:
            if not candidate.resolved_paper_id:
                continue
            paper = paper_bank.get(candidate.resolved_paper_id)
            processed = processed_bank.get(candidate.resolved_paper_id)
            datasets: List[str] = []
            acus: List[str] = []
            payload = _load_processed_payload(processed.processed_json_path) if processed and processed.processed_json_path else None
            if payload:
                for dataset in payload.get("datasets", []):
                    datasets.append(dataset.get("name", ""))
                    acus.extend(dataset.get("acus", []))
            linked_prior_papers.append(
                PriorPaperSummary(
                    paper_id=candidate.resolved_paper_id,
                    title=paper.title if paper else candidate.paper_title or candidate.dataset_name or candidate.candidate_id,
                    arxiv_id=paper.arxiv_id if paper else candidate.resolved_arxiv_id,
                    processed_json_path=processed.processed_json_path if processed else None,
                    dataset_names=[name for name in datasets if name],
                    acus=acus[:20],
                )
            )

        suggested_gold_prior_paper_ids = [
            candidate.resolved_paper_id
            for candidate in sorted_candidates
            if candidate.resolved_paper_id and candidate.relationship_type in {"closest_prior_dataset", "source_dataset"}
        ][:2]
        suggested_hard_negative_ids = [
            candidate.candidate_id
            for candidate in sorted_candidates
            if candidate.relationship_type in {"parallel_benchmark", "evaluation_baseline"}
        ][:2]
        suggested_soft_negative_ids = [
            candidate.candidate_id
            for candidate in sorted_candidates
            if candidate.candidate_id not in suggested_hard_negative_ids
            and candidate.relationship_type in {"loosely_related", "parallel_benchmark", "evaluation_baseline"}
        ][:2]

        status = summarize_query_status(query_paper_id)
        draft = BenchmarkDraftRecord(
            query_paper_id=query_paper_id,
            query_dataset_name=query_dataset.get("name", ""),
            query_acus=query_dataset.get("acus", []),
            candidate_ids=[candidate.candidate_id for candidate in query_candidates],
            gold_prior_paper_ids=existing_drafts.get(query_paper_id, BenchmarkDraftRecord(query_paper_id=query_paper_id, query_dataset_name=query_dataset.get("name", ""))).gold_prior_paper_ids if query_paper_id in existing_drafts else [],
            gold_prior_dataset_names=existing_drafts.get(query_paper_id, BenchmarkDraftRecord(query_paper_id=query_paper_id, query_dataset_name=query_dataset.get("name", ""))).gold_prior_dataset_names if query_paper_id in existing_drafts else [],
            hard_negative_ids=existing_drafts.get(query_paper_id, BenchmarkDraftRecord(query_paper_id=query_paper_id, query_dataset_name=query_dataset.get("name", ""))).hard_negative_ids if query_paper_id in existing_drafts else [],
            soft_negative_ids=existing_drafts.get(query_paper_id, BenchmarkDraftRecord(query_paper_id=query_paper_id, query_dataset_name=query_dataset.get("name", ""))).soft_negative_ids if query_paper_id in existing_drafts else [],
            suggested_gold_prior_paper_ids=suggested_gold_prior_paper_ids,
            suggested_hard_negative_ids=suggested_hard_negative_ids,
            suggested_soft_negative_ids=suggested_soft_negative_ids,
            gold_prior_support_acus=existing_drafts.get(query_paper_id, BenchmarkDraftRecord(query_paper_id=query_paper_id, query_dataset_name=query_dataset.get("name", ""))).gold_prior_support_acus if query_paper_id in existing_drafts else query_dataset.get("previous_work_acus", []),
            gold_added_information_label=existing_drafts.get(query_paper_id, BenchmarkDraftRecord(query_paper_id=query_paper_id, query_dataset_name=query_dataset.get("name", ""))).gold_added_information_label if query_paper_id in existing_drafts else None,
            annotation_notes=existing_drafts.get(query_paper_id, BenchmarkDraftRecord(query_paper_id=query_paper_id, query_dataset_name=query_dataset.get("name", ""))).annotation_notes if query_paper_id in existing_drafts else "",
            draft_status=status,
            linked_prior_papers=linked_prior_papers,
            canonical=True,
        )
        drafts.append(draft)

    query_ids = {row.query_paper_id for row in drafts}
    return replace_benchmark_drafts_for_queries(query_ids, drafts) if query_ids else []


def serialize_query_view(query_paper_id: str, query_jsonl_path: str) -> Optional[Dict]:
    query_rows = load_processed_query_papers(query_jsonl_path)
    paper_bank = {row.paper_id: row for row in load_paper_bank()}
    drafts = {row.query_paper_id: row for row in load_benchmark_drafts()}
    candidates = [row for row in load_previous_work_candidates() if row.query_paper_id == query_paper_id]
    for row in query_rows:
        paper_id = canonical_paper_id(row.get("arxiv_id"), row.get("title", ""))
        if paper_id != query_paper_id:
            continue
        introduced = [dataset.get("info", dataset) for dataset in row.get("datasets", []) if dataset.get("info", dataset).get("is_introduced")]
        return {
            "paper_id": query_paper_id,
            "title": row.get("metadata", {}).get("title", row.get("title", "")),
            "arxiv_id": row.get("arxiv_id"),
            "status": summarize_query_status(query_paper_id),
            "query_datasets": introduced,
            "paper_bank": paper_bank.get(query_paper_id).model_dump() if paper_bank.get(query_paper_id) else None,
            "candidates": [candidate.model_dump() for candidate in candidates],
            "draft": drafts.get(query_paper_id).model_dump() if drafts.get(query_paper_id) else None,
        }
    return None


def patch_candidate(candidate_id_value: str, patch: Dict) -> PreviousWorkCandidateRecord:
    return patch_record(DEFAULT_PREVIOUS_WORK_PATH, PreviousWorkCandidateRecord, "candidate_id", candidate_id_value, patch)


def patch_draft(query_paper_id: str, patch: Dict) -> BenchmarkDraftRecord:
    existing = {row.query_paper_id: row for row in load_benchmark_drafts()}
    current = existing.get(query_paper_id)
    if current:
        candidate_data = current.model_dump()
        candidate_data.update(patch)
        candidate = BenchmarkDraftRecord(**candidate_data)
        if candidate.draft_status == "complete" and not is_human_complete_draft(candidate):
            patch = {**patch, "draft_status": "needs_annotation"}
    return patch_record(DEFAULT_BENCHMARK_DRAFTS_PATH, BenchmarkDraftRecord, "query_paper_id", query_paper_id, patch)


def patch_paper(paper_id: str, patch: Dict) -> PaperBankRecord:
    return patch_record(DEFAULT_PAPER_BANK_PATH, PaperBankRecord, "paper_id", paper_id, patch)


def create_manual_paper_record(
    *,
    title: str,
    query_paper_id: str,
    year: Optional[int] = None,
    authors: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> PaperBankRecord:
    paper_id = canonical_paper_id(None, f"{query_paper_id}-{title}")
    existing = {paper.paper_id: paper for paper in load_paper_bank()}
    if paper_id in existing:
        return existing[paper_id]
    paper = PaperBankRecord(
        paper_id=paper_id,
        title=title,
        source="manual_upload",
        year=year,
        authors=authors or [],
        status="unresolved",
        notes=notes,
    )
    upsert_paper_bank([paper])
    return paper


def create_arxiv_paper_record(
    *,
    arxiv_id: str,
    title: str,
    year: Optional[int] = None,
    authors: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> PaperBankRecord:
    paper_id = canonical_paper_id(arxiv_id, title)
    existing = {paper.paper_id: paper for paper in load_paper_bank()}
    if paper_id in existing:
        return existing[paper_id]
    paper = PaperBankRecord(
        paper_id=paper_id,
        arxiv_id=arxiv_id,
        title=title or arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        source="manual_arxiv",
        year=year,
        authors=authors or [],
        status="resolved_metadata",
        notes=notes,
    )
    upsert_paper_bank([paper])
    return paper


def link_candidate_to_paper(candidate_id_value: str, paper_id: str) -> PreviousWorkCandidateRecord:
    paper_bank = {paper.paper_id: paper for paper in load_paper_bank()}
    paper = paper_bank[paper_id]
    patch = {
        "resolved_paper_id": paper_id,
        "resolved_arxiv_id": paper.arxiv_id,
        "resolved_url": paper.url,
        "resolution_status": "resolved_in_db" if paper.status != "resolved_metadata" else "resolved_arxiv",
        "resolution_source": "manual",
    }
    return patch_candidate(candidate_id_value, patch)


def attach_pdf_to_paper(paper_id: str, source_pdf_path: str) -> PaperBankRecord:
    paper_bank = {paper.paper_id: paper for paper in load_paper_bank()}
    paper = paper_bank[paper_id]
    target_dir = DEFAULT_PDF_DIR / paper_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "paper.pdf"
    with open(source_pdf_path, "rb") as src, open(target_path, "wb") as dst:
        dst.write(src.read())
    paper.pdf_path = str(target_path)
    paper.source = "manual_upload"
    paper.status = "fetched"
    upsert_paper_bank([paper])
    return paper
