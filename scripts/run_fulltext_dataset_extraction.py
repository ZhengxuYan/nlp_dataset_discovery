#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import re
import tarfile
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scv.fulltext_dataset_schema import (
    FullTextDatasetExtraction,
    parse_model_payload,
    quality_warnings_for_bank,
    validate_extraction_for_bank,
)


DEFAULT_CENSUS = "data/census/acl_gemini_flashlite_all.clean.jsonl"
DEFAULT_ACL_CORPUS = os.environ.get("ACL_ANTHOLOGY_CORPUS", "ACL-anthology-corpus")
DEFAULT_OUTPUT = "data/census/fulltext_dataset_extractions_pilot.jsonl"
PROMPT_VERSION = "fulltext_dataset_schema_v1"


SYSTEM_PROMPT = """You are extracting structured information from full-text NLP dataset papers.

Goal:
Build a dataset paper bank for studying NLP datasets introduced in ACL Anthology papers. Extract only information grounded in the provided paper text. If the paper does not say something, use null, [], "unclear", or "unknown"; do not guess.

Critical framing:
- Do not assign subjective novelty labels.
- Extract dataset-centric Atomic Content Units (ACUs): short, auditable claims about what the introduced dataset contributes.
- Each ACU must include a short evidence span copied or tightly paraphrased from the paper text and a section hint when possible.
- Prefer claims that can later be compared against prior dataset papers: task/domain, data/source, annotation/protocol, scale/coverage, evaluation/use, availability/quality, governance/ethics.

Return JSON only. Use exactly this top-level shape:
{
  "paper_id": "...",
  "acl_id": "...",
  "title": "...",
  "year": 2025,
  "venue_prefix": "...",
  "event": "...",
  "booktitle": "...",
  "abstract": "...",
  "anthology_url": "...",
  "pdf_url": "...",
  "paper_contribution_summary": "...",
  "institution_profile": {
    "authors": [{"name": "...", "affiliations": ["..."], "sector": "academic|industry|government|nonprofit|independent|mixed|unknown"}],
    "lead_author_sector": "academic|industry|government|nonprofit|independent|mixed|unknown",
    "paper_sector": "academic_only|industry_only|academic_industry_collab|government|nonprofit|other|unknown",
    "industry_orgs": [],
    "academic_orgs": [],
    "countries_or_regions": []
  },
  "datasets": [
    {
      "dataset_id": "",
      "dataset_identity": {
        "canonical_name": "...",
        "aliases": [],
        "acronym": null,
        "version": null,
        "is_new_dataset": true,
        "is_dataset_family": false,
        "parent_dataset_family": null
      },
      "role": "introduced_dataset|benchmark|training_data|evaluation_set|pretraining_corpus|instruction_tuning_data|preference_data|lexical_resource|knowledge_resource|shared_task_dataset|other",
      "resource_type": "dataset|corpus|benchmark|annotation_set|lexicon|knowledge_base|multimodal_resource|tool_output|other",
      "primary_use": "training|evaluation|analysis|pretraining|fine_tuning|instruction_tuning|rlhf_preference|benchmarking|other",
      "is_reusable_resource": true,
      "usage_description": "...",
      "coverage": {
        "tasks": [],
        "domains": [],
        "languages": [],
        "language_family_or_region": [],
        "modality": [],
        "genre": [],
        "unit_of_analysis": "sentence|document|dialogue|token|span|image_text_pair|audio_text_pair|other|unclear",
        "input_output_format": "...",
        "label_space": "..."
      },
      "construction": {
        "source_data_origin": "...",
        "source_datasets": [{"name": "...", "relationship": "derived_from|extended_from|translated_from|filtered_from|combined_with|sampled_from|inspired_by|unclear", "evidence": "..."}],
        "collection_method": "...",
        "transformation_types": ["collection|annotation|translation|filtering|synthetic_generation|augmentation|mixture|extraction|reformatting|other"],
        "annotation_protocol": "...",
        "annotator_type": "expert|crowdworker|student|author|LLM|mixed|unknown",
        "num_annotators": null,
        "quality_control": "...",
        "synthetic_generation": {"uses_llm": false, "model_names": [], "human_verification": "yes|no|partial|unclear"}
      },
      "scale": {
        "size_text": "...",
        "num_instances": null,
        "num_tokens": null,
        "num_documents": null,
        "num_dialogues": null,
        "num_images": null,
        "num_audio_hours": null,
        "num_languages": null,
        "num_domains": null,
        "splits": {}
      },
      "availability": {
        "release_status": "released|partially_released|promised|available_on_request|not_released|unclear",
        "artifacts": {
          "dataset_urls": [],
          "project_page_urls": [],
          "code_urls": [],
          "huggingface_ids": [],
          "github_repos": [],
          "zenodo_urls": [],
          "osf_urls": [],
          "kaggle_urls": [],
          "paperswithcode_urls": [],
          "other_urls": []
        },
        "license": "...",
        "access_restrictions": "open|gated|request_only|commercial_restricted|not_available|unclear",
        "documentation_type": "datasheet|data_statement|model_card|paper_only|website|none|unclear",
        "maintenance_status": "maintained|static|deprecated|unclear"
      },
      "governance": {
        "ethics_discussed": "yes|no|unclear",
        "pii_discussed": "yes|no|unclear",
        "consent_discussed": "yes|no|unclear",
        "copyright_discussed": "yes|no|unclear",
        "bias_or_fairness_discussed": "yes|no|unclear",
        "known_limitations": []
      },
      "evaluation": {
        "used_for_training": true,
        "used_for_evaluation": true,
        "benchmark_metrics": [],
        "baseline_models": [],
        "compared_datasets": [],
        "reported_improvement": "...",
        "human_evaluation": "yes|no|unclear",
        "ablation_or_data_study": "yes|no|unclear"
      },
      "added_information_summary": "...",
      "acus": [
        {
          "id": "q0",
          "text": "...",
          "type": "task/domain|data/source|annotation/protocol|scale/coverage|evaluation/use|availability/quality|governance/ethics|other",
          "importance": "low|medium|high",
          "evidence": "...",
          "section": "..."
        }
      ],
      "prior_dataset_mentions": [
        {
          "name": "...",
          "relationship_type": "source_dataset|closest_prior_dataset|baseline_benchmark|comparison_dataset|shared_task|loosely_related",
          "cited_paper_title": null,
          "cited_paper_id": null,
          "evidence": "...",
          "prior_support_acus": []
        }
      ],
      "confidence": "low|medium|high",
      "ambiguities": [],
      "missing_information": []
    }
  ],
  "extraction_quality": {
    "confidence": "low|medium|high",
    "text_source": "full_text|partial_text|abstract_only|unknown",
    "missing_sections": [],
    "ambiguous_dataset_identity": false,
    "possible_false_positive_dataset_paper": false,
    "needs_human_review": false,
    "notes": []
  }
}

ACU requirements:
- For each introduced dataset, extract 4-12 ACUs when the text supports them.
- ACUs must be atomic. Split compound claims.
- Do not include generic claims such as "the paper proposes a dataset" unless the dataset identity itself is important.
- Every ACU must have evidence.

Prior dataset mentions:
- Include source datasets, closest prior datasets, comparison benchmarks, and baseline datasets when the paper discusses them substantively.
- Do not include every dataset in an experiment table if it is not relevant to the introduced dataset's added information.
"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_jsonl(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


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
            candidate = cleaned[start : end + 1]
            decoder = json.JSONDecoder()
            try:
                payload, _ = decoder.raw_decode(candidate)
                return payload
            except json.JSONDecodeError:
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    payload, _ = decoder.raw_decode(repaired)
                    return payload
        raise


def ensure_openai_client():
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for OpenAI extraction.") from exc
    return OpenAI()


def ensure_gemini_client(*, timeout_ms: int | None = None):
    load_dotenv()
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise RuntimeError("The google-genai package is required for Gemini extraction.") from exc
    http_options = genai_types.HttpOptions(timeout=timeout_ms) if timeout_ms else None
    return genai.Client(http_options=http_options), genai_types


def census_positive_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("is_dataset_introducing") and row.get("datasets")]


def existing_ids(path: str | Path) -> set[str]:
    if not Path(path).exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        paper_id = str(row.get("paper_id") or "").strip()
        if not paper_id:
            continue
        ids.add(paper_id)
        if not paper_id.startswith(("ACL:", "PRIOR:")) and re.match(r"^\d{4}\.", paper_id):
            ids.add(f"ACL:{paper_id}")
    return ids


def acl_id_from_paper_id(paper_id: str) -> str:
    return paper_id.removeprefix("ACL:")


def tei_path_for(row: dict[str, Any], acl_corpus_dir: Path) -> Path:
    acl_id = acl_id_from_paper_id(str(row.get("paper_id") or ""))
    return acl_corpus_dir / "data" / "grobid_xml" / f"{acl_id}.tei.xml"


def pdf_path_for(row: dict[str, Any], acl_corpus_dir: Path) -> Path:
    acl_id = acl_id_from_paper_id(str(row.get("paper_id") or ""))
    return acl_corpus_dir / "data" / "pdfs" / f"{acl_id}.pdf"


def arxiv_safe_id(row: dict[str, Any]) -> str:
    paper_id = str(row.get("arxiv_id") or row.get("paper_id") or "")
    return paper_id.replace("/", "_")


def arxiv_source_path_for(row: dict[str, Any], acl_corpus_dir: Path) -> Path:
    return acl_corpus_dir / "data" / "arxiv_sources" / f"{arxiv_safe_id(row)}.src"


def arxiv_source_url_for(row: dict[str, Any]) -> str:
    arxiv_id = str(row.get("arxiv_id") or row.get("paper_id") or "")
    return f"https://arxiv.org/e-print/{arxiv_id}"


def pdf_url_for(row: dict[str, Any]) -> str:
    if row.get("pdf_url"):
        return str(row.get("pdf_url"))
    if row.get("arxiv_id") or re.match(r"^\d{4}\.\d+", str(row.get("paper_id") or "")):
        return f"https://arxiv.org/pdf/{row.get('arxiv_id') or row.get('paper_id')}"
    acl_id = acl_id_from_paper_id(str(row.get("paper_id") or ""))
    return f"https://aclanthology.org/{acl_id}.pdf"


def has_local_text(row: dict[str, Any], acl_corpus_dir: Path, source_types: set[str]) -> bool:
    return (
        ("grobid_xml" in source_types and tei_path_for(row, acl_corpus_dir).exists())
        or ("pdf_text" in source_types and pdf_path_for(row, acl_corpus_dir).exists())
        or ("arxiv_source_latex" in source_types and arxiv_source_path_for(row, acl_corpus_dir).exists())
        or ("downloaded_pdf_text" in source_types)
    )


def tei_to_text(path: Path) -> str:
    root = ET.parse(path).getroot()
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    chunks: list[str] = []

    title = root.findtext(".//tei:titleStmt/tei:title", namespaces=ns)
    if title:
        chunks.append(f"# Title\n{title.strip()}")

    abstract_chunks = [" ".join(elem.itertext()).strip() for elem in root.findall(".//tei:profileDesc/tei:abstract//tei:p", ns)]
    if abstract_chunks:
        chunks.append("# Abstract\n" + "\n".join(chunk for chunk in abstract_chunks if chunk))

    for div in root.findall(".//tei:text/tei:body//tei:div", ns):
        head = " ".join(div.findtext("tei:head", default="", namespaces=ns).split())
        paragraphs = []
        for paragraph in div.findall(".//tei:p", ns):
            text = " ".join(" ".join(paragraph.itertext()).split())
            if text:
                paragraphs.append(text)
        if paragraphs:
            section = head or "Unknown Section"
            chunks.append(f"# {section}\n" + "\n".join(paragraphs))

    return "\n\n".join(chunks)


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to extract local PDF text.") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def strip_latex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        output = []
        escaped = False
        for char in line:
            if char == "\\" and not escaped:
                escaped = True
                output.append(char)
                continue
            if char == "%" and not escaped:
                break
            output.append(char)
            escaped = False
        lines.append("".join(output))
    return "\n".join(lines)


def latex_to_plain_text(text: str) -> str:
    text = strip_latex_comments(text)
    text = re.sub(r"\\(begin|end)\{(figure|table|algorithm|equation|align|tikzpicture|lstlisting)[^}]*\}.*?\\end\{\2\}", " ", text, flags=re.S)
    text = re.sub(r"\\(?:documentclass|usepackage|newcommand|renewcommand|def|DeclareMathOperator)(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\bibliography\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\{document\}", " ", text)
    text = re.sub(r"\\(?:input|include)\{([^}]+)\}", r"\n# Included file: \1\n", text)
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^{}]*)\}", r"\n# \1\n", text)
    text = re.sub(r"\\(?:title|author)\{([^{}]*)\}", r"\n# \1\n", text)
    text = re.sub(r"\\(?:textbf|textit|emph|underline|texttt)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:cite|citep|citet|ref|label|url|href)(?:\[[^\]]*\])?\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}$]", " ", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def decode_source_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def is_probable_latex(text: str) -> bool:
    return "\\begin{document}" in text or "\\documentclass" in text or "\\section" in text


def tex_sort_key(item: tuple[str, bytes]) -> tuple[int, int, str]:
    name, data = item
    text = decode_source_bytes(data[:20000])
    main_score = 0
    if "\\begin{document}" in text:
        main_score -= 3
    if "\\documentclass" in text:
        main_score -= 2
    if Path(name).name.lower() in {"main.tex", "paper.tex", "ms.tex", "article.tex"}:
        main_score -= 1
    return (main_score, len(name.split("/")), name)


def tex_files_from_source_package(data: bytes) -> list[tuple[str, bytes]]:
    tex_files: list[tuple[str, bytes]] = []
    buffer = io.BytesIO(data)
    try:
        with tarfile.open(fileobj=buffer, mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if not name.lower().endswith((".tex", ".bbl")):
                    continue
                if any(part.startswith(".") for part in Path(name).parts):
                    continue
                extracted = archive.extractfile(member)
                if extracted is not None:
                    tex_files.append((name, extracted.read()))
            if tex_files:
                return sorted(tex_files, key=tex_sort_key)
    except tarfile.TarError:
        pass

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if name.lower().endswith((".tex", ".bbl")) and not name.endswith("/"):
                    tex_files.append((name, archive.read(name)))
            if tex_files:
                return sorted(tex_files, key=tex_sort_key)
    except zipfile.BadZipFile:
        pass

    for payload in (data,):
        try:
            payload = gzip.decompress(payload)
        except OSError:
            pass
        text = decode_source_bytes(payload)
        if is_probable_latex(text):
            return [("source.tex", payload)]

    return []


def arxiv_source_to_text(path: Path) -> str:
    tex_files = tex_files_from_source_package(path.read_bytes())
    if not tex_files:
        raise ValueError("arXiv source package contains no readable LaTeX files")
    chunks = []
    for name, data in tex_files[:30]:
        text = latex_to_plain_text(decode_source_bytes(data))
        if text:
            chunks.append(f"# Source file: {name}\n{text}")
    output = "\n\n".join(chunks)
    if len(output) < 500:
        raise ValueError("arXiv LaTeX source text is too short after cleaning")
    return output


def download_pdf(row: dict[str, Any], pdf_path: Path, timeout: int) -> None:
    if pdf_path.exists() and pdf_path.stat().st_size > 500:
        return
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pdf_path.with_suffix(".pdf.tmp")
    with requests.get(pdf_url_for(row), stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(pdf_path)


def download_arxiv_source(row: dict[str, Any], source_path: Path, timeout: int) -> None:
    if source_path.exists() and source_path.stat().st_size > 500:
        return
    source_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = source_path.with_suffix(".src.tmp")
    with requests.get(arxiv_source_url_for(row), stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(source_path)


def condense_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    head_chars = int(max_chars * 0.72)
    tail_chars = max_chars - head_chars
    return f"{text[:head_chars]}\n\n[... middle truncated ...]\n\n{text[-tail_chars:]}", True


def load_source_text(
    row: dict[str, Any],
    acl_corpus_dir: Path,
    max_chars: int,
    source_types: set[str],
    download_timeout: int,
    keep_pdfs: bool,
    keep_arxiv_sources: bool,
) -> tuple[dict[str, Any], str]:
    source_errors: list[str] = []

    if "arxiv_source_latex" in source_types:
        source_path = arxiv_source_path_for(row, acl_corpus_dir)
        try:
            download_arxiv_source(row, source_path, download_timeout)
            text = arxiv_source_to_text(source_path)
            condensed, truncated = condense_text(text, max_chars)
            return {
                "type": "arxiv_source_latex",
                "path": str(source_path) if keep_arxiv_sources else None,
                "source_url": arxiv_source_url_for(row),
                "char_count": len(text),
                "truncated": truncated,
                "source_deleted": not keep_arxiv_sources,
            }, condensed
        except Exception as exc:  # noqa: BLE001
            source_errors.append(f"arxiv_source_latex: {exc}")
            if source_types == {"arxiv_source_latex"}:
                raise
        finally:
            if not keep_arxiv_sources and source_path.exists():
                try:
                    source_path.unlink()
                except OSError:
                    pass

    tei_path = tei_path_for(row, acl_corpus_dir)
    if "grobid_xml" in source_types and tei_path.exists():
        text = tei_to_text(tei_path)
        condensed, truncated = condense_text(text, max_chars)
        return {
            "type": "grobid_xml",
            "path": str(tei_path),
            "char_count": len(text),
            "truncated": truncated,
        }, condensed

    pdf_path = pdf_path_for(row, acl_corpus_dir)
    if "pdf_text" in source_types and pdf_path.exists():
        text = pdf_to_text(pdf_path)
        condensed, truncated = condense_text(text, max_chars)
        return {
            "type": "pdf_text",
            "path": str(pdf_path),
            "char_count": len(text),
            "truncated": truncated,
        }, condensed

    if "downloaded_pdf_text" in source_types:
        download_pdf(row, pdf_path, download_timeout)
        try:
            text = pdf_to_text(pdf_path)
            condensed, truncated = condense_text(text, max_chars)
            return {
                "type": "downloaded_pdf_text",
                "path": str(pdf_path) if keep_pdfs else None,
                "pdf_url": pdf_url_for(row),
                "char_count": len(text),
                "truncated": truncated,
                "pdf_deleted": not keep_pdfs,
            }, condensed
        finally:
            if not keep_pdfs and pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError:
                    pass

    allowed = ", ".join(sorted(source_types))
    detail = "; ".join(source_errors)
    suffix = f" Details: {detail}" if detail else ""
    raise FileNotFoundError(f"No source text ({allowed}) for {row.get('paper_id')}.{suffix}")


def compact_metadata(row: dict[str, Any]) -> dict[str, Any]:
    paper_id = str(row.get("paper_id") or "")
    acl_id = acl_id_from_paper_id(paper_id)
    return {
        "paper_id": paper_id,
        "acl_id": acl_id,
        "title": row.get("title") or "",
        "year": row.get("year"),
        "venue_prefix": row.get("venue_prefix") or "",
        "event": row.get("event") or "",
        "booktitle": row.get("booktitle") or "",
        "abstract": row.get("abstract") or "",
        "anthology_url": row.get("anthology_url") or (f"https://aclanthology.org/{acl_id}/" if acl_id else ""),
        "pdf_url": pdf_url_for(row),
        "abstract_census_datasets": row.get("datasets") or [],
    }


def build_prompt(row: dict[str, Any], source_text: dict[str, Any], text: str) -> str:
    payload = {
        "metadata": compact_metadata(row),
        "source_text": source_text,
        "paper_text": text,
    }
    return SYSTEM_PROMPT + "\n\nPaper input:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_record(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    source_text: dict[str, Any],
    model: str,
) -> FullTextDatasetExtraction:
    metadata = compact_metadata(row)
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
        else:
            raise ValueError("Model returned a JSON list instead of a single paper object")
    if isinstance(payload, dict) and isinstance(payload.get("papers"), list):
        matches = [item for item in payload["papers"] if str(item.get("paper_id") or "") == metadata["paper_id"]]
        payload = matches[0] if matches else payload["papers"][0]
    if not isinstance(payload, dict):
        raise ValueError(f"Model returned {type(payload).__name__}, expected JSON object")
    payload = dict(payload)
    for key in [
        "paper_id",
        "acl_id",
        "title",
        "year",
        "venue_prefix",
        "event",
        "booktitle",
        "abstract",
        "anthology_url",
        "pdf_url",
    ]:
        payload[key] = metadata.get(key)
    payload["source_text"] = source_text
    payload["model"] = model
    payload["prompt_version"] = PROMPT_VERSION
    payload["extracted_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    record = parse_model_payload(payload)
    for dataset_index, dataset in enumerate(record.datasets):
        if not dataset.dataset_id:
            safe_name = re.sub(r"[^a-zA-Z0-9]+", "-", dataset.dataset_identity.canonical_name.lower()).strip("-")[:60]
            dataset.dataset_id = f"{record.paper_id}::dataset::{dataset_index}::{safe_name}"
        for acu_index, acu in enumerate(dataset.acus):
            if not acu.id:
                acu.id = f"q{acu_index}"
    return record


def extract_openai(client, prompt: str, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    response = client.responses.create(model=model.removeprefix("openai/"), input=prompt)
    usage = getattr(response, "usage", None)
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage_dict = usage.dict()
        else:
            usage_dict = {
                key: getattr(usage, key)
                for key in ["input_tokens", "output_tokens", "total_tokens"]
                if hasattr(usage, key)
            }
    return parse_json_object(response.output_text), usage_dict


def extract_gemini(client_bundle, prompt: str, model: str, *, max_output_tokens: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    client, genai_types = client_bundle
    response = client.models.generate_content(
        model=model.removeprefix("gemini/"),
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            max_output_tokens=max_output_tokens,
        ),
    )
    usage = getattr(response, "usage_metadata", None)
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage_dict = usage.dict()
        else:
            usage_dict = {
                key: getattr(usage, key)
                for key in [
                    "prompt_token_count",
                    "candidates_token_count",
                    "total_token_count",
                    "cached_content_token_count",
                ]
                if hasattr(usage, key)
            }
    return parse_json_object(response.text or ""), usage_dict


def record_to_dict(record: FullTextDatasetExtraction) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return record.model_dump()
    return record.dict()


def error_row(row: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "error": str(exc),
        "error_type": type(exc).__name__,
        "failed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def run_one(
    client_bundle,
    row: dict[str, Any],
    *,
    backend: str,
    model: str,
    acl_corpus_dir: Path,
    text_char_limit: int,
    source_types: set[str],
    download_timeout: int,
    keep_pdfs: bool,
    keep_arxiv_sources: bool,
    max_output_tokens: int | None,
    max_retries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    started = time.monotonic()
    try:
        source_text, text = load_source_text(
            row,
            acl_corpus_dir,
            text_char_limit,
            source_types,
            download_timeout,
            keep_pdfs,
            keep_arxiv_sources,
        )
    except Exception as exc:  # noqa: BLE001
        return None, error_row(row, exc)

    prompt = build_prompt(row, source_text, text)
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            payload, usage = (
                extract_openai(client_bundle, prompt, model)
                if backend == "openai"
                else extract_gemini(client_bundle, prompt, model, max_output_tokens=max_output_tokens)
            )
            record = normalize_record(row, payload, source_text=source_text, model=model)
            validation_errors = validate_extraction_for_bank(record)
            quality_warnings = quality_warnings_for_bank(record)
            output = record_to_dict(record)
            output["validation_errors"] = validation_errors
            output["quality_warnings"] = quality_warnings
            output["llm_usage"] = usage
            output["runtime_seconds"] = round(time.monotonic() - started, 3)
            return output, None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))
    assert last_exc is not None
    return None, error_row(row, last_exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract full-text dataset bank records from ACL dataset-introducing papers.")
    parser.add_argument("--census-jsonl", default=DEFAULT_CENSUS)
    parser.add_argument("--acl-corpus-dir", default=DEFAULT_ACL_CORPUS)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--error-jsonl", default=None)
    parser.add_argument("--backend", choices=["openai", "gemini"], default="gemini")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--text-char-limit", type=int, default=40000)
    parser.add_argument("--request-timeout-ms", type=int, default=120000)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--source-types",
        nargs="+",
        choices=["grobid_xml", "pdf_text", "arxiv_source_latex", "downloaded_pdf_text"],
        default=["grobid_xml", "pdf_text"],
    )
    parser.add_argument("--local-text-only", action="store_true")
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--keep-pdfs", action="store_true")
    parser.add_argument("--keep-arxiv-sources", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    error_path = args.error_jsonl or str(Path(args.output_jsonl).with_suffix(".errors.jsonl"))
    if args.overwrite:
        clear_jsonl(args.output_jsonl)
        clear_jsonl(error_path)

    rows = census_positive_rows(read_jsonl(args.census_jsonl))
    source_types = set(args.source_types)
    total_positive_rows = len(rows)
    if args.local_text_only:
        acl_corpus_dir = Path(args.acl_corpus_dir)
        rows = [row for row in rows if has_local_text(row, acl_corpus_dir, source_types)]
    local_text_rows = len(rows)
    if args.sample_mode == "random":
        rng = random.Random(args.sample_seed)
        rows = list(rows)
        rng.shuffle(rows)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not args.overwrite:
        done = existing_ids(args.output_jsonl)
        rows = [row for row in rows if str(row.get("paper_id") or "") not in done]

    status = {
        "census_jsonl": args.census_jsonl,
        "acl_corpus_dir": args.acl_corpus_dir,
        "output_jsonl": args.output_jsonl,
        "error_jsonl": error_path,
        "backend": args.backend,
        "model": args.model,
        "total_positive_rows": total_positive_rows,
        "eligible_rows_after_local_text_filter": local_text_rows,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "remaining_rows": len(rows),
        "workers": args.workers,
        "text_char_limit": args.text_char_limit,
        "request_timeout_ms": args.request_timeout_ms,
        "max_output_tokens": args.max_output_tokens,
        "source_types": args.source_types,
        "local_text_only": args.local_text_only,
        "download_timeout": args.download_timeout,
        "keep_pdfs": args.keep_pdfs,
        "keep_arxiv_sources": args.keep_arxiv_sources,
        "dry_run": args.dry_run,
    }
    print(json.dumps(status, indent=2))

    if args.dry_run:
        for row in rows[: min(3, len(rows))]:
            try:
                source_text, text = load_source_text(
                    row,
                    Path(args.acl_corpus_dir),
                    args.text_char_limit,
                    source_types,
                    args.download_timeout,
                    args.keep_pdfs,
                    args.keep_arxiv_sources,
                )
                print(json.dumps({
                    "metadata": compact_metadata(row),
                    "source_text": source_text,
                    "text_preview": text[:1200],
                }, ensure_ascii=False, indent=2))
            except Exception as exc:  # noqa: BLE001
                print(json.dumps(error_row(row, exc), ensure_ascii=False, indent=2))
        return

    client_bundle = ensure_openai_client() if args.backend == "openai" else ensure_gemini_client(timeout_ms=args.request_timeout_ms)
    processed = 0
    failed = 0
    total_runtime = 0.0
    total_tokens = 0

    if args.workers <= 1:
        for row in rows:
            output, error = run_one(
                client_bundle,
                row,
                backend=args.backend,
                model=args.model,
                acl_corpus_dir=Path(args.acl_corpus_dir),
                text_char_limit=args.text_char_limit,
                source_types=source_types,
                download_timeout=args.download_timeout,
                keep_pdfs=args.keep_pdfs,
                keep_arxiv_sources=args.keep_arxiv_sources,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
            )
            if output:
                append_jsonl(args.output_jsonl, [output])
                processed += 1
                total_runtime += float(output.get("runtime_seconds") or 0)
                total_tokens += int((output.get("llm_usage") or {}).get("total_token_count") or (output.get("llm_usage") or {}).get("total_tokens") or 0)
            if error:
                append_jsonl(error_path, [error])
                failed += 1
            print(json.dumps({
                "processed": processed,
                "failed": failed,
                "last_paper_id": row.get("paper_id"),
                "avg_runtime_seconds": round(total_runtime / processed, 3) if processed else None,
                "avg_total_tokens": round(total_tokens / processed, 1) if processed and total_tokens else None,
            }, indent=2))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_one,
                    client_bundle,
                    row,
                    backend=args.backend,
                    model=args.model,
                    acl_corpus_dir=Path(args.acl_corpus_dir),
                    text_char_limit=args.text_char_limit,
                    source_types=source_types,
                    download_timeout=args.download_timeout,
                    keep_pdfs=args.keep_pdfs,
                    keep_arxiv_sources=args.keep_arxiv_sources,
                    max_output_tokens=args.max_output_tokens,
                    max_retries=args.max_retries,
                )
                for row in rows
            ]
            for future in as_completed(futures):
                output, error = future.result()
                if output:
                    append_jsonl(args.output_jsonl, [output])
                    processed += 1
                    total_runtime += float(output.get("runtime_seconds") or 0)
                    total_tokens += int((output.get("llm_usage") or {}).get("total_token_count") or (output.get("llm_usage") or {}).get("total_tokens") or 0)
                if error:
                    append_jsonl(error_path, [error])
                    failed += 1
                print(json.dumps({
                    "processed": processed,
                    "failed": failed,
                    "avg_runtime_seconds": round(total_runtime / processed, 3) if processed else None,
                    "avg_total_tokens": round(total_tokens / processed, 1) if processed and total_tokens else None,
                }, indent=2))


if __name__ == "__main__":
    main()
