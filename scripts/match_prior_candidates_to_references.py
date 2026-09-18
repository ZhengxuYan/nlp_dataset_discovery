#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv


DEFAULT_ACL_CORPUS = os.environ.get("ACL_ANTHOLOGY_CORPUS", "ACL-anthology-corpus")

MATCH_LABELS = {"matched_reference", "no_bibliography_match", "uncertain"}


SYSTEM_PROMPT = """You are resolving prior-support dataset candidates against a query paper's bibliography.

Task:
For each candidate prior dataset/paper, find the most likely matching reference in the query paper's bibliography.

Use only the provided references. Do not use external knowledge except to recognize dataset names and title variants.

Labels:
- matched_reference: a specific bibliography reference clearly corresponds to this candidate.
- no_bibliography_match: no listed reference corresponds to this candidate.
- uncertain: a possible match exists but evidence is insufficient.

Rules:
- Prefer exact or near-exact title matches.
- Dataset-name-only candidates can match a paper if the reference title strongly indicates that dataset.
- If the candidate is a direct source dataset mentioned in query evidence but no bibliography entry is present, use no_bibliography_match.
- Do not match vague families such as "multilingual OCR benchmarks" unless a concrete reference is clearly named.
- Return one judgment per candidate_index.

Return JSON only:
{
  "matches": [
    {
      "candidate_index": 0,
      "match_label": "matched_reference|no_bibliography_match|uncertain",
      "reference_id": "r12",
      "matched_title": "...",
      "matched_authors": ["..."],
      "matched_year": 2023,
      "matched_url": "...",
      "matched_doi": "...",
      "matched_arxiv_id": "...",
      "confidence": "low|medium|high",
      "rationale": "..."
    }
  ],
  "summary": "..."
}
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
                payload, _ = decoder.raw_decode(repaired)
                return payload
        raise


def ensure_openai_client():
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for reference matching.") from exc
    return OpenAI()


def existing_ids(path: str | Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {str(row.get("benchmark_id") or "") for row in read_jsonl(path)}


def acl_id_from_row(row: dict[str, Any]) -> str:
    return str(row.get("query_acl_id") or row.get("query_paper_id") or "").removeprefix("ACL:")


def tei_path_for(row: dict[str, Any], acl_corpus_dir: Path) -> Path:
    return acl_corpus_dir / "data" / "grobid_xml" / f"{acl_id_from_row(row)}.tei.xml"


def pdf_path_for(row: dict[str, Any], acl_corpus_dir: Path) -> Path:
    return acl_corpus_dir / "data" / "pdfs" / f"{acl_id_from_row(row)}.pdf"


def pdf_url_for(row: dict[str, Any]) -> str:
    return f"https://aclanthology.org/{acl_id_from_row(row)}.pdf"


def extract_arxiv_id(text: str) -> str:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", text, re.I)
    return match.group(1) if match else ""


def clean_space(text: str) -> str:
    return " ".join((text or "").split())


def parse_tei_references(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    references = []
    for idx, bibl in enumerate(root.findall(".//tei:listBibl/tei:biblStruct", ns)):
        title = clean_space(" ".join(bibl.findtext(".//tei:title[@level='a']", default="", namespaces=ns).split()))
        if not title:
            title = clean_space(" ".join(bibl.findtext(".//tei:title", default="", namespaces=ns).split()))
        authors = []
        for author in bibl.findall(".//tei:analytic/tei:author", ns) or bibl.findall(".//tei:monogr/tei:author", ns):
            name = clean_space(" ".join(author.itertext()))
            if name:
                authors.append(name)
        year = ""
        date = bibl.find(".//tei:date", ns)
        if date is not None:
            year = date.attrib.get("when", "")[:4] or clean_space(" ".join(date.itertext()))[:4]
        raw = clean_space(" ".join(bibl.itertext()))
        urls = []
        for ref in bibl.findall(".//tei:ref", ns):
            target = ref.attrib.get("target")
            if target:
                urls.append(target)
        doi = ""
        for ident in bibl.findall(".//tei:idno", ns):
            if ident.attrib.get("type", "").lower() == "doi":
                doi = clean_space(" ".join(ident.itertext()))
        references.append({
            "reference_id": f"r{idx}",
            "title": title,
            "authors": authors[:8],
            "year": year,
            "url": urls[0] if urls else "",
            "doi": doi,
            "arxiv_id": extract_arxiv_id(raw + " " + " ".join(urls)),
            "raw_reference": raw[:1200],
            "source": "grobid_tei",
        })
    return [ref for ref in references if ref["title"] or ref["raw_reference"]]


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required to extract local PDF text.") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


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


def split_pdf_references(text: str, max_references: int) -> list[dict[str, Any]]:
    match = re.search(r"(?im)^\s*(references|bibliography)\s*$", text)
    if match:
        ref_text = text[match.end() :]
    else:
        lowered = text.lower()
        idx = max(lowered.rfind("\nreferences"), lowered.rfind("\nbibliography"))
        ref_text = text[idx:] if idx >= 0 else text[-20000:]
    ref_text = re.split(r"(?im)^\s*(appendix|supplementary|acknowledg(e)?ments)\s*$", ref_text)[0]
    lines = [clean_space(line) for line in ref_text.splitlines()]
    lines = [line for line in lines if line]
    entries: list[str] = []
    current: list[str] = []
    start_pat = re.compile(r"^(\[\d+\]|\d+\.|[A-Z][A-Za-z'’-]+,\s+[A-Z]|[A-Z][a-z]+ et al\.)")
    for line in lines:
        if current and start_pat.search(line) and len(" ".join(current)) > 80:
            entries.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(" ".join(current))
    if len(entries) < 5:
        entries = re.split(r"(?<=[.!?])\s+(?=[A-Z][A-Za-z'’-]+,\s+[A-Z])", " ".join(lines))
    refs = []
    for idx, raw in enumerate(entries[:max_references]):
        raw = clean_space(raw)
        if len(raw) < 30:
            continue
        year_match = re.search(r"\b(19|20)\d{2}\b", raw)
        refs.append({
            "reference_id": f"r{idx}",
            "title": "",
            "authors": [],
            "year": year_match.group(0) if year_match else "",
            "url": "",
            "doi": "",
            "arxiv_id": extract_arxiv_id(raw),
            "raw_reference": raw[:1200],
            "source": "pdf_reference_section",
        })
    return refs


def load_references(
    row: dict[str, Any],
    *,
    acl_corpus_dir: Path,
    download_timeout: int,
    keep_pdfs: bool,
    max_references: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tei_path = tei_path_for(row, acl_corpus_dir)
    if tei_path.exists():
        refs = parse_tei_references(tei_path)
        if refs:
            return refs[:max_references], {"reference_source": "grobid_tei", "reference_count": len(refs), "tei_path": str(tei_path)}

    pdf_path = pdf_path_for(row, acl_corpus_dir)
    download_pdf(row, pdf_path, download_timeout)
    try:
        text = pdf_to_text(pdf_path)
        refs = split_pdf_references(text, max_references=max_references)
        return refs, {
            "reference_source": "downloaded_pdf_text",
            "reference_count": len(refs),
            "pdf_url": pdf_url_for(row),
            "pdf_deleted": not keep_pdfs,
        }
    finally:
        if not keep_pdfs and pdf_path.exists():
            try:
                pdf_path.unlink()
            except OSError:
                pass


def selected_candidates(row: dict[str, Any], include_uncertain: bool) -> list[dict[str, Any]]:
    selected = []
    for candidate in row.get("llm_candidate_priors_judged") or []:
        judgment = candidate.get("second_pass_judgment") or {}
        label = judgment.get("support_label")
        if label == "gold_prior_support" or (include_uncertain and label == "uncertain"):
            item = {
                "candidate_index": candidate.get("candidate_index"),
                "dataset_name": candidate.get("dataset_name"),
                "paper_title": candidate.get("paper_title"),
                "relationship_type": candidate.get("relationship_type"),
                "evidence_from_query_paper": candidate.get("evidence_from_query_paper"),
                "search_query": candidate.get("search_query"),
                "second_pass_label": label,
                "second_pass_confidence": judgment.get("confidence"),
                "second_pass_rationale": judgment.get("rationale"),
            }
            selected.append(item)
    return selected


def build_prompt(row: dict[str, Any], candidates: list[dict[str, Any]], references: list[dict[str, Any]]) -> str:
    payload = {
        "query": {
            "benchmark_id": row.get("benchmark_id"),
            "query_paper_id": row.get("query_paper_id"),
            "query_title": row.get("query_title"),
            "query_year": row.get("query_year"),
            "query_dataset_name": row.get("query_dataset_name"),
            "query_acus": row.get("query_acus") or [],
        },
        "candidate_priors": candidates,
        "references": references,
    }
    return SYSTEM_PROMPT + "\n\nInput:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_matches(candidates: list[dict[str, Any]], references: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    refs_by_id = {ref["reference_id"]: ref for ref in references}
    by_index = {}
    for item in payload.get("matches") or []:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("candidate_index"))
        except (TypeError, ValueError):
            continue
        label = item.get("match_label") or "uncertain"
        if label not in MATCH_LABELS:
            label = "uncertain"
        ref_id = str(item.get("reference_id") or "")
        ref = refs_by_id.get(ref_id, {})
        by_index[idx] = {
            "match_label": label,
            "reference_id": ref_id if label == "matched_reference" else "",
            "matched_title": item.get("matched_title") or ref.get("title") or "",
            "matched_authors": item.get("matched_authors") if isinstance(item.get("matched_authors"), list) else ref.get("authors", []),
            "matched_year": item.get("matched_year") or ref.get("year") or "",
            "matched_url": item.get("matched_url") or ref.get("url") or "",
            "matched_doi": item.get("matched_doi") or ref.get("doi") or "",
            "matched_arxiv_id": item.get("matched_arxiv_id") or ref.get("arxiv_id") or "",
            "matched_raw_reference": ref.get("raw_reference", ""),
            "confidence": item.get("confidence") or "medium",
            "rationale": item.get("rationale") or "",
        }
    matched = []
    for candidate in candidates:
        idx = int(candidate.get("candidate_index") or 0)
        item = dict(candidate)
        item["reference_match"] = by_index.get(idx) or {
            "match_label": "uncertain",
            "reference_id": "",
            "matched_title": "",
            "matched_authors": [],
            "matched_year": "",
            "matched_url": "",
            "matched_doi": "",
            "matched_arxiv_id": "",
            "matched_raw_reference": "",
            "confidence": "low",
            "rationale": "No valid match returned.",
        }
        matched.append(item)
    return matched


def normalize_for_match(text: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def title_similarity(left: str, right: str) -> float:
    left_norm = normalize_for_match(left)
    right_norm = normalize_for_match(right)
    if not left_norm or not right_norm:
        return 0.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def dataset_name_in_reference(dataset_name: str, matched_title: str, raw_reference: str) -> bool:
    name = normalize_for_match(dataset_name)
    if not name:
        return False
    haystack = normalize_for_match(f"{matched_title} {raw_reference}")
    if name in haystack:
        return True
    compact_name = name.replace(" ", "")
    compact_haystack = haystack.replace(" ", "")
    return len(compact_name) >= 4 and compact_name in compact_haystack


def rationale_disclaims_match(rationale: str) -> bool:
    normalized = normalize_for_match(rationale)
    disclaimer_patterns = [
        "no bibliography entry",
        "does not match",
        "not clearly correspond",
        "not clearly corresponds",
        "does not clearly correspond",
        "no listed reference matches",
        "no reference clearly corresponds",
        "closest listed",
        "closest available",
    ]
    return any(pattern in normalized for pattern in disclaimer_patterns)


def apply_match_sanity_checks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade bibliography matches that do not actually identify the candidate."""
    checked = []
    for candidate in candidates:
        item = dict(candidate)
        match = dict(item.get("reference_match") or {})
        if match.get("match_label") != "matched_reference":
            checked.append(item)
            continue

        candidate_title = str(item.get("paper_title") or "")
        dataset_name = str(item.get("dataset_name") or "")
        matched_title = str(match.get("matched_title") or "")
        raw_reference = str(match.get("matched_raw_reference") or "")
        confidence = str(match.get("confidence") or "").lower()
        rationale = str(match.get("rationale") or "")
        title_ok = bool(candidate_title) and title_similarity(candidate_title, matched_title) >= 0.55
        name_ok = dataset_name_in_reference(dataset_name, matched_title, raw_reference)
        weak_or_self_disclaimed = confidence == "low" or rationale_disclaims_match(rationale)

        if weak_or_self_disclaimed or not (title_ok or name_ok):
            previous = match.get("match_label")
            match["match_label"] = "uncertain"
            match["confidence"] = "low"
            match["rationale"] = (
                rationale
                + f" Sanity check downgraded this from {previous}: the matched reference title/raw text does not clearly contain the candidate dataset name or closely match the candidate paper title."
            ).strip()
        item["reference_match"] = match
        checked.append(item)
    return checked


def match_row(client, row: dict[str, Any], *, model: str, references: list[dict[str, Any]], include_uncertain: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = selected_candidates(row, include_uncertain)
    if not candidates:
        output = dict(row)
        output["reference_matched_candidates"] = []
        output["reference_match_summary"] = "No selected candidates."
        output["reference_match_usage"] = {}
        return output, {}
    response = client.responses.create(model=model.removeprefix("openai/"), input=build_prompt(row, candidates, references))
    payload = parse_json_object(response.output_text)
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage.dict() if hasattr(usage, "dict") else {})
    output = dict(row)
    output["reference_matched_candidates"] = apply_match_sanity_checks(normalize_matches(candidates, references, payload))
    output["reference_match_summary"] = payload.get("summary") or ""
    output["reference_match_model"] = model
    output["reference_match_usage"] = usage_dict
    output["reference_match_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return output, usage_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Match judged prior-support candidates to query-paper bibliography references using an LLM.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="data/benchmark/acl_prior_support_reference_matches.jsonl")
    parser.add_argument("--error-jsonl", default=None)
    parser.add_argument("--acl-corpus-dir", default=DEFAULT_ACL_CORPUS)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--max-references", type=int, default=80)
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--keep-pdfs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    error_path = args.error_jsonl or str(Path(args.output_jsonl).with_suffix(".errors.jsonl"))
    if args.overwrite:
        clear_jsonl(args.output_jsonl)
        clear_jsonl(error_path)

    rows = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not args.overwrite:
        done = existing_ids(args.output_jsonl)
        rows = [row for row in rows if str(row.get("benchmark_id") or "") not in done]

    print(json.dumps({
        "input_jsonl": args.input_jsonl,
        "output_jsonl": args.output_jsonl,
        "remaining_rows": len(rows),
        "model": args.model,
        "include_uncertain": args.include_uncertain,
        "max_references": args.max_references,
        "dry_run": args.dry_run,
    }, indent=2))

    acl_corpus_dir = Path(args.acl_corpus_dir)
    if args.dry_run:
        for row in rows[:2]:
            refs, meta = load_references(row, acl_corpus_dir=acl_corpus_dir, download_timeout=args.download_timeout, keep_pdfs=args.keep_pdfs, max_references=args.max_references)
            print(json.dumps({
                "benchmark_id": row.get("benchmark_id"),
                "reference_meta": meta,
                "selected_candidates": selected_candidates(row, args.include_uncertain),
                "references_preview": refs[:5],
            }, ensure_ascii=False, indent=2)[:8000])
        return

    client = ensure_openai_client()
    processed = 0
    failed = 0
    total_input = 0
    total_output = 0
    for row in rows:
        started = time.monotonic()
        try:
            refs, meta = load_references(row, acl_corpus_dir=acl_corpus_dir, download_timeout=args.download_timeout, keep_pdfs=args.keep_pdfs, max_references=args.max_references)
            output, usage = match_row(client, row, model=args.model, references=refs, include_uncertain=args.include_uncertain)
            output["reference_extraction_meta"] = meta
            output["reference_match_runtime_seconds"] = round(time.monotonic() - started, 3)
            append_jsonl(args.output_jsonl, [output])
            processed += 1
            total_input += int(usage.get("input_tokens") or 0)
            total_output += int(usage.get("output_tokens") or 0)
        except Exception as exc:  # noqa: BLE001
            append_jsonl(error_path, [{
                "benchmark_id": row.get("benchmark_id"),
                "query_paper_id": row.get("query_paper_id"),
                "query_dataset_name": row.get("query_dataset_name"),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "failed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }])
            failed += 1
        print(json.dumps({
            "processed": processed,
            "failed": failed,
            "last": row.get("benchmark_id"),
            "avg_input_tokens": round(total_input / processed, 1) if processed else None,
            "avg_output_tokens": round(total_output / processed, 1) if processed else None,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
