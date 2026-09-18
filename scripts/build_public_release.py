#!/usr/bin/env python3
"""Build a compact, sanitized, deterministic public research snapshot."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release_dist" / "nlp-dataset-added-information"
BASE_RELEASE = ROOT / "publish" / "nlp-dataset-added-information"

SKIP_NAMES = {
    ".DS_Store",
    ".git",
    ".pytest_cache",
    "__pycache__",
}
TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}
SECRET_PATTERNS = {
    "absolute_user_path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"\b(?:github_pat_|ghp_)[A-Za-z0-9_]{20,}"),
}

EXPANSION_SCRIPTS = {
    "audit_enrichment_stable_ids.py",
    "audit_expansion_regression.py",
    "audit_metadata_schema.py",
    "build_expansion_completion_audit.py",
    "build_expansion_status_packet.py",
    "build_integrated_fulltext_banks.py",
    "build_metadata_review_sample.py",
    "build_public_release.py",
    "enrich_dataset_metadata.py",
    "finalize_paper_enrichment.py",
    "merge_arxiv_raw_catalogs.py",
    "prepare_arxiv_screening_catalog.py",
    "prepare_main_analysis_expansion.py",
    "run_continuous_paper_enrichment.py",
    "run_corpus_expansion_2020_2025.py",
    "summarize_metadata_coverage.py",
    "validate_expansion_readiness.py",
}

EXPANSION_TESTS = {
    "test_audit_enrichment_stable_ids.py",
    "test_audit_expansion_regression.py",
    "test_audit_metadata_schema.py",
    "test_build_expansion_completion_audit.py",
    "test_build_expansion_status_packet.py",
    "test_build_metadata_review_sample.py",
    "test_build_public_release.py",
    "test_enrich_dataset_metadata.py",
    "test_finalize_paper_enrichment.py",
    "test_prepare_main_analysis_expansion.py",
    "test_run_continuous_paper_enrichment.py",
    "test_run_corpus_expansion_2020_2025.py",
    "test_summarize_metadata_coverage.py",
    "test_validate_expansion_readiness.py",
}


def is_hydrated_file(path: Path) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    return stat.st_size == 0 or stat.st_blocks > 0


def copy_file(source: Path, target: Path) -> None:
    if not is_hydrated_file(source):
        raise RuntimeError(f"Source is missing or is a cloud placeholder: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        if any(part in SKIP_NAMES for part in path.relative_to(source).parts):
            continue
        if path.is_file():
            if not is_hydrated_file(path):
                continue
            copy_file(path, target / path.relative_to(source))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> tuple[int, Counter[int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    years: Counter[int] = Counter()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
                    if row.get("year") is not None:
                        years[int(row["year"])] += 1
    return count, years


def arxiv_catalog_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "paper_id",
            "arxiv_id",
            "arxiv_base_id",
            "title",
            "authors",
            "published_date",
            "updated_date",
            "year",
            "categories",
            "primary_category",
            "journal_reference",
            "doi",
            "arxiv_url",
            "pdf_url",
        )
    }


def acl_catalog_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "paper_id",
            "acl_id",
            "title",
            "authors",
            "year",
            "venue_prefix",
            "booktitle",
            "journal",
            "doi",
            "url",
            "pdf_url",
            "has_abstract",
        )
    }


def screening_row(row: dict[str, Any], corpus: str) -> dict[str, Any]:
    datasets = []
    for dataset in row.get("datasets") or []:
        datasets.append(
            {
                key: dataset.get(key)
                for key in (
                    "name",
                    "is_introduced",
                    "role",
                    "source_dataset",
                    "transformation_type",
                    "usage_description",
                    "confidence",
                )
            }
        )
    return {
        "corpus": corpus,
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "year": row.get("year"),
        "is_nlp_paper": row.get("is_nlp_paper"),
        "is_dataset_mentioned": row.get("is_dataset_mentioned"),
        "is_dataset_introducing": row.get("is_dataset_introducing"),
        "datasets": datasets,
        "exclusion_reason": row.get("exclusion_reason"),
        "classified_at": row.get("classified_at"),
    }


def sanitize_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "acus",
        "prior_dataset_mentions",
        "search_text",
        "ambiguities",
        "missing_information",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def transformed_rows(
    path: Path,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for row in read_jsonl(path):
        yield transform(row)


def screening_rows(paths: list[tuple[str, Path]]) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for corpus, path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            paper_id = str(row.get("paper_id") or "")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            yield screening_row(row, corpus)


def copy_release_code(output: Path) -> None:
    if BASE_RELEASE.exists():
        copy_tree(BASE_RELEASE, output)

    for stale_doc in ("DATA.md", "PIPELINE.md", "RELEASE_CHECKLIST.md", "RESULTS.md"):
        path = output / "docs" / stale_doc
        if path.exists():
            path.unlink()

    for name in ("README.md", "README_EXPANSION_2020_2025.md", "README_SCV.md", ".env.example", "requirements.txt"):
        copy_file(ROOT / name, output / name)
    copy_tree(ROOT / "docs", output / "docs")

    for source in sorted((ROOT / "scripts").glob("*.py")):
        if is_hydrated_file(source):
            copy_file(source, output / "scripts" / source.name)

    for source in sorted((ROOT / "tests").glob("*.py")):
        if is_hydrated_file(source):
            copy_file(source, output / "tests" / source.name)

    for source in sorted((ROOT / "scv").glob("*.py")):
        copy_file(source, output / "scv" / source.name)

    scraper_root = ROOT / "scrapers" / "arxiv_scraper"
    for source in sorted(scraper_root.rglob("*.py")):
        if "__pycache__" not in source.parts:
            copy_file(source, output / source.relative_to(ROOT))
    for name in ("scrapy.cfg",):
        source = scraper_root / name
        if source.exists():
            copy_file(source, output / source.relative_to(ROOT))


def copy_paper(output: Path) -> None:
    paper_source = ROOT / "publish" / "acl-style-files-master"
    names = (
        "acl_latex_clean.tex",
        "acl_latex_clean.pdf",
        "custom.bib",
        "acl.sty",
        "acl_natbib.bst",
    )
    for name in names:
        source = paper_source / name
        if source.exists():
            copy_file(source, output / "paper" / name)
    for source in sorted((paper_source / "figures").glob("*")):
        if source.suffix.lower() in {".png", ".pdf"}:
            copy_file(source, output / "paper" / "figures" / source.name)


def copy_summaries(output: Path) -> None:
    sources = (
        ROOT / "artifacts" / "arxiv_2020_2025_paper_metadata_s2_enriched_final_summary.json",
        ROOT / "artifacts" / "arxiv_2020_2025_paper_metadata_s2_enriched_final_summary.md",
        ROOT / "artifacts" / "arxiv_2020_2025_paper_metadata_s2_enriched_final_id_audit.json",
        ROOT / "artifacts" / "arxiv_2020_2025_paper_metadata_s2_enriched_final_id_audit.md",
        ROOT / "artifacts" / "main_analysis_expansion_2020_2025_manifest.json",
        ROOT / "artifacts" / "main_analysis_expansion_2020_2025_manifest.md",
        ROOT / "data" / "census" / "integrated_fulltext_banks_2023_2025_summary.json",
        ROOT / "data" / "census" / "acl_anthology" / "acl_anthology_2020_2025_all_with_abstracts_summary.json",
        ROOT / "data" / "census" / "acl_anthology" / "acl_anthology_2020_2025_all_with_abstracts_summary.md",
    )
    for source in sources:
        if source.exists():
            copy_file(source, output / "data" / "public" / "summaries" / source.name)


def scan_release(output: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append({"file": str(path.relative_to(output)), "pattern": name})
    return findings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output: Path) -> None:
    paths = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    lines = [f"{sha256_file(path)}  {path.relative_to(output)}" for path in paths]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_release(output: Path) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    copy_release_code(output)
    copy_paper(output)
    copy_summaries(output)

    public = output / "data" / "public"
    arxiv_source = ROOT / "data" / "processed" / "arxiv_2020_2025_dedup_no_acl_for_dataset_screening_v2.jsonl"
    acl_source = ROOT / "data" / "census" / "acl_anthology" / "acl_anthology_2020_2025_all_with_abstracts.jsonl"
    dataset_source = ROOT / "data" / "census" / "integrated_fulltext_dataset_bank_2023_2025.jsonl"
    screening_sources = [
        ("arxiv", ROOT / "data" / "processed" / "arxiv_2020_2022_dataset_screening_gemini31_flashlite.jsonl"),
        ("acl", ROOT / "data" / "processed" / "acl_2020_2022_dataset_screening_gemini31_flashlite.jsonl"),
    ]

    arxiv_count, arxiv_years = write_jsonl_gz(
        public / "arxiv_catalog_2020_2025.jsonl.gz",
        transformed_rows(arxiv_source, arxiv_catalog_row),
    )
    acl_count, acl_years = write_jsonl_gz(
        public / "acl_catalog_2020_2025.jsonl.gz",
        transformed_rows(acl_source, acl_catalog_row),
    )
    dataset_count, dataset_years = write_jsonl_gz(
        public / "dataset_bank_2023_2025.jsonl.gz",
        transformed_rows(dataset_source, sanitize_dataset_row),
    )
    screening_count, screening_years = write_jsonl_gz(
        public / "screening_labels_2020_2022.jsonl.gz",
        screening_rows(screening_sources),
    )

    expected_screening = {"arxiv": 64710, "acl": 22538}
    available_by_corpus: Counter[str] = Counter()
    with gzip.open(public / "screening_labels_2020_2022.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            available_by_corpus[json.loads(line)["corpus"]] += 1

    manifest = {
        "release_date": str(date.today()),
        "status": "research_snapshot",
        "main_analysis_years": [2023, 2025],
        "catalog_years": [2020, 2025],
        "files": {
            "arxiv_catalog_2020_2025.jsonl.gz": {
                "rows": arxiv_count,
                "rows_by_year": dict(sorted(arxiv_years.items())),
                "complete": True,
                "abstracts_included": False,
            },
            "acl_catalog_2020_2025.jsonl.gz": {
                "rows": acl_count,
                "rows_by_year": dict(sorted(acl_years.items())),
                "complete": True,
                "abstracts_included": False,
            },
            "dataset_bank_2023_2025.jsonl.gz": {
                "rows": dataset_count,
                "rows_by_year": dict(sorted(dataset_years.items())),
                "complete": True,
                "quoted_evidence_included": False,
            },
            "screening_labels_2020_2022.jsonl.gz": {
                "rows": screening_count,
                "rows_by_year": dict(sorted(screening_years.items())),
                "expected_by_corpus": expected_screening,
                "available_by_corpus": dict(sorted(available_by_corpus.items())),
                "coverage_pct": round(100 * screening_count / sum(expected_screening.values()), 2),
                "complete": screening_count == sum(expected_screening.values()),
            },
        },
        "notes": [
            "The manuscript's dataset/DCU conclusions currently cover 2023-2025.",
            "The 2020-2025 paper catalogs and arXiv citation enrichment are complete.",
            "Consult docs/DATA_CARD.md before reusing derived annotations.",
        ],
    }
    (public / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    findings = scan_release(output)
    if findings:
        raise RuntimeError(f"Release scan failed: {json.dumps(findings, indent=2)}")
    write_checksums(output)
    return manifest


def create_archive(output: Path) -> Path:
    archive = output.parent / f"{output.name}-{date.today()}.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as handle:
        handle.add(output, arcname=output.name, filter=lambda info: _normalize_tar(info))
    return archive


def _normalize_tar(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    manifest = build_release(output)
    archive = None if args.no_archive else create_archive(output)
    result = {
        "output": str(output),
        "archive": str(archive) if archive else None,
        "manifest": manifest,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
