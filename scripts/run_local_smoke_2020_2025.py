#!/usr/bin/env python3
"""Run a local 2020-2025 smoke pipeline with fixture data and cached metadata."""

from __future__ import annotations

import json
import subprocess
import argparse
from pathlib import Path
from typing import Any, Sequence


DEFAULT_OUTPUT_DIR = Path("artifacts/local_smoke_2020_2025")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def fixture_extractions() -> list[dict[str, Any]]:
    return [
        {
            "paper_id": "ACL:2020.acl-main.1",
            "acl_id": "2020.acl-main.1",
            "title": "A Local Smoke Dataset Paper",
            "year": 2020,
            "venue_prefix": "acl-main",
            "event": "acl-2020",
            "anthology_url": "https://aclanthology.org/2020.acl-main.1/",
            "pdf_url": "https://aclanthology.org/2020.acl-main.1.pdf",
            "doi": "10.0000/local-smoke",
            "datasets": [
                {
                    "dataset_id": "local_smoke_dataset",
                    "dataset_identity": {
                        "canonical_name": "Local Smoke Dataset",
                        "aliases": ["LSD"],
                        "acronym": "LSD",
                        "is_new_dataset": True,
                    },
                    "role": "introduced",
                    "resource_type": "dataset",
                    "primary_use": "benchmarking",
                    "is_reusable_resource": True,
                    "usage_description": (
                        "Released at https://huggingface.co/datasets/org/name and "
                        "https://github.com/org/repo for local smoke validation."
                    ),
                    "coverage": {
                        "tasks": ["classification"],
                        "domains": ["news"],
                        "languages": ["English"],
                        "modalities": ["text"],
                        "unit_of_analysis": "document",
                    },
                    "construction": {
                        "source_data_origin": "curated web documents",
                        "collection_method": "manual curation",
                        "source_datasets": [],
                        "transformation_types": ["filtering"],
                        "synthetic_generation": {"uses_llm": False, "model_names": []},
                    },
                    "scale": {"size_text": "1,000 examples", "num_instances": 1000},
                    "availability": {
                        "release_status": "released",
                        "license": "mit",
                        "documentation_type": "dataset card",
                        "artifacts": ["https://huggingface.co/datasets/org/name", "https://github.com/org/repo"],
                    },
                    "evaluation": {"used_for_training": False, "used_for_evaluation": True},
                    "added_information_summary": "Smoke dataset adds curated labels.",
                    "acus": [
                        {
                            "id": "a1",
                            "text": "The dataset contains 1,000 English news examples.",
                            "type": "scale",
                            "importance": "high",
                            "evidence": "1,000 examples",
                            "section": "Dataset",
                        }
                    ],
                },
                {
                    "dataset_id": "name_only_dataset",
                    "dataset_identity": {
                        "canonical_name": "Name Only Dataset",
                        "aliases": [],
                        "acronym": None,
                        "is_new_dataset": True,
                    },
                    "role": "introduced",
                    "resource_type": "dataset",
                    "primary_use": "evaluation",
                    "is_reusable_resource": True,
                    "usage_description": "Released publicly, but the smoke fixture omits direct resource URLs.",
                    "coverage": {
                        "tasks": ["question answering"],
                        "domains": ["web"],
                        "languages": ["English"],
                        "modalities": ["text"],
                        "unit_of_analysis": "question",
                    },
                    "construction": {
                        "source_data_origin": "curated web questions",
                        "collection_method": "manual curation",
                        "source_datasets": [],
                        "transformation_types": ["filtering"],
                        "synthetic_generation": {"uses_llm": False, "model_names": []},
                    },
                    "scale": {"size_text": "500 examples", "num_instances": 500},
                    "availability": {
                        "release_status": "released",
                        "license": "cc-by-4.0",
                        "documentation_type": "dataset card",
                        "artifacts": [],
                    },
                    "evaluation": {"used_for_training": False, "used_for_evaluation": True},
                    "added_information_summary": "Smoke dataset validates name-based resource lookup.",
                    "acus": [
                        {
                            "id": "a2",
                            "text": "The dataset contains 500 English web questions.",
                            "type": "scale",
                            "importance": "medium",
                            "evidence": "500 examples",
                            "section": "Dataset",
                        }
                    ],
                },
            ],
            "extraction_quality": {"confidence": "high", "ambiguities": [], "missing_information": []},
        },
        {
            "paper_id": "ACL:2021.acl-main.2",
            "acl_id": "2021.acl-main.2",
            "title": "A Low Citation Smoke Dataset Paper",
            "year": 2021,
            "venue_prefix": "acl-main",
            "event": "acl-2021",
            "anthology_url": "https://aclanthology.org/2021.acl-main.2/",
            "pdf_url": "https://aclanthology.org/2021.acl-main.2.pdf",
            "doi": "10.0000/low-smoke",
            "datasets": [
                {
                    "dataset_id": "low_citation_dataset",
                    "dataset_identity": {
                        "canonical_name": "Low Citation Dataset",
                        "aliases": [],
                        "acronym": None,
                        "is_new_dataset": True,
                    },
                    "role": "introduced",
                    "resource_type": "dataset",
                    "primary_use": "analysis",
                    "is_reusable_resource": True,
                    "usage_description": "Released at https://github.com/org/lowrepo for low-citation smoke validation.",
                    "coverage": {
                        "tasks": ["tagging"],
                        "domains": ["social media"],
                        "languages": ["English"],
                        "modalities": ["text"],
                        "unit_of_analysis": "post",
                    },
                    "construction": {
                        "source_data_origin": "curated social posts",
                        "collection_method": "manual curation",
                        "source_datasets": [],
                        "transformation_types": ["deduplication"],
                        "synthetic_generation": {"uses_llm": False, "model_names": []},
                    },
                    "scale": {"size_text": "250 examples", "num_instances": 250},
                    "availability": {
                        "release_status": "released",
                        "license": "apache-2.0",
                        "documentation_type": "README",
                        "artifacts": ["https://github.com/org/lowrepo"],
                    },
                    "evaluation": {"used_for_training": True, "used_for_evaluation": True},
                    "added_information_summary": "Smoke dataset validates low-citation review sampling.",
                    "acus": [
                        {
                            "id": "a1",
                            "text": "The dataset contains 250 social media posts.",
                            "type": "scale",
                            "importance": "low",
                            "evidence": "250 examples",
                            "section": "Dataset",
                        }
                    ],
                }
            ],
            "extraction_quality": {"confidence": "medium", "ambiguities": [], "missing_information": []},
        },
    ]


def cached_api_payload() -> dict[str, Any]:
    return {
        "json:https://api.openalex.org/works/doi:10.0000/local-smoke": {
            "ok": True,
            "status": 200,
            "data": {
                "id": "https://openalex.org/WLOCAL",
                "cited_by_count": 42,
                "referenced_works": ["https://openalex.org/WREF"],
                "publication_year": 2020,
                "primary_location": {"source": {"display_name": "ACL"}},
                "authorships": [{"author": {"display_name": "A. Researcher"}}],
            },
        },
        "json:https://api.semanticscholar.org/graph/v1/paper/DOI%3A10.0000%2Flocal-smoke?fields=paperId,title,year,venue,citationCount,influentialCitationCount,referenceCount,authors": {
            "ok": True,
            "status": 200,
            "data": {
                "paperId": "SLOCAL",
                "year": 2020,
                "venue": "ACL",
                "citationCount": 42,
                "influentialCitationCount": 5,
                "referenceCount": 12,
                "authors": [{"name": "A. Researcher"}],
            },
        },
        "json:https://api.openalex.org/works/doi:10.0000/low-smoke": {
            "ok": True,
            "status": 200,
            "data": {
                "id": "https://openalex.org/WLOW",
                "cited_by_count": 0,
                "referenced_works": [],
                "publication_year": 2021,
                "primary_location": {"source": {"display_name": "ACL"}},
                "authorships": [{"author": {"display_name": "C. Author"}}],
            },
        },
        "json:https://api.semanticscholar.org/graph/v1/paper/DOI%3A10.0000%2Flow-smoke?fields=paperId,title,year,venue,citationCount,influentialCitationCount,referenceCount,authors": {
            "ok": True,
            "status": 200,
            "data": {
                "paperId": "SLOW",
                "year": 2021,
                "venue": "ACL",
                "citationCount": 0,
                "influentialCitationCount": 0,
                "referenceCount": 3,
                "authors": [{"name": "C. Author"}],
            },
        },
        "json:https://huggingface.co/api/datasets/org/name": {
            "ok": True,
            "status": 200,
            "data": {
                "downloads": 1234,
                "likes": 56,
                "lastModified": "2026-01-01T00:00:00.000Z",
                "tags": ["license:mit"],
                "cardData": {"license": "mit"},
            },
        },
        "json:https://api.github.com/repos/org/repo": {
            "ok": True,
            "status": 200,
            "data": {
                "stargazers_count": 99,
                "forks_count": 10,
                "watchers_count": 99,
                "open_issues_count": 1,
                "pushed_at": "2026-01-02T00:00:00Z",
                "license": {"spdx_id": "MIT"},
            },
        },
        "json:https://api.github.com/repos/org/lowrepo": {
            "ok": True,
            "status": 200,
            "data": {
                "stargazers_count": 3,
                "forks_count": 1,
                "watchers_count": 3,
                "open_issues_count": 0,
                "pushed_at": "2026-01-05T00:00:00Z",
                "license": {"spdx_id": "Apache-2.0"},
            },
        },
        "json:https://huggingface.co/api/datasets?search=Name%20Only%20Dataset&limit=5": {
            "ok": True,
            "status": 200,
            "data": [
                {
                    "id": "org/name-only-dataset",
                    "downloads": 777,
                    "likes": 12,
                    "lastModified": "2026-01-03T00:00:00.000Z",
                    "tags": ["license:cc-by-4.0"],
                    "cardData": {"license": "cc-by-4.0"},
                }
            ],
        },
        "json:https://paperswithcode.com/api/v1/datasets/?q=Name%20Only%20Dataset": {
            "ok": True,
            "status": 200,
            "data": {
                "results": [
                    {
                        "name": "Name Only Dataset",
                        "slug": "name-only-dataset",
                        "url": "https://paperswithcode.com/dataset/name-only-dataset",
                    }
                ]
            },
        },
        "health:https://aclanthology.org/2020.acl-main.1/": {
            "url": "https://aclanthology.org/2020.acl-main.1/",
            "ok": True,
            "status": 200,
            "resolved_url": "https://aclanthology.org/2020.acl-main.1/",
            "downloadable": False,
            "checked_at": "2026-01-04T00:00:00+00:00",
        },
        "health:https://aclanthology.org/2020.acl-main.1.pdf": {
            "url": "https://aclanthology.org/2020.acl-main.1.pdf",
            "ok": True,
            "status": 200,
            "resolved_url": "https://aclanthology.org/2020.acl-main.1.pdf",
            "downloadable": True,
            "checked_at": "2026-01-04T00:00:00+00:00",
        },
        "health:https://huggingface.co/datasets/org/name": {
            "url": "https://huggingface.co/datasets/org/name",
            "ok": True,
            "status": 200,
            "resolved_url": "https://huggingface.co/datasets/org/name",
            "downloadable": False,
            "checked_at": "2026-01-04T00:00:00+00:00",
        },
        "health:https://github.com/org/repo": {
            "url": "https://github.com/org/repo",
            "ok": True,
            "status": 200,
            "resolved_url": "https://github.com/org/repo",
            "downloadable": False,
            "checked_at": "2026-01-04T00:00:00+00:00",
        },
        "health:https://aclanthology.org/2021.acl-main.2/": {
            "url": "https://aclanthology.org/2021.acl-main.2/",
            "ok": True,
            "status": 200,
            "resolved_url": "https://aclanthology.org/2021.acl-main.2/",
            "downloadable": False,
            "checked_at": "2026-01-05T00:00:00+00:00",
        },
        "health:https://aclanthology.org/2021.acl-main.2.pdf": {
            "url": "https://aclanthology.org/2021.acl-main.2.pdf",
            "ok": True,
            "status": 200,
            "resolved_url": "https://aclanthology.org/2021.acl-main.2.pdf",
            "downloadable": True,
            "checked_at": "2026-01-05T00:00:00+00:00",
        },
        "health:https://github.com/org/lowrepo": {
            "url": "https://github.com/org/lowrepo",
            "ok": True,
            "status": 200,
            "resolved_url": "https://github.com/org/lowrepo",
            "downloadable": False,
            "checked_at": "2026-01-05T00:00:00+00:00",
        },
    }


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local fixture-based 2020-2025 smoke pipeline.")
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir
    root = args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_jsonl = output_dir / "fixture_fulltext_extractions_2020_2025.jsonl"
    integrated_extractions = output_dir / "integrated_fulltext_dataset_extractions_2020_2025.jsonl"
    dataset_bank = output_dir / "integrated_fulltext_dataset_bank_2020_2025.jsonl"
    acu_bank = output_dir / "integrated_fulltext_acu_bank_2020_2025.jsonl"
    integrated_summary = output_dir / "integrated_fulltext_banks_2020_2025_summary.json"
    enriched = output_dir / "integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl"
    enriched_summary = output_dir / "integrated_fulltext_dataset_bank_2020_2025_enriched_summary.json"
    cache = output_dir / "public_metadata_api_cache.json"
    stable_id_audit_json = output_dir / "enrichment_stable_id_audit_2020_2025.json"
    stable_id_audit_md = output_dir / "enrichment_stable_id_audit_2020_2025.md"
    coverage_json = output_dir / "metadata_coverage_2020_2025.json"
    coverage_md = output_dir / "metadata_coverage_2020_2025.md"
    schema_audit_json = output_dir / "metadata_schema_audit_2020_2025.json"
    schema_audit_md = output_dir / "metadata_schema_audit_2020_2025.md"
    review_sample_json = output_dir / "metadata_review_sample_2020_2025.json"
    review_sample_jsonl = output_dir / "metadata_review_sample_2020_2025.jsonl"
    review_sample_md = output_dir / "metadata_review_sample_2020_2025.md"

    write_jsonl(fixture_jsonl, fixture_extractions())
    cache.write_text(json.dumps(cached_api_payload(), indent=2, sort_keys=True), encoding="utf-8")
    run(
        [
            "python",
            "scripts/build_integrated_fulltext_banks.py",
            "--input-jsonl",
            str(fixture_jsonl),
            "--source-corpus",
            "local_smoke",
            "--output-extractions-jsonl",
            str(integrated_extractions),
            "--output-dataset-bank-jsonl",
            str(dataset_bank),
            "--output-acu-bank-jsonl",
            str(acu_bank),
            "--summary-json",
            str(integrated_summary),
        ],
        root,
    )
    run(
        [
            "python",
            "scripts/enrich_dataset_metadata.py",
            "--input",
            str(dataset_bank),
            "--output",
            str(enriched),
            "--summary-output",
            str(enriched_summary),
            "--cache",
            str(cache),
            "--offline",
            "--allow-semantic-batch-failures",
            "--resource-name-fallback",
            "--check-url-health",
        ],
        root,
    )
    run(
        [
            "python",
            "scripts/audit_enrichment_stable_ids.py",
            "--input-jsonl",
            str(dataset_bank),
            "--enriched-jsonl",
            str(enriched),
            "--output-json",
            str(stable_id_audit_json),
            "--output-md",
            str(stable_id_audit_md),
        ],
        root,
    )
    run(
        [
            "python",
            "scripts/summarize_metadata_coverage.py",
            "--input-jsonl",
            str(enriched),
            "--output-json",
            str(coverage_json),
            "--output-md",
            str(coverage_md),
        ],
        root,
    )
    run(
        [
            "python",
            "scripts/audit_metadata_schema.py",
            "--input-jsonl",
            str(enriched),
            "--output-json",
            str(schema_audit_json),
            "--output-md",
            str(schema_audit_md),
        ],
        root,
    )
    run(
        [
            "python",
            "scripts/build_metadata_review_sample.py",
            "--input-jsonl",
            str(enriched),
            "--output-json",
            str(review_sample_json),
            "--output-jsonl",
            str(review_sample_jsonl),
            "--output-md",
            str(review_sample_md),
        ],
        root,
    )
    manifest = {
        "fixture_jsonl": str(fixture_jsonl),
        "dataset_bank": str(dataset_bank),
        "acu_bank": str(acu_bank),
        "integrated_summary": str(integrated_summary),
        "enriched_jsonl": str(enriched),
        "enriched_summary": str(enriched_summary),
        "stable_id_audit_json": str(stable_id_audit_json),
        "stable_id_audit_md": str(stable_id_audit_md),
        "coverage_json": str(coverage_json),
        "coverage_md": str(coverage_md),
        "schema_audit_json": str(schema_audit_json),
        "schema_audit_md": str(schema_audit_md),
        "review_sample_json": str(review_sample_json),
        "review_sample_jsonl": str(review_sample_jsonl),
        "review_sample_md": str(review_sample_md),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
