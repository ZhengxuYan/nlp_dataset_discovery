import json
import hashlib
from pathlib import Path

from scripts.enrich_dataset_metadata import (
    CachedHttpClient,
    classify_dataset_urls,
    collect_urls,
    enrich_row,
    extract_doi,
    github_repo,
    hf_repo_id,
    is_transient_result,
    normalize_doi,
    prefetch_semantic_scholar_exact,
    query_resource_metadata,
    query_paper_metadata,
    select_rows,
    title_similarity,
)


def test_identifier_extraction_exact_values():
    row = {
        "doi": "https://doi.org/10.18653/v1/2020.acl-main.1",
        "project": "https://huggingface.co/datasets/org/name and https://github.com/org/repo",
    }
    urls = collect_urls(row)
    assert normalize_doi(row["doi"]) == "10.18653/v1/2020.acl-main.1"
    assert extract_doi(row, urls) == "10.18653/v1/2020.acl-main.1"
    assert hf_repo_id("https://huggingface.co/datasets/org/name") == "org/name"
    assert github_repo("https://github.com/org/repo/tree/main") == "org/repo"


def test_classify_dataset_urls():
    urls = [
        "https://huggingface.co/datasets/org/name",
        "https://github.com/org/repo",
        "https://paperswithcode.com/dataset/squad",
        "https://example.com/data.csv",
        "https://example.com/project",
    ]
    classified = classify_dataset_urls(urls)
    assert classified["huggingface"] == ["https://huggingface.co/datasets/org/name"]
    assert classified["github"] == ["https://github.com/org/repo"]
    assert classified["paperswithcode"] == ["https://paperswithcode.com/dataset/squad"]
    assert classified["downloads"] == ["https://example.com/data.csv"]
    assert classified["project_pages"] == ["https://example.com/project"]


def test_collect_urls_removes_unmatched_markdown_bracket():
    urls = collect_urls({"text": "Download from [https://ngc.nvidia.com]"})

    assert urls == ["https://ngc.nvidia.com"]
    assert classify_dataset_urls(urls)["project_pages"] == ["https://ngc.nvidia.com"]


def test_collect_urls_preserves_valid_ipv6_and_skips_invalid_ipv6():
    urls = collect_urls(
        {
            "valid": "http://[::1]/dataset",
            "invalid": "http://[invalid-ipv6/dataset",
        }
    )

    assert urls == ["http://[::1]/dataset"]


def test_enrich_row_offline_records_cache_misses(tmp_path: Path):
    client = CachedHttpClient(tmp_path / "cache.json", offline=True)
    row = {
        "title": "A Dataset Paper",
        "doi": "10.0000/example",
        "url": "https://huggingface.co/datasets/org/name",
    }
    enriched = enrich_row(row, client, check_health=False)
    metadata = enriched["public_metadata"]
    assert metadata["paper_identifiers"]["doi"] == "10.0000/example"
    assert metadata["dataset_urls"]["huggingface"] == ["https://huggingface.co/datasets/org/name"]
    assert metadata["hf_metadata"][0]["error"] == "offline cache miss"


def test_paper_only_skips_resource_api_lookups(tmp_path: Path):
    client = CachedHttpClient(tmp_path / "cache.json", offline=True)
    row = {
        "title": "A Dataset Paper",
        "url": "https://huggingface.co/datasets/org/name https://github.com/org/repo",
    }
    enriched = enrich_row(row, client, check_health=True, paper_only=True)
    metadata = enriched["public_metadata"]

    assert metadata["dataset_urls"]["huggingface"] == ["https://huggingface.co/datasets/org/name"]
    assert metadata["hf_metadata"] == []
    assert metadata["github_metadata"] == []
    assert metadata["resource_health"] == []
    assert metadata["metadata_enrichment"]["paper_only"] is True


def test_title_similarity_normalizes_punctuation_and_case():
    assert title_similarity("A Dataset: For NLP!", "a dataset for nlp") == 1.0
    assert title_similarity("A Dataset for NLP", "Completely Different Paper") < 0.5


def test_fuzzy_title_fallback_records_confidence_score(tmp_path: Path):
    cache = {
        "json:https://api.openalex.org/works?search=A%20Dataset%20for%20NLP&per-page=1": {
            "ok": True,
            "status": 200,
            "data": {
                "results": [
                    {
                        "id": "https://openalex.org/WFUZZY",
                        "display_name": "A Dataset for NLP",
                        "cited_by_count": 7,
                        "publication_year": 2021,
                        "authorships": [{"author": {"display_name": "B. Author"}}],
                    }
                ]
            },
        },
        "json:https://api.semanticscholar.org/graph/v1/paper/search?limit=1&fields=paperId,title,year,venue,citationCount,influentialCitationCount,referenceCount,authors&query=A%20Dataset%20for%20NLP": {
            "ok": True,
            "status": 200,
            "data": {
                "data": [
                    {
                        "paperId": "SFUZZY",
                        "title": "A Dataset for NLP",
                        "year": 2021,
                        "citationCount": 7,
                        "authors": [{"name": "B. Author"}],
                    }
                ]
            },
        },
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    client = CachedHttpClient(cache_path, offline=True)

    metadata = query_paper_metadata({"title": "A Dataset for NLP"}, [], client)
    sources = {source["source"]: source for source in metadata["paper_metadata_sources"]}

    assert metadata["paper_identifiers"]["openalex_work_id"] == "https://openalex.org/WFUZZY"
    assert metadata["paper_identifiers"]["semantic_scholar_paper_id"] == "SFUZZY"
    assert sources["openalex"]["match_method"] == "title_fuzzy"
    assert sources["openalex"]["match_confidence_score"] == 1.0
    assert sources["semantic_scholar"]["matched_title"] == "A Dataset for NLP"


def test_low_similarity_fuzzy_title_fallback_is_rejected(tmp_path: Path):
    cache = {
        "json:https://api.openalex.org/works?search=A%20Dataset%20for%20NLP&per-page=1": {
            "ok": True,
            "status": 200,
            "data": {"results": [{"id": "https://openalex.org/WBAD", "display_name": "Unrelated Vision Paper"}]},
        },
        "json:https://api.semanticscholar.org/graph/v1/paper/search?limit=1&fields=paperId,title,year,venue,citationCount,influentialCitationCount,referenceCount,authors&query=A%20Dataset%20for%20NLP": {
            "ok": True,
            "status": 200,
            "data": {"data": [{"paperId": "SBAD", "title": "Unrelated Vision Paper"}]},
        },
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    client = CachedHttpClient(cache_path, offline=True)

    metadata = query_paper_metadata({"title": "A Dataset for NLP"}, [], client)

    assert metadata["paper_identifiers"].get("openalex_work_id") is None
    assert metadata["paper_identifiers"].get("semantic_scholar_paper_id") is None
    assert metadata["paper_metadata_sources"] == []


def test_resource_name_fallback_finds_hf_and_pwc_metadata(tmp_path: Path):
    cache = {
        "json:https://huggingface.co/api/datasets?search=Local%20Smoke%20Dataset&limit=5": {
            "ok": True,
            "status": 200,
            "data": [
                {
                    "id": "org/local-smoke-dataset",
                    "downloads": 321,
                    "likes": 9,
                    "lastModified": "2026-01-01T00:00:00.000Z",
                    "tags": ["license:mit"],
                    "cardData": {"license": "mit"},
                }
            ],
        },
        "json:https://paperswithcode.com/api/v1/datasets/?q=Local%20Smoke%20Dataset": {
            "ok": True,
            "status": 200,
            "data": {
                "results": [
                    {
                        "name": "Local Smoke Dataset",
                        "slug": "local-smoke-dataset",
                        "url": "https://paperswithcode.com/dataset/local-smoke-dataset",
                    }
                ]
            },
        },
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    client = CachedHttpClient(cache_path, offline=True)

    metadata = query_resource_metadata(
        {"all": [], "huggingface": [], "github": [], "paperswithcode": []},
        client,
        check_health=False,
        dataset_names=["Local Smoke Dataset"],
    )

    assert metadata["hf_metadata"][0]["repo_id"] == "org/local-smoke-dataset"
    assert metadata["hf_metadata"][0]["downloads"] == 321
    assert metadata["hf_metadata"][0]["match_method"] == "dataset_name_fuzzy"
    assert metadata["hf_metadata"][0]["match_confidence_score"] >= 0.72
    assert metadata["pwc_metadata"][0]["slug"] == "local-smoke-dataset"
    assert metadata["pwc_metadata"][0]["match_method"] == "dataset_name_fuzzy"


def test_resource_name_fallback_rejects_low_similarity_matches(tmp_path: Path):
    cache = {
        "json:https://huggingface.co/api/datasets?search=Local%20Smoke%20Dataset&limit=5": {
            "ok": True,
            "status": 200,
            "data": [{"id": "org/unrelated-vision-corpus", "downloads": 999}],
        },
        "json:https://paperswithcode.com/api/v1/datasets/?q=Local%20Smoke%20Dataset": {
            "ok": True,
            "status": 200,
            "data": {"results": [{"name": "Unrelated Vision Corpus", "slug": "unrelated"}]},
        },
    }
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    client = CachedHttpClient(cache_path, offline=True)

    metadata = query_resource_metadata(
        {"all": [], "huggingface": [], "github": [], "paperswithcode": []},
        client,
        check_health=False,
        dataset_names=["Local Smoke Dataset"],
    )

    assert metadata["hf_metadata"][0]["error"] == "no confident dataset name match"
    assert metadata["hf_metadata"][0]["match_confidence"] == "fuzzy_low"
    assert metadata["pwc_metadata"][0]["error"] == "no confident dataset name match"


def test_transient_result_detection():
    assert is_transient_result({"ok": False, "status": 429, "error": "rate limited"})
    assert is_transient_result({"ok": False, "status": None, "error": "timeout"})
    assert not is_transient_result({"ok": False, "status": 404, "error": "not found"})
    assert not is_transient_result({"ok": True, "status": 200})


def test_stratified_year_sampling_is_deterministic():
    rows = [
        {"paper_id": "2020-a", "year": 2020},
        {"paper_id": "2020-b", "year": 2020},
        {"paper_id": "2021-a", "year": 2021},
        {"paper_id": "2021-b", "year": 2021},
        {"paper_id": "2022-a", "year": 2022},
        {"paper_id": "2022-b", "year": 2022},
    ]

    sampled = list(select_rows(rows, limit=3, sample_strategy="stratified-year"))

    assert [row["paper_id"] for row in sampled] == ["2020-a", "2021-a", "2022-a"]


def test_first_sampling_supports_offset_and_limit():
    rows = [{"paper_id": str(idx)} for idx in range(5)]

    sampled = list(select_rows(rows, offset=2, limit=2, sample_strategy="first"))

    assert [row["paper_id"] for row in sampled] == ["2", "3"]


def test_semantic_scholar_batch_prefetch_uses_exact_arxiv_ids(tmp_path: Path):
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=paperId,title,year,venue,citationCount,influentialCitationCount,referenceCount,authors"
    payload = {"ids": ["ARXIV:2001.00119"]}
    cache_key = "json-post:" + endpoint + ":" + hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                cache_key: {
                    "ok": True,
                    "status": 200,
                    "data": [
                        {
                            "paperId": "S2PAPER",
                            "title": "Example Paper",
                            "year": 2020,
                            "citationCount": 11,
                            "influentialCitationCount": 2,
                            "referenceCount": 5,
                            "authors": [{"name": "A. Author"}],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    client = CachedHttpClient(cache_path, offline=True)
    row = {"paper_id": "2001.00119v2", "arxiv_id": "2001.00119v2", "title": "Example Paper"}

    prefetched = prefetch_semantic_scholar_exact([row], client)
    metadata = query_paper_metadata(row, [], client, semantic_override=prefetched[row["paper_id"]])

    assert prefetched[row["paper_id"]]["paperId"] == "S2PAPER"
    assert metadata["paper_identifiers"]["semantic_scholar_paper_id"] == "S2PAPER"
    assert metadata["paper_metrics"]["citation_count"] == 11


def test_openalex_can_be_disabled_for_scale(tmp_path: Path):
    client = CachedHttpClient(tmp_path / "cache.json", offline=True)
    semantic = {
        "paperId": "S2PAPER",
        "year": 2020,
        "citationCount": 5,
        "influentialCitationCount": 1,
        "referenceCount": 3,
        "authors": ["A. Author"],
        "query_key": "arxiv:2001.00119",
        "match_confidence": "exact",
        "match_method": "arxiv",
        "match_confidence_score": 1.0,
        "matched_title": "Example Paper",
    }

    metadata = query_paper_metadata(
        {"paper_id": "2001.00119v2", "arxiv_id": "2001.00119v2", "title": "Example Paper"},
        [],
        client,
        semantic_override=semantic,
        openalex_mode="off",
    )

    assert metadata["paper_identifiers"].get("openalex_work_id") is None
    assert metadata["paper_identifiers"]["semantic_scholar_paper_id"] == "S2PAPER"
    assert metadata["paper_metrics"]["citation_count"] == 5


def test_empty_semantic_override_disables_per_row_fallback(tmp_path: Path):
    client = CachedHttpClient(tmp_path / "cache.json", offline=True)

    metadata = query_paper_metadata(
        {"paper_id": "2001.00119v2", "arxiv_id": "2001.00119v2", "title": "Example Paper"},
        [],
        client,
        semantic_override={},
        openalex_mode="off",
    )

    assert metadata["paper_identifiers"].get("semantic_scholar_paper_id") is None
    assert metadata["paper_metrics"]["citation_count"] is None
