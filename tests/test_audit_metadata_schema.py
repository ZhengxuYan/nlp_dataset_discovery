from scripts import audit_metadata_schema


def test_audit_rows_counts_structure_and_missing_reasons():
    rows = [
        {
            "public_metadata": {
                "paper_identifiers": {
                    "doi": "10.1/example",
                    "arxiv_id": None,
                    "acl_anthology_id": "2020.acl-main.1",
                    "openalex_work_id": "https://openalex.org/W1",
                    "semantic_scholar_paper_id": "S1",
                },
                "paper_metrics": {
                    "citation_count": 10,
                    "influential_citation_count": 2,
                    "reference_count": 20,
                    "publication_year": 2020,
                    "venue": "ACL",
                    "authors": ["A"],
                },
                "paper_metadata_sources": [],
                "dataset_urls": {
                    "all": ["https://huggingface.co/datasets/org/name"],
                    "huggingface": ["https://huggingface.co/datasets/org/name"],
                    "github": [],
                    "paperswithcode": [],
                    "project_pages": [],
                    "downloads": [],
                },
                "hf_metadata": [],
                "github_metadata": [],
                "pwc_metadata": [],
                "resource_health": [],
                "metadata_enrichment": {"queried_at": "now"},
            }
        },
        {
            "public_metadata": {
                "paper_identifiers": {
                    "doi": None,
                    "arxiv_id": None,
                    "acl_anthology_id": None,
                    "openalex_work_id": None,
                    "semantic_scholar_paper_id": None,
                },
                "paper_metrics": {"citation_count": None},
                "dataset_urls": {"all": [], "huggingface": [], "github": []},
            }
        },
    ]

    audit = audit_metadata_schema.audit_rows(rows)

    assert audit["rows"] == 2
    assert audit["existing_fields"]["paper_identifiers.doi"] == 2
    assert audit["present_values"]["paper_identifiers.doi"] == 1
    assert audit["present_values"]["paper_metrics.citation_count"] == 1
    assert audit["missing_reason_counts"]["no_exact_paper_identifier"] == 1
    assert audit["missing_reason_counts"]["no_citation_count"] == 1
    assert audit["missing_reason_counts"]["no_github_url"] == 2


def test_audit_rows_counts_missing_public_metadata():
    audit = audit_metadata_schema.audit_rows([{"dataset_name": "x"}])

    assert audit["missing_structure"]["public_metadata"] == 1
    assert audit["missing_reason_counts"]["missing_public_metadata"] == 1


def test_audit_rows_counts_resource_match_provenance():
    rows = [
        {
            "public_metadata": {
                "paper_identifiers": {},
                "paper_metrics": {},
                "paper_metadata_sources": [
                    {
                        "source": "openalex",
                        "match_method": "title_fuzzy",
                        "match_confidence": "fuzzy_high",
                        "match_confidence_score": 0.98,
                    }
                ],
                "dataset_urls": {
                    "all": [],
                    "huggingface": [],
                    "github": [],
                    "paperswithcode": [],
                    "project_pages": [],
                    "downloads": [],
                },
                "dataset_name_candidates": ["Name Only Dataset"],
                "hf_metadata": [
                    {
                        "source": "huggingface",
                        "match_method": "dataset_name_fuzzy",
                        "match_confidence": "fuzzy_high",
                        "match_confidence_score": 1.0,
                        "downloads": 777,
                    }
                ],
                "github_metadata": [],
                "pwc_metadata": [
                    {
                        "source": "paperswithcode",
                        "match_method": "dataset_name_fuzzy",
                        "match_confidence": "fuzzy_high",
                        "match_confidence_score": 1.0,
                    }
                ],
                "resource_health": [],
                "metadata_enrichment": {"queried_at": "now"},
            }
        }
    ]

    audit = audit_metadata_schema.audit_rows(rows)

    assert audit["present_values"]["public_metadata.dataset_name_candidates"] == 1
    assert audit["present_values"]["paper_metadata_sources.match_confidence_score"] == 1
    assert audit["present_values"]["hf_metadata.match_method"] == 1
    assert audit["present_values"]["hf_metadata.match_confidence_score"] == 1
    assert audit["present_values"]["pwc_metadata.match_confidence_score"] == 1
    assert audit["missing_reason_counts"]["no_github_metadata"] == 1
    assert audit["missing_reason_counts"]["no_resource_health"] == 1


def test_audit_rows_counts_resource_health_fields():
    rows = [
        {
            "public_metadata": {
                "paper_identifiers": {},
                "paper_metrics": {},
                "paper_metadata_sources": [],
                "dataset_urls": {
                    "all": ["https://example.com/data.csv"],
                    "huggingface": [],
                    "github": [],
                    "paperswithcode": [],
                    "project_pages": [],
                    "downloads": ["https://example.com/data.csv"],
                },
                "dataset_name_candidates": ["Example"],
                "hf_metadata": [],
                "github_metadata": [],
                "pwc_metadata": [],
                "resource_health": [
                    {
                        "url": "https://example.com/data.csv",
                        "status": 200,
                        "resolved_url": "https://example.com/data.csv",
                        "downloadable": True,
                        "checked_at": "now",
                    }
                ],
                "metadata_enrichment": {"queried_at": "now"},
            }
        }
    ]

    audit = audit_metadata_schema.audit_rows(rows)

    assert audit["present_values"]["resource_health.url"] == 1
    assert audit["present_values"]["resource_health.status"] == 1
    assert audit["present_values"]["resource_health.resolved_url"] == 1
    assert audit["present_values"]["resource_health.downloadable"] == 1
    assert audit["present_values"]["resource_health.checked_at"] == 1


def test_render_markdown_includes_missing_reason_table():
    audit = audit_metadata_schema.audit_rows([{"dataset_name": "x"}])
    markdown = audit_metadata_schema.render_markdown(audit, "input.jsonl")

    assert "# Metadata Schema Audit" in markdown
    assert "Missing Metadata Reasons" in markdown
    assert "missing_public_metadata" in markdown
