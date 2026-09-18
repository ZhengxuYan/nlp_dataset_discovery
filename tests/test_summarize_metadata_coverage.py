from scripts.summarize_metadata_coverage import render_markdown, summarize_enriched_rows


def test_summarize_enriched_rows_counts_metadata_coverage():
    rows = [
        {
            "public_metadata": {
                "metadata_enrichment": {"paper_only": False, "resource_only": False},
                "paper_identifiers": {
                    "doi": "10.1/example",
                    "arxiv_id": "2001.00001",
                    "openalex_work_id": "https://openalex.org/W1",
                    "semantic_scholar_paper_id": "abc",
                },
                "paper_metadata_sources": [{"source": "semantic_scholar"}],
                "paper_metrics": {"citation_count": 12, "publication_year": 2020},
                "dataset_urls": {
                    "all": ["https://huggingface.co/datasets/org/name", "https://github.com/org/repo"],
                    "huggingface": ["https://huggingface.co/datasets/org/name"],
                    "github": ["https://github.com/org/repo"],
                    "paperswithcode": [],
                },
                "hf_metadata": [{"downloads": 100}],
                "github_metadata": [{"stars": 5}],
                "resource_health": [{"ok": True}, {"ok": False}],
            }
        },
        {"public_metadata": {"paper_identifiers": {}, "paper_metrics": {}, "dataset_urls": {"all": []}}},
    ]
    summary = summarize_enriched_rows(rows)
    assert summary["rows"] == 2
    assert summary["paper_enrichment_rows"] == 2
    assert summary["resource_enrichment_rows"] == 2
    assert summary["paper_citation_count"] == 1
    assert summary["paper_metadata_coverage"]["citation_count"]["pct_of_attempted"] == 50.0
    assert summary["paper_metadata_coverage"]["source_counts"] == {"semantic_scholar": 1}
    assert summary["hf_metadata_candidates"] == 1
    assert summary["hf_download_counts"] == 1
    assert summary["github_star_counts"] == 1
    assert summary["resource_metadata_coverage"]["hf"]["pct_of_hf_resources"] == 100.0
    assert summary["coverage_rates"]["citation_count_pct"] == 50.0
    assert summary["coverage_rates"]["hf_download_count_per_hf_link_pct"] == 100.0
    assert summary["coverage_rates"]["hf_download_count_per_hf_resource_pct"] == 100.0


def test_hf_name_fallback_coverage_cannot_exceed_100_percent():
    rows = [
        {
            "public_metadata": {
                "paper_identifiers": {},
                "paper_metrics": {},
                "dataset_urls": {
                    "all": ["https://huggingface.co/datasets/org/name"],
                    "huggingface": ["https://huggingface.co/datasets/org/name"],
                    "github": [],
                    "paperswithcode": [],
                },
                "hf_metadata": [{"downloads": 100}],
            }
        },
        {
            "public_metadata": {
                "paper_identifiers": {},
                "paper_metrics": {},
                "dataset_urls": {"all": [], "huggingface": [], "github": [], "paperswithcode": []},
                "hf_metadata": [{"downloads": 200, "match_method": "dataset_name_fuzzy"}],
            }
        },
    ]

    summary = summarize_enriched_rows(rows)

    assert summary["hf_dataset_links"] == 1
    assert summary["hf_metadata_candidates"] == 2
    assert summary["hf_download_counts"] == 2
    assert summary["coverage_rates"]["hf_download_count_per_hf_resource_pct"] == 100.0
    assert summary["coverage_rates"]["hf_download_count_per_hf_link_pct"] == 100.0


def test_paper_only_run_does_not_make_resource_coverage_look_attempted():
    rows = [
        {
            "public_metadata": {
                "metadata_enrichment": {"paper_only": True},
                "paper_identifiers": {"semantic_scholar_paper_id": "abc"},
                "paper_metrics": {"citation_count": 3},
                "dataset_urls": {"github": ["https://github.com/org/repo"], "all": ["https://github.com/org/repo"]},
                "github_metadata": [],
            }
        }
    ]

    summary = summarize_enriched_rows(rows)

    assert summary["paper_metadata_coverage"]["attempted_rows"] == 1
    assert summary["resource_metadata_coverage"]["attempted_rows"] == 0
    assert summary["paper_metadata_coverage"]["semantic_scholar_id"]["pct_of_attempted"] == 100.0
    assert summary["resource_metadata_coverage"]["github"]["star_counts"] == 0


def test_resource_only_run_does_not_make_paper_coverage_look_attempted():
    rows = [
        {
            "public_metadata": {
                "metadata_enrichment": {"resource_only": True},
                "paper_identifiers": {"acl_anthology_id": "2023.acl-main.1"},
                "paper_metrics": {"citation_count": None},
                "dataset_urls": {
                    "huggingface": ["https://huggingface.co/datasets/org/name"],
                    "all": ["https://huggingface.co/datasets/org/name"],
                },
                "hf_metadata": [{"downloads": 10}],
            }
        }
    ]

    summary = summarize_enriched_rows(rows)

    assert summary["paper_metadata_coverage"]["attempted_rows"] == 0
    assert summary["paper_metadata_coverage"]["citation_count"]["pct_of_attempted"] == 0.0
    assert summary["resource_metadata_coverage"]["attempted_rows"] == 1
    assert summary["resource_metadata_coverage"]["hf"]["pct_of_hf_resources"] == 100.0


def test_render_markdown_includes_core_sections():
    summary = summarize_enriched_rows([])
    markdown = render_markdown(summary, "input.jsonl")
    assert "# Public Metadata Coverage" in markdown
    assert "Paper Metadata" in markdown
    assert "Dataset Resource Metadata" in markdown
    assert "Paper enrichment attempted rows" in markdown
