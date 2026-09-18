from scripts import build_metadata_review_sample


def row(title: str, citations, hf=None, github=None, hf_metadata=None, pwc_metadata=None, health=None):
    return {
        "title": title,
        "public_metadata": {
            "paper_identifiers": {"doi": f"10.0/{title}"},
            "paper_metrics": {"citation_count": citations, "publication_year": 2020},
            "dataset_urls": {
                "all": (hf or []) + (github or []),
                "huggingface": hf or [],
                "github": github or [],
            },
            "hf_metadata": hf_metadata or [],
            "github_metadata": [],
            "pwc_metadata": pwc_metadata or [],
            "resource_health": health or [],
            "metadata_enrichment": {"record_id": title},
        },
    }


def test_build_sample_creates_expected_buckets():
    sample = build_metadata_review_sample.build_sample(
        [
            row("high", 100, hf=["https://huggingface.co/datasets/org/name"]),
            row("low", 1, github=["https://github.com/org/repo"]),
            row(
                "fallback",
                10,
                hf_metadata=[{"match_method": "dataset_name_fuzzy"}],
                pwc_metadata=[{"slug": "fallback", "match_method": "dataset_name_fuzzy"}],
                health=[{"ok": True, "downloadable": True}],
            ),
            row("missing", None),
        ],
        per_bucket=2,
    )

    buckets = {item["bucket"] for item in sample["samples"]}
    assert "high_citation" in buckets
    assert "low_citation" in buckets
    assert "no_citation" in buckets
    assert "huggingface_linked" in buckets
    assert "github_linked" in buckets
    assert "huggingface_name_fallback" in buckets
    assert "paperswithcode_matched" in buckets
    assert "healthy_url" in buckets
    assert "downloadable_url" in buckets


def test_render_markdown_includes_manual_audit_note():
    sample = build_metadata_review_sample.build_sample([row("high", 100)], per_bucket=1)
    markdown = build_metadata_review_sample.render_markdown(sample, "input.jsonl")

    assert "# Metadata Review Sample" in markdown
    assert "Manual Audit Use" in markdown
    assert "high_citation" in markdown
