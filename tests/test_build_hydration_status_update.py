from scripts import build_hydration_status_update


def test_build_status_keeps_full_run_blocked_until_hydration():
    status = build_hydration_status_update.build_status(
        {"placeholder_count": 12, "by_priority": {"high": 2, "medium": 3, "low": 7}},
        {"safe_to_execute_full_pipeline": True},
        {"rows": 1, "coverage_rates": {"citation_count_pct": 100.0}},
    )

    assert status["placeholder_count"] == 12
    assert status["by_priority"]["high"] == 2
    assert status["safe_to_execute_full_pipeline"] is False
    assert status["next_gate"] == "hydrate_high_priority_files"
    assert status["local_smoke_rows"] == 1


def test_build_status_allows_full_run_when_hydration_is_clear():
    status = build_hydration_status_update.build_status(
        {"placeholder_count": 0, "by_priority": {}},
        {"safe_to_execute_full_pipeline": True},
        {},
    )

    assert status["safe_to_execute_full_pipeline"] is True
    assert status["next_gate"] == "run_full_pipeline"


def test_render_markdown_includes_evidence_and_commands():
    markdown = build_hydration_status_update.render_markdown(
        {
            "generated_at": "now",
            "placeholder_count": 2,
            "by_priority": {"high": 1, "medium": 1, "low": 0},
            "safe_to_execute_full_pipeline": False,
            "next_gate": "hydrate_high_priority_files",
            "local_smoke_rows": 1,
            "local_smoke_coverage_rates": {
                "citation_count_pct": 100.0,
                "openalex_id_pct": 100.0,
                "semantic_scholar_id_pct": 100.0,
                "hf_download_count_per_hf_link_pct": 100.0,
                "github_star_count_per_github_link_pct": 100.0,
            },
            "professor_update_status": "Blocked on hydration.",
        }
    )

    assert "# Hydration Status" in markdown
    assert "Local smoke citation coverage: `100.0%`" in markdown
    assert "python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps" in markdown
    assert "artifacts/high_priority_hydration_files_2020_2025.txt" in markdown
