from scripts import build_professor_update_brief


def test_render_brief_includes_progress_evidence_and_blocker():
    markdown = build_professor_update_brief.render_brief(
        {
            "placeholder_count": 12,
            "steps": [{"name": "a"}, {"name": "b"}],
        },
        {
            "placeholder_count": 12,
            "by_priority": {"high": 3},
        },
        {
            "coverage_rates": {
                "citation_count_pct": 50.0,
                "openalex_id_pct": 40.0,
                "semantic_scholar_id_pct": 60.0,
                "hf_download_count_per_hf_link_pct": 70.0,
                "github_star_count_per_github_link_pct": 80.0,
            }
        },
        {"rows": 5},
    )

    assert "# Professor Update Brief" in markdown
    assert "`2`-step run plan" in markdown
    assert "Local smoke citation coverage: `50.0%`" in markdown
    assert "Current total placeholder count is `12`" in markdown
    assert "`3` high-priority files" in markdown
    assert "staged 2020-2021 smoke run plan" in markdown
    assert "metadata_schema_audit_2020_2025.md" in markdown
    assert "metadata_review_sample_2020_2025.md" in markdown


def test_render_brief_zh_includes_progress_evidence_and_blocker():
    markdown = build_professor_update_brief.render_brief_zh(
        {
            "placeholder_count": 12,
            "steps": [{"name": "a"}, {"name": "b"}],
        },
        {
            "placeholder_count": 12,
            "by_priority": {"high": 3},
        },
        {
            "coverage_rates": {
                "citation_count_pct": 50.0,
                "openalex_id_pct": 40.0,
                "semantic_scholar_id_pct": 60.0,
                "hf_download_count_per_hf_link_pct": 70.0,
                "github_star_count_per_github_link_pct": 80.0,
            }
        },
        {"rows": 5},
    )

    assert "# 教授汇报简版" in markdown
    assert "`2` 步 run plan" in markdown
    assert "Local smoke citation coverage: `50.0%`" in markdown
    assert "当前总 placeholder 数是 `12`" in markdown
    assert "`3` 个 high-priority 文件" in markdown
    assert "2020-2021 staged smoke" in markdown
    assert "hydration_action_guide_2020_2025.md" in markdown
    assert "metadata_review_sample_2020_2025.md" in markdown
