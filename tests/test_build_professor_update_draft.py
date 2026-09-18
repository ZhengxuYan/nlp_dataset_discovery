from scripts import build_professor_update_draft


def test_build_draft_summarizes_current_progress():
    draft = build_professor_update_draft.build_draft(
        {
            "status": "blocked_by_hydration",
            "placeholder_count": 10,
            "high_priority_placeholder_count": 2,
            "artifact_count": 5,
            "missing_artifact_count": 0,
            "run_plan_steps": 12,
        },
        {
            "requirements": [
                {"status": "done"},
                {"status": "done"},
                {"status": "blocked_by_hydration"},
            ]
        },
        {
            "current_gate": "hydrate_cloud_placeholders",
            "next_commands": ["python scripts/check_cloud_placeholders.py --summary-only"],
        },
        {
            "done_summary": {
                "local_smoke_papers": 2,
                "local_smoke_datasets": 3,
                "local_smoke_acus": 3,
                "review_sample_count": 17,
                "stable_id_audit_status": "pass",
                "full_plan_url_health_enabled": True,
            },
            "coverage_rates": {
                "citation_count_pct": 100.0,
                "openalex_id_pct": 100.0,
                "semantic_scholar_id_pct": 100.0,
                "github_star_count_per_github_resource_pct": 100.0,
                "healthy_url_pct": 100.0,
            },
        },
    )

    assert draft["current_gate"] == "hydrate_cloud_placeholders"
    assert draft["completion_requirement_counts"] == {"done": 2, "blocked_by_hydration": 1}
    assert draft["local_smoke"]["stable_id_audit_status"] == "pass"
    assert draft["next_commands"] == ["python scripts/check_cloud_placeholders.py --summary-only"]


def test_render_markdown_is_send_ready_and_includes_evidence():
    markdown = build_professor_update_draft.render_markdown(
        {
            "generated_at": "now",
            "subject": "subject",
            "status": "blocked_by_hydration",
            "current_gate": "hydrate_cloud_placeholders",
            "placeholder_count": 10,
            "high_priority_placeholder_count": 2,
            "artifact_count": 5,
            "missing_artifact_count": 0,
            "run_plan_steps": 12,
            "local_smoke": {
                "papers": 2,
                "datasets": 3,
                "acus": 3,
                "stable_id_audit_status": "pass",
            },
            "coverage_rates": {
                "citation_count_pct": 100.0,
                "openalex_id_pct": 100.0,
                "semantic_scholar_id_pct": 100.0,
                "github_star_count_per_github_resource_pct": 100.0,
                "healthy_url_pct": 100.0,
            },
            "completion_requirement_counts": {
                "done": 12,
                "ready_after_hydration": 1,
                "ready_after_full_run": 1,
                "blocked_by_hydration": 1,
            },
            "next_commands": ["python scripts/check_cloud_placeholders.py --summary-only"],
            "supporting_artifacts": {"meeting_packet": "artifacts/professor_meeting_packet_2020_2025.md"},
        }
    )

    assert "可直接发送版本" in markdown
    assert "12" in markdown
    assert "stable-ID audit" in markdown
    assert "check_cloud_placeholders.py" in markdown
    assert "professor_meeting_packet_2020_2025.md" in markdown
