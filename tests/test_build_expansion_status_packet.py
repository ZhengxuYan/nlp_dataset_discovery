from scripts import build_expansion_status_packet


def test_build_packet_combines_status_inputs():
    packet = build_expansion_status_packet.build_packet(
        {"steps": [{"name": "a"}]},
        {"ready_for_full_pipeline": False, "blockers": ["hydration"]},
        {
            "placeholder_count": 10,
            "by_priority": {"high": 2},
            "local_smoke_rows": 1,
            "local_smoke_coverage_rates": {"citation_count_pct": 50.0},
        },
        {},
        {"artifact_count": 3, "missing_count": 0},
    )

    assert packet["status"] == "blocked_by_hydration"
    assert packet["run_plan_steps"] == 1
    assert packet["high_priority_placeholder_count"] == 2
    assert packet["coverage_rates"]["citation_count_pct"] == 50.0
    assert packet["missing_artifact_count"] == 0


def test_render_markdown_includes_next_action_and_artifacts():
    markdown = build_expansion_status_packet.render_markdown(
        {
            "generated_at": "now",
            "status": "blocked_by_hydration",
            "run_plan_steps": 11,
            "placeholder_count": 10,
            "high_priority_placeholder_count": 2,
            "artifact_count": 3,
            "missing_artifact_count": 0,
            "readiness_blockers": ["hydration"],
            "coverage_rates": {"citation_count_pct": 50.0},
            "primary_artifacts": {"chinese_brief": "artifacts/brief.md"},
        }
    )

    assert "# 2020-2025 Expansion Status Packet" in markdown
    assert "blocked_by_hydration" in markdown
    assert "Citation count: `50.0%`" in markdown
    assert "remaining_high_priority_hydration_files_2020_2025.txt" in markdown
    assert "artifacts/brief.md" in markdown
