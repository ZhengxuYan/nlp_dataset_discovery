from scripts import build_next_action_handoff


def test_build_handoff_points_to_hydration_when_placeholders_block():
    handoff = build_next_action_handoff.build_handoff(
        {"ready_for_full_pipeline": False, "blockers": ["high_priority_placeholders_remaining"]},
        {
            "status": "blocked_by_hydration",
            "placeholder_count": 10,
            "high_priority_placeholder_count": 2,
            "artifact_count": 4,
            "missing_artifact_count": 0,
            "run_plan_steps": 12,
        },
        {"complete": False},
        {"steps": [{"name": "refresh"}]},
    )

    assert handoff["current_gate"] == "hydrate_cloud_placeholders"
    assert handoff["high_priority_placeholder_count"] == 2
    assert handoff["post_hydration_step_count"] == 1
    assert any("check_cloud_placeholders.py" in command for command in handoff["next_commands"])


def test_build_handoff_points_to_full_pipeline_when_ready():
    handoff = build_next_action_handoff.build_handoff(
        {"ready_for_full_pipeline": True, "blockers": []},
        {"status": "ready_for_full_pipeline"},
        {"complete": False},
        {"steps": []},
    )

    assert handoff["current_gate"] == "run_full_pipeline"
    assert any("run_post_hydration_expansion_sequence.py" in command for command in handoff["next_commands"])


def test_render_markdown_includes_gate_commands_and_success_condition():
    markdown = build_next_action_handoff.render_markdown(
        {
            "generated_at": "now",
            "current_gate": "hydrate_cloud_placeholders",
            "status": "blocked_by_hydration",
            "ready_for_full_pipeline": False,
            "completion_audit_complete": False,
            "placeholder_count": 10,
            "high_priority_placeholder_count": 2,
            "artifact_count": 4,
            "missing_artifact_count": 0,
            "run_plan_steps": 12,
            "blockers": ["high_priority_placeholders_remaining"],
            "next_commands": ["python scripts/check_cloud_placeholders.py --summary-only"],
            "success_condition": "Readiness changes to ready_for_full_pipeline.",
            "handoff_artifacts": {"readiness_gate": "artifacts/expansion_readiness_2020_2025.md"},
        }
    )

    assert "# 2020-2025 Expansion Next Action Handoff" in markdown
    assert "hydrate_cloud_placeholders" in markdown
    assert "check_cloud_placeholders.py" in markdown
    assert "Success Condition" in markdown
