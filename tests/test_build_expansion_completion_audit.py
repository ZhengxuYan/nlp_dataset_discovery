from scripts import build_expansion_completion_audit


def test_build_audit_classifies_requirements_and_keeps_completion_false():
    audit = build_expansion_completion_audit.build_audit(
        checklist={
            "items": [
                {"requirement": "done req", "status": "done", "evidence": ["a"]},
                {"requirement": "smoke req", "status": "ready_after_hydration", "evidence": ["b"]},
                {"requirement": "full req", "status": "ready_after_full_run", "evidence": ["c"]},
                {"requirement": "blocked req", "status": "blocked_by_hydration", "evidence": ["d"]},
            ]
        },
        readiness={
            "ready_for_full_pipeline": False,
            "blockers": ["hydration"],
            "full_placeholder_count": 10,
            "high_priority_placeholder_count": 2,
        },
        status_packet={"status": "blocked_by_hydration"},
    )

    assert audit["complete"] is False
    assert audit["completion_counts"]["proved"] == 1
    assert audit["completion_counts"]["pending_after_hydration"] == 1
    assert audit["completion_counts"]["pending_after_full_run"] == 1
    assert audit["completion_counts"]["blocked_by_hydration"] == 1
    assert audit["blockers"] == ["hydration"]


def test_render_markdown_includes_requirements_and_blockers():
    audit = build_expansion_completion_audit.build_audit(
        checklist={"items": [{"requirement": "done req", "status": "done", "evidence": ["a"]}]},
        readiness={"ready_for_full_pipeline": True, "blockers": []},
        status_packet={"status": "ready_for_full_pipeline"},
    )

    markdown = build_expansion_completion_audit.render_markdown(audit)

    assert "# 2020-2025 Expansion Completion Audit" in markdown
    assert "done req" in markdown
    assert "proved" in markdown
    assert "Complete: `True`" in markdown
