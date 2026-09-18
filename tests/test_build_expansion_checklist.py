from scripts.build_expansion_checklist import build_items, render_markdown


def test_build_items_marks_full_run_blocked_when_placeholders_remain():
    items = build_items(
        {
            "placeholder_count": 5,
            "steps": [
                {"name": "preflight_placeholder_check"},
                {"name": "public_metadata_enrichment"},
            ],
        }
    )
    statuses = {item["requirement"]: item["status"] for item in items}
    assert statuses["Run full 2020-2025 pipeline and produce final expanded artifacts"] == "blocked_by_hydration"
    assert statuses["Provide smoke/full execution plan"] == "done"
    assert statuses["Run local end-to-end smoke for integrated bank and metadata enrichment"] == "done"


def test_render_markdown_contains_status_table():
    items = build_items({"placeholder_count": 0, "safe_to_execute_full_pipeline": True, "steps": []})
    markdown = render_markdown(items, {"placeholder_count": 0, "safe_to_execute_full_pipeline": True})
    assert "# 2020-2025 Expansion Plan Checklist" in markdown
    assert "| Requirement | Status | Evidence |" in markdown
    assert "ready_to_run" in markdown
