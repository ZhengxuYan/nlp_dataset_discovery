from pathlib import Path

from scripts import validate_expansion_readiness


def test_validate_ready_when_all_gates_clear(tmp_path: Path, monkeypatch):
    manifest_paths = {}
    for key in validate_expansion_readiness.REQUIRED_LOCAL_SMOKE_KEYS:
        path = tmp_path / f"{key}.txt"
        path.write_text("ok", encoding="utf-8")
        manifest_paths[key] = str(path)
    monkeypatch.setattr(validate_expansion_readiness, "count_placeholders", lambda paths: 0)

    readiness = validate_expansion_readiness.validate(
        {"steps": [{"name": name} for name in validate_expansion_readiness.REQUIRED_STEPS]},
        [],
        manifest_paths,
        {"ok": True},
        [],
    )

    assert readiness["ready_for_full_pipeline"] is True
    assert readiness["blockers"] == []


def test_validate_blocks_on_placeholders_and_missing_steps(monkeypatch):
    monkeypatch.setattr(validate_expansion_readiness, "count_placeholders", lambda paths: 3 if paths else 0)

    readiness = validate_expansion_readiness.validate(
        {"steps": [{"name": "preflight_placeholder_check"}]},
        [Path("placeholder")],
        {},
        {"ok": False},
        [Path("data")],
    )

    assert readiness["ready_for_full_pipeline"] is False
    assert "run_plan_missing_required_steps" in readiness["blockers"]
    assert "high_priority_placeholders_remaining" in readiness["blockers"]
    assert "local_smoke_artifacts_missing" in readiness["blockers"]
    assert "refresh_summary_missing_or_failed" in readiness["blockers"]


def test_render_markdown_includes_next_commands():
    markdown = validate_expansion_readiness.render_markdown(
        {
            "generated_at": "now",
            "ready_for_full_pipeline": False,
            "run_plan_step_count": 10,
            "high_priority_placeholder_count": 1,
            "full_placeholder_count": 2,
            "refresh_ok": True,
            "blockers": ["high_priority_placeholders_remaining"],
            "local_smoke_artifacts": {"dataset_bank": True},
        }
    )

    assert "# Expansion Readiness" in markdown
    assert "high_priority_placeholders_remaining" in markdown
    assert "run_corpus_expansion_2020_2025.py --mode full" in markdown
