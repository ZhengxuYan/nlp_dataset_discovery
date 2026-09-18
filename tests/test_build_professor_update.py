from scripts.build_professor_update import render_update


def test_render_update_includes_blocker_and_steps():
    markdown = render_update(
        {
            "placeholder_count": 12,
            "safe_to_execute_full_pipeline": False,
            "steps": [{"name": "preflight", "command_string": "python check.py"}],
        }
    )
    assert "Professor Update" in markdown
    assert "12" in markdown
    assert "python check.py" in markdown
    assert "local end-to-end smoke" in markdown
    assert "Safe to execute full pipeline now: `False`" in markdown
