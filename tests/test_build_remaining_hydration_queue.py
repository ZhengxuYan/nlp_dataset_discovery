from pathlib import Path

from scripts import build_remaining_hydration_queue


def test_build_queue_groups_only_remaining_placeholders(tmp_path: Path, monkeypatch):
    remaining = tmp_path / "data" / "census" / "a.jsonl"
    local = tmp_path / "data" / "census" / "b.jsonl"
    other = tmp_path / "data" / "processed" / "c.csv"
    for path in (remaining, local, other):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        build_remaining_hydration_queue,
        "is_cloud_placeholder",
        lambda path: path in {remaining, other},
    )

    queue = build_remaining_hydration_queue.build_queue([remaining, local, other])

    assert queue["remaining_count"] == 2
    assert queue["by_parent"][str(remaining.parent)] == 1
    assert queue["by_parent"][str(other.parent)] == 1
    assert local.as_posix() not in queue["remaining_files"]


def test_render_markdown_includes_verification_commands():
    markdown = build_remaining_hydration_queue.render_markdown(
        {
            "generated_at": "now",
            "remaining_count": 1,
            "by_parent": {"data/census": 1},
            "samples_by_parent": {"data/census": ["data/census/a.jsonl"]},
        },
        Path("artifacts/high_priority_hydration_files_2020_2025.txt"),
    )

    assert "# Remaining High-Priority Hydration Queue" in markdown
    assert "data/census" in markdown
    assert "remaining_high_priority_hydration_files_2020_2025.txt" in markdown
