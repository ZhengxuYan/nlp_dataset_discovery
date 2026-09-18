from pathlib import Path

from scripts import hydrate_cloud_placeholders_macos


def test_build_plan_dry_run_targets_only_placeholders(tmp_path, monkeypatch):
    placeholder = tmp_path / "placeholder.py"
    hydrated = tmp_path / "hydrated.py"
    placeholder.write_text("placeholder", encoding="utf-8")
    hydrated.write_text("hydrated", encoding="utf-8")
    monkeypatch.setattr(
        hydrate_cloud_placeholders_macos,
        "is_cloud_placeholder",
        lambda path: Path(path).name == "placeholder.py",
    )

    plan = hydrate_cloud_placeholders_macos.build_plan([placeholder, hydrated], execute=False)

    assert plan["execute"] is False
    assert plan["input_count"] == 2
    assert plan["target_count"] == 1
    assert plan["target_files"] == [placeholder.as_posix()]
    assert plan["download_results"] == []


def test_build_plan_execute_reports_missing_downloader(tmp_path, monkeypatch):
    placeholder = tmp_path / "placeholder.py"
    placeholder.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(hydrate_cloud_placeholders_macos, "is_cloud_placeholder", lambda path: True)
    monkeypatch.setattr(hydrate_cloud_placeholders_macos, "find_downloader", lambda downloader=None: None)

    plan = hydrate_cloud_placeholders_macos.build_plan([placeholder], execute=True)

    assert plan["execute"] is True
    assert plan["error"] == "brctl_not_found"
    assert plan["download_results"] == []


def test_render_markdown_includes_dry_run_execute_and_verify_commands():
    markdown = hydrate_cloud_placeholders_macos.render_markdown(
        {
            "generated_at": "now",
            "execute": False,
            "input_count": 1,
            "existing_unique_count": 1,
            "target_count": 1,
            "unlimited_placeholder_count": 1,
            "remaining_target_placeholders": 1,
            "downloader_command": ["/usr/bin/brctl", "download"],
            "error": None,
            "target_files": ["scripts/example.py"],
        },
        Path("artifacts/remaining_high_priority_hydration_files_2020_2025.txt"),
    )

    assert "# macOS Hydration Helper Plan" in markdown
    assert "--execute" in markdown
    assert "check_cloud_placeholders.py" in markdown
    assert "scripts/example.py" in markdown
