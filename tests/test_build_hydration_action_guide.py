from scripts import build_hydration_action_guide


def test_render_guide_includes_counts_and_commands():
    markdown = build_hydration_action_guide.render_guide(
        {
            "placeholder_count": 10,
            "by_priority": {"high": 2, "medium": 3, "low": 5},
        }
    )

    assert "# Hydration Action Guide" in markdown
    assert "Total placeholder files: `10`" in markdown
    assert "High-priority placeholders: `2`" in markdown
    assert "high_priority_hydration_files_2020_2025.txt" in markdown
    assert "remaining_high_priority_hydration_queue_2020_2025.md" in markdown
    assert "check_cloud_placeholders.py --paths-file" in markdown
    assert "--start-year 2020 --end-year 2021 --mode smoke" in markdown
    assert "--mode full --execute --allow-network-steps" in markdown
