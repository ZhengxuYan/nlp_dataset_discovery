from pathlib import Path

from scripts import build_hydration_manifest


def test_priority_for_marks_cache_low_and_data_high():
    assert build_hydration_manifest.priority_for("data/processed/file.jsonl") == "high"
    assert build_hydration_manifest.priority_for("data/benchmark/retrieval_cache/file.json") == "low"
    assert build_hydration_manifest.priority_for("scrapers/arxiv_scraper/.scrapy/httpcache/file") == "low"
    assert build_hydration_manifest.priority_for("other/file.txt") == "medium"


def test_build_manifest_counts_mocked_placeholders(tmp_path: Path, monkeypatch):
    high = tmp_path / "data" / "processed" / "a.jsonl"
    low = tmp_path / "data" / "benchmark" / "retrieval_cache" / "b.json"
    high.parent.mkdir(parents=True)
    low.parent.mkdir(parents=True)
    high.write_text("x", encoding="utf-8")
    low.write_text("x", encoding="utf-8")
    monkeypatch.setattr(build_hydration_manifest, "is_cloud_placeholder", lambda path: path in {high, low})
    manifest = build_hydration_manifest.build_manifest(tmp_path, [tmp_path / "data"], sample_limit=2)
    assert manifest["placeholder_count"] == 2
    assert manifest["by_priority"]["high"] == 1
    assert manifest["by_priority"]["low"] == 1
    assert manifest["priority_files"]["high"] == ["data/processed/a.jsonl"]
    assert "data/processed/a.jsonl" in manifest["priority_samples"]["high"]


def test_render_markdown_includes_suggested_next_step():
    manifest = {
        "generated_at": "now",
        "placeholder_count": 1,
        "placeholder_bytes": 10,
        "by_priority": {"high": 1},
        "by_root": {"data": 1},
        "by_suffix": {".jsonl": 1},
        "by_parent_top": {"data/processed": 1},
        "priority_samples": {"high": ["data/processed/a.jsonl"]},
    }
    markdown = build_hydration_manifest.render_markdown(manifest)
    assert "# Hydration Manifest" in markdown
    assert "Suggested Next Step" in markdown
    assert "high_priority_hydration_files_2020_2025.txt" in markdown
