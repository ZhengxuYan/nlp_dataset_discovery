from pathlib import Path

from scripts.build_progress_update import render_markdown
from scripts.corpus_expansion import YearRange, output_paths


def test_render_markdown_contains_update_sections(tmp_path: Path):
    year_range = YearRange(2020, 2025)
    statuses = {
        key: {"path": str(path), "exists": False, "cloud_placeholder": False, "size_bytes": None}
        for key, path in output_paths(tmp_path, year_range).items()
    }
    markdown = render_markdown(
        tmp_path,
        year_range,
        {str(tmp_path / "scripts"): 2, str(tmp_path / "data"): 3},
        statuses,
    )
    assert "# 2020-2025 Dataset Discovery Progress Update" in markdown
    assert "public metadata enrichment" in markdown
    assert "Parameterized real pipeline entrypoints" in markdown
    assert "safe orchestrator" in markdown
    assert "coverage reporting" in markdown
    assert "evidence-backed checklist" in markdown
    assert "local end-to-end smoke" in markdown
    assert "consolidated repository runbook" in markdown
    assert "hydration manifest" in markdown
    assert "high_priority_hydration_files_2020_2025.txt" in markdown
    assert "run_corpus_expansion_2020_2025.py" in markdown
    assert "run_local_smoke_2020_2025.py" in markdown
    assert "build_expansion_checklist.py" in markdown
    assert "summarize_metadata_coverage.py" in markdown
    assert "5 cloud placeholder files" in markdown
    assert "--summary-only" in markdown
    assert "integrated_dataset_bank_jsonl" in markdown
