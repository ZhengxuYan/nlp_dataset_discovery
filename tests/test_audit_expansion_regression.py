import json
from pathlib import Path

from scripts import audit_expansion_regression
from scripts.corpus_expansion import YearRange


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compare_artifact_passes_when_overlap_is_preserved():
    comparison = audit_expansion_regression.compare_artifact(
        {"status": "counted", "overlap_rows": 2},
        {"status": "counted", "overlap_rows": 3},
    )

    assert comparison["status"] == "pass"


def test_compare_artifact_fails_when_overlap_shrinks():
    comparison = audit_expansion_regression.compare_artifact(
        {"status": "counted", "overlap_rows": 3},
        {"status": "counted", "overlap_rows": 2},
    )

    assert comparison["status"] == "fail_shrunk_overlap"


def test_compare_artifact_marks_missing_new_as_pending():
    comparison = audit_expansion_regression.compare_artifact(
        {"status": "counted", "overlap_rows": 3},
        {"status": "missing"},
    )

    assert comparison["status"] == "pending_new_artifact"


def test_build_audit_compares_overlap_years(tmp_path: Path):
    old_range = YearRange(2023, 2025)
    new_range = YearRange(2020, 2025)
    old_path = tmp_path / "data" / "raw" / "arxiv_results_2023_2025.csv"
    new_path = tmp_path / "data" / "raw" / "arxiv_results_2020_2025.csv"
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_text("year,title\n2023,A\n2024,B\n", encoding="utf-8")
    new_path.write_text("year,title\n2020,Z\n2023,A\n2024,B\n2025,C\n", encoding="utf-8")

    audit = audit_expansion_regression.build_audit(tmp_path, old_range, new_range)

    raw_catalog = audit["artifacts"]["raw_catalog"]
    assert raw_catalog["old"]["overlap_rows"] == 2
    assert raw_catalog["new"]["overlap_rows"] == 3
    assert raw_catalog["comparison"]["status"] == "pass"


def test_audit_file_skips_cloud_placeholders(tmp_path: Path, monkeypatch):
    path = tmp_path / "placeholder.jsonl"
    write_jsonl(path, [{"year": 2023}])
    monkeypatch.setattr(audit_expansion_regression, "is_cloud_placeholder", lambda candidate: candidate == path)

    result = audit_expansion_regression.audit_file(path, YearRange(2023, 2025))

    assert result["status"] == "cloud_placeholder"


def test_render_markdown_includes_interpretation():
    markdown = audit_expansion_regression.render_markdown(
        {
            "generated_at": "now",
            "old_range": {"label": "2023_2025"},
            "new_range": {"label": "2020_2025"},
            "overlap_range": {"label": "2023_2025"},
            "summary": {"pass": 1},
            "artifacts": {
                "raw_catalog": {
                    "comparison": {
                        "status": "pass",
                        "old_overlap_rows": 2,
                        "new_overlap_rows": 3,
                        "old_status": "counted",
                        "new_status": "counted",
                    }
                }
            },
        }
    )

    assert "# Expansion Regression Audit" in markdown
    assert "fail_shrunk_overlap" in markdown
