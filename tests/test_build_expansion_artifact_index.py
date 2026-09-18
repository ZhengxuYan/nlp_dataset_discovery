from scripts import build_expansion_artifact_index


def test_build_index_tracks_required_artifacts(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "README_EXPANSION_2020_2025.md").write_text("runbook", encoding="utf-8")
    (tmp_path / "artifacts" / "professor_update_brief_zh_2020_2025.md").write_text("brief", encoding="utf-8")

    index = build_expansion_artifact_index.build_index(tmp_path)
    keys = {record["key"] for record in index["artifacts"]}

    assert "professor_brief_zh" in keys
    assert "runbook" in keys
    assert "readiness_gate" in keys
    assert index["artifact_count"] == len(build_expansion_artifact_index.ARTIFACTS)
    assert index["missing_count"] >= 1


def test_render_markdown_includes_table():
    index = {
        "generated_at": "now",
        "artifact_count": 1,
        "missing_count": 0,
        "artifacts": [
            {
                "key": "runbook",
                "description": "Expansion runbook",
                "exists": True,
                "path": "README_EXPANSION_2020_2025.md",
            }
        ],
    }

    markdown = build_expansion_artifact_index.render_markdown(index)

    assert "# 2020-2025 Expansion Artifact Index" in markdown
    assert "| `runbook` |" in markdown
