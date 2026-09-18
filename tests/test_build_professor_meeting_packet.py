from scripts import build_professor_meeting_packet


def test_build_packet_summarizes_meeting_evidence():
    packet = build_professor_meeting_packet.build_packet(
        status={
            "status": "blocked_by_hydration",
            "run_plan_steps": 12,
            "placeholder_count": 10,
            "high_priority_placeholder_count": 2,
            "artifact_count": 5,
            "missing_artifact_count": 0,
        },
        coverage={
            "rows": 3,
            "coverage_rates": {
                "citation_count_pct": 100.0,
                "openalex_id_pct": 100.0,
                "semantic_scholar_id_pct": 100.0,
                "hf_download_count_per_hf_resource_pct": 66.7,
                "github_star_count_per_github_resource_pct": 100.0,
                "healthy_url_pct": 100.0,
            },
        },
        schema={
            "present_values": {
                "paper_metadata_sources.match_confidence_score": 4,
                "hf_metadata.match_confidence_score": 2,
                "pwc_metadata.match_confidence_score": 1,
                "resource_health.status": 9,
                "resource_health.downloadable": 9,
            }
        },
        review={
            "sample_count": 2,
            "samples": [
                {"bucket": "high_citation"},
                {"bucket": "low_citation"},
                {"bucket": "huggingface_name_fallback"},
            ],
        },
        stable_id={"status": "pass"},
        integrated={"papers": 2, "datasets": 3, "acus": 3},
        full_plan={"steps": [{"name": "public_metadata_enrichment", "command": ["python", "--check-url-health"]}]},
    )

    assert packet["status"] == "blocked_by_hydration"
    assert packet["done_summary"]["full_plan_url_health_enabled"] is True
    assert packet["done_summary"]["local_smoke_datasets"] == 3
    assert packet["coverage_rates"]["hf_download_count_per_hf_resource_pct"] == 66.7
    assert "huggingface_name_fallback" in packet["review_buckets"]


def test_render_markdown_includes_next_action_and_key_metrics():
    packet = {
        "generated_at": "now",
        "status": "blocked_by_hydration",
        "placeholder_count": 10,
        "high_priority_placeholder_count": 2,
        "done_summary": {
            "run_plan_steps": 12,
            "artifact_count": 5,
            "missing_artifact_count": 0,
            "local_smoke_papers": 2,
            "local_smoke_datasets": 3,
            "local_smoke_acus": 3,
            "stable_id_audit_status": "pass",
            "full_plan_url_health_enabled": True,
            "review_sample_count": 17,
        },
        "coverage_rates": {"citation_count_pct": 100.0, "healthy_url_pct": 100.0},
        "schema_provenance_counts": {"resource_health_status": 9},
        "review_buckets": ["high_citation", "low_citation"],
        "artifacts": {"status_packet": "artifacts/status.md"},
    }

    markdown = build_professor_meeting_packet.render_markdown(packet)

    assert "# 教授 Meeting Packet" in markdown
    assert "2 篇 paper、3 个 dataset、3 个 ACU" in markdown
    assert "Citation count: `100.0%`" in markdown
    assert "run_post_hydration_expansion_sequence.py" in markdown
