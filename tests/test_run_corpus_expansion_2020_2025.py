from pathlib import Path

from scripts.corpus_expansion import YearRange
from scripts.run_corpus_expansion_2020_2025 import build_steps, plan_payload


def test_build_steps_smoke_uses_dry_run_and_2020_2025_paths(tmp_path: Path):
    steps = build_steps(tmp_path, YearRange(2020, 2025), "smoke")
    by_name = {step.name: step for step in steps}
    assert "--download-bib-if-needed" in by_name["acl_anthology_catalog_all"].command
    assert "--dry-run" in by_name["arxiv_interval_scrape"].command
    assert "--output-file" in by_name["arxiv_interval_scrape"].command
    assert "arxiv_results_2020_2025.csv" in " ".join(by_name["arxiv_interval_scrape"].command)
    assert "--limit" in by_name["fulltext_dataset_extraction"].command
    assert "integrated_fulltext_dataset_bank_2020_2025.jsonl" in " ".join(
        by_name["integrated_fulltext_banks"].command
    )
    assert "integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl" in " ".join(
        by_name["public_metadata_enrichment"].command
    )
    assert "enrichment_stable_id_audit_2020_2025.md" in " ".join(
        by_name["enrichment_stable_id_audit"].command
    )
    assert "metadata_coverage_2020_2025.md" in " ".join(by_name["metadata_coverage_report"].command)
    assert "metadata_schema_audit_2020_2025.md" in " ".join(by_name["metadata_schema_audit"].command)
    assert "metadata_review_sample_2020_2025.md" in " ".join(by_name["metadata_review_sample"].command)
    assert "expansion_regression_audit_2020_2025.md" in " ".join(
        by_name["expansion_regression_audit"].command
    )
    assert "--check-url-health" not in by_name["public_metadata_enrichment"].command


def test_build_steps_full_checks_resource_url_health(tmp_path: Path):
    steps = build_steps(tmp_path, YearRange(2020, 2025), "full")
    by_name = {step.name: step for step in steps}
    assert "--check-url-health" in by_name["public_metadata_enrichment"].command


def test_plan_payload_marks_placeholders_unsafe(tmp_path: Path):
    steps = build_steps(tmp_path, YearRange(2020, 2025), "smoke")
    payload = plan_payload(tmp_path, YearRange(2020, 2025), "smoke", steps, placeholders=3)
    assert payload["safe_to_execute_full_pipeline"] is False
    assert payload["placeholder_count"] == 3
    assert payload["steps"][0]["name"] == "preflight_placeholder_check"
    assert payload["steps"][-5]["name"] == "enrichment_stable_id_audit"
    assert payload["steps"][-4]["name"] == "metadata_coverage_report"
    assert payload["steps"][-3]["name"] == "metadata_schema_audit"
    assert payload["steps"][-2]["name"] == "metadata_review_sample"
    assert payload["steps"][-1]["name"] == "expansion_regression_audit"
