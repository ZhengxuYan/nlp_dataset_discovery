import json

from scripts.run_local_smoke_2020_2025 import build_arg_parser, cached_api_payload, fixture_extractions, main


def test_fixture_extractions_have_dataset_and_acu():
    rows = fixture_extractions()
    assert len(rows) == 2
    assert rows[0]["year"] == 2020
    assert rows[0]["datasets"][0]["dataset_identity"]["canonical_name"] == "Local Smoke Dataset"
    assert rows[0]["datasets"][1]["dataset_identity"]["canonical_name"] == "Name Only Dataset"
    assert rows[1]["year"] == 2021
    assert rows[1]["datasets"][0]["dataset_identity"]["canonical_name"] == "Low Citation Dataset"
    assert rows[0]["datasets"][0]["acus"][0]["type"] == "scale"


def test_cached_api_payload_contains_expected_public_metadata():
    cache = cached_api_payload()
    assert any("openalex" in key for key in cache)
    assert cache["json:https://huggingface.co/api/datasets/org/name"]["data"]["downloads"] == 1234
    assert cache["json:https://huggingface.co/api/datasets?search=Name%20Only%20Dataset&limit=5"]["data"][0]["downloads"] == 777
    assert cache["json:https://api.github.com/repos/org/repo"]["data"]["stargazers_count"] == 99
    assert cache["json:https://api.github.com/repos/org/lowrepo"]["data"]["stargazers_count"] == 3
    assert cache["json:https://api.semanticscholar.org/graph/v1/paper/DOI%3A10.0000%2Flow-smoke?fields=paperId,title,year,venue,citationCount,influentialCitationCount,referenceCount,authors"]["data"]["citationCount"] == 0
    assert cache["health:https://aclanthology.org/2020.acl-main.1.pdf"]["downloadable"] is True


def test_arg_parser_accepts_output_dir(tmp_path):
    args = build_arg_parser().parse_args([str(tmp_path), "--root", str(tmp_path)])
    assert args.output_dir == tmp_path
    assert args.root == tmp_path


def test_local_smoke_manifest_includes_schema_audit(tmp_path):
    output_dir = tmp_path / "smoke"
    code = main([str(output_dir)])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    stable_id_audit = json.loads((output_dir / "enrichment_stable_id_audit_2020_2025.json").read_text(encoding="utf-8"))
    coverage = json.loads((output_dir / "metadata_coverage_2020_2025.json").read_text(encoding="utf-8"))
    review_sample = json.loads((output_dir / "metadata_review_sample_2020_2025.json").read_text(encoding="utf-8"))
    enriched_rows = [
        json.loads(line)
        for line in (output_dir / "integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    name_only = next(row for row in enriched_rows if row["dataset_name"] == "Name Only Dataset")
    low_citation = next(row for row in enriched_rows if row["dataset_name"] == "Low Citation Dataset")
    review_buckets = {row["bucket"] for row in review_sample["samples"]}

    assert code == 0
    assert stable_id_audit["status"] == "pass"
    assert stable_id_audit["input_rows"] == 3
    assert coverage["healthy_urls"] == coverage["total_urls"]
    assert coverage["coverage_rates"]["healthy_url_pct"] == 100.0
    assert name_only["public_metadata"]["hf_metadata"][0]["downloads"] == 777
    assert name_only["public_metadata"]["hf_metadata"][0]["match_method"] == "dataset_name_fuzzy"
    assert name_only["public_metadata"]["pwc_metadata"][0]["slug"] == "name-only-dataset"
    assert any(item["downloadable"] for item in name_only["public_metadata"]["resource_health"])
    assert low_citation["public_metadata"]["paper_metrics"]["citation_count"] == 0
    assert low_citation["public_metadata"]["github_metadata"][0]["stars"] == 3
    assert {"high_citation", "low_citation", "huggingface_name_fallback", "paperswithcode_matched", "healthy_url", "downloadable_url"} <= review_buckets
    assert "stable_id_audit_json" in manifest
    assert "stable_id_audit_md" in manifest
    assert "schema_audit_json" in manifest
    assert "schema_audit_md" in manifest
    assert "review_sample_jsonl" in manifest
    assert "review_sample_md" in manifest
