# 2020-2025 Dataset Discovery Expansion Runbook

This runbook tracks the expansion from the current 2023-2025 dataset-discovery workflow to 2020-2025 and the added public metadata enrichment layer.

## Current Status

- Real pipeline entrypoints now accept 2020-2025 year/date parameters.
- 2020_2025 output paths are separate from existing 2023_2025 artifacts.
- Public metadata enrichment is implemented for paper identifiers/citations and dataset-resource signals.
- Title/name fuzzy paper matching now records confidence scores and rejects low-similarity fallbacks.
- Dataset-resource enrichment can fall back from missing URLs to scored dataset-name search for Hugging Face and Papers with Code.
- Full metadata enrichment checks URL health and records status/resolved/downloadable fields.
- Local end-to-end smoke is passing for integrated bank -> enrichment -> stable-ID audit -> coverage reporting.
- Full-data execution is blocked until cloud placeholder files are hydrated locally.

Check the current blocker:

```bash
python scripts/refresh_expansion_status_2020_2025.py
python scripts/validate_expansion_readiness.py --allow-blocked
python scripts/check_cloud_placeholders.py scripts scv scrapers data --summary-only
python scripts/build_hydration_manifest.py
python scripts/build_hydration_status_update.py
python scripts/check_cloud_placeholders.py --paths-file artifacts/high_priority_hydration_files_2020_2025.txt --summary-only
```

Full execution should wait until `placeholder_count=0`.

## Main Artifacts

- Professor update: `artifacts/professor_update_2020_2025.md`
- Professor meeting packet: `artifacts/professor_meeting_packet_2020_2025.md`
- Professor update brief: `artifacts/professor_update_brief_2020_2025.md`
- Chinese professor update brief: `artifacts/professor_update_brief_zh_2020_2025.md`
- Requirement checklist: `artifacts/expansion_plan_checklist_2020_2025.md`
- Completion audit: `artifacts/expansion_completion_audit_2020_2025.md`
- Hydration manifest: `artifacts/hydration_manifest_2020_2025.md`
- Hydration action guide: `artifacts/hydration_action_guide_2020_2025.md`
- Hydration status update: `artifacts/hydration_status_2020_2025.md`
- High-priority hydration file list: `artifacts/high_priority_hydration_files_2020_2025.txt`
- Remaining high-priority hydration queue: `artifacts/remaining_high_priority_hydration_queue_2020_2025.md`
- Expansion regression audit: `artifacts/expansion_regression_audit_2020_2025.md`
- Smoke run plan: `artifacts/corpus_expansion_2020_2025_run_plan.json`
- Full run plan: `artifacts/corpus_expansion_2020_2025_full_run_plan.json`
- 2020-2021 staged smoke plan: `artifacts/corpus_expansion_2020_2021_smoke_run_plan.json`
- Local smoke coverage: `artifacts/local_smoke_2020_2025/metadata_coverage_2020_2025.md`
- Local smoke stable-ID audit: `artifacts/local_smoke_2020_2025/enrichment_stable_id_audit_2020_2025.md`
- Local smoke schema audit: `artifacts/local_smoke_2020_2025/metadata_schema_audit_2020_2025.md`
- Local smoke metadata review sample: `artifacts/local_smoke_2020_2025/metadata_review_sample_2020_2025.md`
- Technical progress report: `artifacts/progress_update_2020_2025.md`
- Refresh summary: `artifacts/refresh_expansion_status_2020_2025.json`
- Full-run readiness gate: `artifacts/expansion_readiness_2020_2025.md`
- Post-hydration sequence summary: `artifacts/post_hydration_expansion_sequence_2020_2025.json`
- Artifact index: `artifacts/expansion_artifact_index_2020_2025.md`
- One-page status packet: `artifacts/expansion_status_packet_2020_2025.md`

## Safe Smoke Commands

Generate the planned smoke/full commands without running heavy jobs:

```bash
python scripts/run_corpus_expansion_2020_2025.py --mode smoke
python scripts/run_corpus_expansion_2020_2025.py --start-year 2020 --end-year 2021 --mode smoke --plan-output artifacts/corpus_expansion_2020_2021_smoke_run_plan.json
```

Run the local fixture-based end-to-end smoke:

```bash
python scripts/run_local_smoke_2020_2025.py artifacts/local_smoke_2020_2025
```

This local smoke does not require network access or cloud-hydrated data. It validates:

- integrated fulltext dataset bank creation
- ACU bank creation
- cached/offline OpenAlex and Semantic Scholar citation metadata
- exact DOI/arXiv matching first, with scored title fuzzy fallback when identifiers are missing
- cached/offline Hugging Face download metadata
- name-based Hugging Face/Papers with Code fallback when direct dataset URLs are missing
- cached/offline GitHub star/fork metadata
- cached/offline URL health status, resolved URL, and downloadable checks
- stable-ID preservation between original and enriched dataset banks
- metadata coverage reporting
- metadata review sample for manual high/low citation and HF/GitHub-linked checks

## Full Pipeline After Hydration

After `placeholder_count=0`, run:

```bash
python scripts/run_post_hydration_expansion_sequence.py --execute --allow-network-steps
```

Or run the full orchestrator directly:

```bash
python scripts/run_corpus_expansion_2020_2025.py --mode full --execute --allow-network-steps
```

The orchestrator records the exact steps and refuses execution while placeholders remain unless `--ignore-placeholders` is explicitly passed.
It also appends the post-run metadata coverage report, metadata schema audit, and expansion regression audit to the run plan.
It audits enrichment with stable paper/dataset record IDs so metadata is not trusted based on row position.

After the full run, check that the 2023-2025 overlap did not shrink:

```bash
python scripts/audit_expansion_regression.py
```

## Individual Entry Points

ACL Anthology catalog:

```bash
python scripts/build_acl_anthology_catalog.py --start-year 2020 --end-year 2025 --scope all
```

arXiv interval scraper:

```bash
python scrapers/arxiv_scraper/run_arxiv_intervals.py --start-date 2020-01-01 --end-date 2025-12-31
```

arXiv screening catalog:

```bash
python scripts/prepare_arxiv_screening_catalog.py --start-year 2020 --end-year 2025
```

Metadata enrichment:

```bash
python scripts/enrich_dataset_metadata.py \
  --input data/census/integrated_fulltext_dataset_bank_2020_2025.jsonl \
  --output data/census/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl \
  --summary-output data/census/integrated_fulltext_dataset_bank_2020_2025_enriched_summary.json \
  --cache data/cache/public_metadata_api_cache.json \
  --check-url-health
```

Coverage report:

```bash
python scripts/audit_enrichment_stable_ids.py \
  --input-jsonl data/census/integrated_fulltext_dataset_bank_2020_2025.jsonl \
  --enriched-jsonl data/census/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl \
  --output-json artifacts/enrichment_stable_id_audit_2020_2025.json \
  --output-md artifacts/enrichment_stable_id_audit_2020_2025.md

python scripts/summarize_metadata_coverage.py \
  --input-jsonl data/census/integrated_fulltext_dataset_bank_2020_2025_enriched.jsonl \
  --output-json artifacts/metadata_coverage_2020_2025.json \
  --output-md artifacts/metadata_coverage_2020_2025.md
```

## Verification

Focused verification used for this work:

```bash
python -m pytest \
  tests/test_corpus_expansion.py \
  tests/test_enrich_dataset_metadata.py \
  tests/test_build_progress_update.py \
  tests/test_check_cloud_placeholders.py \
  tests/test_pipeline_year_parameterization.py \
  tests/test_run_corpus_expansion_2020_2025.py \
  tests/test_summarize_metadata_coverage.py \
  tests/test_audit_enrichment_stable_ids.py \
  tests/test_build_professor_update.py \
  tests/test_build_professor_update_brief.py \
  tests/test_build_expansion_checklist.py \
  tests/test_build_expansion_artifact_index.py \
  tests/test_build_expansion_status_packet.py \
  tests/test_run_local_smoke_2020_2025.py \
  tests/test_refresh_expansion_status_2020_2025.py \
  tests/test_run_post_hydration_expansion_sequence.py \
  tests/test_validate_expansion_readiness.py \
  tests/test_build_hydration_action_guide.py \
  tests/test_build_remaining_hydration_queue.py \
  tests/test_build_hydration_manifest.py \
  tests/test_build_hydration_status_update.py \
  tests/test_audit_expansion_regression.py \
  tests/test_audit_metadata_schema.py \
  -q
```
