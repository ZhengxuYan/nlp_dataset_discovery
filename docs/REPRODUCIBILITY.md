# Reproducibility

## Stage Order

1. Build the ACL and arXiv 2020-2025 paper catalogs.
2. Deduplicate arXiv versions/titles and remove ACL title overlap.
3. Classify papers that introduce datasets.
4. Extract structured dataset records from full text for positive papers.
5. Build the integrated dataset and DCU banks.
6. Retrieve prior-support DCUs and run added-information attribution.
7. Enrich stable paper/dataset identifiers with public metadata.
8. Generate aggregate tables, figures, audits, and the public release.

## Quality Gates

- Year bounds include 2020 and 2025 and exclude 2019.
- Paper IDs are unique after source-specific deduplication.
- ACL/arXiv overlap removal reports a nonzero ACL key set.
- Enrichment merges by stable record ID, never row position.
- Screening output coverage is checked against the input queue.
- Full-text extraction runs only on positive screening rows.
- Published data are scanned for secrets, absolute local paths, source text,
  and excluded evidence fields.

## Main Commands

Prepare the added years:

```bash
python scripts/prepare_main_analysis_expansion.py \
  --arxiv-catalog data/processed/arxiv_2020_2025_dedup_no_acl_for_dataset_screening_v2.jsonl \
  --acl-catalog data/census/acl_anthology/acl_anthology_2020_2025_all_with_abstracts.jsonl \
  --arxiv-pilot data/processed/arxiv_2020_2025_dataset_screening_stratified_3000_gemini_clean.jsonl
```

Build the public snapshot:

```bash
python scripts/build_public_release.py
```

Run focused tests without external pytest plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_prepare_main_analysis_expansion.py \
  tests/test_enrich_dataset_metadata.py \
  tests/test_build_public_release.py
```

LLM stages require provider credentials and incur provider-specific cost. The
public release builder performs only local processing and makes no API calls.
