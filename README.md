# NLP Dataset Discovery and Dataset Contribution Units

This repository contains the code and release artifacts for studying how NLP
dataset papers describe new resources and how their claims relate to prior
dataset contributions.

The project has two connected components:

1. A reproducible ACL Anthology and arXiv corpus pipeline for discovering
   dataset-introducing papers, extracting dataset records, and enriching public
   paper/resource metadata.
2. A Dataset Contribution Unit (DCU) pipeline for retrieving prior evidence and
   measuring added information at the claim level.

## Current Scope

- Completed paper catalog: ACL Anthology and arXiv, 2020-2025.
- arXiv screening catalog: 188,687 deduplicated papers after ACL overlap removal.
- Semantic Scholar enrichment: 187,712 of 188,687 arXiv papers matched (99.48%).
- Completed main-analysis bank currently used by the draft paper: 2023-2025,
  containing 22,133 papers, 22,874 dataset records, and 77,571 DCUs.
- In progress: extending dataset-paper screening, full-text extraction, and DCU
  attribution so the main conclusions cover 2020-2025.

The release status and known limitations are documented in
[`docs/STATUS.md`](docs/STATUS.md).

## Repository Layout

```text
scrapers/   ACL/arXiv/OpenReview collection code
scripts/    screening, extraction, enrichment, audit, and analysis entrypoints
scv/        retrieval and evidence-attribution library
tests/      focused unit and pipeline tests
paper/      manuscript source, figures, bibliography, and compiled draft
data/public compact release data; other large local data are ignored by Git
artifacts/  local logs and reports (ignored except selected release summaries)
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

LLM-backed stages require a configured provider key. Public metadata and local
audit stages can be run independently.

## Reproduce the Public-Metadata Result

```bash
python scripts/finalize_paper_enrichment.py \
  --input data/processed/arxiv_2020_2025_dedup_no_acl_for_dataset_screening_v2.jsonl \
  --prefix artifacts/arxiv_paper_metadata_s2_batch_verified_000000_012250.jsonl \
  --shard-glob 'data/processed/arxiv_2020_2025_paper_metadata_s2_batch_enriched_*.jsonl' \
  --output data/processed/arxiv_2020_2025_paper_metadata_s2_enriched_final.jsonl \
  --summary-json artifacts/arxiv_2020_2025_paper_metadata_s2_enriched_final_summary.json \
  --summary-md artifacts/arxiv_2020_2025_paper_metadata_s2_enriched_final_summary.md
```

## Prepare the 2020-2025 Main-Analysis Expansion

```bash
python scripts/prepare_main_analysis_expansion.py \
  --arxiv-catalog data/processed/arxiv_2020_2025_dedup_no_acl_for_dataset_screening_v2.jsonl \
  --acl-catalog data/census/acl_anthology/acl_anthology_2020_2025_all_with_abstracts.jsonl \
  --arxiv-pilot data/processed/arxiv_2020_2025_dataset_screening_stratified_3000_gemini_clean.jsonl
```

This produces resumable 2020-2022 queues while preserving the completed
2023-2025 dataset/DCU bank. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
for the end-to-end stage order and quality gates.

## Tests

```bash
python -m pytest -q
```

## Data Release Policy

Large raw files, paper full text, API caches, credentials, and model logs are not
committed. The public release package contains compact IDs/labels, sanitized
dataset records, aggregate reports, schemas, and checksums. Source-specific
licenses and redistribution constraints are described in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Paper

The current manuscript is a working draft. Its present main analysis covers
2023-2025, with 2020-2025 corpus construction and metadata enrichment reported
separately. The manuscript will be updated after the full 2020-2025 extraction
and attribution run is complete.

## Citation and License

Citation metadata and a software license will be finalized with the paper
release. Until a license is added, the repository is available for inspection
and collaboration, but no additional reuse rights are granted by default.
