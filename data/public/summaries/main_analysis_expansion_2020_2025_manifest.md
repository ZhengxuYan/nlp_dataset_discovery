# 2020-2025 Main Analysis Expansion

- Generated: 2026-09-17T04:45:17+00:00
- Incremental query years: 2020-2022
- Existing 2023-2025 dataset/DCU bank is preserved and reused.
- A pre-2020 prior-only DCU bank is required before scoring 2020 queries with earlier-year evidence.

## Screening Queues

| Source | Rows | Existing seed rows | Remaining before new calls | Abstract coverage |
| --- | ---: | ---: | ---: | ---: |
| arXiv 2020-2022 | 64,710 | 1,500 | 63,210 | 100.0% |
| ACL 2020-2022 | 22,538 | 0 | 22,538 | 92.4% |

## Execution Order

1. Human-audit the stratified screening sample and freeze the classifier prompt/model.
2. Run resumable abstract screening on the two incremental queues.
3. Build full-text queues from dataset-introducing positives and run extraction.
4. Merge 2020-2022 extraction records with the existing 2023-2025 dataset/DCU bank.
5. Build a pre-2020 prior-only DCU bank, then run year-consistent attribution for 2020-2025.
6. Rebuild all corpus tables, trend figures, significance analyses, abstract, results, and limitations.

## Resumable Screening Commands

The arXiv output is pre-seeded with completed pilot rows. Both commands skip IDs already present.

```bash
python scripts/run_dataset_census_classifier.py --catalog data/processed/arxiv_2020_2022_for_dataset_screening.jsonl --output-jsonl data/processed/arxiv_2020_2022_dataset_screening_gemini31_flashlite.jsonl --error-jsonl data/processed/arxiv_2020_2022_dataset_screening_gemini31_flashlite.errors.jsonl --backend gemini --model gemini-3.1-flash-lite --batch-size 10 --workers 2 --max-retries 3 --sleep 0.2

python scripts/run_dataset_census_classifier.py --catalog data/processed/acl_2020_2022_for_dataset_screening.jsonl --output-jsonl data/processed/acl_2020_2022_dataset_screening_gemini31_flashlite.jsonl --error-jsonl data/processed/acl_2020_2022_dataset_screening_gemini31_flashlite.errors.jsonl --backend gemini --model gemini-3.1-flash-lite --batch-size 10 --workers 2 --max-retries 3 --sleep 0.2
```
