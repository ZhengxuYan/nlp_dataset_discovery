# Project Status

Snapshot date: 2026-09-18.

## Completed

- The ACL Anthology catalog covers 2020-2025: 58,026 papers, including 53,927
  records with abstracts.
- The deduplicated arXiv screening catalog covers 2020-2025: 188,687 papers
  after ACL-overlap removal.
- Semantic Scholar enrichment matched 187,712 arXiv papers (99.48%). Matches
  use exact arXiv or DOI identifiers; citation counts are a 2026-08-28 snapshot.
- The current paper analysis covers 2023-2025: 22,133 papers, 22,874 dataset
  records, and 77,571 Dataset Contribution Units (DCUs).
- The citation-grounded retrieval benchmark contains 100 query dataset rows,
  128 processed prior papers, and 619 linked prior-support DCUs.

## In Progress

- Dataset-paper screening for the added 2020-2022 records is nearly complete,
  with failed batches being retried.
- Full-text extraction, prior-work construction, and DCU attribution have not
  yet been completed for 2020-2022.
- Therefore, the manuscript's substantive dataset/DCU conclusions still refer
  to 2023-2025. The 2020-2025 result currently established is corpus coverage
  and paper-metadata coverage, not the final longitudinal analysis.

## Release Interpretation

The public snapshot separates completed results from work in progress. Counts
in aggregate reports are authoritative for their named stage. Generated JSONL
files should not be interpreted as complete unless their coverage entry in
`data/public/release_manifest.json` is 100%.
