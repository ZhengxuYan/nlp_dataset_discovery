# Data Card

## Scope

This release describes NLP and adjacent dataset papers collected from ACL
Anthology and selected arXiv computer-science categories for 2020-2025. It also
contains the structured 2023-2025 dataset bank used by the current manuscript.

## Public Files

- `acl_catalog_2020_2025.jsonl.gz`: ACL identifiers and bibliographic metadata;
  abstracts are excluded.
- `arxiv_catalog_2020_2025.jsonl.gz`: arXiv identifiers and bibliographic
  metadata after title deduplication and ACL-overlap removal; abstracts are
  excluded.
- `screening_labels_2020_2022.jsonl.gz`: model-generated paper-level screening
  labels available at release time. Consult the manifest before treating these
  as complete.
- `dataset_bank_2023_2025.jsonl.gz`: structured dataset records with quoted
  evidence, full-text search fields, and per-claim DCUs removed.
- `summaries/`: aggregate corpus, enrichment, validation, and paper-result
  reports.

## Sources and Provenance

ACL bibliographic metadata comes from ACL Anthology. arXiv metadata comes from
the arXiv API. Citation aggregates come from Semantic Scholar and record their
query date. Screening and extraction fields are model-generated and should be
treated as research annotations rather than source-of-truth metadata.

## Exclusions

The release does not redistribute paper PDFs, full text, abstracts, API response
caches, provider logs, credentials, model prompts containing source text, or
claim-level quoted evidence. These remain local or must be reacquired from the
original provider under its terms.

## Limitations

- arXiv category selection is broader than NLP and is screened downstream.
- Citation counts are time-dependent.
- Screening and extraction can contain model errors.
- The 2020-2022 screening snapshot may be incomplete while retry work is active.
- The current dataset/DCU analysis covers 2023-2025, not the full six years.

## Responsible Use

Do not use model-generated fields to make decisions about individual authors,
institutions, or datasets without checking the source paper. Follow the source
providers' terms for any reacquired content.
