from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.build_public_release import (
    acl_catalog_row,
    arxiv_catalog_row,
    sanitize_dataset_row,
    screening_row,
    write_jsonl_gz,
)


def test_catalog_rows_exclude_abstracts() -> None:
    row = {
        "paper_id": "p1",
        "title": "Title",
        "year": 2020,
        "abstract": "source text",
        "authors": "A; B",
    }
    assert "abstract" not in arxiv_catalog_row(row)
    assert "abstract" not in acl_catalog_row(row)


def test_dataset_sanitizer_removes_evidence_fields() -> None:
    row = {
        "bank_id": "b1",
        "dataset_name": "Example",
        "acus": [{"evidence": "quoted text"}],
        "prior_dataset_mentions": [{"evidence": "quoted text"}],
        "search_text": "source text",
        "ambiguities": ["x"],
        "missing_information": ["y"],
    }
    cleaned = sanitize_dataset_row(row)
    assert cleaned == {"bank_id": "b1", "dataset_name": "Example"}


def test_screening_row_removes_acus() -> None:
    row = {
        "paper_id": "p1",
        "title": "Title",
        "year": 2021,
        "datasets": [{"name": "D", "acus": ["quoted claim"]}],
    }
    cleaned = screening_row(row, "arxiv")
    assert cleaned["corpus"] == "arxiv"
    assert "acus" not in cleaned["datasets"][0]


def test_gzip_writer_is_readable_and_counts_years(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl.gz"
    count, years = write_jsonl_gz(path, [{"year": 2020}, {"year": 2025}])
    assert count == 2
    assert years == {2020: 1, 2025: 1}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert [json.loads(line) for line in handle] == [{"year": 2020}, {"year": 2025}]
