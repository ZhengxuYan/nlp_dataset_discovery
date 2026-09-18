from scripts.prepare_main_analysis_expansion import filter_years, row_year, summarize_rows


def test_year_filter_includes_2020_through_2022_only():
    rows = [
        {"paper_id": "a", "year": 2019},
        {"paper_id": "b", "year": 2020},
        {"paper_id": "c", "published_date": "2022-12-01"},
        {"paper_id": "d", "year": 2023},
    ]
    selected = filter_years(rows, 2020, 2022)
    assert [row["paper_id"] for row in selected] == ["b", "c"]
    assert row_year(selected[1]) == 2022


def test_summary_reports_duplicate_ids_and_abstract_coverage():
    summary = summarize_rows([
        {"paper_id": "a", "year": 2020, "abstract": "text"},
        {"paper_id": "a", "year": 2020, "abstract": ""},
    ])
    assert summary["rows"] == 2
    assert summary["unique_ids"] == 1
    assert summary["duplicate_ids"] == 1
    assert summary["rows_with_abstract"] == 1
