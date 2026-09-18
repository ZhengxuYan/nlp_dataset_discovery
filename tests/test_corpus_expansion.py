from scripts.corpus_expansion import YearRange, extract_record_year, output_paths


def test_year_range_boundaries():
    years = YearRange(2020, 2025)
    assert not years.includes(2019)
    assert years.includes(2020)
    assert years.includes(2025)
    assert not years.includes(2026)


def test_extract_record_year_from_date_fields():
    assert extract_record_year({"published": "2020-02-03"}) == 2020
    assert extract_record_year({"publication_year": "2025"}) == 2025
    assert extract_record_year({"title": "No year"}) is None


def test_output_paths_use_new_label(tmp_path):
    paths = output_paths(tmp_path, YearRange(2020, 2025))
    assert "2020_2025" in str(paths["integrated_dataset_bank_jsonl"])
    assert "2023_2025" not in str(paths["metadata_enriched_jsonl"])
