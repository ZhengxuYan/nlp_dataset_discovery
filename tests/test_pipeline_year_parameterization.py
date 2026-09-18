import importlib.util
import gzip
from pathlib import Path

from scripts import build_acl_anthology_catalog as acl_catalog
from scripts import build_dataset_census
from scripts import prepare_arxiv_screening_catalog as arxiv_screening
from scripts.build_integrated_fulltext_banks import flatten_dataset


def load_run_arxiv_intervals():
    path = Path("scrapers/arxiv_scraper/run_arxiv_intervals.py")
    spec = importlib.util.spec_from_file_location("run_arxiv_intervals_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_acl_filter_scope_accepts_2020_to_2025():
    rows = [
        {"acl_id": "2020.acl-main.1", "year": 2020, "venue_prefix": "acl-long", "is_front_matter": False},
        {"acl_id": "2022.findings-acl.1", "year": 2022, "venue_prefix": "findings-acl", "is_front_matter": False},
        {"acl_id": "2025.acl-long.1", "year": 2025, "venue_prefix": "acl-long", "is_front_matter": False},
        {"acl_id": "2019.acl-long.1", "year": 2019, "venue_prefix": "acl-long", "is_front_matter": False},
        {"acl_id": "2020.acl-main.0", "year": 2020, "venue_prefix": "acl-long", "is_front_matter": True},
    ]
    scoped = acl_catalog.filter_scope(rows, "all", 2020, 2025)
    assert [row["year"] for row in scoped] == [2020, 2022, 2025]


def test_acl_catalog_reads_bib_gz_fallback(tmp_path):
    bib = tmp_path / "anthology.bib.gz"
    with gzip.open(bib, "wt", encoding="utf-8") as handle:
        handle.write(
            "@inproceedings{doe-smith-2020-dataset,\n"
            "  title={A Dataset Paper},\n"
            "  author={Doe, Jane and Smith, John},\n"
            "  year={2020},\n"
            "  url={https://aclanthology.org/2020.acl-main.1/}\n"
            "}\n"
        )

    rows = acl_catalog.read_bib_gz(bib)
    normalized = acl_catalog.normalize_row(rows[0])

    assert normalized["acl_id"] == "2020.acl-main.1"
    assert normalized["year"] == 2020
    assert normalized["venue_prefix"] == "acl-main"
    assert normalized["authors"] == "Doe, Jane; Smith, John"


def test_arxiv_screening_defaults_follow_year_label():
    paths = arxiv_screening.default_paths(2020, 2025)
    assert paths["input_csv"].endswith("arxiv_results_2020_2025.csv")
    assert paths["output_jsonl"].endswith("arxiv_2020_2025_dedup_no_acl_for_dataset_screening.jsonl")
    assert paths["acl_jsonl"].endswith("acl_anthology_2020_2025_all_with_abstracts.jsonl")


def test_dataset_census_defaults_follow_year_label():
    assert build_dataset_census.default_catalog(2020, 2025).endswith("arxiv_nlp_conf_papers_2020_2025.csv")
    assert build_dataset_census.default_analysis(2020, 2025)[0].endswith(
        "arxiv_nlp_conf_papers_2020_2025_dataset_analysis(gpt-5-mini).jsonl"
    )


def test_run_arxiv_intervals_dry_run_outputs_expected_ranges(capsys):
    module = load_run_arxiv_intervals()
    module.main([
        "--start-date",
        "2020-01-01",
        "--end-date",
        "2020-04-30",
        "--interval-months",
        "3",
        "--output-file",
        "data/raw/arxiv_results_2020_2025.csv",
        "--sleep-seconds",
        "0",
        "--dry-run",
    ])
    output = capsys.readouterr().out
    assert "start_date=2020-01-01" in output
    assert "end_date=2020-03-31" in output
    assert "start_date=2020-04-01" in output
    assert "end_date=2020-04-30" in output
    assert "output_file=data/raw/arxiv_results_2020_2025.csv" in output


def test_integrated_dataset_preserves_paper_identifiers():
    row = flatten_dataset(
        {"paper_id": "p1", "doi": "10.1/example", "arxiv_id": "2001.00001", "arxiv_url": "https://arxiv.org/abs/2001.00001"},
        {"dataset_identity": {"canonical_name": "Dataset"}, "acus": []},
        dataset_index=0,
        source_corpus="test",
    )
    assert row["doi"] == "10.1/example"
    assert row["arxiv_id"] == "2001.00001"
    assert row["arxiv_url"] == "https://arxiv.org/abs/2001.00001"
