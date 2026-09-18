import json
from pathlib import Path

import pytest

from scripts import sample_jsonl_by_year


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_year_stratified_sampler_writes_deterministic_rows(tmp_path: Path):
    input_path = tmp_path / "catalog.jsonl"
    output_path = tmp_path / "sample.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(
        input_path,
        [
            {"paper_id": "2019-a", "year": 2019},
            {"paper_id": "2020-a", "year": 2020},
            {"paper_id": "2020-b", "year": 2020},
            {"paper_id": "2021-a", "published_date": "2021-05-01"},
            {"paper_id": "2021-b", "year": 2021},
            {"paper_id": "2022-a", "year": 2022},
            {"paper_id": "2022-b", "year": 2022},
        ],
    )

    sample_jsonl_by_year.main(
        [
            "--input-jsonl",
            str(input_path),
            "--output-jsonl",
            str(output_path),
            "--summary-json",
            str(summary_path),
            "--start-year",
            "2020",
            "--end-year",
            "2022",
            "--per-year",
            "1",
        ]
    )

    assert [row["paper_id"] for row in read_jsonl(output_path)] == ["2020-a", "2021-a", "2022-a"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rows_written"] == 3
    assert summary["sample_year_counts"] == {"2020": 1, "2021": 1, "2022": 1}
    assert summary["input_year_counts"] == {"2020": 2, "2021": 2, "2022": 2}


def test_year_stratified_sampler_fails_when_year_is_short(tmp_path: Path):
    input_path = tmp_path / "catalog.jsonl"
    output_path = tmp_path / "sample.jsonl"
    write_jsonl(input_path, [{"paper_id": "2020-a", "year": 2020}, {"paper_id": "2021-a", "year": 2021}])

    with pytest.raises(RuntimeError, match="2020: 1/2"):
        sample_jsonl_by_year.main(
            [
                "--input-jsonl",
                str(input_path),
                "--output-jsonl",
                str(output_path),
                "--start-year",
                "2020",
                "--end-year",
                "2021",
                "--per-year",
                "2",
            ]
        )
