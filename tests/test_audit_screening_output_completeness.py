import json
from pathlib import Path

from scripts import audit_screening_output_completeness as audit


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_audit_writes_clean_output_and_missing_catalog(tmp_path: Path):
    catalog = tmp_path / "catalog.jsonl"
    output = tmp_path / "output.jsonl"
    clean = tmp_path / "clean.jsonl"
    missing = tmp_path / "missing.jsonl"
    summary = tmp_path / "summary.json"
    write_jsonl(catalog, [{"paper_id": "p1"}, {"paper_id": "p2"}, {"paper_id": "p3"}])
    write_jsonl(output, [{"paper_id": "p1"}, {"paper_id": "p1"}, {"paper_id": "p-extra"}, {"paper_id": "p3"}])

    audit.main(
        [
            "--catalog-jsonl",
            str(catalog),
            "--output-jsonl",
            str(output),
            "--clean-output-jsonl",
            str(clean),
            "--missing-catalog-jsonl",
            str(missing),
            "--summary-json",
            str(summary),
        ]
    )

    assert [row["paper_id"] for row in read_jsonl(clean)] == ["p1", "p3"]
    assert [row["paper_id"] for row in read_jsonl(missing)] == ["p2"]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["catalog_rows"] == 3
    assert payload["raw_output_rows"] == 4
    assert payload["clean_rows"] == 2
    assert payload["duplicate_ids"] == ["p1"]
    assert payload["extra_ids"] == ["p-extra"]
    assert payload["missing_ids"] == ["p2"]
    assert payload["is_complete"] is False
