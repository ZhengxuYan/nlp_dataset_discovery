import json
from pathlib import Path

from scripts import audit_enrichment_stable_ids
from scripts.enrich_dataset_metadata import stable_record_id


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def enriched(row: dict, record_id: str | None = None) -> dict:
    copied = dict(row)
    copied["public_metadata"] = {"metadata_enrichment": {"record_id": record_id or stable_record_id(row)}}
    return copied


def test_stable_id_audit_passes_when_enriched_rows_are_reordered(tmp_path: Path):
    rows = [
        {"dataset_id": "d1", "paper_id": "p1", "title": "First"},
        {"dataset_id": "d2", "paper_id": "p2", "title": "Second"},
    ]
    input_jsonl = tmp_path / "input.jsonl"
    enriched_jsonl = tmp_path / "enriched.jsonl"
    write_jsonl(input_jsonl, rows)
    write_jsonl(enriched_jsonl, [enriched(rows[1]), enriched(rows[0])])

    audit = audit_enrichment_stable_ids.build_audit(input_jsonl, enriched_jsonl)

    assert audit["status"] == "pass"
    assert audit["matched_count"] == 2
    assert audit["uses_row_position"] is False


def test_stable_id_audit_fails_on_duplicate_enriched_ids(tmp_path: Path):
    rows = [
        {"dataset_id": "d1", "paper_id": "p1"},
        {"dataset_id": "d2", "paper_id": "p2"},
    ]
    input_jsonl = tmp_path / "input.jsonl"
    enriched_jsonl = tmp_path / "enriched.jsonl"
    write_jsonl(input_jsonl, rows)
    write_jsonl(enriched_jsonl, [enriched(rows[0]), enriched(rows[1], record_id=stable_record_id(rows[0]))])

    audit = audit_enrichment_stable_ids.build_audit(input_jsonl, enriched_jsonl)

    assert audit["status"] == "fail"
    assert audit["enriched_duplicate_ids"] == {stable_record_id(rows[0]): 2}


def test_stable_id_audit_fails_on_missing_and_extra_ids(tmp_path: Path):
    rows = [
        {"dataset_id": "d1", "paper_id": "p1"},
        {"dataset_id": "d2", "paper_id": "p2"},
    ]
    input_jsonl = tmp_path / "input.jsonl"
    enriched_jsonl = tmp_path / "enriched.jsonl"
    write_jsonl(input_jsonl, rows)
    write_jsonl(enriched_jsonl, [enriched(rows[0]), {"dataset_id": "d3", "paper_id": "p3"}])

    audit = audit_enrichment_stable_ids.build_audit(input_jsonl, enriched_jsonl)

    assert audit["status"] == "fail"
    assert audit["missing_in_enriched"] == [stable_record_id(rows[1])]
    assert audit["extra_in_enriched"] == ["d3"]
