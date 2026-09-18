import json
from pathlib import Path

import pytest

from scripts.run_continuous_paper_enrichment import (
    count_jsonl_rows,
    load_state,
    progress_snapshot,
    save_state,
)


def test_state_round_trip_and_progress(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    state_path = tmp_path / "state.json"
    state = load_state(state_path, input_path=input_path, start_offset=1, total_rows=2)
    state["next_offset"] = 2
    state["status"] = "complete"
    save_state(state_path, state)

    loaded = load_state(state_path, input_path=input_path, start_offset=0, total_rows=2)
    assert count_jsonl_rows(input_path) == 2
    assert progress_snapshot(loaded)["percent_complete"] == 100.0
    assert loaded["next_offset"] == 2


def test_state_rejects_different_input(tmp_path: Path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")
    state_path = tmp_path / "state.json"
    save_state(state_path, load_state(state_path, input_path=first, start_offset=0, total_rows=1))

    with pytest.raises(ValueError, match="State input mismatch"):
        load_state(state_path, input_path=second, start_offset=0, total_rows=1)
