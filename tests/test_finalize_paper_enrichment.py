from pathlib import Path

from scripts.finalize_paper_enrichment import quantiles, shard_bounds


def test_shard_bounds():
    assert shard_bounds(Path("enriched_012250_013250.jsonl")) == (12250, 13250)


def test_quantiles_empty_and_values():
    assert quantiles([])["median"] is None
    result = quantiles([0, 1, 2, 10])
    assert result["median"] == 1.5
    assert result["max"] == 10
