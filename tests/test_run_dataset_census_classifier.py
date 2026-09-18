import pytest

from scripts.run_dataset_census_classifier import normalize_batch_outputs


def test_normalize_batch_outputs_rejects_missing_paper():
    rows = [{"paper_id": "p1", "title": "One"}, {"paper_id": "p2", "title": "Two"}]
    payload = {"papers": [{"paper_id": "p1", "is_nlp_paper": True}]}

    with pytest.raises(ValueError, match="Model omitted paper_id"):
        normalize_batch_outputs(rows, payload)


def test_normalize_batch_outputs_rejects_duplicate_paper():
    rows = [{"paper_id": "p1", "title": "One"}]
    payload = {"papers": [{"paper_id": "p1"}, {"paper_id": "p1"}]}

    with pytest.raises(ValueError, match="duplicate paper_id"):
        normalize_batch_outputs(rows, payload)


def test_normalize_batch_outputs_accepts_arxiv_version_alias():
    rows = [{"paper_id": "2201.00001v2", "title": "Versioned"}]
    payload = {"papers": [{"paper_id": "2201.00001v1", "is_dataset_mentioned": True}]}

    normalized = normalize_batch_outputs(rows, payload)

    assert normalized[0]["paper_id"] == "2201.00001v2"
    assert normalized[0]["is_dataset_mentioned"] is True
