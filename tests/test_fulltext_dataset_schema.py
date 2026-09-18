from scv.fulltext_dataset_schema import (
    parse_model_payload,
    quality_warnings_for_bank,
    validate_extraction_for_bank,
)


def test_fulltext_payload_accepts_wide_schema():
    record = parse_model_payload({
        "paper_id": "ACL:2025.acl-long.1",
        "acl_id": "2025.acl-long.1",
        "title": "A Dataset Paper",
        "datasets": [
            {
                "dataset_identity": {"canonical_name": "ExampleSet"},
                "acus": [
                    {
                        "id": "q0",
                        "text": "ExampleSet contains 10,000 examples.",
                        "type": "scale/coverage",
                        "importance": "high",
                        "evidence": "The dataset contains 10,000 examples.",
                        "section": "Dataset",
                    },
                    {
                        "id": "q1",
                        "text": "ExampleSet covers sentiment analysis.",
                        "type": "task/domain",
                        "importance": "medium",
                        "evidence": "We use the dataset for sentiment analysis.",
                        "section": "Tasks",
                    },
                    {
                        "id": "q2",
                        "text": "ExampleSet is manually annotated.",
                        "type": "annotation/protocol",
                        "importance": "medium",
                        "evidence": "Annotators labeled each example.",
                        "section": "Annotation",
                    },
                    {
                        "id": "q3",
                        "text": "ExampleSet is released on GitHub.",
                        "type": "availability/quality",
                        "importance": "low",
                        "evidence": "The data is available on GitHub.",
                        "section": "Availability",
                    },
                ],
            }
        ],
    })

    assert record.paper_id == "ACL:2025.acl-long.1"
    assert record.datasets[0].dataset_identity.canonical_name == "ExampleSet"
    assert validate_extraction_for_bank(record) == []


def test_validator_rejects_missing_evidence_and_bad_acu_type():
    record = parse_model_payload({
        "paper_id": "ACL:2025.acl-long.1",
        "datasets": [
            {
                "dataset_identity": {"canonical_name": "ExampleSet"},
                "acus": [
                    {
                        "text": "ExampleSet is useful.",
                        "type": "novelty",
                        "importance": "medium",
                        "evidence": "",
                    }
                ],
            }
        ],
    })

    errors = validate_extraction_for_bank(record)
    warnings = quality_warnings_for_bank(record)

    assert "datasets[0] has fewer than 4 ACUs" in warnings
    assert "datasets[0].acus[0] invalid type: novelty" in errors
    assert "datasets[0].acus[0] missing evidence" in errors
