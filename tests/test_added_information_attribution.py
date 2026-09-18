import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_completed_benchmark_rows.py"
spec = importlib.util.spec_from_file_location("evaluate_completed_benchmark_rows", MODULE_PATH)
eval_rows = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["evaluate_completed_benchmark_rows"] = eval_rows
spec.loader.exec_module(eval_rows)

AddedInformationAttributionOutput = eval_rows.AddedInformationAttributionOutput
ClaimAttributionOutput = eval_rows.ClaimAttributionOutput
added_information_profile = eval_rows.added_information_profile
heuristic_claim_attributions = eval_rows.heuristic_claim_attributions
normalize_attributions = eval_rows.normalize_attributions
parse_json_object = eval_rows.parse_json_object


def claim(query_id, status, importance="medium", delta_type="other", prior_ids=None):
    return ClaimAttributionOutput(
        query_acu_id=query_id,
        query_acu=f"query {query_id}",
        support_status=status,
        best_prior_acu_ids=prior_ids or [],
        delta_type=delta_type,
        importance=importance,
        rationale="grounded test rationale",
    )


def test_added_information_profile_uses_weighted_delta_formula():
    profile = added_information_profile([
        claim("q0", "supported", importance="high"),
        claim("q1", "partially_supported", importance="medium"),
        claim("q2", "unsupported", importance="low", delta_type="scale/coverage"),
    ])

    assert profile["n_query_acus"] == 3
    assert profile["support_counts"] == {
        "supported": 1,
        "partially_supported": 1,
        "unsupported": 1,
    }
    assert profile["unsupported_by_delta_type"] == {"scale/coverage": 1}
    assert profile["added_information_score"] == pytest.approx((0.0 * 1.5 + 0.5 * 1.0 + 1.0 * 0.5) / 3.0)


def test_added_information_profile_excludes_contradicted_and_not_comparable_from_score():
    profile = added_information_profile([
        claim("q0", "supported"),
        claim("q1", "contradicted", importance="high"),
        claim("q2", "not_comparable", importance="high"),
    ])

    assert profile["added_information_score"] == 0.0
    assert profile["excluded_from_score_count"] == 2
    assert profile["support_percentages"]["contradicted"] == pytest.approx(1 / 3)
    assert profile["support_percentages"]["not_comparable"] == pytest.approx(1 / 3)


def test_heuristic_attribution_empty_prior_marks_query_acus_unsupported():
    attributions = heuristic_claim_attributions(["new dataset claim", "new annotation claim"], [])
    profile = added_information_profile(attributions)

    assert [item.support_status for item in attributions] == ["unsupported", "unsupported"]
    assert profile["added_information_score"] == 1.0
    assert profile["support_percentages"]["unsupported"] == 1.0


def test_normalize_attributions_rejects_unknown_prior_id():
    output = AddedInformationAttributionOutput(attributions=[
        ClaimAttributionOutput(
            query_acu_id="q0",
            query_acu="query claim",
            support_status="supported",
            best_prior_acu_ids=["p7"],
            delta_type="other",
            importance="medium",
            rationale="bad prior id",
        )
    ])

    with pytest.raises(ValueError, match="Unknown best_prior_acu_ids"):
        normalize_attributions(output, ["query claim"], ["prior claim"])


def test_normalize_attributions_rejects_missing_query_decision():
    output = AddedInformationAttributionOutput(attributions=[
        ClaimAttributionOutput(
            query_acu_id="q0",
            query_acu="query 0",
            support_status="supported",
            best_prior_acu_ids=["p0"],
            delta_type="other",
            importance="medium",
            rationale="only first query covered",
        )
    ])

    with pytest.raises(ValueError, match="Missing attribution"):
        normalize_attributions(output, ["query 0", "query 1"], ["prior claim"])


def test_parse_json_object_repairs_common_llm_json_wrapping():
    parsed = parse_json_object(
        '```json\n{"attributions":[{"query_acu_id":"q0",}],}\n```'
    )

    assert parsed == {"attributions": [{"query_acu_id": "q0"}]}
