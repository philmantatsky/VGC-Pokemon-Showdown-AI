from __future__ import annotations

import json

from evaluation.run_rollout_gate import _audit_fallbacks, _valid_result


def test_rollout_result_requires_schema_and_matching_residual(tmp_path):
    residual = tmp_path / "residual.pt"
    residual.write_bytes(b"model")
    result = tmp_path / "eval.json"
    result.write_text(
        json.dumps(
            {
                "evaluation_schema": 3,
                "arms": {
                    "distilled_policy": {
                        "battles": 500,
                        "residual_ranker": str(residual),
                    },
                    "live_exact_search": {
                        "battles": 500,
                        "residual_ranker": str(residual),
                    },
                },
            }
        )
    )
    assert _valid_result(result, residual, 500)
    assert not _valid_result(result, residual, 501)


def test_rollout_review_extracts_every_decision_fallback(tmp_path):
    audit = tmp_path / "audit.jsonl"
    audit.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "battle": "battle-1",
                        "turn": 3,
                        "exact_search": {
                            "decision_fallback": {
                                "to": "champion_plus_guards",
                                "error_type": "ValueError",
                                "error": "bad mapping",
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "battle": "battle-1",
                        "turn": 4,
                        "exact_search": {"result": {"choice": "move protect"}},
                    }
                ),
            ]
        )
        + "\n"
    )
    payload = {
        "arms": {
            "live_exact_search": {"move_search": {"audit": str(audit)}}
        }
    }
    assert _audit_fallbacks(payload) == [
        {
            "battle": "battle-1",
            "turn": 3,
            "to": "champion_plus_guards",
            "error_type": "ValueError",
            "error": "bad mapping",
        }
    ]
