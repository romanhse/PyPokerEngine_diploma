from __future__ import annotations

import json
import time

import pytest

from poker_research.llm_policy import (
    CompletionRequest,
    CompletionResponse,
    LLMPolicyConfig,
    StrictJSONLLMPolicy,
    build_prompt,
    observation_payload,
    parse_response,
    response_schema,
)
from poker_research.types import ActionOption, Observation, PublicAction


def _observation() -> Observation:
    return Observation(
        hand_id="secret-seed-like-id",
        seat=1,
        street="flop",
        position="button_sb",
        hole_cards=("As", "Kh"),
        board_cards=("2c", "7d", "Th"),
        stacks=(180, 176),
        street_bets=(4, 8),
        pot=20,
        call_amount=4,
        effective_stack=176,
        spr=8.8,
        legal_actions=(
            ActionOption("fold"),
            ActionOption("check_call"),
            ActionOption("raise_min", 12),
        ),
        history=(PublicAction(0, "raise_min", 8),),
    )


class _Backend:
    def __init__(self, responses: list[str | Exception | CompletionResponse]) -> None:
        self.responses = responses
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, CompletionResponse):
            return response
        return CompletionResponse(
            response,
            model_revision="fixed-revision",
            prompt_tokens=101,
            completion_tokens=19,
            request_id="request-id-is-not-logged",
        )


def _valid_response(action: str = "check_call") -> str:
    return json.dumps(
        {
            "action": action,
            "probabilities": {"fold": 0.1, "check_call": 0.8, "raise_min": 0.1},
            "reason": "concise",
        }
    )


def _policy_config(**overrides: object) -> LLMPolicyConfig:
    values: dict[str, object] = {
        "provider": "fake",
        "model": "offline-model",
    }
    values.update(overrides)
    return LLMPolicyConfig(**values)  # type: ignore[arg-type]


def _attempt_audit(metadata: dict[str, object]) -> list[dict[str, object]]:
    payload = json.loads(str(metadata["attempt_audit_json"]))
    assert isinstance(payload, list)
    return payload


def test_observation_payload_exposes_public_boundary_only() -> None:
    observation = _observation()
    payload = observation_payload(observation)

    assert payload["hole_cards"] == ["As", "Kh"]
    assert "hand_id" not in payload
    assert "deck" not in json.dumps(payload).lower()
    assert "opponent_hole" not in json.dumps(payload).lower()
    assert payload["history"] == [{"seat": 0, "action": "raise_min", "amount": 8}]


def test_prompt_and_schema_are_canonical_and_action_specific() -> None:
    observation = _observation()

    assert build_prompt(observation) == build_prompt(observation)
    schema = response_schema(("fold", "check_call", "raise_min"))
    assert schema["properties"]["action"]["enum"] == ["fold", "check_call", "raise_min"]
    assert schema["properties"]["probabilities"]["additionalProperties"] is False


def test_strict_policy_returns_valid_audited_decision() -> None:
    backend = _Backend([_valid_response()])
    policy = StrictJSONLLMPolicy(
        backend,
        _policy_config(
            model="model@revision",
            expected_model_revision="fixed-revision",
            seed=42,
        ),
    )

    decision = policy.decide(_observation())

    assert decision.action == "check_call"
    assert dict(decision.probabilities) == {"fold": 0.1, "check_call": 0.8, "raise_min": 0.1}
    metadata = dict(decision.metadata)
    assert metadata["fallback"] is False
    assert metadata["parse_status"] == "ok"
    assert metadata["prompt_version"] == "hunl_policy_json_v1"
    assert metadata["model_revision"] == "fixed-revision"
    assert metadata["prompt_tokens"] == 101
    assert metadata["completion_tokens"] == 19
    assert metadata["total_prompt_tokens"] == 101
    assert metadata["total_completion_tokens"] == 19
    assert metadata["usage_incomplete_attempts"] == 0
    assert metadata["response_json"] == _valid_response()
    assert "request-id-is-not-logged" not in str(metadata)
    audit = _attempt_audit(metadata)
    assert audit == [
        {
            "attempt": 1,
            "call_number": 1,
            "completion_tokens": 19,
            "deadline_exceeded": False,
            "error_code": None,
            "error_type": None,
            "latency_ms": audit[0]["latency_ms"],
            "model_revision": "fixed-revision",
            "prompt_tokens": 101,
            "response_hash": metadata["response_hash"],
            "response_text": _valid_response(),
            "status": "ok",
        }
    ]
    assert metadata["deadline_contract"] == "backend_enforces_request_timeout_v1"
    assert backend.requests[0].timeout_seconds == 15.0


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        json.dumps({"action": "fold", "probabilities": {"fold": 1.0}}),
        json.dumps(
            {
                "action": "illegal",
                "probabilities": {"fold": 0.0, "check_call": 1.0, "raise_min": 0.0},
            }
        ),
        json.dumps(
            {
                "action": "fold",
                "probabilities": {"fold": 0.2, "check_call": 0.2, "raise_min": 0.2},
            }
        ),
        json.dumps(
            {
                "action": "fold",
                "probabilities": {"fold": 1.0, "check_call": 0.0, "raise_min": 0.0},
                "extra": "forbidden",
            }
        ),
        (
            '{"action":"fold","action":"check_call","probabilities":'
            '{"fold":0.1,"check_call":0.8,"raise_min":0.1}}'
        ),
        (
            '{"action":"fold","probabilities":'
            '{"fold":NaN,"check_call":0.0,"raise_min":0.0}}'
        ),
        json.dumps(
            {
                "action": "fold",
                "probabilities": {"fold": 1.0, "check_call": 0.0, "raise_min": 0.0},
                "reason": "x" * 501,
            }
        ),
    ],
)
def test_parser_rejects_malformed_or_non_strict_responses(response: str) -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_response(response, _observation())


def test_bounded_retry_then_deterministic_fallback() -> None:
    backend = _Backend([RuntimeError("timeout secret"), "bad json"])
    policy = StrictJSONLLMPolicy(
        backend,
        LLMPolicyConfig(max_attempts=2, provider="fake", model="offline"),
    )

    decision = policy.decide(_observation())

    assert len(backend.requests) == 2
    assert decision.action == "check_call"
    assert dict(decision.probabilities) == {"fold": 0.0, "check_call": 1.0, "raise_min": 0.0}
    metadata = dict(decision.metadata)
    assert metadata["fallback"] is True
    assert metadata["attempts"] == 2
    assert metadata["parse_status"] == "fallback"
    assert metadata["last_response"] == "bad json"
    assert metadata["error_type"] == "ResponseValidationError"
    assert metadata["error_code"] == "invalid_json"
    assert metadata["total_prompt_tokens"] == 101
    assert metadata["total_completion_tokens"] == 19
    assert metadata["usage_incomplete_attempts"] == 1
    audit = _attempt_audit(metadata)
    assert [entry["status"] for entry in audit] == ["backend_error", "response_error"]
    assert [entry["error_code"] for entry in audit] == [
        "backend_exception",
        "invalid_json",
    ]
    assert audit[0]["error_type"] == "RuntimeError"
    assert audit[1]["response_text"] == "bad json"
    assert "timeout secret" not in json.dumps(metadata, sort_keys=True)


def test_successful_second_attempt_is_not_marked_fallback() -> None:
    backend = _Backend(["invalid first response", _valid_response("raise_min")])
    policy = StrictJSONLLMPolicy(backend, _policy_config(max_attempts=2))

    decision = policy.decide(_observation())

    assert decision.action == "raise_min"
    metadata = dict(decision.metadata)
    assert metadata["attempts"] == 2
    assert metadata["fallback"] is False
    assert metadata["total_prompt_tokens"] == 202
    assert metadata["total_completion_tokens"] == 38
    assert metadata["usage_incomplete_attempts"] == 0
    audit = _attempt_audit(metadata)
    assert [entry["status"] for entry in audit] == ["response_error", "ok"]
    assert audit[0]["response_text"] == "invalid first response"


@pytest.mark.parametrize(
    ("revision", "expected", "error_code"),
    [
        (None, None, "missing_model_revision"),
        ("", None, "missing_model_revision"),
        ("actual-revision", "frozen-revision", "model_revision_mismatch"),
    ],
)
def test_missing_or_mismatched_model_revision_forces_audited_fallback(
    revision: str | None,
    expected: str | None,
    error_code: str,
) -> None:
    response = CompletionResponse(
        _valid_response(),
        model_revision=revision,
        prompt_tokens=7,
        completion_tokens=3,
    )
    backend = _Backend([response])
    policy = StrictJSONLLMPolicy(
        backend,
        _policy_config(expected_model_revision=expected),
    )

    decision = policy.decide(_observation())

    metadata = dict(decision.metadata)
    assert metadata["fallback"] is True
    assert metadata["error_code"] == error_code
    assert metadata["total_prompt_tokens"] == 7
    assert metadata["total_completion_tokens"] == 3
    assert _attempt_audit(metadata)[0]["error_code"] == error_code


def test_total_call_budget_prevents_additional_backend_requests() -> None:
    backend = _Backend([_valid_response()])
    policy = StrictJSONLLMPolicy(backend, _policy_config(max_total_calls=1))

    assert policy.decide(_observation()).action == "check_call"
    exhausted = policy.decide(_observation())

    assert len(backend.requests) == 1
    assert policy.calls_made == 1
    assert policy.calls_remaining == 0
    metadata = dict(exhausted.metadata)
    assert metadata["fallback"] is True
    assert metadata["attempts"] == 0
    assert metadata["error_code"] == "call_budget_exhausted"
    assert metadata["call_budget_exhausted"] is True
    assert _attempt_audit(metadata) == []


def test_backend_that_returns_after_deadline_is_rejected_and_audited() -> None:
    class _LateBackend:
        def complete(self, request: CompletionRequest) -> CompletionResponse:
            time.sleep(0.01)
            return CompletionResponse(
                _valid_response(),
                model_revision="fixed-revision",
                prompt_tokens=5,
                completion_tokens=2,
            )

    policy = StrictJSONLLMPolicy(
        _LateBackend(),
        _policy_config(timeout_seconds=0.001),
    )

    decision = policy.decide(_observation())

    metadata = dict(decision.metadata)
    assert metadata["fallback"] is True
    assert metadata["error_code"] == "deadline_exceeded"
    assert metadata["total_prompt_tokens"] == 5
    assert metadata["total_completion_tokens"] == 2
    audit = _attempt_audit(metadata)
    assert audit[0]["status"] == "deadline_exceeded"
    assert audit[0]["deadline_exceeded"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_seconds", 0.0),
        ("temperature", -0.1),
        ("max_output_tokens", 0),
        ("max_attempts", 4),
        ("max_total_calls", 0),
        ("prompt_version", "mutable_prompt"),
        ("provider", ""),
        ("provider", "unspecified"),
        ("model", "   "),
        ("expected_model_revision", ""),
        ("seed", True),
    ],
)
def test_config_rejects_unsafe_values(field: str, value: object) -> None:
    values = {
        "timeout_seconds": 15.0,
        "temperature": 0.0,
        "max_output_tokens": 300,
        "max_attempts": 1,
        "max_total_calls": 10,
        "prompt_version": "hunl_policy_json_v1",
        "provider": "fake",
        "model": "offline-model",
    }
    values[field] = value

    with pytest.raises(ValueError):
        LLMPolicyConfig(**values).validate()  # type: ignore[arg-type]
