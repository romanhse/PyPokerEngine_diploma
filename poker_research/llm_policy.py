"""Strict, provider-independent adapter for language-model poker policies.

The adapter intentionally knows nothing about PokerKit.  A backend receives a
versioned prompt containing only :class:`~poker_research.types.Observation` and
must return one bounded JSON object.  This keeps LLM experiments behind the
same information boundary and action validator as every local policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from poker_research.types import ActionName, Decision, MetadataValue, Observation

PROMPT_VERSION = "hunl_policy_json_v1"
DEADLINE_CONTRACT = "backend_enforces_request_timeout_v1"
_MISSING = object()


class ResponseValidationError(ValueError):
    """A response failure with a fixed, non-secret audit code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """One provider-neutral, reproducible completion request."""

    prompt: str
    json_schema: dict[str, Any]
    model: str
    temperature: float
    seed: int | None
    timeout_seconds: float
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """Provider-neutral response plus the fields required for an audit."""

    text: str
    model_revision: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None


class CompletionBackend(Protocol):
    """Synchronous provider boundary with backend-enforced deadlines.

    Implementations must enforce ``request.timeout_seconds`` inside the provider
    transport or an isolated worker process and raise ``TimeoutError`` when it
    expires.  ``StrictJSONLLMPolicy`` deliberately does not start an unkillable
    helper thread: it can audit a late return, but cannot safely interrupt an
    arbitrary synchronous Python call by itself.
    """

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Return before the request deadline or raise a transport exception."""


@dataclass(frozen=True, slots=True)
class LLMPolicyConfig:
    """Frozen knobs that define one LLM policy version."""

    policy_name: str = "llm_json_v1"
    provider: str = ""
    model: str = ""
    expected_model_revision: str | None = None
    prompt_version: str = PROMPT_VERSION
    temperature: float = 0.0
    seed: int | None = 0
    timeout_seconds: float = 15.0
    max_output_tokens: int = 300
    max_attempts: int = 1
    max_total_calls: int = 10_000

    def validate(self) -> None:
        if not self.policy_name or any(character.isspace() for character in self.policy_name):
            raise ValueError("policy_name must be non-empty and contain no whitespace")
        _validate_non_placeholder(self.provider, "provider")
        _validate_non_placeholder(self.model, "model")
        if self.expected_model_revision is not None:
            _validate_non_placeholder(
                self.expected_model_revision,
                "expected_model_revision",
            )
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(f"unsupported prompt version: {self.prompt_version}")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= 3
        ):
            raise ValueError("max_attempts must be between one and three")
        if (
            isinstance(self.max_total_calls, bool)
            or not isinstance(self.max_total_calls, int)
            or self.max_total_calls < self.max_attempts
        ):
            raise ValueError("max_total_calls must be at least max_attempts")


@dataclass(frozen=True, slots=True)
class _AttemptAudit:
    """Compact audit record for exactly one backend invocation."""

    attempt: int
    call_number: int
    status: str
    latency_ms: float
    deadline_exceeded: bool
    error_type: str | None
    error_code: str | None
    model_revision: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    response_hash: str | None
    response_text: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ResponseAuditFields:
    text: str | None
    model_revision: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    response_hash: str | None


class StrictJSONLLMPolicy:
    """Validate an LLM response exactly, falling back safely on any failure.

    Retries are an explicit bounded loop, never recursion.  A frozen total-call
    budget bounds the whole policy instance.  The fallback is check/call, then
    fold, then the first legal action.  Every attempt and fallback is visible in
    metadata, including aggregate usage, without logging exception messages.
    """

    def __init__(
        self,
        backend: CompletionBackend,
        config: LLMPolicyConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or LLMPolicyConfig()
        self.config.validate()
        self.name = self.config.policy_name
        self._calls_made = 0

    @property
    def calls_made(self) -> int:
        """Number of backend calls consumed by this policy instance."""

        return self._calls_made

    @property
    def calls_remaining(self) -> int:
        """Remaining calls in the frozen match-level budget."""

        return self.config.max_total_calls - self._calls_made

    def decide(self, observation: Observation) -> Decision:
        prompt = build_prompt(observation, prompt_version=self.config.prompt_version)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        schema = response_schema(tuple(option.name for option in observation.legal_actions))
        request = CompletionRequest(
            prompt=prompt,
            json_schema=schema,
            model=self.config.model,
            temperature=self.config.temperature,
            seed=self.config.seed,
            timeout_seconds=self.config.timeout_seconds,
            max_output_tokens=self.config.max_output_tokens,
        )

        last_error_type = "CallBudgetError"
        last_error_code = "call_budget_exhausted"
        last_response: str | None = None
        total_latency_ms = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        usage_incomplete_attempts = 0
        attempt_audits: list[_AttemptAudit] = []

        for attempt in range(1, self.config.max_attempts + 1):
            if self._calls_made >= self.config.max_total_calls:
                last_error_type = "CallBudgetError"
                last_error_code = "call_budget_exhausted"
                break

            self._calls_made += 1
            call_number = self._calls_made
            started_ns = time.perf_counter_ns()
            try:
                response = self.backend.complete(request)
            except Exception as error:  # provider failures share one bounded retry path
                latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                total_latency_ms += latency_ms
                error_type, error_code = _safe_error(error, source="backend")
                last_error_type, last_error_code = error_type, error_code
                usage_incomplete_attempts += 1
                attempt_audits.append(
                    _AttemptAudit(
                        attempt,
                        call_number,
                        "backend_error",
                        round(latency_ms, 3),
                        isinstance(error, TimeoutError),
                        error_type,
                        error_code,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                )
                continue

            latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            total_latency_ms += latency_ms
            fields = _response_audit_fields(response)
            if fields.text is not None:
                last_response = fields.text
            if fields.prompt_tokens is None or fields.completion_tokens is None:
                usage_incomplete_attempts += 1
            if fields.prompt_tokens is not None:
                total_prompt_tokens += fields.prompt_tokens
            if fields.completion_tokens is not None:
                total_completion_tokens += fields.completion_tokens

            if latency_ms > self.config.timeout_seconds * 1_000:
                last_error_type = "BackendDeadlineExceeded"
                last_error_code = "deadline_exceeded"
                attempt_audits.append(
                    _attempt_audit(
                        attempt,
                        call_number,
                        "deadline_exceeded",
                        latency_ms,
                        fields,
                        error_type=last_error_type,
                        error_code=last_error_code,
                        deadline_exceeded=True,
                    )
                )
                continue

            try:
                revision, prompt_tokens, completion_tokens = _validate_completion_response(
                    response,
                    self.config,
                )
                if fields.text is None:
                    raise ResponseValidationError(
                        "invalid_response_text",
                        "completion response text must be a string",
                    )
                decision = parse_response(fields.text, observation)
            except Exception as error:  # malformed responses share the same bounded path
                error_type, error_code = _safe_error(error, source="response")
                last_error_type, last_error_code = error_type, error_code
                attempt_audits.append(
                    _attempt_audit(
                        attempt,
                        call_number,
                        "response_error",
                        latency_ms,
                        fields,
                        error_type=error_type,
                        error_code=error_code,
                    )
                )
                continue

            attempt_audits.append(
                _attempt_audit(
                    attempt,
                    call_number,
                    "ok",
                    latency_ms,
                    fields,
                    model_revision=revision,
                )
            )
            metadata: dict[str, MetadataValue] = {
                **_aggregate_audit_metadata(
                    self.config,
                    self._calls_made,
                    attempt_audits,
                    total_latency_ms=total_latency_ms,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    usage_incomplete_attempts=usage_incomplete_attempts,
                ),
                **dict(decision.metadata),
                "completion_tokens": completion_tokens,
                "fallback": False,
                "model": self.config.model,
                "model_revision": revision,
                "parse_status": "ok",
                "prompt_hash": prompt_hash,
                "prompt_tokens": prompt_tokens,
                "prompt_version": self.config.prompt_version,
                "provider": self.config.provider,
                "response_hash": fields.response_hash,
                "response_json": fields.text,
            }
            return Decision(
                decision.action,
                decision.probabilities,
                tuple(sorted(metadata.items())),
            )

        return _fallback_decision(
            observation,
            attempt_audits=attempt_audits,
            error_type=last_error_type,
            error_code=last_error_code,
            latency_ms=total_latency_ms,
            prompt_hash=prompt_hash,
            last_response=last_response,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            usage_incomplete_attempts=usage_incomplete_attempts,
            config=self.config,
            calls_made=self._calls_made,
        )


def observation_payload(observation: Observation) -> dict[str, Any]:
    """Serialize only information visible to the acting player."""

    return {
        "board_cards": list(observation.board_cards),
        "call_amount": observation.call_amount,
        "effective_stack": observation.effective_stack,
        "history": [asdict(action) for action in observation.history],
        "hole_cards": list(observation.hole_cards),
        "legal_actions": [asdict(option) for option in observation.legal_actions],
        "position": observation.position,
        "pot": observation.pot,
        "seat": observation.seat,
        "spr": observation.spr,
        "stacks": list(observation.stacks),
        "street": observation.street,
        "street_bets": list(observation.street_bets),
    }


def build_prompt(observation: Observation, *, prompt_version: str = PROMPT_VERSION) -> str:
    """Build a canonical prompt whose hash is stable across processes."""

    if prompt_version != PROMPT_VERSION:
        raise ValueError(f"unsupported prompt version: {prompt_version}")
    legal_names = [option.name for option in observation.legal_actions]
    instructions = {
        "contract": {
            "action": "Choose exactly one legal action name.",
            "probabilities": (
                "Return every legal action exactly once; values must be finite, "
                "non-negative, and sum to 1."
            ),
            "reason": "Optional concise explanation; do not reveal chain-of-thought.",
        },
        "legal_action_names": legal_names,
        "observation": observation_payload(observation),
        "role": "You are a heads-up no-limit hold'em policy.",
        "version": prompt_version,
    }
    return json.dumps(instructions, sort_keys=True, separators=(",", ":"))


def response_schema(legal_actions: tuple[ActionName, ...]) -> dict[str, Any]:
    """Return a strict JSON Schema specialized to this decision's action set."""

    properties = {
        name: {"type": "number", "minimum": 0.0, "maximum": 1.0}
        for name in legal_actions
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "probabilities"],
        "properties": {
            "action": {"type": "string", "enum": list(legal_actions)},
            "probabilities": {
                "type": "object",
                "additionalProperties": False,
                "required": list(legal_actions),
                "properties": properties,
            },
            "reason": {"type": "string", "maxLength": 500},
        },
    }


def parse_response(text: str, observation: Observation) -> Decision:
    """Strictly parse the small response schema without an extra dependency."""

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ResponseValidationError("invalid_json", "response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ResponseValidationError("response_not_object", "response must be a JSON object")
    allowed_top_level = {"action", "probabilities", "reason"}
    if set(payload).difference(allowed_top_level):
        raise ResponseValidationError(
            "unknown_top_level_fields",
            "response contains unknown top-level fields",
        )
    if "action" not in payload or "probabilities" not in payload:
        raise ResponseValidationError(
            "missing_required_fields",
            "response requires action and probabilities",
        )

    legal_names = tuple(option.name for option in observation.legal_actions)
    action = payload["action"]
    if not isinstance(action, str) or action not in legal_names:
        raise ResponseValidationError("illegal_action", "action is not legal")
    raw_probabilities = payload["probabilities"]
    if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != set(legal_names):
        raise ResponseValidationError(
            "invalid_probability_keys",
            "probabilities must contain every legal action exactly once",
        )

    probabilities: list[tuple[ActionName, float]] = []
    for name in legal_names:
        raw_value = raw_probabilities[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ResponseValidationError(
                "invalid_probability_type",
                "probabilities must be numbers",
            )
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ResponseValidationError(
                "invalid_probability_range",
                "probabilities must be finite and in [0, 1]",
            )
        probabilities.append((name, value))
    if not math.isclose(
        math.fsum(value for _, value in probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ResponseValidationError(
            "invalid_probability_sum",
            "probabilities must sum to one",
        )
    if dict(probabilities)[action] <= 0:
        raise ResponseValidationError(
            "zero_selected_probability",
            "selected action must have positive probability",
        )

    reason = payload.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ResponseValidationError("invalid_reason_type", "reason must be a string")
    if isinstance(reason, str) and len(reason) > 500:
        raise ResponseValidationError("reason_too_long", "reason must not exceed 500 characters")
    return Decision(
        action,
        tuple(probabilities),
        (("reason", reason),) if reason else (),
    )


def _fallback_decision(
    observation: Observation,
    *,
    attempt_audits: list[_AttemptAudit],
    error_type: str,
    error_code: str,
    latency_ms: float,
    prompt_hash: str,
    last_response: str | None,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    usage_incomplete_attempts: int,
    config: LLMPolicyConfig,
    calls_made: int,
) -> Decision:
    selected = next(
        (
            option
            for name in ("check_call", "fold")
            if (option := observation.option(name)) is not None
        ),
        observation.legal_actions[0],
    )
    probabilities = tuple(
        (option.name, float(option.name == selected.name))
        for option in observation.legal_actions
    )
    metadata: dict[str, MetadataValue] = {
        **_aggregate_audit_metadata(
            config,
            calls_made,
            attempt_audits,
            total_latency_ms=latency_ms,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            usage_incomplete_attempts=usage_incomplete_attempts,
        ),
        "error_code": error_code,
        "error_type": error_type,
        "fallback": True,
        "fallback_action": selected.name,
        "fallback_reason_code": error_code,
        "last_response": last_response,
        "model": config.model,
        "parse_status": "fallback",
        "prompt_hash": prompt_hash,
        "prompt_version": config.prompt_version,
        "provider": config.provider,
    }
    return Decision(selected.name, probabilities, tuple(sorted(metadata.items())))


def _aggregate_audit_metadata(
    config: LLMPolicyConfig,
    calls_made: int,
    attempt_audits: list[_AttemptAudit],
    *,
    total_latency_ms: float,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    usage_incomplete_attempts: int,
) -> dict[str, MetadataValue]:
    return {
        "attempt_audit_json": json.dumps(
            [audit.to_dict() for audit in attempt_audits],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "attempts": len(attempt_audits),
        "call_budget_exhausted": calls_made >= config.max_total_calls,
        "calls_made_total": calls_made,
        "calls_remaining": config.max_total_calls - calls_made,
        "deadline_contract": DEADLINE_CONTRACT,
        "expected_model_revision": config.expected_model_revision,
        "llm_latency_ms": round(total_latency_ms, 3),
        "max_total_calls": config.max_total_calls,
        "timeout_seconds": config.timeout_seconds,
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "usage_incomplete_attempts": usage_incomplete_attempts,
    }


def _attempt_audit(
    attempt: int,
    call_number: int,
    status: str,
    latency_ms: float,
    fields: _ResponseAuditFields,
    *,
    error_type: str | None = None,
    error_code: str | None = None,
    deadline_exceeded: bool = False,
    model_revision: str | None = None,
) -> _AttemptAudit:
    return _AttemptAudit(
        attempt=attempt,
        call_number=call_number,
        status=status,
        latency_ms=round(latency_ms, 3),
        deadline_exceeded=deadline_exceeded,
        error_type=error_type,
        error_code=error_code,
        model_revision=model_revision or fields.model_revision,
        prompt_tokens=fields.prompt_tokens,
        completion_tokens=fields.completion_tokens,
        response_hash=fields.response_hash,
        response_text=fields.text,
    )


def _response_audit_fields(response: object) -> _ResponseAuditFields:
    text = _optional_string(response, "text")
    return _ResponseAuditFields(
        text=text,
        model_revision=_optional_string(response, "model_revision"),
        prompt_tokens=_optional_usage(response, "prompt_tokens"),
        completion_tokens=_optional_usage(response, "completion_tokens"),
        response_hash=hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
    )


def _validate_completion_response(
    response: object,
    config: LLMPolicyConfig,
) -> tuple[str, int | None, int | None]:
    if not isinstance(getattr(response, "text", _MISSING), str):
        raise ResponseValidationError(
            "invalid_response_text",
            "completion response text must be a string",
        )
    revision = getattr(response, "model_revision", _MISSING)
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision.strip().lower() == "unspecified"
    ):
        raise ResponseValidationError(
            "missing_model_revision",
            "completion response requires a non-empty model revision",
        )
    if (
        config.expected_model_revision is not None
        and revision != config.expected_model_revision
    ):
        raise ResponseValidationError(
            "model_revision_mismatch",
            "completion response model revision does not match the frozen revision",
        )
    prompt_tokens = _validate_usage(response, "prompt_tokens")
    completion_tokens = _validate_usage(response, "completion_tokens")
    return revision, prompt_tokens, completion_tokens


def _validate_usage(response: object, field: str) -> int | None:
    value = getattr(response, field, _MISSING)
    if value is None:
        return None
    if value is _MISSING or isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponseValidationError(
            f"invalid_{field}",
            f"{field} must be a non-negative integer or None",
        )
    return value


def _optional_string(response: object, field: str) -> str | None:
    value = getattr(response, field, None)
    return value if isinstance(value, str) else None


def _optional_usage(response: object, field: str) -> int | None:
    value = getattr(response, field, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_error(error: Exception, *, source: str) -> tuple[str, str]:
    # Only adapter-owned response validation errors carry trusted fixed codes.
    # A backend can raise an arbitrary exception object and must not control logs.
    if source == "response" and isinstance(error, ResponseValidationError):
        return "ResponseValidationError", error.code
    error_type = _safe_error_type(error)
    if isinstance(error, TimeoutError):
        return error_type, "backend_timeout"
    return error_type, f"{source}_exception"


def _safe_error_type(error: Exception) -> str:
    raw_name = type(error).__name__
    sanitized = "".join(
        character
        if character.isascii() and (character.isalnum() or character == "_")
        else "_"
        for character in raw_name
    )
    return sanitized[:80] or "Exception"


def _validate_non_placeholder(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value.strip().lower() == "unspecified"
    ):
        raise ValueError(f"{field} must be a non-empty concrete identifier")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ResponseValidationError(
                "duplicate_json_key",
                "response JSON contains a duplicate object key",
            )
        payload[key] = value
    return payload


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ResponseValidationError(
        "nonstandard_json_constant",
        f"non-standard JSON constant is forbidden: {value}",
    )
