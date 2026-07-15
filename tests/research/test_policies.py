"""Tests for frozen baseline policy contracts."""

from __future__ import annotations

import pytest

from poker_research.policies import CallingStationPolicy, EquityValueConfig, EquityValuePolicy
from poker_research.types import ActionOption, Observation


def _observation(
    *,
    street: str = "preflop",
    call_amount: int = 1,
    legal_actions: tuple[ActionOption, ...] | None = None,
) -> Observation:
    return Observation(
        hand_id="policy-test",
        seat=1,
        street=street,
        position="button_sb",
        hole_cards=("As", "Ah"),
        board_cards=(),
        stacks=(198, 199),
        street_bets=(2, 1),
        pot=3,
        call_amount=call_amount,
        effective_stack=198,
        spr=66.0,
        legal_actions=legal_actions
        or (
            ActionOption("fold"),
            ActionOption("check_call"),
            ActionOption("raise_min", 4),
        ),
    )


def test_calling_station_always_selects_check_call_and_logs_full_distribution() -> None:
    observation = _observation()
    decision = CallingStationPolicy().decide(observation)

    assert decision.action == "check_call"
    assert dict(decision.probabilities) == {
        "fold": 0.0,
        "check_call": 1.0,
        "raise_min": 0.0,
    }
    assert dict(decision.metadata) == {"policy": "always_check_call"}


def test_calling_station_fails_loudly_if_engine_omits_check_call() -> None:
    observation = _observation(legal_actions=(ActionOption("fold"),))

    with pytest.raises(RuntimeError, match="no check/call"):
        CallingStationPolicy().decide(observation)


def test_observation_option_returns_named_action_or_none() -> None:
    observation = _observation()

    assert observation.option("raise_min") == ActionOption("raise_min", 4)
    assert observation.option("raise_all_in") is None


@pytest.mark.parametrize(
    ("street", "expected"),
    [
        ("preflop", 0.60),
        ("flop", 0.61),
        ("turn", 0.59),
        ("river", 0.53),
    ],
)
def test_equity_policy_uses_preregistered_street_thresholds(
    street: str, expected: float
) -> None:
    policy = EquityValuePolicy(EquityValueConfig())

    assert policy._raise_threshold(street) == expected


def test_equity_policy_rejects_unknown_street() -> None:
    with pytest.raises(ValueError, match="unknown street"):
        EquityValuePolicy()._raise_threshold("fifth_street")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EquityValueConfig(preflop_samples=0), "preflop_samples must be positive"),
        (EquityValueConfig(postflop_samples=-1), "postflop_samples must be positive"),
        (EquityValueConfig(preflop_samples=True), "preflop_samples must be an integer"),
        (EquityValueConfig(call_margin=float("nan")), "call_margin must be finite"),
        (EquityValueConfig(call_margin=-0.01), r"call_margin must be in \[0, 1\]"),
        (
            EquityValueConfig(flop_raise_equity=1.01),
            r"flop_raise_equity must be in \[0, 1\]",
        ),
        (
            EquityValueConfig(strong_equity=0.9, nut_equity=0.8),
            "strong_equity must not exceed nut_equity",
        ),
    ],
)
def test_equity_value_config_rejects_invalid_samples_and_ranges(
    config: EquityValueConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_equity_policy_validates_config_before_first_decision() -> None:
    with pytest.raises(ValueError, match="postflop_samples"):
        EquityValuePolicy(EquityValueConfig(postflop_samples=0))
