from __future__ import annotations

from dataclasses import replace

import pytest

from poker_research.baselines import CallCheckPolicy
from poker_research.preflop import (
    LAG_CHART,
    NIT_CHART,
    TAG_CHART,
    PreflopChartConfig,
    PreflopChartPolicy,
    canonical_starting_hand,
    chen_score,
)
from poker_research.types import ActionOption, Observation, PublicAction

LEGAL = (
    ActionOption("fold"),
    ActionOption("check_call"),
    ActionOption("raise_min", 4),
    ActionOption("raise_half_pot", 6),
    ActionOption("raise_pot", 8),
)


def _observation(
    hole: tuple[str, str],
    *,
    street: str = "preflop",
    history: tuple[PublicAction, ...] = (),
    call_amount: int = 1,
    position: str = "button_sb",
) -> Observation:
    board = () if street == "preflop" else ("2c", "7d", "Th")
    return Observation(
        hand_id="chart-test",
        seat=1 if position == "button_sb" else 0,
        street=street,
        position=position,
        hole_cards=hole,
        board_cards=board,
        stacks=(198, 199),
        street_bets=(2, 1),
        pot=3,
        call_amount=call_amount,
        effective_stack=198,
        spr=66.0,
        legal_actions=LEGAL,
        history=history,
    )


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        (("As", "Ah"), "AA"),
        (("Kh", "As"), "AKo"),
        (("2s", "As"), "A2s"),
        (("Td", "9d"), "T9s"),
    ],
)
def test_canonical_starting_hand(cards: tuple[str, str], expected: str) -> None:
    assert canonical_starting_hand(cards) == expected


def test_chen_score_orders_intuitive_hands() -> None:
    assert chen_score(("As", "Ah")) > chen_score(("As", "Ks"))
    assert chen_score(("As", "Ks")) > chen_score(("7c", "2d"))
    assert chen_score(("Td", "9d")) > chen_score(("Tc", "3d"))


@pytest.mark.parametrize("cards", [("As", "As"), ("ZZ", "Kh"), ("A", "Kh")])
def test_hand_helpers_reject_invalid_cards(cards: tuple[str, str]) -> None:
    with pytest.raises(ValueError):
        canonical_starting_hand(cards)


def test_tag_chart_raises_premium_and_folds_trash_facing_raise() -> None:
    policy = PreflopChartPolicy(CallCheckPolicy(), config=TAG_CHART)
    facing_raise = (PublicAction(0, "raise_min", 4),)

    premium = policy.decide(_observation(("As", "Ah"), history=facing_raise))
    trash = policy.decide(_observation(("7c", "2d"), history=facing_raise))

    assert premium.action == "raise_pot"
    assert trash.action == "fold"
    assert dict(premium.metadata)["hand_class"] == "AA"
    assert dict(trash.metadata)["chart_context"] == "facing_raise"


def test_lag_opens_wider_than_nit() -> None:
    hand = ("8c", "6c")
    lag = PreflopChartPolicy(CallCheckPolicy(), config=LAG_CHART, name="chart_lag_v1")
    nit = PreflopChartPolicy(CallCheckPolicy(), config=NIT_CHART, name="chart_nit_v1")

    assert lag.decide(_observation(hand)).action.startswith("raise_")
    assert nit.decide(_observation(hand)).action == "fold"


def test_free_check_is_never_folded() -> None:
    policy = PreflopChartPolicy(CallCheckPolicy(), config=NIT_CHART)

    decision = policy.decide(_observation(("7c", "2d"), call_amount=0, position="big_blind"))

    assert decision.action == "check_call"


def test_postflop_delegates_and_preserves_full_distribution() -> None:
    policy = PreflopChartPolicy(CallCheckPolicy(name="postflop_call_v1"))
    observation = replace(
        _observation(("As", "Kh"), street="flop"),
        legal_actions=(ActionOption("fold"), ActionOption("check_call")),
    )

    decision = policy.decide(observation)

    assert decision.action == "check_call"
    assert dict(decision.probabilities) == {"fold": 0.0, "check_call": 1.0}
    assert dict(decision.metadata)["postflop_policy"] == "postflop_call_v1"


@pytest.mark.parametrize(
    "config",
    [
        PreflopChartConfig(button_open=-1),
        PreflopChartConfig(facing_raise_call=12, facing_raise_reraise=10),
        PreflopChartConfig(open_sizing=("check_call",)),
    ],
)
def test_chart_config_validation(config: PreflopChartConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
