"""Contract tests for the authoritative PokerKit heads-up arena."""

from __future__ import annotations

import random
from dataclasses import fields

import pytest

from poker_research.arena import ArenaConfig, HandResult, play_hand
from poker_research.types import ActionOption, Decision, Observation
from tests.research.helpers import (
    RaiseMinimumOncePolicy,
    RecordingCheckCallPolicy,
    deterministic_decision,
)


def _semantic_result(result: HandResult) -> tuple[object, ...]:
    """Discard wall-clock latency while retaining every game-relevant field."""

    decisions = tuple(
        (
            record.index,
            record.seat,
            record.policy,
            record.street,
            record.hole_cards,
            record.board_cards,
            record.pot,
            record.call_amount,
            record.stacks,
            record.street_bets,
            record.legal_actions,
            record.decision,
        )
        for record in result.decisions
    )
    return (
        result.hand_id,
        result.deal_seed,
        result.policy_names,
        result.deck_hash,
        result.hole_cards,
        result.board_cards,
        result.starting_stacks,
        result.finishing_stacks,
        result.payoffs,
        decisions,
        result.public_actions,
        result.hand_history.dumps(),
    )


def test_heads_up_button_blinds_and_action_order_match_pokerkit() -> None:
    big_blind = RecordingCheckCallPolicy("big_blind_recorder")
    button = RecordingCheckCallPolicy("button_recorder")

    result = play_hand((big_blind, button), deal_seed=11, hand_id="order")

    first = button.observations[0]
    assert first.seat == 1
    assert first.position == "button_sb"
    assert first.street == "preflop"
    assert first.stacks == (198, 199)
    assert first.street_bets == (2, 1)
    assert first.pot == 3
    assert first.call_amount == 1
    assert result.decisions[0].seat == 1
    assert result.public_actions[0].action == "check_call"
    assert result.public_actions[0].amount == 1

    first_flop = next(
        observation
        for observation in big_blind.observations
        if observation.street == "flop"
    )
    assert first_flop.seat == 0
    assert first_flop.position == "big_blind"
    assert len(first_flop.board_cards) == 3
    assert [record.seat for record in result.decisions if record.street == "flop"] == [0, 1]
    assert [record.seat for record in result.decisions if record.street == "turn"] == [0, 1]
    assert [record.seat for record in result.decisions if record.street == "river"] == [0, 1]
    first_flop_action = next(
        action
        for action, record in zip(result.public_actions, result.decisions, strict=True)
        if record.street == "flop"
    )
    assert first_flop_action.amount == 0


def test_same_seed_and_policies_replay_exactly() -> None:
    first = play_hand(
        (RecordingCheckCallPolicy("seat_zero"), RecordingCheckCallPolicy("seat_one")),
        deal_seed=987654321,
        hand_id="deterministic",
    )
    second = play_hand(
        (RecordingCheckCallPolicy("seat_zero"), RecordingCheckCallPolicy("seat_one")),
        deal_seed=987654321,
        hand_id="deterministic",
    )

    assert _semantic_result(first) == _semantic_result(second)


def test_deal_seeding_does_not_mutate_process_global_rng() -> None:
    random.seed(331)
    expected_before = random.random()
    expected_after = random.random()

    random.seed(331)
    assert random.random() == expected_before
    play_hand(
        (RecordingCheckCallPolicy("rng_zero"), RecordingCheckCallPolicy("rng_one")),
        deal_seed=73,
        hand_id="rng-isolation",
    )
    assert random.random() == expected_after


def test_same_deal_survives_policy_seat_swap() -> None:
    first = play_hand(
        (RecordingCheckCallPolicy("alice"), RecordingCheckCallPolicy("bob")),
        deal_seed=20260715,
        hand_id="duplicate-a",
    )
    swapped = play_hand(
        (RecordingCheckCallPolicy("bob"), RecordingCheckCallPolicy("alice")),
        deal_seed=20260715,
        hand_id="duplicate-b",
    )

    assert first.deck_hash == swapped.deck_hash
    assert first.hole_cards == swapped.hole_cards
    assert first.board_cards == swapped.board_cards
    assert first.policy_names == tuple(reversed(swapped.policy_names))


@pytest.mark.parametrize("deal_seed", [0, 1, 2, 101, 2**63 - 1])
def test_terminal_results_conserve_chips_and_are_zero_sum(deal_seed: int) -> None:
    config = ArenaConfig(starting_stack=80, small_blind=1, big_blind=2)
    result = play_hand(
        (RecordingCheckCallPolicy("zero"), RecordingCheckCallPolicy("one")),
        deal_seed=deal_seed,
        hand_id=f"conservation-{deal_seed}",
        config=config,
    )

    assert sum(result.starting_stacks) == 160
    assert sum(result.finishing_stacks) == 160
    assert sum(result.payoffs) == 0
    assert result.payoffs == tuple(
        finish - start
        for finish, start in zip(result.finishing_stacks, result.starting_stacks, strict=True)
    )


def test_initial_raise_abstraction_has_unique_in_bounds_targets() -> None:
    big_blind = RecordingCheckCallPolicy("bounds_big_blind")
    button = RecordingCheckCallPolicy("bounds_button")
    play_hand((big_blind, button), deal_seed=5, hand_id="raise-bounds")

    first = button.observations[0]
    assert first.legal_actions == (
        ActionOption("fold"),
        ActionOption("check_call"),
        ActionOption("raise_min", 4),
        ActionOption("raise_pot", 6),
        ActionOption("raise_2pot", 10),
        ActionOption("raise_all_in", 200),
    )

    for observation in (*big_blind.observations, *button.observations):
        raises = [option for option in observation.legal_actions if option.raise_to is not None]
        targets = [option.raise_to for option in raises]
        assert len(targets) == len(set(targets))
        assert all(
            max(observation.street_bets) < target
            <= observation.street_bets[observation.seat] + observation.stacks[observation.seat]
            for target in targets
        )


def test_valid_minimum_raise_is_applied_and_logged() -> None:
    button = RaiseMinimumOncePolicy("minimum_raiser")
    result = play_hand(
        (RecordingCheckCallPolicy("raise_caller"), button),
        deal_seed=4,
        hand_id="raise-min",
    )

    assert result.public_actions[0].seat == 1
    assert result.public_actions[0].action == "raise_min"
    assert result.public_actions[0].amount == 4
    assert result.decisions[0].decision.action == "raise_min"
    assert sum(result.payoffs) == 0


class _MalformedPolicy:
    name = "malformed"

    def __init__(self, case: str) -> None:
        self.case = case

    def decide(self, observation: Observation) -> Decision:
        valid = deterministic_decision(observation, "check_call")
        if self.case == "illegal_action":
            return Decision("teleport", valid.probabilities)
        if self.case == "missing_probability":
            return Decision("check_call", valid.probabilities[:-1])
        if self.case == "duplicate_probability":
            return Decision("check_call", (*valid.probabilities, valid.probabilities[0]))
        if self.case == "outside_unit_interval":
            probabilities = tuple(
                (name, 1.1 if name == "check_call" else 0.0)
                for name, _ in valid.probabilities
            )
            return Decision("check_call", probabilities)
        if self.case == "not_normalized":
            probabilities = tuple(
                (name, 0.9 if name == "check_call" else 0.0)
                for name, _ in valid.probabilities
            )
            return Decision("check_call", probabilities)
        if self.case == "selected_zero":
            fallback = next(name for name, _ in valid.probabilities if name != "check_call")
            probabilities = tuple(
                (name, 1.0 if name == fallback else 0.0)
                for name, _ in valid.probabilities
            )
            return Decision("check_call", probabilities)
        if self.case == "nan_probability":
            probabilities = tuple(
                (name, float("nan") if name == "check_call" else 0.0)
                for name, _ in valid.probabilities
            )
            return Decision("check_call", probabilities)
        raise AssertionError(f"unknown malformed case: {self.case}")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("illegal_action", "illegal action"),
        ("missing_probability", "cover every legal action"),
        ("duplicate_probability", "duplicate probability"),
        ("outside_unit_interval", r"in \[0, 1\]"),
        ("not_normalized", "sum to one"),
        ("selected_zero", "positive probability"),
        ("nan_probability", "finite"),
    ],
)
def test_malformed_policy_decisions_are_rejected(case: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        play_hand(
            (RecordingCheckCallPolicy("valid"), _MalformedPolicy(case)),
            deal_seed=8,
            hand_id=f"malformed-{case}",
        )


def test_policy_observation_excludes_engine_and_opponent_private_cards() -> None:
    zero = RecordingCheckCallPolicy("privacy_zero")
    one = RecordingCheckCallPolicy("privacy_one")
    result = play_hand((zero, one), deal_seed=91, hand_id="privacy")

    observation_fields = {field.name for field in fields(Observation)}
    assert "state" not in observation_fields
    assert "deck_cards" not in observation_fields
    assert "opponent_hole_cards" not in observation_fields
    for record in result.decisions:
        assert record.hole_cards == result.hole_cards[record.seat]
        assert not hasattr((zero, one)[record.seat].observations[0], "state")
        opponent_hole = set(result.hole_cards[1 - record.seat])
        assert opponent_hole.isdisjoint(record.hole_cards)
        assert set(record.board_cards).isdisjoint(opponent_hole)


@pytest.mark.parametrize(
    "config",
    [
        ArenaConfig(starting_stack=0),
        ArenaConfig(small_blind=0),
        ArenaConfig(small_blind=3, big_blind=2),
        ArenaConfig(starting_stack=1, big_blind=2),
        ArenaConfig(ante=-1),
    ],
)
def test_invalid_arena_config_is_rejected(config: ArenaConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_arena_requires_two_distinct_policy_names() -> None:
    only = RecordingCheckCallPolicy("only")
    with pytest.raises(ValueError, match="exactly two"):
        play_hand((only,), deal_seed=1, hand_id="too-few")

    with pytest.raises(ValueError, match="names must be distinct"):
        play_hand(
            (RecordingCheckCallPolicy("same"), RecordingCheckCallPolicy("same")),
            deal_seed=1,
            hand_id="same-name",
        )
