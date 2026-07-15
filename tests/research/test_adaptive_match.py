"""Tests for the separate sequential stateful-policy match runner."""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from poker_research.adaptive import AdaptiveExploitPolicy, UCBMetaPolicy
from poker_research.adaptive_match import AdaptiveMatchConfig, run_adaptive_match
from poker_research.arena import ArenaConfig
from poker_research.types import Decision, Observation, PublicAction


class _PointPolicy:
    def __init__(self, name: str, action: str = "check_call") -> None:
        self.name = name
        self.action = action

    def decide(self, observation: Observation) -> Decision:
        return Decision(
            self.action,
            tuple(
                (option.name, float(option.name == self.action))
                for option in observation.legal_actions
            ),
        )


class _HookedCheckCallPolicy(_PointPolicy):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.first_decision_reward_counts: dict[str, int] = {}
        self.histories: list[tuple[str, int, tuple[PublicAction, ...]]] = []
        self.rewards: list[tuple[str, float]] = []

    def decide(self, observation: Observation) -> Decision:
        self.first_decision_reward_counts.setdefault(observation.hand_id, len(self.rewards))
        return super().decide(observation)

    def update_public_history(
        self,
        hand_id: str,
        hero_seat: int,
        history: Sequence[PublicAction],
    ) -> object:
        self.histories.append((hand_id, hero_seat, tuple(history)))
        return None

    def update_reward(self, hand_id: str, reward: float) -> None:
        self.rewards.append((hand_id, reward))


def _config(*, pair_count: int = 3) -> AdaptiveMatchConfig:
    return AdaptiveMatchConfig(
        pair_count=pair_count,
        master_seed=2_026_0715,
        match_id="pytest-adaptive",
        game=ArenaConfig(starting_stack=20, small_blind=1, big_blind=2),
    )


def test_runner_reuses_objects_swaps_seats_and_batches_terminal_hooks() -> None:
    policy_a = _HookedCheckCallPolicy("stateful_a")
    policy_b = _HookedCheckCallPolicy("stateful_b")
    random.seed(91_317)
    global_state = random.getstate()

    result = run_adaptive_match(policy_a, policy_b, config=_config())

    assert result.policy_names == ("stateful_a", "stateful_b")
    assert len(result.pairs) == 3
    assert len(result.hands) == 6
    assert [pair.play_order for pair in result.pairs] == [
        ("a", "b"),
        ("b", "a"),
        ("a", "b"),
    ]
    assert random.getstate() == global_state
    assert len(policy_a.histories) == len(policy_b.histories) == 6
    assert len(policy_a.rewards) == len(policy_b.rewards) == 6
    assert [seat for _, seat, _ in policy_a.histories] == [0, 1, 0, 1, 0, 1]
    assert [seat for _, seat, _ in policy_b.histories] == [1, 0, 1, 0, 1, 0]

    expected_pre_pair_rewards = [0, 0, 2, 2, 4, 4]
    ordered_hand_ids = [hand.hand_id for hand in result.hands]
    assert [
        policy_a.first_decision_reward_counts[hand_id] for hand_id in ordered_hand_ids
    ] == expected_pre_pair_rewards
    assert [
        policy_b.first_decision_reward_counts[hand_id] for hand_id in ordered_hand_ids
    ] == expected_pre_pair_rewards

    for pair, leg_a, leg_b in zip(
        result.pairs,
        result.hands[::2],
        result.hands[1::2],
        strict=True,
    ):
        assert pair.deck_hash == leg_a.deck_hash == leg_b.deck_hash
        assert pair.deal_seed == leg_a.deal_seed == leg_b.deal_seed
        assert pair.policy_a_net_chips == leg_a.payoffs[0] + leg_b.payoffs[1]
        assert pair.policy_a_bb100 == 25.0 * pair.policy_a_net_chips

    expected_a_rewards = [
        reward
        for leg_a, leg_b in zip(result.hands[::2], result.hands[1::2], strict=True)
        for reward in (float(leg_a.payoffs[0]), float(leg_b.payoffs[1]))
    ]
    assert [reward for _, reward in policy_a.rewards] == expected_a_rewards
    assert all(
        history == hand.public_actions
        for (_, _, history), hand in zip(policy_a.histories, result.hands, strict=True)
    )


def test_runner_drives_real_public_history_and_ucb_reward_hooks() -> None:
    adaptive = AdaptiveExploitPolicy(
        _PointPolicy("adaptive_blueprint"),
        name="adaptive_under_test",
    )
    ucb = UCBMetaPolicy(
        (
            _PointPolicy("ucb_call"),
            _PointPolicy("ucb_call_alternative"),
        ),
        seed=37,
        name="ucb_under_test",
    )

    result = run_adaptive_match(adaptive, ucb, config=_config(pair_count=2))

    assert len(result.pairs) == 2
    assert adaptive.tracker.snapshot().hands == 4
    assert sum(arm.pulls for arm in ucb.arm_statistics()) == 4


@pytest.mark.parametrize(
    ("first", "second", "config", "message"),
    [
        (
            _PointPolicy("same"),
            _PointPolicy("same"),
            _config(),
            "names must be distinct",
        ),
        (
            _PointPolicy("a"),
            _PointPolicy("b"),
            AdaptiveMatchConfig(pair_count=0),
            "pair_count",
        ),
        (
            _PointPolicy("a"),
            _PointPolicy("b"),
            AdaptiveMatchConfig(match_id="bad id"),
            "match_id",
        ),
    ],
)
def test_runner_rejects_invalid_match_contracts(
    first: _PointPolicy,
    second: _PointPolicy,
    config: AdaptiveMatchConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_adaptive_match(first, second, config=config)


def test_runner_rejects_the_same_policy_object_in_both_seats() -> None:
    policy = _PointPolicy("shared")

    with pytest.raises(ValueError, match="distinct policy objects"):
        run_adaptive_match(policy, policy, config=_config(pair_count=1))
