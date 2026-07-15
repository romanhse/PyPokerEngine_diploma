"""Contract tests for compositional and explicitly adaptive policies."""

from __future__ import annotations

import math
import random

import pytest

from poker_research.adaptive import (
    AdaptiveExploitPolicy,
    AdaptiveThresholds,
    EpsilonExplorationPolicy,
    MixedPolicy,
    OpponentStatsTracker,
    UCBMetaPolicy,
    mixture,
)
from poker_research.types import ActionOption, Decision, Observation, PublicAction


class _PointPolicy:
    def __init__(self, name: str, action: str) -> None:
        self.name = name
        self.action = action
        self.calls = 0

    def decide(self, observation: Observation) -> Decision:
        self.calls += 1
        return Decision(
            self.action,
            tuple(
                (option.name, float(option.name == self.action))
                for option in observation.legal_actions
            ),
            (("point_policy", self.name),),
        )

    def reset(self) -> None:
        self.calls = 0


class _MalformedPolicy:
    name = "malformed"

    def decide(self, observation: Observation) -> Decision:
        return Decision("check_call", (("check_call", 1.0),))


def _observation(
    *,
    hand_id: str = "adaptive-test",
    seat: int = 0,
    history: tuple[PublicAction, ...] = (),
    legal_actions: tuple[ActionOption, ...] | None = None,
) -> Observation:
    return Observation(
        hand_id=hand_id,
        seat=seat,
        street="river",
        position="big_blind" if seat == 0 else "button_sb",
        hole_cards=("As", "Ah"),
        board_cards=("2c", "3d", "4h", "5s", "9c"),
        stacks=(150, 150),
        street_bets=(0, 0),
        pot=100,
        call_amount=0,
        effective_stack=150,
        spr=1.5,
        legal_actions=legal_actions
        or (
            ActionOption("fold"),
            ActionOption("check_call"),
            ActionOption("raise_min", 20),
        ),
        history=history,
    )


def _fold_to_bet_history() -> tuple[PublicAction, ...]:
    return (
        PublicAction(1, "check_call", None),
        PublicAction(0, "check_call", None),
        PublicAction(0, "raise_half_pot", 4),
        PublicAction(1, "fold", None),
    )


def test_mixed_policy_aggregates_full_distributions_and_replays_seed() -> None:
    fold = _PointPolicy("fold_point", "fold")
    call = _PointPolicy("call_point", "check_call")
    policy = MixedPolicy((fold, call), (1.0, 3.0), seed=1)
    observation = _observation(
        legal_actions=(ActionOption("fold"), ActionOption("check_call"))
    )

    first = policy.decide(observation)

    assert dict(first.probabilities) == {"fold": 0.25, "check_call": 0.75}
    assert dict(first.probabilities)[first.action] > 0
    assert dict(first.metadata) == {
        "composition": "mixture",
        "components": "fold_point,call_point",
        "weights": "0.25,0.75",
    }
    assert fold.calls == call.calls == 1

    continuation = policy.clone(reset_state=False)
    assert policy.decide(observation) == continuation.decide(observation)

    policy.reset()
    assert policy.decide(observation) == first
    assert fold.calls == call.calls == 1


def test_mixture_rng_is_reproducible_across_fresh_per_hand_instances() -> None:
    def fresh_policy() -> MixedPolicy:
        return MixedPolicy(
            (
                _PointPolicy("fresh_fold", "fold"),
                _PointPolicy("fresh_call", "check_call"),
            ),
            (1.0, 1.0),
            seed=4_271,
        )

    observations = tuple(
        _observation(
            hand_id=f"fresh-mixture-{index}",
            legal_actions=(ActionOption("fold"), ActionOption("check_call")),
        )
        for index in range(32)
    )
    reused = fresh_policy()
    replay = fresh_policy()
    random.seed(8_119)
    global_state = random.getstate()

    reused_actions = [reused.decide(observation).action for observation in observations]
    replay_actions = [replay.decide(observation).action for observation in observations]
    fresh_actions = [fresh_policy().decide(observation).action for observation in observations]

    assert reused_actions == replay_actions == fresh_actions
    assert len(set(fresh_actions)) == 2
    assert random.getstate() == global_state


def test_mixture_convenience_constructor_and_weight_validation() -> None:
    fold = _PointPolicy("fold", "fold")
    call = _PointPolicy("call", "check_call")

    policy = mixture((2.0, fold), (1.0, call), seed=9, name="two_to_one")

    assert isinstance(policy, MixedPolicy)
    assert policy.name == "two_to_one"
    assert policy.weights == pytest.approx((2 / 3, 1 / 3))
    with pytest.raises(ValueError, match="at least one"):
        MixedPolicy((fold, call), (0.0, 0.0))
    with pytest.raises(ValueError, match="equal length"):
        MixedPolicy((fold, call), (1.0,))
    with pytest.raises(ValueError, match="unique"):
        MixedPolicy((fold, _PointPolicy("fold", "check_call")), (1.0, 1.0))


def test_compositions_reject_malformed_child_distributions() -> None:
    observation = _observation()

    with pytest.raises(ValueError, match="cover every legal action"):
        MixedPolicy((_MalformedPolicy(),), (1.0,)).decide(observation)
    with pytest.raises(ValueError, match="cover every legal action"):
        EpsilonExplorationPolicy(_MalformedPolicy(), 0.1).decide(observation)


def test_epsilon_wrapper_mixes_with_uniform_and_has_resettable_rng() -> None:
    base = _PointPolicy("always_call", "check_call")
    observation = _observation(
        legal_actions=(ActionOption("fold"), ActionOption("check_call"))
    )
    policy = EpsilonExplorationPolicy(base, 0.2, seed=0)

    first = policy.decide(observation)

    assert first.action == "check_call"
    assert dict(first.probabilities) == pytest.approx({"fold": 0.1, "check_call": 0.9})
    assert dict(first.metadata) == {
        "base_action": "check_call",
        "base_policy": "always_call",
        "composition": "epsilon_exploration",
        "epsilon": 0.2,
    }
    policy.reset()
    assert policy.decide(observation) == first
    assert base.calls == 1

    with pytest.raises(ValueError, match="epsilon"):
        EpsilonExplorationPolicy(base, math.nan)
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonExplorationPolicy(base, 1.01)


def test_epsilon_rng_matches_fresh_per_hand_factory_semantics() -> None:
    def fresh_policy() -> EpsilonExplorationPolicy:
        return EpsilonExplorationPolicy(
            _PointPolicy("epsilon_base", "check_call"),
            1.0,
            seed=6_181,
        )

    observations = tuple(
        _observation(
            hand_id=f"fresh-epsilon-{index}",
            legal_actions=(ActionOption("fold"), ActionOption("check_call")),
        )
        for index in range(32)
    )
    reused = fresh_policy()

    reused_actions = [reused.decide(observation).action for observation in observations]
    fresh_actions = [fresh_policy().decide(observation).action for observation in observations]

    assert reused_actions == fresh_actions
    assert len(set(fresh_actions)) == 2


def test_opponent_tracker_reconstructs_vpip_pfr_aggression_and_fold_to_bet() -> None:
    raised_then_folded = (
        PublicAction(1, "raise_pot", 6),
        PublicAction(0, "check_call", None),
        PublicAction(0, "raise_half_pot", 4),
        PublicAction(1, "fold", None),
    )
    limped_called_then_bet = (
        PublicAction(1, "check_call", None),
        PublicAction(0, "check_call", None),
        PublicAction(0, "raise_half_pot", 4),
        PublicAction(1, "check_call", None),
        PublicAction(0, "check_call", None),
        PublicAction(1, "raise_min", 4),
        PublicAction(0, "check_call", None),
    )
    tracker = OpponentStatsTracker()

    tracker.update(_observation(hand_id="raised", history=raised_then_folded))
    stats = tracker.update(_observation(hand_id="limped", history=limped_called_then_bet))
    repeated = tracker.update(
        _observation(hand_id="limped", history=limped_called_then_bet)
    )

    assert repeated == stats
    assert stats.hands == 2
    assert stats.vpip_hands == 2
    assert stats.pfr_hands == 1
    assert stats.vpip_rate == 1.0
    assert stats.pfr_rate == 0.5
    assert stats.postflop_aggressive_actions == 1
    assert stats.postflop_calls == 1
    assert stats.aggression_factor == 1.0
    assert stats.aggression_frequency == 0.5
    assert stats.fold_to_bet_opportunities == 2
    assert stats.folds_to_bet == 1
    assert stats.fold_to_bet_rate == 0.5


def test_tracker_counts_big_blind_call_facing_raise_as_vpip_without_pfr() -> None:
    tracker = OpponentStatsTracker()
    history = (
        PublicAction(1, "raise_pot", 6),
        PublicAction(0, "check_call", None),
    )

    stats = tracker.update(_observation(hand_id="bb-call", seat=1, history=history))

    assert stats.vpip_rate == 1.0
    assert stats.pfr_rate == 0.0


def test_tracker_rejects_divergent_hand_history_and_reset_clone_are_explicit() -> None:
    tracker = OpponentStatsTracker()
    full = _fold_to_bet_history()
    tracker.update(_observation(hand_id="same", history=full))

    retained = tracker.clone(reset_state=False)
    fresh = tracker.clone()

    assert retained.snapshot() == tracker.snapshot()
    assert fresh.snapshot().hands == 0
    with pytest.raises(ValueError, match="extend"):
        tracker.update(_observation(hand_id="same", history=full[:-1]))
    with pytest.raises(ValueError, match="heads-up seat"):
        tracker.update(_observation(hand_id="bad-seat", seat=4))
    tracker.reset()
    assert tracker.snapshot().hands == 0


def test_explicit_post_hand_history_hook_captures_terminal_opponent_action() -> None:
    tracker = OpponentStatsTracker()
    complete = _fold_to_bet_history()

    partial = tracker.update(
        _observation(hand_id="terminal-fold", history=complete[:-1])
    )
    finished = tracker.update_public_history("terminal-fold", 0, complete)

    assert partial.fold_to_bet_opportunities == 0
    assert partial.folds_to_bet == 0
    assert finished.fold_to_bet_opportunities == 1
    assert finished.folds_to_bet == 1


def test_adaptive_exploit_uses_frozen_public_stats_and_reports_profile() -> None:
    tracker = OpponentStatsTracker()
    tracker.update(_observation(hand_id="fold-1", history=_fold_to_bet_history()))
    tracker.update(_observation(hand_id="fold-2", history=_fold_to_bet_history()))
    blueprint = _PointPolicy("blueprint", "check_call")
    overfolder = _PointPolicy("overfolder_exploit", "raise_min")
    adaptive = AdaptiveExploitPolicy(
        blueprint,
        overfolder_policy=overfolder,
        tracker=tracker,
        thresholds=AdaptiveThresholds(
            minimum_hands=2,
            minimum_fold_opportunities=2,
            minimum_aggression_opportunities=2,
            overfold_rate=0.75,
        ),
        observe_online=False,
    )

    decision = adaptive.decide(_observation(hand_id="frozen-evaluation"))

    assert decision.action == "raise_min"
    metadata = dict(decision.metadata)
    assert metadata["adaptive_profile"] == "overfolder"
    assert metadata["adaptive_policy"] == "overfolder_exploit"
    assert metadata["observed_hands"] == 2
    assert metadata["fold_to_bet"] == 1.0
    assert tracker.snapshot().hands == 2
    assert blueprint.calls == 0
    assert overfolder.calls == 1

    retained = adaptive.clone(reset_state=False)
    fresh = adaptive.clone()
    assert retained.tracker.snapshot().hands == 2
    assert fresh.tracker.snapshot().hands == 0


def test_adaptive_exploit_falls_back_until_evidence_threshold() -> None:
    blueprint = _PointPolicy("blueprint", "check_call")
    exploit = _PointPolicy("premature_exploit", "raise_min")
    adaptive = AdaptiveExploitPolicy(
        blueprint,
        loose_passive_policy=exploit,
        overfolder_policy=exploit,
        aggressive_policy=exploit,
        thresholds=AdaptiveThresholds(minimum_hands=3),
    )

    decision = adaptive.decide(
        _observation(hand_id="one-hand", history=_fold_to_bet_history())
    )

    assert decision.action == "check_call"
    assert dict(decision.metadata)["adaptive_profile"] == "insufficient_evidence"
    assert blueprint.calls == 1
    assert exploit.calls == 0


def test_ucb_meta_policy_requires_explicit_once_per_hand_reward_updates() -> None:
    fold = _PointPolicy("fold_arm", "fold")
    call = _PointPolicy("call_arm", "check_call")
    policy = UCBMetaPolicy((fold, call), seed=3)
    first_observation = _observation(hand_id="ucb-1")

    first = policy.decide(first_observation)
    repeated = policy.decide(first_observation)

    assert dict(first.metadata)["selected_arm"] == dict(repeated.metadata)["selected_arm"]
    assert sum(arm.pulls for arm in policy.arm_statistics()) == 0

    policy.update_reward("ucb-1", 2.0)
    second = policy.decide(_observation(hand_id="ucb-2"))
    assert dict(first.metadata)["selected_arm"] != dict(second.metadata)["selected_arm"]
    policy.update_reward("ucb-2", -1.0)

    statistics = policy.arm_statistics()
    assert {arm.policy for arm in statistics} == {"fold_arm", "call_arm"}
    assert {arm.pulls for arm in statistics} == {1}
    assert {arm.mean_reward for arm in statistics} == {2.0, -1.0}
    with pytest.raises(ValueError, match="already recorded"):
        policy.update_reward("ucb-1", 0.0)
    with pytest.raises(KeyError, match="no UCB assignment"):
        policy.update_reward("never-selected", 0.0)
    with pytest.raises(ValueError, match="finite"):
        policy.update_reward("ucb-3", math.inf)

    retained = policy.clone(reset_state=False)
    fresh = policy.clone()
    assert retained.arm_statistics() == policy.arm_statistics()
    assert all(arm.pulls == 0 and arm.mean_reward is None for arm in fresh.arm_statistics())


def test_ucb_initial_arm_rng_matches_fresh_per_hand_factory_semantics() -> None:
    def fresh_policy() -> UCBMetaPolicy:
        return UCBMetaPolicy(
            (
                _PointPolicy("fresh_ucb_fold", "fold"),
                _PointPolicy("fresh_ucb_call", "check_call"),
            ),
            seed=2_903,
        )

    observations = tuple(
        _observation(
            hand_id=f"fresh-ucb-{index}",
            legal_actions=(ActionOption("fold"), ActionOption("check_call")),
        )
        for index in range(32)
    )
    reused = fresh_policy()

    reused_arms = [
        dict(reused.decide(observation).metadata)["selected_arm"]
        for observation in observations
    ]
    fresh_arms = [
        dict(fresh_policy().decide(observation).metadata)["selected_arm"]
        for observation in observations
    ]

    assert reused_arms == fresh_arms
    assert len(set(reused_arms)) == 2


@pytest.mark.parametrize(
    "thresholds",
    [
        AdaptiveThresholds(minimum_hands=0),
        AdaptiveThresholds(minimum_fold_opportunities=0),
        AdaptiveThresholds(loose_vpip=1.1),
        AdaptiveThresholds(overfold_rate=math.nan),
    ],
)
def test_adaptive_thresholds_reject_invalid_preregistration(
    thresholds: AdaptiveThresholds,
) -> None:
    with pytest.raises(ValueError):
        thresholds.validate()
