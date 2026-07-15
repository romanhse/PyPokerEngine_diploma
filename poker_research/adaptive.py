"""Composable and explicitly stateful policies for controlled adaptation experiments.

The authoritative arena intentionally exposes no terminal callback.  Consequently this
module never pretends that rewards are learned implicitly: opponent statistics are derived
only from public histories, while terminal actions and reward-based meta-selection use
explicit hooks that an experiment runner must call.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from poker_research.equity import stable_seed
from poker_research.types import Decision, MetadataValue, Observation, Policy, PublicAction


@runtime_checkable
class ResettablePolicy(Protocol):
    """Optional lifecycle supported by stateful policies in this module."""

    def reset(self) -> None:
        """Return runtime state and RNGs to their initial state."""


def _reset_children(policies: Sequence[Policy]) -> None:
    seen: set[int] = set()
    for policy in policies:
        identity = id(policy)
        if identity not in seen and isinstance(policy, ResettablePolicy):
            policy.reset()
        seen.add(identity)


def _probabilities(observation: Observation, decision: Decision) -> dict[str, float]:
    """Validate a child policy before composing its distribution."""

    legal = tuple(option.name for option in observation.legal_actions)
    if not legal:
        raise ValueError("an observation must contain at least one legal action")
    if len(set(legal)) != len(legal):
        raise ValueError("legal action names must be unique")
    probabilities = dict(decision.probabilities)
    if len(probabilities) != len(decision.probabilities):
        raise ValueError("child decision contains duplicate probability entries")
    if set(probabilities) != set(legal):
        raise ValueError("child probabilities must cover every legal action exactly once")
    if decision.action not in probabilities:
        raise ValueError(f"child selected illegal action: {decision.action!r}")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
        raise ValueError("child probabilities must be finite values in [0, 1]")
    if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-9):
        raise ValueError("child probabilities must sum to one")
    if probabilities[decision.action] <= 0:
        raise ValueError("child selected action must have positive probability")
    return probabilities


def _sample_action(
    observation: Observation,
    probabilities: dict[str, float],
    rng: random.Random,
) -> str:
    draw = rng.random()
    cumulative = 0.0
    for option in observation.legal_actions:
        cumulative += probabilities[option.name]
        if draw < cumulative:
            return option.name
    return observation.legal_actions[-1].name


def _decision_rng(seed: int, namespace: str, observation: Observation) -> random.Random:
    """Return a local RNG keyed only by the explicit seed and public decision identity."""

    return random.Random(
        stable_seed(
            seed,
            namespace,
            observation.rng_key or observation.hand_id,
            observation.seat,
            observation.street,
            len(observation.history),
        )
    )


def _decision_with_metadata(
    decision: Decision,
    **metadata: MetadataValue,
) -> Decision:
    additions = tuple(sorted(metadata.items()))
    return Decision(decision.action, decision.probabilities, (*decision.metadata, *additions))


class MixedPolicy:
    """Sample from the weighted mixture of complete child action distributions."""

    def __init__(
        self,
        policies: Sequence[Policy],
        weights: Sequence[float],
        *,
        seed: int = 0,
        name: str = "mixed_v1",
    ) -> None:
        if not policies:
            raise ValueError("a mixture requires at least one child policy")
        if len(policies) != len(weights):
            raise ValueError("policies and weights must have equal length")
        if len({policy.name for policy in policies}) != len(policies):
            raise ValueError("mixture child policy names must be unique")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("mixture weights must be finite and non-negative")
        total = math.fsum(weights)
        if total <= 0:
            raise ValueError("at least one mixture weight must be positive")
        if not name:
            raise ValueError("policy name must be non-empty")
        self.name = name
        self.policies = tuple(policies)
        self.weights = tuple(weight / total for weight in weights)
        self.seed = seed

    def decide(self, observation: Observation) -> Decision:
        aggregate = {option.name: 0.0 for option in observation.legal_actions}
        child_decisions = tuple(policy.decide(observation) for policy in self.policies)
        for weight, child_decision in zip(self.weights, child_decisions, strict=True):
            child_probabilities = _probabilities(observation, child_decision)
            for action, probability in child_probabilities.items():
                aggregate[action] += weight * probability
        total = math.fsum(aggregate.values())
        probabilities = {action: probability / total for action, probability in aggregate.items()}
        action = _sample_action(
            observation,
            probabilities,
            _decision_rng(self.seed, "mixed_policy_action_v1", observation),
        )
        return Decision(
            action,
            tuple(
                (option.name, probabilities[option.name])
                for option in observation.legal_actions
            ),
            (
                ("composition", "mixture"),
                ("components", ",".join(policy.name for policy in self.policies)),
                ("weights", ",".join(f"{weight:.12g}" for weight in self.weights)),
            ),
        )

    def reset(self) -> None:
        _reset_children(self.policies)

    def clone(self, *, reset_state: bool = True) -> MixedPolicy:
        clone = copy.deepcopy(self)
        if reset_state:
            clone.reset()
        return clone


def mixture(
    *weighted_policies: tuple[float, Policy],
    seed: int = 0,
    name: str = "mixed_v1",
) -> MixedPolicy:
    """Convenience constructor accepting ``(weight, policy)`` pairs."""

    return MixedPolicy(
        tuple(policy for _, policy in weighted_policies),
        tuple(weight for weight, _ in weighted_policies),
        seed=seed,
        name=name,
    )


class EpsilonExplorationPolicy:
    """Mix a child's distribution with uniform legal-action exploration."""

    def __init__(
        self,
        policy: Policy,
        epsilon: float,
        *,
        seed: int = 0,
        name: str = "epsilon_exploration_v1",
    ) -> None:
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be a finite value in [0, 1]")
        if not name:
            raise ValueError("policy name must be non-empty")
        self.name = name
        self.policy = policy
        self.epsilon = epsilon
        self.seed = seed

    def decide(self, observation: Observation) -> Decision:
        child = self.policy.decide(observation)
        child_probabilities = _probabilities(observation, child)
        uniform = 1.0 / len(observation.legal_actions)
        probabilities = {
            option.name: (1 - self.epsilon) * child_probabilities[option.name]
            + self.epsilon * uniform
            for option in observation.legal_actions
        }
        action = _sample_action(
            observation,
            probabilities,
            _decision_rng(self.seed, "epsilon_exploration_action_v1", observation),
        )
        return Decision(
            action,
            tuple(
                (option.name, probabilities[option.name])
                for option in observation.legal_actions
            ),
            (
                ("base_action", child.action),
                ("base_policy", self.policy.name),
                ("composition", "epsilon_exploration"),
                ("epsilon", self.epsilon),
            ),
        )

    def reset(self) -> None:
        _reset_children((self.policy,))

    def clone(self, *, reset_state: bool = True) -> EpsilonExplorationPolicy:
        clone = copy.deepcopy(self)
        if reset_state:
            clone.reset()
        return clone


@dataclass(frozen=True, slots=True)
class OpponentStats:
    """Auditable sufficient statistics derived only from public actions."""

    hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    postflop_aggressive_actions: int = 0
    postflop_calls: int = 0
    folds_to_bet: int = 0
    fold_to_bet_opportunities: int = 0

    @property
    def vpip_rate(self) -> float | None:
        return self.vpip_hands / self.hands if self.hands else None

    @property
    def pfr_rate(self) -> float | None:
        return self.pfr_hands / self.hands if self.hands else None

    @property
    def aggression_factor(self) -> float | None:
        return (
            self.postflop_aggressive_actions / self.postflop_calls
            if self.postflop_calls
            else None
        )

    @property
    def aggression_frequency(self) -> float | None:
        opportunities = self.postflop_aggressive_actions + self.postflop_calls
        return self.postflop_aggressive_actions / opportunities if opportunities else None

    @property
    def fold_to_bet_rate(self) -> float | None:
        return (
            self.folds_to_bet / self.fold_to_bet_opportunities
            if self.fold_to_bet_opportunities
            else None
        )


@dataclass(frozen=True, slots=True)
class _TrackedHand:
    hero_seat: int
    history: tuple[PublicAction, ...]


@dataclass(frozen=True, slots=True)
class _HandStats:
    vpip: bool = False
    pfr: bool = False
    aggressive_actions: int = 0
    postflop_calls: int = 0
    folds_to_bet: int = 0
    fold_to_bet_opportunities: int = 0


class OpponentStatsTracker:
    """Idempotently track VPIP/PFR/aggression/fold-to-bet from public history.

    ``PublicAction`` has no street field.  Heads-up betting rounds can nevertheless be
    reconstructed: a call after a raise closes a round, as do two checks when no raise is
    pending.  The tracker rejects histories that cease to extend a previously observed
    snapshot, preventing silent double counting or hand-id reuse.  An action that terminates
    a hand after the policy's last decision cannot appear in another ``Observation``; pass
    ``HandResult.public_actions`` to ``update_public_history`` after the hand to include it.
    """

    def __init__(self) -> None:
        self._hands: dict[str, _TrackedHand] = {}

    def update(self, observation: Observation) -> OpponentStats:
        return self.update_public_history(
            observation.hand_id,
            observation.seat,
            observation.history,
        )

    def update_public_history(
        self,
        hand_id: str,
        hero_seat: int,
        history: Sequence[PublicAction],
    ) -> OpponentStats:
        """Ingest an extending public snapshot, including an optional terminal one."""

        if hero_seat not in (0, 1):
            raise ValueError("opponent tracker requires a heads-up seat 0 or 1")
        public_history = tuple(history)
        if any(action.seat not in (0, 1) for action in public_history):
            raise ValueError("public history contains a non-heads-up seat")
        previous = self._hands.get(hand_id)
        if previous is not None:
            if previous.hero_seat != hero_seat:
                raise ValueError("hero seat changed within one hand id")
            prefix_length = len(previous.history)
            if public_history[:prefix_length] != previous.history:
                raise ValueError("public history must extend its previous snapshot")
        self._hands[hand_id] = _TrackedHand(hero_seat, public_history)
        return self.snapshot()

    def snapshot(self) -> OpponentStats:
        vpip_hands = 0
        pfr_hands = 0
        aggressive_actions = 0
        postflop_calls = 0
        folds_to_bet = 0
        fold_opportunities = 0
        for hand in self._hands.values():
            summary = _summarize_hand(hand.hero_seat, hand.history)
            vpip_hands += int(summary.vpip)
            pfr_hands += int(summary.pfr)
            aggressive_actions += summary.aggressive_actions
            postflop_calls += summary.postflop_calls
            folds_to_bet += summary.folds_to_bet
            fold_opportunities += summary.fold_to_bet_opportunities
        return OpponentStats(
            hands=len(self._hands),
            vpip_hands=vpip_hands,
            pfr_hands=pfr_hands,
            postflop_aggressive_actions=aggressive_actions,
            postflop_calls=postflop_calls,
            folds_to_bet=folds_to_bet,
            fold_to_bet_opportunities=fold_opportunities,
        )

    def reset(self) -> None:
        self._hands.clear()

    def clone(self, *, reset_state: bool = True) -> OpponentStatsTracker:
        clone = copy.deepcopy(self)
        if reset_state:
            clone.reset()
        return clone


def _summarize_hand(hero_seat: int, history: tuple[PublicAction, ...]) -> _HandStats:
    opponent = 1 - hero_seat
    street_index = 0
    pending_raiser: int | None = None
    passive_actions = 0
    opponent_vpip = False
    opponent_pfr = False
    aggressive_actions = 0
    postflop_calls = 0
    folds_to_bet = 0
    fold_opportunities = 0
    opponent_button_action_seen = False

    for action in history:
        if street_index > 3:
            break
        is_raise = action.action.startswith("raise_")
        if action.action not in {"fold", "check_call"} and not is_raise:
            raise ValueError(f"unsupported public action: {action.action!r}")
        facing_bet = pending_raiser is not None and pending_raiser != action.seat

        if action.seat == opponent:
            if facing_bet:
                fold_opportunities += 1
                folds_to_bet += int(action.action == "fold")
            if street_index == 0:
                if is_raise:
                    opponent_vpip = True
                    opponent_pfr = True
                elif (
                    action.action == "check_call"
                    and (facing_bet or (opponent == 1 and not opponent_button_action_seen))
                ):
                    opponent_vpip = True
            elif is_raise:
                aggressive_actions += 1
            elif action.action == "check_call" and facing_bet:
                postflop_calls += 1

        if street_index == 0 and action.seat == opponent and opponent == 1:
            opponent_button_action_seen = True
        if action.action == "fold":
            break
        if is_raise:
            pending_raiser = action.seat
            passive_actions = 0
        elif facing_bet:
            street_index += 1
            pending_raiser = None
            passive_actions = 0
        else:
            passive_actions += 1
            if passive_actions == 2:
                street_index += 1
                passive_actions = 0

    return _HandStats(
        vpip=opponent_vpip,
        pfr=opponent_pfr,
        aggressive_actions=aggressive_actions,
        postflop_calls=postflop_calls,
        folds_to_bet=folds_to_bet,
        fold_to_bet_opportunities=fold_opportunities,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveThresholds:
    """Pre-registered profile thresholds for ``AdaptiveExploitPolicy``."""

    minimum_hands: int = 20
    minimum_fold_opportunities: int = 8
    minimum_aggression_opportunities: int = 8
    loose_vpip: float = 0.55
    passive_pfr: float = 0.20
    overfold_rate: float = 0.60
    aggressive_pfr: float = 0.35
    aggressive_frequency: float = 0.65

    def validate(self) -> None:
        if self.minimum_hands < 1:
            raise ValueError("minimum_hands must be positive")
        if self.minimum_fold_opportunities < 1 or self.minimum_aggression_opportunities < 1:
            raise ValueError("minimum action opportunities must be positive")
        rates = (
            self.loose_vpip,
            self.passive_pfr,
            self.overfold_rate,
            self.aggressive_pfr,
            self.aggressive_frequency,
        )
        if any(not math.isfinite(rate) or not 0 <= rate <= 1 for rate in rates):
            raise ValueError("adaptive rates must be finite values in [0, 1]")


class AdaptiveExploitPolicy:
    """Select a pre-registered child policy from public opponent statistics.

    Online observation is explicit via ``observe_online``.  With it disabled, the tracker is
    a frozen prior/snapshot and ``decide`` cannot learn.  The current arena creates policies
    per hand and has no match callback, so complete match-long adaptation requires a runner
    that deliberately reuses this object and calls ``update_public_history`` after each hand.
    """

    def __init__(
        self,
        blueprint: Policy,
        *,
        loose_passive_policy: Policy | None = None,
        overfolder_policy: Policy | None = None,
        aggressive_policy: Policy | None = None,
        tracker: OpponentStatsTracker | None = None,
        thresholds: AdaptiveThresholds | None = None,
        observe_online: bool = True,
        name: str = "adaptive_exploit_v1",
    ) -> None:
        self.name = name
        self.blueprint = blueprint
        self.loose_passive_policy = loose_passive_policy or blueprint
        self.overfolder_policy = overfolder_policy or blueprint
        self.aggressive_policy = aggressive_policy or blueprint
        self.tracker = tracker or OpponentStatsTracker()
        self.thresholds = thresholds or AdaptiveThresholds()
        self.thresholds.validate()
        self.observe_online = observe_online

    @property
    def children(self) -> tuple[Policy, ...]:
        return (
            self.blueprint,
            self.loose_passive_policy,
            self.overfolder_policy,
            self.aggressive_policy,
        )

    def decide(self, observation: Observation) -> Decision:
        stats = self.tracker.update(observation) if self.observe_online else self.tracker.snapshot()
        profile, policy = self._select_policy(stats)
        child = policy.decide(observation)
        _probabilities(observation, child)
        return _decision_with_metadata(
            child,
            adaptive_profile=profile,
            adaptive_policy=policy.name,
            aggression_frequency=stats.aggression_frequency,
            fold_to_bet=stats.fold_to_bet_rate,
            observed_hands=stats.hands,
            pfr=stats.pfr_rate,
            vpip=stats.vpip_rate,
        )

    def update_public_history(
        self,
        hand_id: str,
        hero_seat: int,
        history: Sequence[PublicAction],
    ) -> OpponentStats:
        """Explicit post-hand hook for terminal public actions omitted from observations."""

        return self.tracker.update_public_history(hand_id, hero_seat, history)

    def _select_policy(self, stats: OpponentStats) -> tuple[str, Policy]:
        thresholds = self.thresholds
        if stats.hands < thresholds.minimum_hands:
            return "insufficient_evidence", self.blueprint
        fold_rate = stats.fold_to_bet_rate
        if (
            stats.fold_to_bet_opportunities >= thresholds.minimum_fold_opportunities
            and fold_rate is not None
            and fold_rate >= thresholds.overfold_rate
        ):
            return "overfolder", self.overfolder_policy
        aggression_opportunities = stats.postflop_aggressive_actions + stats.postflop_calls
        aggression_frequency = stats.aggression_frequency
        pfr = stats.pfr_rate
        if (
            (pfr is not None and pfr >= thresholds.aggressive_pfr)
            or (
                aggression_opportunities >= thresholds.minimum_aggression_opportunities
                and aggression_frequency is not None
                and aggression_frequency >= thresholds.aggressive_frequency
            )
        ):
            return "aggressive", self.aggressive_policy
        vpip = stats.vpip_rate
        if (
            vpip is not None
            and pfr is not None
            and vpip >= thresholds.loose_vpip
            and pfr <= thresholds.passive_pfr
        ):
            return "loose_passive", self.loose_passive_policy
        return "unclassified", self.blueprint

    def reset(self) -> None:
        self.tracker.reset()
        _reset_children(self.children)

    def clone(self, *, reset_state: bool = True) -> AdaptiveExploitPolicy:
        clone = copy.deepcopy(self)
        if reset_state:
            clone.reset()
        return clone


@dataclass(frozen=True, slots=True)
class MetaPolicyArmStats:
    policy: str
    pulls: int
    mean_reward: float | None


class UCBMetaPolicy:
    """Choose one child per hand and learn only through explicit terminal rewards.

    ``play_hand`` currently does not call ``update_reward``.  An outer match runner must call
    it exactly once per completed hand; without that call the object deliberately performs no
    reward learning.
    """

    def __init__(
        self,
        policies: Sequence[Policy],
        *,
        exploration: float = math.sqrt(2.0),
        seed: int = 0,
        name: str = "ucb_meta_v1",
    ) -> None:
        if not policies:
            raise ValueError("UCB meta-policy requires at least one child")
        if len({policy.name for policy in policies}) != len(policies):
            raise ValueError("UCB child policy names must be unique")
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("UCB exploration must be finite and non-negative")
        self.name = name
        self.policies = tuple(policies)
        self.exploration = exploration
        self.seed = seed
        self._pulls = [0 for _ in policies]
        self._reward_sums = [0.0 for _ in policies]
        self._assignments: dict[str, int] = {}
        self._updated_hands: set[str] = set()

    def decide(self, observation: Observation) -> Decision:
        arm = self._assignments.get(observation.hand_id)
        if arm is None:
            arm = self._select_arm(observation)
            self._assignments[observation.hand_id] = arm
        child = self.policies[arm].decide(observation)
        _probabilities(observation, child)
        return _decision_with_metadata(
            child,
            meta_policy="ucb1",
            selected_arm=self.policies[arm].name,
            selected_arm_pulls=self._pulls[arm],
        )

    def _select_arm(self, observation: Observation) -> int:
        rng = _decision_rng(self.seed, "ucb_meta_arm_v1", observation)
        untried = [index for index, pulls in enumerate(self._pulls) if pulls == 0]
        if untried:
            return rng.choice(untried)
        total_pulls = sum(self._pulls)
        scores = [
            self._reward_sums[index] / pulls
            + self.exploration * math.sqrt(math.log(total_pulls) / pulls)
            for index, pulls in enumerate(self._pulls)
        ]
        best = max(scores)
        tied = [
            index for index, score in enumerate(scores) if math.isclose(score, best, abs_tol=1e-12)
        ]
        return rng.choice(tied)

    def update_reward(self, hand_id: str, reward: float) -> None:
        """Apply one terminal reward to the child assigned to ``hand_id``."""

        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if hand_id in self._updated_hands:
            raise ValueError(f"reward already recorded for hand: {hand_id}")
        try:
            arm = self._assignments[hand_id]
        except KeyError as error:
            raise KeyError(f"no UCB assignment for hand: {hand_id}") from error
        self._pulls[arm] += 1
        self._reward_sums[arm] += reward
        self._updated_hands.add(hand_id)

    def arm_statistics(self) -> tuple[MetaPolicyArmStats, ...]:
        return tuple(
            MetaPolicyArmStats(
                policy.name,
                pulls,
                self._reward_sums[index] / pulls if pulls else None,
            )
            for index, (policy, pulls) in enumerate(zip(self.policies, self._pulls, strict=True))
        )

    def reset(self) -> None:
        self._pulls = [0 for _ in self.policies]
        self._reward_sums = [0.0 for _ in self.policies]
        self._assignments.clear()
        self._updated_hands.clear()
        _reset_children(self.policies)

    def clone(self, *, reset_state: bool = True) -> UCBMetaPolicy:
        clone = copy.deepcopy(self)
        if reset_state:
            clone.reset()
        return clone
