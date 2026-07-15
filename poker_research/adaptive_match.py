"""Sequential duplicate matches for explicitly stateful poker policies.

Unlike :mod:`poker_research.league`, this exploratory runner deliberately reuses the same
policy objects across pairs.  Terminal public-history and reward hooks are delivered only
after both paired-deck legs finish, so the first leg's outcome cannot influence the second
through those callbacks.  Physical play order alternates AB/BA by pair to balance remaining
decision-time state effects.  Decisions may still update a policy's documented online state;
callers must preregister that behavior and must not treat these observations as standard
duplicate-poker i.i.d. samples from a frozen strategy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from poker_research.arena import ArenaConfig, HandResult, play_hand
from poker_research.equity import stable_seed
from poker_research.types import Policy, PublicAction


@runtime_checkable
class PublicHistoryUpdater(Protocol):
    """Optional policy hook for a complete terminal public history."""

    def update_public_history(
        self,
        hand_id: str,
        hero_seat: int,
        history: Sequence[PublicAction],
    ) -> object:
        """Ingest one completed hand without hidden cards."""


@runtime_checkable
class RewardUpdater(Protocol):
    """Optional policy hook for a terminal chip reward."""

    def update_reward(self, hand_id: str, reward: float) -> None:
        """Ingest one reward after the policy has acted in the hand."""


@dataclass(frozen=True, slots=True)
class AdaptiveMatchConfig:
    """Frozen rules for one sequential adaptive duplicate match."""

    pair_count: int = 20
    master_seed: int = 0
    policy_schedule_seed: int = 1
    match_id: str = "adaptive-match-v1"
    game: ArenaConfig = ArenaConfig()

    def validate(self) -> None:
        if self.pair_count <= 0:
            raise ValueError("pair_count must be positive")
        if self.policy_schedule_seed < 0:
            raise ValueError("policy_schedule_seed must be non-negative")
        if not self.match_id or any(character.isspace() for character in self.match_id):
            raise ValueError("match_id must be non-empty and contain no whitespace")
        self.game.validate()


@dataclass(frozen=True, slots=True)
class AdaptivePairResult:
    """One auditable pair-level payoff for policy A."""

    pair_id: int
    deal_seed: int
    deck_hash: str
    hand_id_leg_a: str
    hand_id_leg_b: str
    play_order: tuple[str, str]
    policy_a_payoff_leg_a: int
    policy_a_payoff_leg_b: int
    policy_a_net_chips: int
    policy_a_bb100: float


@dataclass(frozen=True, slots=True)
class AdaptiveMatchResult:
    """In-memory hands and pair observations from a sequential adaptive match."""

    policy_names: tuple[str, str]
    pairs: tuple[AdaptivePairResult, ...]
    hands: tuple[HandResult, ...]


def run_adaptive_match(
    policy_a: Policy,
    policy_b: Policy,
    *,
    config: AdaptiveMatchConfig | None = None,
) -> AdaptiveMatchResult:
    """Play sequential duplicate pairs while retaining both policies' runtime state.

    The exact objects supplied as ``policy_a`` and ``policy_b`` are used for every hand.
    Public-history hooks receive all completed hands.  Reward hooks are called only when the
    policy made a decision in that hand, matching ``UCBMetaPolicy``'s per-hand assignment
    contract.
    """

    rules = config or AdaptiveMatchConfig()
    rules.validate()
    if policy_a is policy_b:
        raise ValueError("adaptive match requires two distinct policy objects")
    if policy_a.name == policy_b.name:
        raise ValueError("adaptive match policy names must be distinct")

    pairs: list[AdaptivePairResult] = []
    hands: list[HandResult] = []
    for pair_id in range(rules.pair_count):
        deal_seed = stable_seed(
            rules.master_seed,
            "adaptive_duplicate_pair_v1",
            policy_a.name,
            policy_b.name,
            pair_id,
        )
        hand_id_a = f"{rules.match_id}-{pair_id:06d}-a"
        hand_id_b = f"{rules.match_id}-{pair_id:06d}-b"
        play_order = ("a", "b") if pair_id % 2 == 0 else ("b", "a")
        first_leg = _play_adaptive_leg(
            policy_a,
            policy_b,
            rules,
            pair_id=pair_id,
            deal_seed=deal_seed,
            hand_id_a=hand_id_a,
            hand_id_b=hand_id_b,
            label=play_order[0],
        )
        second_leg = _play_adaptive_leg(
            policy_a,
            policy_b,
            rules,
            pair_id=pair_id,
            deal_seed=deal_seed,
            hand_id_a=hand_id_a,
            hand_id_b=hand_id_b,
            label=play_order[1],
        )
        if play_order == ("a", "b"):
            leg_a, leg_b = first_leg, second_leg
        else:
            leg_b, leg_a = first_leg, second_leg
        if leg_a.deck_hash != leg_b.deck_hash:
            raise RuntimeError("duplicate legs received different decks")

        policy_a_payoff_a = leg_a.payoffs[0]
        policy_a_payoff_b = leg_b.payoffs[1]
        net_chips = policy_a_payoff_a + policy_a_payoff_b
        bb100 = 50.0 * net_chips / rules.game.big_blind
        if not math.isfinite(bb100):
            raise RuntimeError("adaptive pair produced a non-finite payoff")

        # Batch terminal learning after the complete pair.  This deliberately keeps the
        # first leg's terminal reward out of the second leg played with the same deck.
        _notify_completed_hand(policy_a, leg_a, seat=0)
        _notify_completed_hand(policy_b, leg_a, seat=1)
        _notify_completed_hand(policy_b, leg_b, seat=0)
        _notify_completed_hand(policy_a, leg_b, seat=1)

        pairs.append(
            AdaptivePairResult(
                pair_id=pair_id,
                deal_seed=deal_seed,
                deck_hash=leg_a.deck_hash,
                hand_id_leg_a=hand_id_a,
                hand_id_leg_b=hand_id_b,
                play_order=play_order,
                policy_a_payoff_leg_a=policy_a_payoff_a,
                policy_a_payoff_leg_b=policy_a_payoff_b,
                policy_a_net_chips=net_chips,
                policy_a_bb100=bb100,
            )
        )
        hands.extend((leg_a, leg_b))

    return AdaptiveMatchResult(
        policy_names=(policy_a.name, policy_b.name),
        pairs=tuple(pairs),
        hands=tuple(hands),
    )


def _play_adaptive_leg(
    policy_a: Policy,
    policy_b: Policy,
    config: AdaptiveMatchConfig,
    *,
    pair_id: int,
    deal_seed: int,
    hand_id_a: str,
    hand_id_b: str,
    label: str,
) -> HandResult:
    policies = (policy_a, policy_b) if label == "a" else (policy_b, policy_a)
    return play_hand(
        policies,
        deal_seed=deal_seed,
        hand_id=hand_id_a if label == "a" else hand_id_b,
        config=config.game,
        policy_rng_key=str(
            stable_seed(
                config.policy_schedule_seed,
                "adaptive_policy_schedule_v1",
                pair_id,
                label,
            )
        ),
    )


def _notify_completed_hand(policy: Policy, hand: HandResult, *, seat: int) -> None:
    if isinstance(policy, PublicHistoryUpdater):
        policy.update_public_history(hand.hand_id, seat, hand.public_actions)
    if isinstance(policy, RewardUpdater) and any(
        decision.seat == seat for decision in hand.decisions
    ):
        policy.update_reward(hand.hand_id, float(hand.payoffs[seat]))
