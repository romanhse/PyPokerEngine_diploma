"""Fast, auditable preflop-chart policies with pluggable postflop play."""

from __future__ import annotations

import math
from dataclasses import dataclass

from poker_research.types import ActionOption, Decision, MetadataValue, Observation, Policy

RANKS = "23456789TJQKA"
CHEN_HIGH_CARD_POINTS: dict[str, float] = {
    "A": 10.0,
    "K": 8.0,
    "Q": 7.0,
    "J": 6.0,
    "T": 5.0,
    "9": 4.5,
    "8": 4.0,
    "7": 3.5,
    "6": 3.0,
    "5": 2.5,
    "4": 2.0,
    "3": 1.5,
    "2": 1.0,
}


def canonical_starting_hand(hole_cards: tuple[str, str]) -> str:
    """Map two concrete cards to one of the standard 169 hand classes."""

    if len(hole_cards) != 2 or hole_cards[0] == hole_cards[1]:
        raise ValueError("starting hand requires two distinct cards")
    for card in hole_cards:
        if len(card) != 2 or card[0] not in RANKS or card[1] not in "cdhs":
            raise ValueError(f"invalid card: {card!r}")
    first, second = sorted(
        hole_cards,
        key=lambda card: RANKS.index(card[0]),
        reverse=True,
    )
    if first[0] == second[0]:
        return first[0] * 2
    suited = first[1] == second[1]
    return f"{first[0]}{second[0]}{'s' if suited else 'o'}"


def chen_score(hole_cards: tuple[str, str]) -> int:
    """Return the classic explainable Chen-style preflop score.

    The score is a compact chart feature, not a claim about exact all-in equity.
    It rewards pairs, suitedness and connectedness and penalizes large gaps.
    """

    hand_class = canonical_starting_hand(hole_cards)
    high_rank, low_rank = hand_class[0], hand_class[1]
    score = CHEN_HIGH_CARD_POINTS[high_rank]
    is_pair = high_rank == low_rank
    if is_pair:
        score = max(5.0, score * 2.0)
    else:
        if hand_class.endswith("s"):
            score += 2.0
        gap = RANKS.index(high_rank) - RANKS.index(low_rank) - 1
        gap_penalty = 0.0 if gap <= 0 else (1.0 if gap == 1 else (2.0 if gap == 2 else 4.0))
        if gap >= 4:
            gap_penalty = 5.0
        score -= gap_penalty
        if gap <= 1 and RANKS.index(high_rank) < RANKS.index("Q"):
            score += 1.0
    return max(0, math.ceil(score))


@dataclass(frozen=True, slots=True)
class PreflopChartConfig:
    """Thresholds for unopened, raised and reraised preflop pots."""

    button_open: int = 7
    big_blind_continue: int = 6
    facing_raise_call: int = 8
    facing_raise_reraise: int = 11
    facing_reraise_call: int = 10
    facing_reraise_raise: int = 13
    open_sizing: tuple[str, ...] = ("raise_half_pot", "raise_min", "raise_pot")
    reraise_sizing: tuple[str, ...] = ("raise_pot", "raise_half_pot", "raise_min")

    def validate(self) -> None:
        thresholds = (
            self.button_open,
            self.big_blind_continue,
            self.facing_raise_call,
            self.facing_raise_reraise,
            self.facing_reraise_call,
            self.facing_reraise_raise,
        )
        if any(not 0 <= threshold <= 20 for threshold in thresholds):
            raise ValueError("chart thresholds must be in [0, 20]")
        if self.facing_raise_reraise < self.facing_raise_call:
            raise ValueError("reraise threshold must not be below call threshold")
        if self.facing_reraise_raise < self.facing_reraise_call:
            raise ValueError("four-bet threshold must not be below call threshold")
        for preferences in (self.open_sizing, self.reraise_sizing):
            if not preferences or any(not action.startswith("raise_") for action in preferences):
                raise ValueError("sizing preferences must contain raise actions")


TAG_CHART = PreflopChartConfig()
LAG_CHART = PreflopChartConfig(
    button_open=5,
    big_blind_continue=4,
    facing_raise_call=6,
    facing_raise_reraise=10,
    facing_reraise_call=8,
    facing_reraise_raise=12,
)
NIT_CHART = PreflopChartConfig(
    button_open=9,
    big_blind_continue=8,
    facing_raise_call=10,
    facing_raise_reraise=13,
    facing_reraise_call=12,
    facing_reraise_raise=15,
)


class PreflopChartPolicy:
    """Use a deterministic preflop range chart and delegate later streets."""

    def __init__(
        self,
        postflop_policy: Policy,
        *,
        config: PreflopChartConfig | None = None,
        name: str = "preflop_chart_v1",
    ) -> None:
        self.postflop_policy = postflop_policy
        self.config = config or TAG_CHART
        self.config.validate()
        if not name or any(character.isspace() for character in name):
            raise ValueError("policy name must be non-empty and contain no whitespace")
        self.name = name

    def decide(self, observation: Observation) -> Decision:
        if observation.street != "preflop":
            child = self.postflop_policy.decide(observation)
            return Decision(
                child.action,
                child.probabilities,
                (
                    *child.metadata,
                    ("chart_phase", "postflop_delegate"),
                    ("postflop_policy", self.postflop_policy.name),
                ),
            )

        score = chen_score(observation.hole_cards)
        hand_class = canonical_starting_hand(observation.hole_cards)
        raise_count = sum(action.action.startswith("raise_") for action in observation.history)
        if raise_count == 0:
            continue_threshold = (
                self.config.button_open
                if observation.position == "button_sb"
                else self.config.big_blind_continue
            )
            raise_threshold = continue_threshold
            sizing = self.config.open_sizing
            context = "unopened_or_limped"
        elif raise_count == 1:
            continue_threshold = self.config.facing_raise_call
            raise_threshold = self.config.facing_raise_reraise
            sizing = self.config.reraise_sizing
            context = "facing_raise"
        else:
            continue_threshold = self.config.facing_reraise_call
            raise_threshold = self.config.facing_reraise_raise
            sizing = self.config.reraise_sizing
            context = "facing_reraise"

        selected: ActionOption
        reason: str
        if score >= raise_threshold and (raise_option := _select_raise(observation, sizing)):
            selected = raise_option
            reason = "chart_raise"
        elif score >= continue_threshold:
            selected = observation.option("check_call") or _fallback(observation)
            reason = "chart_continue"
        elif observation.call_amount == 0:
            selected = observation.option("check_call") or _fallback(observation)
            reason = "free_check"
        else:
            selected = observation.option("fold") or _fallback(observation)
            reason = "below_chart"
        metadata: dict[str, MetadataValue] = {
            "chart_context": context,
            "chart_phase": "preflop",
            "chen_score": score,
            "continue_threshold": continue_threshold,
            "hand_class": hand_class,
            "raise_count": raise_count,
            "raise_threshold": raise_threshold,
            "reason": reason,
        }
        return _point_decision(observation, selected, metadata)


def _select_raise(
    observation: Observation,
    preferences: tuple[str, ...],
) -> ActionOption | None:
    for name in preferences:
        option = observation.option(name)
        if option is not None:
            return option
    return next(
        (option for option in observation.legal_actions if option.name.startswith("raise_")),
        None,
    )


def _fallback(observation: Observation) -> ActionOption:
    for name in ("check_call", "fold"):
        option = observation.option(name)
        if option is not None:
            return option
    if not observation.legal_actions:
        raise RuntimeError("chart policy received no legal action")
    return observation.legal_actions[0]


def _point_decision(
    observation: Observation,
    selected: ActionOption,
    metadata: dict[str, MetadataValue],
) -> Decision:
    probabilities = tuple(
        (option.name, float(option.name == selected.name))
        for option in observation.legal_actions
    )
    return Decision(selected.name, probabilities, tuple(sorted(metadata.items())))


__all__ = [
    "LAG_CHART",
    "NIT_CHART",
    "TAG_CHART",
    "PreflopChartConfig",
    "PreflopChartPolicy",
    "canonical_starting_hand",
    "chen_score",
]
