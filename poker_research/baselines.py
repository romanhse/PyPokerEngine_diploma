"""Auditable heuristic baselines for heads-up no-limit hold'em experiments.

Every policy consumes only :class:`~poker_research.types.Observation`, samples
through a policy-local RNG, and returns a complete probability distribution over
the arena's current legal action abstraction.  The policies are benchmarks and
opponent archetypes, not claims of equilibrium play.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from poker_research.equity import equity_vs_uniform, stable_seed
from poker_research.types import (
    ActionOption,
    Decision,
    MetadataValue,
    Observation,
)


def _validate_name(name: str) -> None:
    if not name or any(character.isspace() for character in name):
        raise ValueError("policy name must be non-empty and contain no whitespace")


def _legal_options(observation: Observation) -> dict[str, ActionOption]:
    options = {option.name: option for option in observation.legal_actions}
    if not options:
        raise RuntimeError("policy received an observation with no legal actions")
    if len(options) != len(observation.legal_actions):
        raise RuntimeError("legal action names must be unique")
    return options


def _point_decision(
    observation: Observation,
    selected: ActionOption,
    **metadata: MetadataValue,
) -> Decision:
    """Build an auditable deterministic decision over the current legal set."""

    options = _legal_options(observation)
    if options.get(selected.name) != selected:
        raise ValueError(f"selected action is not legal: {selected}")
    probabilities = tuple(
        (option.name, float(option.name == selected.name))
        for option in observation.legal_actions
    )
    return Decision(selected.name, probabilities, tuple(sorted(metadata.items())))


def _normalized_distribution(
    observation: Observation,
    weights: dict[str, float],
) -> tuple[tuple[str, float], ...]:
    """Normalize non-negative weights while retaining every legal action."""

    options = _legal_options(observation)
    unknown = set(weights).difference(options)
    if unknown:
        raise ValueError(f"weights reference illegal actions: {sorted(unknown)}")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("action weights must be non-negative")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one legal action must have positive weight")
    return tuple(
        (option.name, weights.get(option.name, 0.0) / total)
        for option in options.values()
    )


def _sample_action(
    probabilities: tuple[tuple[str, float], ...],
    rng: random.Random,
) -> str:
    """Sample a categorical distribution without touching process-global RNG."""

    draw = rng.random()
    cumulative = 0.0
    last_positive: str | None = None
    for action, probability in probabilities:
        if probability > 0:
            last_positive = action
        cumulative += probability
        if draw < cumulative:
            return action
    if last_positive is None:
        raise RuntimeError("probability distribution has no positive action")
    return last_positive


def _decision_rng(seed: int, namespace: str, observation: Observation) -> random.Random:
    """Return a reproducible RNG unique to one public decision identity.

    League factories intentionally create a fresh policy for every hand.  A mutable RNG
    initialized only from ``seed`` would therefore replay its first draw in every hand.
    Keying the local stream by public observation identity keeps fresh and reused policy
    instances equivalent without exposing cards or touching the process-global RNG.
    """

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


def _fallback(observation: Observation, *, prefer_fold: bool = False) -> ActionOption:
    options = _legal_options(observation)
    preference = ("fold", "check_call") if prefer_fold else ("check_call", "fold")
    for name in preference:
        if name in options:
            return options[name]
    return next(iter(options.values()))


@dataclass(frozen=True, slots=True)
class BetSizingProfile:
    """Ordered preferences over the arena's legal raise-size abstraction."""

    name: str
    preferences: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("bet-sizing profile name must be non-empty")
        if not self.preferences:
            raise ValueError("bet-sizing profile must contain at least one preference")
        if len(set(self.preferences)) != len(self.preferences):
            raise ValueError("bet-sizing preferences must be unique")
        if any(not action.startswith("raise_") for action in self.preferences):
            raise ValueError("bet-sizing preferences must be raise actions")

    def select(self, observation: Observation) -> ActionOption | None:
        """Return the first preferred legal raise, then any remaining legal raise."""

        options = _legal_options(observation)
        for action in self.preferences:
            option = options.get(action)
            if option is not None and option.raise_to is not None:
                return option
        return next((option for option in options.values() if option.raise_to is not None), None)


MIN_RAISE_PROFILE = BetSizingProfile(
    "min_raise",
    ("raise_min", "raise_half_pot", "raise_pot", "raise_2pot", "raise_all_in"),
)
HALF_POT_PROFILE = BetSizingProfile(
    "half_pot",
    ("raise_half_pot", "raise_pot", "raise_min", "raise_2pot", "raise_all_in"),
)
POT_PROFILE = BetSizingProfile(
    "pot",
    ("raise_pot", "raise_half_pot", "raise_2pot", "raise_min", "raise_all_in"),
)
OVERBET_PROFILE = BetSizingProfile(
    "overbet",
    ("raise_2pot", "raise_pot", "raise_all_in", "raise_half_pot", "raise_min"),
)
ALL_IN_PROFILE = BetSizingProfile(
    "all_in",
    ("raise_all_in", "raise_2pot", "raise_pot", "raise_half_pot", "raise_min"),
)


@dataclass(frozen=True, slots=True)
class CheckFoldPolicy:
    """Check when checking is free; otherwise fold whenever folding is legal."""

    name: str = "check_fold_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)

    def decide(self, observation: Observation) -> Decision:
        options = _legal_options(observation)
        if observation.call_amount == 0 and "check_call" in options:
            selected = options["check_call"]
            reason = "free_check"
        elif "fold" in options:
            selected = options["fold"]
            reason = "facing_bet"
        else:
            selected = _fallback(observation)
            reason = "legal_fallback"
        return _point_decision(observation, selected, policy=self.name, reason=reason)


@dataclass(frozen=True, slots=True)
class CallCheckPolicy:
    """Always check or call; equivalent to a robust calling-station baseline."""

    name: str = "call_check_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)

    def decide(self, observation: Observation) -> Decision:
        selected = observation.option("check_call") or _fallback(observation)
        return _point_decision(observation, selected, policy=self.name, reason="check_call")


@dataclass(slots=True)
class RandomValidPolicy:
    """Sample uniformly from the current abstract legal actions.

    Because each raise size is a distinct abstract action, this intentionally
    weights games with more available raise sizes toward aggression.
    """

    seed: int = 0
    name: str = "random_valid_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)

    def decide(self, observation: Observation) -> Decision:
        options = _legal_options(observation)
        weight = 1.0 / len(options)
        probabilities = tuple((name, weight) for name in options)
        selected = _sample_action(
            probabilities,
            _decision_rng(self.seed, "random_valid_action_v1", observation),
        )
        return Decision(
            selected,
            probabilities,
            (("policy", self.name), ("seed", self.seed)),
        )


@dataclass(slots=True)
class BetSizingPolicy:
    """Raise with a configurable frequency and ordered sizing profile."""

    profile: BetSizingProfile = POT_PROFILE
    raise_probability: float = 1.0
    seed: int = 0
    name: str = "bet_sizing_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not 0.0 <= self.raise_probability <= 1.0:
            raise ValueError("raise_probability must be in [0, 1]")

    def decide(self, observation: Observation) -> Decision:
        raise_option = self.profile.select(observation)
        fallback = _fallback(observation)
        if raise_option is None or raise_option.name == fallback.name:
            return _point_decision(
                observation,
                fallback,
                policy=self.name,
                profile=self.profile.name,
                reason="no_raise_available",
            )
        weights = {
            raise_option.name: self.raise_probability,
            fallback.name: 1.0 - self.raise_probability,
        }
        probabilities = _normalized_distribution(observation, weights)
        selected = _sample_action(
            probabilities,
            _decision_rng(self.seed, "bet_sizing_action_v1", observation),
        )
        reason = "profile_raise" if selected == raise_option.name else "declined_raise"
        metadata: dict[str, MetadataValue] = {
            "policy": self.name,
            "profile": self.profile.name,
            "reason": reason,
            "seed": self.seed,
        }
        return Decision(
            selected,
            probabilities,
            tuple(sorted(metadata.items())),
        )


@dataclass(frozen=True, slots=True)
class StreetPressurePolicy:
    """Apply one sizing profile only on explicitly selected streets."""

    target_streets: tuple[str, ...] = ("flop",)
    profile: BetSizingProfile = HALF_POT_PROFILE
    name: str = "street_pressure_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)
        valid_streets = {"preflop", "flop", "turn", "river"}
        if not self.target_streets or set(self.target_streets).difference(valid_streets):
            raise ValueError("target_streets must be non-empty valid poker streets")
        if len(set(self.target_streets)) != len(self.target_streets):
            raise ValueError("target_streets must be unique")

    def decide(self, observation: Observation) -> Decision:
        raise_option = (
            self.profile.select(observation)
            if observation.street in self.target_streets
            else None
        )
        selected = raise_option or _fallback(observation)
        return _point_decision(
            observation,
            selected,
            policy=self.name,
            profile=self.profile.name,
            reason="target_street_raise" if raise_option else "off_street_check_call",
            target_streets=",".join(self.target_streets),
        )


@dataclass(frozen=True, slots=True)
class ButtonPressurePolicy:
    """Pressure from the button and realize equity passively out of position."""

    profile: BetSizingProfile = POT_PROFILE
    name: str = "button_pressure_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)

    def decide(self, observation: Observation) -> Decision:
        raise_option = (
            self.profile.select(observation)
            if observation.position == "button_sb"
            else None
        )
        selected = raise_option or _fallback(observation)
        return _point_decision(
            observation,
            selected,
            policy=self.name,
            profile=self.profile.name,
            reason="button_pressure" if raise_option else "out_of_position_check_call",
        )


@dataclass(frozen=True, slots=True)
class SPRJamPolicy:
    """Jam any holding below a public SPR threshold; otherwise check/call."""

    maximum_spr: float = 2.0
    name: str = "spr_jammer_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)
        if not math.isfinite(self.maximum_spr) or self.maximum_spr < 0:
            raise ValueError("maximum_spr must be finite and non-negative")

    def decide(self, observation: Observation) -> Decision:
        all_in = observation.option("raise_all_in")
        should_jam = observation.spr <= self.maximum_spr and all_in is not None
        selected = all_in if should_jam and all_in is not None else _fallback(observation)
        return _point_decision(
            observation,
            selected,
            maximum_spr=self.maximum_spr,
            policy=self.name,
            reason="low_spr_jam" if should_jam else "spr_above_threshold",
        )


@dataclass(frozen=True, slots=True)
class GeometricSizingPolicy:
    """Use a deterministic street-by-street escalation of raise sizes."""

    name: str = "geometric_sizer_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)

    def decide(self, observation: Observation) -> Decision:
        profiles = {
            "preflop": MIN_RAISE_PROFILE,
            "flop": HALF_POT_PROFILE,
            "turn": POT_PROFILE,
            "river": OVERBET_PROFILE,
        }
        try:
            profile = profiles[observation.street]
        except KeyError as error:
            raise ValueError(f"unknown street: {observation.street}") from error
        raise_option = profile.select(observation)
        selected = raise_option or _fallback(observation)
        return _point_decision(
            observation,
            selected,
            policy=self.name,
            profile=profile.name,
            reason="street_sizing_raise" if raise_option else "no_raise_available",
        )


@dataclass(frozen=True, slots=True)
class ManiacConfig:
    """Action frequencies for an extremely aggressive opponent archetype."""

    raise_probability: float = 0.82
    fold_probability: float = 0.02
    sizing_profile: BetSizingProfile = OVERBET_PROFILE

    def __post_init__(self) -> None:
        if not 0.0 <= self.raise_probability <= 1.0:
            raise ValueError("raise_probability must be in [0, 1]")
        if not 0.0 <= self.fold_probability <= 1.0:
            raise ValueError("fold_probability must be in [0, 1]")
        if self.raise_probability + self.fold_probability > 1.0:
            raise ValueError("raise_probability + fold_probability must not exceed one")


@dataclass(slots=True)
class ManiacPolicy:
    """Raise most opportunities, rarely fold, and otherwise check/call."""

    config: ManiacConfig = ManiacConfig()
    seed: int = 0
    name: str = "maniac_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)

    def decide(self, observation: Observation) -> Decision:
        options = _legal_options(observation)
        raise_option = self.config.sizing_profile.select(observation)
        fold = options.get("fold") if observation.call_amount > 0 else None
        check_call = options.get("check_call")
        weights: dict[str, float] = {}
        raise_probability = self.config.raise_probability if raise_option is not None else 0.0
        fold_probability = self.config.fold_probability if fold is not None else 0.0
        if raise_option is not None:
            weights[raise_option.name] = raise_probability
        if fold is not None:
            weights[fold.name] = fold_probability
        remainder = 1.0 - raise_probability - fold_probability
        receiver = check_call or fold or raise_option or next(iter(options.values()))
        weights[receiver.name] = weights.get(receiver.name, 0.0) + remainder
        probabilities = _normalized_distribution(observation, weights)
        selected = _sample_action(
            probabilities,
            _decision_rng(self.seed, "maniac_action_v1", observation),
        )
        metadata: dict[str, MetadataValue] = {
            "policy": self.name,
            "profile": self.config.sizing_profile.name,
            "seed": self.seed,
        }
        return Decision(
            selected,
            probabilities,
            tuple(sorted(metadata.items())),
        )


@dataclass(frozen=True, slots=True)
class EquityThresholdConfig:
    """Transparent equity, pot-odds, aggression, and sizing parameters."""

    preflop_samples: int = 128
    postflop_samples: int = 192
    preflop_call_equity: float = 0.0
    flop_call_equity: float = 0.0
    turn_call_equity: float = 0.0
    river_call_equity: float = 0.0
    preflop_raise_equity: float = 0.60
    flop_raise_equity: float = 0.62
    turn_raise_equity: float = 0.64
    river_raise_equity: float = 0.66
    call_margin: float = 0.0
    value_raise_probability: float = 1.0
    bluff_probability: float = 0.0
    button_raise_discount: float = 0.0
    jam_spr: float = 0.75
    jam_equity: float = 0.84
    sizing_profile: BetSizingProfile = HALF_POT_PROFILE

    def __post_init__(self) -> None:
        if self.preflop_samples <= 0 or self.postflop_samples <= 0:
            raise ValueError("equity sample counts must be positive")
        unit_values = (
            self.preflop_call_equity,
            self.flop_call_equity,
            self.turn_call_equity,
            self.river_call_equity,
            self.preflop_raise_equity,
            self.flop_raise_equity,
            self.turn_raise_equity,
            self.river_raise_equity,
            self.value_raise_probability,
            self.bluff_probability,
            self.button_raise_discount,
            self.jam_equity,
        )
        if any(not 0.0 <= value <= 1.0 for value in unit_values):
            raise ValueError("equity thresholds and probabilities must be in [0, 1]")
        if not -1.0 <= self.call_margin <= 1.0:
            raise ValueError("call_margin must be in [-1, 1]")
        if not math.isfinite(self.jam_spr) or self.jam_spr < 0:
            raise ValueError("jam_spr must be finite and non-negative")


POT_ODDS_EQUITY_CONFIG = EquityThresholdConfig()
LOOSE_PASSIVE_CONFIG = EquityThresholdConfig(
    preflop_call_equity=0.30,
    flop_call_equity=0.24,
    turn_call_equity=0.25,
    river_call_equity=0.28,
    preflop_raise_equity=0.76,
    flop_raise_equity=0.78,
    turn_raise_equity=0.80,
    river_raise_equity=0.82,
    call_margin=0.10,
    value_raise_probability=0.18,
    bluff_probability=0.005,
    sizing_profile=MIN_RAISE_PROFILE,
)
TIGHT_PASSIVE_CONFIG = EquityThresholdConfig(
    preflop_call_equity=0.58,
    flop_call_equity=0.54,
    turn_call_equity=0.57,
    river_call_equity=0.60,
    preflop_raise_equity=0.72,
    flop_raise_equity=0.75,
    turn_raise_equity=0.78,
    river_raise_equity=0.80,
    call_margin=-0.02,
    value_raise_probability=0.16,
    bluff_probability=0.0,
    sizing_profile=MIN_RAISE_PROFILE,
)
TAG_CONFIG = EquityThresholdConfig(
    preflop_call_equity=0.49,
    flop_call_equity=0.42,
    turn_call_equity=0.43,
    river_call_equity=0.45,
    preflop_raise_equity=0.59,
    flop_raise_equity=0.62,
    turn_raise_equity=0.64,
    river_raise_equity=0.67,
    value_raise_probability=0.85,
    bluff_probability=0.06,
    button_raise_discount=0.025,
    sizing_profile=POT_PROFILE,
)
LAG_CONFIG = EquityThresholdConfig(
    preflop_call_equity=0.38,
    flop_call_equity=0.31,
    turn_call_equity=0.32,
    river_call_equity=0.34,
    preflop_raise_equity=0.53,
    flop_raise_equity=0.55,
    turn_raise_equity=0.57,
    river_raise_equity=0.59,
    call_margin=0.04,
    value_raise_probability=0.92,
    bluff_probability=0.18,
    button_raise_discount=0.05,
    sizing_profile=OVERBET_PROFILE,
)


@dataclass(slots=True)
class EquityThresholdPolicy:
    """Compare private-card equity with public pot odds and fixed thresholds."""

    config: EquityThresholdConfig = POT_ODDS_EQUITY_CONFIG
    seed: int = 0
    name: str = "equity_threshold_v1"

    def __post_init__(self) -> None:
        _validate_name(self.name)

    def decide(self, observation: Observation) -> Decision:
        sample_count = (
            self.config.preflop_samples
            if observation.street == "preflop"
            else self.config.postflop_samples
        )
        equity_seed = stable_seed(
            self.seed,
            "heuristic_equity",
            observation.hole_cards,
            observation.board_cards,
            observation.street,
        )
        equity = equity_vs_uniform(
            observation.hole_cards,
            observation.board_cards,
            sample_count=sample_count,
            seed=equity_seed,
        )
        pot_after_call = observation.pot + observation.call_amount
        pot_odds = observation.call_amount / pot_after_call if pot_after_call else 0.0
        call_threshold = max(pot_odds, self._street_value("call", observation.street))
        raise_threshold = self._street_value("raise", observation.street)
        if observation.position == "button_sb":
            raise_threshold = max(0.0, raise_threshold - self.config.button_raise_discount)

        fold = observation.option("fold")
        should_fold = (
            observation.call_amount > 0
            and fold is not None
            and equity + self.config.call_margin < call_threshold
        )
        fallback = fold if should_fold and fold is not None else _fallback(observation)
        raise_option = self._select_raise(observation, equity)
        if raise_option is None:
            raise_probability = 0.0
        elif equity >= raise_threshold:
            raise_probability = self.config.value_raise_probability
        else:
            raise_probability = self.config.bluff_probability

        weights = {fallback.name: 1.0 - raise_probability}
        if raise_option is not None and raise_probability > 0:
            weights[raise_option.name] = weights.get(raise_option.name, 0.0) + raise_probability
        probabilities = _normalized_distribution(observation, weights)
        selected = _sample_action(
            probabilities,
            _decision_rng(self.seed, "equity_threshold_action_v1", observation),
        )
        if raise_option is not None and selected == raise_option.name:
            reason = "value_raise" if equity >= raise_threshold else "bluff_raise"
        elif should_fold:
            reason = "below_call_threshold"
        else:
            reason = "realize_equity"
        metadata: dict[str, MetadataValue] = {
            "call_threshold": round(call_threshold, 6),
            "equity": round(equity, 6),
            "policy": self.name,
            "pot_odds": round(pot_odds, 6),
            "reason": reason,
            "seed": self.seed,
        }
        return Decision(
            selected,
            probabilities,
            tuple(sorted(metadata.items())),
        )

    def _street_value(self, kind: str, street: str) -> float:
        values = {
            ("call", "preflop"): self.config.preflop_call_equity,
            ("call", "flop"): self.config.flop_call_equity,
            ("call", "turn"): self.config.turn_call_equity,
            ("call", "river"): self.config.river_call_equity,
            ("raise", "preflop"): self.config.preflop_raise_equity,
            ("raise", "flop"): self.config.flop_raise_equity,
            ("raise", "turn"): self.config.turn_raise_equity,
            ("raise", "river"): self.config.river_raise_equity,
        }
        try:
            return values[(kind, street)]
        except KeyError as error:
            raise ValueError(f"unknown equity threshold: {kind}/{street}") from error

    def _select_raise(self, observation: Observation, equity: float) -> ActionOption | None:
        if observation.spr <= self.config.jam_spr and equity >= self.config.jam_equity:
            all_in = observation.option("raise_all_in")
            if all_in is not None:
                return all_in
        return self.config.sizing_profile.select(observation)


@dataclass(slots=True)
class PotOddsEquityPolicy(EquityThresholdPolicy):
    """Pure pot-odds caller with configurable equity value-betting thresholds."""

    config: EquityThresholdConfig = POT_ODDS_EQUITY_CONFIG
    name: str = "pot_odds_equity_v1"


@dataclass(slots=True)
class LoosePassivePolicy(EquityThresholdPolicy):
    """Call too wide, fold reluctantly, and raise only a small strong range."""

    config: EquityThresholdConfig = LOOSE_PASSIVE_CONFIG
    name: str = "loose_passive_v1"


@dataclass(slots=True)
class TightPassivePolicy(EquityThresholdPolicy):
    """Nit archetype: enter few pots and apply little pressure after entering."""

    config: EquityThresholdConfig = TIGHT_PASSIVE_CONFIG
    name: str = "tight_passive_v1"


@dataclass(slots=True)
class TightAggressivePolicy(EquityThresholdPolicy):
    """TAG archetype: continue a tight range and bet strong equity assertively."""

    config: EquityThresholdConfig = TAG_CONFIG
    name: str = "tag_v1"


@dataclass(slots=True)
class LooseAggressivePolicy(EquityThresholdPolicy):
    """LAG archetype: continue widely and combine thin value with frequent bluffs."""

    config: EquityThresholdConfig = LAG_CONFIG
    name: str = "lag_v1"


NitPolicy = TightPassivePolicy
TAGPolicy = TightAggressivePolicy
LAGPolicy = LooseAggressivePolicy


__all__ = [
    "ALL_IN_PROFILE",
    "HALF_POT_PROFILE",
    "LAGPolicy",
    "LAG_CONFIG",
    "LOOSE_PASSIVE_CONFIG",
    "MIN_RAISE_PROFILE",
    "NitPolicy",
    "OVERBET_PROFILE",
    "POT_ODDS_EQUITY_CONFIG",
    "POT_PROFILE",
    "TAGPolicy",
    "TAG_CONFIG",
    "TIGHT_PASSIVE_CONFIG",
    "BetSizingPolicy",
    "BetSizingProfile",
    "CallCheckPolicy",
    "ButtonPressurePolicy",
    "CheckFoldPolicy",
    "EquityThresholdConfig",
    "EquityThresholdPolicy",
    "LooseAggressivePolicy",
    "LoosePassivePolicy",
    "GeometricSizingPolicy",
    "ManiacConfig",
    "ManiacPolicy",
    "PotOddsEquityPolicy",
    "RandomValidPolicy",
    "SPRJamPolicy",
    "StreetPressurePolicy",
    "TightAggressivePolicy",
    "TightPassivePolicy",
]
