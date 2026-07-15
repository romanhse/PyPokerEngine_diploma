"""Frozen baseline policies for evaluation and sanity checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from poker_research.equity import equity_vs_uniform, stable_seed
from poker_research.types import ActionOption, Decision, Observation


def _deterministic_decision(
    observation: Observation,
    selected: ActionOption,
    **metadata: str | int | float | bool | None,
) -> Decision:
    probabilities = tuple(
        (option.name, 1.0 if option.name == selected.name else 0.0)
        for option in observation.legal_actions
    )
    return Decision(selected.name, probabilities, tuple(sorted(metadata.items())))


class CallingStationPolicy:
    """Canonical fish: never bets or folds and always checks/calls."""

    name = "calling_station_v1"

    def decide(self, observation: Observation) -> Decision:
        option = observation.option("check_call")
        if option is None:
            raise RuntimeError("calling station received no check/call action")
        return _deterministic_decision(observation, option, policy="always_check_call")


@dataclass(frozen=True, slots=True)
class EquityValueConfig:
    """Pre-registered knobs for the transparent value-betting baseline."""

    preflop_samples: int = 256
    postflop_samples: int = 384
    call_margin: float = 0.015
    preflop_raise_equity: float = 0.60
    flop_raise_equity: float = 0.61
    turn_raise_equity: float = 0.59
    river_raise_equity: float = 0.53
    strong_equity: float = 0.70
    nut_equity: float = 0.84

    def validate(self) -> None:
        """Reject non-reproducible sampling budgets and nonsensical thresholds."""

        for field_name in ("preflop_samples", "postflop_samples"):
            sample_count = getattr(self, field_name)
            if isinstance(sample_count, bool) or not isinstance(sample_count, int):
                raise ValueError(f"{field_name} must be an integer")
            if sample_count <= 0:
                raise ValueError(f"{field_name} must be positive")

        bounded_fields = (
            "call_margin",
            "preflop_raise_equity",
            "flop_raise_equity",
            "turn_raise_equity",
            "river_raise_equity",
            "strong_equity",
            "nut_equity",
        )
        for field_name in bounded_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.strong_equity > self.nut_equity:
            raise ValueError("strong_equity must not exceed nut_equity")


class EquityValuePolicy:
    """Range-aware, no-bluff baseline designed to exploit loose-passive play.

    It is intentionally transparent: estimate equity against the opponent's
    uncensored uniform range, respect pot odds, and value-bet with fixed,
    auditable size thresholds. It never learns during evaluation.
    """

    name = "equity_value_v1"

    def __init__(self, config: EquityValueConfig | None = None, *, seed: int = 0) -> None:
        self.config = config or EquityValueConfig()
        self.config.validate()
        self.seed = seed

    def decide(self, observation: Observation) -> Decision:
        sample_count = (
            self.config.preflop_samples
            if observation.street == "preflop"
            else self.config.postflop_samples
        )
        equity_seed = stable_seed(
            self.seed,
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

        fold = observation.option("fold")
        if (
            observation.call_amount > 0
            and fold is not None
            and equity + self.config.call_margin < pot_odds
        ):
            return _deterministic_decision(
                observation,
                fold,
                equity=round(equity, 6),
                pot_odds=round(pot_odds, 6),
                reason="below_pot_odds",
            )

        threshold = self._raise_threshold(observation.street)
        if equity >= threshold:
            raise_option = self._select_raise(observation, equity)
            if raise_option is not None:
                return _deterministic_decision(
                    observation,
                    raise_option,
                    equity=round(equity, 6),
                    pot_odds=round(pot_odds, 6),
                    reason="value_raise",
                )

        check_call = observation.option("check_call")
        if check_call is None:
            raise RuntimeError("equity policy received no check/call action")
        return _deterministic_decision(
            observation,
            check_call,
            equity=round(equity, 6),
            pot_odds=round(pot_odds, 6),
            reason="realize_equity",
        )

    def _raise_threshold(self, street: str) -> float:
        thresholds = {
            "preflop": self.config.preflop_raise_equity,
            "flop": self.config.flop_raise_equity,
            "turn": self.config.turn_raise_equity,
            "river": self.config.river_raise_equity,
        }
        try:
            return thresholds[street]
        except KeyError as error:
            raise ValueError(f"unknown street: {street}") from error

    def _select_raise(self, observation: Observation, equity: float) -> ActionOption | None:
        preference: tuple[str, ...]
        if equity >= self.config.nut_equity and observation.street != "preflop":
            preference = ("raise_2pot", "raise_pot", "raise_half_pot", "raise_min")
        elif equity >= self.config.strong_equity:
            preference = ("raise_pot", "raise_half_pot", "raise_min")
        else:
            preference = ("raise_half_pot", "raise_min")
        return next(
            (option for name in preference if (option := observation.option(name)) is not None),
            None,
        )
