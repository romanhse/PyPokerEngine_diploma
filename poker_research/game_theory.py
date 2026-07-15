"""Exact small-game CFR verification through OpenSpiel.

Full HUNL exploitability is intractable.  Kuhn/Leduc give a falsifiable place to
verify algorithms and metrics before using approximate best responses in HUNL.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    iteration: int
    exploitability: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_cfr_verification(
    *,
    game_name: str = "kuhn_poker",
    iterations: int = 10_000,
    report_every: int = 100,
    algorithm: str = "cfr_plus",
) -> tuple[ConvergencePoint, ...]:
    """Run CFR on a supported two-player zero-sum game and report exact exploitability."""

    if game_name not in {"kuhn_poker", "leduc_poker"}:
        raise ValueError("verification game must be kuhn_poker or leduc_poker")
    if iterations <= 0 or report_every <= 0:
        raise ValueError("iterations and report_every must be positive")
    try:
        import pyspiel
        from open_spiel.python.algorithms import cfr, exploitability
    except ImportError as error:
        raise RuntimeError(
            "OpenSpiel is optional; install it with `uv sync --extra game-theory`"
        ) from error

    game = pyspiel.load_game(game_name)
    solvers = {
        "cfr": cfr.CFRSolver,
        "cfr_plus": cfr.CFRPlusSolver,
    }
    try:
        solver = solvers[algorithm](game)
    except KeyError as error:
        raise ValueError("algorithm must be cfr or cfr_plus") from error

    started = time.perf_counter()
    points: list[ConvergencePoint] = []
    for iteration in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if iteration % report_every == 0 or iteration == iterations:
            value = float(exploitability.exploitability(game, solver.average_policy()))
            points.append(ConvergencePoint(iteration, value, time.perf_counter() - started))
    return tuple(points)
