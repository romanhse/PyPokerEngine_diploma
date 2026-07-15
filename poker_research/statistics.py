"""Pre-registered statistics for independent duplicate-poker pairs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    pair_count: int
    mean_bb100: float
    std_bb100: float
    standard_error: float
    two_sided_ci95: tuple[float, float]
    one_sided_lower95: float
    bootstrap_ci95: tuple[float, float]
    t_statistic_vs_zero: float
    one_sided_p_value_vs_zero: float
    practical_margin_bb100: float
    one_sided_lower95_minus_margin: float
    significant_win: bool
    clears_practical_margin: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["two_sided_ci95"] = list(self.two_sided_ci95)
        payload["bootstrap_ci95"] = list(self.bootstrap_ci95)
        return payload


def summarize_duplicate_pairs(
    values_bb100: Sequence[float],
    *,
    practical_margin_bb100: float = 0.0,
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 20260715,
) -> EvaluationSummary:
    """Summarize pair-level observations without pretending hands are independent."""

    values = np.asarray(values_bb100, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two one-dimensional duplicate-pair values are required")
    if not np.isfinite(values).all():
        raise ValueError("duplicate-pair values must be finite")
    if bootstrap_resamples < 1_000:
        raise ValueError("bootstrap_resamples must be at least 1000")

    pair_count = int(values.size)
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = standard_deviation / math.sqrt(pair_count)
    critical_two_sided = float(stats.t.ppf(0.975, pair_count - 1))
    critical_one_sided = float(stats.t.ppf(0.95, pair_count - 1))
    two_sided = (
        mean - critical_two_sided * standard_error,
        mean + critical_two_sided * standard_error,
    )
    lower = mean - critical_one_sided * standard_error

    if standard_error == 0:
        t_statistic = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        p_value = 0.0 if mean > 0 else (1.0 if mean < 0 else 0.5)
    else:
        t_statistic = mean / standard_error
        p_value = float(stats.t.sf(t_statistic, pair_count - 1))

    rng = np.random.default_rng(bootstrap_seed)
    chunk_size = min(1_000, bootstrap_resamples)
    bootstrap_means: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    completed = 0
    while completed < bootstrap_resamples:
        count = min(chunk_size, bootstrap_resamples - completed)
        indices = rng.integers(0, pair_count, size=(count, pair_count))
        bootstrap_means.append(values[indices].mean(axis=1))
        completed += count
    bootstrap_distribution = np.concatenate(bootstrap_means)
    bootstrap_ci = tuple(
        float(value) for value in np.quantile(bootstrap_distribution, (0.025, 0.975))
    )

    lower_minus_margin = lower - practical_margin_bb100
    return EvaluationSummary(
        pair_count=pair_count,
        mean_bb100=mean,
        std_bb100=standard_deviation,
        standard_error=standard_error,
        two_sided_ci95=(two_sided[0], two_sided[1]),
        one_sided_lower95=lower,
        bootstrap_ci95=(bootstrap_ci[0], bootstrap_ci[1]),
        t_statistic_vs_zero=t_statistic,
        one_sided_p_value_vs_zero=p_value,
        practical_margin_bb100=practical_margin_bb100,
        one_sided_lower95_minus_margin=lower_minus_margin,
        significant_win=lower > 0,
        clears_practical_margin=lower_minus_margin > 0,
    )


def required_pair_count(
    standard_deviation_bb100: float,
    detectable_effect_bb100: float,
    *,
    alpha: float = 0.05,
    power: float = 0.90,
) -> int:
    """Normal-approximation sample size for a one-sided fixed-horizon test."""

    if standard_deviation_bb100 <= 0 or detectable_effect_bb100 <= 0:
        raise ValueError("standard deviation and effect must be positive")
    if not 0 < alpha < 0.5 or not 0.5 < power < 1:
        raise ValueError("require 0 < alpha < .5 and .5 < power < 1")
    z_alpha = float(stats.norm.ppf(1 - alpha))
    z_power = float(stats.norm.ppf(power))
    return math.ceil(
        ((z_alpha + z_power) * standard_deviation_bb100 / detectable_effect_bb100) ** 2
    )
