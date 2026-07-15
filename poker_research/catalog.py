"""Versioned catalog and reproducible factories for ready-to-run policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from poker_research.adaptive import EpsilonExplorationPolicy, MixedPolicy
from poker_research.baselines import (
    ALL_IN_PROFILE,
    HALF_POT_PROFILE,
    MIN_RAISE_PROFILE,
    OVERBET_PROFILE,
    POT_PROFILE,
    BetSizingPolicy,
    ButtonPressurePolicy,
    CheckFoldPolicy,
    GeometricSizingPolicy,
    LooseAggressivePolicy,
    LoosePassivePolicy,
    ManiacPolicy,
    PotOddsEquityPolicy,
    RandomValidPolicy,
    SPRJamPolicy,
    StreetPressurePolicy,
    TightAggressivePolicy,
    TightPassivePolicy,
)
from poker_research.equity import stable_seed
from poker_research.neural import NumpyMLPPolicy
from poker_research.policies import CallingStationPolicy, EquityValuePolicy
from poker_research.preflop import LAG_CHART, NIT_CHART, TAG_CHART, PreflopChartPolicy
from poker_research.types import Policy

PolicyFactory = Callable[[], Policy]


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    """Human- and machine-readable description of one frozen policy version."""

    name: str
    family: str
    style: str
    stochastic: bool
    adaptive: bool
    trainable: bool
    speed: str
    description: str
    limitation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


POLICY_CATALOG: tuple[PolicyDescriptor, ...] = (
    PolicyDescriptor(
        "check_fold_v1",
        "negative_control",
        "ultra-tight passive",
        False,
        False,
        False,
        "fast",
        "Checks for free and folds to every bet.",
        "Deliberately terrible; useful only as a lower-bound control.",
    ),
    PolicyDescriptor(
        "calling_station_v1",
        "fish",
        "loose passive",
        False,
        False,
        False,
        "fast",
        "Never folds or bets; always checks/calls.",
        "Represents one extreme fish archetype, not weak players in general.",
    ),
    PolicyDescriptor(
        "random_valid_v1",
        "negative_control",
        "uniform abstract action",
        True,
        False,
        False,
        "fast",
        "Samples every currently legal abstract action with equal probability.",
        "Several raise sizes make aggregate aggression intentionally high.",
    ),
    PolicyDescriptor(
        "min_raiser_v1",
        "sizing_probe",
        "minimum bet pressure",
        False,
        False,
        False,
        "fast",
        "Raises the minimum whenever possible, otherwise checks/calls.",
        "Ignores cards and is intended to isolate sizing sensitivity.",
    ),
    PolicyDescriptor(
        "half_pot_bettor_v1",
        "sizing_probe",
        "half-pot pressure",
        False,
        False,
        False,
        "fast",
        "Uses half-pot raises whenever available.",
        "Ignores cards and over-bets weak holdings.",
    ),
    PolicyDescriptor(
        "pot_bettor_v1",
        "sizing_probe",
        "pot pressure",
        False,
        False,
        False,
        "fast",
        "Uses pot-sized raises whenever available.",
        "Ignores cards and is a stress test rather than a sound strategy.",
    ),
    PolicyDescriptor(
        "overbettor_v1",
        "sizing_probe",
        "two-pot pressure",
        False,
        False,
        False,
        "fast",
        "Prefers a two-pot overbet whenever legal.",
        "Card-blind pressure probe with very high variance.",
    ),
    PolicyDescriptor(
        "jammer_v1",
        "sizing_probe",
        "all-in pressure",
        False,
        False,
        False,
        "fast",
        "Moves all-in whenever raising is legal.",
        "A pathological opponent used to test risk and calling thresholds.",
    ),
    PolicyDescriptor(
        "button_bully_v1",
        "position_probe",
        "in-position pressure",
        False,
        False,
        False,
        "fast",
        "Raises pot from the button and checks/calls out of position.",
        "Card-blind probe for positional sensitivity, not a sound range strategy.",
    ),
    PolicyDescriptor(
        "preflop_raiser_v1",
        "street_probe",
        "preflop-only aggression",
        False,
        False,
        False,
        "fast",
        "Minimum-raises preflop and checks/calls every later street.",
        "Separates preflop pressure from postflop skill while ignoring cards.",
    ),
    PolicyDescriptor(
        "flop_bettor_v1",
        "street_probe",
        "flop-only aggression",
        False,
        False,
        False,
        "fast",
        "Uses half-pot raises only on the flop.",
        "A card-blind diagnostic for responses to flop pressure.",
    ),
    PolicyDescriptor(
        "turn_bettor_v1",
        "street_probe",
        "turn-only aggression",
        False,
        False,
        False,
        "fast",
        "Uses pot-sized raises only on the turn.",
        "A card-blind diagnostic, deliberately unrealistic as a full strategy.",
    ),
    PolicyDescriptor(
        "river_overbettor_v1",
        "street_probe",
        "river-only overbetting",
        False,
        False,
        False,
        "fast",
        "Checks/calls until the river, then prefers a two-pot raise.",
        "Useful for river-pressure tests but contains no value/bluff range construction.",
    ),
    PolicyDescriptor(
        "postflop_pressure_v1",
        "street_probe",
        "postflop pot pressure",
        False,
        False,
        False,
        "fast",
        "Checks/calls preflop and uses pot-sized raises on all postflop streets.",
        "Card-blind stress policy with intentionally excessive postflop aggression.",
    ),
    PolicyDescriptor(
        "spr_jammer_v1",
        "stack_probe",
        "low-SPR all-in pressure",
        False,
        False,
        False,
        "fast",
        "Moves all-in whenever public SPR is at most two.",
        "Ignores private-card strength and exists to test stack-depth behavior.",
    ),
    PolicyDescriptor(
        "geometric_sizer_v1",
        "sizing_probe",
        "street-escalating sizing",
        False,
        False,
        False,
        "fast",
        "Escalates from min-raise preflop to half-pot, pot and river overbet.",
        "Card-blind sizing diagnostic, not an optimized geometric betting solution.",
    ),
    PolicyDescriptor(
        "maniac_v1",
        "fish",
        "loose hyper-aggressive",
        True,
        False,
        False,
        "fast",
        "Raises about 82% of opportunities, rarely folding.",
        "Fixed frequencies ignore ranges and opponent adaptation.",
    ),
    PolicyDescriptor(
        "loose_passive_v1",
        "heuristic",
        "loose passive",
        True,
        False,
        False,
        "medium",
        "Equity-aware wide caller with rare small value raises.",
        "Equity assumes a uniform opponent range.",
    ),
    PolicyDescriptor(
        "tight_passive_v1",
        "heuristic",
        "nit",
        True,
        False,
        False,
        "medium",
        "Enters a tight equity range and applies little pressure.",
        "Threshold chart is heuristic rather than equilibrium-derived.",
    ),
    PolicyDescriptor(
        "tag_v1",
        "heuristic",
        "tight aggressive",
        True,
        False,
        False,
        "medium",
        "Position-aware equity thresholds with assertive value betting and rare bluffs.",
        "Does not infer an opponent range or solve subgames.",
    ),
    PolicyDescriptor(
        "lag_v1",
        "heuristic",
        "loose aggressive",
        True,
        False,
        False,
        "medium",
        "Wider continuation range, overbet sizing and frequent bluffs.",
        "Fixed bluff frequencies can be heavily exploited.",
    ),
    PolicyDescriptor(
        "pot_odds_equity_v1",
        "heuristic",
        "mathematical baseline",
        True,
        False,
        False,
        "medium",
        "Compares uniform-range equity with pot odds and fixed value thresholds.",
        "Uniform-range equity is misspecified against structured opponents.",
    ),
    PolicyDescriptor(
        "equity_value_v1",
        "exploit",
        "no-bluff value",
        False,
        False,
        False,
        "medium",
        "Purpose-built value-betting exploit for a pure calling station.",
        "Specialized exploit; strong calling-station results do not imply robustness.",
    ),
    PolicyDescriptor(
        "noisy_tag_v1",
        "composition",
        "TAG plus exploration",
        True,
        False,
        False,
        "medium",
        "Mixes TAG action probabilities with 8% uniform exploration.",
        "Exploration intentionally sacrifices frozen-policy strength.",
    ),
    PolicyDescriptor(
        "balanced_mix_v1",
        "composition",
        "mixture of expert styles",
        True,
        False,
        False,
        "slow",
        "Action-level mixture of TAG, LAG and nit distributions.",
        "Runs all three equity policies per decision and is not a learned ensemble.",
    ),
    PolicyDescriptor(
        "chart_tag_v1",
        "chart",
        "range-chart tight aggressive",
        True,
        False,
        False,
        "medium",
        "Uses a deterministic Chen-style TAG chart preflop and equity play postflop.",
        "The hand chart is explainable but heuristic, not solver-derived.",
    ),
    PolicyDescriptor(
        "chart_lag_v1",
        "chart",
        "range-chart loose aggressive",
        True,
        False,
        False,
        "medium",
        "Opens and defends a wider preflop chart, then delegates to LAG postflop play.",
        "Wide fixed ranges and bluff rates remain exploitable.",
    ),
    PolicyDescriptor(
        "chart_nit_v1",
        "chart",
        "range-chart nit",
        True,
        False,
        False,
        "medium",
        "Uses a narrow deterministic preflop chart and tight-passive postflop play.",
        "Intentionally over-folds and serves as an overfolder archetype.",
    ),
)

QUICK_SUITE: tuple[str, ...] = (
    "check_fold_v1",
    "calling_station_v1",
    "random_valid_v1",
    "maniac_v1",
    "tag_v1",
    "equity_value_v1",
)

STANDARD_SUITE: tuple[str, ...] = (
    "check_fold_v1",
    "calling_station_v1",
    "random_valid_v1",
    "min_raiser_v1",
    "overbettor_v1",
    "jammer_v1",
    "button_bully_v1",
    "postflop_pressure_v1",
    "spr_jammer_v1",
    "maniac_v1",
    "loose_passive_v1",
    "tight_passive_v1",
    "tag_v1",
    "lag_v1",
    "pot_odds_equity_v1",
    "equity_value_v1",
    "chart_tag_v1",
    "chart_lag_v1",
)

EXTENDED_SUITE: tuple[str, ...] = tuple(descriptor.name for descriptor in POLICY_CATALOG)


def policy_descriptors() -> Mapping[str, PolicyDescriptor]:
    """Return descriptors keyed by their unique versioned names."""

    descriptors = {descriptor.name: descriptor for descriptor in POLICY_CATALOG}
    if len(descriptors) != len(POLICY_CATALOG):
        raise RuntimeError("policy catalog contains duplicate names")
    return descriptors


def policy_factories(
    *,
    master_seed: int = 0,
    selected: Sequence[str] | None = None,
    checkpoints: Sequence[Path] = (),
) -> dict[str, PolicyFactory]:
    """Build fresh-instance factories with stable policy-specific RNG streams."""

    available = _all_factories(master_seed)
    checkpoint_names: list[str] = []
    for path in checkpoints:
        template = NumpyMLPPolicy.from_checkpoint(path)
        if template.name in available:
            raise ValueError(f"checkpoint policy duplicates catalog name: {template.name}")

        def make_checkpoint_policy(
            frozen: NumpyMLPPolicy = template,
        ) -> NumpyMLPPolicy:
            return NumpyMLPPolicy(
                frozen.weights,
                encoder=frozen.encoder,
                name=frozen.name,
                seed=stable_seed(master_seed, "checkpoint_policy", frozen.name),
                checkpoint_sha256=frozen.checkpoint_sha256,
            )

        available[template.name] = make_checkpoint_policy
        checkpoint_names.append(template.name)

    names = (
        tuple(selected)
        if selected is not None
        else (*STANDARD_SUITE, *checkpoint_names)
    )
    if len(names) != len(set(names)):
        raise ValueError("selected policy names must be unique")
    unknown = set(names).difference(available)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    return {name: available[name] for name in names}


def suite_names(suite: str) -> tuple[str, ...]:
    """Resolve a named convenience suite."""

    suites = {
        "quick": QUICK_SUITE,
        "standard": STANDARD_SUITE,
        "extended": EXTENDED_SUITE,
    }
    try:
        return suites[suite]
    except KeyError as error:
        raise ValueError(f"unknown suite: {suite}") from error


def _all_factories(master_seed: int) -> dict[str, PolicyFactory]:
    def seed(name: str) -> int:
        return stable_seed(master_seed, "policy", name)

    return {
        "check_fold_v1": lambda: CheckFoldPolicy(),
        "calling_station_v1": CallingStationPolicy,
        "random_valid_v1": lambda: RandomValidPolicy(seed=seed("random_valid_v1")),
        "min_raiser_v1": lambda: BetSizingPolicy(
            MIN_RAISE_PROFILE,
            name="min_raiser_v1",
        ),
        "half_pot_bettor_v1": lambda: BetSizingPolicy(
            HALF_POT_PROFILE,
            name="half_pot_bettor_v1",
        ),
        "pot_bettor_v1": lambda: BetSizingPolicy(POT_PROFILE, name="pot_bettor_v1"),
        "overbettor_v1": lambda: BetSizingPolicy(
            OVERBET_PROFILE,
            name="overbettor_v1",
        ),
        "jammer_v1": lambda: BetSizingPolicy(ALL_IN_PROFILE, name="jammer_v1"),
        "button_bully_v1": lambda: ButtonPressurePolicy(name="button_bully_v1"),
        "preflop_raiser_v1": lambda: StreetPressurePolicy(
            ("preflop",),
            MIN_RAISE_PROFILE,
            "preflop_raiser_v1",
        ),
        "flop_bettor_v1": lambda: StreetPressurePolicy(
            ("flop",),
            HALF_POT_PROFILE,
            "flop_bettor_v1",
        ),
        "turn_bettor_v1": lambda: StreetPressurePolicy(
            ("turn",),
            POT_PROFILE,
            "turn_bettor_v1",
        ),
        "river_overbettor_v1": lambda: StreetPressurePolicy(
            ("river",),
            OVERBET_PROFILE,
            "river_overbettor_v1",
        ),
        "postflop_pressure_v1": lambda: StreetPressurePolicy(
            ("flop", "turn", "river"),
            POT_PROFILE,
            "postflop_pressure_v1",
        ),
        "spr_jammer_v1": lambda: SPRJamPolicy(name="spr_jammer_v1"),
        "geometric_sizer_v1": lambda: GeometricSizingPolicy(),
        "maniac_v1": lambda: ManiacPolicy(seed=seed("maniac_v1")),
        "loose_passive_v1": lambda: LoosePassivePolicy(seed=seed("loose_passive_v1")),
        "tight_passive_v1": lambda: TightPassivePolicy(seed=seed("tight_passive_v1")),
        "tag_v1": lambda: TightAggressivePolicy(seed=seed("tag_v1")),
        "lag_v1": lambda: LooseAggressivePolicy(seed=seed("lag_v1")),
        "pot_odds_equity_v1": lambda: PotOddsEquityPolicy(
            seed=seed("pot_odds_equity_v1")
        ),
        "equity_value_v1": lambda: EquityValuePolicy(seed=seed("equity_value_v1")),
        "noisy_tag_v1": lambda: EpsilonExplorationPolicy(
            TightAggressivePolicy(seed=seed("noisy_tag_v1:tag")),
            0.08,
            seed=seed("noisy_tag_v1:wrapper"),
            name="noisy_tag_v1",
        ),
        "balanced_mix_v1": lambda: MixedPolicy(
            (
                TightAggressivePolicy(seed=seed("balanced_mix_v1:tag")),
                LooseAggressivePolicy(seed=seed("balanced_mix_v1:lag")),
                TightPassivePolicy(seed=seed("balanced_mix_v1:nit")),
            ),
            (0.60, 0.20, 0.20),
            seed=seed("balanced_mix_v1:wrapper"),
            name="balanced_mix_v1",
        ),
        "chart_tag_v1": lambda: PreflopChartPolicy(
            PotOddsEquityPolicy(seed=seed("chart_tag_v1:postflop")),
            config=TAG_CHART,
            name="chart_tag_v1",
        ),
        "chart_lag_v1": lambda: PreflopChartPolicy(
            LooseAggressivePolicy(seed=seed("chart_lag_v1:postflop")),
            config=LAG_CHART,
            name="chart_lag_v1",
        ),
        "chart_nit_v1": lambda: PreflopChartPolicy(
            TightPassivePolicy(seed=seed("chart_nit_v1:postflop")),
            config=NIT_CHART,
            name="chart_nit_v1",
        ),
    }


__all__ = [
    "EXTENDED_SUITE",
    "POLICY_CATALOG",
    "QUICK_SUITE",
    "STANDARD_SUITE",
    "PolicyDescriptor",
    "policy_descriptors",
    "policy_factories",
    "suite_names",
]
