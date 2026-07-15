"""Contract and behavior tests for the heuristic baseline policy suite."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import is_dataclass, replace

import pytest

import poker_research.baselines as baselines
from poker_research.arena import ArenaConfig, play_hand
from poker_research.baselines import (
    ALL_IN_PROFILE,
    HALF_POT_PROFILE,
    LAG_CONFIG,
    MIN_RAISE_PROFILE,
    OVERBET_PROFILE,
    POT_PROFILE,
    TAG_CONFIG,
    BetSizingPolicy,
    BetSizingProfile,
    ButtonPressurePolicy,
    CallCheckPolicy,
    CheckFoldPolicy,
    EquityThresholdConfig,
    GeometricSizingPolicy,
    LAGPolicy,
    LoosePassivePolicy,
    ManiacConfig,
    ManiacPolicy,
    NitPolicy,
    PotOddsEquityPolicy,
    RandomValidPolicy,
    SPRJamPolicy,
    StreetPressurePolicy,
    TAGPolicy,
    TightPassivePolicy,
)
from poker_research.types import ActionOption, Decision, Observation, Policy

FULL_ACTIONS = (
    ActionOption("fold"),
    ActionOption("check_call"),
    ActionOption("raise_min", 4),
    ActionOption("raise_half_pot", 6),
    ActionOption("raise_pot", 8),
    ActionOption("raise_2pot", 12),
    ActionOption("raise_all_in", 200),
)


def _observation(
    *,
    hand_id: str = "baseline-test",
    street: str = "preflop",
    position: str = "button_sb",
    call_amount: int = 1,
    pot: int = 3,
    spr: float = 10.0,
    legal_actions: tuple[ActionOption, ...] = FULL_ACTIONS,
) -> Observation:
    boards = {
        "preflop": (),
        "flop": ("2c", "7d", "Th"),
        "turn": ("2c", "7d", "Th", "Js"),
        "river": ("2c", "7d", "Th", "Js", "Qc"),
    }
    return Observation(
        hand_id=hand_id,
        seat=1 if position == "button_sb" else 0,
        street=street,
        position=position,
        hole_cards=("As", "Kh"),
        board_cards=boards.get(street, ()),
        stacks=(198, 199),
        street_bets=(2, 1),
        pot=pot,
        call_amount=call_amount,
        effective_stack=198,
        spr=spr,
        legal_actions=legal_actions,
    )


def _assert_valid_decision(observation: Observation, decision: Decision) -> None:
    legal = {option.name for option in observation.legal_actions}
    probabilities = dict(decision.probabilities)
    assert decision.action in legal
    assert probabilities.keys() == legal
    assert all(0.0 <= probability <= 1.0 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[decision.action] > 0.0


@pytest.mark.parametrize(
    "policy",
    [
        CheckFoldPolicy(),
        CallCheckPolicy(),
        RandomValidPolicy(seed=7),
        BetSizingPolicy(raise_probability=0.6, seed=7),
        ButtonPressurePolicy(),
        StreetPressurePolicy(),
        SPRJamPolicy(),
        GeometricSizingPolicy(),
        ManiacPolicy(seed=7),
        PotOddsEquityPolicy(seed=7),
        LoosePassivePolicy(seed=7),
        TightPassivePolicy(seed=7),
        TAGPolicy(seed=7),
        LAGPolicy(seed=7),
    ],
)
def test_every_public_policy_is_a_dataclass_and_emits_a_full_legal_distribution(
    policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baselines, "equity_vs_uniform", lambda *args, **kwargs: 0.55)
    observation = _observation()

    decision = policy.decide(observation)

    assert is_dataclass(policy)
    _assert_valid_decision(observation, decision)


def test_check_fold_checks_for_free_and_folds_when_facing_a_bet() -> None:
    policy = CheckFoldPolicy()

    assert policy.decide(_observation(call_amount=0)).action == "check_call"
    assert policy.decide(_observation(call_amount=3)).action == "fold"


def test_call_check_uses_a_legal_fallback_if_check_call_is_absent() -> None:
    policy = CallCheckPolicy()
    normal = _observation()
    fold_only = _observation(legal_actions=(ActionOption("fold"),))

    assert policy.decide(normal).action == "check_call"
    decision = policy.decide(fold_only)
    assert decision.action == "fold"
    _assert_valid_decision(fold_only, decision)


def test_random_valid_is_seeded_reproducible_uniform_and_global_rng_isolated() -> None:
    observations = tuple(
        _observation(hand_id=f"random-valid-{index}") for index in range(32)
    )
    first = RandomValidPolicy(seed=917)
    replay = RandomValidPolicy(seed=917)
    random.seed(11_903)
    global_state = random.getstate()

    first_actions = [first.decide(observation).action for observation in observations]
    replay_actions = [replay.decide(observation).action for observation in observations]
    fresh_actions = [
        RandomValidPolicy(seed=917).decide(observation).action
        for observation in observations
    ]
    other_actions = [
        RandomValidPolicy(seed=918).decide(observation).action
        for observation in observations
    ]

    assert first_actions == replay_actions == fresh_actions
    assert first_actions != other_actions
    assert len(set(first_actions)) > 1
    assert first.decide(observations[0]) == first.decide(observations[0])
    assert random.getstate() == global_state
    probabilities = dict(RandomValidPolicy(seed=1).decide(observations[0]).probabilities)
    assert all(
        probability == pytest.approx(1 / len(FULL_ACTIONS))
        for probability in probabilities.values()
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RandomValidPolicy(seed=8_101),
        lambda: BetSizingPolicy(raise_probability=0.5, seed=8_101),
        lambda: ManiacPolicy(
            ManiacConfig(raise_probability=0.45, fold_probability=0.20),
            seed=8_101,
        ),
        lambda: PotOddsEquityPolicy(
            config=replace(
                baselines.POT_ODDS_EQUITY_CONFIG,
                value_raise_probability=0.5,
            ),
            seed=8_101,
        ),
    ],
)
def test_every_stochastic_baseline_matches_fresh_per_hand_factory_semantics(
    factory: Callable[[], Policy],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baselines, "equity_vs_uniform", lambda *args, **kwargs: 0.90)
    observations = tuple(
        _observation(hand_id=f"stochastic-baseline-{index}") for index in range(32)
    )
    reused = factory()
    replay = factory()

    reused_decisions = [reused.decide(observation) for observation in observations]
    replay_actions = [replay.decide(observation).action for observation in observations]
    fresh_actions = [factory().decide(observation).action for observation in observations]

    reused_actions = [decision.action for decision in reused_decisions]
    assert reused_actions == replay_actions == fresh_actions
    assert len(set(reused_actions)) > 1
    for observation, decision in zip(observations, reused_decisions, strict=True):
        _assert_valid_decision(observation, decision)


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (MIN_RAISE_PROFILE, "raise_min"),
        (HALF_POT_PROFILE, "raise_half_pot"),
        (POT_PROFILE, "raise_pot"),
        (OVERBET_PROFILE, "raise_2pot"),
        (ALL_IN_PROFILE, "raise_all_in"),
    ],
)
def test_bet_sizing_profiles_choose_their_preferred_legal_raise(
    profile: BetSizingProfile,
    expected: str,
) -> None:
    observation = _observation()
    policy = BetSizingPolicy(
        profile=profile,
        raise_probability=1.0,
        seed=3,
        name=f"sizing_{profile.name}",
    )

    decision = policy.decide(observation)

    assert decision.action == expected
    assert dict(decision.metadata)["profile"] == profile.name
    _assert_valid_decision(observation, decision)


def test_bet_sizing_profile_falls_back_to_an_available_raise_or_check_call() -> None:
    min_only = _observation(
        legal_actions=(
            ActionOption("fold"),
            ActionOption("check_call"),
            ActionOption("raise_min", 4),
        )
    )
    no_raise = _observation(
        call_amount=0,
        legal_actions=(ActionOption("fold"), ActionOption("check_call")),
    )
    policy = BetSizingPolicy(profile=ALL_IN_PROFILE, raise_probability=1.0)

    assert policy.decide(min_only).action == "raise_min"
    assert policy.decide(no_raise).action == "check_call"


def test_public_state_probe_policies_isolate_position_street_spr_and_sizing() -> None:
    button = _observation(position="button_sb")
    big_blind = _observation(position="big_blind")
    assert ButtonPressurePolicy().decide(button).action == "raise_pot"
    assert ButtonPressurePolicy().decide(big_blind).action == "check_call"

    flop_probe = StreetPressurePolicy(("flop",), HALF_POT_PROFILE)
    assert flop_probe.decide(_observation(street="flop")).action == "raise_half_pot"
    assert flop_probe.decide(_observation(street="turn")).action == "check_call"

    jammer = SPRJamPolicy(maximum_spr=2.0)
    assert jammer.decide(_observation(spr=2.0)).action == "raise_all_in"
    assert jammer.decide(_observation(spr=2.1)).action == "check_call"

    geometric = GeometricSizingPolicy()
    expected = {
        "preflop": "raise_min",
        "flop": "raise_half_pot",
        "turn": "raise_pot",
        "river": "raise_2pot",
    }
    assert {
        street: geometric.decide(_observation(street=street)).action
        for street in expected
    } == expected


def test_maniac_reports_exact_action_mix_and_uses_configured_pressure_size() -> None:
    observation = _observation(call_amount=2)
    policy = ManiacPolicy(
        ManiacConfig(
            raise_probability=0.80,
            fold_probability=0.05,
            sizing_profile=OVERBET_PROFILE,
        ),
        seed=13,
    )

    decision = policy.decide(observation)
    probabilities = dict(decision.probabilities)

    assert probabilities["raise_2pot"] == pytest.approx(0.80)
    assert probabilities["fold"] == pytest.approx(0.05)
    assert probabilities["check_call"] == pytest.approx(0.15)
    _assert_valid_decision(observation, decision)


def test_pot_odds_equity_policy_folds_calls_and_value_raises_at_clear_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = PotOddsEquityPolicy(
        config=replace(
            baselines.POT_ODDS_EQUITY_CONFIG,
            value_raise_probability=1.0,
            sizing_profile=POT_PROFILE,
        ),
        seed=5,
    )

    monkeypatch.setattr(baselines, "equity_vs_uniform", lambda *args, **kwargs: 0.10)
    assert policy.decide(_observation(call_amount=5, pot=5)).action == "fold"

    monkeypatch.setattr(baselines, "equity_vs_uniform", lambda *args, **kwargs: 0.55)
    assert policy.decide(_observation(call_amount=1, pot=9)).action == "check_call"

    monkeypatch.setattr(baselines, "equity_vs_uniform", lambda *args, **kwargs: 0.90)
    value_decision = policy.decide(_observation(call_amount=1, pot=9, spr=8.0))
    assert value_decision.action == "raise_pot"
    assert dict(value_decision.metadata)["reason"] == "value_raise"


def test_named_style_profiles_have_distinct_auditable_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baselines, "equity_vs_uniform", lambda *args, **kwargs: 0.45)
    observation = _observation(position="big_blind", call_amount=1, pot=9)

    loose = LoosePassivePolicy(seed=0).decide(observation)
    nit = NitPolicy(seed=0).decide(observation)
    tag = TAGPolicy(seed=0).decide(observation)
    lag = LAGPolicy(seed=0).decide(observation)

    assert dict(loose.probabilities)["check_call"] == pytest.approx(0.995)
    assert dict(nit.probabilities)["fold"] == pytest.approx(1.0)
    assert dict(tag.probabilities)["fold"] == pytest.approx(0.94)
    assert dict(lag.probabilities)["check_call"] == pytest.approx(0.82)
    assert TAG_CONFIG.preflop_call_equity > LAG_CONFIG.preflop_call_equity
    assert TAG_CONFIG.bluff_probability < LAG_CONFIG.bluff_probability


def test_low_spr_high_equity_prefers_an_available_all_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baselines, "equity_vs_uniform", lambda *args, **kwargs: 0.90)
    policy = PotOddsEquityPolicy(seed=0)

    decision = policy.decide(_observation(spr=0.5))

    assert decision.action == "raise_all_in"


def test_equity_policy_rng_and_monte_carlo_seed_are_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_seeds: list[int] = []

    def fixed_equity(*args: object, **kwargs: object) -> float:
        seen_seeds.append(int(kwargs["seed"]))
        return 0.50

    monkeypatch.setattr(baselines, "equity_vs_uniform", fixed_equity)
    observations = tuple(
        _observation(hand_id=f"equity-action-{index}", position="big_blind")
        for index in range(20)
    )
    first = LAGPolicy(seed=71)
    replay = LAGPolicy(seed=71)

    first_actions = [first.decide(observation).action for observation in observations]
    replay_actions = [replay.decide(observation).action for observation in observations]
    fresh_actions = [
        LAGPolicy(seed=71).decide(observation).action for observation in observations
    ]

    assert first_actions == replay_actions == fresh_actions
    assert len(set(seen_seeds)) == 1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CheckFoldPolicy(name="integration_check_fold"),
        lambda: RandomValidPolicy(seed=3, name="integration_random"),
        lambda: BetSizingPolicy(seed=3, name="integration_sizing"),
        lambda: ManiacPolicy(seed=3, name="integration_maniac"),
        lambda: LoosePassivePolicy(seed=3, name="integration_loose_passive"),
        lambda: TightPassivePolicy(seed=3, name="integration_tight_passive"),
        lambda: TAGPolicy(seed=3, name="integration_tag"),
        lambda: LAGPolicy(seed=3, name="integration_lag"),
    ],
)
def test_baseline_runs_end_to_end_without_illegal_actions_or_chip_loss(
    factory: Callable[[], Policy],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baselines, "equity_vs_uniform", lambda *args, **kwargs: 0.55)
    result = play_hand(
        (factory(), CallCheckPolicy(name="integration_opponent")),
        deal_seed=2_026_0715,
        hand_id="baseline-integration",
        config=ArenaConfig(starting_stack=40, small_blind=1, big_blind=2),
    )

    assert sum(result.finishing_stacks) == 80
    assert sum(result.payoffs) == 0
    for record in result.decisions:
        _assert_valid_decision(
            Observation(
                hand_id=result.hand_id,
                seat=record.seat,
                street=record.street,
                position="button_sb" if record.seat == 1 else "big_blind",
                hole_cards=record.hole_cards,
                board_cards=record.board_cards,
                stacks=record.stacks,
                street_bets=record.street_bets,
                pot=record.pot,
                call_amount=record.call_amount,
                effective_stack=min(record.stacks),
                spr=min(record.stacks) / record.pot if record.pot else float("inf"),
                legal_actions=record.legal_actions,
            ),
            record.decision,
        )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: RandomValidPolicy(name=""),
        lambda: BetSizingPolicy(raise_probability=1.1),
        lambda: ManiacConfig(raise_probability=0.8, fold_probability=0.3),
        lambda: EquityThresholdConfig(preflop_samples=0),
        lambda: EquityThresholdConfig(bluff_probability=-0.1),
        lambda: EquityThresholdConfig(jam_spr=-1.0),
        lambda: BetSizingProfile("bad", ("check_call",)),
        lambda: StreetPressurePolicy(()),
        lambda: StreetPressurePolicy(("fifth_street",)),
        lambda: SPRJamPolicy(maximum_spr=float("nan")),
    ],
)
def test_invalid_policy_configs_fail_fast(constructor: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        constructor()
