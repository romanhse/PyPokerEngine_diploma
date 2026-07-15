from __future__ import annotations

import pytest

from poker_research.arena import ArenaConfig, play_hand
from poker_research.catalog import (
    EXTENDED_SUITE,
    POLICY_CATALOG,
    QUICK_SUITE,
    STANDARD_SUITE,
    policy_descriptors,
    policy_factories,
    suite_names,
)
from poker_research.neural import EncoderConfig, MLPWeights, ObservationEncoder, save_checkpoint


def test_catalog_names_are_unique_versioned_and_described() -> None:
    descriptors = policy_descriptors()

    assert len(descriptors) == len(POLICY_CATALOG) >= 28
    assert tuple(descriptors) == EXTENDED_SUITE
    assert all(name.endswith("_v1") for name in descriptors)
    assert all(descriptor.description for descriptor in descriptors.values())
    assert all(descriptor.limitation for descriptor in descriptors.values())
    assert {descriptor.family for descriptor in descriptors.values()} >= {
        "negative_control",
        "fish",
        "sizing_probe",
        "heuristic",
        "exploit",
        "composition",
        "chart",
        "position_probe",
        "stack_probe",
        "street_probe",
    }


def test_named_suites_are_nested_and_valid() -> None:
    assert set(QUICK_SUITE) < set(STANDARD_SUITE) < set(EXTENDED_SUITE)
    assert suite_names("quick") == QUICK_SUITE
    assert suite_names("standard") == STANDARD_SUITE
    assert suite_names("extended") == EXTENDED_SUITE
    with pytest.raises(ValueError, match="unknown suite"):
        suite_names("secret")


def test_factories_return_fresh_correctly_named_policies() -> None:
    factories = policy_factories(master_seed=17, selected=EXTENDED_SUITE)

    for expected_name, factory in factories.items():
        first = factory()
        second = factory()
        assert first.name == expected_name
        assert second.name == expected_name
        assert first is not second


def test_selected_factories_preserve_requested_order() -> None:
    selected = ("tag_v1", "calling_station_v1", "maniac_v1")
    factories = policy_factories(master_seed=1, selected=selected)

    assert tuple(factories) == selected
    with pytest.raises(ValueError, match="unique"):
        policy_factories(selected=("tag_v1", "tag_v1"))
    with pytest.raises(ValueError, match="unknown policies"):
        policy_factories(selected=("missing_v1",))


def test_every_quick_policy_plays_real_hands_without_illegal_actions() -> None:
    factories = policy_factories(master_seed=99, selected=QUICK_SUITE)
    opponent_factory = factories["calling_station_v1"]
    config = ArenaConfig(starting_stack=40, small_blind=1, big_blind=2)

    for index, (name, factory) in enumerate(factories.items()):
        if name == "calling_station_v1":
            continue
        result = play_hand(
            (factory(), opponent_factory()),
            deal_seed=1_000 + index,
            hand_id=f"catalog-smoke-{index}",
            config=config,
        )
        assert sum(result.payoffs) == 0
        assert sum(result.finishing_stacks) == 80


def test_safe_neural_checkpoint_can_join_any_league(tmp_path) -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    weights = MLPWeights.random(encoder.feature_count, hidden_size=8, seed=7)
    checkpoint = tmp_path / "neural.npz"
    save_checkpoint(checkpoint, weights, encoder, policy_name="trained_neural_v1")

    factories = policy_factories(
        master_seed=42,
        selected=("calling_station_v1", "trained_neural_v1"),
        checkpoints=(checkpoint,),
    )

    first = factories["trained_neural_v1"]()
    second = factories["trained_neural_v1"]()
    assert first.name == second.name == "trained_neural_v1"
    assert first is not second
    assert first.checkpoint_sha256 == second.checkpoint_sha256
