from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from poker_research.neural import (
    ACTION_NAMES,
    ENCODER_VERSION,
    EncoderConfig,
    MLPWeights,
    NumpyMLPPolicy,
    ObservationEncoder,
    save_checkpoint,
)
from poker_research.types import ActionOption, Observation, PublicAction


def _observation(*, hand_id: str = "hand-1") -> Observation:
    return Observation(
        hand_id=hand_id,
        seat=1,
        street="turn",
        position="button_sb",
        hole_cards=("As", "Kh"),
        board_cards=("2c", "7d", "Th", "Jc"),
        stacks=(150, 142),
        street_bets=(8, 16),
        pot=108,
        call_amount=8,
        effective_stack=142,
        spr=1.3148,
        legal_actions=(
            ActionOption("fold"),
            ActionOption("check_call"),
            ActionOption("raise_min", 24),
            ActionOption("raise_all_in", 158),
        ),
        history=(
            PublicAction(1, "raise_half_pot", 8),
            PublicAction(0, "check_call", 8),
            PublicAction(0, "raise_min", 16),
        ),
    )


def test_encoder_is_fixed_width_deterministic_and_public() -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=16, equity_seed=42))

    first = encoder.encode(_observation())
    second = encoder.encode(_observation(hand_id="different-hidden-run-id"))

    assert encoder.to_dict()["version"] == ENCODER_VERSION
    assert first.features.shape == (encoder.feature_count,)
    assert first.features.dtype == np.float32
    assert np.array_equal(first.features, second.features)
    assert first.legal_mask.tolist() == [True, True, True, False, False, False, True]
    assert "opponent_hole" not in " ".join(encoder.feature_names)


def test_encoder_card_slots_and_history_features_have_expected_values() -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    encoded = encoder.encode(_observation())
    values = dict(zip(encoder.feature_names, encoded.features, strict=True))

    assert values["card_0_rank_A"] == 1.0
    assert values["card_0_suit_s"] == 1.0
    assert values["card_6_present"] == 0.0
    assert values["street_turn"] == 1.0
    assert values["position_button_sb"] == 1.0
    assert values["history_self_raise_count"] == pytest.approx(0.05)
    assert values["history_opponent_raise_count"] == pytest.approx(0.05)
    assert values["last_actor_opponent"] == 1.0
    assert values["equity_vs_uniform"] == 0.5


@pytest.mark.parametrize("card", ["ZZ", "A", "Asx", "Aq"])
def test_encoder_rejects_invalid_visible_cards(card: str) -> None:
    malformed = replace(_observation(), hole_cards=(card, "Kh"))
    with pytest.raises(ValueError):
        ObservationEncoder(EncoderConfig(equity_samples=0)).encode(malformed)


def test_neural_policy_masks_illegal_action_even_when_its_logit_is_largest() -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    weights = MLPWeights.random(encoder.feature_count, hidden_size=8, seed=1)
    action_bias = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    action_bias[ACTION_NAMES.index("raise_2pot")] = 1_000.0
    action_bias[ACTION_NAMES.index("check_call")] = 10.0
    biased = MLPWeights(
        weights.input_to_hidden,
        weights.hidden_bias,
        weights.hidden_to_hidden,
        weights.second_hidden_bias,
        weights.hidden_to_action,
        action_bias,
    )

    decision = NumpyMLPPolicy(biased, encoder=encoder).decide(_observation())

    assert decision.action == "check_call"
    assert set(dict(decision.probabilities)) == {
        "fold",
        "check_call",
        "raise_min",
        "raise_all_in",
    }
    assert sum(dict(decision.probabilities).values()) == pytest.approx(1.0)


def test_temperature_sampling_is_reproducible_per_public_decision() -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    weights = MLPWeights.random(encoder.feature_count, hidden_size=8, seed=7)
    policy = NumpyMLPPolicy(weights, encoder=encoder, temperature=1.0, seed=99)

    first = policy.decide(_observation())
    second = policy.decide(_observation())

    assert first == second
    assert sum(dict(first.probabilities).values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in dict(first.probabilities).values())


def test_safe_checkpoint_round_trip_and_hash_guard(tmp_path) -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    weights = MLPWeights.random(encoder.feature_count, hidden_size=16, seed=123)
    path = tmp_path / "policy.npz"

    sha256 = save_checkpoint(path, weights, encoder, policy_name="imitation_mlp_v1")
    loaded = NumpyMLPPolicy.from_checkpoint(path, expected_sha256=sha256)

    assert loaded.name == "imitation_mlp_v1"
    assert loaded.checkpoint_sha256 == sha256
    assert loaded.decide(_observation()).action == NumpyMLPPolicy(
        weights,
        encoder=encoder,
        name="imitation_mlp_v1",
    ).decide(_observation()).action
    with pytest.raises(ValueError, match="SHA-256"):
        NumpyMLPPolicy.from_checkpoint(path, expected_sha256="0" * 64)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_checkpoint(path, weights, encoder, policy_name="imitation_mlp_v1")


def test_checkpoint_rejects_executable_object_arrays(tmp_path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(path, checkpoint_version=np.asarray([{"call": "me"}], dtype=object))

    with pytest.raises(ValueError, match="Object arrays"):
        NumpyMLPPolicy.from_checkpoint(path)


def test_weight_shape_and_dtype_validation() -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    weights = MLPWeights.random(encoder.feature_count, hidden_size=8)
    malformed = MLPWeights(
        weights.input_to_hidden.astype(np.float64),  # type: ignore[arg-type]
        weights.hidden_bias,
        weights.hidden_to_hidden,
        weights.second_hidden_bias,
        weights.hidden_to_action,
        weights.action_bias,
    )

    with pytest.raises(ValueError, match="float32"):
        malformed.validate(encoder.feature_count)


def test_frozen_weight_container_owns_read_only_arrays() -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    source = MLPWeights.random(encoder.feature_count, hidden_size=8)
    weights = MLPWeights(
        source.input_to_hidden,
        source.hidden_bias,
        source.hidden_to_hidden,
        source.second_hidden_bias,
        source.hidden_to_action,
        source.action_bias,
    )

    assert all(
        not getattr(weights, name).flags.writeable
        for name in (
            "input_to_hidden",
            "hidden_bias",
            "hidden_to_hidden",
            "second_hidden_bias",
            "hidden_to_action",
            "action_bias",
        )
    )
    with pytest.raises(ValueError, match="read-only"):
        weights.action_bias[0] = 1.0
