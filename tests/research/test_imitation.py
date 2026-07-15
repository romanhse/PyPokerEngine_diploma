"""Behavior-cloning tests for public data, grouped splits, and safe export."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from poker_research.imitation import (
    BehaviorCloningConfig,
    Demonstration,
    RecordingTeacherPolicy,
    _dataset_arrays,
    _masked_logits,
    collect_demonstrations,
    split_demonstrations,
    train_behavior_cloning,
)
from poker_research.neural import (
    ACTION_NAMES,
    EncoderConfig,
    MLPWeights,
    NumpyMLPPolicy,
    ObservationEncoder,
)
from poker_research.types import Decision, Observation
from tests.research.helpers import deterministic_decision


class _CheckCallPolicy:
    def __init__(self, name: str) -> None:
        self.name = name

    def decide(self, observation: Observation) -> Decision:
        return deterministic_decision(observation, "check_call")


def _check_call_factory(
    name: str,
) -> Callable[[int, int], _CheckCallPolicy]:
    def make(_policy_seed: int, _seat: int) -> _CheckCallPolicy:
        return _CheckCallPolicy(name)

    return make


@pytest.fixture
def encoder() -> ObservationEncoder:
    return ObservationEncoder(EncoderConfig(equity_samples=0))


@pytest.fixture
def demonstrations(encoder: ObservationEncoder) -> tuple[Demonstration, ...]:
    return collect_demonstrations(
        _check_call_factory("teacher_check_call_v1"),
        _check_call_factory("opponent_check_call_v1"),
        (918_273_645, 564_738_291, 192_837_465, 837_261_945),
        encoder=encoder,
        policy_seed=73,
    )


def test_collection_keeps_deck_seed_out_of_policy_inputs_and_features(
    encoder: ObservationEncoder,
) -> None:
    deal_seeds = (918_273_645, 564_738_291)
    teacher_factory_calls: list[tuple[int, int]] = []
    opponent_factory_calls: list[tuple[int, int]] = []

    def teacher_factory(policy_seed: int, seat: int) -> _CheckCallPolicy:
        teacher_factory_calls.append((policy_seed, seat))
        return _CheckCallPolicy("public_teacher_v1")

    def opponent_factory(policy_seed: int, seat: int) -> _CheckCallPolicy:
        opponent_factory_calls.append((policy_seed, seat))
        return _CheckCallPolicy("public_opponent_v1")

    collected = collect_demonstrations(
        teacher_factory,
        opponent_factory,
        deal_seeds,
        encoder=encoder,
        policy_seed=991,
    )

    assert collected
    assert len(teacher_factory_calls) == len(deal_seeds) * 2
    assert len(opponent_factory_calls) == len(deal_seeds) * 2
    supplied_policy_seeds = {
        seed for seed, _seat in (*teacher_factory_calls, *opponent_factory_calls)
    }
    assert supplied_policy_seeds.isdisjoint(deal_seeds)
    assert {seat for _seed, seat in teacher_factory_calls} == {0, 1}
    assert {sample.deal_seed for sample in collected} == set(deal_seeds)

    public_fields = {field.name for field in fields(Observation)}
    assert "deal_seed" not in public_fields
    assert "deck" not in public_fields
    assert not any("opponent_hole" in name for name in encoder.feature_names)
    for sample in collected:
        assert all(str(deal_seed) not in sample.hand_id for deal_seed in deal_seeds)
        reencoded = encoder.encode(sample.observation)
        assert np.array_equal(sample.features, reencoded.features)
        assert np.array_equal(sample.legal_mask, reencoded.legal_mask)
        assert sample.features.flags.writeable is False
        assert sample.legal_mask.flags.writeable is False

    repeated_teacher_calls: list[tuple[int, int]] = []

    def repeated_teacher_factory(policy_seed: int, seat: int) -> _CheckCallPolicy:
        repeated_teacher_calls.append((policy_seed, seat))
        return _CheckCallPolicy("public_teacher_v1")

    repeated = collect_demonstrations(
        repeated_teacher_factory,
        _check_call_factory("public_opponent_v1"),
        deal_seeds,
        encoder=encoder,
        policy_seed=991,
    )
    assert repeated_teacher_calls == teacher_factory_calls
    assert [sample.hand_id for sample in repeated] == [
        sample.hand_id for sample in collected
    ]
    assert [sample.action_index for sample in repeated] == [
        sample.action_index for sample in collected
    ]


def test_recording_wrapper_rejects_malformed_teacher_distribution(
    encoder: ObservationEncoder,
    demonstrations: tuple[Demonstration, ...],
) -> None:
    class _MissingProbabilityPolicy:
        name = "missing_probability_v1"

        def decide(self, observation: Observation) -> Decision:
            valid = deterministic_decision(observation, "check_call")
            return Decision(valid.action, valid.probabilities[:-1])

    recorder = RecordingTeacherPolicy(
        _MissingProbabilityPolicy(),
        encoder,
        deal_seed=123,
    )

    with pytest.raises(ValueError, match="cover every legal action"):
        recorder.decide(demonstrations[0].observation)
    assert recorder.demonstrations == ()


def test_split_is_deterministic_and_keeps_whole_deals_together(
    demonstrations: tuple[Demonstration, ...],
) -> None:
    first = split_demonstrations(
        demonstrations,
        validation_fraction=0.25,
        seed=17,
    )
    repeated = split_demonstrations(
        demonstrations,
        validation_fraction=0.25,
        seed=17,
    )

    assert first.train_deal_seeds == repeated.train_deal_seeds
    assert first.validation_deal_seeds == repeated.validation_deal_seeds
    assert [id(sample) for sample in first.train] == [
        id(sample) for sample in repeated.train
    ]
    assert [id(sample) for sample in first.validation] == [
        id(sample) for sample in repeated.validation
    ]
    assert set(first.train_deal_seeds).isdisjoint(first.validation_deal_seeds)
    for deal_seed in {sample.deal_seed for sample in demonstrations}:
        train_count = sum(sample.deal_seed == deal_seed for sample in first.train)
        validation_count = sum(
            sample.deal_seed == deal_seed for sample in first.validation
        )
        expected_count = sum(
            sample.deal_seed == deal_seed for sample in demonstrations
        )
        assert (train_count, validation_count).count(0) == 1
        assert train_count + validation_count == expected_count


def test_demonstration_owns_read_only_feature_and_mask_copies(
    demonstrations: tuple[Demonstration, ...],
) -> None:
    original = demonstrations[0]
    features = original.features.copy()
    legal_mask = original.legal_mask.copy()
    copied = Demonstration(
        deal_seed=original.deal_seed,
        hand_id=original.hand_id,
        seat=original.seat,
        decision_index=original.decision_index,
        teacher_name=original.teacher_name,
        observation=original.observation,
        features=features,
        legal_mask=legal_mask,
        action_index=original.action_index,
    )

    features[:] = 123.0
    legal_mask[:] = False
    assert not np.all(copied.features == 123.0)
    assert copied.legal_mask[copied.action_index]
    with pytest.raises(ValueError, match="read-only"):
        copied.features[0] = 1.0


def test_training_dataset_recomputes_public_encoding_and_rejects_extra_signal(
    encoder: ObservationEncoder,
    demonstrations: tuple[Demonstration, ...],
) -> None:
    original = demonstrations[0]
    tampered_features = original.features.copy()
    tampered_features[0] = 1.0 - tampered_features[0]
    tampered = replace(original, features=tampered_features)

    with pytest.raises(ValueError, match="features do not match its public observation"):
        _dataset_arrays((tampered,), encoder)


def test_masked_cross_entropy_ignores_illegal_logits_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[1.0, 1_000.0, -1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    legal_mask = torch.tensor([[True, False, True]])
    label = torch.tensor([0])

    masked = _masked_logits(torch, logits, legal_mask)
    loss = torch.nn.functional.cross_entropy(masked, label)
    expected = torch.logsumexp(logits[0, [0, 2]], dim=0) - logits[0, 0]

    assert masked.argmax(dim=1).item() == 0
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1].item() == 0.0


def test_cpu_training_is_deterministic_and_exports_safe_npz(
    tmp_path: Path,
    encoder: ObservationEncoder,
    demonstrations: tuple[Demonstration, ...],
) -> None:
    torch = pytest.importorskip("torch")
    config = BehaviorCloningConfig(
        hidden_size=8,
        epochs=2,
        batch_size=8,
        learning_rate=0.01,
        seed=31,
        split_seed=19,
        validation_fraction=0.25,
        device="cpu",
        policy_name="behavior_cloning_test_v1",
    )
    rng_before = torch.random.get_rng_state().clone()

    first = train_behavior_cloning(
        demonstrations,
        encoder,
        tmp_path / "first.npz",
        config=config,
    )
    rng_after_first = torch.random.get_rng_state().clone()
    second = train_behavior_cloning(
        demonstrations,
        encoder,
        tmp_path / "second.npz",
        config=config,
    )

    assert torch.equal(rng_before, rng_after_first)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert first.device == second.device == "cpu"
    assert first.history == second.history
    for weight_field in fields(MLPWeights):
        assert np.array_equal(
            getattr(first.weights, weight_field.name),
            getattr(second.weights, weight_field.name),
        )
    assert all(
        metrics.train_illegal_predictions == 0
        and metrics.validation_illegal_predictions == 0
        for metrics in first.history
    )

    with np.load(first.checkpoint_path, allow_pickle=False) as checkpoint:
        assert checkpoint.files
        assert all(not checkpoint[name].dtype.hasobject for name in checkpoint.files)
    assert len(first.checkpoint_sha256) == 64
    loaded = NumpyMLPPolicy.from_checkpoint(
        first.checkpoint_path,
        expected_sha256=first.checkpoint_sha256,
    )
    decision = loaded.decide(demonstrations[0].observation)
    assert decision.action in {
        option.name for option in demonstrations[0].observation.legal_actions
    }
    assert set(dict(decision.probabilities)) == {
        option.name for option in demonstrations[0].observation.legal_actions
    }
    assert sum(dict(decision.probabilities).values()) == pytest.approx(1.0)
    assert first.weights.action_bias.shape == (len(ACTION_NAMES),)
