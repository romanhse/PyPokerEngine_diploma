"""Fast contract tests for the corrected Double DQN research baseline."""

from __future__ import annotations

import math
from importlib.util import find_spec

import numpy as np
import pytest

from poker_research.baselines import CallCheckPolicy
from poker_research.dqn import (
    DoubleDQNTrainer,
    DQNConfig,
    ReplayBuffer,
    Transition,
)
from poker_research.neural import (
    ACTION_NAMES,
    EncoderConfig,
    MLPWeights,
    NumpyMLPPolicy,
    ObservationEncoder,
)
from poker_research.types import ActionOption, Observation

pytestmark = pytest.mark.skipif(
    find_spec("torch") is None,
    reason="Double DQN training extra is not installed",
)


def _opponent_factory(policy_seed: int, seat: int) -> CallCheckPolicy:
    return CallCheckPolicy(name=f"opponent_{policy_seed}_{seat}")


def _trainer(
    *,
    master_seed: int = 7,
    updates_per_transition: int = 0,
    target_sync_interval: int = 100,
) -> DoubleDQNTrainer:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    config = DQNConfig(
        hidden_size=8,
        replay_capacity=64,
        batch_size=1,
        min_replay_size=1,
        updates_per_transition=updates_per_transition,
        target_sync_interval=target_sync_interval,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_decisions=1,
        master_seed=master_seed,
        device="cpu",
    )
    trainer = DoubleDQNTrainer(config=config, encoder=encoder)
    _set_constant_q(trainer, online=[0, 10, 0, 0, 0, 0, 0])
    trainer.synchronize_target()
    return trainer


def _set_constant_q(
    trainer: DoubleDQNTrainer,
    *,
    online: list[float] | None = None,
    target: list[float] | None = None,
) -> None:
    for model, values in ((trainer.online, online), (trainer.target, target)):
        if values is None:
            continue
        with trainer.torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model[4].bias.copy_(
                trainer.torch.tensor(values, dtype=trainer.torch.float32)
            )


def _observation() -> Observation:
    return Observation(
        hand_id="safe-inference",
        seat=1,
        street="preflop",
        position="button_sb",
        hole_cards=("As", "Kh"),
        board_cards=(),
        stacks=(198, 199),
        street_bets=(2, 1),
        pot=3,
        call_amount=1,
        effective_stack=198,
        spr=66.0,
        legal_actions=(ActionOption("fold"), ActionOption("check_call")),
    )


def test_arena_collection_records_every_hero_decision_and_terminal_payoff() -> None:
    trainer = _trainer()

    report = trainer.play(_opponent_factory)

    assert report.mode == "train"
    assert report.hero_seat == 0
    assert report.transitions_added == report.hero_decisions == len(trainer.replay)
    hero_records = tuple(
        record
        for record in report.hand_result.decisions
        if record.policy == trainer.config.policy_name
    )
    assert len(hero_records) == report.hero_decisions
    for index, record in enumerate(hero_records):
        transition = trainer.replay.get(index)
        assert ACTION_NAMES[transition.action_index] == record.decision.action
        assert transition.legal_mask[transition.action_index]
        if index < len(hero_records) - 1:
            assert not transition.terminal
            assert transition.reward_bb == 0.0
            assert transition.next_legal_mask.any()
        else:
            assert transition.terminal
            assert transition.reward_bb == pytest.approx(report.hero_payoff_bb)
            assert not transition.next_legal_mask.any()

    # At least one linked next observation includes an intervening opponent
    # action, proving that the next state is the hero's next public decision,
    # not the raw state immediately after the hero acted.
    last_opponent_index = trainer.encoder.feature_names.index("last_actor_opponent")
    assert any(
        trainer.replay.get(index).next_features[last_opponent_index] == 1.0
        for index in range(len(trainer.replay) - 1)
    )


def test_train_and_validation_alternate_seats_and_never_share_deal_seeds() -> None:
    trainer = _trainer(master_seed=29)
    train_reports = trainer.train_for(2, _opponent_factory)
    replay_size = len(trainer.replay)
    decisions = trainer.environment_decisions

    validation_reports = trainer.validate_for(2, _opponent_factory)

    assert [report.hero_seat for report in train_reports] == [0, 1]
    assert [report.hero_seat for report in validation_reports] == [0, 1]
    assert all(report.deal_seed % 2 == 0 for report in train_reports)
    assert all(report.deal_seed % 2 == 1 for report in validation_reports)
    assert {report.deal_seed for report in train_reports}.isdisjoint(
        report.deal_seed for report in validation_reports
    )
    assert all(report.mode == "eval" for report in validation_reports)
    assert all(report.transitions_added == 0 for report in validation_reports)
    assert len(trainer.replay) == replay_size
    assert trainer.environment_decisions == decisions
    assert trainer.optimization_steps == 0
    assert trainer.mode == "eval"


def test_opponent_policy_seed_and_public_hand_id_do_not_reveal_deck_seed() -> None:
    trainer = _trainer(master_seed=37)
    factory_calls: list[tuple[int, int]] = []

    def opponent_factory(policy_seed: int, seat: int) -> CallCheckPolicy:
        factory_calls.append((policy_seed, seat))
        return CallCheckPolicy(name=f"safe_opponent_{seat}")

    report = trainer.play(opponent_factory)

    assert factory_calls[0][0] != report.deal_seed
    assert str(report.deal_seed) not in report.hand_result.hand_id
    assert report.hand_result.hand_id == "dqn-train-0-hero-seat-0"


def test_trainer_can_start_from_compatible_behavior_cloning_weights() -> None:
    encoder = ObservationEncoder(EncoderConfig(equity_samples=0))
    initial = MLPWeights.random(encoder.feature_count, hidden_size=8, seed=123)
    trainer = DoubleDQNTrainer(
        config=DQNConfig(
            hidden_size=8,
            replay_capacity=8,
            batch_size=1,
            min_replay_size=1,
            epsilon_start=0.0,
            epsilon_end=0.0,
            master_seed=99,
            device="cpu",
        ),
        encoder=encoder,
        initial_weights=initial,
    )

    exported = trainer.online[0].weight.detach().cpu().numpy().T
    assert np.array_equal(exported, initial.input_to_hidden)
    incompatible = MLPWeights.random(encoder.feature_count, hidden_size=7, seed=123)
    with pytest.raises(ValueError, match="hidden size"):
        DoubleDQNTrainer(
            config=trainer.config,
            encoder=encoder,
            initial_weights=incompatible,
        )


def test_double_dqn_uses_online_legal_argmax_and_target_gather_then_syncs() -> None:
    trainer = _trainer(updates_per_transition=0, target_sync_interval=1)
    # The illegal all-in has the largest online Q.  Among legal actions online
    # chooses check/call (4 > 3), while target itself would prefer fold (10 > 7).
    _set_constant_q(
        trainer,
        online=[3, 4, 0, 0, 0, 0, 100],
        target=[10, 7, 0, 0, 0, 0, -20],
    )
    width = trainer.encoder.feature_count
    current_mask = np.zeros(len(ACTION_NAMES), dtype=np.bool_)
    current_mask[ACTION_NAMES.index("check_call")] = True
    next_mask = np.zeros(len(ACTION_NAMES), dtype=np.bool_)
    next_mask[ACTION_NAMES.index("fold")] = True
    next_mask[ACTION_NAMES.index("check_call")] = True
    trainer.replay.add(
        Transition(
            features=np.zeros(width, dtype=np.float32),
            legal_mask=current_mask,
            action_index=ACTION_NAMES.index("check_call"),
            reward_bb=0.0,
            next_features=np.ones(width, dtype=np.float32),
            next_legal_mask=next_mask,
            terminal=False,
        )
    )

    (metrics,) = trainer.optimize(1)

    # gamma=.99, so target is .99*7=6.93 and selected online Q is 4.
    expected_td = 0.99 * 7.0 - 4.0
    expected_huber = abs(expected_td) - 0.5
    assert metrics.mean_absolute_td_error == pytest.approx(abs(expected_td), rel=1e-5)
    assert metrics.loss == pytest.approx(expected_huber, rel=1e-5)
    assert math.isfinite(metrics.gradient_norm)
    assert metrics.target_synchronized
    assert all(
        trainer.torch.equal(online, target)
        for online, target in zip(
            trainer.online.state_dict().values(),
            trainer.target.state_dict().values(),
            strict=True,
        )
    )


def test_replay_rejects_illegal_actions_and_invalid_terminal_masks() -> None:
    replay = ReplayBuffer(capacity=2, feature_count=3)
    legal = np.asarray([True, False, False, False, False, False, False])
    transition = Transition(
        features=np.zeros(3, dtype=np.float32),
        legal_mask=legal,
        action_index=1,
        reward_bb=0.0,
        next_features=np.zeros(3, dtype=np.float32),
        next_legal_mask=np.zeros(len(ACTION_NAMES), dtype=np.bool_),
        terminal=True,
    )

    with pytest.raises(ValueError, match="action is illegal"):
        replay.add(transition)

    legal_action = Transition(
        features=transition.features,
        legal_mask=legal,
        action_index=0,
        reward_bb=0.0,
        next_features=transition.next_features,
        next_legal_mask=legal,
        terminal=True,
    )
    with pytest.raises(ValueError, match="terminal transition"):
        replay.add(legal_action)


def test_trusted_resume_restores_replay_rngs_and_safe_numpy_export(tmp_path) -> None:
    trainer = _trainer(master_seed=101, updates_per_transition=1)
    trainer.train_for(1, _opponent_factory)
    trainer_path = tmp_path / "trainer.pt"
    inference_path = tmp_path / "policy.npz"

    trainer_digest = trainer.save_training_checkpoint(trainer_path)
    inference_digest = trainer.export_inference_checkpoint(inference_path)
    assert len(trainer_digest) == len(inference_digest) == 64
    with pytest.raises(ValueError, match="trusted=True"):
        DoubleDQNTrainer.from_training_checkpoint(trainer_path)

    resumed = DoubleDQNTrainer.from_training_checkpoint(
        trainer_path,
        trusted=True,
        device="cpu",
    )
    assert resumed.environment_decisions == trainer.environment_decisions
    assert resumed.training_hands == trainer.training_hands
    assert len(resumed.replay) == len(trainer.replay)
    for index in range(len(trainer.replay)):
        expected = trainer.replay.get(index)
        actual = resumed.replay.get(index)
        assert np.array_equal(actual.features, expected.features)
        assert np.array_equal(actual.next_features, expected.next_features)
        assert np.array_equal(actual.legal_mask, expected.legal_mask)
        assert actual.action_index == expected.action_index
        assert actual.reward_bb == expected.reward_bb
        assert actual.terminal == expected.terminal

    # RNG state and counters resume exactly: the next hand is semantically equal.
    trainer.train_mode()
    resumed.train_mode()
    original_next = trainer.play(_opponent_factory)
    resumed_next = resumed.play(_opponent_factory)
    assert resumed_next.deal_seed == original_next.deal_seed
    assert resumed_next.hero_seat == original_next.hero_seat
    assert resumed_next.hero_payoff_bb == original_next.hero_payoff_bb
    assert resumed_next.hand_result.deck_hash == original_next.hand_result.deck_hash
    assert [record.decision.action for record in resumed_next.hand_result.decisions] == [
        record.decision.action for record in original_next.hand_result.decisions
    ]
    assert all(
        trainer.torch.equal(original, restored)
        for original, restored in zip(
            trainer.online.state_dict().values(),
            resumed.online.state_dict().values(),
            strict=True,
        )
    )

    frozen = NumpyMLPPolicy.from_checkpoint(
        inference_path,
        expected_sha256=inference_digest,
    )
    assert frozen.name == trainer.config.policy_name
    assert frozen.decide(_observation()).action == "check_call"
