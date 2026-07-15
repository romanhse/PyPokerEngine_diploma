"""Research-ready Double DQN baseline for the authoritative PokerKit arena.

The module deliberately separates three concerns:

* :class:`EpisodeDQNPolicy` collects one transition for every hero decision while
  seeing only the public :class:`~poker_research.types.Observation` boundary;
* :class:`ReplayBuffer` persists multi-step experience across hands;
* :class:`DoubleDQNTrainer` owns the optional PyTorch training state and exports
  a non-executable NumPy checkpoint for evaluation.

Trainer checkpoints are PyTorch pickle files and therefore must only be resumed
from trusted local paths.  Confirmatory inference uses the safe ``.npz`` format
from :mod:`poker_research.neural` and never needs PyTorch.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from poker_research.arena import ArenaConfig, HandResult, play_hand
from poker_research.equity import stable_seed
from poker_research.neural import (
    ACTION_NAMES,
    ENCODER_VERSION,
    EncodedObservation,
    EncoderConfig,
    MLPWeights,
    ObservationEncoder,
    save_checkpoint,
)
from poker_research.types import Decision, MetadataValue, Observation, Policy

FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int64]
TrainingDevice: TypeAlias = Literal["auto", "cpu", "mps"]
TrainerMode: TypeAlias = Literal["train", "eval"]
OpponentFactory: TypeAlias = Callable[[int, int], Policy]

TRAINER_CHECKPOINT_VERSION = "double_dqn_trainer_v1"


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Frozen algorithm, optimization, exploration, and RNG settings."""

    hidden_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    replay_capacity: int = 100_000
    batch_size: int = 128
    min_replay_size: int = 1_000
    updates_per_transition: int = 1
    target_sync_interval: int = 1_000
    gradient_clip_norm: float = 10.0
    huber_delta: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_decisions: int = 50_000
    master_seed: int = 0
    policy_schedule_seed: int = 1
    device: TrainingDevice = "auto"
    policy_name: str = "double_dqn_v1"

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.replay_capacity <= 0 or self.batch_size <= 0:
            raise ValueError("replay_capacity and batch_size must be positive")
        if not 1 <= self.min_replay_size <= self.replay_capacity:
            raise ValueError("min_replay_size must be in [1, replay_capacity]")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must not exceed replay_capacity")
        if self.updates_per_transition < 0:
            raise ValueError("updates_per_transition must be non-negative")
        if self.target_sync_interval <= 0:
            raise ValueError("target_sync_interval must be positive")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if self.epsilon_decay_decisions <= 0:
            raise ValueError("epsilon_decay_decisions must be positive")
        if self.master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        if self.policy_schedule_seed < 0:
            raise ValueError("policy_schedule_seed must be non-negative")
        if self.device not in ("auto", "cpu", "mps"):
            raise ValueError("device must be 'auto', 'cpu' or 'mps'")
        if not self.policy_name or any(
            character.isspace() for character in self.policy_name
        ):
            raise ValueError("policy_name must be non-empty and contain no whitespace")


@dataclass(frozen=True, slots=True)
class Transition:
    """One public information-state transition in sparse terminal-reward form."""

    features: FloatArray
    legal_mask: BoolArray
    action_index: int
    reward_bb: float
    next_features: FloatArray
    next_legal_mask: BoolArray
    terminal: bool


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """Uniform replay sample; weights remain explicit for future prioritized replay."""

    features: FloatArray
    legal_masks: BoolArray
    action_indices: IntArray
    rewards_bb: FloatArray
    next_features: FloatArray
    next_legal_masks: BoolArray
    terminals: BoolArray
    weights: FloatArray


class ReplayBuffer:
    """Fixed-capacity ring buffer with an exact resumable array representation."""

    def __init__(self, capacity: int, feature_count: int) -> None:
        if capacity <= 0 or feature_count <= 0:
            raise ValueError("capacity and feature_count must be positive")
        self.capacity = capacity
        self.feature_count = feature_count
        self._features = np.zeros((capacity, feature_count), dtype=np.float32)
        self._legal_masks = np.zeros((capacity, len(ACTION_NAMES)), dtype=np.bool_)
        self._action_indices = np.zeros(capacity, dtype=np.int64)
        self._rewards_bb = np.zeros(capacity, dtype=np.float32)
        self._next_features = np.zeros((capacity, feature_count), dtype=np.float32)
        self._next_legal_masks = np.zeros(
            (capacity, len(ACTION_NAMES)), dtype=np.bool_
        )
        self._terminals = np.zeros(capacity, dtype=np.bool_)
        self._size = 0
        self._position = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        """Copy a validated transition into the current ring position."""

        self._validate_transition(transition)
        index = self._position
        self._features[index] = transition.features
        self._legal_masks[index] = transition.legal_mask
        self._action_indices[index] = transition.action_index
        self._rewards_bb[index] = transition.reward_bb
        self._next_features[index] = transition.next_features
        self._next_legal_masks[index] = transition.next_legal_mask
        self._terminals[index] = transition.terminal
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def get(self, index: int) -> Transition:
        """Return a defensive copy of an active slot for audit/tests."""

        if not 0 <= index < self._size:
            raise IndexError("replay index outside active range")
        return Transition(
            features=self._features[index].copy(),
            legal_mask=self._legal_masks[index].copy(),
            action_index=int(self._action_indices[index]),
            reward_bb=float(self._rewards_bb[index]),
            next_features=self._next_features[index].copy(),
            next_legal_mask=self._next_legal_masks[index].copy(),
            terminal=bool(self._terminals[index]),
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        """Sample uniformly without replacement and expose unit importance weights."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError("cannot sample more transitions than replay contains")
        indices = rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(
            features=self._features[indices].copy(),
            legal_masks=self._legal_masks[indices].copy(),
            action_indices=self._action_indices[indices].copy(),
            rewards_bb=self._rewards_bb[indices].copy(),
            next_features=self._next_features[indices].copy(),
            next_legal_masks=self._next_legal_masks[indices].copy(),
            terminals=self._terminals[indices].copy(),
            weights=np.ones(batch_size, dtype=np.float32),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return exact active ring state for a trusted trainer checkpoint."""

        active = self._size
        return {
            "capacity": self.capacity,
            "feature_count": self.feature_count,
            "size": self._size,
            "position": self._position,
            "features": self._features[:active].copy(),
            "legal_masks": self._legal_masks[:active].copy(),
            "action_indices": self._action_indices[:active].copy(),
            "rewards_bb": self._rewards_bb[:active].copy(),
            "next_features": self._next_features[:active].copy(),
            "next_legal_masks": self._next_legal_masks[:active].copy(),
            "terminals": self._terminals[:active].copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a state emitted by :meth:`state_dict` after strict validation."""

        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay checkpoint capacity does not match config")
        if int(state["feature_count"]) != self.feature_count:
            raise ValueError("replay checkpoint feature width does not match encoder")
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError("replay checkpoint has invalid size")
        if not 0 <= position < self.capacity:
            raise ValueError("replay checkpoint has invalid ring position")
        if size < self.capacity and position != size:
            raise ValueError("partially filled replay must point immediately after its data")

        arrays = {
            "features": np.asarray(state["features"]),
            "legal_masks": np.asarray(state["legal_masks"]),
            "action_indices": np.asarray(state["action_indices"]),
            "rewards_bb": np.asarray(state["rewards_bb"]),
            "next_features": np.asarray(state["next_features"]),
            "next_legal_masks": np.asarray(state["next_legal_masks"]),
            "terminals": np.asarray(state["terminals"]),
        }
        expected_shapes = {
            "features": (size, self.feature_count),
            "legal_masks": (size, len(ACTION_NAMES)),
            "action_indices": (size,),
            "rewards_bb": (size,),
            "next_features": (size, self.feature_count),
            "next_legal_masks": (size, len(ACTION_NAMES)),
            "terminals": (size,),
        }
        expected_dtypes = {
            "features": np.dtype(np.float32),
            "legal_masks": np.dtype(np.bool_),
            "action_indices": np.dtype(np.int64),
            "rewards_bb": np.dtype(np.float32),
            "next_features": np.dtype(np.float32),
            "next_legal_masks": np.dtype(np.bool_),
            "terminals": np.dtype(np.bool_),
        }
        for name, array in arrays.items():
            if array.shape != expected_shapes[name] or array.dtype != expected_dtypes[name]:
                raise ValueError(f"replay checkpoint contains invalid {name}")

        self._features.fill(0)
        self._legal_masks.fill(False)
        self._action_indices.fill(0)
        self._rewards_bb.fill(0)
        self._next_features.fill(0)
        self._next_legal_masks.fill(False)
        self._terminals.fill(False)
        if size:
            self._features[:size] = arrays["features"]
            self._legal_masks[:size] = arrays["legal_masks"]
            self._action_indices[:size] = arrays["action_indices"]
            self._rewards_bb[:size] = arrays["rewards_bb"]
            self._next_features[:size] = arrays["next_features"]
            self._next_legal_masks[:size] = arrays["next_legal_masks"]
            self._terminals[:size] = arrays["terminals"]
        self._size = size
        self._position = position
        for index in range(size):
            self._validate_transition(self.get(index))

    def _validate_transition(self, transition: Transition) -> None:
        if transition.features.shape != (self.feature_count,):
            raise ValueError("transition features have the wrong shape")
        if transition.next_features.shape != (self.feature_count,):
            raise ValueError("transition next_features have the wrong shape")
        if transition.legal_mask.shape != (len(ACTION_NAMES),):
            raise ValueError("transition legal_mask has the wrong shape")
        if transition.next_legal_mask.shape != (len(ACTION_NAMES),):
            raise ValueError("transition next_legal_mask has the wrong shape")
        if transition.features.dtype != np.float32:
            raise ValueError("transition features must be float32")
        if transition.next_features.dtype != np.float32:
            raise ValueError("transition next_features must be float32")
        if transition.legal_mask.dtype != np.bool_:
            raise ValueError("transition legal_mask must be bool")
        if transition.next_legal_mask.dtype != np.bool_:
            raise ValueError("transition next_legal_mask must be bool")
        if not np.isfinite(transition.features).all() or not np.isfinite(
            transition.next_features
        ).all():
            raise ValueError("transition features must be finite")
        if not 0 <= transition.action_index < len(ACTION_NAMES):
            raise ValueError("transition action is outside the action vocabulary")
        if not transition.legal_mask[transition.action_index]:
            raise ValueError("transition action is illegal under its mask")
        if not transition.legal_mask.any():
            raise ValueError("transition state has no legal actions")
        if not math.isfinite(transition.reward_bb):
            raise ValueError("transition reward must be finite")
        if transition.terminal and transition.next_legal_mask.any():
            raise ValueError("terminal transition must not expose next legal actions")
        if not transition.terminal and not transition.next_legal_mask.any():
            raise ValueError("non-terminal transition requires next legal actions")


@dataclass(frozen=True, slots=True)
class OptimizationMetrics:
    """One completed gradient update."""

    step: int
    loss: float
    mean_absolute_td_error: float
    gradient_norm: float
    target_synchronized: bool


@dataclass(frozen=True, slots=True)
class DQNHandReport:
    """Auditable outcome of one training or held-out validation hand."""

    mode: TrainerMode
    deal_seed: int
    hero_seat: int
    hero_payoff_bb: float
    hero_decisions: int
    transitions_added: int
    optimization: tuple[OptimizationMetrics, ...]
    hand_result: HandResult


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    features: FloatArray
    legal_mask: BoolArray
    action_index: int


@dataclass(slots=True)
class EpisodeDQNPolicy:
    """Per-hand arena adapter that links consecutive hero observations."""

    trainer: DoubleDQNTrainer
    collect_experience: bool
    name: str = field(init=False)
    decisions: int = field(default=0, init=False)
    transitions_added: int = field(default=0, init=False)
    _pending: _PendingDecision | None = field(default=None, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = self.trainer.config.policy_name

    def decide(self, observation: Observation) -> Decision:
        if self._finished:
            raise RuntimeError("cannot reuse a finished DQN episode policy")
        encoded = self.trainer.encoder.encode(observation)
        if self.collect_experience and self._pending is not None:
            self._append_nonterminal(self._pending, encoded)
        selected, probabilities, epsilon = self.trainer._select_action(  # noqa: SLF001
            encoded,
            explore=self.collect_experience,
        )
        if self.collect_experience:
            self._pending = _PendingDecision(
                encoded.features.copy(),
                encoded.legal_mask.copy(),
                selected,
            )
        self.decisions += 1
        metadata: dict[str, MetadataValue] = {
            "algorithm": "double_dqn",
            "epsilon": round(epsilon, 8),
            "mode": "train" if self.collect_experience else "eval",
            "selected_action_index": selected,
        }
        return Decision(
            action=ACTION_NAMES[selected],
            probabilities=tuple(
                (option.name, float(probabilities[ACTION_NAMES.index(option.name)]))
                for option in observation.legal_actions
            ),
            metadata=tuple(sorted(metadata.items())),
        )

    def finish(self, payoff_bb: float) -> None:
        """Close the final transition with the hand's zero-sum payoff in BB."""

        if self._finished:
            raise RuntimeError("DQN episode policy was already finished")
        if not math.isfinite(payoff_bb):
            raise ValueError("terminal payoff must be finite")
        if self.collect_experience and self._pending is not None:
            terminal_features = np.zeros(
                self.trainer.encoder.feature_count, dtype=np.float32
            )
            terminal_mask = np.zeros(len(ACTION_NAMES), dtype=np.bool_)
            self.trainer.replay.add(
                Transition(
                    features=self._pending.features,
                    legal_mask=self._pending.legal_mask,
                    action_index=self._pending.action_index,
                    reward_bb=payoff_bb,
                    next_features=terminal_features,
                    next_legal_mask=terminal_mask,
                    terminal=True,
                )
            )
            self.transitions_added += 1
        self._pending = None
        self._finished = True

    def _append_nonterminal(
        self,
        pending: _PendingDecision,
        next_observation: EncodedObservation,
    ) -> None:
        self.trainer.replay.add(
            Transition(
                features=pending.features,
                legal_mask=pending.legal_mask,
                action_index=pending.action_index,
                reward_bb=0.0,
                next_features=next_observation.features,
                next_legal_mask=next_observation.legal_mask,
                terminal=False,
            )
        )
        self.transitions_added += 1


class DoubleDQNTrainer:
    """Persistent online/target Double DQN learner with isolated RNG streams."""

    def __init__(
        self,
        *,
        config: DQNConfig | None = None,
        encoder: ObservationEncoder | None = None,
        arena_config: ArenaConfig | None = None,
        initial_weights: MLPWeights | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.config.validate()
        self.encoder = encoder or ObservationEncoder()
        self.arena_config = arena_config or ArenaConfig()
        self.arena_config.validate()
        _validate_encoder_for_arena(self.encoder, self.arena_config)

        self.torch = _load_torch()
        self.device = _resolve_device(self.torch, self.config.device)
        streams = np.random.SeedSequence(self.config.master_seed).spawn(5)
        self._action_rng = np.random.default_rng(streams[0])
        self._replay_rng = np.random.default_rng(streams[1])
        self._train_deal_rng = np.random.default_rng(streams[2])
        self._validation_deal_rng = np.random.default_rng(streams[3])
        network_seed = int(streams[4].generate_state(1, dtype=np.uint32)[0])

        if initial_weights is None:
            network_weights = MLPWeights.random(
                self.encoder.feature_count,
                hidden_size=self.config.hidden_size,
                seed=network_seed,
            )
        else:
            initial_weights.validate(self.encoder.feature_count)
            if initial_weights.hidden_bias.shape != (self.config.hidden_size,):
                raise ValueError("initial weights hidden size does not match DQN config")
            network_weights = initial_weights
        self.online = _build_network(
            self.torch,
            self.encoder.feature_count,
            self.config.hidden_size,
        ).to(self.device)
        self.target = _build_network(
            self.torch,
            self.encoder.feature_count,
            self.config.hidden_size,
        ).to(self.device)
        _load_weights_into_model(self.torch, self.online, network_weights)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = self.torch.optim.AdamW(
            self.online.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.replay = ReplayBuffer(
            self.config.replay_capacity,
            self.encoder.feature_count,
        )
        self.environment_decisions = 0
        self.optimization_steps = 0
        self.training_hands = 0
        self.validation_hands = 0
        self._mode: TrainerMode = "train"
        self.train_mode()

    @property
    def mode(self) -> TrainerMode:
        return self._mode

    @property
    def epsilon(self) -> float:
        """Current linearly annealed exploration probability."""

        progress = min(
            self.environment_decisions / self.config.epsilon_decay_decisions,
            1.0,
        )
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def train_mode(self) -> None:
        """Enable collection/exploration for subsequent :meth:`play` calls."""

        self._mode = "train"
        self.online.train()
        self.target.eval()

    def eval_mode(self) -> None:
        """Enable frozen greedy held-out evaluation for subsequent calls."""

        self._mode = "eval"
        self.online.eval()
        self.target.eval()

    def play(self, opponent_factory: OpponentFactory) -> DQNHandReport:
        """Play one hand in the explicit current mode.

        Training and validation use disjoint deal-seed namespaces (even versus
        odd seeds) and independent RNG streams.  Hero seats alternate within
        each namespace.
        """

        training = self._mode == "train"
        hand_index = self.training_hands if training else self.validation_hands
        hero_seat = hand_index % 2
        deal_seed = self._next_deal_seed(training=training)
        label = "train" if training else "validation"
        hero = EpisodeDQNPolicy(self, collect_experience=training)
        opponent_seed = stable_seed(
            self.config.master_seed,
            "dqn_opponent_policy",
            label,
            hand_index,
            1 - hero_seat,
        )
        opponent = opponent_factory(opponent_seed, 1 - hero_seat)
        policies: tuple[Policy, Policy] = (
            (hero, opponent) if hero_seat == 0 else (opponent, hero)
        )
        hand_result = play_hand(
            policies,
            deal_seed=deal_seed,
            hand_id=f"dqn-{label}-{hand_index}-hero-seat-{hero_seat}",
            config=self.arena_config,
            policy_rng_key=str(
                stable_seed(
                    self.config.policy_schedule_seed,
                    "dqn_policy_schedule_v1",
                    label,
                    hand_index,
                    hero_seat,
                )
            ),
        )
        payoff_bb = hand_result.payoffs[hero_seat] / self.arena_config.big_blind
        hero.finish(payoff_bb)
        optimization: tuple[OptimizationMetrics, ...] = ()
        if training:
            self.training_hands += 1
            step_count = hero.transitions_added * self.config.updates_per_transition
            optimization = self.optimize(step_count)
        else:
            self.validation_hands += 1
        return DQNHandReport(
            mode=self._mode,
            deal_seed=deal_seed,
            hero_seat=hero_seat,
            hero_payoff_bb=payoff_bb,
            hero_decisions=hero.decisions,
            transitions_added=hero.transitions_added,
            optimization=optimization,
            hand_result=hand_result,
        )

    def train_for(
        self,
        hand_count: int,
        opponent_factory: OpponentFactory,
    ) -> tuple[DQNHandReport, ...]:
        """Collect and learn from ``hand_count`` alternating-seat hands."""

        if hand_count <= 0:
            raise ValueError("hand_count must be positive")
        self.train_mode()
        return tuple(self.play(opponent_factory) for _ in range(hand_count))

    def validate_for(
        self,
        hand_count: int,
        opponent_factory: OpponentFactory,
    ) -> tuple[DQNHandReport, ...]:
        """Run greedy held-out hands without replay or optimizer mutation."""

        if hand_count <= 0:
            raise ValueError("hand_count must be positive")
        self.eval_mode()
        replay_size = len(self.replay)
        optimizer_steps = self.optimization_steps
        reports = tuple(self.play(opponent_factory) for _ in range(hand_count))
        if len(self.replay) != replay_size or self.optimization_steps != optimizer_steps:
            raise RuntimeError("validation mutated training state")
        return reports

    def optimize(self, step_count: int = 1) -> tuple[OptimizationMetrics, ...]:
        """Run Double-DQN updates when replay has reached its warm-up threshold."""

        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        required = max(self.config.batch_size, self.config.min_replay_size)
        if len(self.replay) < required:
            return ()
        metrics: list[OptimizationMetrics] = []
        for _ in range(step_count):
            batch = self.replay.sample(self.config.batch_size, self._replay_rng)
            metrics.append(self._optimize_batch(batch))
        return tuple(metrics)

    def synchronize_target(self) -> None:
        """Explicitly copy the online Q-network into the frozen target network."""

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def export_inference_checkpoint(self, path: Path) -> str:
        """Export safe NumPy MLP weights and return their SHA-256 digest."""

        was_training = bool(self.online.training)
        self.online.eval()
        try:
            weights = _export_weights(self.online, self.encoder.feature_count)
            return save_checkpoint(
                path,
                weights,
                self.encoder,
                policy_name=self.config.policy_name,
            )
        finally:
            self.online.train(was_training)

    def save_training_checkpoint(self, path: Path) -> str:
        """Persist exact resumable state to a trusted-local PyTorch checkpoint."""

        if path.suffix not in (".pt", ".pth"):
            raise ValueError("trainer checkpoints must use .pt or .pth")
        if path.exists():
            raise FileExistsError(f"refusing to overwrite trainer checkpoint: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_version": TRAINER_CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "arena_config": asdict(self.arena_config),
            "encoder": self.encoder.to_dict(),
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "replay_state_dict": self.replay.state_dict(),
            "rng_states": {
                "action": copy.deepcopy(self._action_rng.bit_generator.state),
                "replay": copy.deepcopy(self._replay_rng.bit_generator.state),
                "train_deal": copy.deepcopy(self._train_deal_rng.bit_generator.state),
                "validation_deal": copy.deepcopy(
                    self._validation_deal_rng.bit_generator.state
                ),
            },
            "counters": {
                "environment_decisions": self.environment_decisions,
                "optimization_steps": self.optimization_steps,
                "training_hands": self.training_hands,
                "validation_hands": self.validation_hands,
            },
            "mode": self._mode,
        }
        self.torch.save(payload, path)
        return _sha256_file(path)

    @classmethod
    def from_training_checkpoint(
        cls,
        path: Path,
        *,
        trusted: bool = False,
        device: TrainingDevice | None = None,
    ) -> DoubleDQNTrainer:
        """Resume a trainer, refusing pickle deserialization without opt-in trust."""

        if not trusted:
            raise ValueError(
                "PyTorch trainer checkpoints can execute pickle payloads; "
                "pass trusted=True only for a trusted local checkpoint"
            )
        torch = _load_torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("trainer checkpoint root must be a dictionary")
        if payload.get("checkpoint_version") != TRAINER_CHECKPOINT_VERSION:
            raise ValueError("unsupported trainer checkpoint version")
        raw_config = _require_mapping(payload, "config")
        config = DQNConfig(**dict(raw_config))
        if device is not None:
            config = replace(config, device=device)
        arena_config = ArenaConfig(**dict(_require_mapping(payload, "arena_config")))
        encoder_payload = _require_mapping(payload, "encoder")
        if encoder_payload.get("version") != ENCODER_VERSION:
            raise ValueError("trainer checkpoint encoder version is incompatible")
        encoder_config = EncoderConfig(**dict(_require_mapping(encoder_payload, "config")))
        encoder = ObservationEncoder(encoder_config)
        feature_names = tuple(cast(Sequence[str], encoder_payload["feature_names"]))
        if feature_names != encoder.feature_names:
            raise ValueError("trainer checkpoint encoder schema does not match")

        trainer = cls(config=config, encoder=encoder, arena_config=arena_config)
        trainer.online.load_state_dict(payload["online_state_dict"])
        trainer.target.load_state_dict(payload["target_state_dict"])
        trainer.target.eval()
        for parameter in trainer.target.parameters():
            parameter.requires_grad_(False)
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(trainer.torch, trainer.optimizer, trainer.device)
        replay_state = _require_mapping(payload, "replay_state_dict")
        trainer.replay.load_state_dict(replay_state)

        rng_states = _require_mapping(payload, "rng_states")
        trainer._action_rng.bit_generator.state = copy.deepcopy(rng_states["action"])
        trainer._replay_rng.bit_generator.state = copy.deepcopy(rng_states["replay"])
        trainer._train_deal_rng.bit_generator.state = copy.deepcopy(
            rng_states["train_deal"]
        )
        trainer._validation_deal_rng.bit_generator.state = copy.deepcopy(
            rng_states["validation_deal"]
        )
        counters = _require_mapping(payload, "counters")
        trainer.environment_decisions = int(counters["environment_decisions"])
        trainer.optimization_steps = int(counters["optimization_steps"])
        trainer.training_hands = int(counters["training_hands"])
        trainer.validation_hands = int(counters["validation_hands"])
        if min(
            trainer.environment_decisions,
            trainer.optimization_steps,
            trainer.training_hands,
            trainer.validation_hands,
        ) < 0:
            raise ValueError("trainer checkpoint contains negative counters")
        mode = payload.get("mode")
        if mode == "train":
            trainer.train_mode()
        elif mode == "eval":
            trainer.eval_mode()
        else:
            raise ValueError("trainer checkpoint contains invalid mode")
        return trainer

    def _next_deal_seed(self, *, training: bool) -> int:
        rng = self._train_deal_rng if training else self._validation_deal_rng
        # The low bit is a permanent namespace marker, proving that train and
        # held-out validation deals cannot overlap even if the RNG draws match.
        raw = int(rng.integers(0, 2**62, dtype=np.int64))
        return 2 * raw + (0 if training else 1)

    def _select_action(
        self,
        encoded: EncodedObservation,
        *,
        explore: bool,
    ) -> tuple[int, NDArray[np.float64], float]:
        legal_indices = np.flatnonzero(encoded.legal_mask)
        if legal_indices.size == 0:
            raise ValueError("DQN received an observation with no legal actions")
        q_values = self._q_values(encoded.features)
        greedy = int(legal_indices[np.argmax(q_values[legal_indices])])
        epsilon = self.epsilon if explore else 0.0
        probabilities = np.zeros(len(ACTION_NAMES), dtype=np.float64)
        probabilities[legal_indices] = epsilon / legal_indices.size
        probabilities[greedy] += 1.0 - epsilon
        if explore:
            selected = _sample_index(probabilities, float(self._action_rng.random()))
            self.environment_decisions += 1
        else:
            selected = greedy
        if not encoded.legal_mask[selected]:
            raise RuntimeError("masked DQN selected an illegal action")
        return selected, probabilities, epsilon

    def _q_values(self, features: FloatArray) -> NDArray[np.float32]:
        was_training = bool(self.online.training)
        self.online.eval()
        try:
            with self.torch.no_grad():
                tensor = self.torch.from_numpy(features).to(self.device).unsqueeze(0)
                values = self.online(tensor).squeeze(0).detach().cpu().numpy()
            return np.asarray(values, dtype=np.float32)
        finally:
            self.online.train(was_training)

    def _optimize_batch(self, batch: ReplayBatch) -> OptimizationMetrics:
        torch = self.torch
        self.online.train()
        self.target.eval()
        features = torch.from_numpy(batch.features).to(self.device)
        actions = torch.from_numpy(batch.action_indices).to(self.device)
        rewards = torch.from_numpy(batch.rewards_bb).to(self.device)
        next_features = torch.from_numpy(batch.next_features).to(self.device)
        next_masks = torch.from_numpy(batch.next_legal_masks).to(self.device)
        terminals = torch.from_numpy(batch.terminals).to(self.device)
        weights = torch.from_numpy(batch.weights).to(self.device)

        selected_q = self.online(features).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online chooses the legal next action, target evaluates it.
            online_next_q = self.online(next_features)
            masked_online_next_q = online_next_q.masked_fill(
                ~next_masks,
                torch.finfo(online_next_q.dtype).min,
            )
            next_actions = masked_online_next_q.argmax(dim=1)
            target_next_q = self.target(next_features).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards + self.config.gamma * (~terminals).to(rewards.dtype) * (
                target_next_q
            )

        td_errors = targets - selected_q
        per_sample_loss = torch.nn.functional.smooth_l1_loss(
            selected_q,
            targets,
            reduction="none",
            beta=self.config.huber_delta,
        )
        if per_sample_loss.shape != weights.shape:
            raise RuntimeError("Huber loss and replay weights have incompatible shapes")
        loss = (per_sample_loss * weights).sum() / weights.sum()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        self.optimization_steps += 1
        synchronized = self.optimization_steps % self.config.target_sync_interval == 0
        if synchronized:
            self.synchronize_target()
        return OptimizationMetrics(
            step=self.optimization_steps,
            loss=float(loss.detach().cpu().item()),
            mean_absolute_td_error=float(td_errors.abs().mean().detach().cpu().item()),
            gradient_norm=float(gradient_norm_tensor.detach().cpu().item()),
            target_synchronized=synchronized,
        )


def _build_network(torch: Any, input_size: int, hidden_size: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, len(ACTION_NAMES)),
    )


def _load_weights_into_model(torch: Any, model: Any, weights: MLPWeights) -> None:
    layers = (model[0], model[2], model[4])
    arrays = (
        (weights.input_to_hidden.T, weights.hidden_bias),
        (weights.hidden_to_hidden.T, weights.second_hidden_bias),
        (weights.hidden_to_action.T, weights.action_bias),
    )
    with torch.no_grad():
        for layer, (matrix, bias) in zip(layers, arrays, strict=True):
            layer.weight.copy_(torch.from_numpy(np.array(matrix, copy=True, order="C")))
            layer.bias.copy_(torch.from_numpy(np.array(bias, copy=True, order="C")))


def _export_weights(model: Any, input_size: int) -> MLPWeights:
    first, second, output = model[0], model[2], model[4]

    def array(tensor: Any, *, transpose: bool = False) -> FloatArray:
        value = tensor.detach().cpu().numpy()
        if transpose:
            value = value.T
        return np.asarray(value, dtype=np.float32).copy()

    weights = MLPWeights(
        input_to_hidden=array(first.weight, transpose=True),
        hidden_bias=array(first.bias),
        hidden_to_hidden=array(second.weight, transpose=True),
        second_hidden_bias=array(second.bias),
        hidden_to_action=array(output.weight, transpose=True),
        action_bias=array(output.bias),
    )
    weights.validate(input_size)
    return weights


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "Double DQN training requires PyTorch; install the training extra with "
            "pip install -e '.[training]'"
        ) from error


def _resolve_device(torch: Any, requested: TrainingDevice) -> str:
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available")
    if requested == "auto":
        return "mps" if mps_available else "cpu"
    return requested


def _validate_encoder_for_arena(
    encoder: ObservationEncoder,
    arena_config: ArenaConfig,
) -> None:
    if encoder.config.starting_stack != arena_config.starting_stack:
        raise ValueError("encoder starting_stack must match the arena")
    if encoder.config.big_blind != arena_config.big_blind:
        raise ValueError("encoder big_blind must match the arena")


def _sample_index(probabilities: NDArray[np.float64], draw: float) -> int:
    cumulative = 0.0
    last_positive = -1
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if probability > 0:
            last_positive = index
        if draw < cumulative:
            return index
    if last_positive < 0:
        raise RuntimeError("cannot sample an empty probability distribution")
    return last_positive


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"trainer checkpoint field {key!r} must be a mapping")
    return cast(Mapping[str, Any], value)


def _optimizer_to_device(torch: Any, optimizer: Any, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
