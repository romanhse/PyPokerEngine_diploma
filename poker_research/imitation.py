"""Auditable behavior cloning for public heads-up hold'em observations.

Data collection is deliberately framework-free: a recording wrapper sees the
same :class:`~poker_research.types.Observation` as its teacher and stores no
engine state.  PyTorch is imported only by :func:`train_behavior_cloning`; the
result is exported through the safe NumPy checkpoint format used at evaluation
time.
"""

from __future__ import annotations

import importlib
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from poker_research.arena import ArenaConfig, play_hand
from poker_research.equity import stable_seed
from poker_research.neural import (
    ACTION_NAMES,
    MLPWeights,
    ObservationEncoder,
    save_checkpoint,
)
from poker_research.types import Decision, Observation, Policy

# Factories receive a policy-local seed and the public seat.  In particular, the
# first argument is never the deal seed: exposing it would let a policy replay
# PokerKit's shuffle and reconstruct hidden cards.
PolicyFactory: TypeAlias = Callable[[int, int], Policy]
TrainingDevice: TypeAlias = Literal["auto", "cpu", "mps"]


@dataclass(frozen=True, slots=True)
class Demonstration:
    """One teacher action and its exact public, encoded information set."""

    deal_seed: int
    hand_id: str
    seat: int
    decision_index: int
    teacher_name: str
    observation: Observation
    features: NDArray[np.float32]
    legal_mask: NDArray[np.bool_]
    action_index: int

    def __post_init__(self) -> None:
        if self.seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if self.observation.seat != self.seat:
            raise ValueError("demonstration seat does not match its observation")
        if self.observation.hand_id != self.hand_id:
            raise ValueError("demonstration hand_id does not match its observation")
        if self.decision_index < 0:
            raise ValueError("decision_index must be non-negative")
        if not self.teacher_name:
            raise ValueError("teacher_name must be non-empty")
        if self.features.ndim != 1 or not np.isfinite(self.features).all():
            raise ValueError("features must be a finite one-dimensional vector")
        if self.legal_mask.shape != (len(ACTION_NAMES),):
            raise ValueError("legal_mask has the wrong action dimension")
        if not self.legal_mask.any():
            raise ValueError("legal_mask must contain at least one legal action")
        if not 0 <= self.action_index < len(ACTION_NAMES):
            raise ValueError("action_index is outside the action vocabulary")
        if not self.legal_mask[self.action_index]:
            raise ValueError("teacher action is illegal under the recorded mask")

        # A frozen dataclass does not make NumPy buffers immutable.  Copying here
        # prevents a later caller from silently changing a collected dataset.
        features = np.asarray(self.features, dtype=np.float32).copy()
        legal_mask = np.asarray(self.legal_mask, dtype=np.bool_).copy()
        features.setflags(write=False)
        legal_mask.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "legal_mask", legal_mask)

    @property
    def action_name(self) -> str:
        """Teacher label in the shared discrete action vocabulary."""

        return ACTION_NAMES[self.action_index]


@dataclass(slots=True)
class RecordingTeacherPolicy:
    """Transparent policy wrapper that records only public teacher inputs."""

    teacher: Policy
    encoder: ObservationEncoder
    deal_seed: int
    name: str = field(init=False)
    _demonstrations: list[Demonstration] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.name = self.teacher.name

    @property
    def demonstrations(self) -> tuple[Demonstration, ...]:
        """Immutable view of the actions recorded so far."""

        return tuple(self._demonstrations)

    def decide(self, observation: Observation) -> Decision:
        """Delegate the public observation and record a validated hard label."""

        decision = self.teacher.decide(observation)
        _validate_teacher_decision(observation, decision)
        encoded = self.encoder.encode(observation)
        action_index = ACTION_NAMES.index(decision.action)
        if not encoded.legal_mask[action_index]:
            raise RuntimeError("encoder and arena disagree about teacher action legality")
        self._demonstrations.append(
            Demonstration(
                deal_seed=self.deal_seed,
                hand_id=observation.hand_id,
                seat=observation.seat,
                decision_index=len(self._demonstrations),
                teacher_name=self.teacher.name,
                observation=observation,
                features=encoded.features,
                legal_mask=encoded.legal_mask,
                action_index=action_index,
            )
        )
        return decision


def collect_demonstrations(
    teacher_factory: PolicyFactory,
    opponent_factory: PolicyFactory,
    deal_seeds: Sequence[int],
    *,
    encoder: ObservationEncoder | None = None,
    arena_config: ArenaConfig | None = None,
    teacher_seats: Sequence[int] = (0, 1),
    hand_id_prefix: str = "bc",
    policy_seed: int = 0,
) -> tuple[Demonstration, ...]:
    """Play fresh teacher/opponent instances and collect teacher decisions.

    Factories receive ``(policy_local_seed, seat)``.  Policy-local seeds are
    deterministically derived from ``policy_seed`` and collection indices, never
    from the deck seed.  Each teacher seat is played as a separate hand, while
    records retain the deal seed outside model features solely for grouped splitting.
    """

    seeds = tuple(deal_seeds)
    seats = tuple(teacher_seats)
    if not seeds:
        raise ValueError("deal_seeds must be non-empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("deal_seeds must be unique")
    if not seats or any(seat not in (0, 1) for seat in seats):
        raise ValueError("teacher_seats must contain seat 0, seat 1, or both")
    if len(set(seats)) != len(seats):
        raise ValueError("teacher_seats must not contain duplicates")
    if not hand_id_prefix or any(character.isspace() for character in hand_id_prefix):
        raise ValueError("hand_id_prefix must be non-empty and contain no whitespace")

    rules = arena_config or ArenaConfig()
    rules.validate()
    public_encoder = encoder or ObservationEncoder()
    _validate_encoder_for_arena(public_encoder, rules)

    collected: list[Demonstration] = []
    for deal_index, deal_seed in enumerate(seeds):
        for teacher_seat in seats:
            opponent_seat = 1 - teacher_seat
            teacher_local_seed = stable_seed(
                policy_seed,
                "behavior_cloning_teacher",
                deal_index,
                teacher_seat,
            )
            opponent_local_seed = stable_seed(
                policy_seed,
                "behavior_cloning_opponent",
                deal_index,
                opponent_seat,
            )
            teacher = teacher_factory(teacher_local_seed, teacher_seat)
            opponent = opponent_factory(opponent_local_seed, opponent_seat)
            recorder = RecordingTeacherPolicy(teacher, public_encoder, deal_seed)
            policies: tuple[Policy, Policy] = (
                (recorder, opponent)
                if teacher_seat == 0
                else (opponent, recorder)
            )
            play_hand(
                policies,
                deal_seed=deal_seed,
                hand_id=(
                    f"{hand_id_prefix}-deal-{deal_index:06d}"
                    f"-teacher-seat-{teacher_seat}"
                ),
                config=rules,
                policy_rng_key=str(
                    stable_seed(
                        policy_seed,
                        "behavior_cloning_policy_schedule_v1",
                        deal_index,
                        teacher_seat,
                    )
                ),
            )
            collected.extend(recorder.demonstrations)
    if not collected:
        raise RuntimeError("teacher was never asked to act")
    return tuple(collected)


@dataclass(frozen=True, slots=True)
class DemonstrationSplit:
    """Train/validation partition grouped by deal seed."""

    train: tuple[Demonstration, ...]
    validation: tuple[Demonstration, ...]
    train_deal_seeds: tuple[int, ...]
    validation_deal_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.validation:
            raise ValueError("both train and validation partitions must be non-empty")
        train_seeds = {sample.deal_seed for sample in self.train}
        validation_seeds = {sample.deal_seed for sample in self.validation}
        if train_seeds.intersection(validation_seeds):
            raise ValueError("a deal seed appears in both train and validation")
        if train_seeds != set(self.train_deal_seeds):
            raise ValueError("train_deal_seeds do not describe the train partition")
        if validation_seeds != set(self.validation_deal_seeds):
            raise ValueError(
                "validation_deal_seeds do not describe the validation partition"
            )


def split_demonstrations(
    demonstrations: Sequence[Demonstration],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> DemonstrationSplit:
    """Deterministically split whole deals, never individual decisions."""

    samples = tuple(demonstrations)
    if not samples:
        raise ValueError("demonstrations must be non-empty")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and in (0, 1)")
    deal_seeds = sorted({sample.deal_seed for sample in samples})
    if len(deal_seeds) < 2:
        raise ValueError("at least two distinct deal seeds are required for a split")

    random.Random(seed).shuffle(deal_seeds)
    validation_count = min(
        len(deal_seeds) - 1,
        max(1, math.ceil(len(deal_seeds) * validation_fraction)),
    )
    validation_seed_set = set(deal_seeds[:validation_count])
    train = tuple(
        sample for sample in samples if sample.deal_seed not in validation_seed_set
    )
    validation = tuple(
        sample for sample in samples if sample.deal_seed in validation_seed_set
    )
    return DemonstrationSplit(
        train=train,
        validation=validation,
        train_deal_seeds=tuple(sorted({sample.deal_seed for sample in train})),
        validation_deal_seeds=tuple(
            sorted({sample.deal_seed for sample in validation})
        ),
    )


@dataclass(frozen=True, slots=True)
class BehaviorCloningConfig:
    """Small deterministic MLP training configuration."""

    hidden_size: int = 64
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    split_seed: int = 0
    validation_fraction: float = 0.2
    device: TrainingDevice = "auto"
    policy_name: str = "behavior_cloning_v1"

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("hidden_size, epochs and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            not math.isfinite(self.validation_fraction)
            or not 0 < self.validation_fraction < 1
        ):
            raise ValueError("validation_fraction must be finite and in (0, 1)")
        if self.device not in ("auto", "cpu", "mps"):
            raise ValueError("device must be 'auto', 'cpu' or 'mps'")
        if not self.policy_name or any(
            character.isspace() for character in self.policy_name
        ):
            raise ValueError("policy_name must be non-empty and contain no whitespace")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Masked classification metrics after one completed epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    train_illegal_predictions: int
    validation_loss: float
    validation_accuracy: float
    validation_illegal_predictions: int


@dataclass(frozen=True, slots=True)
class BehaviorCloningResult:
    """Trained NumPy weights, safe checkpoint provenance, and learning curve."""

    weights: MLPWeights
    split: DemonstrationSplit
    history: tuple[EpochMetrics, ...]
    device: str
    checkpoint_path: Path
    checkpoint_sha256: str

    @property
    def final_metrics(self) -> EpochMetrics:
        """Metrics for the exported final epoch."""

        return self.history[-1]


def train_behavior_cloning(
    demonstrations: Sequence[Demonstration],
    encoder: ObservationEncoder,
    checkpoint_path: Path,
    *,
    config: BehaviorCloningConfig | None = None,
) -> BehaviorCloningResult:
    """Fit masked cross-entropy and export an inference-only ``.npz`` file.

    PyTorch is an optional training dependency.  Install it with
    ``pip install -e '.[training]'``; collection, splitting, and NumPy inference
    continue to work without it.
    """

    settings = config or BehaviorCloningConfig()
    settings.validate()
    if checkpoint_path.suffix != ".npz":
        raise ValueError("behavior cloning checkpoints must use the .npz extension")
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint_path}")
    split = split_demonstrations(
        demonstrations,
        validation_fraction=settings.validation_fraction,
        seed=settings.split_seed,
    )
    train_arrays = _dataset_arrays(split.train, encoder)
    validation_arrays = _dataset_arrays(split.validation, encoder)

    torch = _load_torch()
    device = _resolve_device(torch, settings.device)
    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(settings.seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(encoder.feature_count, settings.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(settings.hidden_size, settings.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(settings.hidden_size, len(ACTION_NAMES)),
        )
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(settings.seed)
        history: list[EpochMetrics] = []
        for epoch in range(1, settings.epochs + 1):
            _train_epoch(
                torch,
                model,
                optimizer,
                train_arrays,
                device=device,
                batch_size=settings.batch_size,
                generator=shuffle_generator,
            )
            train_loss, train_accuracy, train_illegal = _evaluate(
                torch,
                model,
                train_arrays,
                device=device,
                batch_size=settings.batch_size,
            )
            validation_loss, validation_accuracy, validation_illegal = _evaluate(
                torch,
                model,
                validation_arrays,
                device=device,
                batch_size=settings.batch_size,
            )
            if train_illegal or validation_illegal:
                raise RuntimeError("masked classifier emitted an illegal prediction")
            history.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    train_illegal_predictions=train_illegal,
                    validation_loss=validation_loss,
                    validation_accuracy=validation_accuracy,
                    validation_illegal_predictions=validation_illegal,
                )
            )
        weights = _export_weights(model, encoder.feature_count)
    finally:
        torch.random.set_rng_state(cpu_rng_state)

    checkpoint_sha256 = save_checkpoint(
        checkpoint_path,
        weights,
        encoder,
        policy_name=settings.policy_name,
    )
    return BehaviorCloningResult(
        weights=weights,
        split=split,
        history=tuple(history),
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )


DatasetArrays: TypeAlias = tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.int64],
]


def _dataset_arrays(
    demonstrations: Sequence[Demonstration],
    encoder: ObservationEncoder,
) -> DatasetArrays:
    if not demonstrations:
        raise ValueError("dataset partition must be non-empty")
    for sample in demonstrations:
        if sample.features.shape != (encoder.feature_count,):
            raise ValueError("demonstration feature width does not match encoder")
        expected = encoder.encode(sample.observation)
        if not np.array_equal(sample.features, expected.features):
            raise ValueError("demonstration features do not match its public observation")
        if not np.array_equal(sample.legal_mask, expected.legal_mask):
            raise ValueError("demonstration mask does not match its public observation")
    features = np.stack([sample.features for sample in demonstrations]).astype(
        np.float32,
        copy=False,
    )
    legal_masks = np.stack([sample.legal_mask for sample in demonstrations]).astype(
        np.bool_,
        copy=False,
    )
    labels = np.asarray(
        [sample.action_index for sample in demonstrations],
        dtype=np.int64,
    )
    if not legal_masks[np.arange(len(labels)), labels].all():
        raise ValueError("dataset contains an illegal teacher label")
    return features, legal_masks, labels


def _train_epoch(
    torch: Any,
    model: Any,
    optimizer: Any,
    arrays: DatasetArrays,
    *,
    device: str,
    batch_size: int,
    generator: Any,
) -> None:
    model.train()
    sample_count = arrays[0].shape[0]
    order = torch.randperm(sample_count, generator=generator).tolist()
    for start in range(0, sample_count, batch_size):
        indices = order[start : start + batch_size]
        features, masks, labels = _torch_batch(torch, arrays, indices, device)
        optimizer.zero_grad(set_to_none=True)
        masked_logits = _masked_logits(torch, model(features), masks)
        loss = torch.nn.functional.cross_entropy(masked_logits, labels)
        loss.backward()
        optimizer.step()


def _evaluate(
    torch: Any,
    model: Any,
    arrays: DatasetArrays,
    *,
    device: str,
    batch_size: int,
) -> tuple[float, float, int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_illegal = 0
    sample_count = arrays[0].shape[0]
    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            indices = list(range(start, min(start + batch_size, sample_count)))
            features, masks, labels = _torch_batch(torch, arrays, indices, device)
            masked_logits = _masked_logits(torch, model(features), masks)
            loss = torch.nn.functional.cross_entropy(masked_logits, labels)
            predictions = masked_logits.argmax(dim=1)
            current_count = len(indices)
            total_loss += float(loss.detach().cpu().item()) * current_count
            total_correct += int((predictions == labels).sum().detach().cpu().item())
            predicted_legal = masks.gather(1, predictions.unsqueeze(1)).squeeze(1)
            total_illegal += int((~predicted_legal).sum().detach().cpu().item())
    return (
        total_loss / sample_count,
        total_correct / sample_count,
        total_illegal,
    )


def _torch_batch(
    torch: Any,
    arrays: DatasetArrays,
    indices: list[int],
    device: str,
) -> tuple[Any, Any, Any]:
    features, masks, labels = arrays
    return (
        torch.from_numpy(features[indices]).to(device),
        torch.from_numpy(masks[indices]).to(device),
        torch.from_numpy(labels[indices]).to(device),
    )


def _masked_logits(torch: Any, logits: Any, legal_masks: Any) -> Any:
    if logits.shape != legal_masks.shape:
        raise RuntimeError("logits and legal masks have different shapes")
    return logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)


def _export_weights(model: Any, input_size: int) -> MLPWeights:
    first, second, output = model[0], model[2], model[4]

    def array(tensor: Any, *, transpose: bool = False) -> NDArray[np.float32]:
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
            "behavior cloning training requires PyTorch; "
            "install the project training extra with pip install -e '.[training]'"
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


def _validate_teacher_decision(
    observation: Observation,
    decision: Decision,
) -> None:
    legal_names = tuple(option.name for option in observation.legal_actions)
    if decision.action not in legal_names:
        raise ValueError(f"teacher selected illegal action {decision.action!r}")
    probabilities = dict(decision.probabilities)
    if len(probabilities) != len(decision.probabilities):
        raise ValueError("teacher decision contains duplicate probabilities")
    if set(probabilities) != set(legal_names):
        raise ValueError("teacher probabilities must cover every legal action")
    if any(
        not math.isfinite(probability) or not 0 <= probability <= 1
        for probability in probabilities.values()
    ):
        raise ValueError("teacher probabilities must be finite and in [0, 1]")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-9):
        raise ValueError("teacher probabilities must sum to one")
    if probabilities[decision.action] <= 0:
        raise ValueError("teacher selected an action with zero probability")
