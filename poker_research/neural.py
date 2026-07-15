"""Safe NumPy inference for trainable HUNL policies.

Training frameworks are deliberately kept out of the evaluation boundary.  A
trainer may use PyTorch/MPS and export the small, non-executable ``.npz``
checkpoint defined here; confirmatory evaluation needs only NumPy and never
unpickles Python objects.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from poker_research.equity import equity_vs_uniform, stable_seed
from poker_research.types import ActionName, Decision, MetadataValue, Observation

ACTION_NAMES: tuple[ActionName, ...] = (
    "fold",
    "check_call",
    "raise_min",
    "raise_half_pot",
    "raise_pot",
    "raise_2pot",
    "raise_all_in",
)
RANKS = "23456789TJQKA"
SUITS = "cdhs"
STREETS = ("preflop", "flop", "turn", "river")
POSITIONS = ("big_blind", "button_sb")
ENCODER_VERSION = "public_hunl_v1"
CHECKPOINT_VERSION = "numpy_mlp_v1"
FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Frozen normalization and optional equity-feature settings."""

    starting_stack: int = 200
    big_blind: int = 2
    max_spr: float = 20.0
    history_cap: int = 20
    equity_samples: int = 64
    equity_seed: int = 0

    def validate(self) -> None:
        if self.starting_stack <= 0 or self.big_blind <= 0:
            raise ValueError("starting_stack and big_blind must be positive")
        if not math.isfinite(self.max_spr) or self.max_spr <= 0:
            raise ValueError("max_spr must be finite and positive")
        if self.history_cap <= 0:
            raise ValueError("history_cap must be positive")
        if self.equity_samples < 0:
            raise ValueError("equity_samples must be non-negative")


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    """Fixed-width public feature vector plus the authoritative legal mask."""

    features: FloatArray
    legal_mask: NDArray[np.bool_]


class ObservationEncoder:
    """Encode cards, betting context and public history without hidden state."""

    card_slots: ClassVar[int] = 7
    card_width: ClassVar[int] = 18  # present + 13 ranks + 4 suits

    def __init__(self, config: EncoderConfig | None = None) -> None:
        self.config = config or EncoderConfig()
        self.config.validate()
        self.feature_names = self._feature_names()

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def encode(self, observation: Observation) -> EncodedObservation:
        cards = (*observation.hole_cards, *observation.board_cards)
        if len(cards) > self.card_slots:
            raise ValueError("hold'em observation contains more than seven visible cards")
        features: list[float] = []
        for slot in range(self.card_slots):
            features.extend(self._encode_card(cards[slot] if slot < len(cards) else None))

        features.extend(float(observation.street == street) for street in STREETS)
        features.extend(float(observation.position == position) for position in POSITIONS)

        own = observation.seat
        opponent = 1 - own
        scale = float(self.config.starting_stack)
        features.extend(
            (
                observation.pot / scale,
                observation.call_amount / scale,
                observation.effective_stack / scale,
                min(observation.spr, self.config.max_spr) / self.config.max_spr,
                observation.stacks[own] / scale,
                observation.stacks[opponent] / scale,
                observation.street_bets[own] / scale,
                observation.street_bets[opponent] / scale,
                float(observation.call_amount > 0),
                min(len(observation.history), self.config.history_cap)
                / self.config.history_cap,
            )
        )
        features.extend(self._history_features(observation))

        legal_names = {option.name for option in observation.legal_actions}
        unknown = legal_names.difference(ACTION_NAMES)
        if unknown:
            raise ValueError(f"encoder received unknown actions: {sorted(unknown)}")
        legal_mask = np.asarray([name in legal_names for name in ACTION_NAMES], dtype=np.bool_)
        if not legal_mask.any():
            raise ValueError("observation must contain at least one legal action")
        features.extend(float(value) for value in legal_mask)

        if self.config.equity_samples:
            seed = stable_seed(
                self.config.equity_seed,
                observation.hole_cards,
                observation.board_cards,
                observation.street,
            )
            equity = equity_vs_uniform(
                observation.hole_cards,
                observation.board_cards,
                sample_count=self.config.equity_samples,
                seed=seed,
            )
        else:
            equity = 0.5
        features.append(equity)

        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (self.feature_count,):
            raise RuntimeError(
                f"encoder schema mismatch: got {vector.shape[0]}, expected {self.feature_count}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("encoded observation contains non-finite values")
        return EncodedObservation(vector, legal_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": ENCODER_VERSION,
            "config": asdict(self.config),
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
        }

    def _encode_card(self, card: str | None) -> list[float]:
        encoded = [0.0] * self.card_width
        if card is None:
            return encoded
        if len(card) != 2 or card[0] not in RANKS or card[1] not in SUITS:
            raise ValueError(f"invalid card: {card!r}")
        encoded[0] = 1.0
        encoded[1 + RANKS.index(card[0])] = 1.0
        encoded[1 + len(RANKS) + SUITS.index(card[1])] = 1.0
        return encoded

    def _history_features(self, observation: Observation) -> list[float]:
        own = observation.seat
        counts = [[0, 0, 0], [0, 0, 0]]
        for action in observation.history:
            relative_seat = 0 if action.seat == own else 1
            counts[relative_seat][_action_family(action.action)] += 1
        denominator = float(self.config.history_cap)
        flattened = [
            min(count, self.config.history_cap) / denominator
            for row in counts
            for count in row
        ]

        last_family = [0.0, 0.0, 0.0]
        last_actor = [0.0, 0.0]
        last_amount = 0.0
        if observation.history:
            last = observation.history[-1]
            last_family[_action_family(last.action)] = 1.0
            last_actor[0 if last.seat == own else 1] = 1.0
            last_amount = (last.amount or 0) / float(self.config.starting_stack)
        return [*flattened, *last_family, *last_actor, last_amount]

    def _feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for slot in range(self.card_slots):
            names.append(f"card_{slot}_present")
            names.extend(f"card_{slot}_rank_{rank}" for rank in RANKS)
            names.extend(f"card_{slot}_suit_{suit}" for suit in SUITS)
        names.extend(f"street_{street}" for street in STREETS)
        names.extend(f"position_{position}" for position in POSITIONS)
        names.extend(
            (
                "pot_stack_fraction",
                "call_stack_fraction",
                "effective_stack_fraction",
                "spr_clipped",
                "own_stack_fraction",
                "opponent_stack_fraction",
                "own_street_bet_fraction",
                "opponent_street_bet_fraction",
                "facing_bet",
                "history_length_clipped",
            )
        )
        names.extend(
            f"history_{seat}_{family}_count"
            for seat in ("self", "opponent")
            for family in ("fold", "check_call", "raise")
        )
        names.extend(
            (
                "last_action_fold",
                "last_action_check_call",
                "last_action_raise",
                "last_actor_self",
                "last_actor_opponent",
                "last_action_amount_stack_fraction",
            )
        )
        names.extend(f"legal_{action}" for action in ACTION_NAMES)
        names.append("equity_vs_uniform")
        return tuple(names)


def _action_family(action: ActionName) -> int:
    if action == "fold":
        return 0
    if action == "check_call":
        return 1
    if action.startswith("raise_"):
        return 2
    raise ValueError(f"unknown public action: {action}")


@dataclass(frozen=True, slots=True)
class MLPWeights:
    """Three affine layers exported from any compatible trainer."""

    input_to_hidden: FloatArray
    hidden_bias: FloatArray
    hidden_to_hidden: FloatArray
    second_hidden_bias: FloatArray
    hidden_to_action: FloatArray
    action_bias: FloatArray

    def __post_init__(self) -> None:
        # ``frozen=True`` protects attributes, not the NumPy buffers they point
        # to.  Own read-only copies make evaluation weights actually frozen.
        for name in (
            "input_to_hidden",
            "hidden_bias",
            "hidden_to_hidden",
            "second_hidden_bias",
            "hidden_to_action",
            "action_bias",
        ):
            value = np.asarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def random(
        cls,
        input_size: int,
        *,
        hidden_size: int = 128,
        seed: int = 0,
    ) -> MLPWeights:
        """Create deterministic Xavier weights for tests and trainer initialization."""

        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("network dimensions must be positive")
        rng = np.random.default_rng(seed)

        def xavier(rows: int, columns: int) -> FloatArray:
            bound = math.sqrt(6.0 / (rows + columns))
            return rng.uniform(-bound, bound, size=(rows, columns)).astype(np.float32)

        return cls(
            xavier(input_size, hidden_size),
            np.zeros(hidden_size, dtype=np.float32),
            xavier(hidden_size, hidden_size),
            np.zeros(hidden_size, dtype=np.float32),
            xavier(hidden_size, len(ACTION_NAMES)),
            np.zeros(len(ACTION_NAMES), dtype=np.float32),
        )

    def validate(self, input_size: int) -> None:
        hidden_size = self.input_to_hidden.shape[1] if self.input_to_hidden.ndim == 2 else -1
        expected = {
            "input_to_hidden": (input_size, hidden_size),
            "hidden_bias": (hidden_size,),
            "hidden_to_hidden": (hidden_size, hidden_size),
            "second_hidden_bias": (hidden_size,),
            "hidden_to_action": (hidden_size, len(ACTION_NAMES)),
            "action_bias": (len(ACTION_NAMES),),
        }
        for field, shape in expected.items():
            value = getattr(self, field)
            if value.shape != shape:
                raise ValueError(f"{field} has shape {value.shape}, expected {shape}")
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{field} must contain finite float32 values")

    def forward(self, features: FloatArray) -> FloatArray:
        hidden = np.maximum(features @ self.input_to_hidden + self.hidden_bias, 0.0)
        second_hidden = np.maximum(
            hidden @ self.hidden_to_hidden + self.second_hidden_bias,
            0.0,
        )
        logits = second_hidden @ self.hidden_to_action + self.action_bias
        return np.asarray(logits, dtype=np.float32)


class NumpyMLPPolicy:
    """Frozen neural policy with mandatory legal masking and auditable sampling."""

    def __init__(
        self,
        weights: MLPWeights,
        *,
        encoder: ObservationEncoder | None = None,
        name: str = "numpy_mlp_v1",
        seed: int = 0,
        temperature: float = 0.0,
        checkpoint_sha256: str | None = None,
    ) -> None:
        if not name or any(character.isspace() for character in name):
            raise ValueError("policy name must be non-empty and contain no whitespace")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        self.encoder = encoder or ObservationEncoder()
        weights.validate(self.encoder.feature_count)
        self.weights = weights
        self.name = name
        self.seed = seed
        self.temperature = temperature
        self.checkpoint_sha256 = checkpoint_sha256

    def decide(self, observation: Observation) -> Decision:
        encoded = self.encoder.encode(observation)
        logits = self.weights.forward(encoded.features).astype(np.float64)
        logits[~encoded.legal_mask] = -math.inf
        legal_indices = np.flatnonzero(encoded.legal_mask)
        if self.temperature == 0:
            selected_index = int(legal_indices[np.argmax(logits[legal_indices])])
            distribution = np.zeros(len(ACTION_NAMES), dtype=np.float64)
            distribution[selected_index] = 1.0
        else:
            legal_logits = logits[legal_indices] / self.temperature
            legal_logits -= legal_logits.max()
            legal_probabilities = np.exp(legal_logits)
            legal_probabilities /= legal_probabilities.sum()
            distribution = np.zeros(len(ACTION_NAMES), dtype=np.float64)
            distribution[legal_indices] = legal_probabilities
            sampling_seed = stable_seed(
                self.seed,
                observation.rng_key or observation.hand_id,
                observation.seat,
                observation.street,
                len(observation.history),
            )
            selected_index = _sample_index(distribution, random.Random(sampling_seed).random())

        selected_action = ACTION_NAMES[selected_index]
        probabilities = tuple(
            (option.name, float(distribution[ACTION_NAMES.index(option.name)]))
            for option in observation.legal_actions
        )
        legal_distribution = distribution[legal_indices]
        entropy = -float(
            np.sum(
                legal_distribution[legal_distribution > 0]
                * np.log(legal_distribution[legal_distribution > 0])
            )
        )
        metadata: dict[str, MetadataValue] = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "encoder_version": ENCODER_VERSION,
            "entropy": round(entropy, 6),
            "selected_logit": round(float(logits[selected_index]), 6),
            "temperature": self.temperature,
        }
        return Decision(selected_action, probabilities, tuple(sorted(metadata.items())))

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        expected_sha256: str | None = None,
        seed: int = 0,
        temperature: float = 0.0,
    ) -> NumpyMLPPolicy:
        sha256 = _sha256_file(path)
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise ValueError("checkpoint SHA-256 does not match expected value")
        with np.load(path, allow_pickle=False) as checkpoint:
            checkpoint_version = str(checkpoint["checkpoint_version"].item())
            if checkpoint_version != CHECKPOINT_VERSION:
                raise ValueError(f"unsupported checkpoint version: {checkpoint_version}")
            encoder_payload = json.loads(str(checkpoint["encoder_json"].item()))
            if encoder_payload.get("version") != ENCODER_VERSION:
                raise ValueError("checkpoint encoder version is incompatible")
            encoder = ObservationEncoder(EncoderConfig(**encoder_payload["config"]))
            if encoder.feature_names != tuple(encoder_payload["feature_names"]):
                raise ValueError("checkpoint feature schema does not match current encoder")
            policy_name = str(checkpoint["policy_name"].item())
            weights = MLPWeights(
                checkpoint["input_to_hidden"].astype(np.float32),
                checkpoint["hidden_bias"].astype(np.float32),
                checkpoint["hidden_to_hidden"].astype(np.float32),
                checkpoint["second_hidden_bias"].astype(np.float32),
                checkpoint["hidden_to_action"].astype(np.float32),
                checkpoint["action_bias"].astype(np.float32),
            )
        return cls(
            weights,
            encoder=encoder,
            name=policy_name,
            seed=seed,
            temperature=temperature,
            checkpoint_sha256=sha256,
        )


def save_checkpoint(
    path: Path,
    weights: MLPWeights,
    encoder: ObservationEncoder,
    *,
    policy_name: str,
) -> str:
    """Write a non-executable, schema-versioned checkpoint and return SHA-256."""

    weights.validate(encoder.feature_count)
    if path.suffix != ".npz":
        raise ValueError("neural checkpoints must use the .npz extension")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite neural checkpoint: {path}")
    if not policy_name or any(character.isspace() for character in policy_name):
        raise ValueError("policy_name must be non-empty and contain no whitespace")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        checkpoint_version=np.asarray(CHECKPOINT_VERSION),
        encoder_json=np.asarray(json.dumps(encoder.to_dict(), sort_keys=True)),
        policy_name=np.asarray(policy_name),
        input_to_hidden=weights.input_to_hidden,
        hidden_bias=weights.hidden_bias,
        hidden_to_hidden=weights.hidden_to_hidden,
        second_hidden_bias=weights.second_hidden_bias,
        hidden_to_action=weights.hidden_to_action,
        action_bias=weights.action_bias,
    )
    return _sha256_file(path)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
