"""Small, engine-independent contracts shared by policies and arenas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

ActionName: TypeAlias = str
MetadataValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ActionOption:
    """One legal action after applying the experiment's bet-size abstraction."""

    name: ActionName
    raise_to: int | None = None


@dataclass(frozen=True, slots=True)
class PublicAction:
    """Public information emitted after a policy decision."""

    seat: int
    action: ActionName
    amount: int | None


@dataclass(frozen=True, slots=True)
class Observation:
    """Information that is legally visible to the acting heads-up player."""

    hand_id: str
    seat: int
    street: str
    position: str
    hole_cards: tuple[str, str]
    board_cards: tuple[str, ...]
    stacks: tuple[int, int]
    street_bets: tuple[int, int]
    pot: int
    call_amount: int
    effective_stack: int
    spr: float
    legal_actions: tuple[ActionOption, ...]
    history: tuple[PublicAction, ...] = ()
    rng_key: str = ""

    def option(self, name: ActionName) -> ActionOption | None:
        """Return a named legal action, if it is available."""

        return next((option for option in self.legal_actions if option.name == name), None)


@dataclass(frozen=True, slots=True)
class Decision:
    """A chosen action plus an auditable distribution over legal actions."""

    action: ActionName
    probabilities: tuple[tuple[ActionName, float], ...]
    metadata: tuple[tuple[str, MetadataValue], ...] = field(default_factory=tuple)


class Policy(Protocol):
    """Minimal policy interface; no engine objects or hidden cards can leak through it."""

    @property
    def name(self) -> str:
        """Stable versioned identifier used in manifests and comparisons."""

        ...

    def decide(self, observation: Observation) -> Decision:
        """Choose one action from ``observation.legal_actions``."""

        ...
