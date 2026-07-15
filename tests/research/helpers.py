"""Small deterministic policies shared by research-stack tests."""

from __future__ import annotations

from poker_research.types import Decision, Observation


def deterministic_decision(observation: Observation, action: str) -> Decision:
    """Build a valid point-mass decision over the current legal action set."""

    return Decision(
        action=action,
        probabilities=tuple(
            (option.name, float(option.name == action))
            for option in observation.legal_actions
        ),
        metadata=(("test_policy", True),),
    )


class RecordingCheckCallPolicy:
    """Check/call policy that preserves every observation for assertions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.observations: list[Observation] = []

    def decide(self, observation: Observation) -> Decision:
        self.observations.append(observation)
        return deterministic_decision(observation, "check_call")


class RaiseMinimumOncePolicy(RecordingCheckCallPolicy):
    """Take the minimum raise at the first opportunity, then check/call."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.raised = False

    def decide(self, observation: Observation) -> Decision:
        self.observations.append(observation)
        if not self.raised and observation.option("raise_min") is not None:
            self.raised = True
            return deterministic_decision(observation, "raise_min")
        return deterministic_decision(observation, "check_call")
