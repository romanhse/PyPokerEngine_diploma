"""Authoritative heads-up no-limit hold'em arena backed by PokerKit.

The policy boundary deliberately exposes no engine state.  That makes hidden-card
leaks harder and lets the same policy be evaluated by another engine later.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from pokerkit import Automation, HandHistory, NoLimitTexasHoldem, State

from poker_research.types import ActionOption, Decision, Observation, Policy, PublicAction

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_DEALING,
    Automation.BOARD_DEALING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)
STREETS = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    """Rules frozen for one benchmark."""

    starting_stack: int = 200
    small_blind: int = 1
    big_blind: int = 2
    ante: int = 0

    def validate(self) -> None:
        if self.starting_stack <= 0:
            raise ValueError("starting_stack must be positive")
        if not 0 < self.small_blind <= self.big_blind:
            raise ValueError("blinds must satisfy 0 < small_blind <= big_blind")
        if self.starting_stack < self.big_blind:
            raise ValueError("starting_stack must cover the big blind")
        if self.ante < 0:
            raise ValueError("ante must be non-negative")


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One auditable policy call."""

    index: int
    seat: int
    policy: str
    street: str
    hole_cards: tuple[str, str]
    board_cards: tuple[str, ...]
    pot: int
    call_amount: int
    stacks: tuple[int, int]
    street_bets: tuple[int, int]
    legal_actions: tuple[ActionOption, ...]
    decision: Decision
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HandResult:
    """Terminal result plus enough provenance to audit and replay a hand."""

    hand_id: str
    policy_rng_key: str
    deal_seed: int
    policy_names: tuple[str, str]
    deck_hash: str
    hole_cards: tuple[tuple[str, str], tuple[str, str]]
    board_cards: tuple[str, ...]
    starting_stacks: tuple[int, int]
    finishing_stacks: tuple[int, int]
    payoffs: tuple[int, int]
    decisions: tuple[DecisionRecord, ...]
    public_actions: tuple[PublicAction, ...]
    hand_history: HandHistory

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("hand_history")
        return payload


def play_hand(
    policies: Sequence[Policy],
    *,
    deal_seed: int,
    hand_id: str,
    config: ArenaConfig | None = None,
    policy_rng_key: str | None = None,
) -> HandResult:
    """Play one deterministic hand without mutating the process-global RNG."""

    global_random_state = random.getstate()
    try:
        random.seed(deal_seed)
        return _play_hand(
            policies,
            deal_seed=deal_seed,
            hand_id=hand_id,
            config=config,
            policy_rng_key=policy_rng_key or hand_id,
        )
    finally:
        random.setstate(global_random_state)


def _play_hand(
    policies: Sequence[Policy],
    *,
    deal_seed: int,
    hand_id: str,
    config: ArenaConfig | None,
    policy_rng_key: str,
) -> HandResult:
    """Internal hand loop; caller owns isolation of PokerKit's module RNG."""

    rules = config or ArenaConfig()
    rules.validate()
    if len(policies) != 2:
        raise ValueError("heads-up arena requires exactly two policies")
    if len({policy.name for policy in policies}) != 2:
        raise ValueError("policy names must be distinct for auditable logs")

    game = NoLimitTexasHoldem(
        AUTOMATIONS,
        True,
        rules.ante,
        (rules.small_blind, rules.big_blind),
        rules.big_blind,
    )
    state = game((rules.starting_stack, rules.starting_stack), 2)
    initial_holes = _hole_cards(state)
    deck_hash = _deck_hash(state)
    history: list[PublicAction] = []
    decisions: list[DecisionRecord] = []

    while state.status:
        actor = state.actor_index
        if actor is None:
            raise RuntimeError("active PokerKit state has no actor")
        observation = _observation(
            state,
            actor,
            hand_id,
            tuple(history),
            policy_rng_key,
        )
        started_ns = time.perf_counter_ns()
        decision = policies[actor].decide(observation)
        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        selected = _validate_decision(observation, decision)
        record = DecisionRecord(
            index=len(decisions),
            seat=actor,
            policy=policies[actor].name,
            street=observation.street,
            hole_cards=observation.hole_cards,
            board_cards=observation.board_cards,
            pot=observation.pot,
            call_amount=observation.call_amount,
            stacks=observation.stacks,
            street_bets=observation.street_bets,
            legal_actions=observation.legal_actions,
            decision=decision,
            latency_ms=latency_ms,
        )
        public_amount = (
            observation.call_amount if selected.name == "check_call" else selected.raise_to
        )
        public_action = PublicAction(actor, selected.name, public_amount)
        decisions.append(record)
        history.append(public_action)
        _apply_action(state, selected)

    finishing_stacks = tuple(int(stack) for stack in state.stacks)
    payoffs = tuple(int(payoff) for payoff in state.payoffs)
    if sum(finishing_stacks) != 2 * rules.starting_stack:
        raise RuntimeError("PokerKit violated chip conservation")
    if sum(payoffs) != 0:
        raise RuntimeError("PokerKit produced non-zero-sum payoffs")
    expected_payoffs = tuple(stack - rules.starting_stack for stack in finishing_stacks)
    if payoffs != expected_payoffs:
        raise RuntimeError("payoffs do not match finishing stacks")

    hand_history = HandHistory.from_game_state(
        game,
        state,
        compression_status=False,
        hand=hand_id,
        players=[policy.name for policy in policies],
    )
    return HandResult(
        hand_id=hand_id,
        policy_rng_key=policy_rng_key,
        deal_seed=deal_seed,
        policy_names=(policies[0].name, policies[1].name),
        deck_hash=deck_hash,
        hole_cards=initial_holes,
        board_cards=_board_cards(state),
        starting_stacks=(rules.starting_stack, rules.starting_stack),
        finishing_stacks=(finishing_stacks[0], finishing_stacks[1]),
        payoffs=(payoffs[0], payoffs[1]),
        decisions=tuple(decisions),
        public_actions=tuple(history),
        hand_history=hand_history,
    )


def _observation(
    state: State,
    actor: int,
    hand_id: str,
    history: tuple[PublicAction, ...],
    policy_rng_key: str,
) -> Observation:
    if state.street_index is None or not 0 <= state.street_index < len(STREETS):
        raise RuntimeError(f"unexpected street index: {state.street_index}")
    hole = tuple(repr(card) for card in state.hole_cards[actor])
    if len(hole) != 2:
        raise RuntimeError("actor does not have exactly two hole cards")
    stacks = tuple(int(stack) for stack in state.stacks)
    bets = tuple(int(bet) for bet in state.bets)
    pot = int(state.total_pot_amount)
    call_amount = int(state.checking_or_calling_amount or 0)
    effective_stack = min(stacks)
    return Observation(
        hand_id=hand_id,
        seat=actor,
        street=STREETS[state.street_index],
        position="button_sb" if actor == 1 else "big_blind",
        hole_cards=(hole[0], hole[1]),
        board_cards=_board_cards(state),
        stacks=(stacks[0], stacks[1]),
        street_bets=(bets[0], bets[1]),
        pot=pot,
        call_amount=call_amount,
        effective_stack=effective_stack,
        spr=effective_stack / pot if pot else float("inf"),
        legal_actions=_legal_actions(state, actor),
        history=history,
        rng_key=policy_rng_key,
    )


def _legal_actions(state: State, actor: int) -> tuple[ActionOption, ...]:
    options: list[ActionOption] = []
    if state.can_fold():
        options.append(ActionOption("fold"))
    if state.can_check_or_call():
        options.append(ActionOption("check_call"))
    if not state.can_complete_bet_or_raise_to():
        return tuple(options)

    minimum = state.min_completion_betting_or_raising_to_amount
    maximum = state.max_completion_betting_or_raising_to_amount
    if minimum is None or maximum is None:
        raise RuntimeError("raise reported legal without min/max bounds")
    minimum_int, maximum_int = int(minimum), int(maximum)
    if minimum_int > maximum_int:
        raise RuntimeError("invalid raise bounds")

    seen = {minimum_int}
    options.append(ActionOption("raise_min", minimum_int))
    current_bet = int(state.bets[actor])
    call_amount = int(state.checking_or_calling_amount or 0)
    pot_after_call = int(state.total_pot_amount) + call_amount
    for name, fraction in (
        ("raise_half_pot", 0.5),
        ("raise_pot", 1.0),
        ("raise_2pot", 2.0),
    ):
        target = round(current_bet + call_amount + fraction * pot_after_call)
        if minimum_int <= target < maximum_int and target not in seen:
            options.append(ActionOption(name, target))
            seen.add(target)
    if maximum_int not in seen:
        options.append(ActionOption("raise_all_in", maximum_int))
    return tuple(options)


def _validate_decision(observation: Observation, decision: Decision) -> ActionOption:
    options = {option.name: option for option in observation.legal_actions}
    if decision.action not in options:
        raise ValueError(f"illegal action {decision.action!r}; legal={tuple(options)}")
    probabilities = dict(decision.probabilities)
    if len(probabilities) != len(decision.probabilities):
        raise ValueError("decision contains duplicate probability entries")
    if set(probabilities) != set(options):
        raise ValueError("decision probabilities must cover every legal action exactly once")
    if any(
        not math.isfinite(probability) or probability < 0 or probability > 1
        for probability in probabilities.values()
    ):
        raise ValueError("decision probabilities must be finite and in [0, 1]")
    if not math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-9):
        raise ValueError("decision probabilities must sum to one")
    if probabilities[decision.action] <= 0:
        raise ValueError("selected action must have positive probability")
    return options[decision.action]


def _apply_action(state: State, action: ActionOption) -> None:
    if action.name == "fold":
        state.fold()
    elif action.name == "check_call":
        state.check_or_call()
    elif action.name.startswith("raise_") and action.raise_to is not None:
        state.complete_bet_or_raise_to(action.raise_to)
    else:
        raise ValueError(f"cannot apply action: {action}")


def _hole_cards(state: State) -> tuple[tuple[str, str], tuple[str, str]]:
    holes = tuple(tuple(repr(card) for card in cards) for cards in state.hole_cards)
    if len(holes) != 2 or any(len(hole) != 2 for hole in holes):
        raise RuntimeError("unexpected initial hole cards")
    return ((holes[0][0], holes[0][1]), (holes[1][0], holes[1][1]))


def _board_cards(state: State) -> tuple[str, ...]:
    return tuple(repr(card) for board in state.board_cards for card in board)


def _deck_hash(state: State) -> str:
    cards = [*(repr(card) for hole in state.hole_cards for card in hole)]
    cards.extend(repr(card) for card in state.deck_cards)
    return hashlib.sha256("|".join(cards).encode()).hexdigest()
