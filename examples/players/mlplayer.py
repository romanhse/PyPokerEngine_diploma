import random
from pypokerengine.players import BasePokerPlayer
from collections import defaultdict
import pickle
import os

ACTIONS = ["fold", "call", "raise"]

class MLPlayer(BasePokerPlayer):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, bluff_chance=0.1):
        self.q_table = self.load_q_table()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.bluff_chance = bluff_chance
        self.prev_state = None
        self.prev_action = None
        self.prev_stack = None

    def save_q_table(self, path="q_table_1.pkl"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, path="q_table_1.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return defaultdict(lambda: [0.0, 0.0, 0.0], pickle.load(f))
        return defaultdict(lambda: [0.0, 0.0, 0.0])

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.get_state(hole_card, round_state)
        action_idx = self.choose_action(state)

        # Smart bluff if hand looks weak and we're in late position
        hand_strength = self.evaluate_hand_strength(hole_card)
        position = self.get_position(round_state)

        if hand_strength < 0.3 and position == 'late' and random.random() < self.bluff_chance:
            action_idx = 2 if any(v['action'] == 'raise' for v in valid_actions) else 1

        self.prev_state = state
        self.prev_action = action_idx
        self.prev_stack = self.get_stack(round_state)

        action = valid_actions[action_idx]['action']
        amount = 0
        if action == "raise":
            amount = valid_actions[action_idx]["amount"]["min"]
        else:
            amount = valid_actions[action_idx].get("amount", 0)

        return action, amount

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        return int(self.argmax(self.q_table[state]))

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.prev_state is None or self.prev_action is None:
            return

        my_stack = self.get_stack(round_state)
        reward = my_stack - (self.prev_stack if self.prev_stack is not None else 1000)

        self.learn(self.prev_state, self.prev_action, reward)
        self.save_q_table()
        self.epsilon = max(0.1, self.epsilon * 0.995)  # decay epsilon

    def learn(self, state, action_idx, reward):
        current_q = self.q_table[state][action_idx]
        updated_q = current_q + self.alpha * (reward - current_q)
        self.q_table[state][action_idx] = updated_q

    def get_state(self, hole_card, round_state):
        community = round_state['community_card']
        strength = round(self.evaluate_hand_strength(hole_card), 1)
        pot = round_state['pot']['main']['amount']
        num_community = len(community)
        position = self.get_position(round_state)
        return (strength, num_community, pot // 50, position)

    def evaluate_hand_strength(self, hole_card):
        ranks = '23456789TJQKA'
        rank_map = {r: i for i, r in enumerate(ranks)}
        try:
            values = [rank_map[c[1]] for c in hole_card]  # FIXED: use c[1] (rank)
        except (IndexError, KeyError):
            return 0.0  # fallback if anything goes wrong
        return sum(values) / (len(values) * 12)

    def get_stack(self, round_state):
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                return seat['stack']
        return 0

    def get_position(self, round_state):
        seats = round_state['seats']
        idx = next(i for i, s in enumerate(seats) if s['uuid'] == self.uuid)
        if idx == 0:
            return 'early'
        elif idx == len(seats) - 1:
            return 'late'
        else:
            return 'middle'

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
        self.prev_state = None
        self.prev_action = None
        self.prev_stack = None

    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass

    @staticmethod
    def argmax(lst):
        return max(range(len(lst)), key=lambda x: lst[x])
