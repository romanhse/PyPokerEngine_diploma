import random
from pypokerengine.players import BasePokerPlayer
from collections import defaultdict

ACTIONS = ["fold", "call", "raise"]

class MLPlayer(BasePokerPlayer):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, bluff_chance=0.05):
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.bluff_chance = bluff_chance
        self.prev_state = None
        self.prev_action = None

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.get_state(hole_card, round_state)
        action_idx = self.choose_action(state)

        # Random bluff
        if random.random() < self.bluff_chance:
            action_idx = random.choice(range(len(valid_actions)))

        self.prev_state = state
        self.prev_action = action_idx

        action = valid_actions[action_idx]['action']

        # Choose appropriate amount for 'raise'
        if action == "raise":
            amount = valid_actions[action_idx]['amount']['min']
        else:
            amount = valid_actions[action_idx].get('amount', 0)

        # print(action, amount)
        return action, amount  # ✅ CORRECT FORMAT

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # explore
        return int(self.argmax(self.q_table[state]))  # exploit

    def receive_round_result_message(self, winners, hand_info, round_state):
        reward = 1 if self.uuid in [w['uuid'] for w in winners] else -1
        self.learn(self.prev_state, self.prev_action, reward)

    def learn(self, state, action_idx, reward):
        if state is None or action_idx is None:
            return  # skip learning if no action was taken

        current_q = self.q_table[state][action_idx]
        updated_q = current_q + self.alpha * (reward - current_q)
        self.q_table[state][action_idx] = updated_q

    def get_state(self, hole_card, round_state):
        community = round_state['community_card']
        return str(sorted(hole_card)) + "|" + str(sorted(community))

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
        self.prev_state = None
        self.prev_action = None

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    @staticmethod
    def argmax(lst):
        return max(range(len(lst)), key=lambda x: lst[x])
