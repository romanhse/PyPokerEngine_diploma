import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from pypokerengine.players import BasePokerPlayer
from treys import Evaluator, Card


ACTIONS = ["fold", "call", "raise"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


class DuelingQNetwork(nn.Module):
    # def __init__(self, input_dim, output_dim):
    #     super(DuelingQNetwork, self).__init__()
    #     self.feature = nn.Sequential(
    #         nn.Linear(input_dim, 128),
    #         nn.ReLU()
    #     )
    #     self.value_stream = nn.Sequential(
    #         nn.Linear(128, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, 1)
    #     )
    #     self.advantage_stream = nn.Sequential(
    #         nn.Linear(128, 128),
    #         nn.ReLU(),
    #         nn.Linear(128, output_dim)
    #     )

    def __init__(self, input_dim, output_dim):
        super(DuelingQNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class MLPlayerDDQN(BasePokerPlayer):
    def __init__(self, model_path, initial_stack, state_dim=9, buffer_size=50000, batch_size=64, gamma=0.99, lr=1e-3,
                 epsilon=1.0, epsilon_min=0.15, epsilon_decay=0.999, tau=0.01, alpha=0.6, beta=0.4):
        self.state_dim = state_dim
        self.action_dim = len(ACTIONS)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingQNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.update_target_net()

        self.buffer = []
        self.priorities = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.initial_stack = initial_stack

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau

        self.alpha = alpha  # prioritization exponent
        self.beta = beta    # importance sampling

        self.prev_state = None
        self.prev_action = None
        self.prev_stack = None

        self.model_path = model_path
        self.rewards_log = []
        self.win_log = []

        self.load_model()

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.encode_state(hole_card, round_state)
        self.prev_state = state
        self.prev_stack = self.get_stack(round_state)
        log_bluff = False

        if random.random() < self.epsilon:
            action_idx = random.choice(range(len(valid_actions)))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()
            log_bluff = True

        self.prev_action = action_idx

        action = valid_actions[action_idx]['action']
        amount = valid_actions[action_idx].get('amount', 0)
        if action == 'raise':
            amount = valid_actions[action_idx]['amount']['min']
        # if log_bluff and amount > 0:
        #     strength = state[0]
        #     with open('log_bluff.txt', 'a') as f:
        #         f.write(f'{strength}, {action}, \n')

        return action, amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.prev_state is None or self.prev_action is None:
            return

        reward = self.get_stack(round_state) - (self.prev_stack or 0)
        self.rewards_log.append(reward)
        next_state = self.encode_state(self.hole_card, round_state)
        done = True

        td_error = abs(reward)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
            self.priorities.pop(0)

        self.buffer.append((self.prev_state, self.prev_action, reward, next_state, done))
        self.priorities.append(td_error + 1e-5)

        # Win tracking
        win = any(w['uuid'] == self.uuid for w in winners)
        self.win_log.append(1 if win else 0)

        if len(self.buffer) >= self.batch_size:
            self.train_model()

        self.update_target_net_soft()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.save_model()

    def train_model(self):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = torch.FloatTensor(weights / weights.max()).unsqueeze(1).to(self.device)

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        q_vals = self.policy_net(states).gather(1, actions)
        next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
        next_q_vals = self.target_net(next_states).gather(1, next_actions)
        targets = rewards + self.gamma * next_q_vals * (~dones)

        loss = (self.loss_fn(q_vals, targets.detach()) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_net_soft(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def encode_state(self, hole_card, round_state):
        community_cards = round_state['community_card']
        strength = self.evaluate_hand_strength(hole_card, community_cards)
        pot = round_state['pot']['main']['amount'] / self.initial_stack
        num_community = len(round_state['community_card']) / 5.0
        pos = self.get_position(round_state)
        stack = self.get_stack(round_state) / self.initial_stack
        if self.win_log:
            win_prop = sum(self.win_log)/len(self.win_log)
        else:
            win_prop = 0.5
        if sum(self.opp_moves) == 0:
            moves = [0.33, 0.33, 0.33]
        else:
            moves = []
            for m in self.opp_moves:
                moves.append(m/sum(self.opp_moves))
        return [strength, pot, num_community, pos, stack,win_prop] + moves

    def evaluate_hand_strength(self, hole_card, community_cards):
        evaluator = Evaluator()
        hand = []
        board = []
        for card in hole_card:
            new_format = card[1] + card[0].lower()
            hand.append(Card.new(new_format))
        for card in community_cards:
            new_format = card[1] + card[0].lower()
            board.append(Card.new(new_format))
        if len(board) < 3:
            return 0.5
        else:
            score = evaluator.evaluate(hand, board)
            percentile = evaluator.get_five_card_rank_percentage(score)
            return 1 - percentile

    def get_stack(self, round_state):
        for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
                return seat['stack']
        return 0

    def get_position(self, round_state):
        seats = round_state['seats']
        idx = next((i for i, s in enumerate(seats) if s['uuid'] == self.uuid), 0)
        return idx / max(1, len(seats) - 1)

    def save_model(self):
        torch.save({
            'model': self.policy_net.state_dict(),
            'epsilon': self.epsilon
        }, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            data = torch.load(self.model_path, map_location=self.device)
            self.policy_net.load_state_dict(data['model'])
            self.epsilon = data.get('epsilon', self.epsilon)

    def receive_game_start_message(self, game_info):
        self.opp_moves = [0,0,0]
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state):
        pass
        if action['player_uuid'] != self.uuid:
            if action['action'] == 'fold':
                self.opp_moves[0] += 1
            elif action['action'] == 'call':
                self.opp_moves[1] += 1
            else:
                self.opp_moves[2] += 1
