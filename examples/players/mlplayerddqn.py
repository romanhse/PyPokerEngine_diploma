import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pypokerengine.players import BasePokerPlayer


ACTIONS = ["fold", "call", "raise"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class MLPlayerDDQN(BasePokerPlayer):
    def __init__(self, state_dim=5, buffer_size=50000, batch_size=64, gamma=0.99, lr=1e-3,
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, tau=0.01):
        self.state_dim = state_dim
        self.action_dim = len(ACTIONS)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.update_target_net()

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau

        self.prev_state = None
        self.prev_action = None
        self.prev_stack = None

        self.model_path = "ddqn_model.pt"
        self.load_model()

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self.encode_state(hole_card, round_state)
        self.prev_state = state
        self.prev_stack = self.get_stack(round_state)

        if random.random() < self.epsilon:
            action_idx = random.choice(range(len(valid_actions)))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()

        self.prev_action = action_idx

        action = valid_actions[action_idx]['action']
        amount = valid_actions[action_idx].get('amount', 0)
        if action == 'raise':
            amount = valid_actions[action_idx]['amount']['min']

        return action, amount

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.prev_state is None or self.prev_action is None:
            return

        reward = self.get_stack(round_state) - (self.prev_stack or 0)
        next_state = self.encode_state(self.hole_card, round_state)
        done = True

        self.buffer.append((self.prev_state, self.prev_action, reward, next_state, done))

        if len(self.buffer) >= self.batch_size:
            self.train_model()

        self.update_target_net_soft()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.save_model()

    def train_model(self):
        batch = random.sample(self.buffer, self.batch_size)
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

        loss = self.loss_fn(q_vals, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_net_soft(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def encode_state(self, hole_card, round_state):
        strength = self.evaluate_hand_strength(hole_card)
        pot = round_state['pot']['main']['amount'] / 1000.0
        num_community = len(round_state['community_card']) / 5.0
        pos = self.get_position(round_state)
        stack = self.get_stack(round_state) / 1000.0
        return [strength, pot, num_community, pos, stack]

    def evaluate_hand_strength(self, hole_card):
        ranks = '23456789TJQKA'
        rank_map = {r: i for i, r in enumerate(ranks)}
        try:
            values = [rank_map[c[1]] for c in hole_card]  # fixed: suit+rank format
        except Exception:
            return 0.0
        return sum(values) / (len(values) * 12.0)

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

    # Required stubs
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hole_card = hole_card
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, new_action, round_state): pass
