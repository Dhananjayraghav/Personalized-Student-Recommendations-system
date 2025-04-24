import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class RecommendationEnv:
    def __init__(self, student_model, reward_function):
        self.student_model = student_model
        self.reward_function = reward_function
        self.reset()

    def reset(self):
        self.current_state = self.student_model.get_initial_state()
        return self.current_state

    def step(self, action):
        # Action is a recommendation
        reward = self.reward_function.calculate(
            self.current_state,
            action
        )
        next_state = self.student_model.update_state(
            self.current_state,
            action
        )
        self.current_state = next_state
        return next_state, reward, False, {}  # (state, reward, done, info)


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


class RLRecommender:
    def __init__(self, env, gamma=0.99, lr=0.001):
        self.env = env
        self.gamma = gamma
        self.memory = deque(maxlen=10000)

        input_size = len(env.reset())
        output_size = 100  # Number of possible recommendations

        self.policy_net = PolicyNetwork(input_size, 128, output_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        action = torch.multinomial(probs, 1).item()
        return action

    def train(self, episodes=1000, batch_size=32):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append((state, action, reward, next_state, done))
                total_reward += reward
                state = next_state

                if len(self.memory) >= batch_size:
                    self.update_model(batch_size)

            print(f"Episode {episode}, Total Reward: {total_reward}")

    def update_model(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.policy_net(next_states).max(1)[0].detach()
        expected_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()