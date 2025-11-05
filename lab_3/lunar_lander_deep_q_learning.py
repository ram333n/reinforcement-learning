import random
from collections import deque

import gym
import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
import torch.optim as optim

def simulate_random_agent():
    env = gym.make("LunarLander-v2")
    env.reset()

    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        #  np.array(state, dtype=np.float32), reward, done, {}
        observation, reward, done, _ = env.step(action)
        print(observation)

# Action space:
# 0: do nothing
# 1: fire left orientation engine
# 2: fire main engine
# 3: fire right orientation engine

# State:
# s[0] - x coordinate of lander
# s[1] - y coordinate of lander
# s[2] - x velocity of lander
# s[3] - y velocity of lander
# s[4] - angle
# s[5] - angular velocity
# s[6] - left leg in contact with ground(Boolean)
# s[7] - right leg in contact with ground(Boolean

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=8, h_size=32, a_size=4):
        super(Policy, self).__init__()

        self.s_size = s_size
        self.h_size = h_size
        self.a_size = a_size

        self.fc_1 = nn.Linear(self.s_size, self.h_size)
        self.fc_2 = nn.Linear(self.h_size, self.a_size)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)

        return x

    def act(self, state, epsilon=0.0):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.a_size)

        q_values = self(torch.tensor(state, dtype=torch.float32, device=DEVICE)).to("cpu")

        return torch.argmax(q_values).item()


class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_size, lr, gamma, epsilon, epsilon_decay, revise_target_every=16):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer_size = replay_buffer_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.revise_target_every = revise_target_every
        self.steps_counter = 0

        self.memory = deque(maxlen=replay_buffer_size)
        self.model = Policy(s_size=state_dim, a_size=action_dim)
        self.target_model = Policy(s_size=state_dim, a_size=action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        self.steps_counter += 1

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward

            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32, device=DEVICE))).item()

            with torch.no_grad():
                target_f = self.target_model(torch.tensor(state, dtype=torch.float32, device=DEVICE)).to("cpu").numpy()
                target_f[action] = target
                target_f = torch.tensor(target_f, dtype=torch.float32, device=DEVICE)

            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(torch.tensor(state, dtype=torch.float32, device=DEVICE)))
            loss.backward()
            self.optimizer.step()

            if self.steps_counter % self.revise_target_every == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > 0.02:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    simulate_random_agent()