import random
from collections import deque

import gym
import torch
import numpy as np

from tqdm import tqdm
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
# s[7] - right leg in contact with ground(Boolean)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=8, h_size=128, a_size=4):
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
    def __init__(self, state_dim, action_dim, replay_buffer_size, lr, gamma, epsilon, epsilon_decay, revise_target_every=1000):
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
        self.model = Policy(s_size=state_dim, a_size=action_dim).to(DEVICE)
        self.target_model = Policy(s_size=state_dim, a_size=action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        self.steps_counter += 1

        batch = random.sample(self.memory, batch_size)
        states = np.array([s for s, a, r, ns, d in batch])
        actions = np.array([a for s, a, r, ns, d in batch])
        rewards = np.array([r for s, a, r, ns, d in batch])
        next_states = np.array([ns for s, a, r, ns, d in batch])
        dones = np.array([d for s, a, r, ns, d in batch])

        states = torch.from_numpy(states).float().to(DEVICE)
        actions = torch.from_numpy(actions).long().to(DEVICE)
        rewards = torch.from_numpy(rewards).float().to(DEVICE)
        next_states = torch.from_numpy(next_states).float().to(DEVICE)
        dones = torch.from_numpy(dones).float().to(DEVICE)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_counter % self.revise_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > 0.02:
            self.epsilon *= self.epsilon_decay

    def act(self, state, epsilon=0.0):
        return self.model.act(state, epsilon)


def train(env, n_episodes=200):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer_size=75000,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995
    )

    scores = []
    scores_deque = deque(maxlen=100)
    best_reward = -np.inf

    for episode in tqdm(range(n_episodes)):
        state = env.reset()
        score = 0

        for t in range(1000):
            action = agent.act(state, agent.epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            score += reward

            agent.replay(64)

            if done:
                break

        agent.decay_epsilon()
        scores_deque.append(score)
        scores.append(score)
        print(f"Episode: {episode}, score: {score}, average score: {np.mean(scores_deque)}")

        if episode % 30 == 29:
            reward = test(env, agent.model, render=False, n_episodes=5)

            if reward > best_reward:
                best_reward = reward
                torch.save(agent.model.state_dict(),f"checkpoint_{env.spec.id}_best.pth")


def test(env, policy, render=True, n_episodes=1):
    total_reward = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            state = env.reset()

            for _ in range(1000):
                action = policy.act(state)

                state, reward, done, _ = env.step(action)

                if render:
                    env.render()

                total_reward += reward

                if done:
                    break

    print(f'Test total reward: {total_reward / n_episodes}')

    return total_reward / n_episodes

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    # train(env, n_episodes=2500)
    test_env = gym.make("LunarLander-v2")

    policy = Policy().to(DEVICE)
    policy.load_state_dict(torch.load(f'checkpoint_{env.spec.id}_best.pth'))

    test(test_env, policy, render=True, n_episodes=100)