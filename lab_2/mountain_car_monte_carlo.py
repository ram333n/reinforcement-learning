import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
import tqdm

from commons.mountain_car_commons import map_to_discrete_state, test_policy


def epsilon_greedy_policy(env, discrete_state, q_table, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda a: q_table[(discrete_state, a)])


def generate_episode(env, states, q_values, epsilon, n_timesteps=250):
    n_pos_bins, n_vel_bins = states.shape
    state = env.reset()[0]
    episode = []

    for t in range(n_timesteps):
        discrete_state = map_to_discrete_state(env, state, n_pos_bins, n_vel_bins)
        action = epsilon_greedy_policy(env, discrete_state, q_values, epsilon)
        next_state, _, done, info, _ = env.step(action)
        # reward = -1 + np.abs(next_state[0] - state[0])
        reward = combined_reward(next_state)

        # if done:
        #     reward = 150
        # else:
        #     reward = evaluate_energy(env, next_state)

        episode.append((discrete_state, action, reward))

        if done:
            break

        state = next_state

    return episode


def evaluate_energy(env, state, kinetic_factor=0.45, potential_factor=0.55):
    position = state[0]
    velocity = state[1]

    max_pos_diff = np.abs(env.observation_space.high[0] - env.observation_space.low[0])

    pos_normalized = (position + np.abs(env.observation_space.low[0])) / max_pos_diff
    vel_normalized = np.abs(velocity) / env.observation_space.high[1]

    return potential_factor * pos_normalized + kinetic_factor * vel_normalized


def combined_reward(state):
    position = state[0]
    velocity = state[1]

    return (position + 200 * velocity ** 2) - 1


def extract_policy(env, states, q_values):
    policy = np.zeros(states.shape)
    states_from_q_values = {state for (state, action) in q_values.keys()}

    for state in states_from_q_values:
        pos_bin, vel_bin = state
        policy[pos_bin, vel_bin] = max(list(range(env.action_space.n)), key=lambda a: q_values[(state, a)])

    return policy


if __name__ == "__main__":
    train_env = gym.make("MountainCar-v0")
    demo_env = gym.make("MountainCar-v0", render_mode="human")

    n_pos_bins = 50
    n_vel_bins = 25
    states = np.zeros((n_pos_bins, n_vel_bins))

    q_values = defaultdict(float)
    total_reward_by_state = defaultdict(float)
    state_visits_count = defaultdict(int)

    n_episodes = 20000
    n_episode_timesteps = 500
    epsilon = 0.5
    gamma = 0.99

    success_count = 0
    print_every = 100

    for i in tqdm.tqdm(range(n_episodes)):
        epsilon = max(0.05, 1 - i / n_episodes)
        episode = generate_episode(train_env, states, q_values, epsilon, n_episode_timesteps)

        if len(episode) < n_episode_timesteps:
            success_count += 1

        visited = set()

        for t, (state, action, reward) in enumerate(episode):
            if (state, action) not in visited:
                visited.add((state, action))

                G = 0
                for j, (_, _, r) in enumerate(episode[t:]):
                    G += (gamma ** j) * r

                total_reward_by_state[(state, action)] = total_reward_by_state[(state, action)] + G
                state_visits_count[(state, action)] += 1

                q_values[(state, action)] = total_reward_by_state[(state, action)] / state_visits_count[(state, action)]

        if (i + 1) % print_every == 0:
            success_rate = success_count / print_every
            print(f"Episodes {i + 1 - print_every}-{i + 1}: success rate = {success_rate:.1%}")
            success_count = 0

    policy = extract_policy(train_env, states, q_values)
    print(f"Total reward: {test_policy(demo_env, states, policy)}")
