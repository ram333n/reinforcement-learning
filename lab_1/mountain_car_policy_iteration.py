import gymnasium as gym
import numpy as np
from tqdm import tqdm

from commons.mountain_car_commons import test_policy, map_pos_bin_to_value, \
    map_vel_bin_to_value, apply_action_to_state, map_pos_value_to_bin, \
    map_vel_value_to_bin


def evaluate_value_function(env, states, policy, n_iter=1000, eps=10e-6, gamma=0.99):
    value_table = np.zeros(states.shape)

    n_pos_bins = states.shape[0]
    n_vel_bins = states.shape[1]

    for _ in range(n_iter):
        previous_value_table = np.copy(value_table)

        for pos_bin_idx in range(n_pos_bins):
            for vel_bin_idx in range(n_vel_bins):
                pos = map_pos_bin_to_value(env, pos_bin_idx, n_pos_bins)
                vel = map_vel_bin_to_value(env, vel_bin_idx, n_vel_bins)
                a = policy[pos_bin_idx, vel_bin_idx]

                next_pos, next_vel = apply_action_to_state(env, (pos, vel), a)
                next_pos_bin = map_pos_value_to_bin(env, next_pos, n_pos_bins)
                next_vel_bin = map_vel_value_to_bin(env, next_vel,n_vel_bins)
                # reward = 0 if has_reached_the_flag(env, next_pos) else -1
                reward = -1 + np.abs(next_pos - pos)
                value_table[pos_bin_idx, vel_bin_idx] = reward + gamma * previous_value_table[next_pos_bin, next_vel_bin]

        if np.sum(np.fabs(previous_value_table - value_table)) <= eps:
            break

    return value_table

def extract_policy(env, states, value_table, gamma=0.99):
    policy = np.zeros(states.shape)

    n_pos_bins = states.shape[0]
    n_vel_bins = states.shape[1]

    for pos_bin_idx in range(n_pos_bins):
        for vel_bin_idx in range(n_vel_bins):
            pos = map_pos_bin_to_value(env, pos_bin_idx, n_pos_bins)
            vel = map_vel_bin_to_value(env, vel_bin_idx, n_vel_bins)

            q_values = []

            for a in range(env.action_space.n):
                next_pos, next_vel = apply_action_to_state(env, (pos, vel), a)
                next_pos_bin = map_pos_value_to_bin(env, next_pos, n_pos_bins)
                next_vel_bin = map_vel_value_to_bin(env, next_vel, n_vel_bins)
                # reward = 0 if has_reached_the_flag(env, next_pos) else -1
                reward = -1 + np.abs(next_pos - pos)
                q_value = reward + gamma * value_table[next_pos_bin, next_vel_bin]
                q_values.append(q_value)

            policy[pos_bin_idx, vel_bin_idx] = np.argmax(q_values)

    return policy


def policy_iteration(env, states, n_iter=1000, n_value_table_iter=50, eps=10e-6, gamma=0.99):
    value_table = np.zeros(states.shape)
    policy = np.zeros(states.shape)

    for _ in tqdm(range(n_iter)):
        old_policy = np.copy(policy)
        value_table = evaluate_value_function(env, states, old_policy, n_iter=n_value_table_iter, eps=eps, gamma=gamma)
        policy = extract_policy(env, states, value_table, gamma=gamma)

        if np.array_equal(old_policy, policy):
            break

    return value_table, policy


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="human")
    n_pos_bins = 50
    n_vel_bins = 25
    states = np.zeros((n_pos_bins, n_vel_bins))

    value_table, policy = policy_iteration(env, states, n_iter=30)

    total_reward = test_policy(env, states, policy)
    print(f"Total reward for Policy iteration policy: {total_reward}")