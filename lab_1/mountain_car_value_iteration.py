import gymnasium as gym
import numpy as np
import tqdm

from commons.mountain_car_commons import test_policy, map_pos_bin_to_value, \
    map_vel_bin_to_value, apply_action_to_state, map_pos_value_to_bin, \
    map_vel_value_to_bin


def value_iteration(env, states, n_iter=1000, eps=10e-6, gamma=0.99):
    value_table = np.zeros(states.shape)
    policy = np.zeros(states.shape)

    n_pos_bins = states.shape[0]
    n_vel_bins = states.shape[1]

    for _ in tqdm.tqdm(range(n_iter)):
        previous_value_table = np.copy(value_table)

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
                    q_value = reward + gamma * previous_value_table[next_pos_bin, next_vel_bin]
                    q_values.append(q_value)

                value_table[pos_bin_idx, vel_bin_idx] = max(q_values)
                policy[pos_bin_idx, vel_bin_idx] = np.argmax(q_values)

        print(f"eps = {np.sum(np.fabs(previous_value_table - value_table))}")
        if np.sum(np.fabs(previous_value_table - value_table)) <= eps:
            break

    return value_table, policy


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="human")
    n_pos_bins = 50
    n_vel_bins = 25
    states = np.zeros((n_pos_bins, n_vel_bins))

    value_table, policy = value_iteration(env, states, n_iter=10)

    total_reward = test_policy(env, states, policy)
    print(f"Total reward for Value iteration policy: {total_reward}")