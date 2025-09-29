import gymnasium as gym
import numpy as np
import tqdm


def simulate_random_agent(env):
    env.reset()
    total_reward = 0

    for _ in range(1000):
        env.render()
        state, reward, done, _, _ = env.step(env.action_space.sample())
        total_reward += reward

        if done:
            break

    print(f"Total reward for random agent: {total_reward}")
    print(f"Total reward for random agent: {env.unwrapped}")


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


def map_pos_bin_to_value(env, pos_bin_idx, n_pos_bins):
    pos_low, pos_high = env.observation_space.low[0], env.observation_space.high[0]

    return map_bin_to_value(pos_bin_idx, pos_low, pos_high, n_pos_bins)


def map_bin_to_value(bin_idx, low, high, n_bins):
    if n_bins == 1:
        return (low + high) / 2

    return low + bin_idx * (high - low) / (n_bins - 1)


def map_vel_bin_to_value(env, vel_bin_idx, n_vel_bins):
    vel_low, vel_high = env.observation_space.low[1], env.observation_space.high[1]

    return map_bin_to_value(vel_bin_idx, vel_low, vel_high, n_vel_bins)


def apply_action_to_state(env, state, action):
    force = env.unwrapped.force
    gravity = env.unwrapped.gravity

    pos, vel = state

    vel_next = vel + (action - 1) * force - np.cos(3 * pos) * gravity
    vel_next = np.clip(vel_next, -0.07, 0.07)

    pos_next = pos + vel_next
    pos_next = np.clip(pos_next, -1.2, 0.6)

    return pos_next, vel_next


def map_pos_value_to_bin(env, pos_value, n_pos_bins):
    pos_low, pos_high = env.observation_space.low[0], env.observation_space.high[0]
    return map_value_to_bin(pos_value, pos_low, pos_high, n_pos_bins)


def map_value_to_bin(value, low, high, n_bins):
    if n_bins == 1:
        return 0

    step = (high - low) / (n_bins - 1)
    bin_idx = int(round((value - low) / step))

    return int(np.clip(bin_idx, 0, n_bins - 1))


def map_vel_value_to_bin(env, vel_value, n_vel_bins):
    vel_low, vel_high = env.observation_space.low[1], env.observation_space.high[1]
    return map_value_to_bin(vel_value, vel_low, vel_high, n_vel_bins)


def has_reached_the_flag(env, pos):
    return pos >= env.unwrapped.goal_position


def test_policy(env, states, policy, render=True):
    n_pos_bins, n_vel_bins = states.shape
    state, _ = env.reset()

    if render:
        env.render()

    total_reward = 0
    for _ in range(1000):
        pos, vel = state
        pos_bin = map_pos_value_to_bin(env, pos, n_pos_bins)
        vel_bin = map_vel_value_to_bin(env, vel, n_vel_bins)

        action = int(policy[pos_bin, vel_bin])
        state, reward, done, info, _ = env.step(action)

        if render:
            env.render()

        total_reward += reward
        if done:
            break

    return total_reward


def main():
    env = gym.make("MountainCar-v0", render_mode="human")
    n_pos_bins = 50
    n_vel_bins = 25
    states = np.zeros((n_pos_bins, n_vel_bins))

    value_table, policy = value_iteration(env, states, n_iter=500)

    total_reward = test_policy(env, states, policy)
    print(f"Total reward for Value iteration policy: {total_reward}")


if __name__ == "__main__":
    main()