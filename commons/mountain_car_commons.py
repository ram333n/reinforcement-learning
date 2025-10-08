import numpy as np

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