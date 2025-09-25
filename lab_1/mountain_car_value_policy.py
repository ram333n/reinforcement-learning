import gymnasium as gym

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


def main():
    env = gym.make('MountainCar-v0')
    simulate_random_agent(env)


if __name__ == "__main__":
    main()