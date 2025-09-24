import gymnasium as gym
import time
from main import Policy
import torch
from gymnasium.wrappers import RecordVideo

torch.manual_seed(0) # set random seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dict = torch.load('gymnasium/checkpoint_best.pth')

policy = Policy()
policy.load_state_dict(state_dict)
policy = policy.to(device)

def show_smart_agent():
    env = gym.make('Acrobot-v1', render_mode="human")
    state, _ = env.reset()
    actions = []
    score = 0
    for t in range(1000):
        action, _ = policy.act(state)
        # action = 1
        # actions = [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        # action = actions[t % actions.__len__()]
        actions.append(action)
        env.render()
        state, reward, done, _, _ = env.step(action)
        score += reward
        if done:
            break
        # time.sleep(0.05)
    print(f'Score: {score}')
    print(f'Actions: {actions}')

    env.close()


if __name__ == '__main__':
    show_smart_agent()