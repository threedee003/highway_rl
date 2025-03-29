import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import highway_env
from itertools import count

from torch.utils.tensorboard import SummaryWriter



from .reinforce import REINFORCE
from ..utils.folder_utils import make_log_dir



env = gym.make('highway-v0', render_mode='human', config={
    "action": {
        "type": "DiscreteMetaAction"
    }
})


env.unwrapped.config['observation']['vehicles_count'] = 5

agent = REINFORCE(state_dim=5, action_dim=5)


n = make_log_dir("/home/hw1896957/HighwayEnv/rl_av/runs/REINFORCE")


writer = SummaryWriter(log_dir = f"/home/bikram/HighwayEnv/rl_av/runs/REINFORCE/run_{n}")
SAVE_PATH = f"/home/bikram/HighwayEnv/rl_av/REINFORCE/weights/run_{n}.pth"


num_episodes = 250

running_reward = 10

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    episode_len = 0

    while not done:
        action = agent.select_action(obs)
        obs_state, reward, done, _, _ = env.step(action)

        agent.rewards.append(reward)
        obs = obs_state
        episode_reward += reward
        episode_len += 1




    running_reward = 0.05 * episode_reward + (1-0.05)*running_reward
    agent.update_policy()

    # for name, param in agent.policy.sequential.named_parameters():
    #     print(f"{name}: {param.grad.abs().sum().item()}")
    writer.add_scalar("Rewards/Episode Reward", running_reward, ep)
    writer.add_scalar("Episode Lenght", episode_len, ep)
    for name, param in agent.policy.sequential.named_parameters():
        writer.add_histogram(f"policy_weights/{name}", param, ep)
        if param.grad is not None:
            writer.add_histogram(f"policy_gradients/{name}", param.grad, ep)



    
    print(f"Episode {ep+1}, Total Reward: {running_reward}")

torch.save(agent.policy.state_dict(), SAVE_PATH)
env.close()



