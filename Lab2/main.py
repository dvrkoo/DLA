import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pygame
import wandb
import os

# Initialize Weights & Biases
wandb.init(
    project="reinforce-cartpole",
    name="run_reinforce",
    config={"env": "CartPole-v1", "episodes": 1000, "gamma": 0.99, "lr": 1e-2},
)

pygame.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


def select_action(obs, policy):
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    dist = Categorical(policy(obs))
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def run_episode(env, policy, maxlen=500):
    obs, _ = env.reset()
    log_probs = []
    rewards = []

    for _ in range(maxlen):
        action, log_prob = select_action(obs, policy)
        obs, reward, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        if terminated or truncated:
            break

    return torch.stack(log_probs), rewards


def reinforce(
    policy,
    env,
    render_env=None,
    gamma=0.99,
    num_episodes=1000,
    lr=1e-2,
    checkpoint_dir="checkpoints",
):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)

    running_rewards = [0.0]
    best_running_reward = float("-inf")

    for episode in range(num_episodes):
        log_probs, rewards = run_episode(env, policy)
        returns = compute_returns(rewards, gamma)

        # Standardize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Loss computation
        loss = -(log_probs * returns).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Episode statistics
        episode_reward = sum(rewards)
        episode_length = len(rewards)
        entropy = -torch.mean(log_probs)
        mean_return = returns.mean().item()
        std_return = returns.std().item()

        # Moving average
        running_reward = 0.05 * episode_reward + 0.95 * running_rewards[-1]
        running_rewards.append(running_reward)

        # Logging to Weights & Biases
        wandb.log(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "running_avg_reward": running_reward,
                "loss": loss.item(),
                "entropy": entropy.item(),
                "mean_return": mean_return,
                "std_return": std_return,
            }
        )

        # Checkpoint every 100 episodes
        if episode % 100 == 0:
            torch.save(policy.state_dict(), f"{checkpoint_dir}/policy_ep{episode}.pt")

        # Save best model
        if running_reward > best_running_reward:
            best_running_reward = running_reward
            torch.save(policy.state_dict(), f"{checkpoint_dir}/best_policy.pt")

        if episode % 100 == 0:
            print(
                f"Episode {episode} | Reward: {episode_reward:.2f} | Running Avg: {running_reward:.2f}"
            )

    return running_rewards


def main():
    seed = 2112
    torch.manual_seed(seed)
    np.random.seed(seed)

    render_env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = render_env.observation_space.shape[0]
    n_actions = render_env.action_space.n
    render_env.close()

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    policy = PolicyNet(obs_dim, n_actions).to(device)

    pygame.display.init()
    rewards = reinforce(policy, env, render_env=None, num_episodes=1000)
    pygame.display.quit()

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Running Reward")
    plt.title("REINFORCE Training on CartPole")
    plt.show()

    render_env = gym.make("CartPole-v1", render_mode="human")
    for _ in range(10):
        run_episode(render_env, policy)
    render_env.close()


if __name__ == "__main__":
    main()
