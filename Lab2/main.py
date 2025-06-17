import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import wandb
from utils import make_gif

# Monkey-patch for numpy.bool8 compatibility
np.bool8 = np.bool_

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# Policy and Value Networks
# ============================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


# ============================
# Utility Functions
# ============================
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)


# ============================
# REINFORCE
# ============================
def train_reinforce(env, policy, args):
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rewards_history = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed)
        log_probs, rewards = [], []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            dist = Categorical(policy(obs_t) / args.temperature)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = bool(terminated or truncated)

        returns = compute_returns(rewards, args.gamma)
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * (returns - returns.mean()) / (returns.std() + 1e-9)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        if ep % args.log_interval == 0:
            wandb.log({"reinforce_reward": total_reward, "episode": ep})
            print(f"REINFORCE Ep {ep}: Reward {total_reward:.2f}")
    return rewards_history


# ============================
# PPO
# ============================
def train_ppo(env, policy, value_net, args):
    p_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    v_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
    history = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed)
        memories = []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            dist = Categorical(policy(obs_t) / args.temperature)
            action = dist.sample()
            logp = dist.log_prob(action)
            value = value_net(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            memories.append((obs_t, action, logp, reward, value))
            done = bool(terminated or truncated)
        # compute returns and advantages
        rewards = [m[3] for m in memories]
        returns = compute_returns(rewards, args.gamma)
        values = torch.stack([m[4] for m in memories])
        advantages = returns - values.detach()
        # PPO update
        for _ in range(args.ppo_epochs):
            for idx, (obs_t, action, old_logp, _, _) in enumerate(memories):
                dist = Categorical(policy(obs_t) / args.temperature)
                logp = dist.log_prob(action)
                ratio = (logp - old_logp).exp()
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
                p_loss = -torch.min(surr1, surr2)
                v_loss = F.mse_loss(value_net(obs_t), returns[idx])
                p_optimizer.zero_grad()
                p_loss.backward()
                p_optimizer.step()
                v_optimizer.zero_grad()
                v_loss.backward()
                v_optimizer.step()
        episode_reward = sum(rewards)
        history.append(episode_reward)
        if ep % args.log_interval == 0:
            wandb.log({"ppo_reward": episode_reward, "episode": ep})
            print(f"PPO Ep {ep}: Reward {episode_reward:.2f}")
    return history


# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser(description="RL Trainer")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument(
        "--algo", type=str, choices=["reinforce", "ppo"], default="reinforce"
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2112)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gif", type=str, default="policy.gif")
    args = parser.parse_args()

    wandb.init(
        project="rl-algos",
        name=f"reinforce_{args.env}_{args.algo}_temperature{args.temperature}_lr{args.lr}_gamma{args.gamma}",
        config=vars(args),
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(obs_dim, n_actions).to(device)

    if args.algo == "ppo":
        value_net = ValueNet(obs_dim).to(device)
        history = train_ppo(env, policy, value_net, args)
    else:
        history = train_reinforce(env, policy, args)

    # plot learning curve
    plt.plot(history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{args.algo.upper()} on {args.env}")
    plt.show()

    # generate gif
    make_gif(env, policy, args.gif)


if __name__ == "__main__":
    main()
