import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import wandb
from utils import make_gif, compute_returns
from models import PolicyNet, ValueNet

# Monkey-patch for numpy.bool8 compatibility
np.bool8 = np.bool_

# Device config
device = torch.device("cpu")


def evaluate_policy(
    env, policy, episodes=5, maxlen=500, device="cpu", deterministic=False
):
    policy.eval()
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_r = 0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                probs = policy(obs_t)
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            done = bool(terminated or truncated)
        rewards.append(total_r)
    policy.train()
    return np.mean(rewards), np.std(rewards)


# ============================
# REINFORCE
# ============================
def train_reinforce(env, eval_env, policy, args):
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rewards_history = []

    for ep in range(args.episodes):
        # 1) Rollout (always stochastic for training)
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

        # 2) Compute returns & policy gradient update
        returns = compute_returns(rewards, args.gamma)
        total_reward = sum(rewards)
        rewards_history.append(total_reward)

        # Standardize and compute loss
        log_probs = torch.stack(log_probs)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3) Logging & periodic evaluation
        if ep % args.log_interval == 0:
            # Training metric
            wandb.log(
                {
                    "episode": ep,
                    "train_reward": total_reward,
                    "policy_loss": loss.item(),
                    "episode_length": len(rewards),
                }
            )

            # Deterministic or stochastic evaluation
            eval_mean, eval_std = evaluate_policy(
                eval_env,
                policy,
                episodes=5,
                maxlen=500,
                device=device,
                deterministic=args.deterministic,
            )
            wandb.log(
                {
                    "episode": ep,
                    "eval_mean_reward": eval_mean,
                    "eval_std_reward": eval_std,
                    "deterministic": args.deterministic,
                }
            )

            print(
                f"REINFORCE Ep {ep}: "
                f"Train {total_reward:.2f}, "
                f"Eval {eval_mean:.2f}±{eval_std:.2f}"
            )

    return rewards_history


# ============================
# PPO
# ============================
def train_ppo(env, eval_env, policy, value_net, args):
    p_optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    v_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
    history = []

    for ep in range(args.episodes):
        # 1) Rollout
        obs, _ = env.reset(seed=args.seed)
        memories = []  # list of (obs_t, action, old_logp, reward, value)
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=args.device)
            probs = policy(obs_t)
            dist = Categorical(probs / args.temperature)  # Recompute for value
            action = dist.sample()
            logp = dist.log_prob(action)
            value = value_net(obs_t)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = bool(terminated or truncated)
            # Store detached values to avoid gradient issues
            memories.append(
                (obs_t.detach(), action.detach(), logp.detach(), reward, value.detach())
            )

        # 2) Compute returns & advantages
        rewards = [m[3] for m in memories]
        returns = compute_returns(rewards, args.gamma).detach()
        values = torch.stack([m[4] for m in memories])
        advantages = returns - values

        # 3) PPO update: accumulate losses, then do one backward per network
        for _ in range(args.ppo_epochs):
            policy_losses = []
            value_losses = []

            for idx, (obs_t, action, old_logp, _, _) in enumerate(memories):
                # Re-compute distribution (no in-place)
                probs = policy(obs_t)
                dist = Categorical(probs / args.temperature)
                new_logp = dist.log_prob(action)

                # Ratio & clipped surrogate
                ratio = (new_logp - old_logp).exp()
                adv = advantages[idx].detach()  # Detach advantage

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv
                policy_losses.append(-torch.min(surr1, surr2))

                # Value loss
                value_losses.append(F.mse_loss(value_net(obs_t), returns[idx]))

            # Policy network update
            p_optimizer.zero_grad()
            policy_loss = torch.stack(policy_losses).mean()
            policy_loss.backward()
            p_optimizer.step()

            # Value network update
            v_optimizer.zero_grad()
            value_loss = torch.stack(value_losses).mean()
            value_loss.backward()
            v_optimizer.step()

        episode_reward = sum(rewards)
        history.append(episode_reward)

        if ep % args.log_interval == 0:
            wandb.log(
                {
                    "episode": ep,
                    "ppo_reward": episode_reward,
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                }
            )
            print(f"PPO Ep {ep}: Reward {episode_reward:.2f}")
            # evaluate the policy
            eval_mean, eval_std = evaluate_policy(
                eval_env,
                policy,
                episodes=5,
                maxlen=500,
                device=args.device,
                deterministic=args.deterministic,
            )
            wandb.log(
                {
                    "train_reward": episode_reward,
                    "eval_mean_reward": eval_mean,
                    "eval_std_reward": eval_std,
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "episode": ep,
                    "episode_length": len(rewards),
                }
            )
            print(f"PPo Ep {ep}: Eval Mean {eval_mean:.2f} ± {eval_std:.2f}")

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
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gif", type=str, default="policy.gif")
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic policy (argmax)"
    )

    args = parser.parse_args()

    wandb.init(
        project=f"rl-algos-{args.env}",
        name=f"reinforce_{args.env}_{args.algo}_temperature{args.temperature}_lr{args.lr}_gamma{args.gamma}_deterministic{args.deterministic}",
        group="deterministic" if args.deterministic else "stochastic",
        config=vars(args),
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyNet(obs_dim, n_actions).to(device)

    if args.algo == "ppo":
        value_net = ValueNet(obs_dim).to(device)
        history = train_ppo(env, eval_env, policy, value_net, args)
    else:
        history = train_reinforce(env, eval_env, policy, args)

    # plot learning curve
    plt.plot(history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{args.algo.upper()} on {args.env}")
    plt.show()

    # generate gif
    make_gif(env, policy, args.gif)
    eval_env.close()
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
