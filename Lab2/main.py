import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import wandb
from utils import make_gif, compute_returns
from models import PolicyNet, ValueNet, EnhancedPolicyNet, EnhancedValueNet

# Fix numpy compatibility issues
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "bool9"):
    np.bool9 = np.bool_

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
        steps = 0
        while not done and steps < maxlen:
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
            steps += 1
        rewards.append(total_r)
    policy.train()
    return np.mean(rewards), np.std(rewards)


# ============================
# REINFORCE with improvements
# ============================
def train_reinforce(env, eval_env, policy, args):
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )
    rewards_history = []
    best_reward = float("-inf")

    for ep in range(args.episodes):
        # Rollout (always stochastic for training)
        obs, _ = env.reset(seed=args.seed + ep)  # Different seed per episode
        log_probs, rewards, entropies = [], [], []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            probs = policy(obs_t)
            dist = Categorical(probs / args.temperature)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = bool(terminated or truncated)

        # Compute returns & policy gradient update
        returns = compute_returns(rewards, args.gamma)
        total_reward = sum(rewards)
        rewards_history.append(total_reward)

        # Update best reward
        if total_reward > best_reward:
            best_reward = total_reward

        # Policy loss with entropy regularization
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Advantage normalization (optional baseline)
        if args.normalize_advantages:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_loss = -(log_probs * returns_t).mean()
        entropy_loss = -entropies.mean()
        total_loss = policy_loss + args.entropy_coef * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

        optimizer.step()

        # Learning rate scheduling
        if args.use_lr_scheduler:
            scheduler.step()

        # Logging & periodic evaluation
        if ep % args.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]

            # Evaluation
            eval_mean, eval_std = evaluate_policy(
                eval_env,
                policy,
                episodes=args.eval_episodes,
                maxlen=500,
                device=device,
                deterministic=args.deterministic,
            )
            log_data = {
                "episode": ep,
                "train_reward": total_reward,
                "best_reward": best_reward,
                "policy_loss": policy_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "total_loss": total_loss.item(),
                "episode_length": len(rewards),
                "learning_rate": current_lr,
                "eval_mean_reward": eval_mean,
            }

            wandb.log(log_data)

            print(
                f"REINFORCE Ep {ep}: "
                f"Train {total_reward:.2f}, "
                f"Eval {eval_mean:.2f}±{eval_std:.2f}, "
                f"LR {current_lr:.2e}"
            )

    return rewards_history


# ============================
# PPO with improvements
# ============================
def train_ppo(env, eval_env, policy, value_net, args):
    p_optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    v_optimizer = torch.optim.Adam(
        value_net.parameters(), lr=args.v_lr, weight_decay=args.weight_decay
    )

    p_scheduler = torch.optim.lr_scheduler.StepLR(
        p_optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )
    v_scheduler = torch.optim.lr_scheduler.StepLR(
        v_optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    history = []
    best_reward = float("-inf")

    for ep in range(args.episodes):
        # Rollout - collect trajectories
        obs, _ = env.reset(seed=args.seed + ep)
        states, actions, old_logprobs, rewards, values, entropies = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            # Forward pass through both networks
            with torch.no_grad():  # No gradients needed during rollout
                probs = policy(obs_t)
                value = value_net(obs_t)

            dist = Categorical(probs / args.temperature)
            action = dist.sample()

            # Store everything we need
            states.append(obs_t)
            actions.append(action)
            old_logprobs.append(dist.log_prob(action))
            values.append(value)
            entropies.append(dist.entropy())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = bool(terminated or truncated)

        # Convert to tensors for batch processing
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_logprobs = torch.stack(old_logprobs)
        values = torch.stack(values)
        old_entropies = torch.stack(entropies)

        # Compute returns & advantages
        returns = compute_returns(rewards, args.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = returns_t - values.squeeze()

        # Normalize advantages
        if args.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        episode_reward = sum(rewards)
        history.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward

        # PPO update with batched operations
        for ppo_ep in range(args.ppo_epochs):
            # Batch forward pass through policy network
            probs = policy(states)
            dist = Categorical(probs / args.temperature)
            new_logprobs = dist.log_prob(actions)
            new_entropies = dist.entropy()

            # Batch forward pass through value network
            new_values = value_net(states).squeeze()

            # Compute all losses at once
            ratios = (new_logprobs - old_logprobs).exp()

            # Policy loss with clipping (vectorized)
            surr1 = ratios * advantages.detach()
            surr2 = (
                torch.clamp(ratios, 1 - args.clip, 1 + args.clip) * advantages.detach()
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy loss
            entropy_loss = -new_entropies.mean()

            # Value loss with optional clipping (vectorized)
            if args.clip_value:
                value_clipped = values.squeeze() + torch.clamp(
                    new_values - values.squeeze(), -args.clip, args.clip
                )
                value_loss1 = F.mse_loss(new_values, returns_t)
                value_loss2 = F.mse_loss(value_clipped, returns_t)
                value_loss = torch.max(value_loss1, value_loss2)
            else:
                value_loss = F.mse_loss(new_values, returns_t)

            # Policy network update
            p_optimizer.zero_grad()
            total_policy_loss = policy_loss + args.entropy_coef * entropy_loss
            total_policy_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            p_optimizer.step()

            # Value network update
            v_optimizer.zero_grad()
            value_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), args.grad_clip)
            v_optimizer.step()

        # Learning rate scheduling
        if args.use_lr_scheduler:
            p_scheduler.step()
            v_scheduler.step()

        # Logging
        if ep % args.log_interval == 0:
            current_lr = p_optimizer.param_groups[0]["lr"]
            current_v_lr = v_optimizer.param_groups[0]["lr"]

            # Evaluation
            eval_mean, eval_std = evaluate_policy(
                eval_env,
                policy,
                episodes=args.eval_episodes,
                maxlen=500,
                device=device,
                deterministic=args.deterministic,
            )

            log_data = {
                "episode": ep,
                "train_reward": episode_reward,
                "best_reward": best_reward,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "episode_length": len(rewards),
                "learning_rate": current_lr,
                "value_learning_rate": current_v_lr,
                "eval_mean_reward": eval_mean,
            }

            wandb.log(log_data)
            print(
                f"PPO Ep {ep}: Train {episode_reward:.2f}, Eval {eval_mean:.2f}±{eval_std:.2f}"
            )

    return history


# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser(description="Enhanced RL Trainer")

    # Environment and algorithm
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument(
        "--algo", type=str, choices=["reinforce", "ppo"], default="reinforce"
    )

    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--v_lr", type=float, default=1e-3, help="Value network learning rate"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # Network architecture
    parser.add_argument(
        "--enhanced", action="store_true", help="Use enhanced network architectures"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Hidden layer size for enhanced networks",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of hidden layers for enhanced networks",
    )

    # Optimization improvements
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.5,
        help="Gradient clipping threshold (0 to disable)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="L2 regularization"
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="Entropy regularization coefficient",
    )
    parser.add_argument(
        "--normalize_advantages", action="store_true", help="Normalize advantages"
    )

    # Learning rate scheduling
    parser.add_argument(
        "--use_lr_scheduler", action="store_true", help="Use learning rate scheduler"
    )
    parser.add_argument(
        "--lr_step", type=int, default=200, help="LR scheduler step size"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.9, help="LR scheduler gamma"
    )

    # PPO specific
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument(
        "--clip_value", action="store_true", help="Enable value function clipping"
    )

    # Evaluation and logging
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--eval_episodes", type=int, default=5, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy for evaluation",
    )

    # Output
    parser.add_argument("--gif", type=str, default="policy.gif")

    args = parser.parse_args()
    args.device = device

    # Initialize wandb
    run_group = f"{args.algo.upper()}-{args.env}"

    # 2. Build the run name piece by piece to highlight the "add-on" features.
    #    This makes it easy to see what changed between runs.
    name_parts = [args.algo.upper()]
    if args.normalize_advantages:
        name_parts.append("+ NormAdv")
    if args.grad_clip > 0:
        name_parts.append("+ GradClip")
    # PPO-specific improvements
    if args.algo == "ppo" and args.clip_value:
        name_parts.append("+ ClipVal")
    # General improvements
    if args.entropy_coef > 0:
        name_parts.append("+ Entropy")
    if args.enhanced:
        name_parts.append("+ EnhancedNet")

    # If no improvements are added, mark it as the baseline.
    if len(name_parts) == 1:
        name_parts.append("(Baseline)")

    run_name = " | ".join(name_parts)

    # 3. Initialize wandb with the new, structured names.
    #    We also change the project name to be more specific to this study.
    wandb.init(
        project=f"rl-ablation-study-{args.env}",
        name=run_name,
        group=run_group,
        config=vars(args),
    )
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environments
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create networks
    if args.enhanced:
        policy = EnhancedPolicyNet(
            obs_dim, n_actions, args.hidden_size, args.num_layers
        ).to(device)
        if args.algo == "ppo":
            value_net = EnhancedValueNet(obs_dim, args.hidden_size, args.num_layers).to(
                device
            )
    else:
        policy = PolicyNet(obs_dim, n_actions).to(device)
        if args.algo == "ppo":
            value_net = ValueNet(obs_dim).to(device)

    print(f"Using {'enhanced' if args.enhanced else 'basic'} networks on {device}")
    print(f"Policy network parameters: {sum(p.numel() for p in policy.parameters())}")
    if args.algo == "ppo":
        print(
            f"Value network parameters: {sum(p.numel() for p in value_net.parameters())}"
        )

    # Train
    if args.algo == "ppo":
        history = train_ppo(env, eval_env, policy, value_net, args)
    else:
        history = train_reinforce(env, eval_env, policy, args)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(
        f"{args.algo.upper()} on {args.env} ({'Enhanced' if args.enhanced else 'Basic'})"
    )
    plt.grid(True)
    plt.show()

    # Generate GIF
    make_gif(eval_env, policy, args.gif)

    # Cleanup
    eval_env.close()
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
