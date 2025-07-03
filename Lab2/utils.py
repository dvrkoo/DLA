import gym
import torch
import imageio
import numpy as np
import re


# Fix numpy compatibility issues
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "bool9"):
    np.bool9 = np.bool_


device = torch.device("cpu")


# ============================
# Utility Functions
# ============================


def compute_gae_returns(rewards, values, gamma=0.99, lam=0.95, device="cpu"):
    """
    Computes the Generalized Advantage Estimation (GAE) and corresponding returns.

    Args:
        rewards (list or np.ndarray): List of rewards for an episode.
        values (torch.Tensor): Tensor of value estimates for each state in the episode.
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter.
        device (torch.device): The device to use for tensors.

    Returns:
        A tuple of (advantages, returns) as torch.Tensors.
    """
    # Ensure values is a flat tensor
    values = values.squeeze().detach()

    # Add a dummy value at the end for the last state. If the episode is done,
    # the value of the terminal state is 0.
    next_values = torch.cat([values[1:], torch.tensor([0.0], device=device)])

    # Convert rewards to a tensor
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

    # Calculate GAE
    advantages = torch.zeros_like(rewards_t)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards_t[t] + gamma * next_values[t] - values[t]
        last_gae_lam = delta + gamma * lam * last_gae_lam
        advantages[t] = last_gae_lam

    # Calculate returns by adding advantages to the value estimates
    returns = advantages + values

    return advantages, returns


def compute_returns(rewards, gamma):
    """
    Compute discounted returns efficiently.

    Args:
        rewards: List or numpy array of rewards
        gamma: Discount factor

    Returns:
        Tensor of discounted returns
    """
    # Convert to numpy array if it's a list
    if isinstance(rewards, list):
        rewards = np.array(rewards)

    # Efficient vectorized implementation
    returns = np.zeros_like(rewards, dtype=np.float32)

    # Compute returns in reverse order
    returns[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        returns[i] = rewards[i] + gamma * returns[i + 1]

    return torch.tensor(returns, dtype=torch.float32, device=device)


def make_gif(env_or_name, policy, gif_path, episodes=1, maxlen=500):
    # Determine the env ID string
    if isinstance(env_or_name, str):
        env_id = env_or_name
    else:
        env_id = env_or_name.spec.id

    # If the user passed something like "LunarLander-v3", strip that off
    m = re.match(r"^(?P<base>.*?)(?:-v\d+)$", env_id)
    if m:
        base_id = m.group("base")
    else:
        base_id = env_id

    # Candidate list: first try full env_id, then base_id + "-v2"
    candidates = [env_id, f"{base_id}-v2"]

    for candidate in candidates:
        try:
            render_env = gym.make(candidate, render_mode="rgb_array")
            if candidate != env_id:
                print(f"Falling back to environment '{candidate}' for GIF.")
            break
        except gym.error.Error:
            render_env = None
    else:
        raise gym.error.NameNotFound(
            f"Could not find a renderable version of '{env_id}'. "
            f"Tried: {candidates}"
        )

    frames = []
    for _ in range(episodes):
        obs, _ = render_env.reset()
        done = False
        step = 0
        while not done and step < maxlen:
            obs_t = torch.tensor(
                obs, dtype=torch.float32, device=next(policy.parameters()).device
            )
            action = torch.argmax(policy(obs_t)).item()
            frame = render_env.render()
            if frame is not None:
                frames.append(frame)
            obs, _, terminated, truncated, _ = render_env.step(action)
            done = bool(terminated or truncated)
            step += 1

    render_env.close()

    if frames:
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved GIF to {gif_path}")
    else:
        print("No frames captured; check render_mode and compatibility.")
