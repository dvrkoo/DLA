import gym
import torch
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_gif(env_or_name, policy, gif_path, episodes=1, maxlen=500):
    if isinstance(env_or_name, str):
        env_name = env_or_name
    else:
        env_name = env_or_name.spec.id
    render_env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    for _ in range(episodes):
        obs, _ = render_env.reset()
        done = False
        step = 0
        while not done and step < maxlen:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
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
        print("No frames captured; check render_mode and environment compatibility.")
