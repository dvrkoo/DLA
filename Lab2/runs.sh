#!/bin/bash

# Set up virtual environment activation if needed
# source .venv/bin/activate

# Worst-case: Few episodes, high temperature (more randomness), low gamma (no future reward)
echo "Running REINFORCE on CartPole-v1 (worst-case)"
python main.py --algo reinforce --env CartPole-v1 --episodes 200 --temperature 2.0 --gamma 0.5 --gif gifs/reinforce_cartpole_worst.gif

# Mid-case: Moderate settings
echo "Running REINFORCE on CartPole-v1 (mid-case)"
python main.py --algo reinforce --env CartPole-v1 --episodes 500 --temperature 1.0 --gamma 0.9 --gif gifs/reinforce_cartpole_mid.gif

# Best-case: Optimal tuning
echo "Running REINFORCE on CartPole-v1 (best-case)"
python main.py --algo reinforce --env CartPole-v1 --episodes 1000 --temperature 0.8 --gamma 0.99 --gif gifs/reinforce_cartpole_best.gif

# PPO worst-case
echo "Running PPO on CartPole-v1 (worst-case)"
python main.py --algo ppo --env CartPole-v1 --episodes 200 --temperature 2.0 --gamma 0.5 --gif gifs/ppo_cartpole_worst.gif

# PPO best-case
echo "Running PPO on CartPole-v1 (best-case)"
python main.py --algo ppo --env CartPole-v1 --episodes 1000 --temperature 1.0 --gamma 0.99 --gif gifs/ppo_cartpole_best.gif

# Generalization: Different datasets/environments

# MountainCar-v0: harder task
echo "Running PPO on MountainCar-v0"
python main.py --algo ppo --env MountainCar-v0 --episodes 1000 --temperature 1.0 --gamma 0.99 --gif gifs/ppo_mountaincar.gif

# Acrobot-v1: different dynamics
echo "Running PPO on Acrobot-v1"
python main.py --algo ppo --env Acrobot-v1 --episodes 1000 --temperature 1.0 --gamma 0.99 --gif gifs/ppo_acrobot.gif

# REINFORCE on MountainCar-v0 (likely to fail)
echo "Running REINFORCE on MountainCar-v0"
python main.py --algo reinforce --env MountainCar-v0 --episodes 1000 --temperature 1.0 --gamma 0.99 --gif gifs/reinforce_mountaincar.gif

