#!/bin/bash

# =================================================================================
# RL Ablation Study Script
#
# This script runs a series of experiments for REINFORCE and PPO on CartPole-v1
# and LunarLander-v2. It starts with a minimal baseline and adds one
# improvement at a time to observe its impact.
#
# How to run:
# 1. Make the script executable: chmod +x run_experiments.sh
# 2. Run the script: ./run_experiments.sh
#
# Check the results in your W&B project dashboard.
# =================================================================================

# --- Common settings ---
CARTPOLE_EPISODES=1000
LUNARLANDER_EPISODES=4000
SEED=42

# =================================================================================
# Environment 1: CartPole-v1 (1000 Episodes)
# =================================================================================
echo "--- Starting experiments on CartPole-v1 ($CARTPOLE_EPISODES episodes) ---"
#
# # --- REINFORCE on CartPole ---
# echo "--- Running REINFORCE on CartPole-v1 ---"
#
# # 1. Pure REINFORCE (no improvements)
# echo "Running: Pure REINFORCE"
# python main.py --env CartPole-v1 --algo reinforce --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.0 --entropy_coef 0.0
#
# # 2. REINFORCE + Advantage Normalization
# echo "Running: REINFORCE + Advantage Normalization"
# python main.py --env CartPole-v1 --algo reinforce --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.0 --entropy_coef 0.0 --normalize_advantages
#
# # 3. REINFORCE + Advantage Normalization + Gradient Clipping
# echo "Running: REINFORCE + Advantage Normalization + Gradient Clipping"
# python main.py --env CartPole-v1 --algo reinforce --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.5 --entropy_coef 0.0 --normalize_advantages
#
# #
# # --- PPO on CartPole ---
# echo "--- Running PPO on CartPole-v1 ---"
#
# # 1. PPO Baseline (no advantage normalization or value clipping)
# echo "Running: PPO Baseline"
# python main.py --env CartPole-v1 --algo ppo --episodes $CARTPOLE_EPISODES --seed $SEED
#
# # 2. PPO + Advantage Normalization
# echo "Running: PPO + Advantage Normalization"
# python main.py --env CartPole-v1 --algo ppo --episodes $CARTPOLE_EPISODES --seed $SEED --normalize_advantages
#
# # 3. PPO + Advantage Normalization + Value Clipping
# echo "Running: PPO + Advantage Normalization + Value Clipping"
# python main.py --env CartPole-v1 --algo ppo --episodes $CARTPOLE_EPISODES --seed $SEED --normalize_advantages --clip_value
#

# =================================================================================
# Environment 2: LunarLander-v2 (4000 Episodes)
# # =================================================================================
# echo ""
# echo "--- Starting experiments on LunarLander-v2 ($LUNARLANDER_EPISODES episodes) ---"
#
# # --- REINFORCE on LunarLander ---
# echo "--- Running REINFORCE on LunarLander-v2 ---"
#
# # 1. Pure REINFORCE (no improvements)
# echo "Running: Pure REINFORCE"
# python main.py --env LunarLander-v3 --algo reinforce --episodes $LUNARLANDER_EPISODES --seed $SEED --grad_clip 0.0 --entropy_coef 0.0
#
# # 2. REINFORCE + Advantage Normalization
# echo "Running: REINFORCE + Advantage Normalization"
# python main.py --env LunarLander-v3 --algo reinforce --episodes $LUNARLANDER_EPISODES --seed $SEED --grad_clip 0.0 --entropy_coef 0.0 --normalize_advantages
#
# # 3. REINFORCE + Advantage Normalization + Gradient Clipping
# echo "Running: REINFORCE + Advantage Normalization + Gradient Clipping"
# python main.py --env LunarLander-v3 --algo reinforce --episodes $LUNARLANDER_EPISODES --seed $SEED --grad_clip 0.5 --entropy_coef 0.0 --normalize_advantages
#

# --- PPO on LunarLander ---
echo "--- Running PPO on LunarLander-v2 ---"
#
# # 1. PPO Baseline (no advantage normalization or value clipping)
# echo "Running: PPO Baseline"
# python main.py --env LunarLander-v3 --algo ppo --episodes $LUNARLANDER_EPISODES --seed $SEED
#
# 2. PPO + Advantage Normalization
echo "Running: PPO + Advantage Normalization"
python main.py --env LunarLander-v3 --algo ppo --episodes $LUNARLANDER_EPISODES --seed $SEED --normalize_advantages

# 3. PPO + Advantage Normalization + Value Clipping
echo "Running: PPO + Advantage Normalization + Value Clipping"
python main.py --env LunarLander-v3 --algo ppo --episodes $LUNARLANDER_EPISODES --seed $SEED --normalize_advantages --clip_value


# =================================================================================
echo ""
echo "All experiments complete! Check your wandb dashboard."
# =================================================================================
