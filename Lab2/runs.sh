#!/bin/bash

# =================================================================================
# Comprehensive RL Ablation Study
#
# This script runs 6 experiments on CartPole-v1 and 6 on LunarLander-v2
# to systematically evaluate the impact of different algorithmic components.
# =================================================================================

# --- Configuration ---
SEED=42
CARTPOLE_EPISODES=1000     # A few more episodes to ensure convergence
LUNARLANDER_EPISODES=2000

echo "===== STARTING COMPREHENSIVE RL EXPERIMENT SUITE ====="
echo "SEED: $SEED"
echo "CartPole Episodes: $CARTPOLE_EPISODES"
echo "LunarLander Episodes: $LUNARLANDER_EPISODES"
echo "======================================================"

#
# # =================================================================================
# # Environment 1: CartPole-v1 (6 Experiments)
# # Comparing REINFORCE and PPO and their key improvements.
# # =================================================================================
# echo ""
# echo "--- Starting experiments on CartPole-v1 ($CARTPOLE_EPISODES episodes) ---"
#
# # --- REINFORCE Ablation (3 experiments) ---
# echo ""
# echo "--- Running REINFORCE Ablation on CartPole-v1 ---"
# # 1. REINFORCE Baseline: The simplest version. High variance is expected.
# echo "Running (1/6): REINFORCE | (Baseline)"
# python main.py --env CartPole-v1 --algo reinforce --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.0 --entropy_coef 0.0
#
# # 2. REINFORCE + Advantage Normalization: The most critical improvement for variance reduction.
# echo "Running (2/6): REINFORCE | + NormAdv"
# python main.py --env CartPole-v1 --algo reinforce --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.0 --entropy_coef 0.0 --normalize_advantages
#
# # 3. REINFORCE + All Improvements: Adding everything to give it the best chance.
# echo "Running (3/6): REINFORCE | + NormAdv | + GradClip | + Entropy"
# python main.py --env CartPole-v1 --algo reinforce --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.5 --entropy_coef 0.01 --normalize_advantages
#
#
# # --- PPO Ablation (3 experiments) ---
# echo ""
# echo "--- Running PPO Ablation on CartPole-v1 ---"
# # 4. PPO Baseline: Includes GAE by default, but without other improvements. Can be unstable.
# echo "Running (4/6): PPO | (Baseline)"
# python main.py --env CartPole-v1 --algo ppo --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.0 --entropy_coef 0.0
#
# # 5. PPO + Advantage Normalization: Adding the key stability improvement.
# echo "Running (5/6): PPO | + NormAdv"
# python main.py --env CartPole-v1 --algo ppo --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.0 --entropy_coef 0.0 --normalize_advantages
#
# # 6. PPO + All Improvements: The full suite for robust performance.
# echo "Running (6/6): PPO | + NormAdv | + GradClip | + ClipVal | + Entropy"
# python main.py --env CartPole-v1 --algo ppo --episodes $CARTPOLE_EPISODES --seed $SEED --grad_clip 0.5 --entropy_coef 0.01 --normalize_advantages --clip_value
#

# =================================================================================
# Environment 2: LunarLander-v2 (6 Experiments)
# Demonstrating REINFORCE's failure and a deep dive into PPO's components.
# ALL runs use a larger network (--hidden_size 256) and tuned core parameters.
# =================================================================================
echo ""
echo "--- Starting experiments on LunarLander-v2 ($LUNARLANDER_EPISODES episodes) ---"

# --- Baseline and PPO Ablation (6 experiments) ---
# 1. REINFORCE "Best Effort": Serves as a crucial baseline to show why PPO is needed.
echo "Running (1/6): REINFORCE | + NormAdv | + GradClip | + Entropy (on LunarLander)"
python main.py \
    --env LunarLander-v3 \
    --algo reinforce \
    --episodes $LUNARLANDER_EPISODES \
    --seed $SEED \
    --hidden_size 256 \
    --num_layers 2 \
    --lr 1e-4 \
    --gamma 0.99 \
    --normalize_advantages \
    --grad_clip 1.0 \
    --entropy_coef 0.01

# 2. PPO Baseline: Tuned core parameters (network, lr, gamma) but no "add-on" improvements.
echo "Running (2/6): PPO | (Baseline) (on LunarLander)"
python main.py \
    --env LunarLander-v3 --algo ppo --episodes $LUNARLANDER_EPISODES --seed $SEED \
    --hidden_size 256 --num_layers 2 --lr 3e-4 --gamma 0.99 --ppo_epochs 10 \
    --grad_clip 0.0 --entropy_coef 0.0

# 3. PPO + Advantage Normalization: Adding the most important stabilizer.
echo "Running (3/6): PPO | + NormAdv (on LunarLander)"
python main.py \
    --env LunarLander-v3 --algo ppo --episodes $LUNARLANDER_EPISODES --seed $SEED \
    --hidden_size 256 --num_layers 2 --lr 3e-4 --gamma 0.99 --ppo_epochs 10 \
    --grad_clip 0.0 --entropy_coef 0.0 --normalize_advantages

# 4. PPO + ... + Gradient Clipping: Adding gradient clipping to prevent large, destructive updates.
echo "Running (4/6): PPO | + NormAdv | + GradClip (on LunarLander)"
python main.py \
    --env LunarLander-v3 --algo ppo --episodes $LUNARLANDER_EPISODES --seed $SEED \
    --hidden_size 256 --num_layers 2 --lr 3e-4 --gamma 0.99 --ppo_epochs 10 \
    --grad_clip 1.0 --entropy_coef 0.0 --normalize_advantages

# 5. PPO + ... + Entropy Bonus: Adding entropy to encourage exploration.
echo "Running (5/6): PPO | + NormAdv | + GradClip | + Entropy (on LunarLander)"
python main.py \
    --env LunarLander-v3 --algo ppo --episodes $LUNARLANDER_EPISODES --seed $SEED \
    --hidden_size 256 --num_layers 2 --lr 3e-4 --gamma 0.99 --ppo_epochs 10 \
    --grad_clip 1.0 --entropy_coef 0.01 --normalize_advantages

# 6. PPO + ... + Value Clipping: The complete, modern PPO implementation.
echo "Running (6/6): PPO | + NormAdv | + GradClip | + ClipVal | + Entropy (on LunarLander)"
python main.py \
    --env LunarLander-v3 --algo ppo --episodes $LUNARLANDER_EPISODES --seed $SEED \
    --hidden_size 256 --num_layers 2 --lr 3e-4 --gamma 0.99 --ppo_epochs 10 \
    --grad_clip 1.0 --entropy_coef 0.01 --normalize_advantages --clip_value


echo ""
echo "===== EXPERIMENT SUITE FINISHED ====="
