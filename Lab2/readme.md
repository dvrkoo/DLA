# RL-Algos: REINFORCE & PPO on CartPole & LunarLander

A lightweight reinforcement‑learning pipeline implementing **REINFORCE** and **PPO** on OpenAI Gym environments.  
Out of the box it supports **CartPole‑v1** and **LunarLander‑v2**, with detailed logging (Weights & Biases), GIF creation, and easy CLI control.

---

## 🚀 Features

- **Algorithms**: REINFORCE, PPO
- **Environments**: CartPole‑v1 (discrete), LunarLander‑v2 (discrete)
- **Deterministic vs. stochastic** evaluation (`--deterministic` flag)
- **Temperature** control for exploration
- **Weights & Biases** logging (config, rewards, losses, eval metrics)
- **GIF utility** to record rollouts
- **Bash script** for batch experiments

---

## 📋 Requirements

- Python 3.8+
- PyTorch
- gymnasium
- matplotlib
- wandb
- imageio
- pygame

Install via:

```bash
pip install torch gymnasium matplotlib wandb imageio pygame

```

## 🛠 Usage

python main.py \
 --algo <reinforce|ppo> \
 --env <CartPole-v1|LunarLander-v2> \
 --episodes 1000 \
 --gamma 0.99 \
 --lr 1e-2 \
 --temperature 1.0 \
 [--deterministic] \
 --log_interval 50 \
 --gif path/to/output.gif

## 🔧 CLI Arguments

```bash
--algo           <str>   (default: reinforce)      # Algorithm to use: “reinforce” or “ppo”
--env            <str>   (default: CartPole-v1)    # Gym environment ID: CartPole-v1 or LunarLander-v2
--episodes       <int>   (default: 1000)            # Number of training episodes
--gamma          <float> (default: 0.99)            # Discount factor γ
--lr             <float> (default: 1e-2)            # Learning rate
--temperature    <float> (default: 1.0)             # Exploration temperature
--deterministic  (flag)  (default: False)           # If set, use argmax actions during evaluation
--log_interval   <int>   (default: 50)              # Episodes between logs & console output
--ppo_epochs     <int>   (default: 4)               # Number of PPO epochs per update (PPO only)
--clip           <float> (default: 0.2)             # Clipping parameter ε (PPO only)
--gif            <str>   (default: policy.gif)      # File path for output GIF

```

# Results

## Results on CartPole-v1

![](gifs/cartpole.png)
All the experiments are available at: https://wandb.ai/niccolo-marini-universit-degli-studi-di-firenze/rl-ablation-study-CartPole-v1?nw=nwuserniccolomarini

### Observations

- PPO Dominates in Sample Efficiency: The most striking result is the speed at which all PPO variants solve the environment. The PPO | (Baseline) agent reaches the maximum score of 500 by the second evaluation step (~100 episodes) and stays there. This demonstrates that the core actor-critic structure with GAE is incredibly effective for a simple task like CartPole.

- REINFORCE Shows Clear Benefit from Improvements: Unlike the PPO variants, the REINFORCE runs clearly demonstrate the value of each improvement.

  - The REINFORCE | (Baseline) struggles, learning very slowly and never reliably solving the environment.

  - Adding + NormAdv provides a massive boost, allowing the agent to learn much faster and achieve higher scores.

  - The fully-equipped REINFORCE | + NormAdv | + GradClip | + Entropy performs the best of the three, converging faster and more stably towards the maximum score. This is a perfect illustration of how modern improvements can make a classic (but unstable) algorithm viable.

- CartPole is Too Easy for PPO Ablation: Because even the baseline PPO solves the task almost instantly, it's difficult to meaningfully compare the different PPO improvements on this environment. They all hit the performance ceiling immediately. The true benefit of features like Value Clipping and a tuned Entropy bonus would be more visible on a harder task where stability and exploration are more critical.

- Conclusion: The results confirm that PPO is a significantly more powerful and sample-efficient algorithm than REINFORCE. For REINFORCE, the ablation study successfully shows that improvements like advantage normalization are not just helpful but essential for stable learning. For PPO, CartPole serves as a "sanity check" that the implementation is correct, but it is not complex enough to differentiate between the advanced PPO configurations.

![plots](gifs/cartpole.gif)

## Results on LunarLander-v3

![plots](gifs/lunar.png)

All the experiments are available at: https://wandb.ai/niccolo-marini-universit-degli-studi-di-firenze/rl-ablation-study-LunarLander-v3?nw=nwuserniccolomarini

### Observations

- REINFORCE Fails Completely: As predicted, the fully-equipped REINFORCE agent is unable to learn a viable policy. Its evaluation reward starts negative (around -179) and never achieves a positive score. The agent consistently crashes, demonstrating that even with improvements like advantage normalization and gradient clipping, the high variance of the REINFORCE algorithm prevents it from solving this complex environment. This run serves as an excellent baseline showing the algorithm's limitations.

- PPO Can Learn, but Stability is Key: All PPO variants show a clear ability to learn, quickly moving from large negative scores to positive ones. However, the ablation study reveals a clear hierarchy in performance and stability:

  - Baseline PPO (PPO | (Baseline)) is highly unstable. While it shows flashes of high performance (reaching +244 at step 25), it's erratic and cannot maintain a good policy, often dipping back down to lower scores.

  - Adding Advantage Normalization (PPO | + NormAdv) provides a massive improvement. This agent learns much more stably and is the first to consistently solve the environment, reaching over +200 and staying there in the latter half of training. This highlights normalize_advantages as the single most critical improvement for PPO on this task.

- The Full Suite of Improvements Provides the Most Robust Learning: The agent with all features enabled (PPO | + NormAdv | + GradClip | + ClipVal | + Entropy) achieves the best overall performance. It reaches a score over +200 by step 4 and maintains a high average reward throughout the training process. The added stability from gradient/value clipping and the exploration encouraged by the entropy bonus help it find and refine a good policy faster and more reliably than the other variants.

- Conclusion: The results from the LunarLander-v2 experiments are definitive. They confirm that PPO is vastly superior to REINFORCE for complex tasks. Furthermore, the ablation study within PPO shows that while the core algorithm works, advantage normalization is the critical component for stable learning, with additional improvements like clipping and entropy regularization contributing to even faster and more robust convergence.
