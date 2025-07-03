# 🧠 Deep Learning Applications – Laboratory Assignments

This repository contains the implementation of three laboratory assignments developed for the **Deep Learning Applications** course, part of the MSc in Artificial Intelligence at the **University of Florence**.

Each lab is organized into its own folder and includes code, documentation, and experiments. Dedicated `README.md` files inside each lab provide lab-specific details, but this top-level file offers a general overview and setup instructions.

---

## 📁 Lab Overviews

### 📦 Lab 1 – Deep Networks & Residual Connections

This lab explores deep neural architectures using MLPs and CNNs, with a focus on **residual learning**. It includes:

- Training CNNs on CIFAR-like datasets
- Implementation of **ResNet-style skip connections**, inspired by:
  > _Deep Residual Learning for Image Recognition_ – Kaiming He et al., CVPR 2016
- Exercise 2 focuses on using previous models to perform Knowledge Distillation

---

### 🧠 Lab 2 – Reinforcement Learning with Policy Gradients

This lab explores core ideas in Reinforcement Learning (RL) using Policy Gradient methods, including:

- The REINFORCE algorithm for training stochastic policies aswell as PPO algorithm

- Implementing and visualizing learning in environments like CartPole and LunarLander

- Key concepts: reward signals, return estimation, and gradient-based policy updates

---

### 🎯 Lab 3 – Efficient Fine-Tuning for NLP (Sentiment Classification)

This lab demonstrates two approaches to training a sentiment classifier using DistilBERT on the Rotten Tomatoes and sst2 dataset:

- ✅ **Full fine-tuning** of the DistilBERT model
- ⚡ **Efficient fine-tuning** using LoRA (PEFT):
  > _LoRA: Low-Rank Adaptation of Large Language Models_ – Hu et al., 2021
- Includes a baseline model using SVM + BERT embeddings

---

## ⚙️ Setup Instructions

### 🧬 Clone the repository

```bash
git clone https://github.com/dvrkoo/DLA.git
cd DLA
pip install -r requirements.txt
```
