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
- Class Activation Maps (CAMs), based on:
  > _Learning Deep Features for Discriminative Localization_ – Zhou et al., CVPR 2016


---

### 🤖 Lab 2 – Adversarial Attacks & OOD Detection

This lab introduces techniques in **Adversarial Machine Learning** and **Out-of-Distribution (OOD) Detection**, including:

- Evaluation metrics and pipelines for OOD detection
- Targeted and untargeted **FGSM attacks**:
  > _Explaining and Harnessing Adversarial Examples_ – Goodfellow et al., 2015
- Augmentation strategies to improve adversarial robustness


---

### 🎯 Lab 3 – Efficient Fine-Tuning for NLP (Sentiment Classification)

This lab demonstrates two approaches to training a sentiment classifier using DistilBERT on the Rotten Tomatoes dataset:

- ✅ **Full fine-tuning** of the DistilBERT model
- ⚡ **Efficient fine-tuning** using LoRA (PEFT):
  > _LoRA: Low-Rank Adaptation of Large Language Models_ – Hu et al., 2021
- Optional logging using **Weights & Biases** and **Comet ML**
- Includes a baseline model using SVM + BERT embeddings

---

## ⚙️ Setup Instructions

### 🧬 Clone the repository

```bash
git clone https://github.com/dvrkoo/DLA.git
cd DLA
