# 🌲 ForestGuard Pro: Intelligent Wildfire & Smoke Detection System

<div align="center">

![Project Banner](https://img.shields.io/badge/ForestGuard-Pro-green?style=for-the-badge&logo=leaf)

**An Advanced Real-Time Surveillance System for Early Disaster Prevention using Hybrid TransUNet & CBAM Attention.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [System Demo](#-system-demo)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Dataset & Training](#-dataset--training)
- [Team Members](#-team-members)

---

## 🌍 Project Overview
Wildfires are global catastrophes that cause irreversible damage to ecosystems. **ForestGuard Pro** is an AI-powered solution designed to replace slow traditional sensors. It utilizes **Semantic Segmentation** to identify smoke and fire pixels in real-time video feeds, providing accurate early warnings and visualization on a modern web dashboard.

Unlike standard object detection (YOLO), our system creates a **pixel-perfect mask**, allowing authorities to estimate the **burn area** and **spread rate** precisely.

---

## 💻 System Demo
> *Place a screenshot of your dashboard here. Name it `dashboard.png` and put it in the repo.*

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Upload+Your+Dashboard+Screenshot+Here)

*Figure 1: Real-time analysis dashboard showing Fire (Red) and Smoke (Gray) segmentation.*

---

## ✨ Key Features
* **🧠 Hybrid AI Brain:** Combines **ResNet34** (CNN) for feature extraction with **Transformers** for global context awareness.
* **👁️ Smart Attention:** Manually implemented **CBAM (Convolutional Block Attention Module)** to focus on fire textures and ignore complex backgrounds (clouds, fog).
* **⚡ Real-Time Streaming:** Powered by **FastAPI** Asynchronous streaming, processing video frames with low latency.
* **📊 Interactive Dashboard:** A Glassmorphism UI built with HTML/JS/CSS to visualize live analytics (Fire % vs. Smoke %).
* **⚖️ Balanced Training:** Uses advanced **Data Augmentation** techniques to solve class imbalance between Smoke and Fire.

---

## 🏗️ System Architecture
The core model is a custom **SmokeTransUNet**:

1.  **Encoder:** ResNet34 (Pre-trained on ImageNet).
2.  **Bottleneck:** **Transformer Block** (Self-Attention) to capture long-range dependencies.
3.  **Decoder:** U-Net style upsampling with **Skip Connections**.
4.  **Attention Gates:** **CBAM** modules applied at skip connections to refine features.

| Class ID | Label | Color Code |
| :---: | :--- | :--- |
| **0** | Background | Transparent |
| **1** | Smoke 🌫️ | Gray `(128, 128, 128)` |
| **2** | Fire 🔥 | Red `(255, 50, 0)` |

---

## 📂 Project Structure
```text
ForestGuard-Pro/
├── app.py                   # Main FastAPI Server (Backend)
├── train.py                 # Model Training Script
├── requirements.txt         # Project Dependencies
├── README.md                # Project Documentation
├── weights/                 # Trained Models
│   └── smoke_fire_model.pth # Best Model Weights (Acc: ~83%)
├── templates/               # Frontend UI
│   └── index.html           # Dashboard Interface
├── data/                    # Dataset Directory
│   ├── processed/           # Final Training Data
│   └── D-Fire/              # Original Dataset
└── src/                     # Source Code
    ├── dataset.py           # Custom Dataset Loader
    ├── transforms.py        # Data Augmentation Logic
    └── models/              # Deep Learning Architectures
        ├── smoke_net.py     # Hybrid TransUNet Assembly
        ├── attention.py     # CBAM Module Implementation
        └── unet_parts.py    # Decoder Blocks


## 👥 Team Members & Roles

| Name | | Responsibilities |
| :--- | :--- | :--- |
| **[mohamed ahmed abdelazim ]** | Team Leader | model Building & Implementing TransformerBlock ,smoke_net
| **[ِAbdelRahman Mohamed Abdelrahman  ]** | |  & `CBAM` Attention modules.
| ** Mohamed Sameh Farag Mansour | Building `unet_parts` & Training pipeline optimization. |
| ** Saif Hussam Youssef Al-Khalaily | front end & backend (FastAPI)|
| ** yahya Zakaria Mazid | Implementing Data Augmentation scripts to solve imbalance. |
| ** |Mahmoud Abdel Razek Anbar|  | Implementing Data Augmentation scripts to solve imbalance.
