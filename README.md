# 🔥 ThermalFit-AI: Quantitative Mask Fit Analysis

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/Computer_Vision-Radiometric-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/AI-Face_Mesh-orange?style=for-the-badge&logo=google)
![Status](https://img.shields.io/badge/Status-Master's%20Thesis-red?style=for-the-badge)

> **A Next-Gen Computer Vision system that quantifies respiratory mask leaks (1-100 Score) using thermal physics and Google AI.**

## 📸 Project Overview
Standard respiratory fit tests are subjective. This project proposes a **Quantitative (Objective)** method using Thermal Imaging. By analyzing the temperature differential between the mask surface and exhaled breath, this system detects leaks in real-time.

### 🌟 Key Features
* **AI Face Tracking:** Uses **Google MediaPipe** to lock onto facial landmarks.
* **Physics-Based Calibration:** Converts pixel intensity to Temperature ($^\circ C$).
* **Real-Time Dashboard:** Displays a live "Fit Score" (0-100).
* **Dual Modes:** Auto-AI Mode & Manual Calibration Mode.

## 🛠️ Usage
1. **AI Mode:** `python ai_thermal_analysis.py`
2. **Manual Mode:** `python master_thermal_analysis.py`

---
**Developed by Onkar Yadav**