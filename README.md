<h1 align="center">🧠 Emotion Detection</h1>
<p align="center">
  <img src="https://img.shields.io/github/license/altafpatel01/Emotion-Detection" alt="License">
  <img src="https://img.shields.io/github/languages/top/altafpatel01/Emotion-Detection" alt="Top Language">
  <img src="https://img.shields.io/github/last-commit/altafpatel01/Emotion-Detection" alt="Last Commit">
</p>

<p align="center">
  An AI-powered system that detects human emotions from text and speech using machine learning and NLP techniques.
</p>

---

## 🎯 Project Objective

To build an AI-based system capable of **detecting emotions** from **text input** (and optionally, **speech input**) using advanced **machine learning** and **natural language processing (NLP)** techniques. This system can be used in mental health monitoring, customer feedback analysis, and intelligent chatbots.

---

## 📊 Problem Statement

Machines often struggle to understand the emotional tone behind human communication. This project aims to bridge that gap by accurately identifying emotions such as **joy, sadness, anger, fear, surprise**, and more from user input.

---

## ⚙️ Features

- 🔠 Emotion detection from text
- 🎤 Speech-to-text emotion detection using Whisper
- 🧠 Trained ML/DL models (Logistic Regression, LSTM, etc.)
- 📊 Confidence scores for each emotion
- 📈 Visual representation of results
- 🖥️ Optional Streamlit UI for user interaction

---

## 🛠 Tech Stack

| Area            | Technologies Used                                |
|-----------------|--------------------------------------------------|
| Language        | Python                                           |
| NLP             | NLTK, spaCy                                      |
| ML/DL Models    | scikit-learn, TensorFlow, PyTorch                |
| Speech-to-Text  | OpenAI Whisper                                   |
| Dataset         | GoEmotions by Google                             |

---

## 🧠 Solution Approach



> Visual: A simple workflow image showing: **Input → Preprocessing → Model → Prediction → Output UI**

1. **Data Collection**: GoEmotions dataset (58k+ labeled comments)
2. **Preprocessing**: Tokenization, stopword removal, lemmatization
3. **Model Training**: ML and LSTM-based models
4. **Speech Support**: Whisper model converts speech to text
5. **Emotion Classification**: Multi-label emotion prediction
6. **Output Display**: Console + optional Streamlit UI

---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/altafpatel01/Emotion-Detection.git
cd Emotion-Detection
pip install -r requirements.txt
