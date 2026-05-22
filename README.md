# 🛡️ FakeGuard AI — Fake News Detection with DistilBERT

A production-quality **Fake News Detection system** powered by **DistilBERT**, featuring a complete NLP pipeline, model explainability with **SHAP & LIME**, a **FastAPI** backend, and a stunning dark-themed frontend.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Interview Talking Points](#interview-talking-points)
- [Tech Stack](#tech-stack)

---

## 🧠 Overview

This project fine-tunes a **DistilBERT** transformer model to classify news articles as **REAL** or **FAKE** using the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (~45,000 articles).

### Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **DistilBERT Fine-Tuning** | Transfer learning with HuggingFace Transformers |
| 📊 **Full NLP Pipeline** | Data loading → cleaning → tokenization → training → evaluation |
| 🔍 **SHAP Explainability** | Game-theory based word-level attributions |
| 🟢 **LIME Explainability** | Local interpretable model-agnostic explanations |
| 🚀 **FastAPI Backend** | RESTful API with auto-generated Swagger docs |
| 🎨 **Premium Frontend** | Stunning dark UI with glassmorphism & animations |
| 📈 **Metrics Dashboard** | Accuracy, Precision, Recall, F1-Score visualization |

---

## 🏗️ Architecture

```
┌─────────────┐     HTTP/JSON      ┌──────────────┐     PyTorch      ┌──────────────┐
│             │ ──────────────────► │              │ ──────────────►  │              │
│   Frontend  │                    │   FastAPI    │                  │  DistilBERT  │
│   (HTML/    │ ◄────────────────  │   Backend    │ ◄──────────────  │    Model     │
│   CSS/JS)   │     Results +      │              │   Predictions    │              │
│             │   Explanations     │  /predict    │                  │  Fine-tuned  │
└─────────────┘                    │  /explain    │     SHAP/LIME    │  on 45K      │
                                   │  /metrics    │ ──────────────►  │  articles    │
                                   └──────────────┘                  └──────────────┘
```

---

## 📁 Project Structure

```
fake-news-detector/
├── backend/
│   ├── data/
│   │   └── raw/                     # Place Fake.csv & True.csv here
│   ├── models/
│   │   └── saved/                   # Fine-tuned model checkpoint
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py                # Hyperparameters & paths
│   │   ├── data_loader.py           # Dataset loading & splitting
│   │   ├── preprocessing.py         # Text cleaning & tokenization
│   │   ├── dataset.py               # PyTorch Dataset class
│   │   ├── model.py                 # DistilBERT classifier wrapper
│   │   ├── train.py                 # Training pipeline (HuggingFace Trainer)
│   │   ├── evaluate.py              # Evaluation & confusion matrix
│   │   ├── predict.py               # Single-text inference
│   │   ├── explainability.py        # SHAP & LIME explanations
│   │   └── api.py                   # FastAPI application
│   ├── notebooks/
│   └── requirements.txt
├── frontend/
│   ├── index.html                   # Main page
│   ├── css/styles.css               # Premium dark theme
│   └── js/app.js                    # API integration & interactivity
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip
- (Optional) NVIDIA GPU with CUDA for faster training

### Step 1: Clone & Install Dependencies

```bash
cd fake-news-detector/backend
pip install -r requirements.txt
```

### Step 2: Download the Dataset

1. Go to [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download the dataset (you'll get `Fake.csv` and `True.csv`)
3. Place both files in `backend/data/raw/`

```
backend/data/raw/
├── Fake.csv    (~23,500 fake news articles)
└── True.csv    (~21,400 real news articles)
```

---

## 🏋️ Training the Model

```bash
cd backend
python -m src.train
```

**What happens during training:**
1. Loads & merges Fake.csv + True.csv (adds label column)
2. Cleans text (removes URLs, HTML tags, special characters)
3. Combines title + text with `[SEP]` token
4. Tokenizes with DistilBERT tokenizer (max 256 tokens)
5. Fine-tunes DistilBERT for 3 epochs with AdamW optimizer
6. Evaluates on held-out test set
7. Saves best model to `models/saved/`

**Expected Results:**
| Metric | Score |
|--------|-------|
| Accuracy | ~99%+ |
| F1-Score | ~99%+ |
| Precision | ~99%+ |
| Recall | ~99%+ |

> 💡 This dataset is relatively "easy" for transformer models — the real value of this project is in the **clean pipeline** and **explainability** features.

---

## ▶️ Running the Application

### Start the Unified FastAPI Server

```bash
cd backend
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

The unified server runs both your backend services and serves your premium frontend homepage:
* 🎨 **Frontend Dashboard:** Navigate to **`http://localhost:8000`** in your browser.
* 📖 **Interactive Swagger API Docs:** Navigate to **`http://localhost:8000/docs`** to test your endpoints live.
* 🩺 **API Health Status:** Navigate to **`http://localhost:8000/api/health`**.

---

## 📡 API Documentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the frontend homepage |
| `/api/health` | GET | Health status check |
| `/predict` | POST | Classify text as FAKE or REAL |
| `/explain` | POST | Classify + return word-level SHAP/LIME explanations |
| `/metrics` | GET | Return model evaluation metrics |

### Example: Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists discover that drinking coffee makes you immortal!"}'
```

**Response:**
```json
{
  "label": "FAKE",
  "confidence": 0.97,
  "probabilities": {"FAKE": 0.97, "REAL": 0.03}
}
```

### Example: Explain

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"text": "The president signed the new trade bill today.", "method": "lime", "num_features": 10}'
```

---

## 🎤 Interview Talking Points

### 1. Why DistilBERT over BERT?
> DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its language understanding. For a binary classification task like fake news detection, this trade-off is highly favorable — we get near-identical accuracy with significantly lower computational cost.

### 2. Why combine title + `[SEP]` + text?
> The title carries strong signal about whether news is fake (sensational headlines are a hallmark). The `[SEP]` token lets the model understand these are separate segments, similar to how BERT handles sentence pairs in NLI tasks.

### 3. Why max_length=256 instead of 512?
> Most news articles convey their key claims in the first ~200 tokens. Using 256 reduces GPU memory by ~50% and doubles training speed, with negligible accuracy loss (<0.5%).

### 4. Why AdamW over standard Adam?
> AdamW decouples weight decay from the gradient update. In standard Adam, weight decay interferes with the adaptive learning rate, leading to suboptimal regularization. AdamW fixes this by applying weight decay directly to the weights.

### 5. Why both SHAP and LIME?
> **SHAP** is grounded in game theory (Shapley values) — it's theoretically consistent but computationally expensive. **LIME** is model-agnostic and faster, but relies on random perturbation and may be less stable. Offering both lets users choose based on their needs: SHAP for rigorous analysis, LIME for quick insights.

### 6. Why FastAPI over Flask?
> FastAPI provides async support, automatic request validation via Pydantic, auto-generated OpenAPI docs, and is 2-3x faster than Flask. It's the modern standard for ML model serving.

### 7. How would you deploy this to production?
> Containerize with Docker → serve with Gunicorn + Uvicorn workers → deploy to AWS ECS/GKE → add Redis for caching SHAP/LIME results → API Gateway for rate limiting → model versioning with MLflow.

### 8. How do you handle model drift?
> Monitor prediction confidence distribution over time. If average confidence drops or the label distribution shifts significantly, trigger retraining on fresh labeled data. Use A/B testing to validate new model versions.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Model** | DistilBERT (HuggingFace Transformers) |
| **Training** | PyTorch + HuggingFace Trainer API |
| **Explainability** | SHAP + LIME |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML5 + CSS3 + Vanilla JavaScript |
| **Data Processing** | Pandas + scikit-learn |
| **Visualization** | Matplotlib + Seaborn |

---

## 📊 Dataset Information

**Source:** [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (ISOT Fake News Dataset)

| Property | Value |
|----------|-------|
| Total Articles | ~44,898 |
| Fake Articles | ~23,502 |
| Real Articles | ~21,417 |
| Columns | title, text, subject, date |
| Real News Subjects | politicsNews, worldnews |
| Fake News Subjects | News, politics |

---

## 📄 License

This project is for educational purposes. The dataset is provided by the University of Victoria's ISOT Research Lab.
