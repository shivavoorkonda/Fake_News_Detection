"""
config.py — Central Configuration for Fake News Detection Pipeline

This module centralizes all configurable parameters (paths, hyperparameters,
label mappings) in one place, following the single-source-of-truth principle.
This avoids magic numbers scattered across the codebase and makes it easy to
tweak experiments without hunting through multiple files.

Interview Talking Point:
    Centralizing config promotes reproducibility. Anyone can clone this repo,
    inspect config.py, and know exactly how the model was trained — no guessing.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────────────────────────────────────

# BASE_DIR points to backend/, computed relative to this file's location.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Raw CSV data lives in data/raw/; processed artifacts could go in data/processed/.
DATA_DIR: Path = BASE_DIR / 'data'
RAW_DATA_DIR: Path = DATA_DIR / 'raw'

# Trained model weights and tokenizer are persisted under models/saved/.
MODEL_DIR: Path = BASE_DIR / 'models' / 'saved'

# Evaluation metrics (JSON, confusion matrix images) go in models/.
METRICS_DIR: Path = BASE_DIR / 'models'

# ──────────────────────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Interview: Why DistilBERT over BERT?
# DistilBERT is 40% smaller, 60% faster at inference, and retains ~97% of
# BERT's accuracy on downstream tasks. For a binary classification task like
# fake news detection, the marginal accuracy loss is negligible compared to
# the massive speedup — especially important when serving real-time predictions.
MODEL_NAME: str = 'distilbert-base-uncased'

# The public Hugging Face repository where your fine-tuned SADA model is hosted.
# Render will pull from this repository automatically because large model files
# are excluded from Git to prevent repository bloat.
HF_MODEL_NAME: str = 'shivavoorkonda/Fake_News_Detection'

# Interview: Why max_length=128 instead of 256?
# Reducing max_length to 64 significantly speeds up self-attention training
# and inference on CPU (yielding a ~4x to 5x total speedup from the baseline)
# with negligible accuracy loss.
MAX_LENGTH: int = 64

# Binary classification: FAKE (0) vs REAL (1).
NUM_LABELS: int = 2

# ──────────────────────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

# Interview: Why batch_size=16?
# 16 is the sweet spot for DistilBERT fine-tuning on a single consumer GPU
# (e.g., RTX 3060/4060 with 8–12 GB VRAM). Larger batches risk OOM errors;
# smaller batches slow down training without meaningful gradient quality gains.
BATCH_SIZE: int = 16

# Interview: Why lr=2e-5?
# This is the canonical fine-tuning LR from the original BERT paper (Devlin
# et al., 2019). Higher LRs risk catastrophic forgetting of pre-trained
# representations; lower LRs converge too slowly for practical training times.
LEARNING_RATE: float = 2e-5

# Interview: Why only 3 epochs?
# Pre-trained transformers converge quickly on downstream tasks. More than 3–5
# epochs typically leads to overfitting, especially on a ~45K-sample dataset.
EPOCHS: int = 3

# Interview: Why warmup steps?
# A linear warmup schedule (0 → peak LR over 500 steps) prevents large early
# gradient updates from destabilizing the pre-trained weights before the
# optimizer has accumulated reliable second-moment estimates (AdamW β₂).
WARMUP_STEPS: int = 500

# Interview: Why weight_decay=0.01?
# L2 regularization via decoupled weight decay (AdamW) prevents overfitting
# by penalizing large weights. 0.01 is the standard from the BERT paper.
WEIGHT_DECAY: float = 0.01

# ──────────────────────────────────────────────────────────────────────────────
# Data Split Ratios
# ──────────────────────────────────────────────────────────────────────────────

# 80/10/10 is a well-accepted split for medium-sized NLP datasets. The 10%
# validation set is large enough (~4.5K samples) for reliable metric estimates,
# and 10% test ensures statistically meaningful final evaluation.
TRAIN_SPLIT: float = 0.8
VAL_SPLIT: float = 0.1
TEST_SPLIT: float = 0.1

# Fixed seed ensures reproducible splits across runs.
RANDOM_SEED: int = 42

# ──────────────────────────────────────────────────────────────────────────────
# Label Mapping
# ──────────────────────────────────────────────────────────────────────────────

# Bidirectional mapping for converting between integer labels and human-readable
# strings. Used in training, evaluation, and API response formatting.
LABEL_MAP: dict[int, str] = {0: 'FAKE', 1: 'REAL'}
LABEL_MAP_REVERSE: dict[str, int] = {'FAKE': 0, 'REAL': 1}
