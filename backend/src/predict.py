"""
predict.py — Inference Utilities for Fake News Detection

This module provides a clean inference API for classifying individual articles.
It handles model loading, text preprocessing, forward pass, and probability
computation in a production-ready manner.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import MODEL_DIR, MAX_LENGTH, LABEL_MAP
from src.model import load_saved_model
from src.preprocessing import prepare_input

logger = logging.getLogger(__name__)


def load_predictor(
    model_dir: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the trained model and tokenizer for inference.

    Moves the model to CPU and sets it to evaluation mode.

    Args:
        model_dir: Path to the saved model directory.
                   Defaults to config.MODEL_DIR.

    Returns:
        Tuple of (model, tokenizer) ready for inference.

    Raises:
        FileNotFoundError: If no trained model exists at the given path.
    """
    model_dir = model_dir or str(MODEL_DIR)

    logger.info("Loading predictor from %s", model_dir)
    model, tokenizer = load_saved_model(model_dir)

    # CPU inference for production (no GPU on Render free tier)
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    logger.info("Predictor ready on device: %s", device)
    return model, tokenizer


def predict(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
) -> dict:
    """Classify a single text as FAKE or REAL.

    Args:
        text: Pre-processed article text (ideally via prepare_input()).
        model: Trained classification model.
        tokenizer: Corresponding tokenizer.
        max_length: Maximum token sequence length.

    Returns:
        Dictionary with:
            - label (str): 'FAKE' or 'REAL'.
            - confidence (float): Probability of the predicted class (0.0-1.0).
            - probabilities (dict): {'FAKE': float, 'REAL': float}.

    Raises:
        ValueError: If text is empty or None.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    device = next(model.parameters()).device

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    predicted_class = int(probabilities.argmax())
    confidence = float(probabilities[predicted_class])

    result = {
        'label': LABEL_MAP[predicted_class],
        'confidence': round(confidence, 4),
        'probabilities': {
            LABEL_MAP[0]: round(float(probabilities[0]), 4),  # FAKE
            LABEL_MAP[1]: round(float(probabilities[1]), 4),  # REAL
        },
    }

    logger.info(
        "Prediction: %s (confidence: %.4f)",
        result['label'],
        result['confidence'],
    )

    return result


def predict_from_title_text(
    title: str,
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """Convenience wrapper that preprocesses title + text, then predicts.

    This is the primary entry point for the API endpoint.

    Args:
        title: Article headline (can be empty).
        text: Article body text.
        model: Trained classification model.
        tokenizer: Corresponding tokenizer.

    Returns:
        Same dictionary as predict(): {label, confidence, probabilities}.
    """
    combined_text = prepare_input(title, text)

    if not combined_text:
        raise ValueError(
            "Both title and text are empty. Provide at least one for prediction."
        )

    return predict(combined_text, model, tokenizer)
