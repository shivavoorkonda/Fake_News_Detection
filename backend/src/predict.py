"""
predict.py — Inference Utilities for Fake News Detection

This module provides a clean inference API for classifying individual articles.
It handles model loading, text preprocessing, forward pass, and probability
computation in a production-ready manner.

Interview Talking Points:
    Why separate predict.py from model.py?
        Separation of concerns: model.py handles model lifecycle (create, save,
        load), while predict.py handles the inference pipeline (preprocess →
        forward → postprocess). This makes each module independently testable
        and follows the single-responsibility principle.

    Why softmax on logits?
        The model outputs raw logits (unnormalized scores). Softmax converts
        them into a valid probability distribution that sums to 1.0, making
        the confidence scores interpretable and useful for thresholding
        decisions (e.g., "only flag as fake if confidence > 0.9").
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import MODEL_DIR, MAX_LENGTH, LABEL_MAP
from src.model import load_saved_model
from src.preprocessing import prepare_input, get_tokenizer

logger = logging.getLogger(__name__)


def load_predictor(
    model_dir: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the trained model and tokenizer for inference.

    This is a convenience wrapper around model.load_saved_model() that also
    moves the model to the appropriate device (GPU/CPU) and sets it to
    evaluation mode.

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

    # Move model to GPU if available for faster inference.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Redundant (load_saved_model already calls eval), but explicit is better.

    logger.info("Predictor ready on device: %s", device)
    return model, tokenizer


def predict(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
) -> dict:
    """Classify a single text as FAKE or REAL.

    Interview: Inference pipeline walkthrough:
        1. Tokenize the cleaned text into input_ids and attention_mask.
        2. Move tensors to the same device as the model.
        3. Run a forward pass with torch.no_grad() (no gradient computation).
        4. Apply softmax to logits to get calibrated probabilities.
        5. Select the argmax class and its probability as confidence.

    Args:
        text: Pre-processed article text (ideally via prepare_input()).
        model: Trained classification model.
        tokenizer: Corresponding tokenizer.
        max_length: Maximum token sequence length.

    Returns:
        Dictionary with:
            - label (str): 'FAKE' or 'REAL'.
            - confidence (float): Probability of the predicted class (0.0–1.0).
            - probabilities (dict): {'FAKE': float, 'REAL': float}.

    Raises:
        ValueError: If text is empty or None.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    # Determine the device the model is on.
    device = next(model.parameters()).device

    # Tokenize the input text.
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

    # Interview: Why torch.no_grad()?
    # During inference, we don't need gradients. Disabling them:
    #   - Reduces memory usage (~50% less for transformer models).
    #   - Speeds up computation by skipping graph construction.
    #   - Is a critical best practice for production inference.
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Interview: Why softmax instead of sigmoid for binary classification?
    # While sigmoid works for 2-class problems, softmax generalizes to N classes
    # and ensures probabilities sum to exactly 1.0. HuggingFace's classification
    # head outputs 2 logits (one per class), so softmax is the natural choice.
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

    This is the primary entry point for the API endpoint, which receives
    separate title and text fields from the frontend. It handles the
    preprocessing (cleaning + [SEP] concatenation) before calling predict().

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


# ──────────────────────────────────────────────────────────────────────────────
# Quick interactive test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Loading model...")
    model, tokenizer = load_predictor()

    # Test with a sample article.
    sample_title = "Scientists Discover Cure for All Diseases"
    sample_text = (
        "In a groundbreaking announcement, researchers at an unnamed "
        "university claim to have found a single pill that cures every "
        "known disease. The pill, made from common household ingredients, "
        "will be available for free starting next week."
    )

    print(f"\nTitle: {sample_title}")
    print(f"Text: {sample_text[:100]}...")

    result = predict_from_title_text(sample_title, sample_text, model, tokenizer)
    print(f"\nPrediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {result['probabilities']}")
