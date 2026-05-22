"""
model.py — Model Creation, Saving, and Loading

This module wraps HuggingFace's DistilBertForSequenceClassification with
convenience functions for creating, saving, and loading the model. It keeps
model management logic separate from training logic for cleaner architecture.

Interview Talking Points:
    Why DistilBERT over BERT?
        DistilBERT (Sanh et al., 2019) was trained using knowledge distillation
        from BERT-base. Key advantages:
            - 40% smaller: 66M params vs 110M params (BERT-base).
            - 60% faster inference: Fewer transformer layers (6 vs 12).
            - 97% performance retention: On GLUE benchmark, DistilBERT retains
              97% of BERT's language understanding capability.
        For binary classification tasks like fake news detection, the accuracy
        difference is negligible, but the speed/size gains are massive —
        especially important for real-time API inference and deployment on
        resource-constrained environments.

    Why DistilBertForSequenceClassification?
        This is a pre-built architecture that adds a classification head (linear
        layer) on top of DistilBERT's [CLS] token output. It handles the loss
        computation internally when labels are provided, integrating seamlessly
        with HuggingFace's Trainer API.
"""

import logging
from pathlib import Path
from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config import MODEL_NAME, NUM_LABELS, MODEL_DIR

logger = logging.getLogger(__name__)


def create_model(
    model_name: str = MODEL_NAME,
    num_labels: int = NUM_LABELS,
) -> PreTrainedModel:
    """Create a DistilBERT model with a sequence classification head.

    Interview: What happens under the hood?
        1. Downloads pre-trained DistilBERT weights from HuggingFace Hub
           (or loads from cache if previously downloaded).
        2. Initializes a randomly-weighted classification head on top.
        3. During fine-tuning, both the pre-trained transformer layers AND the
           classification head are updated. This is "full fine-tuning" as opposed
           to "feature extraction" where transformer weights are frozen.

    Args:
        model_name: HuggingFace model identifier (e.g., 'distilbert-base-uncased').
        num_labels: Number of output classes. 2 for binary (FAKE/REAL).

    Returns:
        Pre-trained DistilBERT model with classification head.

    Raises:
        OSError: If the model cannot be downloaded or loaded.
    """
    logger.info("Creating model: %s (num_labels=%d)", model_name, num_labels)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model created — Total params: %s, Trainable: %s",
            f"{total_params:,}",
            f"{trainable_params:,}",
        )
        return model

    except OSError as e:
        logger.error("Failed to create model '%s': %s", model_name, e)
        raise


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Optional[str] = None,
) -> str:
    """Save the fine-tuned model and tokenizer to disk.

    Interview: Why save both model AND tokenizer together?
        The tokenizer's vocabulary and special tokens must match the model's
        embedding layer. Saving them together ensures they stay in sync. If
        you save only the model and later load a different tokenizer version,
        token ID mismatches will produce garbage predictions silently — a
        notoriously hard-to-debug production issue.

    Args:
        model: Trained model to save.
        tokenizer: Tokenizer used during training.
        save_dir: Directory path to save to. Defaults to config.MODEL_DIR.

    Returns:
        The path where the model was saved.
    """
    save_dir = save_dir or str(MODEL_DIR)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving model to %s", save_path)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info("Model and tokenizer saved successfully.")

    return str(save_path)


def load_saved_model(
    save_dir: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a previously saved model and tokenizer from disk.

    Interview: Why use AutoModel/AutoTokenizer instead of DistilBert* directly?
        The Auto* classes inspect the saved config.json to determine the exact
        model architecture. This makes the loading code architecture-agnostic —
        if we later switch from DistilBERT to RoBERTa, this function still
        works without code changes. It's a best practice for production systems
        where models may be swapped during A/B testing.

    Args:
        save_dir: Directory containing the saved model files.
                  Defaults to config.MODEL_DIR.

    Returns:
        Tuple of (model, tokenizer) loaded from disk.

    Raises:
        FileNotFoundError: If the save directory doesn't exist.
        OSError: If model files are corrupted or incompatible.
    """
    save_dir = save_dir or str(MODEL_DIR)
    save_path = Path(save_dir)

    if not save_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {save_path}. "
            "Train the model first using train.py."
        )

    logger.info("Loading model from %s", save_path)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(save_path))
        tokenizer = AutoTokenizer.from_pretrained(str(save_path))

        # Set model to evaluation mode by default for inference safety.
        model.eval()

        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer

    except OSError as e:
        logger.error("Failed to load model from '%s': %s", save_path, e)
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    model = create_model()
    print(f"\nModel type: {type(model).__name__}")
    print(f"Number of layers: {model.config.n_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
