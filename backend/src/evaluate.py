"""
evaluate.py — Model Evaluation and Metrics Generation

This module provides comprehensive evaluation of the trained fake news
detection model on the held-out test set. It generates:
    - Classification metrics (accuracy, precision, recall, F1).
    - A detailed classification report.
    - A confusion matrix visualization saved as an image.

Interview Talking Points:
    Why a separate evaluation module?
        Training metrics (computed on the validation set during training) are
        optimistic because the model's hyperparameters were tuned to maximize
        them. The test set provides an unbiased estimate of real-world
        performance because the model has NEVER seen this data — not during
        training, not during hyperparameter selection.

    Why the confusion matrix?
        Raw metrics like accuracy can hide class-specific failures. The
        confusion matrix shows exactly where the model makes mistakes:
            - False Positives (fake predicted as real): Dangerous, as
              misinformation passes through.
            - False Negatives (real predicted as fake): Annoying but less
              harmful — legitimate news gets flagged.
        This breakdown informs whether the model is biased toward one class.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.config import MODEL_DIR, METRICS_DIR, MAX_LENGTH, LABEL_MAP
from src.model import load_saved_model
from src.dataset import NewsDataset

logger = logging.getLogger(__name__)


def evaluate_model(model_dir: Optional[str] = None) -> dict:
    """Run full evaluation on the held-out test set.

    Workflow:
        1. Load the saved model and tokenizer.
        2. Load the test data saved during training.
        3. Run inference on all test samples.
        4. Compute comprehensive metrics.
        5. Generate and save confusion matrix plot.
        6. Save metrics as JSON for the API to serve.

    Args:
        model_dir: Path to the saved model directory.
                   Defaults to config.MODEL_DIR.

    Returns:
        Dictionary containing:
            - accuracy, precision, recall, f1 (float).
            - confusion_matrix (list of lists).
            - classification_report (str).
    """
    model_dir = model_dir or str(MODEL_DIR)

    # ── Load model ───────────────────────────────────────────────────────────
    logger.info("Loading saved model from %s", model_dir)
    try:
        model, tokenizer = load_saved_model(model_dir)
    except FileNotFoundError:
        logger.error(
            "No saved model found. Train the model first: python -m src.train"
        )
        raise

    # Determine device (GPU if available, else CPU).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    logger.info("Model loaded on device: %s", device)

    # ── Load test data ───────────────────────────────────────────────────────
    test_data_path = Path(METRICS_DIR) / 'test_data.csv'
    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}. "
            "Run training first to generate the test split."
        )

    test_df = pd.read_csv(str(test_data_path))
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    logger.info("Loaded %d test samples", len(test_texts))

    # ── Run inference ────────────────────────────────────────────────────────
    logger.info("Running inference on test set...")
    test_dataset = NewsDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    all_predictions = []
    all_labels = []
    batch_size = 32  # Larger batch for eval since no gradients are stored.

    # Interview: Why torch.no_grad()?
    # During inference, we don't need to compute gradients. Disabling gradient
    # tracking reduces memory usage by ~50% and speeds up computation, because
    # PyTorch doesn't need to build the computational graph for backprop.
    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            batch_end = min(i + batch_size, len(test_dataset))
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels_list = []

            for j in range(i, batch_end):
                sample = test_dataset[j]
                batch_input_ids.append(sample['input_ids'])
                batch_attention_mask.append(sample['attention_mask'])
                batch_labels_list.append(sample['labels'].item())

            input_ids = torch.stack(batch_input_ids).to(device)
            attention_mask = torch.stack(batch_attention_mask).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch_labels_list)

            if (i // batch_size) % 10 == 0:
                logger.info(
                    "  Processed %d/%d samples...", batch_end, len(test_dataset)
                )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    # ── Compute metrics ──────────────────────────────────────────────────────
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=1
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=[LABEL_MAP[0], LABEL_MAP[1]],
    )

    metrics = {
        'accuracy': round(float(accuracy), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1': round(float(f1), 4),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'total_test_samples': len(y_true),
    }

    logger.info("\n=== Evaluation Results ===")
    logger.info("Accuracy:  %.4f", accuracy)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall:    %.4f", recall)
    logger.info("F1 Score:  %.4f", f1)
    logger.info("\nClassification Report:\n%s", report)

    # ── Generate confusion matrix plot ───────────────────────────────────────
    cm_save_path = Path(METRICS_DIR) / 'confusion_matrix.png'
    generate_confusion_matrix(y_true, y_pred, str(cm_save_path))

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics_save_path = Path(METRICS_DIR) / 'evaluation_metrics.json'
    save_metrics(metrics, str(metrics_save_path))

    return metrics


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """Generate and save a confusion matrix heatmap.

    Interview: Why visualize the confusion matrix?
        Numbers alone can be misleading. A heatmap immediately reveals:
            - Whether errors are symmetric (equal FP and FN) or skewed.
            - The absolute magnitude of errors vs correct predictions.
            - Class-specific performance at a glance.
        This is the first plot interviewers look at to assess model quality.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_path: File path to save the plot image.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = [LABEL_MAP[0], LABEL_MAP[1]]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
    )
    plt.title('Confusion Matrix — Fake News Detection', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info("Confusion matrix saved to %s", save_path)


def save_metrics(metrics: dict, save_path: str) -> None:
    """Save evaluation metrics to a JSON file.

    The JSON file is read by the /metrics API endpoint to serve model
    performance data to the frontend without re-running evaluation.

    Args:
        metrics: Dictionary of metric names and values.
        save_path: File path for the output JSON.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Remove non-serializable fields for clean JSON output.
    serializable_metrics = {
        k: v for k, v in metrics.items()
        if k != 'classification_report'
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=2)

    logger.info("Metrics saved to %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point: python -m src.evaluate
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s — %(name)s — %(levelname)s — %(message)s',
    )
    logger.info("=" * 60)
    logger.info("  Fake News Detection — Evaluation Pipeline")
    logger.info("=" * 60)

    results = evaluate_model()

    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print("=" * 60)
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"\n{results['classification_report']}")
