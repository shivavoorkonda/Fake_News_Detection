"""
train.py — Full Training Pipeline for Fake News Detection

This module orchestrates the complete training workflow using HuggingFace's
Trainer API. It ties together data loading, preprocessing, dataset creation,
model initialization, and training with evaluation.

Interview Talking Points:
    Why the HuggingFace Trainer API?
        The Trainer abstracts away the boilerplate training loop (gradient
        accumulation, mixed precision, distributed training, checkpointing,
        logging) while remaining flexible enough for custom metrics, callbacks,
        and data collation. Writing a custom training loop gains nothing for
        standard classification tasks and introduces bugs.

    Why AdamW over Adam?
        AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the
        gradient update, fixing a subtle bug in the original Adam implementation
        where weight decay and L2 regularization behave differently. This is
        now the standard optimizer for transformer fine-tuning and is used
        internally by HuggingFace Trainer by default.

    Why a warmup schedule?
        Pre-trained transformer weights are highly optimized. A large learning
        rate at the start can make drastic updates that destroy these learned
        representations ("catastrophic forgetting"). The warmup schedule
        linearly ramps the LR from 0 to the target over WARMUP_STEPS, giving
        the optimizer time to calibrate its per-parameter adaptive learning
        rates (Adam's second moment estimates) before making large updates.

    Why these hyperparameters?
        - LR=2e-5: Canonical BERT fine-tuning LR (Devlin et al., 2019).
        - Epochs=3: Pre-trained models converge quickly; more epochs overfit.
        - Batch=16: Fits in ~8GB VRAM; balances gradient noise and speed.
        - Warmup=500: ~1% of total steps for a ~36K training set × 3 epochs.
"""

import logging
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    MAX_LENGTH,
    MODEL_DIR,
    METRICS_DIR,
    RANDOM_SEED,
)
from src.data_loader import load_data, split_data, get_data_stats
from src.preprocessing import prepare_input, get_tokenizer
from src.dataset import NewsDataset
from src.model import create_model, save_model

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute classification metrics from model predictions.

    This function is passed to the Trainer as a callback. The Trainer calls it
    after each evaluation pass, providing an EvalPrediction namedtuple with
    logits and label_ids.

    Interview: Why these four metrics?
        - Accuracy: Overall correctness. Easy to understand but misleading
          with imbalanced classes.
        - Precision: Of all articles predicted as fake, what fraction actually
          is? High precision = few false alarms.
        - Recall: Of all actually fake articles, what fraction did we catch?
          High recall = few missed fakes.
        - F1: Harmonic mean of precision and recall. Single metric that
          balances both concerns. This is our primary metric for model selection.

    Args:
        eval_pred: EvalPrediction with .predictions (logits) and .label_ids.

    Returns:
        Dictionary with accuracy, f1, precision, and recall.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': round(float(accuracy), 4),
        'f1': round(float(f1), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
    }


def train_model(
    output_dir: Optional[str] = None,
    model_save_dir: Optional[str] = None,
) -> dict:
    """Execute the full training pipeline.

    Workflow:
        1. Load raw data (Fake.csv + True.csv).
        2. Clean and prepare text (title [SEP] body).
        3. Split into train/val/test with stratification.
        4. Create PyTorch Datasets.
        5. Initialize DistilBERT model.
        6. Configure training arguments.
        7. Train with HuggingFace Trainer.
        8. Save the best model checkpoint.
        9. Return evaluation metrics.

    Args:
        output_dir: Directory for training checkpoints and logs.
                    Defaults to MODEL_DIR parent / 'training_output'.
        model_save_dir: Directory to save the final best model.
                        Defaults to config.MODEL_DIR.

    Returns:
        Dictionary containing training metrics and evaluation results.
    """
    model_save_dir = model_save_dir or str(MODEL_DIR)
    output_dir = output_dir or str(Path(MODEL_DIR).parent / 'training_output')

    # Ensure directories exist.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    logger.info("Step 1/7: Loading data...")
    df_raw = load_data()

    # Check if a dry run is requested via environment variable for rapid verification
    is_dry_run = os.environ.get("DRY_RUN", "").lower() == "true"
    is_quick_train = os.environ.get("QUICK_TRAIN", "").lower() == "true"

    if is_dry_run:
        logger.info("DRY_RUN=true detected: Slicing dataset to a tiny fraction (100 samples) and reducing epochs/warmup steps for rapid CPU verification.")
        df = df_raw.sample(n=100, random_state=RANDOM_SEED).reset_index(drop=True)
    elif is_quick_train:
        logger.info("QUICK_TRAIN=true detected: Slicing dataset to 6,000 samples for fast, high-quality CPU training.")
        df = df_raw.sample(n=6000, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        df = df_raw.copy()

    stats = get_data_stats(df)
    logger.info("Dataset stats: %s", json.dumps(stats, indent=2, default=str))

    # ── Step 2: Clean and prepare text ───────────────────────────────────────
    logger.info("Step 2/7: Preprocessing text...")
    df['clean_text'] = df.apply(
        lambda row: prepare_input(
            str(row.get('title', '')),
            str(row.get('text', '')),
        ),
        axis=1,
    )

    # ── Step 3: Stratified split ─────────────────────────────────────────────
    logger.info("Step 3/7: Splitting data...")
    train_df, val_df, test_df = split_data(df)

    # Apply Style-Adversarial Data Augmentation (SADA) to train_df ONLY
    if not is_dry_run:
        logger.info("Applying Style-Adversarial Data Augmentation (SADA) to train_df...")
        from src.augmentation import generate_sober_fakes, generate_sensational_reals
        import pandas as pd

        # Get all real articles from raw data to generate clickbait real samples
        real_raw = df_raw[df_raw['label'] == 1]

        # Generate 400 Sober Fakes and 400 Sensational Reals
        sober_fakes = generate_sober_fakes(num_samples=400, seed=RANDOM_SEED)
        sensational_reals = generate_sensational_reals(real_raw, num_samples=400, seed=RANDOM_SEED)

        # Concatenate augmented data
        augmented_df = pd.concat([sober_fakes, sensational_reals], ignore_index=True)

        # Apply prepare_input to the augmented articles
        augmented_df['clean_text'] = augmented_df.apply(
            lambda row: prepare_input(
                str(row.get('title', '')),
                str(row.get('text', '')),
            ),
            axis=1,
        )

        # Append to train_df
        train_df = pd.concat([train_df, augmented_df], ignore_index=True)
        # Shuffle train_df
        train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        logger.info("SADA completed. Added 800 adversarial style samples to train_df. New train size: %d", len(train_df))

    logger.info(
        "Split sizes — Train: %d, Val: %d, Test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Step 4: Create datasets ──────────────────────────────────────────────
    logger.info("Step 4/7: Creating PyTorch datasets...")
    tokenizer = get_tokenizer()

    train_dataset = NewsDataset(
        texts=train_df['clean_text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    val_dataset = NewsDataset(
        texts=val_df['clean_text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # Save test data for later evaluation (evaluate.py).
    test_texts = test_df['clean_text'].tolist()
    test_labels = test_df['label'].tolist()

    # ── Step 5: Create model ─────────────────────────────────────────────────
    logger.info("Step 5/7: Initializing model...")
    model = create_model()

    # ── Step 6: Configure training ───────────────────────────────────────────
    logger.info("Step 6/7: Setting up training arguments...")

    # Interview: Training arguments breakdown:
    #   - evaluation_strategy='epoch': Evaluate at the end of each epoch rather
    #     than every N steps. For 3 epochs this gives us 3 evaluation points,
    #     which is sufficient for convergence monitoring.
    #   - save_strategy='epoch': Save a checkpoint at each epoch, aligned with
    #     evaluation for load_best_model_at_end to work correctly.
    #   - load_best_model_at_end: After training, automatically restore the
    #     checkpoint with the best eval metric. Prevents using an overfitted
    #     final-epoch model.
    #   - metric_for_best_model='f1': F1 is our primary metric because it
    #     balances precision (few false alarms) and recall (few missed fakes).
    if is_dry_run:
        epochs = 1
        warmup_steps = 2
        logging_steps = 1
    elif is_quick_train:
        epochs = 2
        warmup_steps = 100
        logging_steps = 10
    else:
        epochs = EPOCHS
        warmup_steps = WARMUP_STEPS
        logging_steps = 100

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Eval doesn't need gradients → larger batch
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,                         # Keep only 2 best checkpoints to save disk
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        seed=RANDOM_SEED,
        report_to='none',                           # Disable wandb/tensorboard for simplicity
        fp16=False,                                  # Set True if GPU supports it for 2x speedup
    )

    # ── Step 7: Train ────────────────────────────────────────────────────────
    logger.info("Step 7/7: Starting training...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Run training.
    train_result = trainer.train()

    # Log training metrics.
    train_metrics = train_result.metrics
    logger.info("Training complete. Metrics: %s", train_metrics)

    # ── Save best model ──────────────────────────────────────────────────────
    logger.info("Saving best model to %s", model_save_dir)
    save_model(trainer.model, tokenizer, model_save_dir)

    # ── Evaluate on validation set ───────────────────────────────────────────
    eval_metrics = trainer.evaluate()
    logger.info("Validation metrics: %s", eval_metrics)

    # ── Save test data for evaluate.py ───────────────────────────────────────
    # Store test set indices so evaluate.py can reproduce the exact same split.
    import pandas as pd
    test_data_path = Path(METRICS_DIR) / 'test_data.csv'
    test_save_df = pd.DataFrame({
        'text': test_texts,
        'label': test_labels,
    })
    test_save_df.to_csv(str(test_data_path), index=False)
    logger.info("Test data saved to %s (%d samples)", test_data_path, len(test_save_df))

    # ── Save metrics ─────────────────────────────────────────────────────────
    all_metrics = {
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'dataset_stats': stats,
    }
    metrics_path = Path(METRICS_DIR) / 'training_metrics.json'
    with open(str(metrics_path), 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    return all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point: python -m src.train
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s — %(name)s — %(levelname)s — %(message)s',
    )
    logger.info("=" * 60)
    logger.info("  Fake News Detection — Training Pipeline")
    logger.info("=" * 60)

    metrics = train_model()

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(json.dumps(metrics, indent=2, default=str))
