"""
data_loader.py — Data Loading and Splitting Utilities

This module handles loading the Kaggle 'Fake and Real News Dataset' from raw
CSVs, assigning labels, merging them into a single DataFrame, and performing
stratified train/val/test splits.

Interview Talking Points:
    - The dataset has NO explicit label column; the label is implicit from the
      filename (Fake.csv → 0, True.csv → 1). We assign labels during loading.
    - Stratified splitting ensures each split has the same class distribution
      as the overall dataset, which prevents the model from seeing a skewed
      class ratio in any split. This is critical because our dataset is slightly
      imbalanced (~23.5K fake vs ~21.4K real).

Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""

import logging
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    LABEL_MAP,
)

logger = logging.getLogger(__name__)


def load_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load and merge Fake.csv and True.csv into a single labeled DataFrame.

    The Kaggle dataset ships as two separate CSV files with no label column.
    This function reads both files, assigns integer labels based on the source
    file (Fake → 0, True → 1), concatenates them, shuffles the result, and
    resets the index for clean downstream processing.

    Args:
        data_dir: Path to the directory containing Fake.csv and True.csv.
                  Defaults to RAW_DATA_DIR from config.

    Returns:
        pd.DataFrame with columns: title, text, subject, date, label (int).

    Raises:
        FileNotFoundError: If Fake.csv or True.csv is missing from data_dir.
        pd.errors.EmptyDataError: If a CSV file is empty.
    """
    data_dir = data_dir or str(RAW_DATA_DIR)

    fake_path = f"{data_dir}/Fake.csv"
    true_path = f"{data_dir}/True.csv"

    logger.info("Loading Fake.csv from %s", fake_path)
    try:
        fake_df = pd.read_csv(fake_path)
        fake_df['label'] = 0  # FAKE
    except FileNotFoundError:
        logger.error("Fake.csv not found at %s", fake_path)
        raise FileNotFoundError(
            f"Fake.csv not found at {fake_path}. "
            "Download the dataset from Kaggle and place it in the raw data directory."
        )

    logger.info("Loading True.csv from %s", true_path)
    try:
        true_df = pd.read_csv(true_path)
        true_df['label'] = 1  # REAL
    except FileNotFoundError:
        logger.error("True.csv not found at %s", true_path)
        raise FileNotFoundError(
            f"True.csv not found at {true_path}. "
            "Download the dataset from Kaggle and place it in the raw data directory."
        )

    # Concatenate and shuffle to avoid ordering bias during training.
    # Interview: Without shuffling, the model would see all fake articles
    # first, then all real articles, leading to catastrophic gradient drift.
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    logger.info(
        "Loaded %d total articles (%d fake, %d real)",
        len(df),
        len(fake_df),
        len(true_df),
    )

    return df


def split_data(
    df: pd.DataFrame,
    train: float = TRAIN_SPLIT,
    val: float = VAL_SPLIT,
    test: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train, validation, and test sets.

    Interview: Why stratified splitting?
        A naive random split can produce skewed label distributions, especially
        with imbalanced datasets. Stratification guarantees that each split
        mirrors the overall class ratio (≈52% fake, ≈48% real). This ensures
        that validation and test metrics are representative of real-world
        performance and prevents the model from appearing artificially better
        or worse due to a lucky/unlucky split.

    The split is done in two stages:
        1. Split off (val + test) from train using stratify.
        2. Split the held-out portion into val and test using stratify.

    Args:
        df: Full dataset with a 'label' column.
        train: Fraction of data for training (default 0.8).
        val: Fraction of data for validation (default 0.1).
        test: Fraction of data for testing (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If split fractions don't sum to ~1.0.
    """
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got {total:.4f} "
            f"(train={train}, val={val}, test={test})"
        )

    # Stage 1: Separate training set from the rest.
    train_df, temp_df = train_test_split(
        df,
        test_size=(val + test),
        random_state=seed,
        stratify=df['label'],
    )

    # Stage 2: Split the remaining data into validation and test.
    # The relative proportion of val in the remaining data:
    val_ratio = val / (val + test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        random_state=seed,
        stratify=temp_df['label'],
    )

    logger.info(
        "Split sizes — Train: %d, Val: %d, Test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return train_df, val_df, test_df


def get_data_stats(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics about the dataset.

    Useful for EDA (Exploratory Data Analysis) and for logging metadata
    alongside model training runs to ensure reproducibility.

    Args:
        df: DataFrame with columns 'label', 'subject', 'title', 'text'.

    Returns:
        Dictionary containing:
            - total_samples: Total number of articles.
            - label_distribution: Count of each label (FAKE/REAL).
            - label_percentages: Percentage of each label.
            - subject_distribution: Count of articles per subject category.
            - avg_title_length: Mean character length of titles.
            - avg_text_length: Mean character length of article bodies.
    """
    stats: dict = {
        'total_samples': len(df),

        # Map integer labels to human-readable names for clarity.
        'label_distribution': {
            LABEL_MAP.get(k, str(k)): v
            for k, v in df['label'].value_counts().to_dict().items()
        },
        'label_percentages': {
            LABEL_MAP.get(k, str(k)): round(v * 100, 2)
            for k, v in df['label'].value_counts(normalize=True).to_dict().items()
        },
    }

    # Subject distribution reveals topical skew (e.g., "politicsNews" dominates
    # the real-news file). This is important for understanding model biases.
    if 'subject' in df.columns:
        stats['subject_distribution'] = df['subject'].value_counts().to_dict()

    # Text length statistics help justify the MAX_LENGTH choice in config.
    if 'title' in df.columns:
        stats['avg_title_length'] = round(df['title'].astype(str).str.len().mean(), 1)
    if 'text' in df.columns:
        stats['avg_text_length'] = round(df['text'].astype(str).str.len().mean(), 1)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity-check when running this module directly.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    data = load_data()
    print("\n=== Dataset Stats ===")
    for key, value in get_data_stats(data).items():
        print(f"  {key}: {value}")

    train_set, val_set, test_set = split_data(data)
    print(f"\nTrain: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
