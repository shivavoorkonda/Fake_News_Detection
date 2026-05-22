"""
dataset.py — PyTorch Dataset for News Article Classification

This module defines a custom PyTorch Dataset that tokenizes articles on-the-fly
during training. This is preferable to pre-tokenizing the entire dataset because:

Interview Talking Points:
    1. Memory efficiency: We don't store ~45K tokenized sequences in RAM. Only
       the raw strings are held; tokenization happens per-batch.
    2. Flexibility: Changing MAX_LENGTH or the tokenizer doesn't require
       re-running a preprocessing pipeline — just update config and retrain.
    3. Data augmentation: On-the-fly processing makes it trivial to add future
       augmentations (e.g., random truncation, synonym replacement) without
       modifying saved data files.
"""

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config import MAX_LENGTH

logger = logging.getLogger(__name__)


class NewsDataset(Dataset):
    """PyTorch Dataset for fake/real news classification.

    Each sample is tokenized on-the-fly and returned as a dictionary
    compatible with HuggingFace's Trainer API, which expects keys
    'input_ids', 'attention_mask', and 'labels'.

    Interview: Why return 'labels' (plural) and not 'label'?
        The HuggingFace Trainer API specifically expects the key 'labels'
        (with an 's') for computing the loss. Using 'label' would silently
        skip loss computation and produce NaN losses — a subtle bug.

    Attributes:
        texts: List of pre-processed text strings (title [SEP] body).
        labels: List of integer labels (0 = FAKE, 1 = REAL).
        tokenizer: HuggingFace tokenizer for encoding texts.
        max_length: Maximum token sequence length for padding/truncation.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
    ) -> None:
        """Initialize the NewsDataset.

        Args:
            texts: List of article text strings (already cleaned and combined).
            labels: List of corresponding integer labels.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum sequence length for tokenization.

        Raises:
            ValueError: If texts and labels have different lengths.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Mismatch: {len(texts)} texts vs {len(labels)} labels. "
                "Each text must have a corresponding label."
            )

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(
            "Created NewsDataset with %d samples (max_length=%d)",
            len(self.texts),
            self.max_length,
        )

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize and return a single sample.

        Interview: Why tokenize on-the-fly instead of pre-tokenizing?
            Pre-tokenizing 45K articles at max_length=256 would consume
            ~45K × 256 × 2 tensors × 4 bytes ≈ 90 MB of RAM just for IDs.
            On-the-fly tokenization uses negligible extra CPU (tokenization
            is fast) and keeps memory footprint proportional to raw text size.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary with:
                - input_ids: Token IDs tensor of shape (max_length,).
                - attention_mask: Mask tensor of shape (max_length,).
                - labels: Scalar label tensor.
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text with padding and truncation.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),          # (max_length,)
            'attention_mask': encoding['attention_mask'].squeeze(0), # (max_length,)
            'labels': torch.tensor(label, dtype=torch.long),        # scalar
        }


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from src.preprocessing import get_tokenizer

    logging.basicConfig(level=logging.INFO)

    sample_texts = [
        "breaking news: world event [SEP] details about the event...",
        "study shows positive results [SEP] researchers confirmed findings...",
    ]
    sample_labels = [0, 1]

    tokenizer = get_tokenizer()
    dataset = NewsDataset(sample_texts, sample_labels, tokenizer)

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"input_ids shape: {sample['input_ids'].shape}")
    print(f"attention_mask shape: {sample['attention_mask'].shape}")
    print(f"label: {sample['labels']}")
