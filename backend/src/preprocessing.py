"""
preprocessing.py — Text Cleaning and Tokenization Utilities

This module provides all text preprocessing steps needed before feeding
articles into the DistilBERT model. It handles:
    1. Raw text cleaning (HTML, URLs, special characters).
    2. Title + body combination with a separator token.
    3. Tokenizer management (cached loading).
    4. Tokenization into model-ready tensors.

Interview Talking Points:
    - We clean text to remove noise that doesn't carry semantic signal (e.g.,
      HTML artifacts from web scraping, URLs). However, we keep the cleaning
      light because DistilBERT's WordPiece tokenizer can handle most messy
      input; aggressive cleaning can actually hurt performance by removing
      useful punctuation patterns.
    - The [SEP] separator between title and body text mirrors BERT's segment
      separation convention, helping the model learn that these are two
      distinct but related text segments.
"""

import re
import logging
from functools import lru_cache
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import MODEL_NAME, MAX_LENGTH

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean raw article text by removing noise artifacts.

    Processing steps (in order):
        1. Remove URLs — They don't carry semantic meaning for classification
           and can confuse the tokenizer with long, unstructured character
           sequences.
        2. Remove HTML tags — Some articles retain <p>, <br>, etc. from scraping.
        3. Remove special characters — Keep only alphanumeric, spaces, and basic
           punctuation. Punctuation patterns can actually be discriminative
           (fake news tends to use more exclamation marks), so we keep them.
        4. Collapse whitespace — Multiple spaces/newlines become a single space.
        5. Lowercase — DistilBERT-base-uncased expects lowercase input anyway.

    Args:
        text: Raw article text string.

    Returns:
        Cleaned text string ready for tokenization.
    """
    if not isinstance(text, str):
        return ""

    # Step 0: Remove publisher prefixes and standalone mentions of "reuters"
    # to eliminate classification shortcut bias and force the model to learn semantic features.
    # 0a. Match location/publisher tag with parentheses: e.g., "WASHINGTON (Reuters) - " or "LONDON (Reuters) - "
    text = re.sub(r'^\s*(?:[A-Za-z0-9\s,\./_()-]+)?\s*\((?:Reuters|REUTERS|reuters)\)\s*[-—–]*\s*', '', text)
    # 0b. Match simple publisher prefix: e.g., "Reuters - "
    text = re.sub(r'^\s*(?:Reuters|REUTERS|reuters)\s*[-—–]+\s*', '', text)
    # 0c. Remove standalone instances of "reuters" in the body text
    text = re.sub(r'\breuters\b', '', text, flags=re.IGNORECASE)

    # Step 1: Remove URLs (http, https, ftp, and www patterns).
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Step 2: Strip HTML tags.
    text = re.sub(r'<[^>]+>', '', text)

    # Step 3: Remove special characters but keep letters, digits, basic
    # punctuation (.,!?;:'-), and whitespace.
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", '', text)

    # Step 4: Collapse multiple whitespace characters into a single space.
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 5: Lowercase for the uncased model variant.
    text = text.lower()

    return text


def prepare_input(title: str, text: str) -> str:
    """Combine article title and body into a single input string.

    Interview: Why combine title and body with [SEP]?
        The title and body carry complementary signals. The title is a concise
        summary written to grab attention (fake titles tend to be more
        sensational), while the body provides supporting detail. By concatenating
        them with a [SEP] token, we let the model's self-attention mechanism
        learn cross-segment relationships — e.g., does the body actually support
        the claim in the title? This mirrors BERT's original next-sentence
        prediction pre-training objective.

    Args:
        title: Article headline (may be empty or None).
        text: Article body text (may be empty or None).

    Returns:
        Combined string in the format: "title [SEP] text".
        If either part is missing, returns the available part only.
    """
    title = clean_text(title) if title else ""
    text = clean_text(text) if text else ""

    if title and text:
        return f"{title} [SEP] {text}"
    elif title:
        return title
    elif text:
        return text
    else:
        return ""


@lru_cache(maxsize=4)
def get_tokenizer(model_name: Optional[str] = None) -> PreTrainedTokenizerBase:
    """Load and cache a HuggingFace tokenizer.

    Interview: Why cache the tokenizer?
        Tokenizer loading involves reading vocabulary files and initializing
        data structures. Caching avoids redundant I/O when the same tokenizer
        is needed across multiple functions (training, evaluation, inference).
        We use functools.lru_cache for thread-safe, automatic caching.

    Args:
        model_name: HuggingFace model identifier. Defaults to config.MODEL_NAME.

    Returns:
        Loaded PreTrainedTokenizer instance.

    Raises:
        OSError: If the model identifier is invalid or not available.
    """
    model_name = model_name or MODEL_NAME
    logger.info("Loading tokenizer: %s", model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully (vocab size: %d)", tokenizer.vocab_size)
        return tokenizer
    except OSError as e:
        logger.error("Failed to load tokenizer '%s': %s", model_name, e)
        raise


def tokenize_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = MAX_LENGTH,
) -> dict:
    """Tokenize a single text string into model-ready tensors.

    Interview: Why truncation + padding?
        - Truncation: Ensures no input exceeds MAX_LENGTH, preventing OOM
          errors during batch processing.
        - Padding: All inputs in a batch must have the same sequence length
          for efficient GPU parallelism. The attention_mask tells the model
          which tokens are real (1) vs padding (0), so padding doesn't affect
          the model's computations.

    Args:
        text: Pre-processed text string to tokenize.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum sequence length (tokens). Defaults to config.MAX_LENGTH.

    Returns:
        Dictionary with keys:
            - input_ids: Tensor of token IDs.
            - attention_mask: Tensor indicating real tokens (1) vs padding (0).
    """
    encoding = tokenizer(
        text,
        add_special_tokens=True,      # Adds [CLS] at start, [SEP] at end.
        max_length=max_length,
        padding='max_length',          # Pad shorter sequences to max_length.
        truncation=True,               # Truncate longer sequences.
        return_attention_mask=True,
        return_tensors='pt',           # Return PyTorch tensors.
    )

    return {
        'input_ids': encoding['input_ids'].squeeze(0),          # Shape: (max_length,)
        'attention_mask': encoding['attention_mask'].squeeze(0), # Shape: (max_length,)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sample_title = "BREAKING: Major Event Happens!!!"
    sample_text = (
        "According to sources at <b>CNN</b>, a major event occurred today. "
        "Visit http://example.com for more info."
    )

    cleaned = prepare_input(sample_title, sample_text)
    print(f"Cleaned input:\n  {cleaned}\n")

    tok = get_tokenizer()
    encoded = tokenize_text(cleaned, tok)
    print(f"input_ids shape: {encoded['input_ids'].shape}")
    print(f"attention_mask shape: {encoded['attention_mask'].shape}")
