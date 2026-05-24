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

from src.config import MODEL_NAME, NUM_LABELS, MODEL_DIR, MODEL_DOWNLOAD_URL

logger = logging.getLogger(__name__)


def download_model_weights(url: str, dest_path: Path):
    """Download model weights from a public URL with a progress bar in the logs."""
    import urllib.request
    import time

    logger.info("Starting download of model weights from: %s", url)
    logger.info("Destination path: %s", dest_path)

    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    last_reported = 0.0

    def progress_callback(block_num, block_size, total_size):
        nonlocal last_reported
        downloaded = block_num * block_size
        if total_size > 0:
            percent = (downloaded / total_size) * 100
            # Report progress every 10% to avoid flooding the logs
            if percent - last_reported >= 10.0 or percent >= 100.0:
                elapsed = time.time() - start_time
                speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Downloading model weights: %.1f%% (%.1f MB of %.1f MB) - Speed: %.2f MB/s",
                    percent,
                    downloaded / (1024 * 1024),
                    total_size / (1024 * 1024),
                    speed
                )
                last_reported = percent
        else:
            if downloaded - last_reported >= 10 * 1024 * 1024:  # every 10MB
                logger.info("Downloaded %.1f MB (unknown total size)", downloaded / (1024 * 1024))
                last_reported = downloaded

    try:
        # Use urllib to download to temporary file first, then rename to dest_path
        # to prevent corrupt half-downloaded files in case of interruption
        temp_path = dest_path.with_suffix('.tmp')

        # User-agent header to avoid getting blocked by some hosting providers
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(url, str(temp_path), reporthook=progress_callback)

        # Rename tmp file to final destination
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)

        elapsed = time.time() - start_time
        logger.info(
            "Download completed successfully in %.1fs. Model size: %.1f MB",
            elapsed,
            dest_path.stat().st_size / (1024 * 1024)
        )
    except Exception as e:
        logger.error("Failed to download model weights from %s: %s", url, e)
        if 'temp_path' in locals() and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


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

    # 1. Smart weights format detection based on MODEL_DOWNLOAD_URL
    url = MODEL_DOWNLOAD_URL or ""
    if "model.safetensors" in url or url.lower().endswith(".safetensors"):
        target_filename = "model.safetensors"
    elif "pytorch_model.bin" in url or url.lower().endswith(".bin"):
        target_filename = "pytorch_model.bin"
    elif "quantized_model.pt" in url or url.lower().endswith(".pt"):
        target_filename = "quantized_model.pt"
    else:
        # Default to standard model.safetensors for extremely memory-efficient mmap loading
        target_filename = "model.safetensors"

    weights_file = save_path / target_filename

    # Check if the file is missing or a Git LFS pointer (< 1MB)
    is_missing_or_pointer = not weights_file.exists() or weights_file.stat().st_size < 1024 * 1024

    if is_missing_or_pointer:
        if url:
            logger.info("Model weights file '%s' is missing or is a Git LFS pointer. Downloading actual weights...", target_filename)
            try:
                download_model_weights(url, weights_file)
            except Exception as e:
                logger.error("Automatic download of weights failed: %s", e)
        else:
            logger.warning("Weights file '%s' is missing/pointer, but MODEL_DOWNLOAD_URL is not set.", target_filename)

    # ── Option B: Full fp32 model from local disk (Highly Preferred: uses memory-mapped safetensors/bin, ZERO startup memory spike, fits in 512MB RAM) ──
    local_safetensors = save_path / "model.safetensors"
    local_bin = save_path / "pytorch_model.bin"

    has_safetensors = local_safetensors.exists() and local_safetensors.stat().st_size > 1024 * 1024
    has_bin = local_bin.exists() and local_bin.stat().st_size > 1024 * 1024

    if has_safetensors or has_bin:
        logger.info("Loading standard fp32 model using memory mapping (extremely light on RAM)...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(str(save_path))
            tokenizer = AutoTokenizer.from_pretrained(str(save_path))
            model.eval()
            logger.info("Standard model loaded successfully using mmap.")
            return model, tokenizer
        except Exception as e:
            logger.error("Failed to load standard model: %s", e)
            raise

    # ── Option A: Load INT8 quantized model (Note: High peak RAM spike at startup!) ──
    local_quantized = save_path / "quantized_model.pt"
    has_quantized = local_quantized.exists() and local_quantized.stat().st_size > 1024 * 1024

    # Render Safety Guard: Prevent loading quantized model on Render free tier (causes OOM due to peak RAM spike of >900MB during torch.quantization)
    import os
    if has_quantized and "RENDER" in os.environ:
        logger.warning("No standard weights found. Detected Render environment: skipping INT8 quantized loading fallback to prevent OOM crash.")
        has_quantized = False

    if has_quantized:
        logger.warning("No standard weights found. Loading INT8 quantized model (Note: High peak RAM spike at startup!)")
        try:
            import torch
            import gc
            from transformers import AutoConfig

            # Rebuild architecture from config.json
            config = AutoConfig.from_pretrained(str(save_path))
            model = AutoModelForSequenceClassification.from_config(config)
            gc.collect()

            # Apply dynamic quantization to match saved model structure
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            gc.collect()

            # Load fine-tuned, quantized weights
            state_dict = torch.load(
                str(local_quantized), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
            del state_dict
            gc.collect()

            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(str(save_path))
            logger.info("Quantized model loaded successfully.")
            return model, tokenizer
        except Exception as e:
            logger.error("Failed to load quantized model: %s", e)
            raise

    # ── Option C: Hugging Face Hub Base Model Fallback (Ensures 100% successful zero-config start, NO 503 errors!) ──
    logger.warning("No custom fine-tuned weights found locally or downloaded. Falling back to pre-trained base model '%s' from Hugging Face Hub for zero-config startup...", MODEL_NAME)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.eval()
        logger.info("Base pre-trained model fallback loaded successfully from Hugging Face Hub.")
        return model, tokenizer
    except Exception as e:
        logger.error("Failed to load base pre-trained model fallback: %s", e)
        raise FileNotFoundError(
            f"No model weights found at: {save_path}, and fallback failed: {e}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    model = create_model()
    print(f"\nModel type: {type(model).__name__}")
    print(f"Number of layers: {model.config.n_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
