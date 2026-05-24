"""
pre_download.py — Build-time Model Pre-downloader for Render

This script runs during the Render Build Step to pre-download and cache model weights 
(both custom Google Drive weights and Hugging Face base models). Because the build step 
has much larger memory limits (>2GB), pre-downloading here completely eliminates 
runtime download overhead and prevents 512MB runtime OOM crashes.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("pre_download")

# Setup python path so we can import src modules
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MODEL_NAME, NUM_LABELS, MODEL_DIR, MODEL_DOWNLOAD_URL
from src.model import download_model_weights

def main():
    logger.info("Starting build-time model pre-download process...")
    save_path = Path(MODEL_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Try downloading custom fine-tuned weights from MODEL_DOWNLOAD_URL
    url = MODEL_DOWNLOAD_URL or ""
    
    # Check weights format
    if "model.safetensors" in url or url.lower().endswith(".safetensors"):
        target_filename = "model.safetensors"
    elif "pytorch_model.bin" in url or url.lower().endswith(".bin"):
        target_filename = "pytorch_model.bin"
    else:
        target_filename = "model.safetensors"

    weights_file = save_path / target_filename
    
    # Check if we should download
    is_missing = not weights_file.exists() or weights_file.stat().st_size < 1024 * 1024

    if is_missing and url:
        logger.info("Custom weights missing from build. Downloading from: %s", url)
        try:
            download_model_weights(url, weights_file)
            logger.info("Successfully downloaded custom weights '%s' during build step.", target_filename)
        except Exception as e:
            logger.error("Failed to download custom weights during build step: %s", e)
    else:
        logger.info("Custom weights '%s' already exist or no download URL provided.", target_filename)

    # 2. Pre-download and cache base model from Hugging Face Hub (Fallback)
    # This ensures that even if custom weights fail or are deleted, the Hugging Face cache
    # is fully populated during build-time so it never downloads during runtime.
    logger.info("Caching base model '%s' from Hugging Face Hub to prevent runtime downloads...", MODEL_NAME)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        # Pre-download and cache model and tokenizer
        AutoTokenizer.from_pretrained(MODEL_NAME)
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        logger.info("Successfully cached base model '%s' from Hugging Face Hub.", MODEL_NAME)
    except Exception as e:
        logger.error("Failed to cache base model from Hugging Face Hub: %s", e)

    logger.info("Build-time pre-download process completed successfully!")

if __name__ == "__main__":
    main()
