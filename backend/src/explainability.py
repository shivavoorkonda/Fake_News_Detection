"""
explainability.py — SHAP and LIME Explanations for Model Predictions

This module provides model interpretability using two complementary approaches:
    - SHAP (SHapley Additive exPlanations): Theoretically grounded in game
      theory, assigns each feature a contribution value based on Shapley values.
    - LIME (Local Interpretable Model-agnostic Explanations): Fits a simple
      interpretable model locally around the prediction to approximate the
      decision boundary.

Interview Talking Points:
    Why both SHAP and LIME?
        - SHAP is theoretically optimal (unique solution satisfying efficiency,
          symmetry, dummy, and additivity axioms), but can be slow on long texts
          because it requires many model evaluations.
        - LIME is faster and produces intuitive word-level importance scores,
          but its explanations can be unstable (different runs may give slightly
          different results due to random perturbation sampling).
        Offering both gives users the choice: SHAP for rigorous analysis, LIME
        for quick insights. In interviews, this shows you understand the
        trade-offs rather than blindly picking one.

    Why explainability matters for fake news detection:
        Users need to trust the model's decisions. Simply saying "this is fake"
        is not enough — showing WHICH words triggered the prediction (e.g.,
        "BREAKING", "anonymous sources", "scientists say") builds trust and
        helps journalists and fact-checkers prioritize their review efforts.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config import MODEL_DIR, LABEL_MAP

logger = logging.getLogger(__name__)


def create_prediction_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> callable:
    """Create a prediction function compatible with SHAP and LIME.

    Interview: Why a wrapper function?
        SHAP and LIME expect a simple function that takes a list of strings
        and returns a 2D numpy array of probabilities (shape: [n_samples, n_classes]).
        Our model requires tokenization, device placement, and post-processing.
        This wrapper hides that complexity behind a clean interface.

    Args:
        model: Trained classification model.
        tokenizer: Corresponding tokenizer.

    Returns:
        A callable that accepts a list of strings and returns probability arrays.
    """
    device = next(model.parameters()).device

    def prediction_fn(texts: list[str]) -> np.ndarray:
        """Predict probabilities for a batch of texts.

        Args:
            texts: List of text strings to classify.

        Returns:
            numpy array of shape (len(texts), num_labels) with probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize the batch.
        encodings = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt',
        )

        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        return probs

    return prediction_fn


def explain_with_shap(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    num_features: int = 15,
) -> list[dict[str, float]]:
    """Generate SHAP explanations for a prediction.

    Interview: How SHAP works for text:
        SHAP uses a "masker" that replaces words with a baseline (e.g., mask
        token) and measures how the prediction changes. It computes approximate
        Shapley values by sampling coalitions of features (words). The Shapley
        value for each word represents its marginal contribution to the
        prediction, averaged over all possible word orderings.

    Note: SHAP can be slow for long texts due to the exponential number of
    feature coalitions. We limit the number of evaluations via max_evals.

    Args:
        text: Input text to explain.
        model: Trained classification model.
        tokenizer: Corresponding tokenizer.
        num_features: Maximum number of top features to return.

    Returns:
        List of dicts with 'token' and 'weight' keys, sorted by absolute weight.
        Positive weights push toward REAL, negative toward FAKE.
    """
    try:
        import shap
    except ImportError:
        logger.error(
            "SHAP is not installed. Install it with: pip install shap"
        )
        raise ImportError(
            "SHAP library is required for explanations. "
            "Install with: pip install shap"
        )

    logger.info("Generating SHAP explanations...")

    prediction_fn = create_prediction_pipeline(model, tokenizer)

    # Interview: Why the text masker?
    # The masker defines the baseline: what to replace masked words with.
    # Using a text masker with the tokenizer automatically handles subword
    # tokens and uses appropriate masking strategies for transformer models.
    masker = shap.maskers.Text(tokenizer)

    explainer = shap.Explainer(
        prediction_fn,
        masker,
        output_names=[LABEL_MAP[0], LABEL_MAP[1]],
    )

    # Compute SHAP values. max_evals controls the approximation quality/speed
    # trade-off. 500 evaluations is a reasonable balance.
    try:
        shap_values = explainer(
            [text],
            max_evals=min(500, 2 * len(text.split()) + 1),
        )
    except Exception as e:
        logger.warning("SHAP explanation failed: %s. Returning empty.", e)
        return []

    # Extract token-level attributions for the predicted class.
    # shap_values.values has shape (1, n_tokens, n_classes).
    if hasattr(shap_values, 'values') and shap_values.values is not None:
        values = shap_values.values[0]  # First (only) sample.

        # Get the predicted class to select the relevant SHAP values.
        probs = prediction_fn([text])
        predicted_class = int(np.argmax(probs[0]))

        # Get SHAP values for the predicted class.
        if len(values.shape) > 1:
            feature_values = values[:, predicted_class]
        else:
            feature_values = values

        # Get corresponding token strings.
        if hasattr(shap_values, 'data') and shap_values.data is not None:
            tokens = shap_values.data[0]
        else:
            tokens = text.split()

        # Build token-weight pairs and sort by absolute importance.
        explanations = []
        for i, (token, weight) in enumerate(zip(tokens, feature_values)):
            token_str = str(token).strip()
            if token_str:  # Skip empty tokens.
                explanations.append({
                    'token': token_str,
                    'weight': round(float(weight), 6),
                })

        # Sort by absolute weight (most important first) and limit.
        explanations.sort(key=lambda x: abs(x['weight']), reverse=True)
        explanations = explanations[:num_features]

        logger.info("SHAP: Returned %d token attributions.", len(explanations))
        return explanations

    logger.warning("SHAP produced no values. Returning empty list.")
    return []


def explain_with_lime(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    num_features: int = 15,
) -> list[dict[str, float]]:
    """Generate LIME explanations for a prediction.

    Interview: How LIME works:
        1. Perturb the input by randomly removing words to create ~5000 variants.
        2. Classify each variant with the black-box model.
        3. Fit a simple linear model (Ridge regression) on the perturbations,
           weighted by their similarity to the original input.
        4. The linear model's coefficients are the word importance scores.
        LIME is model-agnostic — it treats the classifier as a black box and
        only needs input/output access.

    Args:
        text: Input text to explain.
        model: Trained classification model.
        tokenizer: Corresponding tokenizer.
        num_features: Number of top features to include in the explanation.

    Returns:
        List of dicts with 'token' and 'weight' keys, sorted by absolute weight.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.error(
            "LIME is not installed. Install it with: pip install lime"
        )
        raise ImportError(
            "LIME library is required for explanations. "
            "Install with: pip install lime"
        )

    logger.info("Generating LIME explanations...")

    prediction_fn = create_prediction_pipeline(model, tokenizer)

    # Interview: Why these class names?
    # LIME uses class names for labeling its output. Matching our LABEL_MAP
    # ensures consistency across the application.
    explainer = LimeTextExplainer(
        class_names=[LABEL_MAP[0], LABEL_MAP[1]],
        split_expression=r'\W+',  # Split on non-word characters.
    )

    try:
        explanation = explainer.explain_instance(
            text,
            prediction_fn,
            num_features=num_features,
            num_samples=1000,  # Number of perturbations to generate.
        )
    except Exception as e:
        logger.warning("LIME explanation failed: %s. Returning empty.", e)
        return []

    # Extract word importance scores.
    # explanation.as_list() returns [(word, weight), ...] for the predicted class.
    feature_weights = explanation.as_list()

    explanations = [
        {
            'token': word,
            'weight': round(float(weight), 6),
        }
        for word, weight in feature_weights
    ]

    # Sort by absolute weight (most important first).
    explanations.sort(key=lambda x: abs(x['weight']), reverse=True)

    logger.info("LIME: Returned %d word importances.", len(explanations))
    return explanations


# ──────────────────────────────────────────────────────────────────────────────
# Quick interactive test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from src.predict import load_predictor

    logging.basicConfig(level=logging.INFO)

    print("Loading model...")
    model, tokenizer = load_predictor()

    sample_text = (
        "Scientists at Harvard University have published a new study "
        "showing that regular exercise reduces the risk of heart disease "
        "by 30 percent, according to the New England Journal of Medicine."
    )

    print(f"\nInput text: {sample_text[:100]}...")

    print("\n--- LIME Explanation ---")
    lime_result = explain_with_lime(sample_text, model, tokenizer, num_features=10)
    for item in lime_result:
        direction = "→ REAL" if item['weight'] > 0 else "→ FAKE"
        print(f"  {item['token']:20s} {item['weight']:+.6f} {direction}")

    print("\n--- SHAP Explanation ---")
    shap_result = explain_with_shap(sample_text, model, tokenizer, num_features=10)
    for item in shap_result:
        print(f"  {item['token']:20s} {item['weight']:+.6f}")
