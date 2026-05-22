"""
api.py — FastAPI Backend for Fake News Detection

This module defines the REST API that serves model predictions and explanations
to the frontend. It uses FastAPI for automatic OpenAPI documentation, Pydantic
for request/response validation, and a lifespan context manager for efficient
model loading.

Interview Talking Points:
    Why FastAPI?
        - Async by default: Handles concurrent requests efficiently using
          Python's asyncio. Important when multiple users query simultaneously.
        - Automatic docs: Generates Swagger UI at /docs and ReDoc at /redoc,
          making the API self-documenting and easy to test.
        - Pydantic integration: Type-safe request/response validation with
          clear error messages for malformed inputs.
        - Performance: One of the fastest Python web frameworks, comparable
          to Node.js and Go for I/O-bound workloads (TechEmpower benchmarks).

    Why a lifespan context manager for model loading?
        Loading a transformer model takes 2–5 seconds and ~500 MB of RAM.
        Loading it once at startup (via lifespan) and reusing it for all
        requests is essential. Loading per-request would make the API unusable
        (~5s latency per call). The lifespan pattern also ensures clean
        shutdown (model memory is freed when the server stops).

    Why CORS allow all origins?
        During development, the frontend (localhost:3000) and backend
        (localhost:8000) run on different ports. Without CORS headers, the
        browser blocks cross-origin requests. In production, you'd restrict
        this to your actual domain.

Endpoints:
    GET  /          — Health check (is the API alive? is the model loaded?)
    POST /predict   — Classify an article as FAKE or REAL
    POST /explain   — Classify + generate token-level explanations (SHAP/LIME)
    GET  /metrics   — Return saved model evaluation metrics
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from src.config import BASE_DIR, MODEL_DIR, METRICS_DIR, LABEL_MAP

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Global state — populated during app lifespan startup.
# ──────────────────────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_model_loaded = False


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Models — Request/Response Schemas
# ──────────────────────────────────────────────────────────────────────────────

class NewsInput(BaseModel):
    """Request schema for the /predict endpoint.

    Interview: Why is title optional?
        Some use cases provide only the article body (e.g., social media posts,
        copy-pasted text). Making title optional improves API flexibility
        without degrading prediction quality significantly — the body text
        alone carries the majority of the discriminative signal.
    """
    title: Optional[str] = Field(
        default=None,
        description="Article headline (optional). Improves prediction quality.",
        examples=["Scientists Discover New Species in Amazon Rainforest"],
    )
    text: str = Field(
        ...,
        min_length=10,
        description="Article body text (required). Minimum 10 characters.",
        examples=["A team of researchers from the National Geographic Society..."],
    )


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""
    label: str = Field(
        description="Predicted label: 'FAKE' or 'REAL'.",
        examples=["REAL"],
    )
    confidence: float = Field(
        description="Confidence score for the predicted label (0.0 to 1.0).",
        examples=[0.9523],
    )
    probabilities: dict[str, float] = Field(
        description="Probability for each class.",
        examples=[{"FAKE": 0.0477, "REAL": 0.9523}],
    )


class ExplanationRequest(BaseModel):
    """Request schema for the /explain endpoint."""
    title: Optional[str] = Field(
        default=None,
        description="Article headline (optional).",
    )
    text: str = Field(
        ...,
        min_length=10,
        description="Article body text (required).",
    )
    method: str = Field(
        default='lime',
        description="Explanation method: 'shap' or 'lime'.",
        pattern='^(shap|lime)$',
    )
    num_features: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Number of top features to include in the explanation.",
    )


class TokenImportance(BaseModel):
    """A single token's importance score in an explanation."""
    token: str = Field(description="The word/token.")
    weight: float = Field(description="Importance weight (positive or negative).")


class ExplanationResponse(BaseModel):
    """Response schema for the /explain endpoint."""
    label: str
    confidence: float
    method: str
    explanations: list[TokenImportance]


class MetricsResponse(BaseModel):
    """Response schema for the /metrics endpoint."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    total_test_samples: Optional[int] = None
    confusion_matrix: Optional[list[list[int]]] = None


# ──────────────────────────────────────────────────────────────────────────────
# App Lifespan — Load model once at startup
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model at startup and clean up on shutdown.

    Interview: Why a lifespan context manager?
        The @app.on_event("startup") decorator is deprecated in modern FastAPI.
        The lifespan pattern is the recommended approach because:
            1. It clearly separates startup and shutdown logic.
            2. It supports async operations.
            3. Resources allocated during startup are guaranteed to be cleaned
               up during shutdown (context manager protocol).
    """
    global _model, _tokenizer, _model_loaded

    logger.info("Starting up — loading model...")

    try:
        from src.predict import load_predictor
        _model, _tokenizer = load_predictor()
        _model_loaded = True
        logger.info("Model loaded successfully at startup.")
    except FileNotFoundError:
        logger.warning(
            "No trained model found at %s. "
            "The API will start but /predict and /explain will return errors. "
            "Train the model first: python -m src.train",
            MODEL_DIR,
        )
        _model_loaded = False
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        _model_loaded = False

    yield  # App is running and serving requests.

    # Cleanup on shutdown.
    logger.info("Shutting down — cleaning up resources.")
    _model = None
    _tokenizer = None
    _model_loaded = False


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fake News Detection API",
    description=(
        "REST API for classifying news articles as FAKE or REAL using a "
        "fine-tuned DistilBERT model. Supports prediction, explainability "
        "(SHAP & LIME), and model metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Interview: CORS (Cross-Origin Resource Sharing)
# During development, the React/Next.js frontend runs on localhost:3000 while
# the API runs on localhost:8000. Browsers enforce same-origin policy by default,
# blocking requests between different ports. CORSMiddleware adds the necessary
# headers to allow cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (restrict in production).
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods.
    allow_headers=["*"],          # Allow all headers.
)


# ──────────────────────────────────────────────────────────────────────────────
# Serving Static Frontend Files
# ──────────────────────────────────────────────────────────────────────────────
FRONTEND_DIR = BASE_DIR.parent / 'frontend'

# Mount static folders for CSS and JavaScript
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


# ──────────────────────────────────────────────────────────────────────────────
# Helper — Check model availability
# ──────────────────────────────────────────────────────────────────────────────

def _require_model():
    """Raise an HTTP 503 if the model is not loaded.

    This provides a clear error message when the user tries to make predictions
    before training the model, rather than crashing with a cryptic NoneType error.
    """
    if not _model_loaded or _model is None or _tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "message": (
                    "No trained model is available. Please train the model first "
                    "by running: python -m src.train"
                ),
                "hint": (
                    "After training completes, restart the API server to load "
                    "the new model."
                ),
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the premium dark frontend homepage index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Frontend index.html not found at: {index_path}"
        )
    return FileResponse(str(index_path))


@app.get("/api/health", tags=["Health"])
async def health_check() -> dict:
    """Health check endpoint.

    Returns the API status and whether a trained model is loaded.
    Useful for monitoring tools (e.g., Docker health checks, load balancers)
    to verify the service is operational.
    """
    return {
        "status": "healthy",
        "model_loaded": _model_loaded,
        "model_directory": str(MODEL_DIR),
        "available_endpoints": ["/predict", "/explain", "/metrics", "/api/health"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_news(input_data: NewsInput) -> PredictionResponse:
    """Classify a news article as FAKE or REAL.

    Accepts an article's title (optional) and body text, preprocesses them,
    runs inference through the DistilBERT model, and returns the prediction
    with confidence scores.

    Interview: Why async endpoint?
        Although the model inference itself is synchronous (CPU/GPU bound),
        FastAPI runs it in a thread pool automatically for async endpoints.
        This prevents a single slow prediction from blocking other incoming
        requests, improving overall API throughput under concurrent load.
    """
    _require_model()

    try:
        from src.predict import predict_from_title_text

        result = predict_from_title_text(
            title=input_data.title or "",
            text=input_data.text,
            model=_model,
            tokenizer=_tokenizer,
        )

        return PredictionResponse(
            label=result['label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_prediction(request: ExplanationRequest) -> ExplanationResponse:
    """Classify an article and explain the model's decision.

    Uses SHAP or LIME to generate token-level importance scores showing which
    words most influenced the prediction. This is essential for building user
    trust and enabling human review of the model's reasoning.

    Note: Explanations are computationally expensive. LIME typically takes
    2–5 seconds; SHAP can take 10–30 seconds depending on text length.
    """
    _require_model()

    try:
        from src.predict import predict_from_title_text
        from src.preprocessing import prepare_input
        from src.explainability import explain_with_shap, explain_with_lime

        # Get prediction.
        prediction = predict_from_title_text(
            title=request.title or "",
            text=request.text,
            model=_model,
            tokenizer=_tokenizer,
        )

        # Generate explanation.
        combined_text = prepare_input(request.title or "", request.text)

        if request.method == 'shap':
            explanations = explain_with_shap(
                combined_text, _model, _tokenizer, request.num_features
            )
        else:
            explanations = explain_with_lime(
                combined_text, _model, _tokenizer, request.num_features
            )

        return ExplanationResponse(
            label=prediction['label'],
            confidence=prediction['confidence'],
            method=request.method,
            explanations=[
                TokenImportance(token=e['token'], weight=e['weight'])
                for e in explanations
            ],
        )

    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=f"Explanation library not installed: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Explanation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {str(e)}",
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics() -> MetricsResponse:
    """Return saved model evaluation metrics.

    Reads the evaluation_metrics.json file generated by evaluate.py.
    If no metrics file exists, returns an appropriate error message
    guiding the user to run evaluation first.
    """
    metrics_path = Path(METRICS_DIR) / 'evaluation_metrics.json'

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Metrics not found",
                "message": (
                    "No evaluation metrics file found. Run evaluation first: "
                    "python -m src.evaluate"
                ),
                "path": str(metrics_path),
            },
        )

    try:
        with open(str(metrics_path), 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return MetricsResponse(
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1=metrics.get('f1', 0.0),
            total_test_samples=metrics.get('total_test_samples'),
            confusion_matrix=metrics.get('confusion_matrix'),
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Metrics file is corrupted. Re-run: python -m src.evaluate",
        )
    except Exception as e:
        logger.error("Failed to load metrics: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read metrics: {str(e)}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point: python -m src.api
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s — %(name)s — %(levelname)s — %(message)s',
    )
    logger.info("Starting Fake News Detection API server...")
    uvicorn.run(
        'src.api:app',
        host='0.0.0.0',
        port=8000,
        reload=True,
    )
