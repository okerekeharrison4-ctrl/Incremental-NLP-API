"""FastAPI app for intent clarification.

This template uses a Hugging Face zero-shot model to detect intent labels and
returns a clarifying question when confidence is below a configurable threshold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from transformers import pipeline

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Confidence threshold can be overridden with an environment variable.
# Example: INTENT_CONFIDENCE_THRESHOLD=0.75
CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.70"))

# Session storage path can be overridden if you deploy with persistent storage.
SESSION_STORE_PATH = Path(os.getenv("SESSION_STORE_PATH", "sessions.json"))

# Intent labels required by the API contract.
INTENT_LABELS = ["purchase", "support", "inquiry", "other"]

# Initialize FastAPI app.
app = FastAPI(
    title="Intent Clarification API",
    description="Detects user intent and asks a clarifying question when needed.",
    version="1.0.0",
)


# -----------------------------------------------------------------------------
# Request/Response models
# -----------------------------------------------------------------------------

class ClarifyIntentRequest(BaseModel):
    """Incoming request for intent clarification."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="User input text")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID to preserve multi-turn context",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Ensure text is non-empty after trimming whitespace."""
        clean = value.strip()
        if not clean:
            raise ValueError("text must not be empty")
        return clean


class ClarifyIntentResponse(BaseModel):
    """Response object returned by the API."""

    detected_intent: str
    confidence: float
    clarifying_question: str | None
    session_id: str


# -----------------------------------------------------------------------------
# Session storage (JSON file-based)
# -----------------------------------------------------------------------------

class SessionStore:
    """Simple JSON-backed session store for request history.

    This is intentionally lightweight for MVP use. For production, consider
    replacing with Redis or a database.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> dict[str, Any]:
        with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Recover from malformed file by resetting to empty state.
                return {}

    def _write_all(self, data: dict[str, Any]) -> None:
        with self._lock:
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def append_interaction(self, session_id: str, interaction: dict[str, Any]) -> None:
        """Append an interaction record under a session ID."""
        data = self._read_all()
        history = data.get(session_id, [])
        history.append(interaction)
        data[session_id] = history
        self._write_all(data)

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return previous interactions for the session."""
        data = self._read_all()
        return data.get(session_id, [])


session_store = SessionStore(SESSION_STORE_PATH)


# -----------------------------------------------------------------------------
# NLP pipeline setup
# -----------------------------------------------------------------------------

# We use a zero-shot classifier to map free text to required intent labels.
# Model is open-source and downloaded automatically the first time it is used.
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def detect_intent(text: str) -> tuple[str, float, list[tuple[str, float]]]:
    """Run inference and return top label, score, and all ranked candidates."""
    result = classifier(text, candidate_labels=INTENT_LABELS, multi_label=False)

    ranked = list(zip(result["labels"], result["scores"]))
    top_label, top_score = ranked[0]
    return top_label, float(top_score), ranked


def build_clarifying_question(
    ranked_candidates: list[tuple[str, float]],
    session_history: list[dict[str, Any]],
) -> str:
    """Generate a simple clarifying question using top intent candidates."""
    top_3 = [label for label, _ in ranked_candidates[:3]]

    # Optionally include context from previous intent to make the question friendlier.
    if session_history:
        previous_intent = session_history[-1].get("detected_intent")
        return (
            f"Earlier, this seemed related to '{previous_intent}'. "
            f"Do you mean {top_3[0]}, {top_3[1]}, or {top_3[2]}?"
        )

    return f"To help me route this correctly, do you mean {top_3[0]}, {top_3[1]}, or {top_3[2]}?"


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic health endpoint for deployment platforms."""
    return {"status": "ok"}


@app.post("/clarify-intent", response_model=ClarifyIntentResponse)
def clarify_intent(payload: ClarifyIntentRequest) -> ClarifyIntentResponse:
    """Classify intent and return clarifying question if confidence is low."""
    try:
        session_id = payload.session_id or str(uuid4())
        history = session_store.get_history(session_id)

        detected_intent, confidence, ranked = detect_intent(payload.text)

        clarifying_question = None
        if confidence < CONFIDENCE_THRESHOLD:
            clarifying_question = build_clarifying_question(ranked, history)

        record = {
            "text": payload.text,
            "detected_intent": detected_intent,
            "confidence": confidence,
            "clarifying_question": clarifying_question,
        }
        session_store.append_interaction(session_id, record)

        return ClarifyIntentResponse(
            detected_intent=detected_intent,
            confidence=round(confidence, 4),
            clarifying_question=clarifying_question,
            session_id=session_id,
        )
    except ValueError as exc:
        # Handles custom validation/data errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Generic catch for unexpected model/runtime failures.
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
