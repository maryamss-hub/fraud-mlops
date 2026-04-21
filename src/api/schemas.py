"""Pydantic schemas for the fraud detection inference API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for a single transaction prediction."""

    features: Dict[str, Any] = Field(
        ...,
        description="Transaction features as a JSON object (feature_name -> value).",
        examples=[
            {
                "TransactionAmt": 123.45,
                "card1": 1000,
                "DeviceType": "desktop",
            }
        ],
    )


class PredictResponse(BaseModel):
    """Response schema for fraud prediction."""

    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., description="Binary fraud prediction (1=fraud).")
    model_version: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool

