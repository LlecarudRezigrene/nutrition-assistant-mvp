"""Pydantic models for the WhatsApp intake tables (see db/04_whatsapp.sql).

These mirror the DB rows and are used by the FastAPI service for validation
and by console pages for typed access. The main app's SQLAlchemy models in
streamlit_app.py are untouched — these cover only the WhatsApp additions.
"""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Direction(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class MediaNote(str, Enum):
    """Receipt-only log of unsupported media — content is never downloaded."""
    VOICE = "voice"
    IMAGE = "image"
    PDF = "pdf"
    OTHER = "other"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    REJECTED = "rejected"
    SEND_FAILED = "send_failed"


class IntakeStatus(str, Enum):
    NEW = "new"
    PATIENT_CREATED = "patient_created"
    DUPLICATE = "duplicate"


class WhatsAppMessage(BaseModel):
    id: int | None = None
    owner_id: str  # nutritionist's auth uuid — service writes must stamp it
    wa_number: str  # E.164, e.g. +5215512345678
    direction: Direction
    body: str | None = None
    media_note: MediaNote | None = None
    twilio_sid: str | None = None
    patient_id: int | None = None
    created_at: datetime | None = None


class PendingApproval(BaseModel):
    id: int | None = None
    owner_id: str
    wa_number: str
    draft_text: str
    approved_text: str | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime | None = None
    resolved_at: datetime | None = None


class IntakeSubmission(BaseModel):
    id: int | None = None
    owner_id: str
    wa_number: str
    form_data: dict[str, Any] = Field(default_factory=dict)
    consent_accepted_at: datetime
    patient_id: int | None = None
    status: IntakeStatus = IntakeStatus.NEW
    seen_by_nutritionist: bool = False
    created_at: datetime | None = None
