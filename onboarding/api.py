"""In-memory, authenticated mock API. No real scheduling system is connected."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import secrets
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class Booking(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    slot_id: str = Field(min_length=1, max_length=64)
    customer_ref: str = Field(pattern=r"^demo-[a-z0-9-]{1,48}$")
    consent: Literal[True]


class Escalation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: Literal["user_request", "unsupported_request", "tool_failure"]
    customer_ref: str = Field(pattern=r"^demo-[a-z0-9-]{1,48}$")


@dataclass
class Store:
    """Single-process demo state; reset when the app restarts."""
    slots: dict[str, str] = field(default_factory=lambda: {
        "slot-1": "Synthetic morning appointment", "slot-2": "Synthetic afternoon appointment"})
    bookings: dict[str, dict] = field(default_factory=dict)
    requests: dict[str, tuple[dict, dict]] = field(default_factory=dict)
    escalations: dict[tuple[str, str], dict] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)


def create_app(api_key: str | None = None) -> FastAPI:
    key = api_key if api_key is not None else os.environ.get("APPOINTMENT_API_KEY", "")
    if len(key) < 8:
        raise ValueError("Set APPOINTMENT_API_KEY to at least 8 characters")
    app = FastAPI(title="Synthetic Appointment API", version="0.1.0")
    store = Store()
    app.state.store = store

    def authorize(x_api_key: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_api_key.encode(), key.encode()):
            raise HTTPException(401, "Invalid API key")

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok", "mode": "synthetic"}

    @app.get("/availability", dependencies=[Depends(authorize)])
    def check_availability(service: Literal["onboarding"] = "onboarding") -> dict:
        with store.lock:
            return {"slots": [{"id": slot, "label": label} for slot, label in store.slots.items()
                              if slot not in store.bookings], "service": service}

    @app.post("/appointments", dependencies=[Depends(authorize)])
    def book_appointment(body: Booking, idempotency_key: str = Header(min_length=8, max_length=100)) -> dict:
        payload = body.model_dump()
        with store.lock:
            if idempotency_key in store.requests:
                previous, result = store.requests[idempotency_key]
                if previous != payload:
                    raise HTTPException(409, "Idempotency key reused with different request")
                return result
            if body.slot_id not in store.slots:
                raise HTTPException(404, "Unknown slot")
            if body.slot_id in store.bookings:
                raise HTTPException(409, "Slot unavailable")
            result = {"appointment_id": str(uuid4()), "slot_id": body.slot_id, "status": "booked"}
            store.bookings[body.slot_id] = result
            store.requests[idempotency_key] = (payload, result)
            return result

    @app.post("/escalations", dependencies=[Depends(authorize)])
    def escalate_to_human(body: Escalation) -> dict:
        with store.lock:
            pair = (body.customer_ref, body.reason)
            if pair not in store.escalations:
                store.escalations[pair] = {"ticket_id": str(uuid4()), "status": "queued", "reason": body.reason}
            return store.escalations[pair]

    return app
