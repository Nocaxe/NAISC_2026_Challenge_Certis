"""
Feedback & input router.

Handles:
  POST /api/audio        — submit audio transcript for analysis
  POST /api/logs/event   — submit an access-control / alarm event
  POST /api/feedback     — officer feedback on an alert (confirm / dismiss)
"""
import logging
from fastapi import APIRouter, HTTPException

from core.connections import manager
from core.feedback import save_alert, save_feedback, update_alert_status
from core.models import AudioInput, AlertFeedback, LogEvent
from agents.graph import run_pipeline

router = APIRouter(prefix="/api", tags=["inputs"])
logger = logging.getLogger(__name__)


@router.post("/audio")
async def submit_audio(body: AudioInput):
    """Analyse an intercom / distress-call transcript and raise an alert if warranted."""
    alert = await run_pipeline(
        audio_transcript=body.transcript,
        audio_source=body.source,
    )
    if alert:
        await save_alert(alert)
        await manager.broadcast({"type": "alert", "data": alert.model_dump(mode="json")})
        return {"status": "alert_raised", "alert_id": alert.alert_id}
    return {"status": "no_threat"}


@router.post("/logs/event")
async def submit_log_event(event: LogEvent):
    """Ingest an access-control or alarm event and run the log-analysis pipeline."""
    alert = await run_pipeline(log_events=[event.model_dump(mode="json")])
    if alert:
        await save_alert(alert)
        await manager.broadcast({"type": "alert", "data": alert.model_dump(mode="json")})
        return {"status": "alert_raised", "alert_id": alert.alert_id}
    return {"status": "no_threat"}


@router.post("/feedback")
async def submit_feedback(body: AlertFeedback):
    """Record officer feedback (confirm / dismiss) for an alert."""
    updated = await update_alert_status(body.alert_id, body.outcome)
    if not updated:
        raise HTTPException(status_code=404, detail="Alert not found.")
    await save_feedback(body.alert_id, body.outcome, body.officer_note)
    return {"status": "ok"}
