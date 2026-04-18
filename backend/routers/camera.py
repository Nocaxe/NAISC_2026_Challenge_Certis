"""Camera router"""
import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import settings
from core.connections import manager
from core.feedback import save_alert
from agents.graph import run_pipeline

router = APIRouter(tags=["camera"])
logger = logging.getLogger(__name__)


@dataclass
class CameraState:
    last_analysis: float = 0.0
    last_alert_category: Optional[str] = None
    last_alert_time: float = 0.0


_camera_state: dict[str, CameraState] = {}


@router.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """
    Client → Server: {"type": "frame", "camera_id": "webcam_1", "data": "<base64 JPEG>"}
    Server → Client: {"type": "ack"} | {"type": "alert", "data": <Alert JSON>}

    Analysis runs in a background task so the receive loop is never blocked,
    which keeps WebSocket keepalive ping/pong working during long LLM calls.
    """
    await websocket.accept()
    _pending: dict[str, asyncio.Task] = {}

    async def _analyse(frame_b64: str, camera_id: str, state: CameraState, t: float) -> None:
        logger.debug("Analysing frame from %s", camera_id)
        alert = await run_pipeline(frame_b64=frame_b64, camera_id=camera_id)
        try:
            if alert:
                same_category = alert.category == state.last_alert_category
                within_cooldown = (t - state.last_alert_time) < settings.alert_cooldown
                if same_category and within_cooldown:
                    logger.debug(
                        "Suppressed duplicate alert '%s' for %s (cooldown: %.0fs remaining)",
                        alert.category, camera_id, settings.alert_cooldown - (t - state.last_alert_time),
                    )
                    return
                state.last_alert_category = alert.category
                state.last_alert_time = t
                alert.camera_id = camera_id
                alert.frame_snapshot = frame_b64
                await save_alert(alert)
                payload = {"type": "alert", "data": alert.model_dump(mode="json")}
                await websocket.send_json(payload)
                await manager.broadcast(payload)
            else:
                if state.last_alert_category:
                    logger.debug("Threat resolved on %s, cooldown cleared.", camera_id)
                    state.last_alert_category = None
                    state.last_alert_time = 0.0
        except Exception as exc:
            logger.error("Error sending analysis result for %s: %s", camera_id, exc)

    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") != "frame":
                continue

            camera_id: str = msg.get("camera_id", "webcam_1")
            frame_b64: str = msg.get("data", "")
            now = time.time()
            state = _camera_state.setdefault(camera_id, CameraState())

            if now - state.last_analysis < settings.frame_analysis_interval:
                await websocket.send_json({"type": "ack"})
                continue

            # Skip if a pipeline run is already in progress for this camera
            task = _pending.get(camera_id)
            if task and not task.done():
                await websocket.send_json({"type": "ack"})
                continue

            state.last_analysis = now
            _pending[camera_id] = asyncio.create_task(
                _analyse(frame_b64, camera_id, state, now)
            )
            await websocket.send_json({"type": "ack"})

    except WebSocketDisconnect:
        logger.info("Camera WebSocket disconnected: %s", websocket.client)
    except Exception as exc:
        logger.error("Camera WebSocket error: %s", exc)
    finally:
        for task in _pending.values():
            task.cancel()
