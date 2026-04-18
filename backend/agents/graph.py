"""LangGraph pipeline, orchestrates all agents and the coordinator."""
from __future__ import annotations
import asyncio
import logging
from typing import Optional, List, TypedDict

from langgraph.graph import StateGraph, START, END

from core.models import Alert, LogEvent, VisionAssessment, AudioAssessment, LogAssessment
from agents import vision_agent, audio_agent, log_agent, coordinator

logger = logging.getLogger(__name__)

# State

class SecurityState(TypedDict):
    frame_b64: Optional[str]
    camera_id: Optional[str]
    audio_transcript: Optional[str]
    audio_source: str
    log_events: List[dict]
    vision_assessment: Optional[dict]
    audio_assessment: Optional[dict]
    log_assessment: Optional[dict]
    alert: Optional[dict]

# Nodes

async def agents_node(state: SecurityState) -> dict:
    """Run applicable agents concurrently and return their assessments."""
    tasks: dict[str, object] = {}

    if state.get("frame_b64"):
        tasks["vision"] = vision_agent.analyze_frame(
            state["frame_b64"], state.get("camera_id") or "unknown"
        )

    if state.get("audio_transcript"):
        tasks["audio"] = audio_agent.analyze_transcript(
            state["audio_transcript"], state.get("audio_source") or "intercom"
        )

    raw_events = state.get("log_events") or []
    if raw_events:
        for raw in raw_events:
            log_agent.ingest_event(LogEvent(**raw))
        tasks["log"] = asyncio.to_thread(log_agent.analyze_events)

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_map = dict(zip(tasks.keys(), results))

    output: dict = {}
    for key in ("vision", "audio", "log"):
        val = result_map.get(key)
        if val is not None and not isinstance(val, Exception):
            output[f"{key}_assessment"] = val.model_dump()
        elif isinstance(val, Exception):
            logger.error("%s agent raised: %s", key, val)

    return output


async def coordinator_node(state: SecurityState) -> dict:
    """Synthesise agent outputs into a final alert decision."""
    def _maybe(cls, key):
        data = state.get(key)
        return cls(**data) if data else None

    vision = _maybe(VisionAssessment, "vision_assessment")
    audio = _maybe(AudioAssessment, "audio_assessment")
    log = _maybe(LogAssessment, "log_assessment")

    should_alert, alert = await coordinator.coordinate(vision, audio, log)
    return {"alert": alert.model_dump() if alert else None}

# Graph

_workflow = StateGraph(SecurityState)
_workflow.add_node("agents", agents_node)
_workflow.add_node("coordinator", coordinator_node)
_workflow.add_edge(START, "agents")
_workflow.add_edge("agents", "coordinator")
_workflow.add_edge("coordinator", END)

pipeline = _workflow.compile()

# Public entry point

async def run_pipeline(
    frame_b64: Optional[str] = None,
    camera_id: Optional[str] = None,
    audio_transcript: Optional[str] = None,
    audio_source: str = "intercom",
    log_events: Optional[List[dict]] = None,
) -> Optional[Alert]:
    """Run the full security analysis pipeline and return an Alert or None."""
    initial_state: SecurityState = {
        "frame_b64": frame_b64,
        "camera_id": camera_id,
        "audio_transcript": audio_transcript,
        "audio_source": audio_source,
        "log_events": log_events or [],
        "vision_assessment": None,
        "audio_assessment": None,
        "log_assessment": None,
        "alert": None,
    }
    try:
        result = await pipeline.ainvoke(initial_state)
        alert_data = result.get("alert")
        if alert_data:
            return Alert(**alert_data)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
    return None
