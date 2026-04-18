"""Coordinator Agent — fuses agent assessments and generates actionable alerts."""
from __future__ import annotations
import logging
from typing import Optional, Tuple

import anthropic

from core.config import settings
from core.models import Alert, VisionAssessment, AudioAssessment, LogAssessment, SeverityLevel
from core.sop_rag import SOP_GUIDELINES
from agents.utils import parse_llm_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are the Security Operations Coordinator AI for a commercial building.
You receive threat assessments from specialist agents and must decide whether to raise a security alert.

{SOP_GUIDELINES}

When generating an alert, base your recommended_actions strictly on the SOPs above.
Use Singapore emergency numbers: Police 999 · Ambulance/Fire (SCDF) 995.

Respond ONLY with valid JSON (no markdown fences):
{{
  "should_alert": <bool>,
  "severity": <"critical"|"high"|"medium"|"low">,
  "category": <threat category string>,
  "title": "<concise title, max 10 words>",
  "description": "<2–3 sentences explaining the situation>",
  "evidence": ["<evidence point>", "..."],
  "recommended_actions": ["<action 1>", "..."],
  "confidence": <0.0–1.0>,
  "location": "<best-known location>"
}}

Only set should_alert=true when confidence >= 0.65 and evidence clearly indicates an active incident.
Do not alert for normal building activity or ambiguous footage."""




def _to_str_list(items: list) -> list[str]:
    """Flatten a list that may contain strings or single-key dicts into plain strings."""
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.extend(str(v) for v in item.values())
    return out


def _build_context(
    vision: Optional[VisionAssessment],
    audio: Optional[AudioAssessment],
    log: Optional[LogAssessment],
) -> str:
    parts = []
    if vision:
        parts.append(
            f"VISION AGENT (confidence {vision.confidence:.0%}): {vision.description}\n"
            f"  Threat detected: {vision.threat_detected} | Type: {vision.threat_type} | "
            f"Severity: {vision.severity}\n"
            f"  Evidence: {'; '.join(vision.evidence)}"
        )
    if audio:
        parts.append(
            f"AUDIO AGENT (confidence {audio.confidence:.0%}): {audio.description}\n"
            f"  Threat detected: {audio.threat_detected} | Type: {audio.threat_type} | "
            f"Severity: {audio.severity}\n"
            f"  Evidence: {'; '.join(audio.evidence)}"
        )
    if log:
        parts.append(
            f"LOG AGENT (confidence {log.confidence:.0%}): {log.description}\n"
            f"  Threat detected: {log.threat_detected} | Type: {log.threat_type} | "
            f"Severity: {log.severity}\n"
            f"  Evidence: {'; '.join(log.evidence)}"
        )
    return "\n\n".join(parts)


async def coordinate(
    vision: Optional[VisionAssessment],
    audio: Optional[AudioAssessment],
    log: Optional[LogAssessment],
) -> Tuple[bool, Optional[Alert]]:
    """Synthesise agent assessments into a single alert decision."""

    # Shortcircuit: if no agent detected anything, skip the LLM call
    any_threat = any([
        vision and vision.threat_detected,
        audio and audio.threat_detected,
        log and log.threat_detected,
    ])
    if not any_threat:
        return False, None

    context = _build_context(vision, audio, log)
    user_msg = f"Agent assessments:\n\n{context}\n\nShould I raise a security alert? If yes, provide full details."

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        message = await client.messages.create(
            model=settings.claude_model,
            max_tokens=700,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        result = parse_llm_json(message.content[0].text)
    except Exception as exc:
        logger.error("Coordinator error: %s", exc)
        return _rule_based_alert(vision, audio, log)

    if not result.get("should_alert"):
        return False, None

    contributing = []
    if vision and vision.threat_detected:
        contributing.append("Vision")
    if audio and audio.threat_detected:
        contributing.append("Audio")
    if log and log.threat_detected:
        contributing.append("Logs")

    alert = Alert(
        severity=SeverityLevel(result["severity"]),
        category=result["category"],
        title=result["title"],
        description=result["description"],
        evidence=_to_str_list(result.get("evidence", [])),
        recommended_actions=_to_str_list(result.get("recommended_actions", [])),
        contributing_agents=contributing,
        confidence=result["confidence"],
        location=result.get("location", "Unknown"),
    )
    return True, alert


def _rule_based_alert(
    vision: Optional[VisionAssessment],
    audio: Optional[AudioAssessment],
    log: Optional[LogAssessment],
) -> Tuple[bool, Optional[Alert]]:
    """Fallback alert when the LLM coordinator is unavailable."""
    # Pick the highest severity threat
    candidates = [a for a in [vision, audio, log] if a and a.threat_detected]
    if not candidates:
        return False, None

    severity_order = {
        SeverityLevel.CRITICAL: 0,
        SeverityLevel.HIGH: 1,
        SeverityLevel.MEDIUM: 2,
        SeverityLevel.LOW: 3,
        None: 4,
    }
    best = min(candidates, key=lambda a: severity_order.get(a.severity, 4))

    contributing = []
    if vision and vision.threat_detected:
        contributing.append("Vision")
    if audio and audio.threat_detected:
        contributing.append("Audio")
    if log and log.threat_detected:
        contributing.append("Logs")

    alert = Alert(
        severity=best.severity or SeverityLevel.MEDIUM,
        category=best.threat_type or "SUSPICIOUS_BEHAVIOR",
        title=f"Security Alert: {(best.threat_type or 'Incident').replace('_', ' ').title()}",
        description=best.description,
        evidence=best.evidence,
        recommended_actions=["Dispatch officer to location.", "Assess situation.", "Contact supervisor."],
        contributing_agents=contributing,
        confidence=best.confidence,
        location=best.location or "Unknown",
    )
    return True, alert
