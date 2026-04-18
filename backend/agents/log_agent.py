"""Log Agent: rule-based analysis of access control and alarm events without requiring LLM
"""
from __future__ import annotations
from collections import deque
from datetime import datetime, timedelta
from typing import List

from core.models import LogEvent, LogAssessment, SeverityLevel

_event_buffer: deque[LogEvent] = deque(maxlen=500)

# Threat rules:  event_type -> (min_count, window_seconds, severity, threat_type)
RULES: dict[str, tuple[int, int, SeverityLevel, str]] = {
    "DOOR_FORCED":    (1,   60,  SeverityLevel.HIGH,     "FORCED_ENTRY"),
    "FIRE_ALARM":     (1,   60,  SeverityLevel.CRITICAL, "FIRE_EMERGENCY"),
    "PANIC_BUTTON":   (1,   60,  SeverityLevel.CRITICAL, "DISTRESS_CALL"),
    "LIFT_ALARM":     (1,   60,  SeverityLevel.HIGH,     "MEDICAL_EMERGENCY"),
    "GLASS_BREAK":    (1,   60,  SeverityLevel.HIGH,     "FORCED_ENTRY"),
    "MOTION_ALARM":   (1,  120,  SeverityLevel.MEDIUM,   "UNAUTHORIZED_ACCESS"),
    "ACCESS_DENIED":  (3,  300,  SeverityLevel.MEDIUM,   "UNAUTHORIZED_ACCESS"),
    "DOOR_PROPPED":   (1,  300,  SeverityLevel.LOW,      "UNAUTHORIZED_ACCESS"),
}


def ingest_event(event: LogEvent) -> None:
    """Add an event to the in-memory buffer."""
    _event_buffer.append(event)


def analyze_events(events: List[LogEvent] | None = None) -> LogAssessment:
    """
    Evaluate recent events against threat rules.

    If *events* is provided it is used directly (e.g. during pipeline runs).
    Otherwise the global buffer is used.
    """
    source = events if events is not None else list(_event_buffer)
    now = datetime.utcnow()

    best_match = None
    best_timestamp = None

    for event_type, (min_count, window_sec, severity, threat_type) in RULES.items():
        window_start = now - timedelta(seconds=window_sec)
        matching = [
            e for e in source
            if e.event_type == event_type and e.timestamp >= window_start
        ]
        if len(matching) >= min_count:
            latest = max(e.timestamp for e in matching)
            if best_timestamp is None or latest > best_timestamp:
                best_timestamp = latest
                best_match = (event_type, window_sec, severity, threat_type, matching)

    if best_match:
        event_type, window_sec, severity, threat_type, matching = best_match
        location = matching[-1].location
        return LogAssessment(
            threat_detected=True,
            threat_type=threat_type,
            confidence=0.95,
            description=f"{event_type.replace('_', ' ').title()} detected at {location}.",
            evidence=[
                f"{len(matching)}x {event_type} in last {window_sec}s at {location}",
            ],
            severity=severity,
            location=location,
            triggered_rules=[event_type],
        )

    return LogAssessment(
        threat_detected=False,
        confidence=1.0,
        description="No anomalous log events detected.",
    )
