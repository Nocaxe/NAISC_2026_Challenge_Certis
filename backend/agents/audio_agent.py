"""Analyses text transcripts from intercoms and distress calls."""
import logging

import anthropic

from core.config import settings
from core.models import AudioAssessment
from agents.utils import parse_llm_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a security operations AI analysing transcripts from building intercoms,
panic buttons, and distress calls.

Determine whether the transcript indicates an active security incident.

Respond ONLY with valid JSON (no markdown fences):
{
  "threat_detected": <bool>,
  "threat_type": <"PHYSICAL_ALTERCATION"|"MEDICAL_EMERGENCY"|"FIRE_EMERGENCY"|
                   "DISTRESS_CALL"|"UNAUTHORIZED_ACCESS"|"SUSPICIOUS_BEHAVIOR"|null>,
  "confidence": <0.0–1.0>,
  "description": "<1–2 sentences summarising the situation>",
  "evidence": ["<key phrase or detail from transcript>", "..."],
  "severity": <"critical"|"high"|"medium"|"low"|null>,
  "location": "<location mentioned in transcript, or null>"
}

Be accurate. Ambiguous calls with no clear threat should return threat_detected=false."""


async def analyze_transcript(transcript: str, source: str = "intercom") -> AudioAssessment:
    """Analyse a text transcript and return a threat assessment."""
    user_msg = f"Source: {source}\nTranscript: {transcript}\n\nIs there an active security incident?"

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        message = await client.messages.create(
            model=settings.claude_model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        llm_result = AudioAssessment(**parse_llm_json(message.content[0].text), transcript=transcript)

        return llm_result

    except Exception as exc:
        logger.error("Audio agent error: %s", exc)
        return AudioAssessment(
            threat_detected=False,
            threat_type=None,
            confidence=0.0,
            description="Error analysing transcript",
            evidence=[],
            severity=None,
            location=None,
            transcript=transcript,
        )
