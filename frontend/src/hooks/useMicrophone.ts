import { useCallback, useEffect, useRef, useState } from "react";

// Minimum character length before sending a transcript to the backend
const MIN_TRANSCRIPT_LENGTH = 10;

export interface TranscriptEntry {
  text: string;
  timestamp: Date;
  status: "sending" | "sent" | "error";
}

export function useMicrophone() {
  const [active, setActive] = useState(false);
  const [supported, setSupported] = useState(false);
  const [interim, setInterim] = useState(""); // live in-progress text
  const [history, setHistory] = useState<TranscriptEntry[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    const SR = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    setSupported(!!SR);
  }, []);

  const sendTranscript = useCallback(async (text: string) => {
    const entry: TranscriptEntry = { text, timestamp: new Date(), status: "sending" };
    setHistory((prev) => [entry, ...prev].slice(0, 20));

    try {
      await fetch("/api/audio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: text, source: "microphone" }),
      });
      setHistory((prev) =>
        prev.map((e) => (e === entry ? { ...e, status: "sent" } : e))
      );
    } catch {
      setHistory((prev) =>
        prev.map((e) => (e === entry ? { ...e, status: "error" } : e))
      );
    }
  }, []);

  const start = useCallback(() => {
    const SR = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SR) return;

    const recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-SG"; // Singapore English, falls back to en-US

    recognition.onstart = () => {
      setActive(true);
      setErrorMsg(null);
    };

    recognition.onresult = (event) => {
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          const finalText = result[0].transcript.trim();
          if (finalText.length >= MIN_TRANSCRIPT_LENGTH) {
            sendTranscript(finalText);
          }
          setInterim("");
        } else {
          interimText += result[0].transcript;
        }
      }
      if (interimText) setInterim(interimText);
    };

    recognition.onerror = (event) => {
      if (event.error === "no-speech") return;
      setErrorMsg(`Microphone error: ${event.error}`);
      setActive(false);
    };

    recognition.onend = () => {
      // Auto-restart if still supposed to be active
      if (recognitionRef.current === recognition) {
        try { recognition.start(); } catch { setActive(false); }
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [sendTranscript]);

  const stop = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.onend = null; // prevent auto restart
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setActive(false);
    setInterim("");
  }, []);

  return { active, supported, interim, history, errorMsg, start, stop };
}
