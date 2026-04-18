import { useCallback } from "react";
import type { Alert } from "../types";
import { useAlerts } from "../hooks/useAlerts";
import Header from "./Header";
import AlertFeed from "./AlertFeed";
import CameraFeed from "./CameraFeed";
import MicrophoneFeed from "./MicrophoneFeed";

export default function Dashboard() {
  const { alerts, connected, sendFeedback } = useAlerts();

  const handleAlert = useCallback(
    (alert: Alert) => {
      // Camera hook emits alert; add it to the feed (already handled via WS broadcast,
      // but this ensures it appears immediately for the camera client too)
      void alert; // alerts arrive via useAlerts WS subscription
    },
    []
  );

  const activeCount = alerts.filter((a) => a.status === "active").length;

  return (
    <div className="flex flex-col h-screen bg-surface">
      <Header alertCount={activeCount} wsConnected={connected} />

      <div className="flex flex-1 overflow-hidden gap-0">
        {/* Left panel — camera */}
        <div className="w-[420px] shrink-0 p-4 border-r border-border overflow-y-auto">
          <CameraFeed onAlert={handleAlert} />
          <MicrophoneFeed />

          {/* Manual input section for demo */}
          <DemoPanel />
        </div>

        {/* Right panel — alerts */}
        <div className="flex-1 p-4 overflow-hidden">
          <AlertFeed alerts={alerts} onFeedback={sendFeedback} />
        </div>
      </div>
    </div>
  );
}

/** Lets the demo operator inject log events and audio transcripts manually. */
function DemoPanel() {
  const submitLog = async (eventType: string, location: string) => {
    await fetch("/api/logs/event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_type: eventType, location }),
    });
  };

  const submitAudio = async (transcript: string) => {
    await fetch("/api/audio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, source: "intercom" }),
    });
  };

  return (
    <div className="mt-6 border border-border rounded-lg p-4">
      <p className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-3">
        Demo Triggers
      </p>
      <div className="flex flex-col gap-2">
        <button
          onClick={() => submitLog("FIRE_ALARM", "Level 3 — Server Room")}
          className="text-xs px-3 py-1.5 rounded bg-red-900/50 border border-red-700 text-red-300 hover:bg-red-800/50"
        >
          Fire Alarm — Level 3
        </button>
        <button
          onClick={() => submitLog("DOOR_FORCED", "Basement Car Park Exit")}
          className="text-xs px-3 py-1.5 rounded bg-orange-900/50 border border-orange-700 text-orange-300 hover:bg-orange-800/50"
        >
          Forced Entry — Car Park
        </button>
        <button
          onClick={() => submitLog("PANIC_BUTTON", "Reception Lobby")}
          className="text-xs px-3 py-1.5 rounded bg-red-900/50 border border-red-700 text-red-300 hover:bg-red-800/50"
        >
          Panic Button — Lobby
        </button>
        <button
          onClick={() => {
            submitLog("ACCESS_DENIED", "Level 5 Server Room");
            setTimeout(() => submitLog("ACCESS_DENIED", "Level 5 Server Room"), 500);
            setTimeout(() => submitLog("ACCESS_DENIED", "Level 5 Server Room"), 1000);
          }}
          className="text-xs px-3 py-1.5 rounded bg-yellow-900/50 border border-yellow-700 text-yellow-300 hover:bg-yellow-800/50"
        >
          3× Access Denied — Level 5
        </button>
        <button
          onClick={() =>
            submitAudio(
              "Help! There's a man with a knife on level 2 near the lift lobby. Please send security now!"
            )
          }
          className="text-xs px-3 py-1.5 rounded bg-orange-900/50 border border-orange-700 text-orange-300 hover:bg-orange-800/50"
        >
          Distress Call — Knife Threat
        </button>
        <button
          onClick={() =>
            submitAudio(
              "Someone collapsed in the food court on level 1. They're unconscious and not responding."
            )
          }
          className="text-xs px-3 py-1.5 rounded bg-blue-900/50 border border-blue-700 text-blue-300 hover:bg-blue-800/50"
        >
          Medical Emergency — Food Court
        </button>
      </div>
    </div>
  );
}
