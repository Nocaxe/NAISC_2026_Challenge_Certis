import { useMicrophone } from "../hooks/useMicrophone";
import { formatDistanceToNow } from "date-fns";

export default function MicrophoneFeed() {
  const { active, supported, interim, history, errorMsg, start, stop } = useMicrophone();

  if (!supported) {
    return (
      <div className="mt-6 border border-border rounded-lg p-4">
        <p className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-2">
          Live Microphone
        </p>
        <p className="text-xs text-gray-500">
          Not supported in this browser. Use Chrome or Edge.
        </p>
      </div>
    );
  }

  return (
    <div className="mt-6 border border-border rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs font-bold uppercase tracking-widest text-gray-500">
          Live Microphone
        </p>
        <div className="flex items-center gap-1.5 text-xs text-gray-400">
          <div
            className={`w-2 h-2 rounded-full ${
              active ? "bg-red-500 animate-pulse" : "bg-gray-600"
            }`}
          />
          {active ? "Listening" : "Off"}
        </div>
      </div>

      {/* Interim transcript — live preview */}
      <div className="min-h-[36px] mb-3 px-2 py-1.5 rounded bg-black/30 border border-border text-xs text-gray-400 italic">
        {interim || (active ? "Waiting for speech…" : "Start microphone to monitor audio.")}
      </div>

      {/* Error */}
      {errorMsg && (
        <p className="mb-2 text-xs text-red-400">{errorMsg}</p>
      )}

      {/* Start / Stop */}
      <button
        onClick={active ? stop : start}
        className={`w-full py-2 rounded text-sm font-medium ${
          active
            ? "bg-red-900/60 border border-red-700 text-red-300 hover:bg-red-800/60"
            : "bg-gray-700 hover:bg-gray-600 text-gray-200"
        }`}
      >
        {active ? "Stop Microphone" : "Start Microphone"}
      </button>

      {/* Sent transcript history */}
      {history.length > 0 && (
        <div className="mt-3 space-y-1.5">
          <p className="text-xs text-gray-600 uppercase tracking-widest">Recent</p>
          {history.map((entry, i) => (
            <div key={i} className="flex items-start gap-2 text-xs">
              <span
                className={`mt-0.5 shrink-0 w-1.5 h-1.5 rounded-full ${
                  entry.status === "sent"
                    ? "bg-green-500"
                    : entry.status === "error"
                    ? "bg-red-500"
                    : "bg-yellow-500 animate-pulse"
                }`}
              />
              <div className="flex-1 min-w-0">
                <p className="text-gray-300 truncate">{entry.text}</p>
                <p className="text-gray-600">
                  {formatDistanceToNow(entry.timestamp, { addSuffix: true })}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
