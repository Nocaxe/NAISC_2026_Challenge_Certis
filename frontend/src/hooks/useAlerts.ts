import { useEffect, useRef, useState, useCallback } from "react";
import type { Alert, AlertFeedback } from "../types";

const WS_URL = "ws://localhost:8000/api/ws/alerts";

export function useAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Load history on mount
  useEffect(() => {
    fetch("/api/alerts")
      .then((r) => r.json())
      .then((data: Alert[]) => setAlerts(data))
      .catch(() => {});
  }, []);

  // Connect WebSocket
  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000); // reconnect
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "alert") {
          const alert = msg.data as Alert;
          setAlerts((prev) =>
            prev.some((a) => a.alert_id === alert.alert_id)
              ? prev
              : [alert, ...prev]
          );
        }
      };
    }
    connect();
    return () => wsRef.current?.close();
  }, []);

  const sendFeedback = useCallback((feedback: AlertFeedback) => {
    // Optimistic update
    setAlerts((prev) =>
      prev.map((a) =>
        a.alert_id === feedback.alert_id ? { ...a, status: feedback.outcome } : a
      )
    );
    wsRef.current?.send(JSON.stringify({ type: "feedback", data: feedback }));
  }, []);

  return { alerts, connected, sendFeedback };
}
