import { useEffect, useRef, useState, useCallback } from "react";
import type { Alert } from "../types";

const WS_URL = "ws://localhost:8000/ws/camera";
const FRAME_INTERVAL_MS = 500; // send a frame every 500 ms; server throttles analysis

export function useCamera(onAlert: (alert: Alert) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<number | null>(null);

  const [active, setActive] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [analysisText, setAnalysisText] = useState<string>("");

  const stop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    wsRef.current?.close();
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
    setWsConnected(false);
  }, []);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      setAnalysisText("Camera permission denied.");
      return;
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      setActive(true);

      intervalRef.current = window.setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || ws.readyState !== WebSocket.OPEN) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d")!.drawImage(video, 0, 0);
        const frame = canvas.toDataURL("image/jpeg", 0.7);

        ws.send(JSON.stringify({ type: "frame", camera_id: "webcam_1", data: frame }));
      }, FRAME_INTERVAL_MS);
    };

    ws.onclose = () => {
      setWsConnected(false);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "alert") {
        onAlert(msg.data as Alert);
        setAnalysisText(`Threat detected: ${msg.data.title}`);
      } else if (msg.type === "ack") {
        setAnalysisText("Analysing...");
      }
    };
  }, [onAlert]);

  useEffect(() => () => stop(), [stop]);

  return { videoRef, canvasRef, active, wsConnected, analysisText, start, stop };
}
