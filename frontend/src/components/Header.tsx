interface Props {
  alertCount: number;
  wsConnected: boolean;
}

export default function Header({ alertCount, wsConnected }: Props) {
  return (
    <header className="flex items-center justify-between px-6 py-3 bg-panel border-b border-border">
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
        <span className="text-lg font-bold tracking-widest uppercase text-gray-100">
          Security Operations
        </span>
      </div>
      <div className="flex items-center gap-6 text-sm text-gray-400">
        <span>
          Active alerts:{" "}
          <span className={alertCount > 0 ? "text-red-400 font-bold" : "text-gray-300"}>
            {alertCount}
          </span>
        </span>
        <div className="flex items-center gap-1.5">
          <div
            className={`w-2 h-2 rounded-full ${wsConnected ? "bg-green-500" : "bg-gray-600"}`}
          />
          <span>{wsConnected ? "Live" : "Disconnected"}</span>
        </div>
      </div>
    </header>
  );
}
