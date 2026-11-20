import { useEffect, useState } from "react";
import { checkBackendHealth } from "./api";
import "./style.css";

type ConnectionState = "checking" | "connected" | "disconnected";

export default function App() {
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("checking");
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    checkBackendHealth()
      .then(() => setConnectionState("connected"))
      .catch((error: unknown) => {
        setConnectionState("disconnected");
        setErrorMessage(
          error instanceof Error ? error.message : "An unexpected error occurred",
        );
      });
  }, []);

  return (
    <main className="status-card">
      <p className="eyebrow">Regional Media Forensics</p>
      <h1>Backend connection</h1>
      <p
        className={`connection-status ${connectionState}`}
        role="status"
        aria-live="polite"
      >
        {connectionState === "checking" && "Checking backend…"}
        {connectionState === "connected" && "Connected"}
        {connectionState === "disconnected" && "Not connected"}
      </p>
      {errorMessage && <p className="error-message">{errorMessage}</p>}
      <p className="help-text">
        The frontend checks the FastAPI health endpoint when this page loads.
      </p>
    </main>
  );
}
