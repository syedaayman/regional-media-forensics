export interface HealthResponse {
  status: string;
}

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export async function checkBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${apiBaseUrl}/health`);

  if (!response.ok) {
    throw new Error(`Backend returned HTTP ${response.status}`);
  }

  const health = (await response.json()) as HealthResponse;

  if (health.status !== "ok") {
    throw new Error("Backend health endpoint returned an unexpected response");
  }

  return health;
}
