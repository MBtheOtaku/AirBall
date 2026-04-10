const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function getHealth() {
  if (!API_URL) {
    throw new Error("Missing NEXT_PUBLIC_API_URL");
  }
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`API error (${res.status})`);
  return res.json();
}

export function getApiBaseUrl() {
  return API_URL;
}
