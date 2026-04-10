"use client";

import { useEffect, useState } from "react";
import { getApiBaseUrl, getHealth } from "./app";

type HealthResponse = {
  status: string;
};

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    const checkHealth = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getHealth();
        if (mounted) {
          setHealth(data);
        }
      } catch (fetchError) {
        if (mounted) {
          setHealth(null);
          setError(fetchError instanceof Error ? fetchError.message : "Unknown error");
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    checkHealth();
    return () => {
      mounted = false;
    };
  }, []);

  const apiBaseUrl = getApiBaseUrl();

  return (
    <main className="min-h-screen bg-background text-foreground px-6 py-10">
      <section className="mx-auto w-full max-w-5xl space-y-6">
        <header>
          <h1 className="text-3xl font-semibold">AirBall Dashboard</h1>
          <p className="text-sm text-muted-foreground">Live backend status and camera stream preview.</p>
        </header>

        <div className="rounded-lg border border-border p-4">
          <h2 className="text-lg font-medium">Backend Health</h2>
          <p className="mt-2 text-sm">API: {apiBaseUrl}</p>
          {loading && <p className="mt-2 text-sm">Checking health...</p>}
          {!loading && error && <p className="mt-2 text-sm">Error: {error}</p>}
          {!loading && health && <p className="mt-2 text-sm">Status: {health.status}</p>}
        </div>

        <div className="rounded-lg border border-border p-4">
          <h2 className="text-lg font-medium">Live Stream</h2>
          <img
            src={`${apiBaseUrl}/video_feed`}
            alt="AirBall live stream"
            className="mt-3 h-auto w-full rounded-md border border-border"
          />
        </div>
      </section>
    </main>
  );
}
