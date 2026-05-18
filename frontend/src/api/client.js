const API = "";

export async function apiGet(path) {
  const res = await fetch(`${API}${path}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

export async function apiPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

export async function fetchJobProgress(jobId) {
  return apiGet(`/api/jobs/${jobId}/progress`);
}

export async function fetchJobResult(jobId) {
  return apiGet(`/api/jobs/${jobId}/result`);
}

export async function pollJobUntilDone(jobId, onProgress) {
  const delay = (ms) => new Promise((r) => setTimeout(r, ms));
  while (true) {
    const snap = await fetchJobProgress(jobId);
    onProgress?.(snap);
    if (snap.status === "completed") {
      return fetchJobResult(jobId);
    }
    if (snap.status === "failed") {
      throw new Error(snap.error || "Job failed");
    }
    await delay(350);
  }
}
