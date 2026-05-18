import { useCallback, useEffect, useState } from "react";
import { apiGet } from "../api/client";

export function useResults(tool) {
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!tool) return;
    setLoading(true);
    try {
      const data = await apiGet(`/api/results/options?tool=${encodeURIComponent(tool)}`);
      setOptions(data.options || []);
    } catch {
      setOptions([]);
    } finally {
      setLoading(false);
    }
  }, [tool]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { options, loading, refresh };
}

export function formatResultLabel(opt) {
  const date = opt.created_at
    ? new Date(opt.created_at).toLocaleString(undefined, {
        dateStyle: "short",
        timeStyle: "short",
      })
    : "";
  return `${opt.label}${date ? ` · ${date}` : ""}`;
}
