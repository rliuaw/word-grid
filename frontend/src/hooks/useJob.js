import { useCallback, useState } from "react";
import { pollJobUntilDone } from "../api/client";

export function useJob() {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);

  const run = useCallback(async (startFn) => {
    setRunning(true);
    setError(null);
    setResult(null);
    setProgress({ percent: 0, message: "Starting…", stage: "init" });
    try {
      const { job_id } = await startFn();
      setJobId(job_id);
      const final = await pollJobUntilDone(job_id, (snap) => {
        setProgress(snap.progress);
      });
      setResult(final.result);
      return final.result;
    } catch (e) {
      setError(e.message);
      throw e;
    } finally {
      setRunning(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setProgress(null);
    setJobId(null);
  }, []);

  return { run, running, progress, result, error, jobId, reset };
}
