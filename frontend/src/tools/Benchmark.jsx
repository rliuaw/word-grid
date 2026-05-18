import { useEffect, useState } from "react";
import { apiGet, apiPost } from "../api/client";
import ResultSelect from "../components/ResultSelect";
import ToolPanel, { Field, Output, PrimaryButton, SecondaryButton } from "../components/ToolPanel";

export default function Benchmark({ initialGalleryId = "" }) {
  const [galleryId, setGalleryId] = useState(initialGalleryId);

  useEffect(() => {
    if (initialGalleryId) setGalleryId(initialGalleryId);
  }, [initialGalleryId]);
  const [prompt, setPrompt] = useState("");
  const [score, setScore] = useState(7);
  const [special, setSpecial] = useState(false);
  const [saved, setSaved] = useState(null);

  const loadPrompt = async () => {
    if (!galleryId) return;
    const data = await apiGet(`/api/benchmark/${galleryId}/prompt`);
    setPrompt(data.prompt);
  };

  const saveScore = async (e) => {
    e.preventDefault();
    if (!galleryId) return;
    const data = await apiPost(`/api/benchmark/${galleryId}/score`, {
      score: +score,
      special,
    });
    setSaved(data.payload?.metadata);
  };

  return (
    <ToolPanel
      title="Benchmarking"
      description="Gemini scoring prompt and manual benchmark scores for gallery grids."
    >
      <ResultSelect
        tool="gallery"
        value={galleryId}
        onChange={setGalleryId}
        placeholder="Select gallery grid…"
        className="mb-4"
      />
      <SecondaryButton onClick={loadPrompt} disabled={!galleryId}>
        Generate Gemini prompt
      </SecondaryButton>
      <textarea
        className="mt-4 min-h-[200px] w-full rounded-md border border-slate-600 bg-slate-900 p-3 font-mono text-xs"
        value={prompt}
        readOnly
        placeholder="Prompt appears here…"
      />
      <form onSubmit={saveScore} className="mt-6 flex flex-wrap items-end gap-4">
        <Field label="Manual score (0–10)">
          <input
            type="number"
            step="0.1"
            min={0}
            max={10}
            className="w-24 rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={score}
            onChange={(e) => setScore(e.target.value)}
          />
        </Field>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          <input type="checkbox" checked={special} onChange={(e) => setSpecial(e.target.checked)} />
          Mark as special grid
        </label>
        <PrimaryButton disabled={!galleryId}>Save score</PrimaryButton>
      </form>
      {saved && (
        <Output>
          <pre>{JSON.stringify(saved, null, 2)}</pre>
        </Output>
      )}
    </ToolPanel>
  );
}
