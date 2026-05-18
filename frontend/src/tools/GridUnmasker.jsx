import { useEffect, useState } from "react";
import { apiGet, apiPost } from "../api/client";
import { useJob } from "../hooks/useJob";
import ProgressBar from "../components/ProgressBar";
import ResultSelect from "../components/ResultSelect";
import GridTable from "../components/GridTable";
import ModelSelect from "../components/ModelSelect";
import ToolPanel, { Field, Output, PrimaryButton, SecondaryButton } from "../components/ToolPanel";

const DEFAULT_POS = JSON.stringify(
  [
    ["DT", "NN", "VB"],
    ["PRP", "VBD", "RB"],
    ["IN", "DT", "NN"],
  ],
  null,
  2
);

export default function GridUnmasker({ models }) {
  const [posJson, setPosJson] = useState(DEFAULT_POS);
  const [modelId, setModelId] = useState("bert-base-uncased");
  const [topK, setTopK] = useState(5);
  const [posGridResultId, setPosGridResultId] = useState("");
  const [session, setSession] = useState(null);
  const [override, setOverride] = useState({ row: 0, col: 0, word: "" });
  const job = useJob();

  useEffect(() => {
    if (!posGridResultId) return;
    apiGet(`/api/results/${posGridResultId}`)
      .then((rec) => {
        const grids = rec.payload?.grids;
        if (grids?.[0]) setPosJson(JSON.stringify(grids[0], null, 2));
      })
      .catch(() => {});
  }, [posGridResultId]);

  const parseGrid = () => JSON.parse(posJson);

  const applySession = (data) => setSession(data);

  const start = (autoRun = false) => {
    job.run(() =>
      apiPost("/api/tools/grid-unmask/start", {
        pos_grid: parseGrid(),
        model_id: modelId,
        top_k: +topK,
        auto_run: autoRun,
      })
    ).then(applySession);
  };

  const step = (path) => {
    if (!session?.session_id) return;
    job
      .run(() => apiPost(path, { session_id: session.session_id }))
      .then(applySession);
  };

  const doOverride = (e) => {
    e.preventDefault();
    if (!session?.session_id) return;
    job
      .run(() =>
        apiPost("/api/tools/grid-unmask/override", {
          session_id: session.session_id,
          row: +override.row,
          col: +override.col,
          word: override.word,
        })
      )
      .then(applySession);
  };

  const finalize = () => {
    if (!session?.session_id) return;
    job
      .run(() =>
        apiPost("/api/tools/grid-unmask/finalize", {
          session_id: session.session_id,
        })
      )
      .then((r) => alert(`Saved to gallery: ${r.gallery_id}`));
  };

  const lastStep = session?.steps?.[session.steps.length - 1];
  const highlight = lastStep ? [lastStep.row, lastStep.col] : null;

  return (
    <ToolPanel
      title="Grid Unmasker"
      description="Step through filling a POS grid (2N seed words, then BERT). Phase 1 prefills 2N words; BERT candidates are filtered by POS and exclude punctuation."
    >
      <ResultSelect
        tool="pos_grid"
        value={posGridResultId}
        onChange={setPosGridResultId}
        placeholder="Load POS grid from saved result…"
        className="mb-4"
      />
      <Field label="POS grid (JSON)">
        <textarea
          className="min-h-[100px] w-full rounded-md border border-slate-600 bg-slate-900 p-3 font-mono text-sm"
          value={posJson}
          onChange={(e) => setPosJson(e.target.value)}
        />
      </Field>
      <section className="mt-4 flex flex-wrap items-end gap-4">
        <ModelSelect models={models} value={modelId} onChange={setModelId} />
        <Field label="Top K">
          <input
            type="number"
            className="w-16 rounded-md border border-slate-600 bg-slate-900 px-2 py-2"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />
        </Field>
        <SecondaryButton onClick={() => start(false)} disabled={job.running}>
          Start
        </SecondaryButton>
        <SecondaryButton onClick={() => step("/api/tools/grid-unmask/step-forward")} disabled={!session || job.running}>
          Step →
        </SecondaryButton>
        <SecondaryButton onClick={() => step("/api/tools/grid-unmask/step-backward")} disabled={!session || job.running}>
          ← Step
        </SecondaryButton>
        <SecondaryButton onClick={() => start(true)} disabled={job.running}>
          Auto-fill
        </SecondaryButton>
        <PrimaryButton type="button" onClick={finalize} disabled={!session || job.running}>
          Save to gallery
        </PrimaryButton>
      </section>
      <ProgressBar progress={job.progress} running={job.running} />
      {job.error && <p className="mt-4 text-red-400">{job.error}</p>}
      {session && (
        <Output>
          <GridTable grid={session.pos_grid} title="POS tags" />
          <GridTable grid={session.word_grid} highlight={highlight} title="Words" />
          {lastStep && (
            <section className="mt-4 space-y-2 text-slate-300">
              <p>
                Step {lastStep.index + 1}: ({lastStep.row},{lastStep.col}) {lastStep.pos} →{" "}
                <strong className="text-emerald-400">{lastStep.word}</strong>
              </p>
              <p className="text-xs text-slate-500">{lastStep.rule}</p>
              {lastStep.top_k?.length > 0 && (
                <ul className="text-xs">
                  {lastStep.top_k.map((t, i) => (
                    <li key={i}>
                      {t.word} — {t.score?.toFixed?.(4) ?? t.score}
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}
          <form onSubmit={doOverride} className="mt-4 flex flex-wrap items-end gap-3 border-t border-slate-700 pt-4">
            <Field label="Row">
              <input
                type="number"
                className="w-16 rounded-md border border-slate-600 bg-slate-900 px-2 py-1"
                value={override.row}
                onChange={(e) => setOverride({ ...override, row: e.target.value })}
              />
            </Field>
            <Field label="Col">
              <input
                type="number"
                className="w-16 rounded-md border border-slate-600 bg-slate-900 px-2 py-1"
                value={override.col}
                onChange={(e) => setOverride({ ...override, col: e.target.value })}
              />
            </Field>
            <Field label="Word">
              <input
                className="rounded-md border border-slate-600 bg-slate-900 px-2 py-1"
                value={override.word}
                onChange={(e) => setOverride({ ...override, word: e.target.value })}
              />
            </Field>
            <SecondaryButton type="submit" disabled={job.running}>
              Override cell
            </SecondaryButton>
          </form>
        </Output>
      )}
    </ToolPanel>
  );
}
