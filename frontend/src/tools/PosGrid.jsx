import { useState } from "react";
import { apiPost } from "../api/client";
import { useJob } from "../hooks/useJob";
import ProgressBar from "../components/ProgressBar";
import ResultSelect from "../components/ResultSelect";
import GridTable from "../components/GridTable";
import ToolPanel, { Field, Output, PrimaryButton } from "../components/ToolPanel";

export default function PosGrid() {
  const [n, setN] = useState(3);
  const [k, setK] = useState(3);
  const [dictId, setDictId] = useState("");
  const { run, running, progress, result, error } = useJob();

  return (
    <ToolPanel
      title="POS Grid Generator"
      description="Build N×N POS grids from a saved dictionary using the crossword solver."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!dictId) return;
          run(() =>
            apiPost("/api/tools/pos-grid", {
              n: +n,
              k: +k,
              dictionary_result_id: dictId,
            })
          );
        }}
        className="flex flex-wrap items-end gap-4"
      >
        <Field label="N">
          <input
            type="number"
            min={2}
            max={7}
            className="w-20 rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={n}
            onChange={(e) => setN(e.target.value)}
          />
        </Field>
        <Field label="Candidate grids (K)">
          <input
            type="number"
            min={1}
            className="w-20 rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={k}
            onChange={(e) => setK(e.target.value)}
          />
        </Field>
        <ResultSelect
          tool="pos_combinations"
          value={dictId}
          onChange={setDictId}
          placeholder="Select POS combinations result…"
          className="min-w-[280px]"
        />
        <PrimaryButton disabled={running || !dictId}>Generate grids</PrimaryButton>
      </form>
      <ProgressBar progress={progress} running={running} />
      {error && <p className="mt-4 text-red-400">{error}</p>}
      {result && (
        <Output>
          <p className="mb-4 text-slate-300">
            Found {result.found} grid(s) · saved {result.result_id}
          </p>
          {result.grids?.map((g, i) => (
            <GridTable key={i} grid={g} title={`Grid ${i + 1}`} />
          ))}
        </Output>
      )}
    </ToolPanel>
  );
}
