import { useState } from "react";
import { apiPost } from "../api/client";
import { useJob } from "../hooks/useJob";
import ProgressBar from "../components/ProgressBar";
import ModelSelect from "../components/ModelSelect";
import ToolPanel, { Field, Output, PrimaryButton } from "../components/ToolPanel";

export default function PosTest({ models }) {
  const [sequence, setSequence] = useState("DT NN VB");
  const [modelId, setModelId] = useState("bert-base-uncased");
  const { run, running, progress, result, error } = useJob();

  return (
    <ToolPanel
      title="POS Combination Test"
      description="Random sentence from the BERT vocabulary for a POS sequence."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(() =>
            apiPost("/api/tools/pos-test", {
              pos_sequence: sequence,
              model_id: modelId,
            })
          );
        }}
        className="flex flex-wrap items-end gap-4"
      >
        <Field label="POS sequence" className="min-w-[200px] flex-1">
          <input
            className="w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
          />
        </Field>
        <ModelSelect models={models} value={modelId} onChange={setModelId} />
        <PrimaryButton disabled={running}>Generate sentence</PrimaryButton>
      </form>
      <ProgressBar progress={progress} running={running} />
      {error && <p className="mt-4 text-red-400">{error}</p>}
      {result && (
        <Output>
          <p className="text-lg text-emerald-300">{result.sentence}</p>
        </Output>
      )}
    </ToolPanel>
  );
}
