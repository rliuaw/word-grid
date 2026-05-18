import { useState } from "react";
import { apiPost } from "../api/client";
import { useJob } from "../hooks/useJob";
import ProgressBar from "../components/ProgressBar";
import ModelSelect from "../components/ModelSelect";
import ToolPanel, { Field, Output, PrimaryButton } from "../components/ToolPanel";

export default function SinglePos({ models }) {
  const [pos, setPos] = useState("NN");
  const [modelId, setModelId] = useState("bert-base-uncased");
  const { run, running, progress, result, error } = useJob();

  const onSubmit = (e) => {
    e.preventDefault();
    run(() => apiPost("/api/tools/single-pos", { pos, model_id: modelId }));
  };

  return (
    <ToolPanel
      title="Single POS Generator"
      description="Random BERT-vocabulary word matching a Penn Treebank POS tag."
    >
      <form onSubmit={onSubmit} className="flex flex-wrap items-end gap-4">
        <Field label="POS tag">
          <input
            className="rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-slate-100"
            value={pos}
            onChange={(e) => setPos(e.target.value)}
          />
        </Field>
        <ModelSelect models={models} value={modelId} onChange={setModelId} />
        <PrimaryButton disabled={running}>Generate word</PrimaryButton>
      </form>
      <ProgressBar progress={progress} running={running} />
      {error && <p className="mt-4 text-red-400">{error}</p>}
      {result && (
        <Output>
          <p className="text-lg text-emerald-300">{result.word}</p>
          <p className="mt-2 text-slate-500">Saved as {result.result_id}</p>
        </Output>
      )}
    </ToolPanel>
  );
}
