import { useState } from "react";
import { apiPost } from "../api/client";
import { useJob } from "../hooks/useJob";
import ProgressBar from "../components/ProgressBar";
import ModelSelect from "../components/ModelSelect";
import ToolPanel, { Field, Output, PrimaryButton } from "../components/ToolPanel";

export default function Unmasker({ models }) {
  const [sentence, setSentence] = useState("She sits [MASK] the chair");
  const [modelId, setModelId] = useState("bert-base-uncased");
  const [topK, setTopK] = useState(10);
  const { run, running, progress, result, error } = useJob();

  return (
    <ToolPanel title="BERT Unmasker" description="Top-K masked token predictions.">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run(() =>
            apiPost("/api/tools/unmasker", {
              sentence,
              model_id: modelId,
              top_k: +topK,
            })
          );
        }}
        className="flex flex-col gap-4"
      >
        <Field label="Masked sentence">
          <input
            className="w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={sentence}
            onChange={(e) => setSentence(e.target.value)}
          />
        </Field>
        <section className="flex flex-wrap items-end gap-4">
          <ModelSelect models={models} value={modelId} onChange={setModelId} />
          <Field label="Top K">
            <input
              type="number"
              min={1}
              max={50}
              className="w-20 rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
              value={topK}
              onChange={(e) => setTopK(e.target.value)}
            />
          </Field>
          <PrimaryButton disabled={running}>Unmask</PrimaryButton>
        </section>
      </form>
      <ProgressBar progress={progress} running={running} />
      {error && <p className="mt-4 text-red-400">{error}</p>}
      {result && (
        <Output>
          {(result.masks || [{ index: 0, results: result.results || [] }]).map((mask) => (
            <section key={mask.index} className={mask.index > 0 ? "mt-4 border-t border-slate-700 pt-4" : ""}>
              {result.mask_count > 1 && (
                <h4 className="mb-2 text-sm font-medium text-slate-400">
                  Mask {mask.index + 1}
                </h4>
              )}
              <ul className="space-y-1">
                {mask.results?.map((r, i) => (
                  <li key={i}>
                    {r.word}{" "}
                    <span className="text-sky-400">{r.score?.toFixed(4)}</span>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </Output>
      )}
    </ToolPanel>
  );
}
