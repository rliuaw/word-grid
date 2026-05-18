import { useState } from "react";
import { apiPost } from "../api/client";
import { useJob } from "../hooks/useJob";
import ProgressBar from "../components/ProgressBar";
import ToolPanel, { Field, Output, PrimaryButton } from "../components/ToolPanel";

export default function PosCombinations({ algorithms }) {
  const [n, setN] = useState(3);
  const [k, setK] = useState(50);
  const [algorithm, setAlgorithm] = useState("tagger");
  const [maxSentences, setMaxSentences] = useState(300);
  const { run, running, progress, result, error } = useJob();

  const onSubmit = (e) => {
    e.preventDefault();
    run(() =>
      apiPost("/api/tools/pos-combinations", {
        n: +n,
        k: +k,
        algorithm,
        max_sentences: +maxSentences,
      })
    );
  };

  const showOccurrences = result?.has_occurrences;

  return (
    <ToolPanel
      title="POS Combinations"
      description="Generate dictionary.txt from tagger or rule-based algorithms."
    >
      <form onSubmit={onSubmit} className="flex flex-wrap items-end gap-4">
        <Field label="N">
          <input
            type="number"
            min={1}
            max={12}
            className="w-20 rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={n}
            onChange={(e) => setN(e.target.value)}
          />
        </Field>
        <Field label="K">
          <input
            type="number"
            min={1}
            className="w-24 rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={k}
            onChange={(e) => setK(e.target.value)}
          />
        </Field>
        <Field label="Algorithm">
          <select
            className="rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
          >
            {algorithms.map((a) => (
              <option key={a.id} value={a.id}>
                {a.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Max sentences">
          <input
            type="number"
            className="w-28 rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
            value={maxSentences}
            onChange={(e) => setMaxSentences(e.target.value)}
          />
        </Field>
        <PrimaryButton disabled={running}>Generate dictionary</PrimaryButton>
      </form>
      <ProgressBar progress={progress} running={running} />
      {error && <p className="mt-4 text-red-400">{error}</p>}
      {result && (
        <Output>
          <p className="mb-3 text-slate-300">
            {result.count} combinations ·{" "}
            <a
              className="text-sky-400 hover:underline"
              href={`/api/results/${result.result_id}/dictionary.txt`}
              target="_blank"
              rel="noreferrer"
            >
              Download dictionary.txt
            </a>
          </p>
          <section className="max-h-96 overflow-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-700 text-slate-500">
                  <th className="py-2 pr-4">#</th>
                  <th className="py-2 pr-4">POS sequence</th>
                  {showOccurrences && <th className="py-2">Occurrences</th>}
                </tr>
              </thead>
              <tbody>
                {result.combinations?.map((c, i) => (
                  <tr key={i} className="border-b border-slate-800">
                    <td className="py-1.5 pr-4 text-slate-500">{i + 1}</td>
                    <td className="py-1.5 pr-4 font-mono">{c.display}</td>
                    {showOccurrences && (
                      <td className="py-1.5 text-amber-300/90">
                        {c.occurrences?.toLocaleString()}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </Output>
      )}
    </ToolPanel>
  );
}
