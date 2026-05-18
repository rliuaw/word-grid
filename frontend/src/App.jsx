import { useEffect, useState } from "react";
import { apiGet } from "./api/client";
import SinglePos from "./tools/SinglePos";
import PosCombinations from "./tools/PosCombinations";
import PosTest from "./tools/PosTest";
import PosGrid from "./tools/PosGrid";
import Unmasker from "./tools/Unmasker";
import GridUnmasker from "./tools/GridUnmasker";
import Benchmark from "./tools/Benchmark";
import Gallery from "./tools/Gallery";

const TABS = [
  { id: "single-pos", label: "Single POS" },
  { id: "pos-combinations", label: "POS Combinations" },
  { id: "pos-test", label: "POS Test" },
  { id: "pos-grid", label: "POS Grid" },
  { id: "unmasker", label: "BERT Unmasker" },
  { id: "grid-unmask", label: "Grid Unmasker" },
  { id: "benchmark", label: "Benchmark" },
  { id: "gallery", label: "Gallery" },
];

export default function App() {
  const [tab, setTab] = useState("single-pos");
  const [models, setModels] = useState([]);
  const [algorithms, setAlgorithms] = useState([]);
  const [benchmarkGridId, setBenchmarkGridId] = useState("");

  useEffect(() => {
    apiGet("/api/models").then((d) => setModels(d.models || [])).catch(() => {});
    apiGet("/api/algorithms").then((d) => setAlgorithms(d.algorithms || [])).catch(() => {});
  }, []);

  const goBenchmark = (id) => {
    setBenchmarkGridId(id);
    setTab("benchmark");
  };

  return (
    <section className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-900/80 px-6 py-5 backdrop-blur">
        <h1 className="text-2xl font-semibold tracking-tight">Word Grid Explorer</h1>
        <p className="mt-1 text-sm text-slate-400">
          Construct and analyze N×N word grids — POS pipelines, crossword grids, BERT unmasking
        </p>
      </header>
      <section className="flex min-h-[calc(100vh-88px)]">
        <nav className="w-52 shrink-0 border-r border-slate-800 bg-slate-900/40 py-4">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`block w-full border-l-2 px-4 py-2.5 text-left text-sm transition ${
                tab === t.id
                  ? "border-sky-500 bg-sky-950/30 text-sky-300"
                  : "border-transparent text-slate-500 hover:bg-slate-800/50 hover:text-slate-200"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <main className="flex-1 overflow-auto p-8">
          {tab === "single-pos" && <SinglePos models={models} />}
          {tab === "pos-combinations" && <PosCombinations algorithms={algorithms} />}
          {tab === "pos-test" && <PosTest models={models} />}
          {tab === "pos-grid" && <PosGrid />}
          {tab === "unmasker" && <Unmasker models={models} />}
          {tab === "grid-unmask" && <GridUnmasker models={models} />}
          {tab === "benchmark" && <Benchmark key={benchmarkGridId} initialGalleryId={benchmarkGridId} />}
          {tab === "gallery" && <Gallery onSelectGrid={goBenchmark} />}
        </main>
      </section>
    </section>
  );
}
