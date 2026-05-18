import { useCallback, useEffect, useState } from "react";
import { apiGet } from "../api/client";
import GridTable from "../components/GridTable";
import ToolPanel, { SecondaryButton } from "../components/ToolPanel";

export default function Gallery({ onSelectGrid }) {
  const [grids, setGrids] = useState([]);

  const load = useCallback(async () => {
    const data = await apiGet("/api/gallery");
    setGrids(data.grids || []);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <ToolPanel title="Gallery" description="All filled word grids saved to the results folder.">
      <SecondaryButton onClick={load}>Refresh</SecondaryButton>
      <section className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {grids.map((g) => (
          <article
            key={g.id}
            className="cursor-pointer rounded-lg border border-slate-700 bg-slate-900/50 p-3 hover:border-sky-600"
            onClick={() => onSelectGrid?.(g.id)}
          >
            <header className="mb-2 flex justify-between text-xs text-slate-500">
              <span>{g.n}×{g.n}</span>
              <span>{g.metadata?.benchmark_score ?? "—"}</span>
            </header>
            <GridTable grid={g.cells} />
            {g.metadata?.special && (
              <span className="mt-2 inline-block text-xs text-amber-400">★ special</span>
            )}
          </article>
        ))}
      </section>
      {!grids.length && (
        <p className="mt-8 text-center text-slate-500">No grids yet. Use Grid Unmasker to save one.</p>
      )}
    </ToolPanel>
  );
}
