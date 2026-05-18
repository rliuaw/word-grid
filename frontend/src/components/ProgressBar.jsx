export default function ProgressBar({ progress, running }) {
  if (!running && !progress) return null;
  const pct = progress?.percent ?? 0;
  return (
    <section className="mt-4 rounded-lg border border-slate-700 bg-slate-900/80 p-4">
      <header className="mb-2 flex items-center justify-between text-sm">
        <span className="text-slate-300">{progress?.message || "Working…"}</span>
        <span className="font-mono text-sky-400">{pct.toFixed(0)}%</span>
      </header>
      <section className="h-2 overflow-hidden rounded-full bg-slate-800">
        <span
          className="block h-full rounded-full bg-gradient-to-r from-sky-600 to-sky-400 transition-all duration-300"
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </section>
      {progress?.stage && (
        <p className="mt-2 text-xs uppercase tracking-wide text-slate-500">
          {progress.stage}
        </p>
      )}
    </section>
  );
}
