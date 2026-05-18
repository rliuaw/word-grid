import { useResults, formatResultLabel } from "../hooks/useResults";

export default function ResultSelect({
  tool,
  value,
  onChange,
  placeholder = "Select a saved result…",
  className = "",
}) {
  const { options, loading, refresh } = useResults(tool);

  return (
    <label className={`flex flex-col gap-1 text-sm text-slate-400 ${className}`}>
      <span>Saved result</span>
      <select
        className="rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-slate-100"
        value={value || ""}
        onChange={(e) => onChange(e.target.value || null)}
        disabled={loading}
      >
        <option value="">{loading ? "Loading…" : placeholder}</option>
        {options.map((opt) => (
          <option key={opt.id} value={opt.id}>
            {formatResultLabel(opt)}
          </option>
        ))}
      </select>
      <button
        type="button"
        onClick={refresh}
        className="self-start text-xs text-sky-400 hover:text-sky-300"
      >
        Refresh list
      </button>
    </label>
  );
}
