export default function ToolPanel({ title, description, children }) {
  return (
    <section className="max-w-3xl">
      <h2 className="text-xl font-semibold text-slate-100">{title}</h2>
      <p className="mt-1 mb-6 text-sm text-slate-400">{description}</p>
      {children}
    </section>
  );
}

export function Field({ label, children, className = "" }) {
  return (
    <label className={`flex flex-col gap-1 text-sm text-slate-400 ${className}`}>
      <span>{label}</span>
      {children}
    </label>
  );
}

export function Output({ children, className = "" }) {
  return (
    <section
      className={`mt-6 rounded-lg border border-slate-700 bg-slate-900/60 p-4 font-mono text-sm ${className}`}
    >
      {children}
    </section>
  );
}

export function PrimaryButton({ children, disabled, ...props }) {
  return (
    <button
      type="submit"
      disabled={disabled}
      className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-600 disabled:opacity-50"
      {...props}
    >
      {children}
    </button>
  );
}

export function SecondaryButton({ children, ...props }) {
  return (
    <button
      type="button"
      className="rounded-md border border-slate-600 px-3 py-2 text-sm text-slate-200 hover:bg-slate-800"
      {...props}
    >
      {children}
    </button>
  );
}
