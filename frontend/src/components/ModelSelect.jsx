import { Field } from "./ToolPanel";

export default function ModelSelect({ models, value, onChange }) {
  return (
    <Field label="Model">
      <select
        className="rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-slate-100"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label}
          </option>
        ))}
      </select>
    </Field>
  );
}
