export default function GridTable({ grid, highlight, title }) {
  if (!grid?.length) return null;
  return (
    <section className="my-3">
      {title && <h4 className="mb-2 text-sm font-medium text-slate-400">{title}</h4>}
      <table className="border-collapse font-mono text-sm">
        <tbody>
          {grid.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => {
                const isHi = highlight && highlight[0] === i && highlight[1] === j;
                const filled = cell && cell !== "·";
                return (
                  <td
                    key={j}
                    className={`min-w-[3rem] border border-slate-600 px-3 py-1.5 text-center ${
                      filled ? "bg-emerald-950/40" : ""
                    } ${isHi ? "ring-2 ring-sky-500" : ""}`}
                  >
                    {cell || "·"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
