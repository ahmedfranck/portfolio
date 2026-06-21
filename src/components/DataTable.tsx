import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";

export interface DataTableColumn<T> {
  key: keyof T;
  label: string;
  format?: (value: T[keyof T], row: T) => string;
}

interface DataTableProps<T extends Record<string, unknown>> {
  rows: T[];
  columns: DataTableColumn<T>[];
  searchableKey?: keyof T;
  searchPlaceholder?: string;
}

export default function DataTable<T extends Record<string, unknown>>({
  rows,
  columns,
  searchableKey,
  searchPlaceholder = "Filtrer par pays…",
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<keyof T | null>(null);
  const [sortDir, setSortDir] = useState<1 | -1>(1);
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    if (!searchableKey || !query.trim()) return rows;
    const q = query.trim().toLowerCase();
    return rows.filter((r) => String(r[searchableKey]).toLowerCase().includes(q));
  }, [rows, query, searchableKey]);

  const sorted = useMemo(() => {
    if (!sortKey) return filtered;
    return [...filtered].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * sortDir;
      return String(av).localeCompare(String(bv)) * sortDir;
    });
  }, [filtered, sortKey, sortDir]);

  function toggleSort(key: keyof T) {
    if (sortKey === key) {
      setSortDir((d) => (d === 1 ? -1 : 1));
    } else {
      setSortKey(key);
      setSortDir(1);
    }
  }

  return (
    <div className="space-y-3">
      {searchableKey && (
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={searchPlaceholder}
          className="w-full max-w-xs rounded-lg border border-line px-3 py-2 text-sm text-ink"
        />
      )}
      <div className="max-h-[420px] overflow-auto rounded-card border border-line">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-surface font-mono text-[11px] uppercase tracking-wide text-text-2">
            <tr>
              {columns.map((col) => (
                <th key={String(col.key)} className="px-3 py-2.5">
                  <button
                    type="button"
                    onClick={() => toggleSort(col.key)}
                    className="inline-flex items-center gap-1 hover:text-ink"
                  >
                    {col.label}
                    {sortKey === col.key && (sortDir === 1 ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {sorted.map((row, i) => (
              <tr key={i} className="transition-colors duration-150 hover:bg-brand-soft/60">
                {columns.map((col) => (
                  <td key={String(col.key)} className="px-3 py-2 text-ink">
                    {col.format ? col.format(row[col.key], row) : String(row[col.key] ?? "n.d.")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
