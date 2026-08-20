import { useMemo, useState, type CSSProperties, type Key } from "react";
import { ArrowDown, ArrowUp, Download, Search } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { AoDataBadges } from "./badges";

export interface AoDataTableColumn<T> {
  readonly id: string;
  readonly header: string;
  readonly accessor: (row: T) => unknown;
  readonly render?: (value: unknown, row: T) => string;
  readonly align?: "left" | "center" | "right";
}

interface DataTableProps<T extends object> {
  readonly rows: readonly T[];
  readonly columns: readonly AoDataTableColumn<T>[];
  readonly rowKey?: (row: T, index: number) => Key;
  readonly searchPlaceholder?: string;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly exportFilename?: string;
  readonly emptyLabel?: string;
}

function compareValues(a: unknown, b: unknown) {
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), "fr", { numeric: true, sensitivity: "base" });
}

function csvCell(value: unknown) {
  const text = value == null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

export default function DataTable<T extends object>({
  rows,
  columns,
  rowKey,
  searchPlaceholder = "Filtrer le tableau…",
  source,
  illustrative = false,
  exportFilename,
  emptyLabel = "Aucune donnée pour les filtres sélectionnés.",
}: DataTableProps<T>) {
  const { theme } = useAoTheme();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ id: string; direction: 1 | -1 } | null>(null);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase("fr");
    if (!normalized) return [...rows];
    return rows.filter((row) => columns.some((column) => String(column.accessor(row) ?? "").toLocaleLowerCase("fr").includes(normalized)));
  }, [columns, query, rows]);

  const visibleRows = useMemo(() => {
    if (!sort) return filtered;
    const column = columns.find((item) => item.id === sort.id);
    if (!column) return filtered;
    return [...filtered].sort((a, b) => compareValues(column.accessor(a), column.accessor(b)) * sort.direction);
  }, [columns, filtered, sort]);

  function toggleSort(id: string) {
    setSort((current) => current?.id === id
      ? { id, direction: current.direction === 1 ? -1 : 1 }
      : { id, direction: 1 });
  }

  function exportCsv() {
    if (!exportFilename) return;
    const header = columns.map((column) => csvCell(column.header)).join(",");
    const body = visibleRows.map((row) => columns.map((column) => csvCell(column.accessor(row))).join(",")).join("\n");
    const blob = new Blob(["\uFEFF", header, "\n", body], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${exportFilename}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div style={{ fontFamily: theme.typography.body }}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <label className="relative block min-w-56 flex-1 sm:max-w-xs">
          <span className="sr-only">Rechercher dans le tableau</span>
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2" style={{ color: theme.colors.muted }} aria-hidden="true" />
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={searchPlaceholder}
            className="w-full rounded-md border bg-white py-2 pl-8 pr-3 text-[10px] outline-none focus:ring-2"
            style={{ borderColor: theme.colors.border, color: theme.colors.text, "--tw-ring-color": theme.colors.accent } as CSSProperties}
          />
        </label>
        {exportFilename && (
          <button
            type="button"
            onClick={exportCsv}
            className="inline-flex items-center gap-1.5 rounded-md border px-3 py-2 text-[9px] font-bold"
            style={{ borderColor: theme.colors.border, color: theme.colors.primary, background: theme.colors.canvas }}
          >
            <Download size={13} aria-hidden="true" /> CSV
          </button>
        )}
      </div>
      <div className="max-h-[460px] overflow-auto rounded-md border" style={{ borderColor: theme.colors.border }}>
        <table className="w-full min-w-[640px] border-collapse text-left text-[10px]">
          <thead className="sticky top-0 z-10 text-[8px] font-bold uppercase tracking-[0.06em] text-white" style={{ background: theme.colors.primary }}>
            <tr>
              {columns.map((column) => (
                <th key={column.id} className="px-3 py-2.5" style={{ textAlign: column.align ?? "left" }}>
                  <button type="button" onClick={() => toggleSort(column.id)} className="inline-flex items-center gap-1 font-inherit">
                    {column.header}
                    {sort?.id === column.id && (sort.direction === 1 ? <ArrowUp size={11} /> : <ArrowDown size={11} />)}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, index) => (
              <tr key={rowKey?.(row, index) ?? index} className="border-b last:border-0 hover:bg-[var(--ao-soft)]" style={{ borderColor: theme.colors.border }}>
                {columns.map((column) => {
                  const value = column.accessor(row);
                  return (
                    <td key={column.id} className="px-3 py-2.5" style={{ color: theme.colors.text, textAlign: column.align ?? "left" }}>
                      {column.render ? column.render(value, row) : String(value ?? "n.d.")}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        {visibleRows.length === 0 && <p className="p-6 text-center text-[10px]" style={{ color: theme.colors.muted }}>{emptyLabel}</p>}
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <span className="text-[9px]" style={{ color: theme.colors.muted }}>{visibleRows.length.toLocaleString("fr-FR")} ligne(s)</span>
        <AoDataBadges source={source} illustrative={illustrative} />
      </div>
    </div>
  );
}
