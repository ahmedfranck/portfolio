import type { ReactNode } from "react";
import { Table2 } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

export type ChartCell = string | number | null | undefined;
export type ChartTableRow = Readonly<Record<string, ChartCell>>;

export interface ChartTableColumn {
  readonly key: string;
  readonly label: string;
  readonly format?: (value: ChartCell, row: ChartTableRow) => string;
  readonly render?: (value: ChartCell, row: ChartTableRow) => ReactNode;
}

interface ChartScaffoldProps {
  readonly children: ReactNode;
  readonly ariaLabel: string;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly tableColumns: readonly ChartTableColumn[];
  readonly tableRows: readonly ChartTableRow[];
}

export default function ChartScaffold({
  children,
  ariaLabel,
  tableColumns,
  tableRows,
}: ChartScaffoldProps) {
  const { theme } = useAoTheme();

  return (
    <div style={{ fontFamily: theme.typography.body }}>
      <div role="img" aria-label={ariaLabel}>
        {children}
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-end gap-2">
        <details className="group text-[9px]" style={{ color: theme.colors.muted }}>
          <summary className="inline-flex cursor-pointer list-none items-center gap-1 font-semibold" style={{ color: theme.colors.primary }}>
            <Table2 size={12} aria-hidden="true" /> Voir les données du graphique
          </summary>
          <div className="mt-2 max-h-72 max-w-full overflow-auto rounded-md border" style={{ borderColor: theme.colors.border }}>
            <table className="min-w-full border-collapse bg-white text-left text-[9px]">
              <thead className="sticky top-0 text-white" style={{ background: theme.colors.primary }}>
                <tr>
                  {tableColumns.map((column) => <th key={column.key} className="whitespace-nowrap px-2.5 py-2 font-bold">{column.label}</th>)}
                </tr>
              </thead>
              <tbody>
                {tableRows.map((row, index) => (
                  <tr key={index} className="border-b last:border-0" style={{ borderColor: theme.colors.border }}>
                    {tableColumns.map((column) => (
                      <td key={column.key} className="whitespace-nowrap px-2.5 py-2" style={{ color: theme.colors.text }}>
                        {column.render ? column.render(row[column.key], row) : column.format ? column.format(row[column.key], row) : String(row[column.key] ?? "n.d.")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </div>
  );
}
