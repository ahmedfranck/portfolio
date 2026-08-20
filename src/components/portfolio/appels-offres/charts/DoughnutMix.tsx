import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import ChartScaffold, { type ChartTableColumn, type ChartTableRow } from "./ChartScaffold";

export interface DoughnutMixDatum {
  readonly id: string;
  readonly name: string;
  readonly value: number;
  readonly color?: string;
}

interface DoughnutMixProps {
  readonly data: readonly DoughnutMixDatum[];
  readonly height?: number;
  readonly ariaLabel: string;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly colors?: readonly string[];
  readonly valueLabel?: string;
  readonly unit?: string;
  readonly formatValue?: (value: number) => string;
}

export default function DoughnutMix({
  data,
  height = 320,
  ariaLabel,
  source,
  illustrative = false,
  colors,
  valueLabel = "Valeur",
  unit = "",
  formatValue = (value) => value.toLocaleString("fr-FR"),
}: DoughnutMixProps) {
  const { theme } = useAoTheme();
  const palette = colors?.length ? colors : theme.series;
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const tableRows: ChartTableRow[] = data.map((item) => ({
    category: item.name,
    value: item.value,
    share: total > 0 ? item.value / total * 100 : 0,
  }));
  const tableColumns: ChartTableColumn[] = [
    { key: "category", label: "Catégorie" },
    { key: "value", label: valueLabel, format: (value) => typeof value === "number" ? `${formatValue(value)}${unit}` : "n.d." },
    { key: "share", label: "Part", format: (value) => typeof value === "number" ? `${value.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %` : "n.d." },
  ];

  return (
    <ChartScaffold ariaLabel={ariaLabel} source={source} illustrative={illustrative} tableColumns={tableColumns} tableRows={tableRows}>
      <div className="relative">
        <ResponsiveContainer width="100%" height={height}>
          <PieChart>
            <Pie
              data={[...data]}
              dataKey="value"
              nameKey="name"
              innerRadius="52%"
              outerRadius="78%"
              paddingAngle={2}
              cornerRadius={4}
              stroke="#FFFFFF"
              strokeWidth={2}
              isAnimationActive
            >
              {data.map((item, index) => <Cell key={item.id} fill={item.color ?? palette[index % palette.length]} />)}
            </Pie>
            <Tooltip
              formatter={(value: unknown, name: unknown) => [typeof value === "number" ? `${formatValue(value)}${unit}` : String(value ?? "n.d."), String(name)]}
              contentStyle={{ borderRadius: 8, border: `1px solid ${theme.colors.border}`, fontSize: 10, fontFamily: theme.typography.body }}
            />
            <Legend wrapperStyle={{ fontSize: 9, color: theme.colors.muted }} />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-center" aria-hidden="true">
          <strong className="block text-lg" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>{formatValue(total)}</strong>
          <span className="text-[8px] font-bold uppercase tracking-[0.08em]" style={{ color: theme.colors.muted }}>{valueLabel}</span>
        </div>
      </div>
    </ChartScaffold>
  );
}
