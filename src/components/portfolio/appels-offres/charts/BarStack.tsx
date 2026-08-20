import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import ChartScaffold, { type ChartCell, type ChartTableColumn, type ChartTableRow } from "./ChartScaffold";

export type BarStackDatum = Readonly<Record<string, ChartCell>>;

export interface BarStackSeries {
  readonly dataKey: string;
  readonly name: string;
  readonly color?: string;
  readonly unit?: string;
}

interface BarStackProps {
  readonly data: readonly BarStackDatum[];
  readonly xKey: string;
  readonly xLabel?: string;
  readonly series: readonly BarStackSeries[];
  readonly mode?: "stacked" | "grouped" | "percent";
  readonly height?: number;
  readonly ariaLabel: string;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly colors?: readonly string[];
  readonly formatValue?: (value: number, series: BarStackSeries) => string;
}

export default function BarStack({
  data,
  xKey,
  xLabel = "Catégorie",
  series,
  mode = "stacked",
  height = 320,
  ariaLabel,
  source,
  illustrative = false,
  colors,
  formatValue = (value, item) => `${value.toLocaleString("fr-FR")}${item.unit ?? ""}`,
}: BarStackProps) {
  const { theme } = useAoTheme();
  const palette = colors?.length ? colors : theme.series;
  const seriesByKey = new Map(series.map((item) => [item.dataKey, item]));
  const tableColumns: ChartTableColumn[] = [
    { key: xKey, label: xLabel },
    ...series.map((item) => ({
      key: item.dataKey,
      label: item.name,
      format: (value: ChartCell) => typeof value === "number" ? formatValue(value, item) : String(value ?? "n.d."),
    })),
  ];

  return (
    <ChartScaffold ariaLabel={ariaLabel} source={source} illustrative={illustrative} tableColumns={tableColumns} tableRows={data as readonly ChartTableRow[]}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={[...data]}
          stackOffset={mode === "percent" ? "expand" : undefined}
          margin={{ top: 10, right: 14, bottom: 4, left: 0 }}
        >
          <CartesianGrid stroke={theme.colors.border} strokeOpacity={0.55} vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }} tickLine={false} axisLine={{ stroke: theme.colors.border }} />
          <YAxis
            tickFormatter={mode === "percent" ? (value) => `${Math.round(Number(value) * 100)} %` : undefined}
            tick={{ fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            formatter={(value: unknown, name: unknown) => {
              const item = seriesByKey.get(String(name));
              return [typeof value === "number" && item ? formatValue(value, item) : String(value ?? "n.d."), item?.name ?? String(name)];
            }}
            contentStyle={{ borderRadius: 8, border: `1px solid ${theme.colors.border}`, fontSize: 10, fontFamily: theme.typography.body }}
          />
          <Legend formatter={(value) => seriesByKey.get(value)?.name ?? value} wrapperStyle={{ fontSize: 9, color: theme.colors.muted }} />
          {series.map((item, index) => (
            <Bar
              key={item.dataKey}
              dataKey={item.dataKey}
              name={item.dataKey}
              stackId={mode === "grouped" ? undefined : "ao-stack"}
              fill={item.color ?? palette[index % palette.length]}
              radius={[4, 4, 0, 0]}
              maxBarSize={34}
              isAnimationActive
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartScaffold>
  );
}
