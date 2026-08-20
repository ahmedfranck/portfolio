import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import ChartScaffold, { type ChartCell, type ChartTableColumn, type ChartTableRow } from "./ChartScaffold";

export type LineTrendDatum = Readonly<Record<string, ChartCell>>;

export interface LineTrendSeries {
  readonly dataKey: string;
  readonly name: string;
  readonly color?: string;
  readonly unit?: string;
}

interface LineTrendProps {
  readonly data: readonly LineTrendDatum[];
  readonly xKey: string;
  readonly xLabel?: string;
  readonly series: readonly LineTrendSeries[];
  readonly height?: number;
  readonly yLabel?: string;
  readonly ariaLabel: string;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly colors?: readonly string[];
  readonly formatValue?: (value: number, series: LineTrendSeries) => string;
}

export default function LineTrend({
  data,
  xKey,
  xLabel = "Période",
  series,
  height = 320,
  yLabel,
  ariaLabel,
  source,
  illustrative = false,
  colors,
  formatValue = (value, item) => `${value.toLocaleString("fr-FR")}${item.unit ?? ""}`,
}: LineTrendProps) {
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
    <ChartScaffold
      ariaLabel={ariaLabel}
      source={source}
      illustrative={illustrative}
      tableColumns={tableColumns}
      tableRows={data as readonly ChartTableRow[]}
    >
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={[...data]} margin={{ top: 10, right: 14, bottom: 4, left: 0 }}>
          <CartesianGrid stroke={theme.colors.border} strokeOpacity={0.55} vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }} tickLine={false} axisLine={{ stroke: theme.colors.border }} />
          <YAxis
            tick={{ fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }}
            tickLine={false}
            axisLine={false}
            label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fontSize: 9, fill: theme.colors.muted } : undefined}
          />
          <Tooltip
            formatter={(value: unknown, name: unknown) => {
              const item = seriesByKey.get(String(name));
              return [typeof value === "number" && item ? formatValue(value, item) : String(value ?? "n.d."), item?.name ?? String(name)];
            }}
            contentStyle={{ borderRadius: 8, border: `1px solid ${theme.colors.border}`, fontSize: 10, fontFamily: theme.typography.body }}
            labelStyle={{ color: theme.colors.primary, fontWeight: 700 }}
          />
          <Legend formatter={(value) => seriesByKey.get(value)?.name ?? value} wrapperStyle={{ fontSize: 9, color: theme.colors.muted }} />
          {series.map((item, index) => (
            <Line
              key={item.dataKey}
              type="monotone"
              dataKey={item.dataKey}
              name={item.dataKey}
              stroke={item.color ?? palette[index % palette.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
              connectNulls={false}
              isAnimationActive
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartScaffold>
  );
}
