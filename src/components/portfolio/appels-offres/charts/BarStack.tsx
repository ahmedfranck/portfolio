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
import { CountryLabel } from "../shared/CountryFlag";
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
  readonly countryCodeKey?: string;
}

interface AxisTickProps {
  readonly x?: string | number;
  readonly y?: string | number;
  readonly payload?: { readonly value?: string };
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
  countryCodeKey,
  formatValue = (value, item) => `${value.toLocaleString("fr-FR")}${item.unit ?? ""}`,
}: BarStackProps) {
  const { theme } = useAoTheme();
  const palette = colors?.length ? colors : theme.series;
  const seriesByKey = new Map(series.map((item) => [item.dataKey, item]));
  const countryByLabel = new Map(data.map((item) => [String(item[xKey] ?? ""), countryCodeKey ? String(item[countryCodeKey] ?? "") : ""]));
  function CountryTick({ x = 0, y = 0, payload }: AxisTickProps) {
    const label = payload?.value ?? "";
    const iso3 = countryByLabel.get(label);
    return (
      <foreignObject x={(Number(x) || 0) - 46} y={(Number(y) || 0) + 5} width={92} height={28}>
        <div className="flex justify-center text-[8px] font-semibold" style={{ color: theme.colors.primary, fontFamily: theme.typography.body }}>
          {iso3 ? <CountryLabel iso3={iso3} name={label} /> : label}
        </div>
      </foreignObject>
    );
  }
  const tableColumns: ChartTableColumn[] = [
    { key: xKey, label: xLabel, render: (value, row) => countryCodeKey && row[countryCodeKey] ? <CountryLabel iso3={String(row[countryCodeKey])} name={String(value)} /> : String(value ?? "n.d.") },
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
          margin={{ top: 10, right: 14, bottom: countryCodeKey ? 30 : 4, left: 0 }}
        >
          <CartesianGrid stroke={theme.colors.border} strokeOpacity={0.55} vertical={false} />
          <XAxis dataKey={xKey} tick={countryCodeKey ? CountryTick : { fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }} height={countryCodeKey ? 44 : 30} tickLine={false} axisLine={{ stroke: theme.colors.border }} />
          <YAxis
            tickFormatter={mode === "percent" ? (value) => `${Math.round(Number(value) * 100)} %` : undefined}
            tick={{ fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }}
            tickLine={false}
            axisLine={false}
          />
          {countryCodeKey ? (
            <Tooltip content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as BarStackDatum;
              const iso3 = String(row[countryCodeKey] ?? "");
              return <div className="rounded-md border bg-white px-3 py-2 text-[10px] shadow-lg" style={{ borderColor: theme.colors.border, fontFamily: theme.typography.body }}><strong style={{ color: theme.colors.primary }}><CountryLabel iso3={iso3} name={String(label)} /></strong>{payload.map((entry) => { const item = seriesByKey.get(String(entry.name)); return <p key={String(entry.name)} style={{ color: theme.colors.muted }}>{item?.name ?? String(entry.name)} : {typeof entry.value === "number" && item ? formatValue(entry.value, item) : String(entry.value ?? "n.d.")}</p>; })}</div>;
            }} />
          ) : (
            <Tooltip formatter={(value: unknown, name: unknown) => { const item = seriesByKey.get(String(name)); return [typeof value === "number" && item ? formatValue(value, item) : String(value ?? "n.d."), item?.name ?? String(name)]; }} contentStyle={{ borderRadius: 8, border: `1px solid ${theme.colors.border}`, fontSize: 10, fontFamily: theme.typography.body }} />
          )}
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
