import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import ChartScaffold, { type ChartTableColumn, type ChartTableRow } from "./ChartScaffold";

export interface BubbleScatterDatum {
  readonly id: string;
  readonly name: string;
  readonly x: number;
  readonly y: number;
  readonly z?: number;
  readonly group?: string;
  readonly color?: string;
}

interface BubbleScatterProps {
  readonly data: readonly BubbleScatterDatum[];
  readonly xLabel: string;
  readonly yLabel: string;
  readonly zLabel?: string;
  readonly xUnit?: string;
  readonly yUnit?: string;
  readonly zUnit?: string;
  readonly height?: number;
  readonly ariaLabel: string;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly colors?: readonly string[];
  readonly formatValue?: (value: number) => string;
}

export default function BubbleScatter({
  data,
  xLabel,
  yLabel,
  zLabel = "Taille",
  xUnit = "",
  yUnit = "",
  zUnit = "",
  height = 350,
  ariaLabel,
  source,
  illustrative = false,
  colors,
  formatValue = (value) => value.toLocaleString("fr-FR"),
}: BubbleScatterProps) {
  const { theme } = useAoTheme();
  const palette = colors?.length ? colors : theme.series;
  const hasSize = data.some((item) => item.z != null);
  const groupNames = [...new Set(data.map((item) => item.group ?? item.id))];
  const colorFor = (item: BubbleScatterDatum, index: number) => item.color ?? palette[Math.max(0, groupNames.indexOf(item.group ?? item.id)) % palette.length] ?? palette[index % palette.length];
  const tableRows: ChartTableRow[] = data.map((item) => ({
    name: item.name,
    x: item.x,
    y: item.y,
    z: item.z,
    group: item.group,
  }));
  const tableColumns: ChartTableColumn[] = [
    { key: "name", label: "Entité" },
    { key: "x", label: xLabel, format: (value) => typeof value === "number" ? `${formatValue(value)}${xUnit}` : "n.d." },
    { key: "y", label: yLabel, format: (value) => typeof value === "number" ? `${formatValue(value)}${yUnit}` : "n.d." },
    ...(hasSize ? [{ key: "z", label: zLabel, format: (value) => typeof value === "number" ? `${formatValue(value)}${zUnit}` : "n.d." } satisfies ChartTableColumn] : []),
    ...(data.some((item) => item.group) ? [{ key: "group", label: "Groupe" } satisfies ChartTableColumn] : []),
  ];

  return (
    <ChartScaffold ariaLabel={ariaLabel} source={source} illustrative={illustrative} tableColumns={tableColumns} tableRows={tableRows}>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 10, right: 18, bottom: 28, left: 6 }}>
          <CartesianGrid stroke={theme.colors.border} strokeOpacity={0.55} />
          <XAxis
            type="number"
            dataKey="x"
            name={xLabel}
            unit={xUnit}
            tick={{ fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }}
            tickLine={false}
            axisLine={{ stroke: theme.colors.border }}
            label={{ value: xLabel, position: "insideBottom", offset: -15, fontSize: 9, fill: theme.colors.muted }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={yLabel}
            unit={yUnit}
            tick={{ fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }}
            tickLine={false}
            axisLine={false}
            label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 9, fill: theme.colors.muted }}
          />
          {hasSize && <ZAxis type="number" dataKey="z" name={zLabel} unit={zUnit} range={[70, 520]} />}
          <Tooltip
            cursor={{ stroke: theme.colors.border, strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const item = payload[0].payload as BubbleScatterDatum;
              return (
                <div className="rounded-md border bg-white px-3 py-2 text-[10px] shadow-lg" style={{ borderColor: theme.colors.border, fontFamily: theme.typography.body }}>
                  <strong style={{ color: theme.colors.primary }}>{item.name}</strong>
                  <p style={{ color: theme.colors.muted }}>{xLabel} : {formatValue(item.x)}{xUnit}</p>
                  <p style={{ color: theme.colors.muted }}>{yLabel} : {formatValue(item.y)}{yUnit}</p>
                  {item.z != null && <p style={{ color: theme.colors.muted }}>{zLabel} : {formatValue(item.z)}{zUnit}</p>}
                  {item.group && <p style={{ color: theme.colors.muted }}>Groupe : {item.group}</p>}
                </div>
              );
            }}
          />
          <Scatter data={[...data]} fill={theme.colors.primary} isAnimationActive>
            {data.map((item, index) => <Cell key={item.id} fill={colorFor(item, index)} fillOpacity={0.82} />)}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </ChartScaffold>
  );
}
