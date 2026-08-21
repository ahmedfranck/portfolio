import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import ChartScaffold, { type ChartTableRow } from "./ChartScaffold";

export interface HorizontalRankDatum {
  readonly id: string;
  readonly name: string;
  readonly value: number;
  readonly year?: number;
}

interface HorizontalRankBarsProps {
  readonly data: readonly HorizontalRankDatum[];
  readonly axisMax: number;
  readonly unit?: string;
  readonly height?: number;
  readonly ariaLabel: string;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly rampStart?: string;
  readonly rampEnd?: string;
}

function hexToRgb(hex: string) {
  const value = Number.parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255] as const;
}

function mixHex(from: string, to: string, ratio: number) {
  const start = hexToRgb(from);
  const end = hexToRgb(to);
  const channel = (index: number) => Math.round(start[index] + (end[index] - start[index]) * ratio);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

export default function HorizontalRankBars({
  data,
  axisMax,
  unit = "",
  height = 390,
  ariaLabel,
  source,
  illustrative = false,
  rampStart = "#D7B85A",
  rampEnd,
}: HorizontalRankBarsProps) {
  const { theme } = useAoTheme();
  const endColor = rampEnd ?? theme.colors.primary;
  const sorted = [...data].sort((a, b) => b.value - a.value);
  const ticks = Array.from({ length: Math.floor(axisMax / 5) + 1 }, (_, index) => index * 5);
  const tableRows: ChartTableRow[] = sorted.map((item) => ({ country: item.name, value: item.value, year: item.year }));

  return (
    <ChartScaffold
      ariaLabel={ariaLabel}
      source={source}
      illustrative={illustrative}
      tableRows={tableRows}
      tableColumns={[
        { key: "country", label: "Pays" },
        { key: "value", label: "mCPR moderne", format: (value) => typeof value === "number" ? `${value.toLocaleString("fr-FR", { maximumFractionDigits: 1 })}${unit}` : "n.d." },
        { key: "year", label: "Millésime" },
      ]}
    >
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={sorted} layout="vertical" margin={{ top: 8, right: 46, bottom: 14, left: 6 }}>
          <CartesianGrid stroke={theme.colors.border} strokeOpacity={0.65} horizontal={false} />
          <XAxis
            type="number"
            domain={[0, axisMax]}
            ticks={ticks}
            unit={unit}
            tick={{ fontSize: 9, fill: theme.colors.muted, fontFamily: theme.typography.body }}
            tickLine={false}
            axisLine={{ stroke: theme.colors.border }}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={92}
            tick={{ fontSize: 9, fill: theme.colors.primary, fontWeight: 600, fontFamily: theme.typography.body }}
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            formatter={(value: unknown) => [typeof value === "number" ? `${value.toLocaleString("fr-FR", { maximumFractionDigits: 1 })}${unit}` : String(value), "mCPR moderne"]}
            contentStyle={{ borderRadius: 8, border: `1px solid ${theme.colors.border}`, fontSize: 10, fontFamily: theme.typography.body }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive>
            {sorted.map((item) => <Cell key={item.id} fill={mixHex(rampStart, endColor, Math.max(0, Math.min(1, item.value / axisMax)))} />)}
            <LabelList dataKey="value" position="right" formatter={(value: unknown) => typeof value === "number" ? `${value.toLocaleString("fr-FR", { maximumFractionDigits: 1 })}${unit}` : ""} style={{ fill: theme.colors.text, fontSize: 9, fontWeight: 700 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartScaffold>
  );
}
