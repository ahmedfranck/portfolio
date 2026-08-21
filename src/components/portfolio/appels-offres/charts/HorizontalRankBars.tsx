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
  readonly referenceYear?: number;
}

interface YearTickProps {
  readonly x?: string | number;
  readonly y?: string | number;
  readonly payload?: { readonly value?: string };
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
  referenceYear = new Date().getFullYear(),
}: HorizontalRankBarsProps) {
  const { theme } = useAoTheme();
  const endColor = rampEnd ?? theme.colors.primary;
  const sorted = [...data].sort((a, b) => b.value - a.value);
  const ticks = Array.from({ length: Math.floor(axisMax / 5) + 1 }, (_, index) => index * 5);
  const tableRows: ChartTableRow[] = sorted.map((item) => ({ country: item.name, value: item.value, year: item.year }));
  const yearByName = new Map(sorted.map((item) => [item.name, item.year]));

  function yearColors(year?: number) {
    if (year == null) return { background: theme.colors.canvas, color: theme.colors.muted };
    const age = referenceYear - year;
    if (age < 2) return { background: `${theme.colors.positive}18`, color: theme.colors.positive };
    if (age <= 4) return { background: "#FEF0E0", color: theme.colors.warning };
    return { background: `${theme.colors.negative}16`, color: theme.colors.negative };
  }

  function YearTick({ x = 0, y = 0, payload }: YearTickProps) {
    const name = payload?.value ?? "";
    const year = yearByName.get(name);
    const colors = yearColors(year);
    const resolvedX = Number(x) || 0;
    const resolvedY = Number(y) || 0;
    return (
      <g transform={`translate(${resolvedX},${resolvedY})`}>
        <text x={-46} y={3} textAnchor="end" fill={theme.colors.primary} fontSize={9} fontWeight={600} fontFamily={theme.typography.body}>{name}</text>
        <rect x={-40} y={-8} width={34} height={16} rx={8} fill={colors.background} />
        <text x={-23} y={3} textAnchor="middle" fill={colors.color} fontSize={9} fontWeight={700} fontFamily={theme.typography.body}>{year ?? "n.d."}</text>
      </g>
    );
  }

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
      <div className="mb-1 flex flex-wrap justify-end gap-2 text-[8px]" style={{ color: theme.colors.muted }}>
        <span>Millésime :</span>
        <span className="rounded-full px-2 py-0.5" style={yearColors(referenceYear)}>moins de 2 ans</span>
        <span className="rounded-full px-2 py-0.5" style={yearColors(referenceYear - 3)}>2–4 ans</span>
        <span className="rounded-full px-2 py-0.5" style={yearColors(referenceYear - 5)}>plus de 4 ans</span>
      </div>
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
            width={148}
            tick={YearTick}
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
