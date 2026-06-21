import { AreaChart, Area, ResponsiveContainer, YAxis } from "recharts";
import { PALETTE } from "../../lib/palette";

interface IndicatorMiniSeriesProps {
  label: string;
  unit?: string;
  series: { year: number; value: number }[];
  latest: number | null;
  formatValue?: (n: number) => string;
}

export default function IndicatorMiniSeries({
  label,
  unit = "",
  series,
  latest,
  formatValue,
}: IndicatorMiniSeriesProps) {
  const fmt = formatValue ?? ((n: number) => n.toLocaleString("fr-FR"));

  return (
    <div className="rounded-card border border-line bg-white p-4 shadow-card">
      <p className="font-mono text-[11px] uppercase tracking-wide text-text-2">{label}</p>
      <p className="mt-1 font-display text-2xl font-semibold text-ink">
        {latest != null ? fmt(latest) : "n.d."}
        {latest != null && unit ? <span className="ml-1 text-sm font-normal text-text-2">{unit}</span> : null}
      </p>
      {series.length > 1 && (
        <ResponsiveContainer width="100%" height={48}>
          <AreaChart data={series} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
            <YAxis hide domain={["auto", "auto"]} />
            <Area
              type="monotone"
              dataKey="value"
              stroke={PALETTE.brand}
              fill={PALETTE.brand}
              fillOpacity={0.18}
              isAnimationActive
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
