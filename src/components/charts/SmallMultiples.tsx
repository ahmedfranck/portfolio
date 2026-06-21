import { AreaChart, Area, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { PALETTE } from "../../lib/palette";

interface SmallMultipleDatum {
  iso3: string;
  name: string;
  series: { year: number; value: number }[];
  latest: number;
}

interface SmallMultiplesProps {
  data: SmallMultipleDatum[];
  valueSuffix?: string;
}

export default function SmallMultiples({ data, valueSuffix = " %" }: SmallMultiplesProps) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {data.map((d) => (
        <div key={d.iso3} className="rounded-card border border-line bg-surface p-3">
          <div className="flex items-baseline justify-between">
            <p className="text-xs font-medium text-text-2">{d.name}</p>
            <p className="font-display text-sm font-semibold text-ink">
              {d.latest}
              {valueSuffix}
            </p>
          </div>
          <ResponsiveContainer width="100%" height={48}>
            <AreaChart data={d.series} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
              <YAxis hide domain={["auto", "auto"]} />
              <Tooltip
                formatter={(value: unknown) => [`${value}${valueSuffix}`, d.name]}
                labelFormatter={(label) => `Année ${label}`}
                contentStyle={{ borderRadius: 8, border: "1px solid #E4E7EC", fontSize: 12 }}
              />
              <Area type="monotone" dataKey="value" stroke={PALETTE.brand} fill={PALETTE.brand} fillOpacity={0.2} isAnimationActive />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  );
}
