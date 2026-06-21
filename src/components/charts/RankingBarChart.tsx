import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { PALETTE } from "../../lib/palette";

interface RankingDatum {
  name: string;
  value: number;
}

interface RankingBarChartProps {
  data: RankingDatum[];
  valueSuffix?: string;
  height?: number;
  highlightColor?: string;
}

export default function RankingBarChart({
  data,
  valueSuffix = "",
  height = 380,
  highlightColor = PALETTE.brand,
}: RankingBarChartProps) {
  const sorted = [...data].sort((a, b) => b.value - a.value);

  return (
    <ResponsiveContainer width="100%" height={Math.max(height, sorted.length * 26)}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E4E7EC" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 12, fill: "#586072" }} />
        <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12, fill: "#14161B" }} />
        <Tooltip
          formatter={(value: unknown) => [`${value}${valueSuffix}`, "Valeur"]}
          contentStyle={{ borderRadius: 10, border: "1px solid #E4E7EC", fontSize: 13 }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive>
          {sorted.map((_, i) => (
            <Cell key={i} fill={highlightColor} fillOpacity={1 - i * (0.5 / sorted.length)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
