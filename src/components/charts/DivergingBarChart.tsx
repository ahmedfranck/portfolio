import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";

interface DivergingDatum {
  name: string;
  value: number; // peut être positif (filles > garçons) ou négatif (filles < garçons)
}

interface DivergingBarChartProps {
  data: DivergingDatum[];
  valueSuffix?: string;
  height?: number;
  positiveColor?: string;
  negativeColor?: string;
}

export default function DivergingBarChart({
  data,
  valueSuffix = " pts",
  height = 380,
  positiveColor = "#5B4BE3",
  negativeColor = "#F59E0B",
}: DivergingBarChartProps) {
  const sorted = [...data].sort((a, b) => a.value - b.value);

  return (
    <ResponsiveContainer width="100%" height={Math.max(height, sorted.length * 26)}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E4E7EC" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 12, fill: "#586072" }} />
        <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12, fill: "#14161B" }} />
        <ReferenceLine x={0} stroke="#586072" />
        <Tooltip
          formatter={(value: unknown) => [
            `${Number(value) > 0 ? "+" : ""}${value}${valueSuffix}`,
            "Écart filles - garçons",
          ]}
          contentStyle={{ borderRadius: 10, border: "1px solid #E4E7EC", fontSize: 13 }}
        />
        <Bar dataKey="value" radius={4} maxBarSize={18} isAnimationActive>
          {sorted.map((d, i) => (
            <Cell key={i} fill={d.value >= 0 ? positiveColor : negativeColor} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
