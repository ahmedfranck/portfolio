import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface StackedAreaSeries {
  key: string;
  label: string;
  color: string;
}

interface StackedAreaChartProps {
  data: Record<string, number | string>[];
  series: StackedAreaSeries[];
  valueSuffix?: string;
  height?: number;
}

export default function StackedAreaChart({
  data,
  series,
  valueSuffix = " %",
  height = 360,
}: StackedAreaChartProps) {
  const labelByKey = Object.fromEntries(series.map((s) => [s.key, s.label]));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E4E7EC" />
        <XAxis dataKey="year" tick={{ fontSize: 12, fill: "#586072" }} />
        <YAxis tick={{ fontSize: 12, fill: "#586072" }} />
        <Tooltip
          formatter={(value: unknown, name: unknown) => [
            `${value}${valueSuffix}`,
            labelByKey[name as string] ?? String(name),
          ]}
          labelFormatter={(label) => `Année ${label}`}
          contentStyle={{ borderRadius: 10, border: "1px solid #E4E7EC", fontSize: 13 }}
        />
        <Legend formatter={(value) => labelByKey[value] ?? value} wrapperStyle={{ fontSize: 12 }} />
        {series.map((s) => (
          <Area
            key={s.key}
            type="monotone"
            dataKey={s.key}
            stackId="1"
            stroke={s.color}
            fill={s.color}
            fillOpacity={0.85}
            isAnimationActive
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
