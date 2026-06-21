import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { COUNTRY_NAMES } from "../../data/countries";
import { colorForIndex } from "../../lib/palette";

interface MultiLineChartProps {
  data: Record<string, number | string>[]; // [{ year, BEN: 12.3, SEN: 14.1, ... }]
  seriesCodes: string[];
  yLabel?: string;
  valueSuffix?: string;
  height?: number;
}

export default function MultiLineChart({
  data,
  seriesCodes,
  yLabel,
  valueSuffix = "",
  height = 320,
}: MultiLineChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E4E7EC" />
        <XAxis dataKey="year" tick={{ fontSize: 12, fill: "#586072" }} />
        <YAxis
          tick={{ fontSize: 12, fill: "#586072" }}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fontSize: 12, fill: "#586072" } : undefined}
        />
        <Tooltip
          formatter={(value: unknown, name: unknown) => [
            `${value}${valueSuffix}`,
            COUNTRY_NAMES[name as string] ?? String(name),
          ]}
          labelFormatter={(label) => `Année ${label}`}
          contentStyle={{ borderRadius: 10, border: "1px solid #E4E7EC", fontSize: 13 }}
        />
        <Legend formatter={(value) => COUNTRY_NAMES[value] ?? value} wrapperStyle={{ fontSize: 12 }} />
        {seriesCodes.map((code, i) => (
          <Line
            key={code}
            type="monotone"
            dataKey={code}
            stroke={colorForIndex(i)}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
            isAnimationActive
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
