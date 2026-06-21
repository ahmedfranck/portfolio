import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ZAxis,
  Cell,
} from "recharts";
import { colorForIndex } from "../../lib/palette";

interface ScatterDatum {
  iso3: string;
  name: string;
  x: number;
  y: number;
  z?: number;
}

interface ScatterChartCardProps {
  data: ScatterDatum[];
  xLabel: string;
  yLabel: string;
  xSuffix?: string;
  ySuffix?: string;
  height?: number;
}

export default function ScatterChartCard({
  data,
  xLabel,
  yLabel,
  xSuffix = "",
  ySuffix = "",
  height = 360,
}: ScatterChartCardProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 24, bottom: 24, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#E4E7EC" />
        <XAxis
          type="number"
          dataKey="x"
          name={xLabel}
          tick={{ fontSize: 12, fill: "#586072" }}
          label={{ value: xLabel, position: "insideBottom", offset: -10, fontSize: 12, fill: "#586072" }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel}
          tick={{ fontSize: 12, fill: "#586072" }}
          label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 12, fill: "#586072" }}
        />
        {data.some((d) => d.z != null) && <ZAxis type="number" dataKey="z" range={[60, 400]} />}
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          content={({ active, payload }) => {
            if (!active || !payload || payload.length === 0) return null;
            const d = payload[0].payload as ScatterDatum;
            return (
              <div className="rounded-lg border border-line bg-white px-3 py-2 text-xs shadow-cardHover">
                <p className="font-display font-semibold text-ink">{d.name}</p>
                <p className="text-text-2">
                  {xLabel} : {d.x}
                  {xSuffix}
                </p>
                <p className="text-text-2">
                  {yLabel} : {d.y}
                  {ySuffix}
                </p>
              </div>
            );
          }}
        />
        <Scatter data={data} fill="#5B4BE3" isAnimationActive>
          {data.map((_, i) => (
            <Cell key={i} fill={colorForIndex(i)} fillOpacity={0.85} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
