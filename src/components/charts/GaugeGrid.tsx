interface GaugeDatum {
  iso3: string;
  name: string;
  value: number; // 0-100
}

interface GaugeGridProps {
  data: GaugeDatum[];
  color?: string;
}

function Gauge({ name, value, color }: { name: string; value: number; color: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  const radius = 40;
  const circumference = Math.PI * radius; // demi-cercle
  const offset = circumference * (1 - clamped / 100);

  return (
    <div className="flex flex-col items-center gap-1.5">
      <svg viewBox="0 0 100 56" width="100" height="56" role="img" aria-label={`${name} : ${clamped}%`}>
        <path
          d="M 10 50 A 40 40 0 0 1 90 50"
          fill="none"
          stroke="#EDEFF3"
          strokeWidth="9"
          strokeLinecap="round"
        />
        <path
          d="M 10 50 A 40 40 0 0 1 90 50"
          fill="none"
          stroke={color}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
        <text x="50" y="46" textAnchor="middle" fontSize="16" fontWeight="700" fill="#14161B">
          {Math.round(clamped)}%
        </text>
      </svg>
      <span className="text-center text-xs font-medium text-text-2">{name}</span>
    </div>
  );
}

export default function GaugeGrid({ data, color = "#5B4BE3" }: GaugeGridProps) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 md:grid-cols-8">
      {data.map((d) => (
        <Gauge key={d.iso3} name={d.name} value={d.value} color={color} />
      ))}
    </div>
  );
}
