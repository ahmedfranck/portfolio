import { rampStops, type IndicatorPolarity } from "../lib/palette";

interface MapLegendProps {
  min: number;
  max: number;
  unit?: string;
  polarity?: IndicatorPolarity;
  formatValue?: (n: number) => string;
}

export default function MapLegend({ min, max, unit = "", polarity = "positive", formatValue }: MapLegendProps) {
  const stops = rampStops(polarity);
  const fmt = formatValue ?? ((n: number) => Math.round(n).toLocaleString("fr-FR"));

  return (
    <div className="flex flex-wrap items-center gap-5 font-mono text-xs text-text-2">
      <div className="flex items-center gap-2">
        <span>
          {fmt(min)}
          {unit}
        </span>
        <div
          className="h-2.5 w-32 rounded-full"
          style={{ background: `linear-gradient(to right, ${stops.join(", ")})` }}
          aria-hidden="true"
        />
        <span>
          {fmt(max)}
          {unit}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="hatched-missing h-2.5 w-5 rounded-full border border-line" aria-hidden="true" />
        <span>n.d.</span>
      </div>
    </div>
  );
}
