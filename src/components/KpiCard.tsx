import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import CountUpText from "./CountUpText";

interface KpiCardProps {
  label: string;
  value: string;
  unit?: string;
  yoyChange?: number | null;
  helpText?: string;
  /** Si true, une baisse de la valeur est interprétée comme une amélioration (ex. mortalité). */
  lowerIsBetter?: boolean;
}

export default function KpiCard({
  label,
  value,
  unit,
  yoyChange,
  helpText,
  lowerIsBetter = false,
}: KpiCardProps) {
  const isFlat = yoyChange == null || Math.abs(yoyChange) <= 0.05;
  const isIncrease = yoyChange != null && yoyChange > 0.05;
  const isImprovement = !isFlat && (lowerIsBetter ? !isIncrease : isIncrease);

  return (
    <div className="rounded-card border border-line bg-white p-5 shadow-card transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover">
      <p className="font-mono text-[11px] font-medium uppercase tracking-wide text-text-2">{label}</p>
      <p className="mt-2 flex items-baseline gap-1.5">
        <CountUpText text={value} className="font-display text-3xl font-bold text-ink" />
        {unit && <span className="text-sm font-medium text-text-2">{unit}</span>}
      </p>
      {yoyChange != null && (
        <p
          className={`mt-2 flex items-center gap-1 text-sm font-medium ${
            isFlat ? "text-text-2" : isImprovement ? "text-brand-deep" : "text-rose-600"
          }`}
        >
          {isFlat ? (
            <Minus size={16} aria-hidden="true" />
          ) : isIncrease ? (
            <ArrowUpRight size={16} aria-hidden="true" />
          ) : (
            <ArrowDownRight size={16} aria-hidden="true" />
          )}
          {Math.abs(yoyChange).toFixed(1)} % vs année précédente
        </p>
      )}
      {helpText && <p className="mt-1 text-xs text-text-2">{helpText}</p>}
    </div>
  );
}
