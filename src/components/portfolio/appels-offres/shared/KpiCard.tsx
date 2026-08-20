import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";
import type { ReactNode } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

export type KpiTrend = "up" | "down" | "flat";

interface KpiCardProps {
  readonly label: string;
  readonly value: ReactNode;
  readonly unit?: string;
  readonly context?: string;
  readonly delta?: string;
  readonly trend?: KpiTrend;
  readonly trendMagnitude?: number;
  readonly positive?: boolean;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly accent?: string;
}

export default function KpiCard({
  label,
  value,
  unit,
  context,
  delta,
  trend = "flat",
  trendMagnitude = 55,
  positive,
  accent,
}: KpiCardProps) {
  const { theme } = useAoTheme();
  const TrendIcon = trend === "up" ? ArrowUpRight : trend === "down" ? ArrowDownRight : ArrowRight;
  const trendPolarity = positive ?? (trend === "up" ? true : trend === "down" ? false : null);
  const trendColor = trendPolarity == null
    ? theme.colors.muted
    : trendPolarity
      ? theme.colors.positive
      : theme.colors.negative;
  const railWidth = `${Math.max(12, Math.min(100, trendMagnitude))}%`;

  return (
    <article
      className="group relative overflow-hidden rounded-[9px] border bg-white p-4 shadow-sm transition duration-200 hover:-translate-y-0.5 hover:shadow-md"
      style={{ borderColor: theme.colors.border, fontFamily: theme.typography.body }}
    >
      <span className="absolute inset-x-0 bottom-0 h-[4px]" style={{ background: `${theme.colors.border}80` }} aria-hidden="true">
        <span className="block h-full rounded-r-full transition-[width] duration-500" style={{ width: delta ? railWidth : "100%", background: delta ? trendColor : accent ?? theme.colors.accent }} />
      </span>
      <p className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: theme.colors.muted }}>{label}</p>
      <p className="mt-2 flex items-baseline gap-1 text-2xl leading-none" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>
        {value}
        {unit && <span className="text-[10px] font-semibold" style={{ color: theme.colors.muted, fontFamily: theme.typography.body }}>{unit}</span>}
      </p>
      {context && <p className="mt-2 text-[10px] leading-relaxed" style={{ color: theme.colors.muted }}>{context}</p>}
      {delta && (
        <p className="mt-3 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1.5 text-[10px] font-bold" style={{ color: trendColor, borderColor: `${trendColor}38`, background: `${trendColor}12` }}>
          <span className="flex h-5 w-5 items-center justify-center rounded-full" style={{ background: `${trendColor}18` }}>
            <TrendIcon size={14} strokeWidth={2.4} aria-hidden="true" />
          </span>
          {delta}
        </p>
      )}
    </article>
  );
}
