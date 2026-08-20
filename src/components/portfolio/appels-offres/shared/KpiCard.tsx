import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";
import type { ReactNode } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { AoDataBadges } from "./badges";

export type KpiTrend = "up" | "down" | "flat";

interface KpiCardProps {
  readonly label: string;
  readonly value: ReactNode;
  readonly unit?: string;
  readonly context?: string;
  readonly delta?: string;
  readonly trend?: KpiTrend;
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
  positive,
  source,
  illustrative = false,
  accent,
}: KpiCardProps) {
  const { theme } = useAoTheme();
  const TrendIcon = trend === "up" ? ArrowUpRight : trend === "down" ? ArrowDownRight : ArrowRight;
  const trendColor = positive == null
    ? theme.colors.muted
    : positive
      ? theme.colors.positive
      : theme.colors.negative;

  return (
    <article
      className="group relative overflow-hidden rounded-[9px] border bg-white p-4 shadow-sm transition duration-200 hover:-translate-y-0.5 hover:shadow-md"
      style={{ borderColor: theme.colors.border, fontFamily: theme.typography.body }}
    >
      <span className="absolute inset-x-0 bottom-0 h-[3px]" style={{ background: accent ?? theme.colors.accent }} aria-hidden="true" />
      <p className="text-[9px] font-bold uppercase tracking-[0.08em]" style={{ color: theme.colors.muted }}>{label}</p>
      <p className="mt-2 flex items-baseline gap-1 text-2xl leading-none" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>
        {value}
        {unit && <span className="text-[10px] font-semibold" style={{ color: theme.colors.muted, fontFamily: theme.typography.body }}>{unit}</span>}
      </p>
      {context && <p className="mt-2 text-[10px] leading-relaxed" style={{ color: theme.colors.muted }}>{context}</p>}
      {delta && (
        <p className="mt-2 inline-flex items-center gap-1 rounded-full px-2 py-1 text-[9px] font-bold" style={{ color: trendColor, background: `${trendColor}12` }}>
          <TrendIcon size={11} aria-hidden="true" /> {delta}
        </p>
      )}
      {(source || illustrative) && <div className="mt-3"><AoDataBadges source={source} illustrative={illustrative} sourceDisplay="disclosure" /></div>}
    </article>
  );
}
