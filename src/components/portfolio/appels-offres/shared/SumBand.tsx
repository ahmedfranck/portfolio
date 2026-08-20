import type { ReactNode } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { AoDataBadges } from "./badges";

export interface SumBandStat {
  readonly label: string;
  readonly value: ReactNode;
  readonly unit?: string;
  readonly source?: string;
  readonly illustrative?: boolean;
}

interface SumBandProps {
  readonly eyebrow?: string;
  readonly title: string;
  readonly subtitle?: string;
  readonly badge?: string;
  readonly illustrativeNotice?: string;
  readonly stats: readonly SumBandStat[];
}

export default function SumBand({ eyebrow, title, subtitle, badge, illustrativeNotice, stats }: SumBandProps) {
  const { theme } = useAoTheme();

  return (
    <section
      className="relative overflow-hidden rounded-xl px-5 py-5 text-white shadow-sm"
      style={{
        background: `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.primaryDark})`,
        fontFamily: theme.typography.body,
      }}
      aria-label={title}
    >
      <span
        className="pointer-events-none absolute -right-12 -top-12 h-40 w-40 rounded-full"
        style={{ background: `${theme.colors.accent}18` }}
        aria-hidden="true"
      />
      <div className="relative space-y-5">
        <div className="min-w-[320px]">
          {eyebrow && (
            <p className="text-[9px] font-bold uppercase tracking-[0.18em] text-white/55">{eyebrow}</p>
          )}
          <div className="mt-1 flex flex-wrap items-start justify-between gap-3">
            <h2 className="max-w-[720px] text-xl leading-tight sm:text-2xl" style={{ fontFamily: theme.typography.heading }}>{title}</h2>
            {badge && <span className="shrink-0 rounded-full border px-3 py-1 text-[8px] font-bold uppercase tracking-[0.09em]" style={{ borderColor: `${theme.colors.accentLight}66`, color: theme.colors.accentLight, background: `${theme.colors.accent}18` }}>{badge}</span>}
          </div>
          {subtitle && <p className="mt-2 max-w-[720px] text-[11px] leading-relaxed text-white/55">{subtitle}</p>}
        </div>
        {illustrativeNotice && (
          <p className="inline-flex max-w-full items-center rounded-full px-3 py-1.5 text-[8px] font-bold" style={{ color: theme.colors.warning, background: "#FEF0E0" }}>
            ⚠ {illustrativeNotice}
          </p>
        )}
        <dl className="grid grid-cols-2 gap-x-2 gap-y-4 lg:grid-cols-4">
          {stats.slice(0, 4).map((stat, index) => (
            <div
              key={`${stat.label}-${index}`}
              className="min-w-0 border-l border-white/10 px-3 first:border-0 first:pl-0"
            >
              <dd className="flex items-baseline gap-1 text-lg" style={{ color: theme.colors.accentLight, fontFamily: theme.typography.heading }}>
                {stat.value}
                {stat.unit && <span className="text-[10px] text-white/50">{stat.unit}</span>}
              </dd>
              <dt className="mt-1 text-[8px] font-bold uppercase tracking-[0.08em] text-white/40">{stat.label}</dt>
              {(stat.source || stat.illustrative) && (
                <span className="mt-2 block">
                  <AoDataBadges source={stat.source} illustrative={stat.illustrative} compactSource />
                </span>
              )}
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
