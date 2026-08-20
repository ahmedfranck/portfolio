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
  readonly stats: readonly SumBandStat[];
}

export default function SumBand({ eyebrow, title, subtitle, stats }: SumBandProps) {
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
      <div className="relative grid gap-5 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-center">
        <div>
          {eyebrow && (
            <p className="text-[9px] font-bold uppercase tracking-[0.18em] text-white/55">{eyebrow}</p>
          )}
          <h2 className="mt-1 text-xl leading-tight sm:text-2xl" style={{ fontFamily: theme.typography.heading }}>
            {title}
          </h2>
          {subtitle && <p className="mt-2 max-w-2xl text-[11px] leading-relaxed text-white/55">{subtitle}</p>}
        </div>
        <dl className="grid grid-cols-2 gap-x-2 gap-y-4 sm:grid-cols-3 xl:flex xl:items-stretch">
          {stats.slice(0, 5).map((stat, index) => (
            <div
              key={`${stat.label}-${index}`}
              className="min-w-0 px-3 first:pl-0 xl:min-w-28 xl:border-l xl:border-white/10 xl:first:border-0"
            >
              <dd className="flex items-baseline gap-1 text-lg" style={{ color: theme.colors.accentLight, fontFamily: theme.typography.heading }}>
                {stat.value}
                {stat.unit && <span className="text-[10px] text-white/50">{stat.unit}</span>}
              </dd>
              <dt className="mt-1 text-[8px] font-bold uppercase tracking-[0.08em] text-white/40">{stat.label}</dt>
              {(stat.source || stat.illustrative) && (
                <span className="mt-2 block">
                  <AoDataBadges source={stat.source} illustrative={stat.illustrative} />
                </span>
              )}
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
