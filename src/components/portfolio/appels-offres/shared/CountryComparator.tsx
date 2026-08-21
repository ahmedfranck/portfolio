import type { ReactNode } from "react";
import { GitCompareArrows } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { CountryLabel } from "./CountryFlag";
import CountryMultiSelect from "./CountryMultiSelect";

export interface CountryComparatorMetric {
  readonly id: string;
  readonly label: string;
}

export interface CountryComparatorMixItem {
  readonly label: string;
  readonly value: number;
  readonly color?: string;
}

export interface CountryComparatorEntity {
  readonly id: string;
  readonly iso3: string;
  readonly name: string;
  readonly shortName?: string;
  readonly values: Readonly<Record<string, ReactNode>>;
  readonly mix?: readonly CountryComparatorMixItem[];
}

interface CountryComparatorProps {
  readonly title?: string;
  readonly subtitle?: string;
  readonly entities: readonly CountryComparatorEntity[];
  readonly metrics: readonly CountryComparatorMetric[];
  readonly selectedIds: readonly string[];
  readonly onChange: (ids: string[]) => void;
  readonly min?: number;
  readonly max?: number;
  readonly mixLabel?: string;
  readonly source?: string;
  readonly illustrative?: boolean;
}

const MIX_COLORS = ["#C3911F", "#12365A", "#2E6F73", "#6E8CA8", "#D7B85A", "#AAB4BE"] as const;

export default function CountryComparator({
  title = "Comparaison par pays",
  subtitle = "Sélectionnez les pays à aligner sur les mêmes repères.",
  entities,
  metrics,
  selectedIds,
  onChange,
  min = 2,
  max = 4,
  mixLabel = "Mix méthodes",
}: CountryComparatorProps) {
  const { theme } = useAoTheme();
  const selected = entities.filter((entity) => selectedIds.includes(entity.id)).slice(0, max);

  return (
    <section className="overflow-hidden rounded-[9px] border bg-white shadow-sm" style={{ borderColor: theme.colors.border, fontFamily: theme.typography.body }}>
      <header className="flex flex-wrap items-start justify-between gap-3 border-b px-4 py-3" style={{ borderColor: theme.colors.border }}>
        <div>
          <h3 className="flex items-center gap-2 text-xs" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>
            <GitCompareArrows size={15} style={{ color: theme.colors.accent }} aria-hidden="true" />
            {title}
          </h3>
          <p className="mt-1 text-[10px] leading-relaxed" style={{ color: theme.colors.muted }}>{subtitle}</p>
        </div>
        <span className="rounded-full px-2.5 py-1 text-[8px] font-bold uppercase tracking-[0.08em]" style={{ color: theme.colors.primary, background: theme.colors.soft }}>
          {selected.length} / {max} pays
        </span>
      </header>

      <div className="space-y-4 p-4">
        <CountryMultiSelect
          options={entities.map((entity) => ({ value: entity.id, iso3: entity.iso3, label: entity.shortName ?? entity.name }))}
          value={selectedIds}
          onChange={onChange}
          min={min}
          max={max}
        />

        <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-4">
          {selected.map((entity) => (
            <article key={entity.id} className="min-w-0 overflow-hidden rounded-lg border" style={{ borderColor: theme.colors.border, background: theme.colors.canvas }}>
              <header className="border-b px-3 py-3" style={{ borderColor: theme.colors.border, background: "#FFFFFF" }}>
                <h4 className="text-sm" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>
                  <CountryLabel iso3={entity.iso3} name={entity.name} size="md" />
                </h4>
              </header>
              <dl className="grid grid-cols-2 gap-px bg-slate-200/70">
                {metrics.map((metric) => (
                  <div key={metric.id} className="min-w-0 bg-white px-3 py-2.5">
                    <dt className="text-[7px] font-bold uppercase tracking-[0.07em]" style={{ color: theme.colors.muted }}>{metric.label}</dt>
                    <dd className="mt-1 truncate text-sm font-bold" style={{ color: theme.colors.primary }}>{entity.values[metric.id] ?? "—"}</dd>
                  </div>
                ))}
              </dl>
              {entity.mix && entity.mix.length > 0 && (
                <div className="px-3 py-3">
                  <p className="text-[7px] font-bold uppercase tracking-[0.08em]" style={{ color: theme.colors.muted }}>{mixLabel}</p>
                  <div className="mt-2 flex h-2 overflow-hidden rounded-full" aria-label={`${mixLabel} de ${entity.name}`}>
                    {entity.mix.map((item, index) => (
                      <span key={item.label} className="h-full" style={{ width: `${item.value}%`, background: item.color ?? MIX_COLORS[index % MIX_COLORS.length] }} title={`${item.label} : ${item.value} %`} />
                    ))}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1">
                    {entity.mix.slice(0, 3).map((item, index) => (
                      <span key={item.label} className="inline-flex items-center gap-1 text-[7px]" style={{ color: theme.colors.muted }}>
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: item.color ?? MIX_COLORS[index % MIX_COLORS.length] }} aria-hidden="true" />
                        {item.label} {item.value}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </article>
          ))}
        </div>
      </div>

    </section>
  );
}
