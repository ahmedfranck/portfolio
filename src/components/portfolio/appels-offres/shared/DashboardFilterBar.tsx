import type { ReactNode } from "react";
import { CalendarRange, Check, RotateCcw, SlidersHorizontal } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

export interface DashboardFilterOption {
  readonly value: string;
  readonly label: ReactNode;
}

interface DashboardFilterBarProps {
  readonly countryOptions: readonly DashboardFilterOption[];
  readonly selectedCountries: readonly string[];
  readonly onCountriesChange: (countries: string[]) => void;
  readonly yearRange: readonly [number, number];
  readonly onYearRangeChange: (range: [number, number]) => void;
  readonly minYear: number;
  readonly maxYear: number;
  readonly minSelectedCountries?: number;
  readonly allLabel?: string;
}

export default function DashboardFilterBar({
  countryOptions,
  selectedCountries,
  onCountriesChange,
  yearRange,
  onYearRangeChange,
  minYear,
  maxYear,
  minSelectedCountries = 1,
  allLabel = "Tous les pays",
}: DashboardFilterBarProps) {
  const { theme } = useAoTheme();
  const years = Array.from({ length: maxYear - minYear + 1 }, (_, index) => minYear + index);
  const allSelected = selectedCountries.length === countryOptions.length;
  const isDefault = allSelected && yearRange[0] === minYear && yearRange[1] === maxYear;

  function toggleCountry(value: string) {
    if (selectedCountries.includes(value)) {
      if (selectedCountries.length <= minSelectedCountries) return;
      onCountriesChange(selectedCountries.filter((country) => country !== value));
      return;
    }
    onCountriesChange([...selectedCountries, value]);
  }

  function setStart(value: number) {
    onYearRangeChange([value, Math.max(value, yearRange[1])]);
  }

  function setEnd(value: number) {
    onYearRangeChange([Math.min(yearRange[0], value), value]);
  }

  function reset() {
    onCountriesChange(countryOptions.map((option) => option.value));
    onYearRangeChange([minYear, maxYear]);
  }

  return (
    <section className="sticky top-0 z-30 rounded-[9px] border bg-white/95 shadow-sm backdrop-blur" style={{ borderColor: theme.colors.border, fontFamily: theme.typography.body }} aria-label="Filtres globaux du dashboard">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-4 py-2.5" style={{ borderColor: theme.colors.border }}>
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-md" style={{ color: theme.colors.accent, background: theme.colors.soft }}><SlidersHorizontal size={14} aria-hidden="true" /></span>
          <div>
            <h2 className="text-[10px] font-bold uppercase tracking-[0.1em]" style={{ color: theme.colors.primary }}>Filtres globaux</h2>
            <p className="text-[8px]" style={{ color: theme.colors.muted }}>{selectedCountries.length} pays (min. {minSelectedCountries}) · {yearRange[0]}–{yearRange[1]}</p>
          </div>
        </div>
        <button type="button" onClick={reset} disabled={isDefault} className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[8px] font-bold disabled:cursor-default disabled:opacity-35" style={{ borderColor: theme.colors.border, color: theme.colors.primary }}>
          <RotateCcw size={10} aria-hidden="true" /> Réinitialiser
        </button>
      </div>

      <div className="grid gap-3 px-4 py-3 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-center">
        <fieldset>
          <legend className="sr-only">Pays inclus</legend>
          <div className="flex flex-wrap gap-1.5">
            <button type="button" aria-pressed={allSelected} onClick={() => onCountriesChange(countryOptions.map((option) => option.value))} className="rounded-full border px-2.5 py-1 text-[8px] font-bold" style={{ borderColor: allSelected ? theme.colors.accent : theme.colors.border, background: allSelected ? theme.colors.soft : "#FFFFFF", color: theme.colors.primary }}>{allLabel}</button>
            {countryOptions.map((option) => {
              const active = selectedCountries.includes(option.value);
              const locked = active && selectedCountries.length <= minSelectedCountries;
              return (
                <button key={option.value} type="button" aria-pressed={active} aria-disabled={locked} title={locked ? `Conserver au moins ${minSelectedCountries} pays` : undefined} onClick={() => toggleCountry(option.value)} className="inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[8px] font-semibold transition" style={{ borderColor: active ? theme.colors.primary : theme.colors.border, background: active ? theme.colors.primary : "#FFFFFF", color: active ? "#FFFFFF" : theme.colors.primary }}>
                  {option.label}{active && <Check size={9} aria-hidden="true" />}
                </button>
              );
            })}
          </div>
        </fieldset>

        <fieldset className="flex flex-wrap items-center gap-2">
          <legend className="sr-only">Plage d’années</legend>
          <CalendarRange size={14} style={{ color: theme.colors.accent }} aria-hidden="true" />
          <label className="inline-flex items-center gap-1.5 text-[8px] font-bold uppercase tracking-[0.06em]" style={{ color: theme.colors.muted }}>
            De
            <select value={yearRange[0]} onChange={(event) => setStart(Number(event.target.value))} className="rounded-md border bg-white px-2 py-1.5 text-[9px] font-bold" style={{ borderColor: theme.colors.border, color: theme.colors.primary }}>
              {years.filter((year) => year <= yearRange[1]).map((year) => <option key={year} value={year}>{year}</option>)}
            </select>
          </label>
          <label className="inline-flex items-center gap-1.5 text-[8px] font-bold uppercase tracking-[0.06em]" style={{ color: theme.colors.muted }}>
            À
            <select value={yearRange[1]} onChange={(event) => setEnd(Number(event.target.value))} className="rounded-md border bg-white px-2 py-1.5 text-[9px] font-bold" style={{ borderColor: theme.colors.border, color: theme.colors.primary }}>
              {years.filter((year) => year >= yearRange[0]).map((year) => <option key={year} value={year}>{year}</option>)}
            </select>
          </label>
        </fieldset>
      </div>
    </section>
  );
}
