import { RotateCcw, SlidersHorizontal } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import CountryMultiSelect from "./CountryMultiSelect";
import type { CountrySelectOption } from "./CountrySelect";
import YearSlider from "./YearSlider";

export type DashboardFilterOption = CountrySelectOption;

interface DashboardFilterBarProps {
  readonly countryOptions: readonly DashboardFilterOption[];
  readonly selectedCountries: readonly string[];
  readonly onCountriesChange: (countries: string[]) => void;
  readonly maxYearValue: number;
  readonly onMaxYearChange: (year: number) => void;
  readonly minYear: number;
  readonly maxYear: number;
  readonly allLabel?: string;
}

export default function DashboardFilterBar({
  countryOptions,
  selectedCountries,
  onCountriesChange,
  maxYearValue,
  onMaxYearChange,
  minYear,
  maxYear,
  allLabel = "Tous les pays",
}: DashboardFilterBarProps) {
  const { theme } = useAoTheme();
  const validSelection = countryOptions.filter((option) => selectedCountries.includes(option.value));
  const selectedLabel = validSelection.length === countryOptions.length
    ? allLabel
    : validSelection.length === 0
      ? "Aucun pays sélectionné"
      : validSelection.length === 1
        ? validSelection[0].label
        : `${validSelection.length} pays sélectionnés`;
  const isDefault = validSelection.length === countryOptions.length && maxYearValue === maxYear;

  function reset() {
    onCountriesChange(countryOptions.map((option) => option.value));
    onMaxYearChange(maxYear);
  }

  return (
    <section className="sticky top-0 z-30 rounded-[9px] border bg-white/95 shadow-sm backdrop-blur" style={{ borderColor: theme.colors.border, fontFamily: theme.typography.body }} aria-label="Filtres globaux du dashboard">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-4 py-2.5" style={{ borderColor: theme.colors.border }}>
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-md" style={{ color: theme.colors.accent, background: theme.colors.soft }}><SlidersHorizontal size={14} aria-hidden="true" /></span>
          <div>
            <h2 className="text-[10px] font-bold uppercase tracking-[0.1em]" style={{ color: theme.colors.primary }}>Filtres globaux</h2>
            <p className="text-[8px]" style={{ color: theme.colors.muted }}>{selectedLabel} · données jusqu’en {maxYearValue}</p>
          </div>
        </div>
        <button type="button" onClick={reset} disabled={isDefault} className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[8px] font-bold disabled:cursor-default disabled:opacity-35" style={{ borderColor: theme.colors.border, color: theme.colors.primary }}>
          <RotateCcw size={10} aria-hidden="true" /> Réinitialiser
        </button>
      </div>

      <div className="grid gap-4 px-4 py-3 md:grid-cols-2 md:items-end">
        <CountryMultiSelect
          label="Filtre pays"
          options={countryOptions}
          value={selectedCountries}
          onChange={onCountriesChange}
          min={0}
          max={countryOptions.length}
          allOption
          allLabel={allLabel}
          summaryMode="count"
        />
        <YearSlider value={maxYearValue} onChange={onMaxYearChange} min={minYear} max={maxYear} />
      </div>
    </section>
  );
}
