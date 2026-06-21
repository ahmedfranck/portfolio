import { useState } from "react";
import { ChevronDown, Info, ExternalLink } from "lucide-react";

export interface IndicatorSourceInfo {
  key: string;
  code: string;
  label: string;
  unit: string;
  yearRange: [number, number] | null;
  countriesWithData: number;
  totalCountries: number;
}

interface SourcesPanelProps {
  indicators: IndicatorSourceInfo[];
  note?: string;
}

export default function SourcesPanel({ indicators, note }: SourcesPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-card border border-line bg-white shadow-card">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 px-5 py-4 text-left"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="flex items-center gap-2 font-display text-sm font-semibold text-ink">
          <Info size={16} aria-hidden="true" />
          Sources & méthodologie
        </span>
        <ChevronDown
          size={18}
          className={`text-text-2 transition-transform ${open ? "rotate-180" : ""}`}
          aria-hidden="true"
        />
      </button>
      {open && (
        <div className="border-t border-line px-5 py-4 text-sm text-text-2">
          {note && <p className="mb-3">{note}</p>}
          <ul className="space-y-3">
            {indicators.map((ind) => (
              <li key={ind.key} className="border-b border-line pb-3 last:border-0 last:pb-0">
                <p className="font-medium text-ink">{ind.label}</p>
                <p className="mt-0.5 font-mono text-xs">
                  Source : Banque mondiale — Open Data ({ind.code}) ·{" "}
                  {ind.yearRange ? (
                    <>
                      données {ind.yearRange[0]}–{ind.yearRange[1]} · {ind.countriesWithData}/
                      {ind.totalCountries} pays couverts
                    </>
                  ) : (
                    "aucune donnée disponible pour ces pays"
                  )}
                </p>
                <a
                  href={`https://data.worldbank.org/indicator/${ind.code}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-0.5 inline-flex items-center gap-1 text-xs font-medium text-brand hover:underline"
                >
                  Voir sur data.worldbank.org <ExternalLink size={12} aria-hidden="true" />
                </a>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs">
            Licence des données : CC BY-4.0. Les valeurs manquantes sont affichées « n.d. » et ne sont
            jamais estimées.
          </p>
        </div>
      )}
    </div>
  );
}
