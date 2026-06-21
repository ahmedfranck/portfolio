import { useState, useRef, useEffect } from "react";
import { ChevronDown, Check } from "lucide-react";
import { COUNTRIES } from "../../data/countries";

interface CountryMultiSelectProps {
  selected: string[];
  onChange: (codes: string[]) => void;
  label?: string;
}

export default function CountryMultiSelect({
  selected,
  onChange,
  label = "Pays",
}: CountryMultiSelectProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  function toggle(iso3: string) {
    if (selected.includes(iso3)) {
      onChange(selected.filter((c) => c !== iso3));
    } else {
      onChange([...selected, iso3]);
    }
  }

  const summary =
    selected.length === 0
      ? "Aucun pays"
      : selected.length === COUNTRIES.length
        ? "Tous les pays"
        : `${selected.length} pays sélectionnés`;

  return (
    <div className="relative" ref={ref}>
      <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-2">
        {label}
      </label>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full min-w-[180px] cursor-pointer items-center justify-between gap-2 rounded-lg border border-line bg-white px-3 py-2 text-sm text-ink transition-colors duration-200 hover:border-brand focus:border-brand focus:outline-none"
      >
        <span>{summary}</span>
        <ChevronDown size={16} className={`transition-transform ${open ? "rotate-180" : ""}`} aria-hidden="true" />
      </button>
      {open && (
        <div className="absolute z-20 mt-1 max-h-72 w-64 overflow-auto rounded-lg border border-line bg-white p-2 shadow-cardHover">
          <div className="mb-1 flex gap-2 border-b border-line pb-2">
            <button
              type="button"
              className="cursor-pointer text-xs font-medium text-brand underline-offset-4 hover:underline"
              onClick={() => onChange(COUNTRIES.map((c) => c.iso3))}
            >
              Tout sélectionner
            </button>
            <button
              type="button"
              className="cursor-pointer text-xs font-medium text-text-2 underline-offset-4 transition-colors duration-200 hover:text-brand hover:underline"
              onClick={() => onChange([])}
            >
              Tout effacer
            </button>
          </div>
          {COUNTRIES.map((c) => {
            const checked = selected.includes(c.iso3);
            return (
              <button
                key={c.iso3}
                type="button"
                onClick={() => toggle(c.iso3)}
                className="flex w-full cursor-pointer items-center justify-between gap-2 rounded-md px-2 py-1.5 text-sm transition-colors duration-150 hover:bg-brand-soft hover:text-brand"
              >
                <span>{c.name}</span>
                {checked && <Check size={14} className="text-brand" aria-hidden="true" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
