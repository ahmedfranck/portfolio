import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown, Globe2 } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { CountryLabel } from "./CountryFlag";

export interface CountrySelectOption {
  readonly value: string;
  readonly iso3: string;
  readonly label: string;
}

interface CountrySelectProps {
  readonly label?: string;
  readonly value: string | null;
  readonly options: readonly CountrySelectOption[];
  readonly onChange: (value: string | null) => void;
  readonly allLabel?: string;
  readonly allOption?: boolean;
}

export default function CountrySelect({ label = "Filtre pays", value, options, onChange, allLabel = "Tous les pays", allOption = true }: CountrySelectProps) {
  const { theme } = useAoTheme();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const listboxId = useId();
  const selected = options.find((option) => option.value === value) ?? null;

  useEffect(() => {
    function closeOnOutsideClick(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, []);

  function choose(nextValue: string | null) {
    onChange(nextValue);
    setOpen(false);
  }

  return (
    <div ref={rootRef} className="relative min-w-[220px]" style={{ fontFamily: theme.typography.body }}>
      <span className="mb-1.5 block text-[8px] font-bold uppercase tracking-[0.1em]" style={{ color: theme.colors.muted }}>{label}</span>
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        onClick={() => setOpen((current) => !current)}
        className="flex w-full items-center justify-between gap-3 rounded-md border bg-white px-3 py-2 text-left text-[10px] font-semibold shadow-sm transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
        style={{ borderColor: open ? theme.colors.accent : theme.colors.border, color: theme.colors.primary, outlineColor: theme.colors.accent }}
      >
        <span className="inline-flex min-w-0 items-center gap-2">
          {selected ? <CountryLabel iso3={selected.iso3} name={selected.label} size="md" /> : allOption ? <><Globe2 size={15} style={{ color: theme.colors.accent }} aria-hidden="true" /><span>{allLabel}</span></> : <span>Sélectionner un pays</span>}
        </span>
        <ChevronDown size={14} className={`shrink-0 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden="true" />
      </button>

      {open && (
        <div id={listboxId} role="listbox" aria-label={label} className="absolute left-0 top-[calc(100%+.35rem)] z-50 max-h-80 w-full min-w-[240px] overflow-auto rounded-md border bg-white p-1.5 shadow-xl" style={{ borderColor: theme.colors.border }}>
          {allOption && <>
            <button type="button" role="option" aria-selected={value == null} onClick={() => choose(null)} className="flex w-full items-center justify-between gap-2 rounded px-2.5 py-2 text-left text-[10px] font-semibold transition hover:bg-slate-50" style={{ color: theme.colors.primary, background: value == null ? theme.colors.soft : undefined }}>
              <span className="inline-flex items-center gap-2"><Globe2 size={15} style={{ color: theme.colors.accent }} aria-hidden="true" />{allLabel}</span>
              {value == null && <Check size={13} style={{ color: theme.colors.accent }} aria-hidden="true" />}
            </button>
            <div className="my-1 border-t" style={{ borderColor: theme.colors.border }} />
          </>}
          {options.map((option) => {
            const active = option.value === value;
            return (
              <button key={option.value} type="button" role="option" aria-selected={active} onClick={() => choose(option.value)} className="flex w-full items-center justify-between gap-2 rounded px-2.5 py-2 text-left text-[10px] font-semibold transition hover:bg-slate-50" style={{ color: theme.colors.primary, background: active ? theme.colors.soft : undefined }}>
                <CountryLabel iso3={option.iso3} name={option.label} size="md" />
                {active && <Check size={13} style={{ color: theme.colors.accent }} aria-hidden="true" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
