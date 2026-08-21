import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { CountryLabel } from "./CountryFlag";
import type { CountrySelectOption } from "./CountrySelect";

interface CountryMultiSelectProps {
  readonly label?: string;
  readonly options: readonly CountrySelectOption[];
  readonly value: readonly string[];
  readonly onChange: (value: string[]) => void;
  readonly min?: number;
  readonly max?: number;
}

export default function CountryMultiSelect({ label = "Pays à comparer", options, value, onChange, min = 2, max = 4 }: CountryMultiSelectProps) {
  const { theme } = useAoTheme();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const listboxId = useId();
  const selectedOptions = options.filter((option) => value.includes(option.value));
  const summary = selectedOptions.length <= 3
    ? selectedOptions.map((option) => option.label).join(", ")
    : `${selectedOptions.length} pays sélectionnés`;

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

  function toggle(id: string) {
    if (value.includes(id)) {
      if (value.length <= min) return;
      onChange(value.filter((item) => item !== id));
      return;
    }
    if (value.length >= max) return;
    onChange([...value, id]);
  }

  return (
    <div ref={rootRef} className="relative min-w-[260px]" style={{ fontFamily: theme.typography.body }}>
      <span className="mb-1.5 block text-[8px] font-bold uppercase tracking-[0.1em]" style={{ color: theme.colors.muted }}>{label}</span>
      <button type="button" aria-haspopup="listbox" aria-expanded={open} aria-controls={listboxId} onClick={() => setOpen((current) => !current)} className="flex w-full items-center justify-between gap-3 rounded-md border bg-white px-3 py-2 text-left text-[10px] font-semibold shadow-sm transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2" style={{ borderColor: open ? theme.colors.accent : theme.colors.border, color: theme.colors.primary, outlineColor: theme.colors.accent }}>
        <span className="truncate">{summary || `Sélectionnez ${min} à ${max} pays`}</span>
        <span className="inline-flex shrink-0 items-center gap-1.5"><span className="rounded-full px-1.5 py-0.5 text-[8px]" style={{ background: theme.colors.soft }}>{value.length}/{max}</span><ChevronDown size={14} className={`transition-transform ${open ? "rotate-180" : ""}`} aria-hidden="true" /></span>
      </button>

      {open && (
        <div id={listboxId} role="listbox" aria-multiselectable="true" aria-label={label} className="absolute left-0 top-[calc(100%+.35rem)] z-50 max-h-80 w-full min-w-[280px] overflow-auto rounded-md border bg-white p-1.5 shadow-xl" style={{ borderColor: theme.colors.border }}>
          <p className="px-2.5 py-1.5 text-[8px] font-semibold" style={{ color: theme.colors.muted }}>Minimum {min} · maximum {max}</p>
          {options.map((option) => {
            const active = value.includes(option.value);
            const disabled = active ? value.length <= min : value.length >= max;
            return (
              <button key={option.value} type="button" role="option" aria-selected={active} aria-disabled={disabled} disabled={disabled} onClick={() => toggle(option.value)} className="flex w-full items-center justify-between gap-2 rounded px-2.5 py-2 text-left text-[10px] font-semibold transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40" style={{ color: theme.colors.primary, background: active ? theme.colors.soft : undefined }}>
                <CountryLabel iso3={option.iso3} name={option.label} size="md" />
                <span className="flex h-4 w-4 items-center justify-center rounded border" style={{ borderColor: active ? theme.colors.accent : theme.colors.border, background: active ? theme.colors.accent : "#FFFFFF", color: "#FFFFFF" }}>{active && <Check size={11} aria-hidden="true" />}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
