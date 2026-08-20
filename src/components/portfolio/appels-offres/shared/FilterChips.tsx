import type { ReactNode } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

export interface FilterChipOption {
  readonly value: string;
  readonly label: ReactNode;
  readonly count?: number;
}

interface FilterChipsProps {
  readonly label: ReactNode;
  readonly options: readonly FilterChipOption[];
  readonly value: readonly string[];
  readonly onChange: (value: string[]) => void;
  readonly multiple?: boolean;
}

export default function FilterChips({ label, options, value, onChange, multiple = true }: FilterChipsProps) {
  const { theme } = useAoTheme();

  function toggle(optionValue: string) {
    if (!multiple) {
      onChange([optionValue]);
      return;
    }
    onChange(value.includes(optionValue) ? value.filter((item) => item !== optionValue) : [...value, optionValue]);
  }

  return (
    <fieldset className="flex flex-wrap items-center gap-2" style={{ fontFamily: theme.typography.body }}>
      <legend className="mr-1 text-[9px] font-bold uppercase tracking-[0.12em]" style={{ color: theme.colors.muted }}>{label}</legend>
      {options.map((option) => {
        const active = value.includes(option.value);
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => toggle(option.value)}
            className="rounded-full border px-2.5 py-1 text-[9px] font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
            style={{
              borderColor: active ? theme.colors.primary : theme.colors.border,
              background: active ? theme.colors.primary : "#FFFFFF",
              color: active ? "#FFFFFF" : theme.colors.primary,
              outlineColor: theme.colors.accent,
            }}
          >
            {option.label}{option.count != null ? ` · ${option.count}` : ""}
          </button>
        );
      })}
    </fieldset>
  );
}
