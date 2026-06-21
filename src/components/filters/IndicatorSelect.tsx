import { useId } from "react";

interface IndicatorOption {
  value: string;
  label: string;
}

interface IndicatorSelectProps {
  value: string;
  onChange: (value: string) => void;
  options: readonly IndicatorOption[];
  label?: string;
}

export default function IndicatorSelect({
  value,
  onChange,
  options,
  label = "Indicateur",
}: IndicatorSelectProps) {
  const id = useId();

  return (
    <div className="min-w-[200px]">
      <label htmlFor={id} className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-2">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full cursor-pointer rounded-lg border border-line bg-white px-3 py-2 text-sm text-ink transition-colors duration-200 hover:border-brand focus:border-brand focus:outline-none"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}
