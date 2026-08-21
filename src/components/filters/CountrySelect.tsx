import { useId } from "react";
import { COUNTRIES } from "../../data/countries";

interface CountrySelectProps {
  value: string;
  onChange: (iso3: string) => void;
  countries?: string[];
  label?: string;
}

export default function CountrySelect({ value, onChange, countries, label = "Pays" }: CountrySelectProps) {
  const id = useId();
  const options = countries ? COUNTRIES.filter((c) => countries.includes(c.iso3)) : COUNTRIES;

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
        {options.map((c) => (
          <option key={c.iso3} value={c.iso3}>
            {c.name}
          </option>
        ))}
      </select>
    </div>
  );
}
