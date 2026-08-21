import { useId } from "react";
import { MIN_YEAR, MAX_YEAR } from "../../data/countries";

interface YearSliderProps {
  year: number;
  onChange: (year: number) => void;
  min?: number;
  max?: number;
  label?: string;
}

export default function YearSlider({
  year,
  onChange,
  min = MIN_YEAR,
  max = MAX_YEAR,
  label = "Année",
}: YearSliderProps) {
  const id = useId();

  return (
    <div className="min-w-[180px]">
      <label htmlFor={id} className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-2">
        {label} : <span className="font-display font-semibold text-ink">{year}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={1}
        value={year}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-surface-2 accent-brand"
      />
      <div className="mt-1 flex justify-between text-[10px] text-text-2">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
