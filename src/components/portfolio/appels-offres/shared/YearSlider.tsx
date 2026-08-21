import { useId, type CSSProperties } from "react";
import { CalendarRange } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

interface YearSliderProps {
  readonly value: number;
  readonly onChange: (value: number) => void;
  readonly min: number;
  readonly max: number;
  readonly label?: string;
}

export default function YearSlider({ value, onChange, min, max, label = "Année maximale affichée" }: YearSliderProps) {
  const { theme } = useAoTheme();
  const id = useId();
  const progress = max === min ? 100 : Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  const style = {
    "--ao-slider-accent": theme.colors.accent,
    "--ao-slider-border": theme.colors.primaryDark,
    background: `linear-gradient(90deg, ${theme.colors.accent} 0%, ${theme.colors.accent} ${progress}%, #D9DDE2 ${progress}%, #D9DDE2 100%)`,
  } as CSSProperties;

  return (
    <div className="min-w-[260px]" style={{ fontFamily: theme.typography.body }}>
      <label htmlFor={id} className="flex items-center justify-between gap-3 text-[8px] font-bold uppercase tracking-[0.1em]" style={{ color: theme.colors.muted }}>
        <span className="inline-flex items-center gap-1.5"><CalendarRange size={12} style={{ color: theme.colors.accent }} aria-hidden="true" />{label}</span>
        <strong className="rounded-full px-2 py-1 text-[10px]" style={{ color: theme.colors.primary, background: theme.colors.soft }}>{value}</strong>
      </label>
      <input id={id} type="range" min={min} max={max} step={1} value={value} onChange={(event) => onChange(Number(event.target.value))} className="ao-year-slider mt-2 h-2 w-full cursor-pointer appearance-none rounded-full" style={style} aria-valuetext={`Données affichées jusqu’en ${value}`} />
      <div className="mt-1.5 flex justify-between text-[8px] font-semibold" style={{ color: theme.colors.muted }}>
        <span>{min} · inclus</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
