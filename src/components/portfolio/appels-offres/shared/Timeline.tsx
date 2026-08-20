import { useAoTheme } from "../../../../hooks/useAoTheme";

export interface TimelineEntry {
  readonly id: string;
  readonly date: string;
  readonly title: string;
  readonly description: string;
  readonly impact?: string;
  readonly tone?: "default" | "positive" | "warning" | "crisis";
  readonly source?: string;
  readonly illustrative?: boolean;
}

interface TimelineProps {
  readonly entries: readonly TimelineEntry[];
  readonly ariaLabel?: string;
}

export default function Timeline({ entries, ariaLabel = "Événements structurants" }: TimelineProps) {
  const { theme } = useAoTheme();

  function toneColor(tone: TimelineEntry["tone"]) {
    if (tone === "positive") return theme.colors.positive;
    if (tone === "warning") return theme.colors.warning;
    if (tone === "crisis") return theme.colors.negative;
    return theme.colors.accent;
  }

  return (
    <ol className="relative space-y-0" aria-label={ariaLabel} style={{ fontFamily: theme.typography.body }}>
      {entries.map((entry, index) => {
        const color = toneColor(entry.tone);
        return (
          <li key={entry.id} className="relative grid grid-cols-[5rem_1rem_minmax(0,1fr)] gap-2 pb-5 last:pb-0">
            <time className="pt-0.5 text-right text-[9px] font-bold" style={{ color: theme.colors.muted }}>{entry.date}</time>
            <span className="relative flex justify-center">
              {index < entries.length - 1 && <span className="absolute bottom-[-1.25rem] top-3 w-px" style={{ background: theme.colors.border }} aria-hidden="true" />}
              <span className="relative z-10 mt-1 h-2.5 w-2.5 rounded-full border-2 border-white" style={{ background: color, boxShadow: `0 0 0 1px ${color}` }} aria-hidden="true" />
            </span>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h4 className="text-[11px] font-bold" style={{ color: theme.colors.primary }}>{entry.title}</h4>
                {entry.impact && <span className="rounded-full px-2 py-0.5 text-[8px] font-bold" style={{ color, background: `${color}12` }}>{entry.impact}</span>}
              </div>
              <p className="mt-1 text-[10px] leading-relaxed" style={{ color: theme.colors.muted }}>{entry.description}</p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
