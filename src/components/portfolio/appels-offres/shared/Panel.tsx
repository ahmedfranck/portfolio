import { useId, type ReactNode } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { AoDataBadges } from "./badges";

interface PanelProps {
  readonly title: string;
  readonly subtitle?: string;
  readonly action?: ReactNode;
  readonly children: ReactNode;
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly accent?: string;
  readonly className?: string;
}

export default function Panel({ title, subtitle, action, children, source, illustrative = false, accent, className = "" }: PanelProps) {
  const { theme } = useAoTheme();
  const titleId = useId();

  return (
    <section
      className={`overflow-hidden rounded-[9px] border bg-white shadow-sm ${className}`}
      style={{ borderColor: theme.colors.border, fontFamily: theme.typography.body }}
      aria-labelledby={titleId}
    >
      <header className="flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3" style={{ borderColor: theme.colors.border }}>
        <div className="min-w-0">
          <h3 id={titleId} className="flex items-center gap-2 text-xs" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>
            <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: accent ?? theme.colors.accent }} aria-hidden="true" />
            {title}
          </h3>
          {subtitle && <p className="mt-1 text-[10px] leading-relaxed" style={{ color: theme.colors.muted }}>{subtitle}</p>}
        </div>
        {action}
      </header>
      <div className="p-4">{children}</div>
      {(source || illustrative) && (
        <footer className="border-t px-4 py-3" style={{ borderColor: theme.colors.border, background: theme.colors.canvas }}>
          <AoDataBadges source={source} illustrative={illustrative} />
        </footer>
      )}
    </section>
  );
}
