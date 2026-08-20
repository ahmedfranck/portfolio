import type { ReactNode } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import Note, { type NoteVariant } from "./Note";

export interface RecommendationItem {
  readonly id: string;
  readonly title: string;
  readonly body: ReactNode;
  readonly variant?: NoteVariant;
}

export interface RecommendationGroup {
  readonly id: string;
  readonly title: ReactNode;
  readonly description?: string;
  readonly items: readonly RecommendationItem[];
}

interface RecommendationsHubProps {
  readonly title: ReactNode;
  readonly subtitle?: string;
  readonly groups: readonly RecommendationGroup[];
  readonly controls?: ReactNode;
}

export default function RecommendationsHub({ title, subtitle, groups, controls }: RecommendationsHubProps) {
  const { theme } = useAoTheme();

  return (
    <section className="space-y-5" aria-labelledby="ao-recommendations-title" style={{ fontFamily: theme.typography.body }}>
      <header className="rounded-[9px] border bg-white p-4" style={{ borderColor: theme.colors.border }}>
        <p className="text-[8px] font-bold uppercase tracking-[0.14em]" style={{ color: theme.colors.accent }}>Capitalisation</p>
        <h3 id="ao-recommendations-title" className="mt-1 text-lg" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>{title}</h3>
        {subtitle && <p className="mt-2 max-w-3xl text-[10px] leading-relaxed" style={{ color: theme.colors.muted }}>{subtitle}</p>}
        {controls && <div className="mt-4 border-t pt-4" style={{ borderColor: theme.colors.border }}>{controls}</div>}
      </header>

      {groups.map((group) => (
        <section key={group.id} className="rounded-[9px] border bg-white p-4" style={{ borderColor: theme.colors.border }} aria-labelledby={`ao-recommendation-group-${group.id}`}>
          <div className="mb-3">
            <h4 id={`ao-recommendation-group-${group.id}`} className="text-xs font-bold" style={{ color: theme.colors.primary }}>{group.title}</h4>
            {group.description && <p className="mt-1 text-[9px]" style={{ color: theme.colors.muted }}>{group.description}</p>}
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            {group.items.map((item) => <Note key={item.id} title={item.title} variant={item.variant}>{item.body}</Note>)}
          </div>
        </section>
      ))}
    </section>
  );
}
