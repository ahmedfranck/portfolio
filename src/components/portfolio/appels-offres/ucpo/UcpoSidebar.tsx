import { Database, Flag, LayoutDashboard, Lightbulb, Landmark, Menu, ShieldAlert, X, type LucideIcon } from "lucide-react";
import { Fragment, useState } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

export interface UcpoSection {
  readonly id: string;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
  readonly group: string;
  readonly badge?: string;
}

interface UcpoSidebarProps {
  readonly sections: readonly UcpoSection[];
  readonly active: string;
  readonly onChange: (section: string) => void;
}

export default function UcpoSidebar({ sections, active, onChange }: UcpoSidebarProps) {
  const { theme } = useAoTheme();
  const [open, setOpen] = useState(false);

  const icons: Readonly<Record<string, LucideIcon>> = {
    overview: LayoutDashboard,
    financing: Landmark,
    countries: Flag,
    crisis: ShieldAlert,
    recommendations: Lightbulb,
    sources: Database,
  };

  const navigation = (
    <nav aria-label="Sections de l’Observatoire PF" className="space-y-1.5">
      {sections.map((section, index) => {
        const selected = section.id === active;
        const Icon = icons[section.id] ?? LayoutDashboard;
        return (
          <Fragment key={section.id}>
            {section.group !== sections[index - 1]?.group && (
              <p className="px-3 pb-1 pt-4 text-[7px] font-bold uppercase tracking-[0.18em] first:pt-1" style={{ color: theme.colors.accentLight }}>{section.group}</p>
            )}
            <button
              type="button"
              aria-current={selected ? "page" : undefined}
              onClick={() => {
                onChange(section.id);
                setOpen(false);
              }}
              className="group w-full rounded-md border px-3 py-2.5 text-left transition"
              style={{
                borderColor: selected ? theme.colors.accent : "transparent",
                background: selected ? `${theme.colors.accent}16` : "transparent",
                color: selected ? theme.colors.accentLight : "rgba(255,255,255,.72)",
              }}
            >
              <span className="flex items-center gap-2.5">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md" style={{ background: selected ? `${theme.colors.accent}28` : "rgba(255,255,255,.07)", color: selected ? theme.colors.accentLight : "rgba(255,255,255,.55)" }}>
                  <Icon size={14} strokeWidth={1.8} aria-hidden="true" />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-[0.06em]">
                    <span className="text-[7px] opacity-40">{String(index + 1).padStart(2, "0")}</span>
                    <span className="truncate">{section.shortLabel}</span>
                    {section.badge && <span className="ml-auto rounded-full px-1.5 py-0.5 text-[7px]" style={{ color: theme.colors.primaryDark, background: theme.colors.accentLight }}>{section.badge}</span>}
                  </span>
                  <span className="mt-0.5 block text-[8px] leading-relaxed opacity-55">{section.description}</span>
                </span>
              </span>
            </button>
          </Fragment>
        );
      })}
    </nav>
  );

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="mb-3 inline-flex items-center gap-2 rounded-md px-3 py-2 text-[10px] font-bold text-white lg:hidden"
        style={{ background: theme.colors.primary }}
      >
        <Menu size={15} aria-hidden="true" /> Navigation UCPO
      </button>

      <aside
        className="hidden min-h-screen self-stretch rounded-[10px] shadow-sm lg:block"
        style={{ background: theme.colors.primaryDark, fontFamily: theme.typography.body }}
      >
        <div className="sticky top-0 max-h-screen overflow-auto p-3">
          <p className="px-3 pb-3 pt-1 text-[8px] font-bold uppercase tracking-[0.16em] text-white/35">Observatoire PF</p>
          {navigation}
        </div>
      </aside>

      {open && (
        <div className="fixed inset-0 z-[70] lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation UCPO">
          <button type="button" aria-label="Fermer la navigation" onClick={() => setOpen(false)} className="absolute inset-0 bg-slate-950/55" />
          <aside className="absolute inset-y-0 left-0 w-[min(86vw,320px)] overflow-auto p-4 shadow-2xl" style={{ background: theme.colors.primaryDark }}>
            <div className="mb-4 flex items-center justify-between">
              <strong className="text-xs text-white">Observatoire PF</strong>
              <button type="button" onClick={() => setOpen(false)} aria-label="Fermer" className="rounded-full p-2 text-white/70 hover:bg-white/10">
                <X size={17} aria-hidden="true" />
              </button>
            </div>
            {navigation}
          </aside>
        </div>
      )}
    </>
  );
}
