import { Menu, X } from "lucide-react";
import { useState } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

export interface UcpoSection {
  readonly id: string;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
}

interface UcpoSidebarProps {
  readonly sections: readonly UcpoSection[];
  readonly active: string;
  readonly onChange: (section: string) => void;
}

export default function UcpoSidebar({ sections, active, onChange }: UcpoSidebarProps) {
  const { theme } = useAoTheme();
  const [open, setOpen] = useState(false);

  const navigation = (
    <nav aria-label="Sections de l’Observatoire PF" className="space-y-1.5">
      {sections.map((section, index) => {
        const selected = section.id === active;
        return (
          <button
            key={section.id}
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
            <span className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.08em]">
              <span className="text-[8px] opacity-45">{String(index + 1).padStart(2, "0")}</span>
              {section.shortLabel}
            </span>
            <span className="mt-1 block text-[8px] leading-relaxed opacity-55">{section.description}</span>
          </button>
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
        className="sticky top-0 hidden h-screen min-h-screen overflow-auto rounded-[10px] p-3 shadow-sm lg:block"
        style={{ background: theme.colors.primaryDark, fontFamily: theme.typography.body }}
      >
        <p className="px-3 pb-3 pt-1 text-[8px] font-bold uppercase tracking-[0.16em] text-white/35">Observatoire PF</p>
        {navigation}
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
