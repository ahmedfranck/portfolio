import type { ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";

export interface DashboardTab {
  id: string;
  label: string;
  render: () => ReactNode;
}

interface DashboardTabsProps {
  tabs: DashboardTab[];
  active: string;
  onChange: (id: string) => void;
}

/**
 * Shell de dashboard réutilisable : la sélection d'onglet est contrôlée par le parent
 * (qui porte l'état partagé pays/année/indicateur), seule la transition est gérée ici.
 */
export default function DashboardTabs({ tabs, active, onChange }: DashboardTabsProps) {
  const activeTab = tabs.find((t) => t.id === active) ?? tabs[0];

  return (
    <div>
      <div
        role="tablist"
        aria-label="Onglets du tableau de bord"
        className="flex flex-wrap gap-2 border-b border-line pb-4"
      >
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              onClick={() => onChange(tab.id)}
              className={`rounded-full px-4 py-2 text-sm font-medium transition-colors duration-200 ${
                isActive ? "bg-brand text-white" : "bg-surface text-text-2 hover:bg-surface-2 hover:text-ink"
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <div className="pt-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
          >
            {activeTab.render()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
