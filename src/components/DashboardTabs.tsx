import type { ReactNode } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";

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
  const reduceMotion = useReducedMotion();
  const tabTransition = reduceMotion ? { duration: 0 } : { type: "spring" as const, stiffness: 380, damping: 32 };
  const panelTransition = reduceMotion ? { duration: 0 } : { duration: 0.22, ease: [0.22, 1, 0.36, 1] as const };

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
              className={`relative overflow-hidden rounded-full px-4 py-2 text-sm font-medium transition-all duration-200 ${
                isActive
                  ? "text-white"
                  : "cursor-pointer bg-surface text-text-2 hover:scale-[1.02] hover:bg-brand-soft hover:text-brand"
              }`}
            >
              {isActive && (
                <motion.span
                  layoutId="dashboardTabIndicator"
                  className="absolute inset-0 rounded-full bg-brand"
                  transition={tabTransition}
                />
              )}
              <span className="relative z-10">{tab.label}</span>
            </button>
          );
        })}
      </div>
      <div className="pt-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab.id}
            initial={{ opacity: 0, y: reduceMotion ? 0 : 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: reduceMotion ? 0 : -10 }}
            transition={panelTransition}
          >
            {activeTab.render()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
