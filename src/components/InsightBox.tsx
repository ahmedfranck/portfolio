import { Lightbulb } from "lucide-react";
import type { ReactNode } from "react";

export default function InsightBox({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-3 rounded-card border border-brand-soft bg-brand-soft/60 p-4 text-sm text-ink">
      <Lightbulb size={18} className="mt-0.5 shrink-0 text-brand-deep" aria-hidden="true" />
      <div className="space-y-1">{children}</div>
    </div>
  );
}
