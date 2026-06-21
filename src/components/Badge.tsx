import type { ReactNode } from "react";

interface BadgeProps {
  children: ReactNode;
  tone?: "brand" | "ink" | "amber" | "neutral";
}

const TONE_CLASSES: Record<string, string> = {
  brand: "bg-brand-soft text-brand-deep border-brand-soft",
  ink: "bg-surface-2 text-ink border-line",
  amber: "bg-amber-50 text-amber-700 border-amber-200",
  neutral: "bg-surface text-text-2 border-line",
};

export default function Badge({ children, tone = "neutral" }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}
