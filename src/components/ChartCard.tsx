import type { ReactNode } from "react";

interface ChartCardProps {
  title: string;
  description?: string;
  children: ReactNode;
  className?: string;
}

export default function ChartCard({ title, description, children, className = "" }: ChartCardProps) {
  return (
    <div className={`rounded-card border border-line bg-white p-5 shadow-card transition-all duration-200 hover:-translate-y-[3px] hover:shadow-cardHover sm:p-6 ${className}`}>
      <div className="mb-4">
        <h3 className="font-display text-base font-semibold text-ink">{title}</h3>
        {description && <p className="mt-1 text-sm text-text-2">{description}</p>}
      </div>
      <div className="w-full">{children}</div>
      <p className="mt-3 font-mono text-xs text-text-2">Source : Banque mondiale</p>
    </div>
  );
}
