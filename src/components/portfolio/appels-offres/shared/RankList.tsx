import { useMemo } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { CountryLabel } from "./CountryFlag";

export interface RankListItem {
  readonly id: string;
  readonly label: string;
  readonly value: number;
  readonly displayValue?: string;
  readonly detail?: string;
  readonly iso3?: string;
}

interface RankListProps {
  readonly items: readonly RankListItem[];
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly descending?: boolean;
}

export default function RankList({ items, source, illustrative = false, descending = true }: RankListProps) {
  const { theme } = useAoTheme();
  const sorted = useMemo(
    () => [...items].sort((a, b) => (descending ? b.value - a.value : a.value - b.value)),
    [descending, items]
  );
  const max = Math.max(...sorted.map((item) => Math.abs(item.value)), 1);

  return (
    <div style={{ fontFamily: theme.typography.body }}>
      <ol className="space-y-3">
        {sorted.map((item, index) => (
          <li key={item.id} className="grid grid-cols-[1.5rem_minmax(0,1fr)_auto] items-center gap-2">
            <span className="text-[10px] font-bold" style={{ color: theme.colors.accent }}>{String(index + 1).padStart(2, "0")}</span>
            <div className="min-w-0">
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate text-[10px] font-semibold" style={{ color: theme.colors.primary }}>{item.iso3 ? <CountryLabel iso3={item.iso3} name={item.label} /> : item.label}</span>
                {item.detail && <span className="truncate text-[8px]" style={{ color: theme.colors.muted }}>{item.detail}</span>}
              </div>
              <div className="mt-1 h-1 overflow-hidden rounded-full" style={{ background: theme.colors.soft }}>
                <span className="block h-full rounded-full" style={{ width: `${Math.max(2, Math.abs(item.value) / max * 100)}%`, background: theme.colors.accent }} />
              </div>
            </div>
            <strong className="text-[10px]" style={{ color: theme.colors.text }}>{item.displayValue ?? item.value.toLocaleString("fr-FR")}</strong>
          </li>
        ))}
      </ol>
    </div>
  );
}
