import { AlertTriangle, Database } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

interface AoDataBadgesProps {
  readonly source?: string;
  readonly illustrative?: boolean;
}

export function AoDataBadges({ source, illustrative = false }: AoDataBadgesProps) {
  const { theme } = useAoTheme();

  if (!source && !illustrative) return null;

  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      {source && (
        <span
          className="inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[9px] font-semibold"
          style={{ borderColor: theme.colors.border, color: theme.colors.muted, background: theme.colors.canvas }}
        >
          <Database size={10} aria-hidden="true" />
          Source : {source}
        </span>
      )}
      {illustrative && (
        <span
          className="inline-flex items-center gap-1 rounded-full px-2 py-1 text-[9px] font-bold"
          style={{ color: theme.colors.warning, background: "#FEF0E0" }}
        >
          <AlertTriangle size={10} aria-hidden="true" />
          Données illustratives
        </span>
      )}
    </span>
  );
}
