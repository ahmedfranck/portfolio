import { useId } from "react";
import { AlertTriangle, Database } from "lucide-react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

interface AoDataBadgesProps {
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly compactSource?: boolean;
  readonly sourceDisplay?: "inline" | "disclosure";
}

function compactSourceLabel(source: string) {
  const primary = source.split("·")[0]?.trim() || source;
  return `Source · ${primary}`;
}

export function AoDataBadges({ source, illustrative = false, compactSource = false, sourceDisplay = "inline" }: AoDataBadgesProps) {
  const { theme } = useAoTheme();
  const tooltipId = `ao-source-${useId().replaceAll(":", "")}`;

  if (!source && !illustrative) return null;

  return (
    <span className="inline-flex flex-wrap items-center gap-1.5">
      {source && sourceDisplay === "inline" && (
        <span
          className="inline-flex max-w-full items-center gap-1 rounded-full border px-2 py-1 text-[8px] font-semibold"
          style={{ borderColor: theme.colors.border, color: theme.colors.muted, background: theme.colors.canvas }}
          title={compactSource ? source : undefined}
        >
          <Database size={10} aria-hidden="true" />
          <span className={compactSource ? "max-w-44 truncate" : undefined}>{compactSource ? compactSourceLabel(source) : `Source : ${source}`}</span>
        </span>
      )}
      {source && sourceDisplay === "disclosure" && (
        <span className="group/source relative inline-flex">
          <button
            type="button"
            aria-describedby={tooltipId}
            className="inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[8px] font-semibold"
            style={{ borderColor: theme.colors.border, color: theme.colors.muted, background: theme.colors.canvas }}
          >
            <Database size={9} aria-hidden="true" /> Source
          </button>
          <span
            id={tooltipId}
            role="tooltip"
            className="invisible absolute bottom-[calc(100%+.4rem)] left-0 z-40 w-max max-w-64 rounded-md border bg-white px-2.5 py-2 text-left text-[9px] font-medium leading-relaxed opacity-0 shadow-lg transition group-hover/source:visible group-hover/source:opacity-100 group-focus-within/source:visible group-focus-within/source:opacity-100"
            style={{ borderColor: theme.colors.border, color: theme.colors.text }}
          >
            {source}
          </span>
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
