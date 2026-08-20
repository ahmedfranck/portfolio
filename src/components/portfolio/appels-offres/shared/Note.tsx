import { AlertTriangle, CircleAlert, Info, Lightbulb, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { useAoTheme } from "../../../../hooks/useAoTheme";

export type NoteVariant = "default" | "info" | "warning" | "crisis";

interface NoteProps {
  readonly title?: string;
  readonly children: ReactNode;
  readonly variant?: NoteVariant;
}

const NOTE_ICONS: Record<NoteVariant, LucideIcon> = {
  default: Lightbulb,
  info: Info,
  warning: AlertTriangle,
  crisis: CircleAlert,
};

export default function Note({ title, children, variant = "default" }: NoteProps) {
  const { theme } = useAoTheme();
  const color = variant === "crisis"
    ? theme.colors.negative
    : variant === "warning"
      ? theme.colors.warning
      : variant === "info"
        ? theme.colors.info
        : theme.colors.accent;
  const Icon = NOTE_ICONS[variant];

  return (
    <aside
      className="flex gap-3 rounded-md border-l-[3px] px-3 py-2.5 text-[10px] leading-relaxed"
      style={{ borderColor: color, background: `${color}10`, color: theme.colors.muted, fontFamily: theme.typography.body }}
    >
      <Icon size={15} className="mt-0.5 shrink-0" style={{ color }} aria-hidden="true" />
      <div>
        {title && <strong className="mb-0.5 block" style={{ color: theme.colors.primary }}>{title}</strong>}
        {children}
      </div>
    </aside>
  );
}
