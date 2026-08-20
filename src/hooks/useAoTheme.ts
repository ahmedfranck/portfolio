import {
  createContext,
  createElement,
  useContext,
  useMemo,
  type CSSProperties,
  type ReactNode,
} from "react";
import { AO_THEMES, type AoTheme, type AoThemeKey } from "../themes/appelsOffres";

export interface AoThemeContextValue {
  readonly themeKey: AoThemeKey;
  readonly theme: AoTheme;
  readonly isUcpo: boolean;
}

interface AoThemeProviderProps {
  readonly themeKey: AoThemeKey;
  readonly children: ReactNode;
  readonly className?: string;
  readonly style?: CSSProperties;
}

type AoCssVariables = CSSProperties & Record<`--ao-${string}`, string> & {
  "--dash-deep": string;
  "--dash-accent": string;
  "--dash-soft": string;
  "--dash-canvas": string;
};

const AoThemeContext = createContext<AoThemeContextValue | null>(null);

function themeVariables(theme: AoTheme): AoCssVariables {
  return {
    "--ao-primary": theme.colors.primary,
    "--ao-primary-dark": theme.colors.primaryDark,
    "--ao-accent": theme.colors.accent,
    "--ao-accent-light": theme.colors.accentLight,
    "--ao-soft": theme.colors.soft,
    "--ao-canvas": theme.colors.canvas,
    "--ao-text": theme.colors.text,
    "--ao-muted": theme.colors.muted,
    "--ao-border": theme.colors.border,
    "--ao-positive": theme.colors.positive,
    "--ao-negative": theme.colors.negative,
    "--ao-warning": theme.colors.warning,
    "--ao-info": theme.colors.info,
    "--ao-heading-font": theme.typography.heading,
    "--ao-body-font": theme.typography.body,
    "--ao-series-1": theme.series[0],
    "--ao-series-2": theme.series[1],
    "--ao-series-3": theme.series[2],
    "--ao-series-4": theme.series[3],
    "--ao-series-5": theme.series[4],
    "--ao-series-6": theme.series[5],
    "--ao-series-7": theme.series[6],
    "--ao-series-8": theme.series[7],
    "--dash-deep": theme.colors.primaryDark,
    "--dash-accent": theme.colors.accent,
    "--dash-soft": theme.colors.soft,
    "--dash-canvas": theme.colors.canvas,
  };
}

export function AoThemeProvider({ themeKey, children, className, style }: AoThemeProviderProps) {
  const theme = AO_THEMES[themeKey];
  const value = useMemo<AoThemeContextValue>(
    () => ({ themeKey, theme, isUcpo: themeKey === "ucpo" }),
    [theme, themeKey]
  );
  const scopedStyle = { ...themeVariables(theme), ...style } as AoCssVariables;

  return createElement(
    AoThemeContext.Provider,
    { value },
    createElement(
      "div",
      {
        className,
        "data-ao-theme": themeKey,
        style: scopedStyle,
      },
      children
    )
  );
}

export function useAoTheme(): AoThemeContextValue {
  const context = useContext(AoThemeContext);
  if (!context) {
    throw new Error("useAoTheme doit être utilisé dans un AoThemeProvider.");
  }
  return context;
}
