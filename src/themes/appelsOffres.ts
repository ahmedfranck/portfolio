export type AoThemeKey = "ucpo" | "unicef" | "bad" | "pnue" | "unverified";

export interface AoThemeDefinition {
  readonly key: AoThemeKey;
  readonly name: string;
  readonly organizationDisplay: string;
  readonly rationale: string;
  readonly colors: {
    readonly primary: string;
    readonly primaryDark: string;
    readonly accent: string;
    readonly accentLight: string;
    readonly soft: string;
    readonly canvas: string;
    readonly text: string;
    readonly muted: string;
    readonly border: string;
    readonly positive: string;
    readonly negative: string;
    readonly warning: string;
    readonly info: string;
  };
  readonly typography: {
    readonly heading: string;
    readonly body: string;
  };
  readonly series: readonly [string, string, string, string, string, string, string, string];
  readonly footerNote?: string;
}

export const AO_THEMES = {
  ucpo: {
    key: "ucpo",
    name: "UCPO",
    organizationDisplay: "Identité visuelle UCPO confirmée",
    rationale: "Palette institutionnelle UCPO navy, or et crème réservée au projet Santé reproductive confirmé.",
    colors: {
      primary: "#22275A",
      primaryDark: "#181C45",
      accent: "#C3911F",
      accentLight: "#E0B24B",
      soft: "#FDF3DC",
      canvas: "#F7F5F0",
      text: "#333333",
      muted: "#666666",
      border: "#E5E0D6",
      positive: "#2E7D5E",
      negative: "#C0392B",
      warning: "#D4730A",
      info: "#1A4D6E",
    },
    typography: {
      heading: '"Chelsea Market", cursive',
      body: '"Montserrat", sans-serif',
    },
    series: ["#22275A", "#C3911F", "#2E7D5E", "#1A4D6E", "#B5541A", "#5B3A8A", "#1A6E6E", "#E0B24B"],
  },
  unicef: {
    key: "unicef",
    name: "UNICEF",
    organizationDisplay: "Palette inspirée de l’UNICEF",
    rationale: "Bleu cyan public de l’UNICEF, complété par un accent or et un bleu texte à fort contraste.",
    colors: {
      primary: "#1CABE2",
      primaryDark: "#002F5F",
      accent: "#F2A900",
      accentLight: "#FFD166",
      soft: "#EAF8FD",
      canvas: "#F6FBFD",
      text: "#002F5F",
      muted: "#526A7A",
      border: "#CDEAF5",
      positive: "#17845B",
      negative: "#C43D3D",
      warning: "#B86F00",
      info: "#167DA6",
    },
    typography: {
      heading: '"Inter", sans-serif',
      body: '"Inter", sans-serif',
    },
    series: ["#1CABE2", "#F2A900", "#17845B", "#167DA6", "#7B61A8", "#D65A4A", "#4C7A8A", "#FFD166"],
  },
  bad: {
    key: "bad",
    name: "BAD",
    organizationDisplay: "Palette inspirée de la Banque africaine de développement",
    rationale: "Vert institutionnel public de la BAD avec accent or, adapté aux dashboards économiques et d’infrastructure.",
    colors: {
      primary: "#006A4E",
      primaryDark: "#003F2E",
      accent: "#F2A900",
      accentLight: "#FFD166",
      soft: "#E8F4F0",
      canvas: "#F6FAF8",
      text: "#003F2E",
      muted: "#557068",
      border: "#C9E2D9",
      positive: "#2E7D5E",
      negative: "#BC3F3F",
      warning: "#B86F00",
      info: "#236B8E",
    },
    typography: {
      heading: '"Inter", sans-serif',
      body: '"Inter", sans-serif',
    },
    series: ["#006A4E", "#F2A900", "#2E7D5E", "#236B8E", "#A85632", "#6E5A8A", "#2C7771", "#FFD166"],
  },
  pnue: {
    key: "pnue",
    name: "PNUE",
    organizationDisplay: "Palette inspirée du Programme des Nations Unies pour l’environnement",
    rationale: "Turquoise institutionnel public du PNUE et accent ambre pour les lectures climat, risque et biodiversité.",
    colors: {
      primary: "#00A499",
      primaryDark: "#003D3B",
      accent: "#F5A623",
      accentLight: "#FFD27A",
      soft: "#E7F7F5",
      canvas: "#F5FAF9",
      text: "#003D3B",
      muted: "#55706E",
      border: "#C7E7E3",
      positive: "#287D5B",
      negative: "#C4473D",
      warning: "#C87800",
      info: "#197C93",
    },
    typography: {
      heading: '"Inter", sans-serif',
      body: '"Inter", sans-serif',
    },
    series: ["#00A499", "#F5A623", "#287D5B", "#197C93", "#B65738", "#67558A", "#3A7972", "#FFD27A"],
  },
  unverified: {
    key: "unverified",
    name: "Institutionnel neutre",
    organizationDisplay: "Commanditaire non confirmé",
    rationale: "Gris-bleu ardoise et turquoise sobres afin de ne revendiquer aucune identité de commanditaire non prouvée.",
    colors: {
      primary: "#2C3E50",
      primaryDark: "#1F2937",
      accent: "#16A085",
      accentLight: "#69C7B5",
      soft: "#E8F6F3",
      canvas: "#F6F8F9",
      text: "#1F2937",
      muted: "#667085",
      border: "#D6DDE3",
      positive: "#2E7D5E",
      negative: "#B54747",
      warning: "#B76E16",
      info: "#365F7A",
    },
    typography: {
      heading: '"Inter", sans-serif',
      body: '"Inter", sans-serif',
    },
    series: ["#2C3E50", "#16A085", "#365F7A", "#B76E16", "#6B5B8E", "#B54747", "#4B7A72", "#69C7B5"],
    footerNote: "Commanditaire à confirmer — thème institutionnel générique appliqué en attendant validation.",
  },
} as const satisfies Readonly<Record<AoThemeKey, AoThemeDefinition>>;

export type AoTheme = (typeof AO_THEMES)[AoThemeKey];
