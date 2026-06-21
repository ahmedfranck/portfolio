import { scaleSequential } from "d3-scale";

export const PALETTE = {
  ink: "#14161B",
  brand: "#5B4BE3",
  brandDeep: "#3F32B5",
  surface: "#F5F6F8",
  surface2: "#EDEFF3",
  line: "#E4E7EC",
  text2: "#586072",
  text3: "#8A92A3",
};

// Palette catégorielle (séries de graphiques, badges) — §2.1 du brief.
export const CATEGORICAL: string[] = [
  "#5B4BE3", // c1
  "#0EA5A4", // c2
  "#F59E0B", // c3
  "#E0457B", // c4
  "#3B82F6", // c5
  "#10B981", // c6
  "#64748B", // c7
];

export function colorForIndex(index: number): string {
  return CATEGORICAL[index % CATEGORICAL.length];
}

export function colorForCountry(iso3: string, allCodes: string[]): string {
  const idx = allCodes.indexOf(iso3);
  return colorForIndex(idx === -1 ? 0 : idx);
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function lerpColor(a: string, b: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const bl = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}

function multiLerp(stops: string[], t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const seg = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  return lerpColor(stops[i], stops[i + 1], seg - i);
}

export type IndicatorPolarity = "positive" | "negative";

// Indicateur "positif" (mCPR, accès à l'eau...) → rampe claire→indigo.
const POSITIVE_RAMP = ["#EEEBFB", "#0EA5A4", "#3F32B5"];
// Indicateur "négatif" (mortalité, déplacement...) → rampe claire→ambre→brun.
const NEGATIVE_RAMP = ["#FEF3E2", "#F59E0B", "#7C4A03"];

/** Couleur séquentielle à teinte unique pour t ∈ [0,1], jamais rouge/vert seuls. */
export function sequentialColor(t: number, polarity: IndicatorPolarity = "positive"): string {
  return multiLerp(polarity === "negative" ? NEGATIVE_RAMP : POSITIVE_RAMP, t);
}

export const MISSING_DATA_COLOR = "#D7DBE3";

/** Échelle séquentielle d3 (domaine réel -> couleur), choisie selon la polarité de l'indicateur. */
export function makeSequentialScale(
  min: number,
  max: number,
  polarity: IndicatorPolarity = "positive"
): (value: number) => string {
  const scale = scaleSequential<string>((t) => sequentialColor(t, polarity)).domain([min, max]);
  return (value: number) => scale(value);
}

export function rampStops(polarity: IndicatorPolarity = "positive"): string[] {
  return polarity === "negative" ? NEGATIVE_RAMP : POSITIVE_RAMP;
}
