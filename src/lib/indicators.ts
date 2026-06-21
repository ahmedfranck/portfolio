/** Normalise une valeur entre 0 et 100 selon un min/max donné. */
export function normalize(value: number, min: number, max: number): number {
  if (max === min) return 50;
  return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}
