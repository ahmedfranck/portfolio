interface YearRow {
  iso3: string;
  year: number;
}

export interface LatestObservation {
  value: number;
  year: number;
}

/**
 * Pour chaque pays, renvoie la dernière valeur réelle disponible pour `field`
 * à l'année `cutoffYear` ou avant. Ne fabrique jamais de valeur : renvoie
 * `null` si aucune observation n'existe dans cette fenêtre.
 */
export function latestAtOrBefore<T extends YearRow>(
  rows: T[],
  field: keyof T,
  cutoffYear: number
): Record<string, LatestObservation | null> {
  const byCountry: Record<string, LatestObservation | null> = {};
  const countries = Array.from(new Set(rows.map((r) => r.iso3)));

  for (const iso3 of countries) {
    const candidates = rows
      .filter((r) => r.iso3 === iso3 && r.year <= cutoffYear && r[field] != null)
      .sort((a, b) => b.year - a.year);
    const top = candidates[0];
    byCountry[iso3] = top ? { value: top[field] as number, year: top.year } : null;
  }

  return byCountry;
}

/** Les deux observations réelles les plus récentes pour un pays/indicateur, peu importe l'écart d'années. */
export function lastTwoObservations<T extends YearRow>(
  rows: T[],
  field: keyof T,
  iso3: string
): [LatestObservation | null, LatestObservation | null] {
  const sorted = rows
    .filter((r) => r.iso3 === iso3 && r[field] != null)
    .sort((a, b) => b.year - a.year);
  const a = sorted[0];
  const b = sorted[1];
  return [
    a ? { value: a[field] as number, year: a.year } : null,
    b ? { value: b[field] as number, year: b.year } : null,
  ];
}

/** Moyenne ignorant les valeurs manquantes. Renvoie `null` si aucune valeur définie. */
export function averageDefined(values: (number | null | undefined)[]): number | null {
  const defined = values.filter((v): v is number => v != null && Number.isFinite(v));
  if (defined.length === 0) return null;
  return defined.reduce((a, b) => a + b, 0) / defined.length;
}

/** Variation moyenne (%) entre les deux dernières observations réelles disponibles, par pays. */
export function averageYoYAcrossCountries<T extends YearRow>(
  rows: T[],
  field: keyof T,
  countries: string[]
): number | null {
  const changes: number[] = [];
  for (const iso3 of countries) {
    const [latest, previous] = lastTwoObservations(rows, field, iso3);
    if (latest && previous && previous.value !== 0) {
      changes.push(((latest.value - previous.value) / previous.value) * 100);
    }
  }
  return averageDefined(changes);
}

/** Série réelle (année, valeur) pour un pays/indicateur, jusqu'à `cutoffYear`, sans valeur fabriquée. */
export function seriesUpTo<T extends YearRow>(
  rows: T[],
  field: keyof T,
  iso3: string,
  cutoffYear: number
): { year: number; value: number }[] {
  return rows
    .filter((r) => r.iso3 === iso3 && r.year <= cutoffYear && r[field] != null)
    .sort((a, b) => a.year - b.year)
    .map((r) => ({ year: r.year, value: r[field] as number }));
}

export const NO_DATA_LABEL = "n.d.";
