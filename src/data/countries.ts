export interface Country {
  iso3: string;
  name: string;
  region: "Afrique de l'Ouest" | "Afrique Centrale";
}

export const COUNTRIES: Country[] = [
  { iso3: "BEN", name: "Bénin", region: "Afrique de l'Ouest" },
  { iso3: "BFA", name: "Burkina Faso", region: "Afrique de l'Ouest" },
  { iso3: "CMR", name: "Cameroun", region: "Afrique Centrale" },
  { iso3: "CIV", name: "Côte d'Ivoire", region: "Afrique de l'Ouest" },
  { iso3: "GHA", name: "Ghana", region: "Afrique de l'Ouest" },
  { iso3: "GIN", name: "Guinée", region: "Afrique de l'Ouest" },
  { iso3: "LBR", name: "Libéria", region: "Afrique de l'Ouest" },
  { iso3: "MLI", name: "Mali", region: "Afrique de l'Ouest" },
  { iso3: "MRT", name: "Mauritanie", region: "Afrique de l'Ouest" },
  { iso3: "NER", name: "Niger", region: "Afrique de l'Ouest" },
  { iso3: "NGA", name: "Nigéria", region: "Afrique de l'Ouest" },
  { iso3: "COD", name: "RDC", region: "Afrique Centrale" },
  { iso3: "SEN", name: "Sénégal", region: "Afrique de l'Ouest" },
  { iso3: "SLE", name: "Sierra Leone", region: "Afrique de l'Ouest" },
  { iso3: "TCD", name: "Tchad", region: "Afrique Centrale" },
  { iso3: "TGO", name: "Togo", region: "Afrique de l'Ouest" },
];

export const COUNTRY_NAMES: Record<string, string> = Object.fromEntries(
  COUNTRIES.map((c) => [c.iso3, c.name])
);

export const YEARS: number[] = Array.from({ length: 2024 - 2010 + 1 }, (_, i) => 2010 + i);

export const MIN_YEAR = 2010;
export const MAX_YEAR = 2024;
