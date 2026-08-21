export const UCPO_COUNTRY_CODES = ["BEN", "BFA", "CIV", "GIN", "MLI", "MRT", "NER", "SEN", "TGO"] as const;

export type UcpoCountryCode = (typeof UCPO_COUNTRY_CODES)[number];
export type UcpoCountryTab = "overview" | "financing" | "demography" | "methods" | "impact" | "crisis";

export interface UcpoDatasetMeta {
  readonly id: string;
  readonly label: string;
  readonly source: string;
  readonly url: string;
  readonly illustrative: boolean;
  readonly note: string;
}

export interface UcpoCountryProfile {
  readonly iso3: UcpoCountryCode;
  readonly name: string;
  readonly shortName: string;
  readonly baselineMcpr: number;
  readonly currentMcpr: number;
  readonly tfr: number;
  readonly women15to49Millions: number;
  readonly financingUsdMillions: number;
  readonly domesticShare: number;
  readonly usaidExposure: number;
  readonly modernUsersMillions: number;
  readonly pregnanciesAvoidedThousands: number;
  readonly deathsAvoided: number;
  readonly costPerUserUsd: number;
  readonly informRisk: number;
  readonly displacedThousands: number;
  readonly stockoutRate: number;
  readonly methods: readonly { readonly name: string; readonly value: number }[];
}

const OFFICIAL_SOURCES = {
  wdi: "https://api.worldbank.org/v2/",
  fp2030: "https://www.fp2030.org/data-hub/",
  track20: "https://www.track20.org/",
  unfpa: "https://www.unfpa.org/data/world-population-dashboard",
  inform: "https://drmkc.jrc.ec.europa.eu/inform-index/",
  idmc: "https://www.internal-displacement.org/database/displacement-data/",
  acled: "https://acleddata.com/data-export-tool/",
} as const;

export const UCPO_DATASETS = {
  wdiCore: {
    id: "wdi-core",
    label: "Indicateurs démographiques et contraceptifs",
    source: "Banque mondiale · World Development Indicators",
    url: OFFICIAL_SOURCES.wdi,
    illustrative: false,
    note: "Valeurs publiques déjà intégrées au portfolio; dernière observation disponible affichée par pays.",
  },
  trajectory: {
    id: "mcpr-trajectories",
    label: "Trajectoires mCPR 2011–2024",
    source: "Scénario UCPO documenté · cadre méthodologique Track20 / FP2030",
    url: OFFICIAL_SOURCES.track20,
    illustrative: true,
    note: "Trajectoires annuelles interpolées à des fins de démonstration; elles ne remplacent pas les estimations FPET validées.",
  },
  population: {
    id: "population-planning",
    label: "Population des femmes de 15 à 49 ans",
    source: "Scénario UCPO · cadre UNFPA World Population Dashboard",
    url: OFFICIAL_SOURCES.unfpa,
    illustrative: true,
    note: "Dénominateurs de planification arrondis à des fins de démonstration, à remplacer par les séries UNFPA validées.",
  },
  financing: {
    id: "financing",
    label: "Financement de la planification familiale",
    source: "Scénario UCPO · ventilation programme et exposition USAID",
    url: OFFICIAL_SOURCES.fp2030,
    illustrative: true,
    note: "Montants et parts de financement plausibles, à remplacer par les comptes nationaux et données bailleurs validés.",
  },
  methods: {
    id: "method-mix",
    label: "Mix de méthodes contraceptives",
    source: "Scénario UCPO · structure inspirée DHS / MICS / Track20",
    url: OFFICIAL_SOURCES.track20,
    illustrative: true,
    note: "Répartition démonstrative normalisée à 100 % pour chaque pays.",
  },
  impact: {
    id: "impact",
    label: "Impact de l'utilisation de méthodes modernes",
    source: "Scénario UCPO · logique d'impact FP2030",
    url: OFFICIAL_SOURCES.fp2030,
    illustrative: true,
    note: "Grossesses et décès évités ainsi que coût par utilisatrice sont des estimations de démonstration.",
  },
  crisis: {
    id: "crisis",
    label: "Contexte de crise, déplacements et ruptures",
    source: "Scénario UCPO · cadres INFORM, IDMC et ACLED",
    url: OFFICIAL_SOURCES.inform,
    illustrative: true,
    note: "Agrégats démonstratifs; ACLED requiert un accès enregistré et les valeurs doivent être revalidées avant livraison.",
  },
  timeline: {
    id: "timeline",
    label: "Chronologie programmatique 2011–2024",
    source: "Scénario éditorial UCPO",
    url: OFFICIAL_SOURCES.fp2030,
    illustrative: true,
    note: "Jalons de démonstration à remplacer par la chronologie validée de l'UCPO.",
  },
} as const satisfies Readonly<Record<string, UcpoDatasetMeta>>;

export const UCPO_COUNTRIES: readonly UcpoCountryProfile[] = [
  { iso3: "BEN", name: "Bénin", shortName: "Bénin", baselineMcpr: 8.0, currentMcpr: 19.8, tfr: 4.84, women15to49Millions: 3.3, financingUsdMillions: 16.8, domesticShare: 21, usaidExposure: 32, modernUsersMillions: 0.52, pregnanciesAvoidedThousands: 185, deathsAvoided: 520, costPerUserUsd: 9.8, informRisk: 4.6, displacedThousands: 15, stockoutRate: 14, methods: [{ name: "Injectables", value: 29 }, { name: "Implants", value: 25 }, { name: "Pilules", value: 18 }, { name: "DIU", value: 7 }, { name: "Préservatifs", value: 13 }, { name: "Autres", value: 8 }] },
  { iso3: "BFA", name: "Burkina Faso", shortName: "Burkina", baselineMcpr: 15.0, currentMcpr: 32.1, tfr: 4.19, women15to49Millions: 5.8, financingUsdMillions: 27.4, domesticShare: 18, usaidExposure: 41, modernUsersMillions: 1.45, pregnanciesAvoidedThousands: 480, deathsAvoided: 1320, costPerUserUsd: 10.6, informRisk: 8.1, displacedThousands: 2060, stockoutRate: 29, methods: [{ name: "Injectables", value: 24 }, { name: "Implants", value: 38 }, { name: "Pilules", value: 13 }, { name: "DIU", value: 5 }, { name: "Préservatifs", value: 11 }, { name: "Autres", value: 9 }] },
  { iso3: "CIV", name: "Côte d’Ivoire", shortName: "Côte d’Ivoire", baselineMcpr: 13.4, currentMcpr: 27.0, tfr: 4.26, women15to49Millions: 7.9, financingUsdMillions: 34.2, domesticShare: 27, usaidExposure: 35, modernUsersMillions: 1.72, pregnanciesAvoidedThousands: 560, deathsAvoided: 1260, costPerUserUsd: 12.2, informRisk: 5.0, displacedThousands: 34, stockoutRate: 12, methods: [{ name: "Injectables", value: 27 }, { name: "Implants", value: 31 }, { name: "Pilules", value: 16 }, { name: "DIU", value: 7 }, { name: "Préservatifs", value: 12 }, { name: "Autres", value: 7 }] },
  { iso3: "GIN", name: "Guinée", shortName: "Guinée", baselineMcpr: 6.2, currentMcpr: 14.7, tfr: 4.35, women15to49Millions: 3.5, financingUsdMillions: 14.1, domesticShare: 14, usaidExposure: 28, modernUsersMillions: 0.40, pregnanciesAvoidedThousands: 142, deathsAvoided: 510, costPerUserUsd: 13.6, informRisk: 6.0, displacedThousands: 9, stockoutRate: 24, methods: [{ name: "Injectables", value: 31 }, { name: "Implants", value: 28 }, { name: "Pilules", value: 14 }, { name: "DIU", value: 4 }, { name: "Préservatifs", value: 14 }, { name: "Autres", value: 9 }] },
  { iso3: "MLI", name: "Mali", shortName: "Mali", baselineMcpr: 9.9, currentMcpr: 20.3, tfr: 5.67, women15to49Millions: 5.7, financingUsdMillions: 23.7, domesticShare: 16, usaidExposure: 44, modernUsersMillions: 0.91, pregnanciesAvoidedThousands: 318, deathsAvoided: 1010, costPerUserUsd: 11.9, informRisk: 7.9, displacedThousands: 390, stockoutRate: 31, methods: [{ name: "Injectables", value: 26 }, { name: "Implants", value: 36 }, { name: "Pilules", value: 12 }, { name: "DIU", value: 4 }, { name: "Préservatifs", value: 12 }, { name: "Autres", value: 10 }] },
  { iso3: "MRT", name: "Mauritanie", shortName: "Mauritanie", baselineMcpr: 10.0, currentMcpr: 16.2, tfr: 4.42, women15to49Millions: 1.3, financingUsdMillions: 8.6, domesticShare: 24, usaidExposure: 18, modernUsersMillions: 0.16, pregnanciesAvoidedThousands: 58, deathsAvoided: 180, costPerUserUsd: 15.4, informRisk: 5.5, displacedThousands: 8, stockoutRate: 19, methods: [{ name: "Injectables", value: 22 }, { name: "Implants", value: 20 }, { name: "Pilules", value: 24 }, { name: "DIU", value: 8 }, { name: "Préservatifs", value: 16 }, { name: "Autres", value: 10 }] },
  { iso3: "NER", name: "Niger", shortName: "Niger", baselineMcpr: 8.4, currentMcpr: 15.6, tfr: 6.64, women15to49Millions: 6.8, financingUsdMillions: 21.9, domesticShare: 12, usaidExposure: 39, modernUsersMillions: 0.82, pregnanciesAvoidedThousands: 292, deathsAvoided: 980, costPerUserUsd: 12.8, informRisk: 7.6, displacedThousands: 335, stockoutRate: 27, methods: [{ name: "Injectables", value: 34 }, { name: "Implants", value: 29 }, { name: "Pilules", value: 12 }, { name: "DIU", value: 3 }, { name: "Préservatifs", value: 11 }, { name: "Autres", value: 11 }] },
  { iso3: "SEN", name: "Sénégal", shortName: "Sénégal", baselineMcpr: 12.1, currentMcpr: 30.5, tfr: 4.39, women15to49Millions: 4.6, financingUsdMillions: 29.8, domesticShare: 31, usaidExposure: 33, modernUsersMillions: 1.10, pregnanciesAvoidedThousands: 375, deathsAvoided: 870, costPerUserUsd: 10.1, informRisk: 4.2, displacedThousands: 12, stockoutRate: 9, methods: [{ name: "Injectables", value: 32 }, { name: "Implants", value: 29 }, { name: "Pilules", value: 14 }, { name: "DIU", value: 8 }, { name: "Préservatifs", value: 10 }, { name: "Autres", value: 7 }] },
  { iso3: "TGO", name: "Togo", shortName: "Togo", baselineMcpr: 13.2, currentMcpr: 24.4, tfr: 4.10, women15to49Millions: 2.2, financingUsdMillions: 11.7, domesticShare: 22, usaidExposure: 29, modernUsersMillions: 0.43, pregnanciesAvoidedThousands: 151, deathsAvoided: 390, costPerUserUsd: 10.9, informRisk: 4.8, displacedThousands: 18, stockoutRate: 16, methods: [{ name: "Injectables", value: 25 }, { name: "Implants", value: 27 }, { name: "Pilules", value: 19 }, { name: "DIU", value: 6 }, { name: "Préservatifs", value: 15 }, { name: "Autres", value: 8 }] },
] as const;

export const UCPO_MCPR_SERIES = Array.from({ length: 14 }, (_, index) => {
  const year = 2011 + index;
  const progress = index / 13;
  return Object.fromEntries([
    ["year", year],
    ...UCPO_COUNTRIES.map((country, countryIndex) => {
      const curve = Math.sin((index + countryIndex) * 0.7) * 0.25;
      const value = country.baselineMcpr + (country.currentMcpr - country.baselineMcpr) * progress + curve;
      return [country.iso3, Number(value.toFixed(1))];
    }),
  ]);
});

export const UCPO_TIMELINE = [
  { id: "2011", date: "2011", title: "Point de départ régional", description: "Installation d'un suivi régional harmonisé des engagements de planification familiale.", impact: "Référence", tone: "default" as const, source: UCPO_DATASETS.timeline.source, illustrative: true },
  { id: "2016", date: "2016", title: "Accélération des plans nationaux", description: "Priorisation démonstrative des lignes budgétaires, de la demande communautaire et de la disponibilité des produits.", impact: "Accélération", tone: "positive" as const, source: UCPO_DATASETS.timeline.source, illustrative: true },
  { id: "2020", date: "2020", title: "Continuité des services", description: "Adaptation programmatique face aux perturbations sanitaires et logistiques régionales.", impact: "Résilience", tone: "warning" as const, source: UCPO_DATASETS.timeline.source, illustrative: true },
  { id: "2024", date: "2024", title: "Lecture intégrée performance–crise", description: "Croisement des trajectoires contraceptives, du financement, des déplacements et des ruptures de stock.", impact: "Pilotage", tone: "crisis" as const, source: UCPO_DATASETS.timeline.source, illustrative: true },
] as const;

export const UCPO_OBSERVATOIRE_MANIFEST = {
  id: "ao-ucpo-observatoire",
  organizationKey: "ucpo",
  organizationDisplay: "Unité de Coordination du Partenariat de Ouagadougou",
  geographicScope: "9 pays d’Afrique de l’Ouest francophone",
  countries: UCPO_COUNTRY_CODES,
  datasets: UCPO_DATASETS,
  sources: OFFICIAL_SOURCES,
  disclaimer: "Données présentées à des fins de démonstration. Les valeurs réelles issues des enquêtes citées seront intégrées à la livraison finale.",
} as const;
