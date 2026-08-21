import { useEffect, useMemo, useState } from "react";
import rawData from "../../data/project4.json";
import type { ReproductiveHealthRow } from "../../data/types";
import {
  UCPO_COUNTRIES,
  UCPO_COUNTRY_CODES,
  UCPO_DATASETS,
  UCPO_MCPR_SERIES,
  UCPO_OBSERVATOIRE_MANIFEST,
  type UcpoCountryCode,
  type UcpoDatasetMeta,
} from "../../data/projects/ao-ucpo-observatoire";
import { useAoTheme } from "../../hooks/useAoTheme";
import { BarStack, BubbleScatter, DoughnutMix, HorizontalRankBars, LineTrend } from "../../components/portfolio/appels-offres/charts";
import {
  DataTable,
  CountryComparator,
  type CountryComparatorEntity,
  CountrySelect,
  DashboardFilterBar,
  CountryFlag,
  KpiCard,
  MapChoropleth,
  Panel,
  SumBand,
} from "../../components/portfolio/appels-offres/shared";
import {
  CrisisModule,
  UcpoCountryFiche,
  UcpoRecommendations,
  UcpoSidebar,
  type UcpoSection,
} from "../../components/portfolio/appels-offres/ucpo";

const ROWS = rawData.rows as ReproductiveHealthRow[];

const SECTIONS: readonly UcpoSection[] = [
  { id: "overview", label: "Vue régionale", shortLabel: "Vue régionale", description: "9 pays, KPIs et carte", group: "Pilotage" },
  { id: "financing", label: "Financement", shortLabel: "Financement", description: "Mix bailleurs et exposition", group: "Pilotage" },
  { id: "countries", label: "Fiches pays", shortLabel: "Fiches pays", description: "6 angles d’analyse par pays", group: "Pays & résilience", badge: "9" },
  { id: "comparison", label: "Comparaison par pays", shortLabel: "Comparaison", description: "2 à 4 pays côte à côte", group: "Pays & résilience", badge: "2–4" },
  { id: "crisis", label: "Contexte de crise", shortLabel: "Contexte de crise", description: "INFORM, PDI et ruptures", group: "Pays & résilience" },
  { id: "recommendations", label: "Lecture & recommandations", shortLabel: "Recommandations", description: "Interprétation et pilotage", group: "Capitalisation", badge: "30" },
  { id: "sources", label: "Sources & méthode", shortLabel: "Sources & méthode", description: "Traçabilité des datasets", group: "Capitalisation" },
] as const;

type YearRange = readonly [number, number];

function latestReal(iso3: string, field: "mcprModern" | "tfr", yearRange: YearRange = [2010, 2024]) {
  const observations = ROWS
    .filter((row) => row.iso3 === iso3 && row[field] != null && row.year >= yearRange[0] && row.year <= yearRange[1])
    .sort((a, b) => b.year - a.year);
  const row = observations[0];
  return row ? { value: row[field] as number, year: row.year } : null;
}

function average(values: readonly number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function OverviewSection({ countries, yearRange }: { readonly countries: readonly UcpoCountryCode[]; readonly yearRange: YearRange }) {
  const realMcpr = Object.fromEntries(UCPO_COUNTRY_CODES.map((iso3) => [iso3, latestReal(iso3, "mcprModern", yearRange)?.value ?? null]));
  const realYears = Object.fromEntries(UCPO_COUNTRY_CODES.map((iso3) => [iso3, latestReal(iso3, "mcprModern", yearRange)?.year ?? null]));
  const visibleCountries = UCPO_COUNTRIES.filter((country) => countries.includes(country.iso3));
  const ranking = visibleCountries.flatMap((country) => {
    const observation = latestReal(country.iso3, "mcprModern", yearRange);
    return observation ? [{ id: country.iso3, iso3: country.iso3, name: country.name, value: observation.value, year: observation.year }] : [];
  });
  const trajectorySeries = visibleCountries.map((country) => ({ dataKey: country.iso3, name: country.shortName, unit: " %" }));
  const trajectoryData = UCPO_MCPR_SERIES.filter((row) => row.year >= yearRange[0] && row.year <= yearRange[1]);
  const riskPoints = visibleCountries.map((country) => ({ id: country.iso3, iso3: country.iso3, name: country.name, x: country.informRisk, y: country.currentMcpr, z: Math.max(20, country.displacedThousands), group: country.informRisk >= 7 ? "Risque élevé" : "Risque modéré" }));

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(270px,.75fr)]">
        <Panel title="mCPR moderne — dernière observation réelle" subtitle="Périmètre strict des neuf pays membres du Partenariat de Ouagadougou.">
          <MapChoropleth
            values={realMcpr}
            years={realYears}
            countries={UCPO_COUNTRY_CODES}
            activeCountries={countries}
            suffix=" %"
            height={390}
            source={UCPO_DATASETS.wdiCore.source}
            ariaLabel="Prévalence contraceptive moderne dans les neuf pays du Partenariat de Ouagadougou"
            legendTitle="mCPR moderne (%) — Banque mondiale WDI, dernière observation disponible"
            scopeLabel="Périmètre PO (rampe crème → navy)"
            outsideScopeLabel="Hors périmètre PO"
            noDataLabel="Pas de donnée disponible"
            discreteSteps={5}
            rampStart="#D7B85A"
            outsideFill="#E8E4DC"
            outsidePatternId="hors-po"
            scopeStroke="#C3911F"
            scopeStrokeWidth={1}
            projectionCenter={[-2, 14]}
            projectionScale={840}
            countryAdornment={(iso3) => <CountryFlag iso3={iso3} size="sm" />}
          />
        </Panel>
        <Panel title="Repères pays" subtitle="Classement selon la dernière observation WDI disponible.">
          <HorizontalRankBars
            data={ranking}
            axisMax={35}
            unit=" %"
            ariaLabel="Classement mCPR moderne des neuf pays du Partenariat de Ouagadougou, axe commençant à zéro"
            source={UCPO_DATASETS.wdiCore.source}
            rampStart="#D7B85A"
          />
        </Panel>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Trajectoires mCPR comparées" subtitle="Apport analytique : la pente compare la vitesse de progression modélisée entre pays, au-delà du seul niveau le plus récent.">
          <LineTrend data={trajectoryData} xKey="year" xLabel="Année" series={trajectorySeries} yLabel="mCPR (%)" height={310} ariaLabel="Trajectoires mCPR illustratives comparées pour les pays filtrés" source={UCPO_DATASETS.trajectory.source} illustrative formatValue={(value) => `${value.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %`} />
        </Panel>
        <Panel title="Risque opérationnel × mCPR" subtitle="Apport analytique : ce croisement identifie les pays où une progression contraceptive reste exposée à un environnement opérationnel fragile.">
          <BubbleScatter data={riskPoints} xLabel="Indice INFORM" yLabel="mCPR 2024" zLabel="PDI" yUnit=" %" zUnit=" k" height={310} ariaLabel="Relation illustrative entre risque INFORM et mCPR dans les pays PO" source={`${UCPO_DATASETS.trajectory.source} + ${UCPO_DATASETS.crisis.source}`} illustrative />
        </Panel>
      </div>
    </div>
  );
}

function FinancingSection({ countries }: { readonly countries: readonly UcpoCountryCode[] }) {
  const visibleCountries = UCPO_COUNTRIES.filter((country) => countries.includes(country.iso3));
  const rows = visibleCountries.map((country) => ({
    iso3: country.iso3,
    country: country.shortName,
    domestic: Number((country.financingUsdMillions * country.domesticShare / 100).toFixed(1)),
    usaid: Number((country.financingUsdMillions * country.usaidExposure / 100).toFixed(1)),
    others: Number((country.financingUsdMillions * (100 - country.domesticShare - country.usaidExposure) / 100).toFixed(1)),
  }));
  const domestic = rows.reduce((sum, row) => sum + row.domestic, 0);
  const usaid = rows.reduce((sum, row) => sum + row.usaid, 0);
  const others = rows.reduce((sum, row) => sum + row.others, 0);
  const total = domestic + usaid + others;
  const exposureRanking = visibleCountries.map((country) => ({ id: country.iso3, iso3: country.iso3, name: country.name, value: country.usaidExposure }));
  const resiliencePoints = visibleCountries.map((country) => ({ id: country.iso3, iso3: country.iso3, name: country.name, x: country.costPerUserUsd, y: country.domesticShare, z: country.financingUsdMillions, group: country.domesticShare >= 25 ? "Effort domestique renforcé" : "Effort domestique limité" }));

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard label="Enveloppe régionale" value={total.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="M USD" delta="+6,2 % vs 2023" trend="up" trendMagnitude={62} source={UCPO_DATASETS.financing.source} illustrative />
        <KpiCard label="Ressources domestiques" value={(domestic / total * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="%" delta="+1,8 pt vs 2023" trend="up" trendMagnitude={36} source={UCPO_DATASETS.financing.source} illustrative />
        <KpiCard label="Exposition USAID" value={(usaid / total * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="%" delta="−2,4 pt vs 2023" trend="down" positive trendMagnitude={48} source={UCPO_DATASETS.financing.source} illustrative />
        <KpiCard label="Autres partenaires" value={(others / total * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="%" delta="+0,6 pt vs 2023" trend="up" trendMagnitude={22} source={UCPO_DATASETS.financing.source} illustrative />
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(280px,.75fr)]">
        <Panel title="Ventilation par pays" subtitle="Millions USD — scénario de portefeuille.">
          <BarStack data={rows} xKey="country" countryCodeKey="iso3" series={[{ dataKey: "domestic", name: "Domestique", unit: " M USD" }, { dataKey: "usaid", name: "USAID", unit: " M USD" }, { dataKey: "others", name: "Autres", unit: " M USD" }]} height={350} ariaLabel="Financement illustratif de la planification familiale par pays" source={UCPO_DATASETS.financing.source} illustrative />
        </Panel>
        <Panel title="Mix régional" subtitle="Part de l’enveloppe illustrative.">
          <DoughnutMix data={[{ id: "domestic", name: "Domestique", value: domestic }, { id: "usaid", name: "USAID", value: usaid }, { id: "others", name: "Autres partenaires", value: others }]} valueLabel="M USD" unit=" M" ariaLabel="Mix régional illustratif du financement" source={UCPO_DATASETS.financing.source} illustrative />
        </Panel>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Exposition USAID par pays" subtitle="Apport analytique : le classement met en évidence les portefeuilles les plus sensibles à une variation de ce financement.">
          <HorizontalRankBars data={exposureRanking} axisMax={50} unit=" %" showYear={false} height={330} ariaLabel="Exposition illustrative au financement USAID par pays" source={UCPO_DATASETS.financing.source} illustrative rampStart="#D7B85A" />
        </Panel>
        <Panel title="Effort domestique × coût par utilisatrice" subtitle="Apport analytique : la taille de bulle représente l’enveloppe et distingue niveau d’autonomie et efficience apparente.">
          <BubbleScatter data={resiliencePoints} xLabel="Coût / utilisatrice" yLabel="Part domestique" zLabel="Financement" xUnit=" USD" yUnit=" %" zUnit=" M USD" height={330} ariaLabel="Relation illustrative entre coût par utilisatrice, part domestique et financement" source={`${UCPO_DATASETS.financing.source} + ${UCPO_DATASETS.impact.source}`} illustrative />
        </Panel>
      </div>
    </div>
  );
}

function CountriesSection({ countries, yearRange }: { readonly countries: readonly UcpoCountryCode[]; readonly yearRange: YearRange }) {
  const [countrySelection, setCountrySelection] = useState<UcpoCountryCode>("SEN");
  const selected = countries.includes(countrySelection) ? countrySelection : countries[0] ?? null;
  const country = selected ? UCPO_COUNTRIES.find((item) => item.iso3 === selected) ?? null : null;
  const mcpr = selected ? latestReal(selected, "mcprModern", yearRange) : null;
  const tfr = selected ? latestReal(selected, "tfr", yearRange) : null;

  useEffect(() => {
    if (selected && selected !== countrySelection) setCountrySelection(selected);
  }, [countrySelection, selected]);

  return (
    <div className="space-y-4">
      <Panel title="Sélection pays" subtitle="Chaque fiche comprend six angles analytiques et des comparaisons régionales contextualisées.">
        <CountrySelect
          label="Fiche pays active"
          options={UCPO_COUNTRIES.map((item) => ({
            value: item.iso3,
            iso3: item.iso3,
            label: item.name,
            disabled: !countries.includes(item.iso3),
            description: countries.includes(item.iso3) ? undefined : "Exclu par le filtre pays",
          }))}
          value={selected}
          onChange={(value) => value && setCountrySelection(value as UcpoCountryCode)}
          allOption={false}
        />
      </Panel>
      {country ? (
        <UcpoCountryFiche country={country} realMcpr={mcpr?.value} realMcprYear={mcpr?.year} realTfr={tfr?.value} realTfrYear={tfr?.year} yearRange={yearRange} />
      ) : (
        <Panel title="Aucune fiche accessible" subtitle="Le filtre transverse ne contient actuellement aucun pays.">
          <p className="text-[10px] leading-relaxed text-slate-500">Réactivez au moins un pays dans la barre de filtres pour ouvrir sa fiche détaillée.</p>
        </Panel>
      )}
    </div>
  );
}

const COMPARISON_METRICS = [
  { id: "trend", label: "Évolution mCPR — période active", kind: "sparkline" },
  { id: "mcpr", label: "mCPR" },
  { id: "demand", label: "Demande satisfaite" },
  { id: "unmet", label: "Besoins non satisfaits" },
  { id: "users", label: "Utilisatrices" },
  { id: "financing", label: "Financement" },
  { id: "domestic", label: "Financement domestique / total" },
  { id: "usaid", label: "Exposition USAID" },
  { id: "cost", label: "Coût par utilisatrice" },
  { id: "inform", label: "Risque INFORM" },
  { id: "displaced", label: "Personnes déplacées internes" },
  { id: "acled", label: "Événements ACLED" },
  { id: "stockouts", label: "Ruptures de stock" },
  { id: "pregnancies", label: "Grossesses évitées" },
  { id: "deaths", label: "Décès maternels évités" },
] as const;

function buildComparisonEntities(maxYear: number): readonly CountryComparatorEntity[] {
  const trajectory = UCPO_MCPR_SERIES.filter((row) => row.year <= maxYear);
  const indicatorYear = Math.min(2024, maxYear);
  const trajectoryStartYear = trajectory[0]?.year ?? indicatorYear;
  const illustrativeTag = `${indicatorYear} ill.`;

  return UCPO_COUNTRIES.map((country) => {
    const realMcpr = latestReal(country.iso3, "mcprModern", [2010, maxYear]);
    const countryTrend = trajectory.map((row) => Number(row[country.iso3]));
    const modeledMcpr = countryTrend.at(-1) ?? country.baselineMcpr;
    const demandSatisfied = Math.min(82, 34 + modeledMcpr * 1.28);
    const unmetNeed = Math.max(12, 30 - modeledMcpr * 0.35);
    const acledEvents = Math.round(country.informRisk ** 2 * 5 + country.displacedThousands / 8);
    const realTag = realMcpr ? `${realMcpr.year} WDI` : illustrativeTag;

    return {
      id: country.iso3,
      iso3: country.iso3,
      name: country.name,
      shortName: country.shortName,
      values: {
        mcpr: `${(realMcpr?.value ?? modeledMcpr).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %`,
        demand: `${demandSatisfied.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %`,
        unmet: `${unmetNeed.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %`,
        users: `${country.modernUsersMillions.toLocaleString("fr-FR", { maximumFractionDigits: 2 })} M`,
        financing: `${country.financingUsdMillions.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} M USD`,
        domestic: `${country.domesticShare.toLocaleString("fr-FR")} %`,
        usaid: `${country.usaidExposure.toLocaleString("fr-FR")} %`,
        cost: `${country.costPerUserUsd.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} USD`,
        inform: country.informRisk.toLocaleString("fr-FR", { maximumFractionDigits: 1 }),
        displaced: `${country.displacedThousands.toLocaleString("fr-FR")} k`,
        acled: acledEvents.toLocaleString("fr-FR"),
        stockouts: `${country.stockoutRate.toLocaleString("fr-FR")} %`,
        pregnancies: `${country.pregnanciesAvoidedThousands.toLocaleString("fr-FR")} k`,
        deaths: country.deathsAvoided.toLocaleString("fr-FR"),
      },
      tags: {
        trend: `${trajectoryStartYear}–${indicatorYear} ill.`,
        mcpr: realTag,
        demand: illustrativeTag,
        unmet: illustrativeTag,
        users: illustrativeTag,
        financing: illustrativeTag,
        domestic: illustrativeTag,
        usaid: illustrativeTag,
        cost: illustrativeTag,
        inform: "2024 ill.",
        displaced: "2024 ill.",
        acled: "2024 ill.",
        stockouts: "2024 ill.",
        pregnancies: illustrativeTag,
        deaths: illustrativeTag,
      },
      sparklines: {
        trend: {
          values: countryTrend,
          startLabel: `${trajectoryStartYear} · ${countryTrend[0]?.toLocaleString("fr-FR", { maximumFractionDigits: 1 }) ?? "n.d."} %`,
          endLabel: `${indicatorYear} · ${modeledMcpr.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %`,
          ariaLabel: `Évolution illustrative de la mCPR de ${country.name} jusqu’en ${indicatorYear}`,
        },
      },
      mix: country.methods.map((method) => ({ label: method.name, value: method.value })),
    };
  });
}

function ComparisonSection({ countries, selected, onChange, maxYear }: { readonly countries: readonly UcpoCountryCode[]; readonly selected: readonly string[]; readonly onChange: (ids: string[]) => void; readonly maxYear: number }) {
  const entities = buildComparisonEntities(maxYear).filter((entity) => countries.includes(entity.id as UcpoCountryCode));
  const retained = selected.filter((id) => countries.includes(id as UcpoCountryCode));
  const effectiveSelection = [...retained, ...countries.filter((id) => !retained.includes(id))].slice(0, Math.min(4, Math.max(2, retained.length)));
  if (entities.length < 2) {
    return (
      <Panel title="Comparaison par pays" subtitle="Deux pays au minimum sont nécessaires pour ouvrir la vue comparative.">
        <p className="text-[10px] leading-relaxed text-slate-500">Le filtre global ne conserve actuellement qu’un pays. Réactivez au moins un second pays dans la barre de filtres.</p>
      </Panel>
    );
  }
  return (
    <CountryComparator
      entities={entities}
      metrics={COMPARISON_METRICS}
      selectedIds={effectiveSelection}
      onChange={onChange}
      subtitle="Les mêmes indicateurs sont alignés pour comparer performance, financement, exposition et résilience sans scroll horizontal."
      source={`${UCPO_DATASETS.trajectory.source} · ${UCPO_DATASETS.financing.source} · ${UCPO_DATASETS.crisis.source} · ${UCPO_DATASETS.impact.source}`}
      illustrative
    />
  );
}

function SourcesSection() {
  const datasets = Object.values(UCPO_DATASETS) as UcpoDatasetMeta[];
  return (
    <Panel title="Registre de provenance" subtitle="Vue complète non filtrée : un flag explicite distingue chaque dataset démonstratif des données publiques réelles.">
      <DataTable
        rows={datasets}
        rowKey={(row) => row.id}
        columns={[
          { id: "dataset", header: "Dataset", accessor: (row) => row.label },
          { id: "status", header: "Statut", accessor: (row) => row.illustrative ? "Illustratif" : "Réel" },
          { id: "source", header: "Source / cadre", accessor: (row) => row.source },
          { id: "note", header: "Règle d’usage", accessor: (row) => row.note },
        ]}
        exportFilename="ucpo-registre-sources"
      />
    </Panel>
  );
}

export default function Body() {
  const { theme } = useAoTheme();
  const [section, setSection] = useState("overview");
  const [selectedCountriesState, setSelectedCountriesState] = useState<string[]>([...UCPO_COUNTRY_CODES]);
  const [maxYear, setMaxYear] = useState(2024);
  const [comparisonCountries, setComparisonCountries] = useState<string[]>(["BFA", "MLI", "NER", "SEN"]);
  const selectedCountries = useMemo<readonly UcpoCountryCode[]>(() => UCPO_COUNTRY_CODES.filter((country) => selectedCountriesState.includes(country)), [selectedCountriesState]);
  const yearRange = useMemo<YearRange>(() => [2010, maxYear], [maxYear]);
  const selectedProfiles = UCPO_COUNTRIES.filter((country) => selectedCountries.includes(country.iso3));
  const financingTotal = selectedProfiles.reduce((sum, country) => sum + country.financingUsdMillions, 0);
  const modernUsers = selectedProfiles.reduce((sum, country) => sum + country.modernUsersMillions, 0);
  const mcprRow = [...UCPO_MCPR_SERIES].reverse().find((row) => row.year >= yearRange[0] && row.year <= yearRange[1]);
  const currentMcprAverage = mcprRow && selectedProfiles.length > 0 ? average(selectedProfiles.map((country) => Number(mcprRow[country.iso3]))) : null;
  const highRiskCountries = selectedProfiles.filter((country) => country.informRisk >= 7).length;
  const activeLabel = SECTIONS.find((item) => item.id === section)?.label ?? "Observatoire";
  const selectedCountryLabel = selectedCountries.length === UCPO_COUNTRY_CODES.length
    ? "Tous les pays"
    : selectedCountries.length === 0
      ? "Aucun pays sélectionné"
      : selectedCountries.length === 1
        ? UCPO_COUNTRIES.find((country) => country.iso3 === selectedCountries[0])?.shortName ?? selectedCountries[0]
        : `${selectedCountries.length} pays sélectionnés`;
  const unfilteredSection = section === "recommendations" || section === "sources";
  const fixedYearSection = section === "financing" || section === "crisis" || section === "comparison";
  const activeFilterLabel = unfilteredSection ? "Vue complète · filtres non appliqués" : section === "comparison" ? `${selectedCountryLabel} · comparaison locale` : fixedYearSection ? `${selectedCountryLabel} · données 2024` : `${selectedCountryLabel} · jusqu’en ${maxYear}`;

  const content = useMemo(() => {
    if (selectedCountries.length === 0 && section !== "recommendations" && section !== "sources") {
      return (
        <Panel title="Aucun pays sélectionné" subtitle="Les agrégats sont suspendus pour éviter d’afficher des résultats incomplets ou trompeurs.">
          <p className="text-[10px] leading-relaxed text-slate-500">Cochez « Tous les pays » ou au moins un pays dans le filtre transverse pour réactiver cette vue.</p>
        </Panel>
      );
    }
    if (section === "financing") return <FinancingSection countries={selectedCountries} />;
    if (section === "countries") return <CountriesSection countries={selectedCountries} yearRange={yearRange} />;
    if (section === "comparison") return <ComparisonSection countries={selectedCountries} selected={comparisonCountries} onChange={setComparisonCountries} maxYear={maxYear} />;
    if (section === "crisis") return <CrisisModule countries={selectedCountries} />;
    if (section === "recommendations") return <UcpoRecommendations />;
    if (section === "sources") return <SourcesSection />;
    return <OverviewSection countries={selectedCountries} yearRange={yearRange} />;
  }, [comparisonCountries, maxYear, section, selectedCountries, yearRange]);

  return (
    <div className="space-y-5" style={{ fontFamily: theme.typography.body }}>
      <SumBand
        eyebrow="Observatoire régional de la planification familiale"
        title="Neuf pays, une lecture commune de la performance et de la résilience"
        subtitle="Observatoire régional des neuf pays du Partenariat de Ouagadougou réalisé dans le cadre de l'appel d'offres de l'UCPO."
        badge={selectedCountries.length === UCPO_COUNTRY_CODES.length ? "9 pays PO" : selectedCountryLabel}
        illustrativeNotice="Données illustratives"
        stats={[
          { label: mcprRow ? `mCPR ${mcprRow.year}` : "mCPR", value: currentMcprAverage == null ? "n.d." : currentMcprAverage.toLocaleString("fr-FR", { maximumFractionDigits: 1 }), unit: currentMcprAverage == null ? undefined : "%", source: UCPO_DATASETS.trajectory.source },
          { label: "Utilisatrices", value: modernUsers.toLocaleString("fr-FR", { maximumFractionDigits: 1 }), unit: "M", source: UCPO_DATASETS.impact.source },
          { label: "Financement", value: financingTotal.toLocaleString("fr-FR", { maximumFractionDigits: 0 }), unit: "M USD", source: UCPO_DATASETS.financing.source },
          { label: "Risque élevé", value: highRiskCountries, unit: "pays", source: UCPO_DATASETS.crisis.source },
        ]}
      />

      <DashboardFilterBar
        countryOptions={UCPO_COUNTRIES.map((country) => ({ value: country.iso3, iso3: country.iso3, label: country.name }))}
        selectedCountries={selectedCountries}
        onCountriesChange={setSelectedCountriesState}
        maxYearValue={maxYear}
        onMaxYearChange={setMaxYear}
        minYear={2010}
        maxYear={2024}
      />

      <div className="grid items-stretch gap-4 lg:grid-cols-[220px_minmax(0,1fr)]">
        <UcpoSidebar sections={SECTIONS} active={section} onChange={setSection} />
        <main className="min-w-0" aria-label={activeLabel}>
          <div className="mb-4 flex items-end justify-between gap-3 border-b pb-3" style={{ borderColor: theme.colors.border }}>
            <div>
              <p className="text-[8px] font-bold uppercase tracking-[0.16em]" style={{ color: theme.colors.accent }}>Observatoire PF</p>
              <h2 className="mt-1 text-xl" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>{activeLabel}</h2>
            </div>
            <span className="hidden rounded-full px-3 py-1 text-[8px] font-bold uppercase tracking-[0.08em] sm:inline" style={{ color: theme.colors.primary, background: theme.colors.soft }}>{activeFilterLabel}</span>
          </div>
          {content}
        </main>
      </div>

      <footer className="rounded-[9px] border px-4 py-4" style={{ borderColor: theme.colors.border, background: theme.colors.canvas }}>
        <div className="flex flex-wrap gap-x-4 gap-y-2 text-[9px] font-semibold">
          {Object.entries(UCPO_OBSERVATOIRE_MANIFEST.sources).map(([id, url]) => (
            <a key={id} href={url} target="_blank" rel="noreferrer" className="underline decoration-current/30 underline-offset-2" style={{ color: theme.colors.primary }}>{id.toUpperCase()}</a>
          ))}
        </div>
        <p className="mt-3 text-[10px] leading-relaxed" style={{ color: theme.colors.muted }}>{UCPO_OBSERVATOIRE_MANIFEST.disclaimer}</p>
      </footer>
    </div>
  );
}
