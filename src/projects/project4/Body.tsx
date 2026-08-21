import { useMemo, useState } from "react";
import rawData from "../../data/project4.json";
import type { ReproductiveHealthRow } from "../../data/types";
import {
  UCPO_COUNTRIES,
  UCPO_COUNTRY_CODES,
  UCPO_DATASETS,
  UCPO_OBSERVATOIRE_MANIFEST,
  UCPO_TIMELINE,
  type UcpoCountryCode,
  type UcpoDatasetMeta,
} from "../../data/projects/ao-ucpo-observatoire";
import { useAoTheme } from "../../hooks/useAoTheme";
import { BarStack, DoughnutMix, HorizontalRankBars } from "../../components/portfolio/appels-offres/charts";
import {
  DataTable,
  FilterChips,
  KpiCard,
  MapChoropleth,
  Note,
  Panel,
  SumBand,
  Timeline,
} from "../../components/portfolio/appels-offres/shared";
import {
  CrisisModule,
  MotionTracker,
  UcpoCountryFiche,
  UcpoSidebar,
  type UcpoSection,
} from "../../components/portfolio/appels-offres/ucpo";

const ROWS = rawData.rows as ReproductiveHealthRow[];

const SECTIONS: readonly UcpoSection[] = [
  { id: "overview", label: "Vue régionale", shortLabel: "Vue régionale", description: "9 pays, KPIs et carte" },
  { id: "motion", label: "Motion Tracker", shortLabel: "Motion Tracker", description: "Trajectoires mCPR 2011–2024" },
  { id: "financing", label: "Financement", shortLabel: "Financement", description: "Mix bailleurs et exposition" },
  { id: "countries", label: "Fiches pays", shortLabel: "9 fiches pays", description: "6 angles d’analyse par pays" },
  { id: "crisis", label: "Contexte de crise", shortLabel: "Contexte de crise", description: "INFORM, PDI et ruptures" },
  { id: "sources", label: "Sources & méthode", shortLabel: "Sources & méthode", description: "Traçabilité des datasets" },
] as const;

function latestReal(iso3: string, field: "mcprModern" | "tfr") {
  const observations = ROWS
    .filter((row) => row.iso3 === iso3 && row[field] != null)
    .sort((a, b) => b.year - a.year);
  const row = observations[0];
  return row ? { value: row[field] as number, year: row.year } : null;
}

function average(values: readonly number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function OverviewSection() {
  const realMcpr = Object.fromEntries(UCPO_COUNTRY_CODES.map((iso3) => [iso3, latestReal(iso3, "mcprModern")?.value ?? null]));
  const realYears = Object.fromEntries(UCPO_COUNTRY_CODES.map((iso3) => [iso3, latestReal(iso3, "mcprModern")?.year ?? null]));
  const ranking = UCPO_COUNTRIES.flatMap((country) => {
    const observation = latestReal(country.iso3, "mcprModern");
    return observation ? [{ id: country.iso3, name: country.name, value: observation.value, year: observation.year }] : [];
  });

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(270px,.75fr)]">
        <Panel title="mCPR moderne — dernière observation réelle" subtitle="Périmètre strict des neuf pays membres du Partenariat de Ouagadougou.">
          <MapChoropleth
            values={realMcpr}
            years={realYears}
            countries={UCPO_COUNTRY_CODES}
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
      <div className="grid gap-3 md:grid-cols-3">
        <Note title="Périmètre confirmé">La vue régionale exclut tout pays hors des neuf membres du Partenariat de Ouagadougou.</Note>
        <Note title="Dates hétérogènes" variant="warning">Le millésime affiché varie selon le pays; comparer une valeur exige de vérifier son année d’observation.</Note>
        <Note title="Lecture responsable" variant="info">Les KPIs transversaux de démonstration sont séparés des observations WDI réelles par un badge orange.</Note>
      </div>
    </div>
  );
}

function MotionSection() {
  const [countries, setCountries] = useState<UcpoCountryCode[]>([...UCPO_COUNTRY_CODES]);
  return (
    <div className="space-y-4">
      <Panel title="Motion Tracker régional" subtitle="Filtrer les trajectoires sans perdre la traçabilité du scénario.">
        <div className="mb-4">
          <FilterChips
            label="Pays"
            options={UCPO_COUNTRIES.map((country) => ({ value: country.iso3, label: country.shortName }))}
            value={countries}
            onChange={(values) => values.length > 0 && setCountries(values as UcpoCountryCode[])}
          />
        </div>
        <MotionTracker countries={countries} />
      </Panel>
      <div className="grid gap-3 md:grid-cols-3">
        <Note title="Courbe monotone">Le rendu Recharts reprend la tension visuelle du prototype UCPO via une interpolation monotone.</Note>
        <Note title="Série démonstrative" variant="warning">Les points annuels sont interpolés pour le prototype et ne doivent pas être cités comme estimations FPET.</Note>
        <Note title="Substitution finale" variant="info">La livraison finale doit charger les séries pays validées par Track20/FP2030 et conserver leurs intervalles d’incertitude.</Note>
      </div>
    </div>
  );
}

function FinancingSection() {
  const rows = UCPO_COUNTRIES.map((country) => ({
    country: country.shortName,
    domestic: Number((country.financingUsdMillions * country.domesticShare / 100).toFixed(1)),
    usaid: Number((country.financingUsdMillions * country.usaidExposure / 100).toFixed(1)),
    others: Number((country.financingUsdMillions * (100 - country.domesticShare - country.usaidExposure) / 100).toFixed(1)),
  }));
  const domestic = rows.reduce((sum, row) => sum + row.domestic, 0);
  const usaid = rows.reduce((sum, row) => sum + row.usaid, 0);
  const others = rows.reduce((sum, row) => sum + row.others, 0);
  const total = domestic + usaid + others;

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard label="Enveloppe régionale" value={total.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="M USD" source={UCPO_DATASETS.financing.source} illustrative />
        <KpiCard label="Ressources domestiques" value={(domestic / total * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="%" source={UCPO_DATASETS.financing.source} illustrative />
        <KpiCard label="Exposition USAID" value={(usaid / total * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="%" source={UCPO_DATASETS.financing.source} illustrative />
        <KpiCard label="Autres partenaires" value={(others / total * 100).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="%" source={UCPO_DATASETS.financing.source} illustrative />
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(280px,.75fr)]">
        <Panel title="Ventilation par pays" subtitle="Millions USD — scénario de portefeuille.">
          <BarStack data={rows} xKey="country" series={[{ dataKey: "domestic", name: "Domestique", unit: " M USD" }, { dataKey: "usaid", name: "USAID", unit: " M USD" }, { dataKey: "others", name: "Autres", unit: " M USD" }]} height={350} ariaLabel="Financement illustratif de la planification familiale par pays" source={UCPO_DATASETS.financing.source} illustrative />
        </Panel>
        <Panel title="Mix régional" subtitle="Part de l’enveloppe illustrative.">
          <DoughnutMix data={[{ id: "domestic", name: "Domestique", value: domestic }, { id: "usaid", name: "USAID", value: usaid }, { id: "others", name: "Autres partenaires", value: others }]} valueLabel="M USD" unit=" M" ariaLabel="Mix régional illustratif du financement" source={UCPO_DATASETS.financing.source} illustrative />
        </Panel>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        <Note title="Exposition bailleur" variant="warning">Une part USAID élevée signale un besoin de scénario de continuité, sans préjuger des décaissements réels.</Note>
        <Note title="Effort domestique">La part nationale sert d’indicateur de résilience financière et doit être rapprochée de l’exécution budgétaire.</Note>
        <Note title="Validation finale" variant="info">Les montants seront remplacés par les comptes nationaux, rapports FPSA et données bailleurs validés.</Note>
      </div>
    </div>
  );
}

function CountriesSection() {
  const [selected, setSelected] = useState<UcpoCountryCode>("SEN");
  const country = UCPO_COUNTRIES.find((item) => item.iso3 === selected)!;
  const mcpr = latestReal(selected, "mcprModern");
  const tfr = latestReal(selected, "tfr");

  return (
    <div className="space-y-4">
      <Panel title="Sélection pays" subtitle="Chaque fiche comprend six angles et trois callouts d’interprétation.">
        <FilterChips label="Pays UCPO" options={UCPO_COUNTRIES.map((item) => ({ value: item.iso3, label: item.shortName }))} value={[selected]} onChange={(values) => setSelected(values[0] as UcpoCountryCode)} multiple={false} />
      </Panel>
      <UcpoCountryFiche country={country} realMcpr={mcpr?.value} realMcprYear={mcpr?.year} realTfr={tfr?.value} realTfrYear={tfr?.year} />
    </div>
  );
}

function SourcesSection() {
  const datasets = Object.values(UCPO_DATASETS) as UcpoDatasetMeta[];
  return (
    <div className="space-y-4">
      <Panel title="Registre de provenance" subtitle="Un flag explicite distingue chaque dataset démonstratif des données publiques réelles.">
        <DataTable
          rows={datasets}
          rowKey={(row) => row.id}
          columns={[
            { id: "dataset", header: "Dataset", accessor: (row) => row.label },
            { id: "status", header: "Statut", accessor: (row) => row.illustrative ? "Illustratif" : "Réel" },
            { id: "source", header: "Source / cadre", accessor: (row) => row.source },
            { id: "note", header: "Règle d’usage", accessor: (row) => row.note },
          ]}
          source="Manifeste UCPO du projet"
          illustrative={datasets.some((dataset) => dataset.illustrative)}
          exportFilename="ucpo-registre-sources"
        />
      </Panel>
      <Panel title="Chronologie programmatique" subtitle="Jalons éditoriaux du prototype.">
        <Timeline entries={UCPO_TIMELINE} />
      </Panel>
      <div className="grid gap-3 md:grid-cols-3">
        <Note title="Principe de priorité">Une donnée publique gratuite et sans clé remplace le scénario dès qu’elle est disponible et documentée.</Note>
        <Note title="Accès restreint" variant="warning">ACLED requiert un compte; aucune valeur ACLED n’est présentée comme observation réelle dans cette version.</Note>
        <Note title="Reproductibilité" variant="info">Le manifeste centralise statut, URL, source et règle d’usage pour faciliter le remplacement dataset par dataset.</Note>
      </div>
    </div>
  );
}

export default function Body() {
  const { theme } = useAoTheme();
  const [section, setSection] = useState("overview");
  const financingTotal = UCPO_COUNTRIES.reduce((sum, country) => sum + country.financingUsdMillions, 0);
  const modernUsers = UCPO_COUNTRIES.reduce((sum, country) => sum + country.modernUsersMillions, 0);
  const currentMcprAverage = average(UCPO_COUNTRIES.map((country) => country.currentMcpr));
  const highRiskCountries = UCPO_COUNTRIES.filter((country) => country.informRisk >= 7).length;
  const activeLabel = SECTIONS.find((item) => item.id === section)?.label ?? "Observatoire";

  const content = useMemo(() => {
    if (section === "motion") return <MotionSection />;
    if (section === "financing") return <FinancingSection />;
    if (section === "countries") return <CountriesSection />;
    if (section === "crisis") return <CrisisModule />;
    if (section === "sources") return <SourcesSection />;
    return <OverviewSection />;
  }, [section]);

  return (
    <div className="space-y-5" style={{ fontFamily: theme.typography.body }}>
      <SumBand
        eyebrow="Observatoire régional de la planification familiale"
        title="Neuf pays, une lecture commune de la performance et de la résilience"
        subtitle="Périmètre : 9 pays du Partenariat de Ouagadougou. Prototype de référence UCPO avec séparation explicite des données réelles et illustratives."
        badge="9 pays PO"
        illustrativeNotice="Les 4 KPI ci-dessous sont des scénarios illustratifs UCPO — voir Sources & méthode"
        stats={[
          { label: "mCPR 2024", value: currentMcprAverage.toLocaleString("fr-FR", { maximumFractionDigits: 1 }), unit: "%", source: UCPO_DATASETS.motion.source },
          { label: "Utilisatrices", value: modernUsers.toLocaleString("fr-FR", { maximumFractionDigits: 1 }), unit: "M", source: UCPO_DATASETS.impact.source },
          { label: "Financement", value: financingTotal.toLocaleString("fr-FR", { maximumFractionDigits: 0 }), unit: "M USD", source: UCPO_DATASETS.financing.source },
          { label: "Risque élevé", value: highRiskCountries, unit: "pays", source: UCPO_DATASETS.crisis.source },
        ]}
      />

      <div className="grid items-start gap-4 lg:grid-cols-[220px_minmax(0,1fr)]">
        <UcpoSidebar sections={SECTIONS} active={section} onChange={setSection} />
        <main className="min-w-0" aria-label={activeLabel}>
          <div className="mb-4 flex items-end justify-between gap-3 border-b pb-3" style={{ borderColor: theme.colors.border }}>
            <div>
              <p className="text-[8px] font-bold uppercase tracking-[0.16em]" style={{ color: theme.colors.accent }}>Observatoire PF</p>
              <h2 className="mt-1 text-xl" style={{ color: theme.colors.primary, fontFamily: theme.typography.heading }}>{activeLabel}</h2>
            </div>
            <span className="hidden rounded-full px-3 py-1 text-[8px] font-bold uppercase tracking-[0.08em] sm:inline" style={{ color: theme.colors.primary, background: theme.colors.soft }}>9 pays PO</span>
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
