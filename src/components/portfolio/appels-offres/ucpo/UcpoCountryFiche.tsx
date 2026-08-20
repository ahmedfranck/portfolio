import { useState } from "react";
import { UCPO_DATASETS, type UcpoCountryProfile, type UcpoCountryTab } from "../../../../data/projects/ao-ucpo-observatoire";
import { BarStack, DoughnutMix } from "../charts";
import { FilterChips, KpiCard, Panel } from "../shared";

interface UcpoCountryFicheProps {
  readonly country: UcpoCountryProfile;
  readonly realMcpr?: number | null;
  readonly realMcprYear?: number | null;
  readonly realTfr?: number | null;
  readonly realTfrYear?: number | null;
}

const COUNTRY_TABS: readonly { readonly value: UcpoCountryTab; readonly label: string }[] = [
  { value: "overview", label: "Vue générale" },
  { value: "financing", label: "Financement" },
  { value: "demography", label: "Démographie" },
  { value: "methods", label: "Méthodes" },
  { value: "impact", label: "Impact" },
  { value: "crisis", label: "Crise" },
];

export default function UcpoCountryFiche({ country, realMcpr, realMcprYear, realTfr, realTfrYear }: UcpoCountryFicheProps) {
  const [tab, setTab] = useState<UcpoCountryTab>("overview");
  const financingOther = 100 - country.domesticShare - country.usaidExposure;

  return (
    <div className="space-y-4">
      <FilterChips label={`Fiche ${country.name}`} options={COUNTRY_TABS} value={[tab]} onChange={(value) => setTab((value[0] as UcpoCountryTab) ?? "overview")} multiple={false} />

      {tab === "overview" && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <KpiCard label="mCPR" value={(realMcpr ?? country.currentMcpr).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="%" context={realMcprYear ? `Observation WDI ${realMcprYear}` : "Scénario 2024"} source={realMcpr != null ? UCPO_DATASETS.wdiCore.source : UCPO_DATASETS.trajectory.source} illustrative={realMcpr == null} />
          <KpiCard label="Femmes 15–49 ans" value={country.women15to49Millions.toLocaleString("fr-FR")} unit="M" context="Population de planification" source={UCPO_DATASETS.population.source} illustrative />
          <KpiCard label="Utilisatrices modernes" value={country.modernUsersMillions.toLocaleString("fr-FR")} unit="M" context="Estimation de démonstration" source={UCPO_DATASETS.impact.source} illustrative />
          <KpiCard label="Risque INFORM" value={country.informRisk.toFixed(1)} context="Lecture composite" source={UCPO_DATASETS.crisis.source} illustrative />
        </div>
      )}

      {tab === "financing" && (
        <Panel title={`Structure de financement · ${country.name}`} subtitle={`${country.financingUsdMillions.toLocaleString("fr-FR")} M USD illustratifs`}>
          <BarStack data={[{ segment: country.shortName, domestic: country.domesticShare, usaid: country.usaidExposure, others: financingOther }]} xKey="segment" series={[{ dataKey: "domestic", name: "Ressources domestiques", unit: " %" }, { dataKey: "usaid", name: "Exposition USAID", unit: " %" }, { dataKey: "others", name: "Autres partenaires", unit: " %" }]} mode="percent" height={260} ariaLabel={`Structure illustrative du financement de ${country.name}`} source={UCPO_DATASETS.financing.source} illustrative />
        </Panel>
      )}

      {tab === "demography" && (
        <div className="grid gap-3 sm:grid-cols-3">
          <KpiCard label="Indice de fécondité" value={(realTfr ?? country.tfr).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} unit="naiss./femme" context={realTfrYear ? `Observation WDI ${realTfrYear}` : "Valeur de scénario"} source={realTfr != null ? UCPO_DATASETS.wdiCore.source : UCPO_DATASETS.trajectory.source} illustrative={realTfr == null} />
          <KpiCard label="mCPR scénario 2024" value={country.currentMcpr.toLocaleString("fr-FR")} unit="%" source={UCPO_DATASETS.trajectory.source} illustrative />
          <KpiCard label="Gain depuis 2011" value={(country.currentMcpr - country.baselineMcpr).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="points" source={UCPO_DATASETS.trajectory.source} illustrative />
        </div>
      )}

      {tab === "methods" && (
        <Panel title={`Mix de méthodes · ${country.name}`} subtitle="Répartition illustrative des utilisatrices de méthodes modernes.">
          <DoughnutMix data={country.methods.map((method) => ({ id: method.name, name: method.name, value: method.value }))} ariaLabel={`Mix de méthodes contraceptives de ${country.name}`} source={UCPO_DATASETS.methods.source} illustrative unit=" %" valueLabel="Part" />
        </Panel>
      )}

      {tab === "impact" && (
        <div className="grid gap-3 sm:grid-cols-3">
          <KpiCard label="Grossesses évitées" value={country.pregnanciesAvoidedThousands.toLocaleString("fr-FR")} unit="k" source={UCPO_DATASETS.impact.source} illustrative />
          <KpiCard label="Décès maternels évités" value={country.deathsAvoided.toLocaleString("fr-FR")} source={UCPO_DATASETS.impact.source} illustrative />
          <KpiCard label="Coût / utilisatrice" value={country.costPerUserUsd.toLocaleString("fr-FR")} unit="USD" source={UCPO_DATASETS.impact.source} illustrative />
        </div>
      )}

      {tab === "crisis" && (
        <div className="grid gap-3 sm:grid-cols-3">
          <KpiCard label="Indice INFORM" value={country.informRisk.toFixed(1)} source={UCPO_DATASETS.crisis.source} illustrative />
          <KpiCard label="PDI" value={country.displacedThousands.toLocaleString("fr-FR")} unit="milliers" source={UCPO_DATASETS.crisis.source} illustrative />
          <KpiCard label="Ruptures de stock" value={country.stockoutRate.toLocaleString("fr-FR")} unit="%" source={UCPO_DATASETS.crisis.source} illustrative />
        </div>
      )}

    </div>
  );
}
