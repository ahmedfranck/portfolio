import { useState } from "react";
import { UCPO_COUNTRIES, UCPO_DATASETS, UCPO_MCPR_SERIES, type UcpoCountryProfile, type UcpoCountryTab } from "../../../../data/projects/ao-ucpo-observatoire";
import { BarStack, BubbleScatter, DoughnutMix, LineTrend } from "../charts";
import { CountryLabel, FilterChips, KpiCard, Panel } from "../shared";

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
  const trajectory = UCPO_MCPR_SERIES.map((row) => ({ year: row.year, mcpr: row[country.iso3] }));
  const financingComparison = UCPO_COUNTRIES.map((item) => ({ id: item.iso3, iso3: item.iso3, name: item.name, x: item.costPerUserUsd, y: item.domesticShare, z: item.financingUsdMillions, color: item.iso3 === country.iso3 ? "#C3911F" : undefined }));
  const demographyComparison = UCPO_COUNTRIES.map((item) => ({ id: item.iso3, iso3: item.iso3, name: item.name, x: item.tfr, y: item.currentMcpr, z: item.women15to49Millions, color: item.iso3 === country.iso3 ? "#C3911F" : undefined }));
  const methodSeries = country.methods.map((method) => ({ dataKey: method.name, name: method.name, unit: " %" }));
  const methodAverage = Object.fromEntries(country.methods.map((method) => [method.name, Number((UCPO_COUNTRIES.reduce((sum, item) => sum + (item.methods.find((candidate) => candidate.name === method.name)?.value ?? 0), 0) / UCPO_COUNTRIES.length).toFixed(1))]));
  const methodComparison = [{ profile: country.shortName, iso3: country.iso3, ...Object.fromEntries(country.methods.map((method) => [method.name, method.value])) }, { profile: "Moyenne PO", iso3: null, ...methodAverage }];
  const impactTrajectory = UCPO_MCPR_SERIES.map((row) => ({ year: row.year, pregnancies: Math.round(country.pregnanciesAvoidedThousands * Number(row[country.iso3]) / country.currentMcpr) }));
  const crisisComparison = UCPO_COUNTRIES.map((item) => ({ id: item.iso3, iso3: item.iso3, name: item.name, x: item.informRisk, y: item.stockoutRate, z: Math.max(20, item.displacedThousands), color: item.iso3 === country.iso3 ? "#C3911F" : undefined }));

  return (
    <div className="space-y-4">
      <FilterChips label={<span className="normal-case tracking-normal">Fiche <CountryLabel iso3={country.iso3} name={country.name} size="md" /></span>} options={COUNTRY_TABS} value={[tab]} onChange={(value) => setTab((value[0] as UcpoCountryTab) ?? "overview")} multiple={false} />

      {tab === "overview" && (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <KpiCard label="mCPR" value={(realMcpr ?? country.currentMcpr).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="%" context={realMcprYear ? `Observation WDI ${realMcprYear}` : "Scénario 2024"} source={realMcpr != null ? UCPO_DATASETS.wdiCore.source : UCPO_DATASETS.trajectory.source} illustrative={realMcpr == null} />
            <KpiCard label="Femmes 15–49 ans" value={country.women15to49Millions.toLocaleString("fr-FR")} unit="M" context="Population de planification" source={UCPO_DATASETS.population.source} illustrative />
            <KpiCard label="Utilisatrices modernes" value={country.modernUsersMillions.toLocaleString("fr-FR")} unit="M" context="Estimation de démonstration" source={UCPO_DATASETS.impact.source} illustrative />
            <KpiCard label="Risque INFORM" value={country.informRisk.toFixed(1)} context="Indice composite" source={UCPO_DATASETS.crisis.source} illustrative />
          </div>
          <Panel title={<span>Trajectoire synthétique · <CountryLabel iso3={country.iso3} name={country.name} /></span>} subtitle="Apport analytique : la série replace le niveau courant dans sa progression modélisée depuis 2011.">
            <LineTrend data={trajectory} xKey="year" series={[{ dataKey: "mcpr", name: country.name, unit: " %" }]} height={260} ariaLabel={`Trajectoire mCPR illustrative de ${country.name}`} source={UCPO_DATASETS.trajectory.source} illustrative />
          </Panel>
        </div>
      )}

      {tab === "financing" && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Panel title={<span>Structure de financement · <CountryLabel iso3={country.iso3} name={country.name} /></span>} subtitle={`${country.financingUsdMillions.toLocaleString("fr-FR")} M USD illustratifs`}><BarStack data={[{ iso3: country.iso3, segment: country.shortName, domestic: country.domesticShare, usaid: country.usaidExposure, others: financingOther }]} xKey="segment" countryCodeKey="iso3" series={[{ dataKey: "domestic", name: "Ressources domestiques", unit: " %" }, { dataKey: "usaid", name: "Exposition USAID", unit: " %" }, { dataKey: "others", name: "Autres partenaires", unit: " %" }]} mode="percent" height={290} ariaLabel={`Structure illustrative du financement de ${country.name}`} source={UCPO_DATASETS.financing.source} illustrative /></Panel>
          <Panel title="Positionnement régional" subtitle="Apport analytique : la position compare coût par utilisatrice, effort domestique et taille de l’enveloppe."><BubbleScatter data={financingComparison} xLabel="Coût / utilisatrice" yLabel="Part domestique" zLabel="Financement" xUnit=" USD" yUnit=" %" zUnit=" M USD" height={290} ariaLabel={`Positionnement financier illustratif de ${country.name}`} source={UCPO_DATASETS.financing.source} illustrative /></Panel>
        </div>
      )}

      {tab === "demography" && (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3"><KpiCard label="Indice de fécondité" value={(realTfr ?? country.tfr).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} unit="naiss./femme" context={realTfrYear ? `Observation WDI ${realTfrYear}` : "Valeur de scénario"} source={realTfr != null ? UCPO_DATASETS.wdiCore.source : UCPO_DATASETS.trajectory.source} illustrative={realTfr == null} /><KpiCard label="mCPR scénario 2024" value={country.currentMcpr.toLocaleString("fr-FR")} unit="%" source={UCPO_DATASETS.trajectory.source} illustrative /><KpiCard label="Gain depuis 2011" value={(country.currentMcpr - country.baselineMcpr).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} unit="points" source={UCPO_DATASETS.trajectory.source} illustrative /></div>
          <Panel title="Fécondité × mCPR dans le contexte régional" subtitle="Apport analytique : le nuage situe le pays sans transformer la corrélation observée en causalité."><BubbleScatter data={demographyComparison} xLabel="Indice de fécondité" yLabel="mCPR 2024" zLabel="Femmes 15–49 ans" xUnit="" yUnit=" %" zUnit=" M" height={300} ariaLabel={`Position démographique illustrative de ${country.name}`} source={`${UCPO_DATASETS.wdiCore.source} + ${UCPO_DATASETS.trajectory.source}`} illustrative /></Panel>
        </div>
      )}

      {tab === "methods" && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Panel title={<span>Mix de méthodes · <CountryLabel iso3={country.iso3} name={country.name} /></span>} subtitle="Répartition illustrative des utilisatrices de méthodes modernes."><DoughnutMix data={country.methods.map((method) => ({ id: method.name, name: method.name, value: method.value }))} ariaLabel={`Mix de méthodes contraceptives de ${country.name}`} source={UCPO_DATASETS.methods.source} illustrative unit=" %" valueLabel="Part" /></Panel>
          <Panel title="Écart au profil régional" subtitle="Apport analytique : la comparaison avec la moyenne PO met en évidence les dépendances propres à une méthode."><BarStack data={methodComparison} xKey="profile" countryCodeKey="iso3" series={methodSeries} mode="stacked" height={320} ariaLabel={`Comparaison du mix de méthodes de ${country.name} à la moyenne régionale`} source={UCPO_DATASETS.methods.source} illustrative /></Panel>
        </div>
      )}

      {tab === "impact" && (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3"><KpiCard label="Grossesses évitées" value={country.pregnanciesAvoidedThousands.toLocaleString("fr-FR")} unit="k" source={UCPO_DATASETS.impact.source} illustrative /><KpiCard label="Décès maternels évités" value={country.deathsAvoided.toLocaleString("fr-FR")} source={UCPO_DATASETS.impact.source} illustrative /><KpiCard label="Coût / utilisatrice" value={country.costPerUserUsd.toLocaleString("fr-FR")} unit="USD" source={UCPO_DATASETS.impact.source} illustrative /></div>
          <Panel title="Trajectoire d’impact modélisée" subtitle="Apport analytique : l’évolution des grossesses évitées suit ici la progression mCPR et explicite l’hypothèse de calcul."><LineTrend data={impactTrajectory} xKey="year" series={[{ dataKey: "pregnancies", name: "Grossesses évitées", unit: " k" }]} height={280} ariaLabel={`Grossesses évitées modélisées pour ${country.name}`} source={`${UCPO_DATASETS.impact.source} + ${UCPO_DATASETS.trajectory.source}`} illustrative /></Panel>
        </div>
      )}

      {tab === "crisis" && (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3"><KpiCard label="Indice INFORM" value={country.informRisk.toFixed(1)} source={UCPO_DATASETS.crisis.source} illustrative /><KpiCard label="PDI" value={country.displacedThousands.toLocaleString("fr-FR")} unit="milliers" source={UCPO_DATASETS.crisis.source} illustrative /><KpiCard label="Ruptures de stock" value={country.stockoutRate.toLocaleString("fr-FR")} unit="%" source={UCPO_DATASETS.crisis.source} illustrative /></div>
          <Panel title="Position dans la matrice de continuité" subtitle="Apport analytique : la comparaison régionale combine niveau de risque, ruptures et déplacements internes."><BubbleScatter data={crisisComparison} xLabel="Indice INFORM" yLabel="Ruptures" zLabel="PDI" yUnit=" %" zUnit=" k" height={300} ariaLabel={`Position de crise illustrative de ${country.name}`} source={UCPO_DATASETS.crisis.source} illustrative /></Panel>
        </div>
      )}

    </div>
  );
}
