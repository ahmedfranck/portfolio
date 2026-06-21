import { useMemo, useState } from "react";
import rawData from "../../data/project2.json";
import type { NutritionRow } from "../../data/types";
import { COUNTRY_NAMES, MAX_YEAR } from "../../data/countries";
import CountryMultiSelect from "../../components/filters/CountryMultiSelect";
import YearSlider from "../../components/filters/YearSlider";
import IndicatorSelect from "../../components/filters/IndicatorSelect";
import KpiCard from "../../components/KpiCard";
import ChartCard from "../../components/ChartCard";
import ExportButton from "../../components/ExportButton";
import InsightBox from "../../components/InsightBox";
import DashboardTabs, { type DashboardTab } from "../../components/DashboardTabs";
import DataTable, { type DataTableColumn } from "../../components/DataTable";
import ChoroplethMap from "../../components/charts/ChoroplethMap";
import SmallMultiples from "../../components/charts/SmallMultiples";
import ScatterChartCard from "../../components/charts/ScatterChartCard";
import RankingBarChart from "../../components/charts/RankingBarChart";
import GaugeGrid from "../../components/charts/GaugeGrid";
import SourcesPanel, { type IndicatorSourceInfo } from "../../components/SourcesPanel";
import { latestAtOrBefore, averageDefined, averageYoYAcrossCountries, NO_DATA_LABEL } from "../../lib/realdata";
import { formatPercent, formatNumber } from "../../lib/format";

const ROWS = rawData.rows as NutritionRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["BEN", "BFA", "CMR", "CIV", "GHA", "MLI", "NGA", "TCD"];

const INDICATOR_OPTIONS = [
  { value: "stunting", label: "Retard de croissance (stunting)" },
  { value: "wasting", label: "Émaciation (wasting)" },
  { value: "underweight", label: "Insuffisance pondérale" },
] as const;

type IndicatorKey = (typeof INDICATOR_OPTIONS)[number]["value"];

function kpiFor(rows: NutritionRow[], field: keyof NutritionRow, countries: string[], cutoff: number) {
  const latest = latestAtOrBefore(rows, field, cutoff);
  const avg = averageDefined(countries.map((c) => latest[c]?.value));
  const rowsUpToCutoff = rows.filter((r) => r.year <= cutoff);
  const yoy = averageYoYAcrossCountries(rowsUpToCutoff, field, countries);
  return { avg, yoy };
}

export default function Body() {
  const [tab, setTab] = useState("overview");
  const [countries, setCountries] = useState<string[]>(DEFAULT_COUNTRIES);
  const [year, setYear] = useState<number>(MAX_YEAR);
  const [indicator, setIndicator] = useState<IndicatorKey>("stunting");

  const filtered = useMemo(() => ROWS.filter((r) => countries.includes(r.iso3)), [countries]);
  const indicatorLabel = INDICATOR_OPTIONS.find((o) => o.value === indicator)!.label;

  const kpis = useMemo(
    () => ({
      stunting: kpiFor(filtered, "stunting", countries, year),
      wasting: kpiFor(filtered, "wasting", countries, year),
      underweight: kpiFor(filtered, "underweight", countries, year),
      dtp3: kpiFor(filtered, "dtp3Coverage", countries, year),
      measles: kpiFor(filtered, "measlesCoverage", countries, year),
    }),
    [filtered, countries, year]
  );

  const vaccineAvg = useMemo(() => {
    if (kpis.dtp3.avg == null && kpis.measles.avg == null) return null;
    return averageDefined([kpis.dtp3.avg, kpis.measles.avg]);
  }, [kpis]);

  const smallMultiplesData = useMemo(
    () =>
      countries.map((iso3) => {
        const series = ROWS.filter((r) => r.iso3 === iso3 && r.year <= year && r[indicator] != null).map((r) => ({
          year: r.year,
          value: r[indicator] as number,
        }));
        const latest = series[series.length - 1]?.value ?? null;
        return { iso3, name: COUNTRY_NAMES[iso3], series, latest };
      }),
    [countries, indicator, year]
  );

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, indicator, year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [indicator, year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, indicator, year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [indicator, year]);

  const scatterData = useMemo(() => {
    const latestIndicator = latestAtOrBefore(filtered, indicator, year);
    const latestGdp = latestAtOrBefore(filtered, "gdpPerCapita", year);
    return countries
      .map((iso3) => ({
        iso3,
        name: COUNTRY_NAMES[iso3],
        x: latestGdp[iso3]?.value,
        y: latestIndicator[iso3]?.value,
      }))
      .filter((d): d is { iso3: string; name: string; x: number; y: number } => d.x != null && d.y != null);
  }, [filtered, countries, indicator, year]);

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, indicator, year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, indicator, year]);

  const gaugeData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "dtp3Coverage", year);
    return countries
      .map((iso3) => ({ iso3, name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { iso3: string; name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "stunting", label: "Stunting", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "wasting", label: "Wasting", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "underweight", label: "Insuff. pondérale", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "dtp3Coverage", label: "DTC3", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "gdpPerCapita", label: "PIB/habitant", format: (v) => (v != null ? formatNumber(v as number, 0) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <IndicatorSelect value={indicator} onChange={(v) => setIndicator(v as IndicatorKey)} options={INDICATOR_OPTIONS} />
      <div className="ml-auto">
        <ExportButton filename="nutrition-survie-enfant" rows={tableRows} />
      </div>
    </div>
  );

  const tabs: DashboardTab[] = [
    {
      id: "overview",
      label: "Vue d'ensemble",
      render: () => (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <KpiCard label="Retard de croissance" value={kpis.stunting.avg != null ? formatPercent(kpis.stunting.avg) : NO_DATA_LABEL} yoyChange={kpis.stunting.yoy} lowerIsBetter />
            <KpiCard label="Émaciation" value={kpis.wasting.avg != null ? formatPercent(kpis.wasting.avg) : NO_DATA_LABEL} yoyChange={kpis.wasting.yoy} lowerIsBetter />
            <KpiCard label="Insuffisance pondérale" value={kpis.underweight.avg != null ? formatPercent(kpis.underweight.avg) : NO_DATA_LABEL} yoyChange={kpis.underweight.yoy} lowerIsBetter />
            <KpiCard label="Couverture vaccinale moyenne" value={vaccineAvg != null ? formatPercent(vaccineAvg) : NO_DATA_LABEL} helpText="Moyenne DTC3 / rougeole" />
          </div>
          <ChartCard title="Couverture vaccinale DTC3" description="Dernière valeur réelle disponible par pays.">
            <GaugeGrid data={gaugeData} />
          </ChartCard>
          <InsightBox>
            <p>
              Le retard de croissance recule sur la durée dans la plupart des pays, mais reste à des niveaux élevés
              là où l'insuffisance pondérale demeure forte.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title={`${indicatorLabel} par pays`} description="Carte interactive : zoom, survol et clic. Dernière valeur réelle disponible.">
          <ChoroplethMap values={mapValues} years={mapYears} valueSuffix=" %" polarity="negative" />
        </ChartCard>
      ),
    },
    {
      id: "comparison",
      label: "Comparaison",
      render: () => (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title={`Classement — ${indicatorLabel.toLowerCase()}`} description="Dernière valeur réelle disponible par pays.">
            <RankingBarChart data={rankingData} valueSuffix=" %" />
          </ChartCard>
          <ChartCard title={`${indicatorLabel} vs PIB/habitant`} description="Dernière valeur réelle disponible par pays sélectionné.">
            <ScatterChartCard data={scatterData} xLabel="PIB par habitant" yLabel={indicatorLabel} xSuffix=" $" ySuffix=" %" />
          </ChartCard>
        </div>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Évolution par pays (small multiples)" description="Tendance réelle de l'indicateur sélectionné, pays par pays (données manquantes = rupture de courbe).">
          <SmallMultiples data={smallMultiplesData} valueSuffix=" %" />
        </ChartCard>
      ),
    },
    {
      id: "data",
      label: "Données",
      render: () => (
        <div className="space-y-4">
          <DataTable rows={tableRows} columns={columns} searchableKey="pays" />
          <SourcesPanel indicators={INDICATORS} />
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      {filterBar}
      <DashboardTabs tabs={tabs} active={tab} onChange={setTab} />
    </div>
  );
}
