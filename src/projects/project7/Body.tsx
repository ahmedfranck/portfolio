import { useMemo, useState } from "react";
import rawData from "../../data/project7.json";
import type { EconomyRow } from "../../data/types";
import { COUNTRY_NAMES, MAX_YEAR } from "../../data/countries";
import CountryMultiSelect from "../../components/filters/CountryMultiSelect";
import YearSlider from "../../components/filters/YearSlider";
import KpiCard from "../../components/KpiCard";
import ChartCard from "../../components/ChartCard";
import ExportButton from "../../components/ExportButton";
import InsightBox from "../../components/InsightBox";
import DashboardTabs, { type DashboardTab } from "../../components/DashboardTabs";
import DataTable, { type DataTableColumn } from "../../components/DataTable";
import ChoroplethMap from "../../components/charts/ChoroplethMap";
import MultiLineChart from "../../components/charts/MultiLineChart";
import ScatterChartCard from "../../components/charts/ScatterChartCard";
import RankingBarChart from "../../components/charts/RankingBarChart";
import SourcesPanel, { type IndicatorSourceInfo } from "../../components/SourcesPanel";
import { latestAtOrBefore, averageDefined, averageYoYAcrossCountries, NO_DATA_LABEL } from "../../lib/realdata";
import { formatNumber, formatPercent } from "../../lib/format";

const ROWS = rawData.rows as EconomyRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["SEN", "CIV", "GHA", "NGA", "MLI", "COD"];

function kpiFor(rows: EconomyRow[], field: keyof EconomyRow, countries: string[], cutoff: number) {
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

  const filtered = useMemo(() => ROWS.filter((r) => countries.includes(r.iso3)), [countries]);

  const kpis = useMemo(
    () => ({
      gdpGrowth: kpiFor(filtered, "gdpGrowth", countries, year),
      gdpPerCapita: kpiFor(filtered, "gdpPerCapita", countries, year),
      inflation: kpiFor(filtered, "inflation", countries, year),
      fdi: kpiFor(filtered, "fdiPercentGdp", countries, year),
    }),
    [filtered, countries, year]
  );

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "gdpPerCapita", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const lineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      countries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.gdpGrowth != null) row[iso3] = rec.gdpGrowth;
      });
      return row;
    });
  }, [countries, year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "gdpPerCapita", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "gdpPerCapita", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const scatterData = useMemo(() => {
    const inflation = latestAtOrBefore(filtered, "inflation", year);
    const growth = latestAtOrBefore(filtered, "gdpGrowth", year);
    return countries
      .map((iso3) => ({
        iso3,
        name: COUNTRY_NAMES[iso3],
        x: growth[iso3]?.value,
        y: inflation[iso3]?.value,
      }))
      .filter((d): d is { iso3: string; name: string; x: number; y: number } => d.x != null && d.y != null);
  }, [filtered, countries, year]);

  const fdiRankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "fdiPercentGdp", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "gdpGrowth", label: "Croissance PIB", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "gdpPerCapita", label: "PIB / habitant", format: (v) => (v != null ? formatNumber(v as number, 0) : NO_DATA_LABEL) },
    { key: "inflation", label: "Inflation", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "fdiPercentGdp", label: "IDE entrants", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="economie-croissance" rows={tableRows} />
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
            <KpiCard label="Croissance du PIB" value={kpis.gdpGrowth.avg != null ? formatPercent(kpis.gdpGrowth.avg) : NO_DATA_LABEL} yoyChange={kpis.gdpGrowth.yoy} />
            <KpiCard label="PIB par habitant" value={kpis.gdpPerCapita.avg != null ? formatNumber(kpis.gdpPerCapita.avg, 0) : NO_DATA_LABEL} unit={kpis.gdpPerCapita.avg != null ? "USD" : undefined} yoyChange={kpis.gdpPerCapita.yoy} />
            <KpiCard label="Inflation" value={kpis.inflation.avg != null ? formatPercent(kpis.inflation.avg) : NO_DATA_LABEL} yoyChange={kpis.inflation.yoy} lowerIsBetter />
            <KpiCard label="IDE entrants" value={kpis.fdi.avg != null ? formatPercent(kpis.fdi.avg) : NO_DATA_LABEL} helpText="% du PIB" yoyChange={kpis.fdi.yoy} />
          </div>
          <ChartCard title="Classement — PIB par habitant" description="Dernière valeur réelle disponible par pays sélectionné.">
            <RankingBarChart data={rankingData} valueSuffix=" $" />
          </ChartCard>
          <InsightBox>
            <p>
              Les pays affichant le PIB par habitant le plus élevé ne sont pas systématiquement ceux qui attirent
              le plus d'investissements directs étrangers en % du PIB.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Évolution de la croissance du PIB" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
          <MultiLineChart data={lineData} seriesCodes={countries} valueSuffix=" %" />
        </ChartCard>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="PIB par habitant par pays" description="Carte interactive : zoom, survol et clic. Dernière valeur réelle disponible.">
          <ChoroplethMap values={mapValues} years={mapYears} valueSuffix=" $" polarity="positive" />
        </ChartCard>
      ),
    },
    {
      id: "comparison",
      label: "Comparaison",
      render: () => (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title="Croissance vs inflation" description="Dernière valeur réelle disponible par pays sélectionné.">
            <ScatterChartCard data={scatterData} xLabel="Croissance du PIB" yLabel="Inflation" xSuffix=" %" ySuffix=" %" />
          </ChartCard>
          <ChartCard title="Classement — investissements directs étrangers" description="Dernière valeur réelle disponible par pays, en % du PIB.">
            <RankingBarChart data={fdiRankingData} valueSuffix=" %" />
          </ChartCard>
        </div>
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
