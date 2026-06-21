import { useMemo, useState } from "react";
import rawData from "../../data/project9.json";
import type { EnergyRow } from "../../data/types";
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
import ScatterChartCard from "../../components/charts/ScatterChartCard";
import RankingBarChart from "../../components/charts/RankingBarChart";
import MultiLineChart from "../../components/charts/MultiLineChart";
import SourcesPanel, { type IndicatorSourceInfo } from "../../components/SourcesPanel";
import { latestAtOrBefore, averageDefined, averageYoYAcrossCountries, NO_DATA_LABEL } from "../../lib/realdata";
import { formatNumber, formatPercent } from "../../lib/format";

const ROWS = rawData.rows as EnergyRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["BEN", "BFA", "CIV", "GHA", "MLI", "NER", "NGA", "SEN", "TCD", "TGO"];

function kpiFor(rows: EnergyRow[], field: keyof EnergyRow, countries: string[], cutoff: number) {
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
      access: kpiFor(filtered, "electricityAccess", countries, year),
      cooking: kpiFor(filtered, "cleanCookingAccess", countries, year),
      renewable: kpiFor(filtered, "renewableShare", countries, year),
      consumption: kpiFor(filtered, "electricityConsumption", countries, year),
    }),
    [filtered, countries, year]
  );

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "electricityAccess", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "electricityAccess", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "electricityAccess", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const scatterData = useMemo(() => {
    const renewable = latestAtOrBefore(filtered, "renewableShare", year);
    const access = latestAtOrBefore(filtered, "electricityAccess", year);
    return countries
      .map((iso3) => ({
        iso3,
        name: COUNTRY_NAMES[iso3],
        x: access[iso3]?.value,
        y: renewable[iso3]?.value,
      }))
      .filter((d): d is { iso3: string; name: string; x: number; y: number } => d.x != null && d.y != null);
  }, [filtered, countries, year]);

  const cookingRankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "cleanCookingAccess", year);
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
        if (rec && rec.electricityAccess != null) row[iso3] = rec.electricityAccess;
      });
      return row;
    });
  }, [countries, year]);

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "electricityAccess", label: "Accès électricité", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "cleanCookingAccess", label: "Cuisson propre", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "renewableShare", label: "Part renouvelables", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "electricityConsumption", label: "Consommation élec./hab", format: (v) => (v != null ? formatNumber(v as number, 0) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="energie-acces-electricite" rows={tableRows} />
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
            <KpiCard label="Accès à l'électricité" value={kpis.access.avg != null ? formatPercent(kpis.access.avg) : NO_DATA_LABEL} yoyChange={kpis.access.yoy} />
            <KpiCard label="Accès à la cuisson propre" value={kpis.cooking.avg != null ? formatPercent(kpis.cooking.avg) : NO_DATA_LABEL} yoyChange={kpis.cooking.yoy} />
            <KpiCard label="Part des énergies renouvelables" value={kpis.renewable.avg != null ? formatPercent(kpis.renewable.avg) : NO_DATA_LABEL} yoyChange={kpis.renewable.yoy} />
            <KpiCard label="Consommation d'électricité" value={kpis.consumption.avg != null ? formatNumber(kpis.consumption.avg, 0) : NO_DATA_LABEL} unit={kpis.consumption.avg != null ? "kWh/hab" : undefined} yoyChange={kpis.consumption.yoy} helpText="Couverture partielle (11/16 pays)" />
          </div>
          <ChartCard title="Classement — accès à l'électricité" description="Dernière valeur réelle disponible par pays sélectionné.">
            <RankingBarChart data={rankingData} valueSuffix=" %" />
          </ChartCard>
          <InsightBox>
            <p>
              L'accès à des combustibles de cuisson propres reste en retrait par rapport à l'accès à l'électricité
              dans la plupart des pays du panel.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Accès à l'électricité par pays" description="Carte interactive : zoom, survol et clic. Dernière valeur réelle disponible.">
          <ChoroplethMap values={mapValues} years={mapYears} valueSuffix=" %" polarity="positive" />
        </ChartCard>
      ),
    },
    {
      id: "comparison",
      label: "Comparaison",
      render: () => (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title="Accès à l'électricité vs part des renouvelables" description="Dernière valeur réelle disponible par pays sélectionné.">
            <ScatterChartCard data={scatterData} xLabel="Accès à l'électricité" yLabel="Part des renouvelables" xSuffix=" %" ySuffix=" %" />
          </ChartCard>
          <ChartCard title="Classement — accès à la cuisson propre" description="Dernière valeur réelle disponible par pays.">
            <RankingBarChart data={cookingRankingData} valueSuffix=" %" />
          </ChartCard>
        </div>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Évolution de l'accès à l'électricité" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
          <MultiLineChart data={lineData} seriesCodes={countries} valueSuffix=" %" />
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
