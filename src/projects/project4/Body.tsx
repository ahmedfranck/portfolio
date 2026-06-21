import { useMemo, useState } from "react";
import rawData from "../../data/project4.json";
import type { ReproductiveHealthRow } from "../../data/types";
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

const ROWS = rawData.rows as ReproductiveHealthRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["BEN", "CIV", "GHA", "MLI", "SEN", "TGO"];

function kpiFor(
  rows: ReproductiveHealthRow[],
  field: keyof ReproductiveHealthRow,
  countries: string[],
  cutoff: number
) {
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
      mcpr: kpiFor(filtered, "mcpr", countries, year),
      mcprModern: kpiFor(filtered, "mcprModern", countries, year),
      tfr: kpiFor(filtered, "tfr", countries, year),
      demandSatisfied: kpiFor(filtered, "demandSatisfied", countries, year),
    }),
    [filtered, countries, year]
  );

  const lineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      countries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.mcpr != null) row[iso3] = rec.mcpr;
      });
      return row;
    });
  }, [countries, year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "mcprModern", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "mcprModern", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const scatterData = useMemo(() => {
    const tfr = latestAtOrBefore(filtered, "tfr", year);
    const mcprModern = latestAtOrBefore(filtered, "mcprModern", year);
    return countries
      .map((iso3) => ({
        iso3,
        name: COUNTRY_NAMES[iso3],
        x: tfr[iso3]?.value,
        y: mcprModern[iso3]?.value,
      }))
      .filter((d): d is { iso3: string; name: string; x: number; y: number } => d.x != null && d.y != null);
  }, [filtered, countries, year]);

  const modernRankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "mcprModern", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const demandRankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "demandSatisfied", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "mcpr", label: "Prévalence (toutes méthodes)", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "mcprModern", label: "Prévalence moderne", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "tfr", label: "Fécondité (ISF)", format: (v) => (v != null ? formatNumber(v as number, 2) : NO_DATA_LABEL) },
    { key: "demandSatisfied", label: "Demande satisfaite", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="sante-reproductive-fecondite" rows={tableRows} />
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
            <KpiCard label="Prévalence contraceptive (toutes méthodes)" value={kpis.mcpr.avg != null ? formatPercent(kpis.mcpr.avg) : NO_DATA_LABEL} yoyChange={kpis.mcpr.yoy} />
            <KpiCard label="Prévalence, méthodes modernes" value={kpis.mcprModern.avg != null ? formatPercent(kpis.mcprModern.avg) : NO_DATA_LABEL} yoyChange={kpis.mcprModern.yoy} />
            <KpiCard label="Indice synthétique de fécondité" value={kpis.tfr.avg != null ? formatNumber(kpis.tfr.avg, 2) : NO_DATA_LABEL} unit={kpis.tfr.avg != null ? "naiss./femme" : undefined} yoyChange={kpis.tfr.yoy} lowerIsBetter />
            <KpiCard label="Demande satisfaite (méthodes modernes)" value={kpis.demandSatisfied.avg != null ? formatPercent(kpis.demandSatisfied.avg) : NO_DATA_LABEL} yoyChange={kpis.demandSatisfied.yoy} />
          </div>
          <ChartCard title="Classement — prévalence contraceptive moderne" description="Dernière valeur réelle disponible par pays sélectionné.">
            <RankingBarChart data={modernRankingData} valueSuffix=" %" />
          </ChartCard>
          <InsightBox>
            <p>
              La prévalence contraceptive moderne progresse dans la plupart des pays du panel, en lien avec une
              baisse progressive de l'indice de fécondité.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Évolution de la prévalence contraceptive (toutes méthodes)" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
          <MultiLineChart data={lineData} seriesCodes={countries} valueSuffix=" %" />
        </ChartCard>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Prévalence contraceptive moderne par pays" description="Carte interactive : zoom, survol et clic. Dernière valeur réelle disponible.">
          <ChoroplethMap values={mapValues} years={mapYears} valueSuffix=" %" polarity="positive" />
        </ChartCard>
      ),
    },
    {
      id: "comparison",
      label: "Comparaison",
      render: () => (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title="Fécondité vs prévalence contraceptive moderne" description="Dernière valeur réelle disponible par pays sélectionné.">
            <ScatterChartCard data={scatterData} xLabel="Indice de fécondité" yLabel="Prévalence, méthodes modernes" xSuffix=" naiss./femme" ySuffix=" %" />
          </ChartCard>
          <ChartCard title="Classement — demande satisfaite par méthodes modernes" description="Dernière valeur réelle disponible par pays.">
            <RankingBarChart data={demandRankingData} valueSuffix=" %" />
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
