import { useMemo, useState } from "react";
import rawData from "../../data/project5.json";
import type { WashRow } from "../../data/types";
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
import BubbleMap from "../../components/charts/BubbleMap";
import MultiLineChart from "../../components/charts/MultiLineChart";
import ScatterChartCard from "../../components/charts/ScatterChartCard";
import RankingBarChart from "../../components/charts/RankingBarChart";
import SourcesPanel, { type IndicatorSourceInfo } from "../../components/SourcesPanel";
import { latestAtOrBefore, averageDefined, averageYoYAcrossCountries, NO_DATA_LABEL } from "../../lib/realdata";
import { formatNumber, formatPercent } from "../../lib/format";

const ROWS = rawData.rows as WashRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["BEN", "BFA", "CIV", "GHA", "MLI", "NER", "NGA", "SEN", "TCD", "TGO"];

function kpiFor(rows: WashRow[], field: keyof WashRow, countries: string[], cutoff: number) {
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
      water: kpiFor(filtered, "waterAccess", countries, year),
      sanitation: kpiFor(filtered, "basicSanitation", countries, year),
      openDef: kpiFor(filtered, "openDefecation", countries, year),
      childMort: kpiFor(filtered, "childMortality", countries, year),
    }),
    [filtered, countries, year]
  );

  const bubbleValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "waterAccess", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const bubbleSizes = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "populationMillions", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const bubbleYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "waterAccess", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "waterAccess", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "waterAccess", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const scatterData = useMemo(() => {
    const sanitation = latestAtOrBefore(filtered, "basicSanitation", year);
    const childMort = latestAtOrBefore(filtered, "childMortality", year);
    return countries
      .map((iso3) => ({
        iso3,
        name: COUNTRY_NAMES[iso3],
        x: sanitation[iso3]?.value,
        y: childMort[iso3]?.value,
      }))
      .filter((d): d is { iso3: string; name: string; x: number; y: number } => d.x != null && d.y != null);
  }, [filtered, countries, year]);

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "waterAccess", year);
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
        if (rec && rec.waterAccess != null) row[iso3] = rec.waterAccess;
      });
      return row;
    });
  }, [countries, year]);

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "waterAccess", label: "Accès à l'eau", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "basicSanitation", label: "Assainissement", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "openDefecation", label: "Défécation air libre", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "childMortality", label: "Mortalité -5 ans", format: (v) => (v != null ? formatNumber(v as number, 0) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="wash-eau-assainissement" rows={tableRows} />
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
            <KpiCard label="Accès à l'eau potable" value={kpis.water.avg != null ? formatPercent(kpis.water.avg) : NO_DATA_LABEL} yoyChange={kpis.water.yoy} />
            <KpiCard label="Assainissement de base" value={kpis.sanitation.avg != null ? formatPercent(kpis.sanitation.avg) : NO_DATA_LABEL} yoyChange={kpis.sanitation.yoy} />
            <KpiCard label="Défécation à l'air libre" value={kpis.openDef.avg != null ? formatPercent(kpis.openDef.avg) : NO_DATA_LABEL} yoyChange={kpis.openDef.yoy} lowerIsBetter />
            <KpiCard label="Mortalité des moins de 5 ans" value={kpis.childMort.avg != null ? formatNumber(kpis.childMort.avg, 0) : NO_DATA_LABEL} unit={kpis.childMort.avg != null ? "/ 1 000" : undefined} yoyChange={kpis.childMort.yoy} lowerIsBetter />
          </div>
          <ChartCard title="Accès à l'eau potable" description="Carte à bulles : taille = population, couleur = intensité de l'accès (dernière donnée réelle disponible).">
            <BubbleMap values={bubbleValues} sizes={bubbleSizes} years={bubbleYears} valueSuffix=" %" sizeLabel="millions d'habitants" />
          </ChartCard>
          <InsightBox>
            <p>
              L'accès à l'eau potable a généralement progressé plus vite que l'assainissement de base, qui reste
              le maillon faible dans plusieurs pays du panel.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Accès à l'eau potable par pays" description="Carte interactive : zoom, survol et clic. Dernière valeur réelle disponible.">
          <ChoroplethMap values={mapValues} years={mapYears} valueSuffix=" %" polarity="positive" />
        </ChartCard>
      ),
    },
    {
      id: "comparison",
      label: "Comparaison (corrélation)",
      render: () => (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title="Assainissement vs mortalité infantile" description="Dernière valeur réelle disponible par pays sélectionné.">
            <ScatterChartCard data={scatterData} xLabel="Assainissement de base" yLabel="Mortalité -5 ans" xSuffix=" %" ySuffix=" / 1 000" />
          </ChartCard>
          <ChartCard title="Classement — accès à l'eau potable" description="Dernière valeur réelle disponible par pays.">
            <RankingBarChart data={rankingData} valueSuffix=" %" />
          </ChartCard>
        </div>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Évolution de l'accès à l'eau potable" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
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
