import { useMemo, useState } from "react";
import rawData from "../../data/project11.json";
import type { EmploymentRow } from "../../data/types";
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
import { formatPercent } from "../../lib/format";

const ROWS = rawData.rows as EmploymentRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["BEN", "BFA", "CIV", "GHA", "MLI", "NGA", "SEN", "TGO"];

function kpiFor(rows: EmploymentRow[], field: keyof EmploymentRow, countries: string[], cutoff: number) {
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
      youth: kpiFor(filtered, "youthUnemployment", countries, year),
      total: kpiFor(filtered, "totalUnemployment", countries, year),
      vulnerable: kpiFor(filtered, "vulnerableEmployment", countries, year),
      participation: kpiFor(filtered, "laborForceParticipation", countries, year),
    }),
    [filtered, countries, year]
  );

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "youthUnemployment", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const scatterData = useMemo(() => {
    const vulnerable = latestAtOrBefore(filtered, "vulnerableEmployment", year);
    const youth = latestAtOrBefore(filtered, "youthUnemployment", year);
    return countries
      .map((iso3) => ({
        iso3,
        name: COUNTRY_NAMES[iso3],
        x: youth[iso3]?.value,
        y: vulnerable[iso3]?.value,
      }))
      .filter((d): d is { iso3: string; name: string; x: number; y: number } => d.x != null && d.y != null);
  }, [filtered, countries, year]);

  const participationRankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "laborForceParticipation", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "youthUnemployment", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "youthUnemployment", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const lineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      countries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.youthUnemployment != null) row[iso3] = rec.youthUnemployment;
      });
      return row;
    });
  }, [countries, year]);

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "youthUnemployment", label: "Chômage jeunes", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "totalUnemployment", label: "Chômage total", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "vulnerableEmployment", label: "Emploi vulnérable", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "laborForceParticipation", label: "Taux d'activité", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="emploi-jeunesse" rows={tableRows} />
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
            <KpiCard label="Chômage des jeunes (15-24 ans)" value={kpis.youth.avg != null ? formatPercent(kpis.youth.avg) : NO_DATA_LABEL} yoyChange={kpis.youth.yoy} lowerIsBetter />
            <KpiCard label="Chômage total" value={kpis.total.avg != null ? formatPercent(kpis.total.avg) : NO_DATA_LABEL} yoyChange={kpis.total.yoy} lowerIsBetter />
            <KpiCard label="Emploi vulnérable" value={kpis.vulnerable.avg != null ? formatPercent(kpis.vulnerable.avg) : NO_DATA_LABEL} yoyChange={kpis.vulnerable.yoy} lowerIsBetter />
            <KpiCard label="Taux d'activité" value={kpis.participation.avg != null ? formatPercent(kpis.participation.avg) : NO_DATA_LABEL} yoyChange={kpis.participation.yoy} />
          </div>
          <ChartCard title="Classement — chômage des jeunes" description="Dernière valeur réelle disponible par pays sélectionné.">
            <RankingBarChart data={rankingData} valueSuffix=" %" />
          </ChartCard>
          <InsightBox>
            <p>
              Le chômage des jeunes est presque systématiquement supérieur au chômage total, traduisant des
              difficultés d'insertion spécifiques à cette tranche d'âge.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "comparison",
      label: "Comparaison",
      render: () => (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title="Chômage des jeunes vs emploi vulnérable" description="Dernière valeur réelle disponible par pays sélectionné.">
            <ScatterChartCard data={scatterData} xLabel="Chômage des jeunes" yLabel="Emploi vulnérable" xSuffix=" %" ySuffix=" %" />
          </ChartCard>
          <ChartCard title="Classement — taux d'activité" description="Dernière valeur réelle disponible par pays.">
            <RankingBarChart data={participationRankingData} valueSuffix=" %" />
          </ChartCard>
        </div>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Chômage des jeunes par pays" description="Carte interactive : zoom, survol et clic. Dernière valeur réelle disponible.">
          <ChoroplethMap values={mapValues} years={mapYears} valueSuffix=" %" polarity="negative" />
        </ChartCard>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Évolution du chômage des jeunes" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
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
