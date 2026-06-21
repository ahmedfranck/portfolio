import { useMemo, useState } from "react";
import rawData from "../../data/project10.json";
import type { DigitalInclusionRow } from "../../data/types";
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

const ROWS = rawData.rows as DigitalInclusionRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["SEN", "CIV", "GHA", "NGA", "MLI", "COD"];

function kpiFor(rows: DigitalInclusionRow[], field: keyof DigitalInclusionRow, countries: string[], cutoff: number) {
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
      mobile: kpiFor(filtered, "mobileSubscriptions", countries, year),
      internet: kpiFor(filtered, "internetUsers", countries, year),
      account: kpiFor(filtered, "accountOwnership", countries, year),
      broadband: kpiFor(filtered, "fixedBroadband", countries, year),
    }),
    [filtered, countries, year]
  );

  const lineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      countries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.internetUsers != null) row[iso3] = rec.internetUsers;
      });
      return row;
    });
  }, [countries, year]);

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "internetUsers", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "internetUsers", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "internetUsers", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const scatterData = useMemo(() => {
    const mobile = latestAtOrBefore(filtered, "mobileSubscriptions", year);
    const internet = latestAtOrBefore(filtered, "internetUsers", year);
    return countries
      .map((iso3) => ({
        iso3,
        name: COUNTRY_NAMES[iso3],
        x: mobile[iso3]?.value,
        y: internet[iso3]?.value,
      }))
      .filter((d): d is { iso3: string; name: string; x: number; y: number } => d.x != null && d.y != null);
  }, [filtered, countries, year]);

  const accountRankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "accountOwnership", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "mobileSubscriptions", label: "Abonnements mobiles", format: (v) => (v != null ? formatNumber(v as number, 1) : NO_DATA_LABEL) },
    { key: "internetUsers", label: "Internautes", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "accountOwnership", label: "Détention de compte", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "fixedBroadband", label: "Haut débit fixe", format: (v) => (v != null ? formatNumber(v as number, 2) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="inclusion-numerique-financiere" rows={tableRows} />
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
            <KpiCard label="Abonnements mobiles" value={kpis.mobile.avg != null ? formatNumber(kpis.mobile.avg, 1) : NO_DATA_LABEL} unit={kpis.mobile.avg != null ? "/ 100 hab." : undefined} yoyChange={kpis.mobile.yoy} />
            <KpiCard label="Internautes" value={kpis.internet.avg != null ? formatPercent(kpis.internet.avg) : NO_DATA_LABEL} yoyChange={kpis.internet.yoy} />
            <KpiCard label="Détention de compte" value={kpis.account.avg != null ? formatPercent(kpis.account.avg) : NO_DATA_LABEL} helpText="Banque ou mobile money, données Findex éparses" yoyChange={kpis.account.yoy} />
            <KpiCard label="Haut débit fixe" value={kpis.broadband.avg != null ? formatNumber(kpis.broadband.avg, 2) : NO_DATA_LABEL} unit={kpis.broadband.avg != null ? "/ 100 hab." : undefined} yoyChange={kpis.broadband.yoy} />
          </div>
          <ChartCard title="Classement — utilisateurs d'Internet" description="Dernière valeur réelle disponible par pays sélectionné.">
            <RankingBarChart data={rankingData} valueSuffix=" %" />
          </ChartCard>
          <InsightBox>
            <p>
              Les abonnements à la téléphonie mobile dépassent largement le taux d'utilisation d'Internet dans la
              plupart des pays, signe d'un potentiel de croissance numérique encore important.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Évolution des utilisateurs d'Internet" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
          <MultiLineChart data={lineData} seriesCodes={countries} valueSuffix=" %" />
        </ChartCard>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Utilisateurs d'Internet par pays" description="Carte interactive : zoom, survol et clic. Dernière valeur réelle disponible.">
          <ChoroplethMap values={mapValues} years={mapYears} valueSuffix=" %" polarity="positive" />
        </ChartCard>
      ),
    },
    {
      id: "comparison",
      label: "Comparaison",
      render: () => (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title="Abonnements mobiles vs utilisateurs d'Internet" description="Dernière valeur réelle disponible par pays sélectionné.">
            <ScatterChartCard data={scatterData} xLabel="Abonnements mobiles" yLabel="Internautes" xSuffix=" / 100 hab." ySuffix=" %" />
          </ChartCard>
          <ChartCard title="Classement — détention de compte" description="Dernière valeur réelle disponible par pays (données Findex).">
            <RankingBarChart data={accountRankingData} valueSuffix=" %" />
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
