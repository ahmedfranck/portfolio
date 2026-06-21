import { useMemo, useState } from "react";
import rawData from "../../data/project12.json";
import type { ClimateRow } from "../../data/types";
import { COUNTRY_NAMES, MAX_YEAR } from "../../data/countries";
import CountryMultiSelect from "../../components/filters/CountryMultiSelect";
import YearSlider from "../../components/filters/YearSlider";
import CountrySelect from "../../components/filters/CountrySelect";
import KpiCard from "../../components/KpiCard";
import ChartCard from "../../components/ChartCard";
import ExportButton from "../../components/ExportButton";
import InsightBox from "../../components/InsightBox";
import DashboardTabs, { type DashboardTab } from "../../components/DashboardTabs";
import DataTable, { type DataTableColumn } from "../../components/DataTable";
import ChoroplethMap from "../../components/charts/ChoroplethMap";
import MultiLineChart from "../../components/charts/MultiLineChart";
import RankingBarChart from "../../components/charts/RankingBarChart";
import IndicatorMiniSeries from "../../components/charts/IndicatorMiniSeries";
import SourcesPanel, { type IndicatorSourceInfo } from "../../components/SourcesPanel";
import { latestAtOrBefore, averageDefined, averageYoYAcrossCountries, seriesUpTo, NO_DATA_LABEL } from "../../lib/realdata";
import { formatNumber, formatPercent } from "../../lib/format";

const ROWS = rawData.rows as ClimateRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["BEN", "BFA", "CMR", "CIV", "GHA", "MLI", "NGA", "COD", "SEN", "TCD"];

function kpiFor(rows: ClimateRow[], field: keyof ClimateRow, countries: string[], cutoff: number) {
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
  const [selectedCountry, setSelectedCountry] = useState<string>(DEFAULT_COUNTRIES[0]);

  const filtered = useMemo(() => ROWS.filter((r) => countries.includes(r.iso3)), [countries]);

  const kpis = useMemo(
    () => ({
      co2: kpiFor(filtered, "co2PerCapita", countries, year),
      forest: kpiFor(filtered, "forestArea", countries, year),
      pm25: kpiFor(filtered, "pm25Exposure", countries, year),
      protected: kpiFor(filtered, "protectedAreas", countries, year),
    }),
    [filtered, countries, year]
  );

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "forestArea", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "co2PerCapita", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "co2PerCapita", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const lineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      countries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.forestArea != null) row[iso3] = rec.forestArea;
      });
      return row;
    });
  }, [countries, year]);

  const countryProfile = useMemo(
    () => ({
      co2: { series: seriesUpTo(ROWS, "co2PerCapita", selectedCountry, year), latest: latestAtOrBefore(ROWS, "co2PerCapita", year)[selectedCountry]?.value ?? null },
      forest: { series: seriesUpTo(ROWS, "forestArea", selectedCountry, year), latest: latestAtOrBefore(ROWS, "forestArea", year)[selectedCountry]?.value ?? null },
      water: { series: seriesUpTo(ROWS, "renewableWaterPerCapita", selectedCountry, year), latest: latestAtOrBefore(ROWS, "renewableWaterPerCapita", year)[selectedCountry]?.value ?? null },
      pm25: { series: seriesUpTo(ROWS, "pm25Exposure", selectedCountry, year), latest: latestAtOrBefore(ROWS, "pm25Exposure", year)[selectedCountry]?.value ?? null },
      protected: { series: seriesUpTo(ROWS, "protectedAreas", selectedCountry, year), latest: latestAtOrBefore(ROWS, "protectedAreas", year)[selectedCountry]?.value ?? null },
    }),
    [selectedCountry, year]
  );

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "co2PerCapita", label: "CO₂ / habitant", format: (v) => (v != null ? formatNumber(v as number, 2) : NO_DATA_LABEL) },
    { key: "forestArea", label: "Couvert forestier", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "renewableWaterPerCapita", label: "Eau renouvelable / hab.", format: (v) => (v != null ? formatNumber(v as number, 0) : NO_DATA_LABEL) },
    { key: "pm25Exposure", label: "Exposition PM2.5", format: (v) => (v != null ? formatNumber(v as number, 1) : NO_DATA_LABEL) },
    { key: "protectedAreas", label: "Aires protégées", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="climat-environnement" rows={tableRows} />
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
            <KpiCard label="Émissions de CO₂ / habitant" value={kpis.co2.avg != null ? formatNumber(kpis.co2.avg, 2) : NO_DATA_LABEL} unit={kpis.co2.avg != null ? "t/hab." : undefined} yoyChange={kpis.co2.yoy} lowerIsBetter />
            <KpiCard label="Couvert forestier" value={kpis.forest.avg != null ? formatPercent(kpis.forest.avg) : NO_DATA_LABEL} yoyChange={kpis.forest.yoy} />
            <KpiCard label="Exposition aux PM2.5" value={kpis.pm25.avg != null ? formatNumber(kpis.pm25.avg, 1) : NO_DATA_LABEL} unit={kpis.pm25.avg != null ? "µg/m³" : undefined} yoyChange={kpis.pm25.yoy} lowerIsBetter />
            <KpiCard label="Aires protégées" value={kpis.protected.avg != null ? formatPercent(kpis.protected.avg) : NO_DATA_LABEL} yoyChange={kpis.protected.yoy} />
          </div>
          <ChartCard title="Classement — couvert forestier" description="Dernière valeur réelle disponible par pays sélectionné.">
            <RankingBarChart data={rankingData} valueSuffix=" %" />
          </ChartCard>
          <InsightBox>
            <p>
              Le couvert forestier recule dans plusieurs pays sur la période, en lien avec la pression
              démographique et agricole.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Émissions de CO₂ par habitant par pays" description="Carte interactive : zoom, survol et clic pour ouvrir la fiche pays. Dernière valeur réelle disponible.">
          <ChoroplethMap
            values={mapValues}
            years={mapYears}
            valueSuffix=" t/hab."
            polarity="negative"
            selectedIso3={selectedCountry}
            onSelectCountry={(iso3) => {
              setSelectedCountry(iso3);
              setTab("country");
            }}
          />
        </ChartCard>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Évolution du couvert forestier" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
          <MultiLineChart data={lineData} seriesCodes={countries} valueSuffix=" %" />
        </ChartCard>
      ),
    },
    {
      id: "country",
      label: "Fiche pays",
      render: () => (
        <div className="space-y-4">
          <CountrySelect value={selectedCountry} onChange={setSelectedCountry} label="Pays sélectionné" />
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <IndicatorMiniSeries label="CO₂ / habitant" unit="t/hab." series={countryProfile.co2.series} latest={countryProfile.co2.latest} formatValue={(n) => formatNumber(n, 2)} />
            <IndicatorMiniSeries label="Couvert forestier" unit="%" series={countryProfile.forest.series} latest={countryProfile.forest.latest} formatValue={(n) => formatNumber(n, 1)} />
            <IndicatorMiniSeries label="Eau renouvelable / hab." unit="m³" series={countryProfile.water.series} latest={countryProfile.water.latest} formatValue={(n) => formatNumber(n, 0)} />
            <IndicatorMiniSeries label="Exposition PM2.5" unit="µg/m³" series={countryProfile.pm25.series} latest={countryProfile.pm25.latest} formatValue={(n) => formatNumber(n, 1)} />
            <IndicatorMiniSeries label="Aires protégées" unit="%" series={countryProfile.protected.series} latest={countryProfile.protected.latest} formatValue={(n) => formatNumber(n, 1)} />
          </div>
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
