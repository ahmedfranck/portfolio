import { useMemo, useState } from "react";
import rawData from "../../data/project8.json";
import type { AgricultureRow } from "../../data/types";
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

const ROWS = rawData.rows as AgricultureRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["BEN", "BFA", "CIV", "GHA", "MLI", "NER", "NGA", "SEN", "TCD", "TGO"];

function kpiFor(rows: AgricultureRow[], field: keyof AgricultureRow, countries: string[], cutoff: number) {
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
      agri: kpiFor(filtered, "agricultureValueAdded", countries, year),
      yield: kpiFor(filtered, "cerealYield", countries, year),
      foodIndex: kpiFor(filtered, "foodProductionIndex", countries, year),
      undernourishment: kpiFor(filtered, "undernourishment", countries, year),
    }),
    [filtered, countries, year]
  );

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "undernourishment", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "undernourishment", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "undernourishment", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const lineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      countries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.cerealYield != null) row[iso3] = rec.cerealYield;
      });
      return row;
    });
  }, [countries, year]);

  const countryProfile = useMemo(
    () => ({
      agri: { series: seriesUpTo(ROWS, "agricultureValueAdded", selectedCountry, year), latest: latestAtOrBefore(ROWS, "agricultureValueAdded", year)[selectedCountry]?.value ?? null },
      yield: { series: seriesUpTo(ROWS, "cerealYield", selectedCountry, year), latest: latestAtOrBefore(ROWS, "cerealYield", year)[selectedCountry]?.value ?? null },
      foodIndex: { series: seriesUpTo(ROWS, "foodProductionIndex", selectedCountry, year), latest: latestAtOrBefore(ROWS, "foodProductionIndex", year)[selectedCountry]?.value ?? null },
      undernourishment: { series: seriesUpTo(ROWS, "undernourishment", selectedCountry, year), latest: latestAtOrBefore(ROWS, "undernourishment", year)[selectedCountry]?.value ?? null },
      arableLand: { series: seriesUpTo(ROWS, "arableLand", selectedCountry, year), latest: latestAtOrBefore(ROWS, "arableLand", year)[selectedCountry]?.value ?? null },
    }),
    [selectedCountry, year]
  );

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "agricultureValueAdded", label: "Valeur ajoutée agricole", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "cerealYield", label: "Rendement céréalier", format: (v) => (v != null ? formatNumber(v as number, 0) : NO_DATA_LABEL) },
    { key: "foodProductionIndex", label: "Indice production alim.", format: (v) => (v != null ? formatNumber(v as number, 1) : NO_DATA_LABEL) },
    { key: "undernourishment", label: "Sous-alimentation", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "arableLand", label: "Terres arables", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="agriculture-securite-alimentaire" rows={tableRows} />
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
            <KpiCard label="Valeur ajoutée agricole" value={kpis.agri.avg != null ? formatPercent(kpis.agri.avg) : NO_DATA_LABEL} helpText="% du PIB" yoyChange={kpis.agri.yoy} />
            <KpiCard label="Rendement céréalier" value={kpis.yield.avg != null ? formatNumber(kpis.yield.avg, 0) : NO_DATA_LABEL} unit={kpis.yield.avg != null ? "kg/ha" : undefined} yoyChange={kpis.yield.yoy} />
            <KpiCard label="Indice de production alimentaire" value={kpis.foodIndex.avg != null ? formatNumber(kpis.foodIndex.avg, 1) : NO_DATA_LABEL} yoyChange={kpis.foodIndex.yoy} />
            <KpiCard label="Sous-alimentation" value={kpis.undernourishment.avg != null ? formatPercent(kpis.undernourishment.avg) : NO_DATA_LABEL} yoyChange={kpis.undernourishment.yoy} lowerIsBetter />
          </div>
          <ChartCard title="Classement — prévalence de la sous-alimentation" description="Dernière valeur réelle disponible par pays sélectionné.">
            <RankingBarChart data={rankingData} valueSuffix=" %" />
          </ChartCard>
          <InsightBox>
            <p>
              La prévalence de la sous-alimentation ne suit pas toujours l'évolution de l'indice de production
              alimentaire : la disponibilité globale ne garantit pas l'accès de tous aux aliments.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Sous-alimentation par pays" description="Carte interactive : zoom, survol et clic pour ouvrir la fiche pays. Dernière valeur réelle disponible.">
          <ChoroplethMap
            values={mapValues}
            years={mapYears}
            valueSuffix=" %"
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
        <ChartCard title="Évolution du rendement céréalier" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
          <MultiLineChart data={lineData} seriesCodes={countries} valueSuffix=" kg/ha" />
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
            <IndicatorMiniSeries label="Valeur ajoutée agricole" unit="%" series={countryProfile.agri.series} latest={countryProfile.agri.latest} formatValue={(n) => formatNumber(n, 1)} />
            <IndicatorMiniSeries label="Rendement céréalier" unit="kg/ha" series={countryProfile.yield.series} latest={countryProfile.yield.latest} formatValue={(n) => formatNumber(n, 0)} />
            <IndicatorMiniSeries label="Indice production alim." series={countryProfile.foodIndex.series} latest={countryProfile.foodIndex.latest} formatValue={(n) => formatNumber(n, 1)} />
            <IndicatorMiniSeries label="Sous-alimentation" unit="%" series={countryProfile.undernourishment.series} latest={countryProfile.undernourishment.latest} formatValue={(n) => formatNumber(n, 1)} />
            <IndicatorMiniSeries label="Terres arables" unit="%" series={countryProfile.arableLand.series} latest={countryProfile.arableLand.latest} formatValue={(n) => formatNumber(n, 1)} />
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
