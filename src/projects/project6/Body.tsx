import { useMemo, useState } from "react";
import rawData from "../../data/project6.json";
import type { DisplacementRow } from "../../data/types";
import { COUNTRY_NAMES, MAX_YEAR } from "../../data/countries";
import YearSlider from "../../components/filters/YearSlider";
import CountrySelect from "../../components/filters/CountrySelect";
import KpiCard from "../../components/KpiCard";
import ChartCard from "../../components/ChartCard";
import ExportButton from "../../components/ExportButton";
import InsightBox from "../../components/InsightBox";
import DashboardTabs, { type DashboardTab } from "../../components/DashboardTabs";
import DataTable, { type DataTableColumn } from "../../components/DataTable";
import StackedAreaChart from "../../components/charts/StackedAreaChart";
import ChoroplethMap from "../../components/charts/ChoroplethMap";
import RankingBarChart from "../../components/charts/RankingBarChart";
import MultiLineChart from "../../components/charts/MultiLineChart";
import IndicatorMiniSeries from "../../components/charts/IndicatorMiniSeries";
import SourcesPanel, { type IndicatorSourceInfo } from "../../components/SourcesPanel";
import { latestAtOrBefore, seriesUpTo, NO_DATA_LABEL } from "../../lib/realdata";
import { formatCompact } from "../../lib/format";

const ROWS = rawData.rows as DisplacementRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const AVAILABLE_ISO3 = Array.from(new Set(ROWS.map((r) => r.iso3)));
const INITIAL_COUNTRY = "BFA";

function sumLatest(rows: DisplacementRow[], field: keyof DisplacementRow, countries: string[], cutoff: number) {
  const latest = latestAtOrBefore(rows, field, cutoff);
  const values = countries.map((c) => latest[c]?.value).filter((v): v is number => v != null);
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0);
}

export default function Body() {
  const [tab, setTab] = useState("overview");
  const [countryFilter, setCountryFilter] = useState<string>("ALL");
  const [year, setYear] = useState<number>(MAX_YEAR);
  const [selectedCountry, setSelectedCountry] = useState<string>(INITIAL_COUNTRY);

  const effectiveCountries = useMemo(
    () => (countryFilter === "ALL" ? AVAILABLE_ISO3 : [countryFilter]),
    [countryFilter]
  );
  const filtered = useMemo(
    () => (countryFilter === "ALL" ? ROWS : ROWS.filter((r) => r.iso3 === countryFilter)),
    [countryFilter]
  );

  const totals = useMemo(
    () => ({
      refugees: sumLatest(filtered, "refugees", effectiveCountries, year),
      refugeesOrigin: sumLatest(filtered, "refugeesOrigin", effectiveCountries, year),
      idp: sumLatest(filtered, "idp", effectiveCountries, year),
    }),
    [filtered, effectiveCountries, year]
  );

  const totalDisplaced = useMemo(() => {
    if (totals.refugees == null && totals.idp == null) return null;
    return (totals.refugees ?? 0) + (totals.idp ?? 0);
  }, [totals]);

  const flowData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const rows = filtered.filter((r) => r.year === y);
      const idp = rows.reduce((acc, r) => acc + (r.idp ?? 0), 0);
      const refugees = rows.reduce((acc, r) => acc + (r.refugees ?? 0), 0);
      return { year: y, idp, refugees };
    });
  }, [filtered, year]);

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "idp", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "idp", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const rankingData = useMemo(() => {
    const latestRefugees = latestAtOrBefore(filtered, "refugees", year);
    const latestIdp = latestAtOrBefore(filtered, "idp", year);
    return effectiveCountries
      .map((iso3) => {
        const r = latestRefugees[iso3]?.value;
        const i = latestIdp[iso3]?.value;
        if (r == null && i == null) return null;
        return { name: COUNTRY_NAMES[iso3], value: (r ?? 0) + (i ?? 0) };
      })
      .filter((d): d is { name: string; value: number } => d != null);
  }, [filtered, effectiveCountries, year]);

  const idpLineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      effectiveCountries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.idp != null) row[iso3] = rec.idp;
      });
      return row;
    });
  }, [effectiveCountries, year]);

  const countryProfile = useMemo(
    () => ({
      refugees: { series: seriesUpTo(ROWS, "refugees", selectedCountry, year), latest: latestAtOrBefore(ROWS, "refugees", year)[selectedCountry]?.value ?? null },
      refugeesOrigin: { series: seriesUpTo(ROWS, "refugeesOrigin", selectedCountry, year), latest: latestAtOrBefore(ROWS, "refugeesOrigin", year)[selectedCountry]?.value ?? null },
      idp: { series: seriesUpTo(ROWS, "idp", selectedCountry, year), latest: latestAtOrBefore(ROWS, "idp", year)[selectedCountry]?.value ?? null },
    }),
    [selectedCountry, year]
  );

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "refugees", label: "Réfugiés accueillis", format: (v) => (v != null ? formatCompact(v as number) : NO_DATA_LABEL) },
    { key: "refugeesOrigin", label: "Réfugiés originaires", format: (v) => (v != null ? formatCompact(v as number) : NO_DATA_LABEL) },
    { key: "idp", label: "Déplacés internes", format: (v) => (v != null ? formatCompact(v as number) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountrySelect value={countryFilter} onChange={setCountryFilter} countries={AVAILABLE_ISO3} allOption label="Pays" />
      <YearSlider year={year} onChange={setYear} label="Année maximale" />
      <div className="ml-auto">
        <ExportButton filename="populations-deplacees" rows={tableRows} />
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
            <KpiCard label="Réfugiés accueillis" value={totals.refugees != null ? formatCompact(totals.refugees) : NO_DATA_LABEL} lowerIsBetter />
            <KpiCard label="Réfugiés originaires" value={totals.refugeesOrigin != null ? formatCompact(totals.refugeesOrigin) : NO_DATA_LABEL} lowerIsBetter />
            <KpiCard label="Déplacés internes" value={totals.idp != null ? formatCompact(totals.idp) : NO_DATA_LABEL} lowerIsBetter />
            <KpiCard label="Total personnes déplacées" value={totalDisplaced != null ? formatCompact(totalDisplaced) : NO_DATA_LABEL} helpText="Réfugiés accueillis + déplacés internes" lowerIsBetter />
          </div>
          <ChartCard title="Déplacements cumulés — déplacés internes et réfugiés accueillis" description="Somme réelle des pays sélectionnés, 2010 → année sélectionnée.">
            <StackedAreaChart
              data={flowData}
              series={[
                { key: "idp", label: "Déplacés internes", color: "#9A7850" },
                { key: "refugees", label: "Réfugiés accueillis", color: "#29463A" },
              ]}
              valueSuffix=""
            />
          </ChartCard>
          <ChartCard title="Classement — total personnes déplacées" description="Réfugiés accueillis + déplacés internes, dernière valeur réelle disponible.">
            <RankingBarChart data={rankingData} valueSuffix=" pers." />
          </ChartCard>
          <InsightBox>
            <p>
              Le nombre de personnes déplacées est très concentré sur un nombre restreint de pays du panel, plutôt
              que réparti de façon homogène sur la région.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Intensité des déplacements internes" description="Carte interactive : zoom, survol et clic pour ouvrir la fiche pays. Dernière valeur réelle disponible.">
          <ChoroplethMap
            values={mapValues}
            years={mapYears}
            valueSuffix=" pers."
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
        <ChartCard title="Évolution des déplacés internes" description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés.">
          <MultiLineChart data={idpLineData} seriesCodes={effectiveCountries} valueSuffix=" pers." />
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
            <IndicatorMiniSeries label="Réfugiés accueillis" series={countryProfile.refugees.series} latest={countryProfile.refugees.latest} formatValue={formatCompact} />
            <IndicatorMiniSeries label="Réfugiés originaires" series={countryProfile.refugeesOrigin.series} latest={countryProfile.refugeesOrigin.latest} formatValue={formatCompact} />
            <IndicatorMiniSeries label="Déplacés internes" series={countryProfile.idp.series} latest={countryProfile.idp.latest} formatValue={formatCompact} />
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
