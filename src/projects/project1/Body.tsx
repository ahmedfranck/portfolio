import { useMemo, useState } from "react";
import rawData from "../../data/project1.json";
import type { MaternalRow } from "../../data/types";
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
import {
  latestAtOrBefore,
  averageDefined,
  averageYoYAcrossCountries,
  seriesUpTo,
  NO_DATA_LABEL,
} from "../../lib/realdata";
import { formatNumber, formatPercent } from "../../lib/format";

const ROWS = rawData.rows as MaternalRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["SEN", "CIV", "GHA", "MLI", "NGA", "COD"];

function kpiFor(rows: MaternalRow[], field: keyof MaternalRow, countries: string[], cutoff: number) {
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
      mmr: kpiFor(filtered, "maternalMortalityRatio", countries, year),
      anc: kpiFor(filtered, "prenatalCareCoverage", countries, year),
      sba: kpiFor(filtered, "skilledBirthAttendance", countries, year),
      neo: kpiFor(filtered, "neonatalMortality", countries, year),
    }),
    [filtered, countries, year]
  );

  const mapValues = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "maternalMortalityRatio", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.value ?? null]));
  }, [year]);
  const mapYears = useMemo(() => {
    const latest = latestAtOrBefore(ROWS, "maternalMortalityRatio", year);
    return Object.fromEntries(Object.entries(latest).map(([iso3, obs]) => [iso3, obs?.year ?? null]));
  }, [year]);

  const lineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      countries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.skilledBirthAttendance != null) row[iso3] = rec.skilledBirthAttendance;
      });
      return row;
    });
  }, [countries, year]);

  const rankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "prenatalCareCoverage", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const countryProfile = useMemo(
    () => ({
      mmr: { series: seriesUpTo(ROWS, "maternalMortalityRatio", selectedCountry, year), latest: latestAtOrBefore(ROWS, "maternalMortalityRatio", year)[selectedCountry]?.value ?? null },
      anc: { series: seriesUpTo(ROWS, "prenatalCareCoverage", selectedCountry, year), latest: latestAtOrBefore(ROWS, "prenatalCareCoverage", year)[selectedCountry]?.value ?? null },
      sba: { series: seriesUpTo(ROWS, "skilledBirthAttendance", selectedCountry, year), latest: latestAtOrBefore(ROWS, "skilledBirthAttendance", year)[selectedCountry]?.value ?? null },
      neo: { series: seriesUpTo(ROWS, "neonatalMortality", selectedCountry, year), latest: latestAtOrBefore(ROWS, "neonatalMortality", year)[selectedCountry]?.value ?? null },
    }),
    [selectedCountry, year]
  );

  const tableRows = useMemo(
    () => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })),
    [filtered]
  );

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "maternalMortalityRatio", label: "Mortalité maternelle", format: (v) => (v != null ? formatNumber(v as number, 0) : NO_DATA_LABEL) },
    { key: "prenatalCareCoverage", label: "Soins prénatals", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "skilledBirthAttendance", label: "Accouch. assistés", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "neonatalMortality", label: "Mortalité néonatale", format: (v) => (v != null ? formatNumber(v as number, 1) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="sante-maternelle-neonatale" rows={tableRows} />
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
            <KpiCard
              label="Ratio de mortalité maternelle"
              value={kpis.mmr.avg != null ? formatNumber(kpis.mmr.avg, 0) : NO_DATA_LABEL}
              unit={kpis.mmr.avg != null ? "/ 100 000" : undefined}
              yoyChange={kpis.mmr.yoy}
              lowerIsBetter
            />
            <KpiCard
              label="Soins prénatals (≥ 1 visite)"
              value={kpis.anc.avg != null ? formatPercent(kpis.anc.avg) : NO_DATA_LABEL}
              yoyChange={kpis.anc.yoy}
            />
            <KpiCard
              label="Accouchements assistés"
              value={kpis.sba.avg != null ? formatPercent(kpis.sba.avg) : NO_DATA_LABEL}
              yoyChange={kpis.sba.yoy}
            />
            <KpiCard
              label="Mortalité néonatale"
              value={kpis.neo.avg != null ? formatNumber(kpis.neo.avg, 1) : NO_DATA_LABEL}
              unit={kpis.neo.avg != null ? "/ 1 000" : undefined}
              yoyChange={kpis.neo.yoy}
              lowerIsBetter
            />
          </div>
          <ChartCard
            title="Classement — soins prénatals (≥ 1 visite)"
            description="Dernière valeur réelle disponible par pays sélectionné."
          >
            <RankingBarChart data={rankingData} valueSuffix=" %" />
          </ChartCard>
          <InsightBox>
            <p>
              Les pays affichant les plus forts taux d'accouchements assistés ne sont pas toujours ceux avec le
              ratio de mortalité maternelle le plus bas : d'autres déterminants (distance aux structures, qualité
              des soins) entrent en jeu.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard
          title="Mortalité maternelle par pays"
          description="Carte interactive : zoom, survol et clic pour ouvrir la fiche pays. Dernière valeur réelle disponible (au plus tard l'année sélectionnée)."
        >
          <ChoroplethMap
            values={mapValues}
            years={mapYears}
            valueSuffix=" / 100 000"
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
        <ChartCard
          title="Accouchements assistés par personnel qualifié"
          description="Évolution réelle 2010 → année sélectionnée, pour les pays sélectionnés."
        >
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
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <IndicatorMiniSeries label="Mortalité maternelle" unit="/ 100 000" series={countryProfile.mmr.series} latest={countryProfile.mmr.latest} formatValue={(n) => formatNumber(n, 0)} />
            <IndicatorMiniSeries label="Soins prénatals" unit="%" series={countryProfile.anc.series} latest={countryProfile.anc.latest} formatValue={(n) => formatNumber(n, 1)} />
            <IndicatorMiniSeries label="Accouch. assistés" unit="%" series={countryProfile.sba.series} latest={countryProfile.sba.latest} formatValue={(n) => formatNumber(n, 1)} />
            <IndicatorMiniSeries label="Mortalité néonatale" unit="/ 1 000" series={countryProfile.neo.series} latest={countryProfile.neo.latest} formatValue={(n) => formatNumber(n, 1)} />
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
