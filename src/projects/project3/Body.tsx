import { useMemo, useState } from "react";
import rawData from "../../data/project3.json";
import type { EducationRow } from "../../data/types";
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
import DivergingBarChart from "../../components/charts/DivergingBarChart";
import MultiLineChart from "../../components/charts/MultiLineChart";
import RankingBarChart from "../../components/charts/RankingBarChart";
import SourcesPanel, { type IndicatorSourceInfo } from "../../components/SourcesPanel";
import { latestAtOrBefore, averageDefined, averageYoYAcrossCountries, NO_DATA_LABEL } from "../../lib/realdata";
import { normalize } from "../../lib/indicators";
import { formatNumber, formatPercent } from "../../lib/format";

const ROWS = rawData.rows as EducationRow[];
const INDICATORS = rawData.indicators as IndicatorSourceInfo[];
const DEFAULT_COUNTRIES = ["BEN", "BFA", "CIV", "GHA", "MLI", "NER", "NGA", "SEN", "TCD", "TGO"];

function kpiFor(rows: EducationRow[], field: keyof EducationRow, countries: string[], cutoff: number) {
  const latest = latestAtOrBefore(rows, field, cutoff);
  const avg = averageDefined(countries.map((c) => latest[c]?.value));
  const rowsUpToCutoff = rows.filter((r) => r.year <= cutoff);
  const yoy = averageYoYAcrossCountries(rowsUpToCutoff, field, countries);
  return { avg, yoy };
}

/** Index dérivé (0-100) combinant parité de scolarisation secondaire et alphabétisation des jeunes femmes. */
function derivedProgressIndex(secF: number | null, secM: number | null, literacy: number | null): number | null {
  if (secF == null || secM == null || secM === 0 || literacy == null) return null;
  const parityScore = normalize(secF / secM, 0.5, 1.1);
  const literacyScore = normalize(literacy, 30, 95);
  return averageDefined([parityScore, literacyScore]);
}

export default function Body() {
  const [tab, setTab] = useState("overview");
  const [countries, setCountries] = useState<string[]>(DEFAULT_COUNTRIES);
  const [year, setYear] = useState<number>(MAX_YEAR);

  const filtered = useMemo(() => ROWS.filter((r) => countries.includes(r.iso3)), [countries]);

  const kpis = useMemo(
    () => ({
      parityPrimary: kpiFor(filtered, "parityPrimary", countries, year),
      secF: kpiFor(filtered, "enrollmentSecondaryFemale", countries, year),
      literacy: kpiFor(filtered, "literacyYoungWomen", countries, year),
    }),
    [filtered, countries, year]
  );

  const indexByCountry = useMemo(() => {
    const secF = latestAtOrBefore(filtered, "enrollmentSecondaryFemale", year);
    const secM = latestAtOrBefore(filtered, "enrollmentSecondaryMale", year);
    const literacy = latestAtOrBefore(filtered, "literacyYoungWomen", year);
    return countries.map((iso3) => ({
      iso3,
      value: derivedProgressIndex(secF[iso3]?.value ?? null, secM[iso3]?.value ?? null, literacy[iso3]?.value ?? null),
    }));
  }, [filtered, countries, year]);

  const derivedAvg = useMemo(() => averageDefined(indexByCountry.map((d) => d.value)), [indexByCountry]);

  const mapValues = useMemo(
    () => Object.fromEntries(indexByCountry.map((d) => [d.iso3, d.value != null ? Math.round(d.value) : null])),
    [indexByCountry]
  );

  const divergingData = useMemo(() => {
    const secF = latestAtOrBefore(filtered, "enrollmentSecondaryFemale", year);
    const secM = latestAtOrBefore(filtered, "enrollmentSecondaryMale", year);
    return countries
      .map((iso3) => {
        const f = secF[iso3]?.value;
        const m = secM[iso3]?.value;
        return f != null && m != null ? { name: COUNTRY_NAMES[iso3], value: Number((f - m).toFixed(1)) } : null;
      })
      .filter((d): d is { name: string; value: number } => d != null);
  }, [filtered, countries, year]);

  const parityRankingData = useMemo(() => {
    const latest = latestAtOrBefore(filtered, "parityPrimary", year);
    return countries
      .map((iso3) => ({ name: COUNTRY_NAMES[iso3], value: latest[iso3]?.value }))
      .filter((d): d is { name: string; value: number } => d.value != null);
  }, [filtered, countries, year]);

  const literacyLineData = useMemo(() => {
    const years = Array.from({ length: year - 2010 + 1 }, (_, i) => 2010 + i);
    return years.map((y) => {
      const row: Record<string, number | string> = { year: y };
      countries.forEach((iso3) => {
        const rec = ROWS.find((r) => r.iso3 === iso3 && r.year === y);
        if (rec && rec.literacyYoungWomen != null) row[iso3] = rec.literacyYoungWomen;
      });
      return row;
    });
  }, [countries, year]);

  const rankingData = useMemo(
    () =>
      indexByCountry
        .map((d) => ({ name: COUNTRY_NAMES[d.iso3], value: d.value }))
        .filter((d): d is { name: string; value: number } => d.value != null)
        .map((d) => ({ name: d.name, value: Math.round(d.value) })),
    [indexByCountry]
  );

  const tableRows = useMemo(() => filtered.map((r) => ({ ...r, pays: COUNTRY_NAMES[r.iso3] })), [filtered]);

  const columns: DataTableColumn<(typeof tableRows)[number]>[] = [
    { key: "pays", label: "Pays" },
    { key: "year", label: "Année" },
    { key: "parityPrimary", label: "Parité primaire (GPI)", format: (v) => (v != null ? formatNumber(v as number, 2) : NO_DATA_LABEL) },
    { key: "enrollmentSecondaryFemale", label: "Scol. secondaire F", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "enrollmentSecondaryMale", label: "Scol. secondaire M", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
    { key: "literacyYoungWomen", label: "Alphabétisation 15-24F", format: (v) => (v != null ? formatPercent(v as number) : NO_DATA_LABEL) },
  ];

  const filterBar = (
    <div className="flex flex-wrap items-end gap-4 rounded-card border border-line bg-white p-4 shadow-card">
      <CountryMultiSelect selected={countries} onChange={setCountries} />
      <YearSlider year={year} onChange={setYear} label="Données jusqu'à l'année" />
      <div className="ml-auto">
        <ExportButton filename="education-filles-genre" rows={tableRows} />
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
            <KpiCard label="Indice de parité primaire (GPI)" value={kpis.parityPrimary.avg != null ? formatNumber(kpis.parityPrimary.avg, 2) : NO_DATA_LABEL} yoyChange={kpis.parityPrimary.yoy} helpText="1 = parité filles/garçons" />
            <KpiCard label="Scolarisation secondaire, filles" value={kpis.secF.avg != null ? formatPercent(kpis.secF.avg) : NO_DATA_LABEL} yoyChange={kpis.secF.yoy} />
            <KpiCard label="Alphabétisation jeunes femmes" value={kpis.literacy.avg != null ? formatPercent(kpis.literacy.avg) : NO_DATA_LABEL} yoyChange={kpis.literacy.yoy} />
            <KpiCard label="Index dérivé de progrès" value={derivedAvg != null ? formatNumber(derivedAvg, 0) : NO_DATA_LABEL} unit={derivedAvg != null ? "/ 100" : undefined} helpText="Parité secondaire + alphabétisation (normalisées)" />
          </div>
          <ChartCard title="Classement — index dérivé de progrès" description="Combine parité de scolarisation secondaire et alphabétisation des jeunes femmes (voir méthodologie).">
            <RankingBarChart data={rankingData} valueSuffix=" / 100" />
          </ChartCard>
          <InsightBox>
            <p>
              La parité est largement atteinte ou dépassée au primaire dans plusieurs pays du panel, mais l'écart
              se creuse souvent au secondaire.
            </p>
          </InsightBox>
        </div>
      ),
    },
    {
      id: "comparison",
      label: "Comparaison (écart F/M)",
      render: () => (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title="Écart de scolarisation secondaire filles–garçons" description="Valeurs négatives : les filles sont en retrait. Dernière valeur réelle disponible par pays.">
            <DivergingBarChart data={divergingData} valueSuffix=" pts" />
          </ChartCard>
          <ChartCard title="Classement — parité primaire (GPI)" description="Dernière valeur réelle disponible par pays sélectionné.">
            <RankingBarChart data={parityRankingData} valueSuffix="" />
          </ChartCard>
        </div>
      ),
    },
    {
      id: "trends",
      label: "Tendances",
      render: () => (
        <ChartCard title="Alphabétisation des jeunes femmes (15-24 ans)" description="Évolution réelle 2010 → année sélectionnée.">
          <MultiLineChart data={literacyLineData} seriesCodes={countries} valueSuffix=" %" />
        </ChartCard>
      ),
    },
    {
      id: "map",
      label: "Carte",
      render: () => (
        <ChartCard title="Index dérivé de progrès par pays" description="Carte interactive : zoom, survol et clic. Combine parité secondaire et alphabétisation (voir méthodologie).">
          <ChoroplethMap values={mapValues} valueSuffix=" / 100" unit="" polarity="positive" />
        </ChartCard>
      ),
    },
    {
      id: "data",
      label: "Données",
      render: () => (
        <div className="space-y-4">
          <DataTable rows={tableRows} columns={columns} searchableKey="pays" />
          <SourcesPanel
            indicators={INDICATORS}
            note="L'index dérivé de progrès est calculé par ce portfolio (non officiel) : il moyenne deux sous-scores normalisés (0–100) — la parité de scolarisation secondaire (filles ÷ garçons) et l'alphabétisation des jeunes femmes — à partir des dernières données disponibles pour chaque indicateur, qui peuvent provenir d'années légèrement différentes selon les pays."
          />
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
