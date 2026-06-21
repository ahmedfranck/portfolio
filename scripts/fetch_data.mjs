/**
 * Récupère de VRAIES données ouvertes (API Banque mondiale — Open Data) pour les 16 pays
 * du portfolio, 2010–2024, et écrit un fichier JSON par projet dans src/data/.
 *
 * Exécution : node scripts/fetch_data.mjs
 * Nécessite un accès Internet. Les fichiers générés sont committés pour que le site
 * fonctionne ensuite hors-ligne (build statique).
 *
 * Aucune valeur manquante n'est fabriquée : un champ sans donnée réelle reste `null`.
 */
import { writeFileSync, mkdirSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "..", "src", "data");

const MIN_YEAR = 2010;
const MAX_YEAR = 2024;
const YEARS = Array.from({ length: MAX_YEAR - MIN_YEAR + 1 }, (_, i) => MIN_YEAR + i);

const COUNTRIES = [
  "BEN", "BFA", "CMR", "CIV", "GHA", "GIN", "LBR", "MLI",
  "MRT", "NER", "NGA", "COD", "SEN", "SLE", "TCD", "TGO",
];

const WB_BASE = "https://api.worldbank.org/v2/country";

const PROJECTS = {
  project1: {
    file: "project1.json",
    indicators: [
      { key: "maternalMortalityRatio", code: "SH.STA.MMRT", label: "Ratio de mortalité maternelle", unit: "pour 100 000 naissances vivantes", decimals: 0 },
      { key: "prenatalCareCoverage", code: "SH.STA.ANVC.ZS", label: "Femmes enceintes ayant reçu des soins prénatals", unit: "%", decimals: 1 },
      { key: "skilledBirthAttendance", code: "SH.STA.BRTC.ZS", label: "Accouchements assistés par personnel qualifié", unit: "%", decimals: 1 },
      { key: "neonatalMortality", code: "SH.DYN.NMRT", label: "Mortalité néonatale", unit: "pour 1 000 naissances vivantes", decimals: 1 },
    ],
  },
  project2: {
    file: "project2.json",
    indicators: [
      { key: "stunting", code: "SH.STA.STNT.ZS", label: "Retard de croissance (stunting)", unit: "%", decimals: 1 },
      { key: "wasting", code: "SH.STA.WAST.ZS", label: "Émaciation (wasting)", unit: "%", decimals: 1 },
      { key: "underweight", code: "SH.STA.MALN.ZS", label: "Insuffisance pondérale", unit: "%", decimals: 1 },
      { key: "dtp3Coverage", code: "SH.IMM.IDPT", label: "Couverture vaccinale DTC3", unit: "%", decimals: 1 },
      { key: "measlesCoverage", code: "SH.IMM.MEAS", label: "Couverture vaccinale rougeole", unit: "%", decimals: 1 },
      { key: "gdpPerCapita", code: "NY.GDP.PCAP.CD", label: "PIB par habitant", unit: "USD courants", decimals: 0 },
    ],
  },
  project3: {
    file: "project3.json",
    indicators: [
      { key: "enrollmentPrimaryFemale", code: "SE.PRM.ENRR.FE", label: "Scolarisation primaire, filles (brut)", unit: "%", decimals: 1 },
      { key: "enrollmentPrimaryMale", code: "SE.PRM.ENRR.MA", label: "Scolarisation primaire, garçons (brut)", unit: "%", decimals: 1 },
      { key: "enrollmentSecondaryFemale", code: "SE.SEC.ENRR.FE", label: "Scolarisation secondaire, filles (brut)", unit: "%", decimals: 1 },
      { key: "enrollmentSecondaryMale", code: "SE.SEC.ENRR.MA", label: "Scolarisation secondaire, garçons (brut)", unit: "%", decimals: 1 },
      { key: "parityPrimary", code: "SE.ENR.PRIM.FM.ZS", label: "Indice de parité filles/garçons, primaire (GPI)", unit: "ratio", decimals: 2 },
      { key: "literacyYoungWomen", code: "SE.ADT.1524.LT.FE.ZS", label: "Alphabétisation des jeunes femmes (15-24 ans)", unit: "%", decimals: 1 },
    ],
  },
  project4: {
    file: "project4.json",
    indicators: [
      { key: "mcpr", code: "SP.DYN.CONU.ZS", label: "Prévalence contraceptive, toutes méthodes", unit: "% des femmes mariées 15-49 ans", decimals: 1 },
      { key: "mcprModern", code: "SP.DYN.CONM.ZS", label: "Prévalence contraceptive, méthodes modernes", unit: "% des femmes mariées 15-49 ans", decimals: 1 },
      { key: "tfr", code: "SP.DYN.TFRT.IN", label: "Indice synthétique de fécondité", unit: "naissances par femme", decimals: 2 },
      { key: "demandSatisfied", code: "SH.FPL.SATM.ZS", label: "Demande satisfaite par méthodes modernes", unit: "% des femmes mariées avec demande", decimals: 1 },
    ],
  },
  project5: {
    file: "project5.json",
    indicators: [
      { key: "waterAccess", code: "SH.H2O.BASW.ZS", label: "Accès à l'eau potable de base", unit: "%", decimals: 1 },
      { key: "basicSanitation", code: "SH.STA.BASS.ZS", label: "Accès à l'assainissement de base", unit: "%", decimals: 1 },
      { key: "openDefecation", code: "SH.STA.ODFC.ZS", label: "Défécation à l'air libre", unit: "%", decimals: 1 },
      { key: "childMortality", code: "SH.DYN.MORT", label: "Mortalité des moins de 5 ans", unit: "pour 1 000 naissances vivantes", decimals: 1 },
      { key: "populationMillions", code: "SP.POP.TOTL", label: "Population totale", unit: "millions d'habitants", decimals: 2, transform: (v) => v / 1e6 },
    ],
  },
  project6: {
    file: "project6.json",
    indicators: [
      // SM.POP.REFG / SM.POP.REFG.OR / VC.IDP.TOCV sont archivés côté API (WDI Database Archives,
      // aucune donnée renvoyée) : remplacés par leurs équivalents actifs ci-dessous.
      { key: "refugees", code: "SM.POP.RHCR.EA", label: "Réfugiés sous mandat HCR (pays d'asile)", unit: "personnes", decimals: 0 },
      { key: "refugeesOrigin", code: "SM.POP.RHCR.EO", label: "Réfugiés sous mandat HCR (pays d'origine)", unit: "personnes", decimals: 0 },
      { key: "idp", code: "SM.POP.IDPC", label: "Déplacés internes (pays d'asile/origine)", unit: "personnes", decimals: 0 },
      { key: "populationMillions", code: "SP.POP.TOTL", label: "Population totale", unit: "millions d'habitants", decimals: 2, transform: (v) => v / 1e6 },
    ],
  },
  project7: {
    file: "project7.json",
    indicators: [
      { key: "gdpGrowth", code: "NY.GDP.MKTP.KD.ZG", label: "Croissance du PIB", unit: "% annuel", decimals: 1 },
      { key: "gdpPerCapita", code: "NY.GDP.PCAP.CD", label: "PIB par habitant", unit: "USD courants", decimals: 0 },
      { key: "inflation", code: "FP.CPI.TOTL.ZG", label: "Inflation des prix à la consommation", unit: "% annuel", decimals: 1 },
      { key: "fdiPercentGdp", code: "BX.KLT.DINV.WD.GD.ZS", label: "Investissements directs étrangers, entrants", unit: "% du PIB", decimals: 1 },
      // GC.DOD.TOTL.GD.ZS (dette publique/PIB) prévu au cahier des charges : couverture quasi
      // inexistante sur ces 16 pays (2/16, 7 observations) — exclu pour éviter une colonne vide.
    ],
  },
  project8: {
    file: "project8.json",
    indicators: [
      { key: "agricultureValueAdded", code: "NV.AGR.TOTL.ZS", label: "Valeur ajoutée agricole", unit: "% du PIB", decimals: 1 },
      { key: "cerealYield", code: "AG.YLD.CREL.KG", label: "Rendement céréalier", unit: "kg/hectare", decimals: 0 },
      { key: "foodProductionIndex", code: "AG.PRD.FOOD.XD", label: "Indice de production alimentaire", unit: "indice (2014-2016=100)", decimals: 1 },
      { key: "undernourishment", code: "SN.ITK.DEFC.ZS", label: "Prévalence de la sous-alimentation", unit: "% de la population", decimals: 1 },
      { key: "arableLand", code: "AG.LND.ARBL.ZS", label: "Terres arables", unit: "% de la surface du territoire", decimals: 1 },
    ],
  },
  project9: {
    file: "project9.json",
    indicators: [
      { key: "electricityAccess", code: "EG.ELC.ACCS.ZS", label: "Accès à l'électricité", unit: "% de la population", decimals: 1 },
      { key: "cleanCookingAccess", code: "EG.CFT.ACCS.ZS", label: "Accès à des combustibles de cuisson propres", unit: "% de la population", decimals: 1 },
      { key: "renewableShare", code: "EG.FEC.RNEW.ZS", label: "Part des énergies renouvelables", unit: "% de la consommation finale", decimals: 1 },
      { key: "electricityConsumption", code: "EG.USE.ELEC.KH.PC", label: "Consommation d'électricité par habitant", unit: "kWh/habitant", decimals: 0 },
    ],
  },
  project10: {
    file: "project10.json",
    indicators: [
      { key: "mobileSubscriptions", code: "IT.CEL.SETS.P2", label: "Abonnements à la téléphonie mobile", unit: "pour 100 habitants", decimals: 1 },
      { key: "internetUsers", code: "IT.NET.USER.ZS", label: "Utilisateurs d'Internet", unit: "% de la population", decimals: 1 },
      { key: "accountOwnership", code: "FX.OWN.TOTL.ZS", label: "Détention d'un compte (banque/mobile money)", unit: "% des 15 ans et plus", decimals: 1 },
      { key: "fixedBroadband", code: "IT.NET.BBND.P2", label: "Abonnements au haut débit fixe", unit: "pour 100 habitants", decimals: 2 },
    ],
  },
  project11: {
    file: "project11.json",
    indicators: [
      { key: "youthUnemployment", code: "SL.UEM.1524.ZS", label: "Chômage des jeunes (15-24 ans)", unit: "% de la population active 15-24 ans", decimals: 1 },
      { key: "totalUnemployment", code: "SL.UEM.TOTL.ZS", label: "Chômage total", unit: "% de la population active", decimals: 1 },
      { key: "vulnerableEmployment", code: "SL.EMP.VULN.ZS", label: "Emploi vulnérable", unit: "% de l'emploi total", decimals: 1 },
      { key: "laborForceParticipation", code: "SL.TLF.CACT.ZS", label: "Taux d'activité", unit: "% de la population 15+", decimals: 1 },
    ],
  },
  project12: {
    file: "project12.json",
    indicators: [
      // EN.ATM.CO2E.PC prévu au cahier des charges est archivé côté API (WDI Database Archives,
      // aucune donnée renvoyée) : remplacé par son équivalent actif ci-dessous.
      { key: "co2PerCapita", code: "EN.GHG.CO2.PC.CE.AR5", label: "Émissions de CO₂ par habitant", unit: "tonnes/habitant", decimals: 2 },
      { key: "forestArea", code: "AG.LND.FRST.ZS", label: "Couvert forestier", unit: "% de la surface du territoire", decimals: 1 },
      { key: "renewableWaterPerCapita", code: "ER.H2O.INTR.PC", label: "Ressources en eau douce renouvelables par habitant", unit: "m³/habitant", decimals: 0 },
      { key: "pm25Exposure", code: "EN.ATM.PM25.MC.M3", label: "Exposition aux particules fines (PM2.5)", unit: "µg/m³", decimals: 1 },
      { key: "protectedAreas", code: "ER.LND.PTLD.ZS", label: "Aires terrestres protégées", unit: "% de la surface du territoire", decimals: 1 },
    ],
  },
};

async function fetchIndicator(code) {
  const url = `${WB_BASE}/${COUNTRIES.join(";")}/indicator/${code}?format=json&date=${MIN_YEAR}:${MAX_YEAR}&per_page=20000`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} pour ${code}`);
  const json = await res.json();
  const records = json[1];
  if (!records) return [];
  return records
    .filter((r) => r.value != null)
    .map((r) => ({ iso3: r.countryiso3code, year: Number(r.date), value: r.value }));
}

function round(value, decimals) {
  const f = Math.pow(10, decimals);
  return Math.round(value * f) / f;
}

async function buildProject(projectKey, def) {
  const rows = new Map();
  for (const iso3 of COUNTRIES) {
    for (const year of YEARS) {
      rows.set(`${iso3}-${year}`, { iso3, year });
    }
  }

  const indicatorsMeta = [];

  for (const ind of def.indicators) {
    let records = [];
    try {
      records = await fetchIndicator(ind.code);
    } catch (err) {
      console.warn(`  ⚠ ${ind.code} (${ind.key}) — échec de récupération : ${err.message}`);
    }

    const countriesWithData = new Set();
    let minYear = Infinity;
    let maxYear = -Infinity;

    for (const rec of records) {
      const rowKey = `${rec.iso3}-${rec.year}`;
      const row = rows.get(rowKey);
      if (!row) continue; // pays ou année hors périmètre
      const raw = ind.transform ? ind.transform(rec.value) : rec.value;
      row[ind.key] = round(raw, ind.decimals);
      countriesWithData.add(rec.iso3);
      minYear = Math.min(minYear, rec.year);
      maxYear = Math.max(maxYear, rec.year);
    }

    // Initialise explicitement à null les cases sans donnée réelle (pas de fabrication).
    for (const row of rows.values()) {
      if (!(ind.key in row)) row[ind.key] = null;
    }

    indicatorsMeta.push({
      key: ind.key,
      code: ind.code,
      label: ind.label,
      unit: ind.unit,
      countriesWithData: countriesWithData.size,
      totalCountries: COUNTRIES.length,
      yearRange: countriesWithData.size > 0 ? [minYear, maxYear] : null,
    });

    console.log(
      `  ${ind.code.padEnd(20)} ${ind.key.padEnd(26)} ${countriesWithData.size}/${COUNTRIES.length} pays` +
        (countriesWithData.size > 0 ? ` · ${minYear}-${maxYear}` : " · aucune donnée")
    );
  }

  const output = {
    rows: Array.from(rows.values()).sort((a, b) => (a.iso3 < b.iso3 ? -1 : a.iso3 > b.iso3 ? 1 : a.year - b.year)),
    indicators: indicatorsMeta,
    fetchedAt: new Date().toISOString().slice(0, 10),
  };

  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(join(OUT_DIR, def.file), JSON.stringify(output, null, 2));
  console.log(`✓ ${def.file} écrit\n`);
}

async function main() {
  console.log("Récupération des données réelles — Banque mondiale (Open Data)\n");
  for (const [key, def] of Object.entries(PROJECTS)) {
    console.log(`▸ ${key} (${def.file})`);
    await buildProject(key, def);
  }
  console.log("Terminé. Toutes les données sont réelles ; les cases sans donnée disponible sont à `null`.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
