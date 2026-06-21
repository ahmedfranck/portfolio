// Génère une fois src/data/geo/africa.json : GeoJSON simplifié du continent africain,
// avec les codes ISO3 en propriété, à partir de world-atlas (Natural Earth 110m).
// Pas de fetch à l'exécution : ce script est exécuté manuellement, le résultat est commité.
import { feature } from "topojson-client";
import world from "world-atlas/countries-110m.json" with { type: "json" };
import { writeFileSync } from "node:fs";

// Numeric ISO 3166-1 id (topojson) -> ISO3.
const NUMERIC_TO_ISO3 = {
  "012": "DZA", "024": "AGO", "204": "BEN", "072": "BWA", "854": "BFA", "108": "BDI",
  "120": "CMR", "140": "CAF", "148": "TCD", "178": "COG", "384": "CIV", "180": "COD",
  "262": "DJI", "818": "EGY", "226": "GNQ", "232": "ERI", "748": "SWZ", "231": "ETH",
  "266": "GAB", "270": "GMB", "288": "GHA", "324": "GIN", "624": "GNB", "404": "KEN",
  "426": "LSO", "430": "LBR", "434": "LBY", "450": "MDG", "454": "MWI", "466": "MLI",
  "478": "MRT", "504": "MAR", "508": "MOZ", "516": "NAM", "562": "NER", "566": "NGA",
  "646": "RWA", "728": "SSD", "686": "SEN", "694": "SLE", "706": "SOM", "710": "ZAF",
  "729": "SDN", "834": "TZA", "768": "TGO", "788": "TUN", "800": "UGA", "732": "ESH",
  "894": "ZMB", "716": "ZWE",
};

const PORTFOLIO_ISO3 = new Set([
  "BEN", "BFA", "CMR", "CIV", "GHA", "GIN", "LBR", "MLI", "MRT", "NER", "NGA", "COD",
  "SEN", "SLE", "TCD", "TGO",
]);

const geo = feature(world, world.objects.countries);

const features = geo.features
  .map((f) => {
    const iso3 = NUMERIC_TO_ISO3[String(f.id).padStart(3, "0")];
    if (!iso3) return null;
    return {
      ...f,
      id: iso3,
      properties: { name: f.properties?.name ?? iso3, inPortfolio: PORTFOLIO_ISO3.has(iso3) },
    };
  })
  .filter((f) => f !== null);

const out = { type: "FeatureCollection", features };
writeFileSync(
  new URL("../src/data/geo/africa.json", import.meta.url),
  JSON.stringify(out)
);
console.log(`Écrit ${features.length} pays (dont ${features.filter((f) => f.properties.inPortfolio).length} du portfolio).`);
