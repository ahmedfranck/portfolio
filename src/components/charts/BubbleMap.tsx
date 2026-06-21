import { useMemo, useState } from "react";
import { ComposableMap, Geographies, Geography, Marker } from "react-simple-maps";
import geoData from "../../data/geo/africa.json";
import { CENTROIDS } from "../../data/geo/centroids";
import { COUNTRY_NAMES } from "../../data/countries";
import { PALETTE } from "../../lib/palette";

interface GeoFeatureProps {
  inPortfolio: boolean;
}

interface BubbleMapProps {
  values: Record<string, number | null>; // iso3 -> valeur affichée (ex. accès à l'eau %)
  sizes: Record<string, number | null>; // iso3 -> taille de bulle (ex. population)
  years?: Record<string, number | null>; // iso3 -> année de la donnée affichée
  valueSuffix?: string;
  sizeLabel?: string;
  height?: number;
}

export default function BubbleMap({
  values,
  sizes,
  years,
  valueSuffix = "",
  sizeLabel = "population",
  height = 420,
}: BubbleMapProps) {
  const [hover, setHover] = useState<{
    name: string;
    value: number;
    size: number;
    year?: number | null;
    x: number;
    y: number;
  } | null>(null);

  const maxSize = useMemo(() => {
    const sizeValues = Object.values(sizes).filter((v): v is number => v != null);
    return sizeValues.length > 0 ? Math.max(...sizeValues) : 0;
  }, [sizes]);

  function radiusFor(size: number): number {
    const minR = 5;
    const maxR = 22;
    if (!maxSize) return minR;
    return minR + (maxR - minR) * Math.sqrt(size / maxSize);
  }

  return (
    <div className="relative">
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ center: [4, 8], scale: 620 }}
        width={760}
        height={height}
        style={{ width: "100%", height: "auto" }}
        role="img"
        aria-label="Carte à bulles de l'Afrique de l'Ouest et Centrale"
      >
        <Geographies geography={geoData as object}>
          {({ geographies }) =>
            geographies.map((geo) => {
              const inPortfolio = (geo.properties as GeoFeatureProps).inPortfolio;
              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill={inPortfolio ? "#F5F6F8" : "#EDEFF3"}
                  stroke="#FFFFFF"
                  strokeWidth={0.6}
                  style={{ default: { outline: "none" }, hover: { outline: "none" }, pressed: { outline: "none" } }}
                />
              );
            })
          }
        </Geographies>
        {Object.entries(CENTROIDS).map(([iso3, coords]) => {
          const value = values[iso3];
          const size = sizes[iso3] ?? 0;
          if (value == null) return null;
          return (
            <Marker key={iso3} coordinates={coords}>
              <circle
                r={radiusFor(size)}
                fill={PALETTE.brand}
                fillOpacity={0.55}
                stroke={PALETTE.brand}
                strokeWidth={1.5}
                onMouseEnter={(e) =>
                  setHover({ name: COUNTRY_NAMES[iso3], value, size, year: years?.[iso3], x: e.clientX, y: e.clientY })
                }
                onMouseMove={(e) => setHover((h) => (h ? { ...h, x: e.clientX, y: e.clientY } : h))}
                onMouseLeave={() => setHover(null)}
                style={{ cursor: "pointer" }}
              />
            </Marker>
          );
        })}
      </ComposableMap>
      {hover && (
        <div
          className="pointer-events-none fixed z-50 rounded-lg border border-line bg-white px-3 py-2 text-xs shadow-cardHover"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
        >
          <p className="font-display font-semibold text-ink">{hover.name}</p>
          <p className="text-text-2">
            Valeur : {hover.value}
            {valueSuffix}
            {hover.year ? ` (${hover.year})` : ""}
          </p>
          <p className="text-text-2">
            Taille de bulle : {hover.size.toLocaleString("fr-FR")} ({sizeLabel})
          </p>
        </div>
      )}
    </div>
  );
}
