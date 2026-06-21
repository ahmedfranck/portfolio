import { useMemo, useState } from "react";
import { ComposableMap, Geographies, Geography, ZoomableGroup } from "react-simple-maps";
import { Minus, Plus } from "lucide-react";
import geoData from "../../data/geo/africa.json";
import { COUNTRY_NAMES } from "../../data/countries";
import { makeSequentialScale, type IndicatorPolarity } from "../../lib/palette";
import MapLegend from "../MapLegend";

interface ChoroplethMapProps {
  values: Record<string, number | null>; // iso3 -> valeur (null si donnée indisponible)
  years?: Record<string, number | null>; // iso3 -> année de la donnée affichée
  valueSuffix?: string;
  unit?: string;
  polarity?: IndicatorPolarity;
  height?: number;
  selectedIso3?: string | null;
  onSelectCountry?: (iso3: string) => void;
  formatValue?: (n: number) => string;
}

interface GeoFeatureProps {
  name: string;
  inPortfolio: boolean;
}

interface HoverState {
  name: string;
  value: number | null;
  year?: number | null;
  inPortfolio: boolean;
  x: number;
  y: number;
}

export default function ChoroplethMap({
  values,
  years,
  valueSuffix = "",
  unit,
  polarity = "positive",
  height = 460,
  selectedIso3,
  onSelectCountry,
  formatValue,
}: ChoroplethMapProps) {
  const [hover, setHover] = useState<HoverState | null>(null);
  const [zoom, setZoom] = useState(1);
  const [center, setCenter] = useState<[number, number]>([4, 8]);

  const { min, max } = useMemo(() => {
    const vals = Object.values(values).filter((v): v is number => v != null && Number.isFinite(v));
    if (vals.length === 0) return { min: 0, max: 1 };
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, [values]);

  const colorScale = useMemo(() => makeSequentialScale(min, max, polarity), [min, max, polarity]);
  const fmt = formatValue ?? ((n: number) => n.toLocaleString("fr-FR"));

  return (
    <div className="relative">
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ center: [4, 8], scale: 620 }}
        width={760}
        height={height}
        style={{ width: "100%", height: "auto", touchAction: "none" }}
        role="img"
        aria-label="Carte interactive de l'Afrique de l'Ouest et Centrale, pays voisins grisés pour le contexte"
      >
        <defs>
          <pattern id="hatch-missing" width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
            <rect width="6" height="6" fill="#EDEFF3" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="#C7CCD6" strokeWidth="2" />
          </pattern>
        </defs>
        <ZoomableGroup
          center={center}
          zoom={zoom}
          minZoom={1}
          maxZoom={6}
          onMoveEnd={(position) => {
            setCenter(position.coordinates);
            setZoom(position.zoom);
          }}
        >
          <Geographies geography={geoData as object}>
            {({ geographies }) =>
              geographies.map((geo) => {
                const iso3 = geo.id as string;
                const props = geo.properties as GeoFeatureProps;
                const inPortfolio = props.inPortfolio;
                const name = COUNTRY_NAMES[iso3] ?? props.name;
                const value = inPortfolio ? values[iso3] ?? null : null;
                const isSelected = selectedIso3 === iso3;
                const fill = !inPortfolio
                  ? "#EDEFF3"
                  : value != null
                    ? colorScale(value)
                    : "url(#hatch-missing)";
                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={fill}
                    stroke={isSelected ? "#5B4BE3" : "#FFFFFF"}
                    strokeWidth={isSelected ? 2 : 0.6}
                    tabIndex={inPortfolio ? 0 : -1}
                    aria-label={inPortfolio ? `${name} : ${value != null ? `${fmt(value)}${valueSuffix}` : "donnée non disponible"}` : undefined}
                    onMouseEnter={(e) => {
                      if (!inPortfolio) return;
                      setHover({ name, value, year: years?.[iso3], inPortfolio, x: e.clientX, y: e.clientY });
                    }}
                    onMouseMove={(e) => setHover((h) => (h ? { ...h, x: e.clientX, y: e.clientY } : h))}
                    onMouseLeave={() => setHover(null)}
                    onClick={() => inPortfolio && onSelectCountry?.(iso3)}
                    onKeyDown={(e) => {
                      if (inPortfolio && (e.key === "Enter" || e.key === " ")) {
                        e.preventDefault();
                        onSelectCountry?.(iso3);
                      }
                    }}
                    style={{
                      default: { outline: "none", transition: "fill 0.2s ease" },
                      hover: {
                        outline: "none",
                        filter: inPortfolio ? "brightness(0.93)" : "none",
                        cursor: inPortfolio ? "pointer" : "default",
                      },
                      pressed: { outline: "none" },
                    }}
                  />
                );
              })
            }
          </Geographies>
        </ZoomableGroup>
      </ComposableMap>

      <div className="absolute right-2 top-2 flex flex-col gap-1">
        <button
          type="button"
          aria-label="Zoomer"
          onClick={() => setZoom((z) => Math.min(6, z + 0.6))}
          className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full border border-line bg-white text-ink shadow-card transition-all duration-200 hover:scale-105 hover:bg-brand-soft hover:text-brand"
        >
          <Plus size={14} aria-hidden="true" />
        </button>
        <button
          type="button"
          aria-label="Dézoomer"
          onClick={() => setZoom((z) => Math.max(1, z - 0.6))}
          className="flex h-7 w-7 cursor-pointer items-center justify-center rounded-full border border-line bg-white text-ink shadow-card transition-all duration-200 hover:scale-105 hover:bg-brand-soft hover:text-brand"
        >
          <Minus size={14} aria-hidden="true" />
        </button>
      </div>

      {hover && (
        <div
          className="pointer-events-none fixed z-50 rounded-lg border border-line bg-white px-3 py-2 text-xs shadow-cardHover"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
        >
          <p className="font-display font-semibold text-ink">{hover.name}</p>
          <p className="text-text-2">
            {hover.value != null ? `${fmt(hover.value)}${valueSuffix}${hover.year ? ` (${hover.year})` : ""}` : "n.d."}
          </p>
        </div>
      )}

      <div className="mt-3">
        <MapLegend min={min} max={max} unit={unit ?? valueSuffix} polarity={polarity} formatValue={fmt} />
      </div>
    </div>
  );
}
