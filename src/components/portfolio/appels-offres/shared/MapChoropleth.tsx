import { useId, useMemo, useState } from "react";
import { ComposableMap, Geographies, Geography, ZoomableGroup } from "react-simple-maps";
import { Minus, Plus } from "lucide-react";
import geoData from "../../../../data/geo/africa.json";
import { COUNTRY_NAMES } from "../../../../data/countries";
import { useAoTheme } from "../../../../hooks/useAoTheme";
import { AoDataBadges } from "./badges";

interface MapChoroplethProps {
  readonly values: Readonly<Record<string, number | null>>;
  readonly years?: Readonly<Record<string, number | null>>;
  readonly countries?: readonly string[];
  readonly selectedIso3?: string | null;
  readonly onSelectCountry?: (iso3: string) => void;
  readonly formatValue?: (value: number) => string;
  readonly suffix?: string;
  readonly height?: number;
  readonly polarity?: "positive" | "negative";
  readonly source?: string;
  readonly illustrative?: boolean;
  readonly ariaLabel?: string;
}

interface GeoProperties {
  readonly name: string;
  readonly inPortfolio: boolean;
}

interface HoverState {
  readonly iso3: string;
  readonly name: string;
  readonly value: number | null;
  readonly year?: number | null;
  readonly x: number;
  readonly y: number;
}

function hexToRgb(hex: string) {
  const value = Number.parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255] as const;
}

function mixHex(from: string, to: string, ratio: number) {
  const start = hexToRgb(from);
  const end = hexToRgb(to);
  const channel = (index: number) => Math.round(start[index] + (end[index] - start[index]) * ratio);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

export default function MapChoropleth({
  values,
  years,
  countries,
  selectedIso3,
  onSelectCountry,
  formatValue = (value) => value.toLocaleString("fr-FR"),
  suffix = "",
  height = 430,
  polarity = "positive",
  source,
  illustrative = false,
  ariaLabel = "Carte choroplèthe interactive",
}: MapChoroplethProps) {
  const { theme } = useAoTheme();
  const [hover, setHover] = useState<HoverState | null>(null);
  const [zoom, setZoom] = useState(1);
  const [center, setCenter] = useState<[number, number]>([4, 8]);
  const patternId = `ao-map-missing-${useId().replaceAll(":", "")}`;
  const allowed = useMemo(() => new Set(countries ?? Object.keys(values)), [countries, values]);
  const numericValues = useMemo(() => Object.entries(values).filter(([iso3, value]) => allowed.has(iso3) && value != null).map(([, value]) => value as number), [allowed, values]);
  const min = numericValues.length ? Math.min(...numericValues) : 0;
  const max = numericValues.length ? Math.max(...numericValues) : 1;
  const range = max - min || 1;
  const endColor = polarity === "negative" ? theme.colors.negative : theme.colors.primary;

  function colorFor(value: number) {
    return mixHex(theme.colors.soft, endColor, Math.max(0, Math.min(1, (value - min) / range)));
  }

  return (
    <div className="relative" style={{ fontFamily: theme.typography.body }}>
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ center: [4, 8], scale: 620 }}
        width={760}
        height={height}
        style={{ width: "100%", height: "auto", touchAction: "none" }}
        role="img"
        aria-label={ariaLabel}
      >
        <defs>
          <pattern id={patternId} width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
            <rect width="6" height="6" fill={theme.colors.canvas} />
            <line x1="0" y1="0" x2="0" y2="6" stroke={theme.colors.border} strokeWidth="2" />
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
            {({ geographies }) => geographies.map((geo) => {
              const iso3 = geo.id as string;
              const properties = geo.properties as GeoProperties;
              const isAllowed = allowed.has(iso3);
              const value = isAllowed ? values[iso3] ?? null : null;
              const name = COUNTRY_NAMES[iso3] ?? properties.name;
              const fill = !isAllowed ? theme.colors.canvas : value == null ? `url(#${patternId})` : colorFor(value);
              const label = `${name} : ${value == null ? "donnée non disponible" : `${formatValue(value)}${suffix}`}`;
              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill={fill}
                  stroke={selectedIso3 === iso3 ? theme.colors.accent : "#FFFFFF"}
                  strokeWidth={selectedIso3 === iso3 ? 2 : 0.6}
                  tabIndex={isAllowed ? 0 : -1}
                  aria-label={isAllowed ? label : undefined}
                  onMouseEnter={(event) => isAllowed && setHover({ iso3, name, value, year: years?.[iso3], x: event.clientX, y: event.clientY })}
                  onMouseMove={(event) => setHover((current) => current ? { ...current, x: event.clientX, y: event.clientY } : current)}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => isAllowed && onSelectCountry?.(iso3)}
                  onKeyDown={(event) => {
                    if (isAllowed && (event.key === "Enter" || event.key === " ")) {
                      event.preventDefault();
                      onSelectCountry?.(iso3);
                    }
                  }}
                  style={{
                    default: { outline: "none", transition: "fill 180ms ease" },
                    hover: { outline: "none", cursor: isAllowed ? "pointer" : "default", filter: isAllowed ? "brightness(0.94)" : "none" },
                    pressed: { outline: "none" },
                  }}
                />
              );
            })}
          </Geographies>
        </ZoomableGroup>
      </ComposableMap>

      <div className="absolute right-2 top-2 flex flex-col gap-1">
        {([
          { label: "Zoomer", icon: Plus, action: () => setZoom((value) => Math.min(6, value + 0.6)) },
          { label: "Dézoomer", icon: Minus, action: () => setZoom((value) => Math.max(1, value - 0.6)) },
        ] as const).map(({ label, icon: Icon, action }) => (
          <button key={label} type="button" aria-label={label} onClick={action} className="flex h-8 w-8 items-center justify-center rounded-full border bg-white shadow-sm" style={{ borderColor: theme.colors.border, color: theme.colors.primary }}>
            <Icon size={14} aria-hidden="true" />
          </button>
        ))}
      </div>

      {hover && (
        <div className="pointer-events-none fixed z-50 rounded-md border bg-white px-3 py-2 text-[10px] shadow-lg" style={{ left: hover.x + 12, top: hover.y + 12, borderColor: theme.colors.border }}>
          <strong style={{ color: theme.colors.primary }}>{hover.name}</strong>
          <p style={{ color: theme.colors.muted }}>{hover.value == null ? "n.d." : `${formatValue(hover.value)}${suffix}${hover.year ? ` · ${hover.year}` : ""}`}</p>
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-52 items-center gap-2 text-[9px]" style={{ color: theme.colors.muted }}>
          <span>{formatValue(min)}{suffix}</span>
          <span className="h-2 flex-1 rounded-full" style={{ background: `linear-gradient(90deg, ${theme.colors.soft}, ${endColor})` }} aria-hidden="true" />
          <span>{formatValue(max)}{suffix}</span>
        </div>
        <AoDataBadges source={source} illustrative={illustrative} />
      </div>
    </div>
  );
}
