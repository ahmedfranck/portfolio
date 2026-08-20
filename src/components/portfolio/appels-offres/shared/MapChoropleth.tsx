import { useId, useMemo, useState, type ReactNode } from "react";
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
  readonly legendTitle?: string;
  readonly scopeLabel?: string;
  readonly outsideScopeLabel?: string;
  readonly noDataLabel?: string;
  readonly discreteSteps?: number;
  readonly rampStart?: string;
  readonly outsideFill?: string;
  readonly outsidePatternId?: string;
  readonly scopeStroke?: string;
  readonly scopeStrokeWidth?: number;
  readonly projectionCenter?: readonly [number, number];
  readonly projectionScale?: number;
  readonly countryAdornment?: (iso3: string) => ReactNode;
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
  readonly inScope: boolean;
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
  legendTitle,
  scopeLabel = "Périmètre couvert",
  outsideScopeLabel = "Hors périmètre",
  noDataLabel = "Pas de donnée disponible",
  discreteSteps = 0,
  rampStart,
  outsideFill,
  outsidePatternId,
  scopeStroke,
  scopeStrokeWidth = 0.6,
  projectionCenter = [4, 8],
  projectionScale = 620,
  countryAdornment,
}: MapChoroplethProps) {
  const { theme } = useAoTheme();
  const [hover, setHover] = useState<HoverState | null>(null);
  const [zoom, setZoom] = useState(1);
  const [center, setCenter] = useState<[number, number]>([...projectionCenter]);
  const idSuffix = useId().replaceAll(":", "");
  const missingPatternId = `ao-map-missing-${idSuffix}`;
  const resolvedOutsidePatternId = outsidePatternId ?? `ao-map-outside-${idSuffix}`;
  const allowed = useMemo(() => new Set(countries ?? Object.keys(values)), [countries, values]);
  const numericValues = useMemo(() => Object.entries(values).filter(([iso3, value]) => allowed.has(iso3) && value != null).map(([, value]) => value as number), [allowed, values]);
  const min = numericValues.length ? Math.min(...numericValues) : 0;
  const max = numericValues.length ? Math.max(...numericValues) : 1;
  const range = max - min || 1;
  const endColor = polarity === "negative" ? theme.colors.negative : theme.colors.primary;
  const startColor = rampStart ?? theme.colors.soft;
  const resolvedOutsideFill = outsideFill ?? theme.colors.canvas;
  const stepCount = Math.max(1, discreteSteps || 1);

  function colorFor(value: number) {
    const normalized = Math.max(0, Math.min(1, (value - min) / range));
    const ratio = discreteSteps > 1
      ? Math.min(stepCount - 1, Math.floor(normalized * stepCount)) / (stepCount - 1)
      : normalized;
    return mixHex(startColor, endColor, ratio);
  }

  const rampColors = Array.from({ length: stepCount }, (_, index) => mixHex(startColor, endColor, stepCount === 1 ? 1 : index / (stepCount - 1)));

  return (
    <div className="relative" style={{ fontFamily: theme.typography.body }}>
      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ center: [...projectionCenter], scale: projectionScale }}
        width={760}
        height={height}
        style={{ width: "100%", height: "auto", touchAction: "none" }}
        role="img"
        aria-label={ariaLabel}
      >
        <defs>
          <pattern id={missingPatternId} width="6" height="6" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
            <rect width="6" height="6" fill={startColor} />
            <line x1="0" y1="0" x2="0" y2="6" stroke={theme.colors.accent} strokeWidth="1.2" />
          </pattern>
          <pattern id={resolvedOutsidePatternId} width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
            <rect width="7" height="7" fill={resolvedOutsideFill} />
            <line x1="0" y1="0" x2="0" y2="7" stroke="#C9C3B8" strokeWidth="1.2" />
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
              const fill = !isAllowed ? `url(#${resolvedOutsidePatternId})` : value == null ? `url(#${missingPatternId})` : colorFor(value);
              const label = !isAllowed ? `${name} : ${outsideScopeLabel}` : `${name} : ${value == null ? noDataLabel : `${formatValue(value)}${suffix}`}, pays du périmètre`;
              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill={fill}
                  stroke={selectedIso3 === iso3 ? theme.colors.accentLight : isAllowed ? scopeStroke ?? theme.colors.accent : "#FFFFFF"}
                  strokeWidth={selectedIso3 === iso3 ? 2 : isAllowed ? scopeStrokeWidth : 0.55}
                  tabIndex={isAllowed ? 0 : -1}
                  aria-label={label}
                  onMouseEnter={(event) => setHover({ iso3, name, value, year: years?.[iso3], x: event.clientX, y: event.clientY, inScope: isAllowed })}
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
          <strong className="inline-flex items-center gap-1.5" style={{ color: theme.colors.primary }}>{countryAdornment?.(hover.iso3)}<span>{hover.name}</span></strong>
          <p style={{ color: theme.colors.muted }}>{hover.inScope ? hover.value == null ? noDataLabel : `${formatValue(hover.value)}${suffix}${hover.year ? ` · ${hover.year}` : ""} · pays du périmètre` : outsideScopeLabel}</p>
        </div>
      )}

      <div className="mt-3 space-y-3">
        <div>
          {legendTitle && <p className="mb-2 text-[9px] font-bold" style={{ color: theme.colors.primary }}>{legendTitle}</p>}
          <div className="flex min-w-52 items-center gap-2 text-[9px]" style={{ color: theme.colors.muted }}>
            <span>{formatValue(min)}{suffix}</span>
            <span className="grid h-3 flex-1 overflow-hidden rounded-sm" style={{ gridTemplateColumns: `repeat(${stepCount}, minmax(0, 1fr))` }} aria-label={`${scopeLabel}, ${stepCount} classes`}>
              {rampColors.map((color, index) => <span key={`${color}-${index}`} style={{ background: color }} />)}
            </span>
            <span>{formatValue(max)}{suffix}</span>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-2 text-[8px]" style={{ color: theme.colors.muted }}>
            <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-5 rounded-sm" style={{ background: `linear-gradient(90deg, ${startColor}, ${endColor})`, border: `1px solid ${scopeStroke ?? theme.colors.accent}` }} />{scopeLabel}</span>
            <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-5 rounded-sm" style={{ background: `repeating-linear-gradient(135deg, ${startColor} 0 3px, ${theme.colors.accent} 3px 4px)` }} />{noDataLabel}</span>
            <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-5 rounded-sm" style={{ background: `repeating-linear-gradient(135deg, ${resolvedOutsideFill} 0 3px, #C9C3B8 3px 4px)` }} />{outsideScopeLabel}</span>
          </div>
        </div>
        <div><AoDataBadges source={source} illustrative={illustrative} /></div>
      </div>
    </div>
  );
}
