import type { ReactNode } from "react";

interface CountryFlagProps {
  readonly iso3: string;
  readonly size?: "sm" | "md" | "lg";
  readonly className?: string;
}

const DIMENSIONS = {
  sm: { width: 16, height: 11 },
  md: { width: 20, height: 14 },
  lg: { width: 30, height: 20 },
} as const;

const STAR = "15,5.2 16.2,8.6 19.8,8.6 16.9,10.7 18,14.2 15,12.1 12,14.2 13.1,10.7 10.2,8.6 13.8,8.6";

function FlagArtwork({ iso3 }: { readonly iso3: string }): ReactNode {
  if (iso3 === "BEN") return <><rect width="12" height="20" fill="#008751" /><rect x="12" width="18" height="10" fill="#FCD116" /><rect x="12" y="10" width="18" height="10" fill="#E8112D" /></>;
  if (iso3 === "BFA") return <><rect width="30" height="10" fill="#EF2B2D" /><rect y="10" width="30" height="10" fill="#009E49" /><polygon points={STAR} fill="#FCD116" /></>;
  if (iso3 === "CIV") return <><rect width="10" height="20" fill="#F77F00" /><rect x="10" width="10" height="20" fill="#FFFFFF" /><rect x="20" width="10" height="20" fill="#009E60" /></>;
  if (iso3 === "GIN") return <><rect width="10" height="20" fill="#CE1126" /><rect x="10" width="10" height="20" fill="#FCD116" /><rect x="20" width="10" height="20" fill="#009460" /></>;
  if (iso3 === "MLI") return <><rect width="10" height="20" fill="#14B53A" /><rect x="10" width="10" height="20" fill="#FCD116" /><rect x="20" width="10" height="20" fill="#CE1126" /></>;
  if (iso3 === "MRT") return <><rect width="30" height="20" fill="#00A95C" /><rect width="30" height="2.5" fill="#D01C1F" /><rect y="17.5" width="30" height="2.5" fill="#D01C1F" /><circle cx="15" cy="9" r="5" fill="#FFD700" /><circle cx="15" cy="7.2" r="4.2" fill="#00A95C" /><polygon points="15,9.2 15.7,11 17.6,11 16.1,12.1 16.7,14 15,12.8 13.3,14 13.9,12.1 12.4,11 14.3,11" fill="#FFD700" /></>;
  if (iso3 === "NER") return <><rect width="30" height="6.67" fill="#E05206" /><rect y="6.67" width="30" height="6.66" fill="#FFFFFF" /><rect y="13.33" width="30" height="6.67" fill="#0DB02B" /><circle cx="15" cy="10" r="2.4" fill="#E05206" /></>;
  if (iso3 === "SEN") return <><rect width="10" height="20" fill="#00853F" /><rect x="10" width="10" height="20" fill="#FDEF42" /><rect x="20" width="10" height="20" fill="#E31B23" /><polygon points={STAR} fill="#00853F" /></>;
  if (iso3 === "TGO") return <><rect width="30" height="4" fill="#006A4E" /><rect y="4" width="30" height="4" fill="#FFCE00" /><rect y="8" width="30" height="4" fill="#006A4E" /><rect y="12" width="30" height="4" fill="#FFCE00" /><rect y="16" width="30" height="4" fill="#006A4E" /><rect width="12" height="12" fill="#D21034" /><polygon points="6,2.1 6.9,4.7 9.7,4.7 7.4,6.3 8.3,9 6,7.4 3.7,9 4.6,6.3 2.3,4.7 5.1,4.7" fill="#FFFFFF" /></>;
  return <><rect width="30" height="20" fill="#F2F0EA" /><text x="15" y="13" textAnchor="middle" fontSize="7" fontWeight="700" fill="#667085">{iso3.slice(0, 2)}</text></>;
}

export default function CountryFlag({ iso3, size = "sm", className = "" }: CountryFlagProps) {
  const dimensions = DIMENSIONS[size];
  return (
    <svg
      viewBox="0 0 30 20"
      width={dimensions.width}
      height={dimensions.height}
      className={`inline-block shrink-0 overflow-hidden rounded-[2px] border border-black/10 shadow-[0_1px_1px_rgba(0,0,0,.08)] ${className}`}
      aria-hidden="true"
      focusable="false"
    >
      <FlagArtwork iso3={iso3} />
    </svg>
  );
}

interface CountryLabelProps {
  readonly iso3: string;
  readonly name: ReactNode;
  readonly size?: "sm" | "md" | "lg";
  readonly className?: string;
}

export function CountryLabel({ iso3, name, size = "sm", className = "" }: CountryLabelProps) {
  return <span className={`inline-flex items-center gap-1.5 ${className}`}><CountryFlag iso3={iso3} size={size} /><span>{name}</span></span>;
}
