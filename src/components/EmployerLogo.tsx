import { useState } from "react";
import { getEmployerLogo } from "../lib/employerLogos";

interface EmployerLogoProps {
  logo?: string;
  name: string;
}

function monogram(name: string): string {
  const words = name.split(/[\s()]+/).filter(Boolean);
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
  return (words[0]?.slice(0, 2) ?? "?").toUpperCase();
}

/** Logo employeur (~40px) avec fallback monogramme si le fichier est absent sous src/assets/logos/. */
export default function EmployerLogo({ logo, name }: EmployerLogoProps) {
  const src = logo ? getEmployerLogo(logo) : undefined;
  const [failed, setFailed] = useState(false);

  if (src && !failed) {
    return (
      <span className="flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-line bg-white p-1.5 grayscale transition duration-200 group-hover:scale-105 group-hover:grayscale-0">
        <img
          src={src}
          alt={`Logo ${name}`}
          className="h-full w-full object-contain"
          onError={() => setFailed(true)}
        />
      </span>
    );
  }

  return (
    <span
      className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-brand-soft font-display text-sm font-semibold text-brand transition duration-200 group-hover:scale-105"
      aria-hidden="true"
    >
      {monogram(name)}
    </span>
  );
}
