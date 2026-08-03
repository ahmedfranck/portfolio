import afdbLogo from "../assets/organizations/afdb.webp";
import partenariatOuagadougouLogo from "../assets/organizations/partenariat-ouagadougou.png";
import unepLogo from "../assets/organizations/unep.svg";
import unicefLogo from "../assets/organizations/unicef.webp";

export interface PortfolioOrganization {
  id: "ucpo" | "bad" | "unicef" | "pnue";
  acronym: string;
  name: string;
  focus: string;
  accent: string;
  soft: string;
  logo?: string;
  projectSlugs: string[];
}

export const CALL_FOR_OFFERS_ORGANIZATIONS: PortfolioOrganization[] = [
  {
    id: "ucpo",
    acronym: "UCPO",
    name: "Partenariat de Ouagadougou",
    focus: "Planification familiale, santé reproductive, santé maternelle, nutrition et égalité de genre.",
    accent: "#c99a2e",
    soft: "#fbf7e9",
    logo: partenariatOuagadougouLogo,
    projectSlugs: [
      "sante-maternelle-neonatale",
      "education-filles-genre",
      "nutrition-survie-enfant",
      "sante-reproductive-fecondite",
    ],
  },
  {
    id: "bad",
    acronym: "BAD",
    name: "Banque africaine de développement",
    focus: "Agriculture, accès à l'énergie, inclusion numérique et financière, emploi et autonomisation des jeunes.",
    accent: "#27784a",
    soft: "#edf7f1",
    logo: afdbLogo,
    projectSlugs: [
      "agriculture-securite-alimentaire",
      "energie-acces-electricite",
      "inclusion-numerique-financiere",
      "emploi-jeunesse",
    ],
  },
  {
    id: "unicef",
    acronym: "UNICEF",
    name: "Fonds des Nations Unies pour l'enfance",
    focus: "Eau, assainissement et hygiène, services essentiels et protection des populations vulnérables.",
    accent: "#1cabe2",
    soft: "#edf9fd",
    logo: unicefLogo,
    projectSlugs: ["wash-eau-assainissement", "populations-deplacees"],
  },
  {
    id: "pnue",
    acronym: "PNUE",
    name: "Programme des Nations Unies pour l'environnement",
    focus: "Action climatique, protection des écosystèmes, biodiversité et suivi des pressions environnementales.",
    accent: "#167d69",
    soft: "#edf8f5",
    logo: unepLogo,
    projectSlugs: ["climat-environnement"],
  },
];

export function getPortfolioOrganizationByProject(slug: string) {
  return CALL_FOR_OFFERS_ORGANIZATIONS.find((organization) => organization.projectSlugs.includes(slug));
}
