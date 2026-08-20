import afdbLogo from "../assets/organizations/afdb.webp";
import partenariatOuagadougouLogo from "../assets/organizations/partenariat-ouagadougou.png";
import unepLogo from "../assets/organizations/unep.svg";
import unicefLogo from "../assets/organizations/unicef.webp";
import unverifiedLogo from "../assets/organizations/unverified.svg";

export interface PortfolioOrganization {
  id: "ucpo" | "bad" | "unicef" | "pnue" | "unverified";
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
    focus: "Planification familiale et santé reproductive dans les neuf pays du Partenariat de Ouagadougou.",
    accent: "#c3911f",
    soft: "#fdf3dc",
    logo: partenariatOuagadougouLogo,
    projectSlugs: ["sante-reproductive-fecondite"],
  },
  {
    id: "unverified",
    acronym: "À CONFIRMER",
    name: "Commanditaire à confirmer",
    focus: "Association institutionnelle laissée ouverte : aucune identité de commanditaire n'est revendiquée sans preuve documentaire.",
    accent: "#16a085",
    soft: "#e8f6f3",
    logo: unverifiedLogo,
    projectSlugs: ["sante-maternelle-neonatale", "nutrition-survie-enfant", "education-filles-genre"],
  },
  {
    id: "bad",
    acronym: "BAD",
    name: "Banque africaine de développement",
    focus: "Croissance économique, agriculture, accès à l'énergie, inclusion numérique et financière, emploi et autonomisation des jeunes.",
    accent: "#006a4e",
    soft: "#e8f4f0",
    logo: afdbLogo,
    projectSlugs: [
      "economie-croissance",
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
    soft: "#eaf8fd",
    logo: unicefLogo,
    projectSlugs: ["wash-eau-assainissement", "populations-deplacees"],
  },
  {
    id: "pnue",
    acronym: "PNUE",
    name: "Programme des Nations Unies pour l'environnement",
    focus: "Action climatique, protection des écosystèmes, biodiversité et suivi des pressions environnementales.",
    accent: "#00a499",
    soft: "#e7f7f5",
    logo: unepLogo,
    projectSlugs: ["climat-environnement"],
  },
];

export function getPortfolioOrganizationByProject(slug: string) {
  return CALL_FOR_OFFERS_ORGANIZATIONS.find((organization) => organization.projectSlugs.includes(slug));
}
