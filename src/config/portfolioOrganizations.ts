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
    focus: "Planification familiale et santé reproductive dans les neuf pays du Partenariat de Ouagadougou.",
    accent: "#8a6e38",
    soft: "#f4f0e5",
    logo: partenariatOuagadougouLogo,
    projectSlugs: ["sante-reproductive-fecondite"],
  },
  {
    id: "bad",
    acronym: "BAD",
    name: "Banque africaine de développement",
    focus: "Croissance économique, agriculture, accès à l'énergie, inclusion numérique et financière, emploi et autonomisation des jeunes.",
    accent: "#355a49",
    soft: "#edf1ec",
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
    accent: "#356b81",
    soft: "#edf2f3",
    logo: unicefLogo,
    projectSlugs: ["wash-eau-assainissement", "populations-deplacees"],
  },
  {
    id: "pnue",
    acronym: "PNUE",
    name: "Programme des Nations Unies pour l'environnement",
    focus: "Action climatique, protection des écosystèmes, biodiversité et suivi des pressions environnementales.",
    accent: "#3c665d",
    soft: "#edf2ef",
    logo: unepLogo,
    projectSlugs: ["climat-environnement"],
  },
];

export function getPortfolioOrganizationByProject(slug: string) {
  return CALL_FOR_OFFERS_ORGANIZATIONS.find((organization) => organization.projectSlugs.includes(slug));
}
