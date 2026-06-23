import {
  Activity,
  BadgeCheck,
  Code2,
  Database,
  Handshake,
  LayoutDashboard,
  Presentation,
  ShieldCheck,
  Workflow,
  type LucideIcon,
} from "lucide-react";
import { MONITORING, WEB_DEV } from "../config/content";
import type { ProjectFamily } from "../projects/types";

export type PortfolioFamily = ProjectFamily | "applications-data-science";

export const EXPERTISE: { icon: LucideIcon; title: string; description: string }[] = [
  {
    icon: Database,
    title: "Observatoires & data hubs",
    description:
      "Conception de hubs de données et d'observatoires pour le suivi d'indicateurs stratégiques, de la collecte à la restitution.",
  },
  {
    icon: LayoutDashboard,
    title: "Tableaux de bord décisionnels",
    description:
      "Dashboards Power BI, Tableau et Dataiku pour le pilotage de la performance et l'aide à la décision du COMEX.",
  },
  {
    icon: Workflow,
    title: "Pipelines & gouvernance des données",
    description:
      "Pipelines ETL (Python/pandas), règles de qualité et traçabilité des sources pour des données fiables et auditables.",
  },
  {
    icon: Activity,
    title: "Monitoring santé / PF / SSR",
    description: MONITORING.groups[0].items[0],
  },
  {
    icon: Code2,
    title: "Développement web & intégration",
    description: WEB_DEV.items[0],
  },
];

export const APPROACH: { icon: LucideIcon; title: string; description: string }[] = [
  {
    icon: Handshake,
    title: "Co-construction",
    description: "Ateliers de cadrage avec les équipes métier et les partenaires pour définir des indicateurs partagés.",
  },
  {
    icon: ShieldCheck,
    title: "Traçabilité des sources",
    description: "Documentation des sources, fréquences et méthodologies : rien n'est affiché sans origine claire.",
  },
  {
    icon: BadgeCheck,
    title: "Qualité des données",
    description: "Procédures de contrôle qualité et de validation avant toute restitution.",
  },
  {
    icon: Presentation,
    title: "Formation & accompagnement",
    description: "Formation des utilisateurs et accompagnement post-lancement pour ancrer la culture data.",
  },
];

export const FAMILIES: { id: PortfolioFamily; label: string }[] = [
  { id: "sante-developpement-humain", label: "Santé & développement humain" },
  { id: "economie-societe-environnement", label: "Économie, société & environnement" },
  { id: "applications-data-science", label: "Applications & data science (Streamlit)" },
];

export const FEATURE_POINTS: { icon: LucideIcon; title: string; description: string }[] = [
  {
    icon: Database,
    title: "Double compétence data + web",
    description: "Analyse, modélisation et développement d'interfaces décisionnelles dans un même flux projet.",
  },
  {
    icon: Workflow,
    title: "Projet de bout en bout",
    description: "Cadrage, architecture, indicateurs, UX/UI, dashboards, intégration, formation et support.",
  },
  {
    icon: Activity,
    title: "Santé / PF / suivi-évaluation",
    description: "Spécialisation sur les programmes de santé publique et de développement en Afrique de l'Ouest.",
  },
  {
    icon: ShieldCheck,
    title: "Données ouvertes & institutionnelles",
    description: "Documentation des sources et restitution claire pour des indicateurs auditables.",
  },
];

export const DATA_TOOLS = [
  "Banque mondiale",
  "DHS / EDS",
  "Track20",
  "OMS",
  "UNFPA",
  "OCHA",
  "HCR",
  "UNICEF",
  "Power BI",
  "Tableau",
  "Dataiku",
  "PostgreSQL",
  "Python",
  "React",
];

export const FEATURED_PROJECT_SLUGS = ["sante-maternelle-neonatale", "economie-croissance", "climat-environnement"];
