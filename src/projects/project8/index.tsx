import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "agriculture-securite-alimentaire",
  shortTitle: "Agriculture & alimentation",
  title: "Agriculture & sécurité alimentaire",
  domain: "Agriculture",
  family: "economie-societe-environnement",
  keywords: ["rendement céréalier", "sous-alimentation", "production alimentaire"],
  angle: "Nourrir la région : rendements, production et sous-alimentation.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) de valeur ajoutée agricole, de rendement céréalier, de production alimentaire et de sous-alimentation dans 16 pays entre 2010 et 2024.",
  insights: [
    "Les rendements céréaliers varient fortement d'un pays à l'autre, sans corrélation stricte avec la part de terres arables disponibles.",
    "La prévalence de la sous-alimentation ne suit pas toujours l'évolution de l'indice de production alimentaire : la disponibilité globale ne garantit pas l'accès de tous aux aliments.",
    "L'agriculture pèse encore fortement dans le PIB de plusieurs pays du panel, signe d'économies encore peu diversifiées.",
  ],
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
