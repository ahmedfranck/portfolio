import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "populations-deplacees",
  shortTitle: "Populations déplacées",
  title: "Populations déplacées & accès aux services",
  domain: "Fragilité & accès aux soins",
  keywords: ["déplacement", "réfugiés", "fragilité"],
  angle: "Suivre les dynamiques de déplacement forcé dans la région.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data — données HCR) de réfugiés et de déplacés internes dans 16 pays d'Afrique de l'Ouest et Centrale entre 2010 et 2024.",
  insights: [
    "Le nombre de personnes déplacées est très concentré sur un nombre restreint de pays du panel, plutôt que répartie de façon homogène sur la région.",
    "Les pays accueillant le plus de réfugiés ne sont pas toujours ceux qui en comptent le plus parmi leurs propres ressortissants déplacés à l'étranger.",
    "Les déplacés internes restent l'indicateur le moins systématiquement renseigné de ce projet : sa couverture pays est légèrement inférieure aux autres séries.",
  ],
  family: "sante-developpement-humain",
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
