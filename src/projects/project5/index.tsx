import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "wash-eau-assainissement",
  shortTitle: "WASH",
  title: "Eau, assainissement & hygiène (WASH)",
  domain: "WASH",
  keywords: ["eau potable", "assainissement", "santé environnementale"],
  angle: "L'eau et l'assainissement comme déterminants de santé.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) mettant en relation l'accès à l'eau potable, l'assainissement de base et la mortalité des moins de 5 ans dans 16 pays.",
  insights: [
    "Les pays qui réduisent le plus la défécation à l'air libre affichent souvent aussi des baisses notables de la mortalité infantile, sans que la relation soit strictement linéaire.",
    "L'accès à l'eau potable a généralement progressé plus vite que l'assainissement de base, qui reste le maillon faible dans plusieurs pays du panel.",
    "La taille des bulles (population réelle) met en évidence l'enjeu d'échelle : les pays les plus peuplés concentrent une part importante des besoins non couverts.",
  ],
  family: "sante-developpement-humain",
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
