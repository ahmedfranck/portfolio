import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "climat-environnement",
  shortTitle: "Climat & environnement",
  title: "Climat & environnement",
  domain: "Climat & environnement",
  family: "economie-societe-environnement",
  keywords: ["émissions de CO2", "couvert forestier", "aires protégées"],
  angle: "Pressions environnementales et réponses de conservation.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) d'émissions de CO₂ par habitant, de couvert forestier, de ressources en eau et d'aires protégées dans 16 pays entre 2010 et 2024.",
  insights: [
    "Les émissions de CO₂ par habitant restent globalement faibles à l'échelle mondiale pour la plupart des pays du panel, mais progressent dans certains d'entre eux.",
    "Le couvert forestier recule dans plusieurs pays sur la période, en lien avec la pression démographique et agricole.",
    "L'extension des aires protégées ne compense pas toujours le recul du couvert forestier observé par ailleurs.",
  ],
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
