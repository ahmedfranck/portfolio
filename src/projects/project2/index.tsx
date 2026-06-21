import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "nutrition-survie-enfant",
  shortTitle: "Nutrition enfant",
  title: "Nutrition & survie de l'enfant",
  domain: "Santé de l'enfant",
  keywords: ["stunting", "malnutrition", "vaccination"],
  angle: "La double charge de la malnutrition infantile.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) de retard de croissance, d'émaciation et de couverture vaccinale des enfants de moins de 5 ans dans 16 pays, en lien avec le PIB par habitant.",
  insights: [
    "Le retard de croissance recule sur la durée dans la plupart des pays, mais reste à des niveaux élevés là où l'insuffisance pondérale demeure forte.",
    "La relation entre PIB par habitant et stunting est visible mais imparfaite : certains pays à revenu modeste obtiennent de meilleurs résultats nutritionnels que des pays plus riches.",
    "Les indicateurs de malnutrition (stunting, wasting, insuffisance pondérale) reposent sur des enquêtes ponctuelles : leur fréquence de mise à jour est plus faible que celle des indicateurs de couverture vaccinale.",
  ],
  family: "sante-developpement-humain",
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
