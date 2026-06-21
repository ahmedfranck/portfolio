import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "emploi-jeunesse",
  shortTitle: "Emploi & jeunesse",
  title: "Emploi & jeunesse",
  domain: "Emploi",
  family: "economie-societe-environnement",
  keywords: ["chômage des jeunes", "emploi vulnérable", "taux d'activité"],
  angle: "L'insertion des jeunes sur le marché du travail.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) de chômage des jeunes, de chômage total, d'emploi vulnérable et de taux d'activité dans 16 pays entre 2010 et 2024.",
  insights: [
    "Le chômage des jeunes est presque systématiquement supérieur au chômage total, traduisant des difficultés d'insertion spécifiques à cette tranche d'âge.",
    "Une part élevée d'emploi vulnérable coexiste souvent avec un chômage officiel relativement bas : une partie importante de l'activité reste informelle et précaire.",
    "Le taux d'activité varie peu dans le temps pour la plupart des pays, suggérant des dynamiques structurelles plus que conjoncturelles.",
  ],
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
