import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "sante-reproductive-fecondite",
  shortTitle: "Santé reproductive",
  title: "Santé reproductive & fécondité",
  domain: "Santé reproductive",
  keywords: ["contraception", "fécondité", "demande satisfaite"],
  angle: "Quel accès réel à la contraception, et avec quel effet sur la fécondité ?",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) de prévalence contraceptive, de fécondité et de demande satisfaite par des méthodes modernes dans 16 pays entre 2010 et 2024.",
  insights: [
    "La prévalence contraceptive moderne progresse dans la plupart des pays du panel, en lien avec une baisse progressive de l'indice de fécondité.",
    "Les pays où la demande de planification familiale est le mieux satisfaite par des méthodes modernes affichent généralement une prévalence contraceptive globale plus élevée.",
    "L'écart entre prévalence toutes méthodes et prévalence méthodes modernes reste significatif dans certains pays, signe d'un recours encore important aux méthodes traditionnelles.",
  ],
  family: "sante-developpement-humain",
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
