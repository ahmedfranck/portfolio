import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "sante-reproductive-fecondite",
  shortTitle: "Observatoire PF",
  title: "Observatoire PF — santé reproductive & fécondité",
  domain: "Santé reproductive",
  keywords: ["contraception", "fécondité", "financement", "crise", "trajectoires mCPR"],
  angle: "Quel accès réel à la contraception, et avec quel effet sur la fécondité ?",
  pitch:
    "Observatoire régional des neuf pays du Partenariat de Ouagadougou : indicateurs Banque mondiale réels, trajectoires mCPR, financement, impact et contexte de crise avec scénarios illustratifs explicitement signalés.",
  insights: [
    "La prévalence contraceptive moderne progresse dans la plupart des pays du panel, en lien avec une baisse progressive de l'indice de fécondité.",
    "Les pays où la demande de planification familiale est le mieux satisfaite par des méthodes modernes affichent généralement une prévalence contraceptive globale plus élevée.",
    "L'écart entre prévalence toutes méthodes et prévalence méthodes modernes reste significatif dans certains pays, signe d'un recours encore important aux méthodes traditionnelles.",
  ],
  family: "sante-developpement-humain",
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
