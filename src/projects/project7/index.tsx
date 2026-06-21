import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "economie-croissance",
  shortTitle: "Économie & croissance",
  title: "Économie & croissance",
  domain: "Économie",
  family: "economie-societe-environnement",
  keywords: ["PIB", "inflation", "investissements directs étrangers"],
  angle: "Croissance, prix et attractivité : la photographie macroéconomique de la région.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) de croissance du PIB, de PIB par habitant, d'inflation et d'investissements directs étrangers dans 16 pays d'Afrique de l'Ouest et Centrale entre 2010 et 2024.",
  insights: [
    "La croissance du PIB reste très volatile d'une année à l'autre selon les pays, sans tendance régionale unique sur la période.",
    "Les pays affichant le PIB par habitant le plus élevé ne sont pas systématiquement ceux qui attirent le plus d'investissements directs étrangers en % du PIB.",
    "L'inflation a connu des pics marqués sur certaines années (chocs sur les prix alimentaires et énergétiques), visibles dans les séries temporelles.",
  ],
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
