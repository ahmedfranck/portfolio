import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "sante-maternelle-neonatale",
  shortTitle: "Santé maternelle",
  title: "Santé maternelle & néonatale",
  domain: "Santé publique",
  keywords: ["mortalité maternelle", "soins prénatals", "accouchements assistés"],
  angle: "Où en est l'accès des femmes à des soins sûrs ?",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) de santé maternelle et néonatale pour suivre l'accès aux soins obstétricaux dans 16 pays d'Afrique de l'Ouest et Centrale entre 2010 et 2024.",
  insights: [
    "La couverture des soins prénatals et des accouchements assistés a globalement progressé sur la période, mais à des rythmes très inégaux selon les pays.",
    "Les pays affichant les plus forts taux d'accouchements assistés ne sont pas toujours ceux avec le ratio de mortalité maternelle le plus bas : d'autres déterminants (distance aux structures, qualité des soins) entrent en jeu.",
    "Certains indicateurs (soins prénatals, accouchements assistés) ont un décalage de publication : les dernières données disponibles ne couvrent pas toujours l'année la plus récente, ce qui est signalé dans chaque encart Sources.",
  ],
  family: "sante-developpement-humain",
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
