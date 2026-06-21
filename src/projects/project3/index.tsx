import { lazy } from "react";
import type { ProjectConfig } from "../types";

export const config: ProjectConfig = {
  slug: "education-filles-genre",
  shortTitle: "Éducation & genre",
  title: "Éducation des filles & égalité de genre",
  domain: "Genre & éducation",
  keywords: ["scolarisation", "parité", "alphabétisation"],
  angle: "L'autonomisation commence par l'école.",
  pitch:
    "Indicateurs réels (Banque mondiale, Open Data) d'écarts de scolarisation entre filles et garçons et d'alphabétisation des jeunes femmes, synthétisés dans un index dérivé de progrès.",
  insights: [
    "La parité est largement atteinte ou dépassée au primaire dans plusieurs pays du panel (indice de parité proche de 1), mais l'écart se creuse souvent au secondaire.",
    "L'alphabétisation des jeunes femmes varie fortement d'un pays à l'autre, sans toujours suivre le même classement que la parité de scolarisation.",
    "L'indice de parité primaire (GPI) a une fréquence de publication irrégulière selon les pays : certaines données récentes peuvent manquer, ce qui est visible dans l'encart Sources.",
  ],
  family: "sante-developpement-humain",
  hasMap: true,
  Body: lazy(() => import("./Body")),
};
