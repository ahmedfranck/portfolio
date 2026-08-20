import { useState } from "react";
import { UCPO_COUNTRIES, type UcpoCountryCode, type UcpoCountryProfile } from "../../../../data/projects/ao-ucpo-observatoire";
import { CountryLabel, FilterChips, RecommendationsHub, type RecommendationGroup } from "../shared";

function countryGroups(country: UcpoCountryProfile): readonly RecommendationGroup[] {
  return [
    {
      id: "country-overview",
      title: <span>Fiche pays · Vue générale · <CountryLabel iso3={country.iso3} name={country.name} /></span>,
      items: [
        { id: "country-overview-1", title: "Lecture 1", body: "La progression de la mCPR doit être lue avec l’évolution de la demande satisfaite." },
        { id: "country-overview-2", title: "Lecture 2", body: "Les écarts annuels relèvent ici d’un scénario et non d’une série FPET validée.", variant: "warning" },
        { id: "country-overview-3", title: "Lecture 3", body: "La fiche combine données publiques réelles et modules démonstratifs clairement signalés.", variant: "info" },
      ],
    },
    {
      id: "country-financing",
      title: <span>Fiche pays · Financement · <CountryLabel iso3={country.iso3} name={country.name} /></span>,
      items: [
        { id: "country-financing-1", title: "Lecture 1", body: `La dépendance illustrée à USAID atteint ${country.usaidExposure} % de l’enveloppe.` },
        { id: "country-financing-2", title: "Lecture 2", body: "La part domestique indique la résilience potentielle, pas la qualité d’exécution budgétaire.", variant: "warning" },
        { id: "country-financing-3", title: "Lecture 3", body: "Les montants doivent être rapprochés des comptes nationaux de santé et décaissements bailleurs.", variant: "info" },
      ],
    },
    {
      id: "country-demography",
      title: <span>Fiche pays · Démographie · <CountryLabel iso3={country.iso3} name={country.name} /></span>,
      items: [
        { id: "country-demography-1", title: "Lecture 1", body: "L’ISF donne le contexte démographique mais ne mesure pas à lui seul la performance du programme." },
        { id: "country-demography-2", title: "Lecture 2", body: "La population des femmes de 15–49 ans sert de dénominateur de planification dans ce prototype.", variant: "warning" },
        { id: "country-demography-3", title: "Lecture 3", body: "Comparer les dates d’observation avant toute lecture causale entre mCPR et fécondité.", variant: "info" },
      ],
    },
    {
      id: "country-methods",
      title: <span>Fiche pays · Méthodes · <CountryLabel iso3={country.iso3} name={country.name} /></span>,
      items: [
        { id: "country-methods-1", title: "Lecture 1", body: "Un mix diversifié réduit la dépendance à une seule chaîne d’approvisionnement." },
        { id: "country-methods-2", title: "Lecture 2", body: "La part d’implants et d’injectables éclaire les besoins de formation et de logistique.", variant: "warning" },
        { id: "country-methods-3", title: "Lecture 3", body: "La répartition présentée est illustrative et doit être remplacée par les enquêtes DHS/MICS validées.", variant: "info" },
      ],
    },
    {
      id: "country-impact",
      title: <span>Fiche pays · Impact · <CountryLabel iso3={country.iso3} name={country.name} /></span>,
      items: [
        { id: "country-impact-1", title: "Lecture 1", body: "Les grossesses évitées sont un résultat modélisé, pas un décompte administratif." },
        { id: "country-impact-2", title: "Lecture 2", body: "Le coût par utilisatrice facilite la comparaison, sans remplacer une analyse coût-efficacité complète.", variant: "warning" },
        { id: "country-impact-3", title: "Lecture 3", body: "Les décès maternels évités exigent des hypothèses documentées et une validation technique.", variant: "info" },
      ],
    },
    {
      id: "country-crisis",
      title: <span>Fiche pays · Crise · <CountryLabel iso3={country.iso3} name={country.name} /></span>,
      items: [
        { id: "country-crisis-1", title: "Lecture 1", body: "La continuité des services doit être priorisée dans les zones à déplacement prolongé." },
        { id: "country-crisis-2", title: "Lecture 2", body: "Les ruptures nationales peuvent masquer de fortes disparités infranationales.", variant: "warning" },
        { id: "country-crisis-3", title: "Lecture 3", body: "Le croisement INFORM–IDMC–ACLED doit utiliser des périodes d’observation alignées.", variant: "info" },
      ],
    },
  ];
}

const GLOBAL_GROUPS: readonly RecommendationGroup[] = [
  {
    id: "scope",
    title: "Périmètre & mesure",
    items: [
      { id: "scope-1", title: "Périmètre confirmé", body: "La vue régionale exclut tout pays hors des neuf membres du Partenariat de Ouagadougou." },
      { id: "scope-2", title: "Dates hétérogènes", body: "Le millésime affiché varie selon le pays; comparer une valeur exige de vérifier son année d’observation.", variant: "warning" },
      { id: "scope-3", title: "Lecture responsable", body: "Les indicateurs illustratifs sont explicitement séparés des observations WDI réelles et signalés par un badge dédié.", variant: "info" },
    ],
  },
  {
    id: "financing",
    title: "Financement & résilience",
    items: [
      { id: "financing-1", title: "Exposition bailleur", body: "Une part USAID élevée signale un besoin de scénario de continuité, sans préjuger des décaissements réels.", variant: "warning" },
      { id: "financing-2", title: "Effort domestique", body: "La part nationale sert d’indicateur de résilience financière et doit être rapprochée de l’exécution budgétaire." },
      { id: "financing-3", title: "Validation finale", body: "Les montants seront remplacés par les comptes nationaux, rapports FPSA et données bailleurs validés.", variant: "info" },
    ],
  },
  {
    id: "risk",
    title: "Risque & continuité",
    items: [
      { id: "risk-1", title: "Signal de risque", body: "Un indice élevé devient critique lorsqu’il coïncide avec une rupture de produits et des déplacements massifs.", variant: "crisis" },
      { id: "risk-2", title: "Décision logistique", body: "Le tableau sert à prioriser le prépositionnement; il ne constitue pas une alerte opérationnelle réelle.", variant: "warning" },
      { id: "risk-3", title: "Validation requise", body: "Les agrégats INFORM, IDMC et ACLED devront être datés, rapprochés et validés avant publication finale.", variant: "info" },
    ],
  },
  {
    id: "methods",
    title: "Méthode & sources",
    items: [
      { id: "methods-1", title: "Principe de priorité", body: "Une donnée publique gratuite et sans clé remplace le scénario dès qu’elle est disponible et documentée." },
      { id: "methods-2", title: "Accès restreint", body: "ACLED requiert un compte; aucune valeur ACLED n’est présentée comme observation réelle dans cette version.", variant: "warning" },
      { id: "methods-3", title: "Reproductibilité", body: "Le manifeste centralise statut, URL, source et règle d’usage pour faciliter le remplacement dataset par dataset.", variant: "info" },
    ],
  },
];

export default function UcpoRecommendations() {
  const [selected, setSelected] = useState<UcpoCountryCode>("SEN");
  const country = UCPO_COUNTRIES.find((item) => item.iso3 === selected)!;

  return (
    <RecommendationsHub
      title="Lecture & recommandations"
      subtitle="Les commentaires humains sont regroupés ici afin de laisser les onglets analytiques centrés sur les données."
      groups={[...GLOBAL_GROUPS, ...countryGroups(country)]}
      controls={(
        <FilterChips
          label="Lectures des fiches pays"
          options={UCPO_COUNTRIES.map((item) => ({ value: item.iso3, label: <CountryLabel iso3={item.iso3} name={item.shortName} /> }))}
          value={[selected]}
          onChange={(values) => setSelected(values[0] as UcpoCountryCode)}
          multiple={false}
        />
      )}
    />
  );
}
