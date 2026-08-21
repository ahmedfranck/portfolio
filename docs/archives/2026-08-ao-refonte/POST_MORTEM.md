# Post-mortem factuel — refonte « Appels d'offres » abandonnée

Cette archive décrit le chantier annulé le 21 août 2026 et conservé dans l'historique Git.

## Choix de conception documentés
- Thème UCPO navy/or limité au seul projet Santé reproductive confirmé UCPO.
- Maintien de Recharts et création de wrappers thémés pour les graphiques institutionnels.
- Création de primitives partagées pour KPI, panels, tableaux, cartes, notes et chronologies.
- Séparation des filtres transverses pays/année et des sélections locales de comparaison.
- Comparateur multi-pays configurable avec métriques alignées, sparklines et millésimes.
- Séparation explicite entre données publiques réelles et jeux illustratifs signalés.
- Retrait progressif des badges source et illustratifs répétés dans chaque bloc analytique.

## Fichiers et patterns introduits
- `src/themes/appelsOffres.ts` centralisait les identités des organismes détectés.
- `src/components/portfolio/appels-offres/shared/` regroupait les composants génériques.
- `src/components/portfolio/appels-offres/charts/` regroupait les wrappers Recharts.
- `src/components/portfolio/appels-offres/ucpo/` contenait la navigation et les vues UCPO.
- `src/data/projects/ao-ucpo-observatoire.ts` portait manifest, sources et scénarios.
- `ProjectPage.tsx`, `portfolioOrganizations.ts` et `src/index.css` assuraient le scoping.

## Chantier consulting/PowerBI
- Le chantier consulting/PowerBI existait déjà à `db6ff71` et revient à son état d'origine après revert.
- Cinq fichiers WIP non commités étaient extérieurs au périmètre annulé : `ConsultingPortfolio.tsx`, `PowerBIEmbed.tsx`, `DashboardTabs.tsx`, `ChartCard.tsx` et `KpiCard.tsx`.
- Ces évolutions dépendaient de classes CSS ajoutées par la refonte AO dans `bf33ce4`.
- Elles sont préservées dans le stash nommé et dans `pre-revert-worktree.diff`, sans restauration dans le worktree.
- Pour reprendre ces évolutions, il faudra soit réintroduire un sous-ensemble des classes CSS concernées dans un fichier dédié type `src/styles/consulting.css`, soit refactorer les 5 composants WIP pour ne plus dépendre de ces classes.
## Récupération
- Le tag `abandon/ao-refonte-20260821` référence le dernier état complet du chantier avant annulation.
- L'audit métier reste archivé avec ce document pour une éventuelle reprise sélective.
