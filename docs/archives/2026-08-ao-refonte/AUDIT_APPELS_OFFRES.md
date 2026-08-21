# Audit préalable — dashboards « Appels d'offres »

Date de l'audit : 20 août 2026  
Périmètre audité : application React 18 + Vite, routes `/portfolio?category=appels-offres` et `/portfolio/:slug`.

## Synthèse

- Les 12 dashboards enregistrés dans `src/projects/index.ts` sont actuellement classés dans la catégorie `appels-offres`.
- Après correction métier, **un seul projet est confirmé UCPO** : `sante-reproductive-fecondite`. `sante-maternelle-neonatale` est traité non-UCPO tant qu'aucune preuve de financement ou de pilotage UCPO/PO/FP2030 via UCPO n'est fournie. `nutrition-survie-enfant` et `education-filles-genre` sont explicitement hors périmètre UCPO.
- Chaque dashboard possède déjà 4 KPI, une carte interactive, au moins une visualisation temporelle ou comparative, un tableau triable/filtrable, un export CSV et un panneau de sources.
- Les données analytiques couvrent 16 pays et la période 2010–2024. Elles ont été extraites le 21 juin 2026 depuis Banque mondiale Open Data. La dernière observation réellement disponible varie selon l'indicateur et le pays.
- Le portfolio décrit ces projets comme « réalisés dans le cadre d'appels d'offres ». Le résultat administratif de chaque appel d'offres (`gagné`, `soumis`, `en cours`) et sa date contractuelle ne sont pas stockés dans le dépôt. L'audit ne les invente donc pas.
- La refonte UCPO a été amorcée dans `ProjectPage.tsx` et `index.css`, mais elle s'applique actuellement à quatre projets à cause d'une association trop large dans `portfolioOrganizations.ts`. Cette association devra être corrigée après validation de l'audit.
- Le moteur graphique retenu est **Recharts exclusivement**.
- Les futurs enrichissements peuvent utiliser des données illustratives, à condition de respecter le marquage explicite défini dans la section « Règles validées pour les données illustratives ».

## Inventaire, commanditaires et détection UCPO corrigée

| # | Slug | Projet | Commanditaire retenu pour la refonte | isUcpo | Preuve ou limite de preuve | Périmètre thématique |
|---:|---|---|---|:---:|---|---|
| 1 | `sante-maternelle-neonatale` | Santé maternelle & néonatale | **Non confirmé** | **false** | La fiche `src/projects/project1/index.tsx:5-12` ne cite qu'une source de données Banque mondiale. L'affectation UCPO de `portfolioOrganizations.ts:28` n'est accompagnée d'aucun champ de financement/pilotage. En application de la correction utilisateur, le projet reste non-UCPO jusqu'à preuve UCPO/PO/FP2030 via UCPO. | Mortalité maternelle, soins prénatals, accouchement assisté, mortalité néonatale |
| 2 | `nutrition-survie-enfant` | Nutrition & survie de l'enfant | **Non confirmé** | **false** | Correction utilisateur explicite : hors mandat UCPO. La fiche `src/projects/project2/index.tsx:5-12` décrit les indicateurs, sans commanditaire. L'ancienne ligne UCPO `portfolioOrganizations.ts:30` est donc invalide comme classification métier. | Retard de croissance, émaciation, insuffisance pondérale, vaccination, PIB/habitant |
| 3 | `education-filles-genre` | Éducation des filles & égalité de genre | **Non confirmé** | **false** | Correction utilisateur explicite : hors mandat UCPO. La fiche `src/projects/project3/index.tsx:5-12` ne contient aucun commanditaire. L'ancienne ligne UCPO `portfolioOrganizations.ts:29` est donc invalide comme classification métier. | Parité scolaire, scolarisation secondaire, alphabétisation des jeunes femmes |
| 4 | `sante-reproductive-fecondite` | Santé reproductive & fécondité | **UCPO / Partenariat de Ouagadougou** | **true** | Confirmation métier explicite de l'utilisateur le 20 août 2026, cohérente avec `portfolioOrganizations.ts:19-27` et avec la fiche `src/projects/project4/index.tsx:5-16` centrée sur mCPR, méthodes modernes et demande PF satisfaite. La confirmation utilisateur est la preuve décisive ; la proximité thématique seule ne l'aurait pas été. | mCPR, contraception moderne, fécondité, demande satisfaite |
| 5 | `wash-eau-assainissement` | Eau, assainissement & hygiène | **UNICEF — association configurée, provenance contractuelle non documentée** | false | `portfolioOrganizations.ts:50-57` nomme UNICEF et inclut explicitement ce slug. La fiche projet ne contient toutefois pas de champ de financement : l'association est utilisable pour le thème, pas comme preuve d'un contrat. | Eau potable, assainissement, défécation à l'air libre, mortalité des moins de 5 ans |
| 6 | `populations-deplacees` | Populations déplacées & accès aux services | **UNICEF — association configurée, provenance contractuelle non documentée** | false | `portfolioOrganizations.ts:50-57` inclut explicitement ce slug sous UNICEF. Les données HCR mentionnées dans la fiche sont une source statistique, pas un commanditaire. | Réfugiés accueillis/originaires, déplacements internes, fragilité |
| 7 | `economie-croissance` | Économie & croissance | **BAD — association configurée, provenance contractuelle non documentée** | false | `portfolioOrganizations.ts:34-47` rattache explicitement ce slug à la BAD. Aucun champ contractuel distinct n'existe. | Croissance du PIB, PIB/habitant, inflation, IDE |
| 8 | `agriculture-securite-alimentaire` | Agriculture & sécurité alimentaire | **BAD — association configurée, provenance contractuelle non documentée** | false | `portfolioOrganizations.ts:34-47` rattache explicitement ce slug à la BAD. Aucun champ contractuel distinct n'existe. | Valeur ajoutée agricole, rendements, production alimentaire, sous-alimentation |
| 9 | `energie-acces-electricite` | Énergie & accès à l'électricité | **BAD — association configurée, provenance contractuelle non documentée** | false | `portfolioOrganizations.ts:34-47` rattache explicitement ce slug à la BAD. Aucun champ contractuel distinct n'existe. | Électricité, cuisson propre, renouvelables, consommation électrique |
| 10 | `inclusion-numerique-financiere` | Inclusion numérique & financière | **BAD — association configurée, provenance contractuelle non documentée** | false | `portfolioOrganizations.ts:34-47` rattache explicitement ce slug à la BAD. Aucun champ contractuel distinct n'existe. | Mobile, Internet, détention de compte, haut débit fixe |
| 11 | `emploi-jeunesse` | Emploi & jeunesse | **BAD — association configurée, provenance contractuelle non documentée** | false | `portfolioOrganizations.ts:34-47` rattache explicitement ce slug à la BAD. Aucun champ contractuel distinct n'existe. | Chômage des jeunes, chômage total, emploi vulnérable, activité |
| 12 | `climat-environnement` | Climat & environnement | **PNUE — association configurée, provenance contractuelle non documentée** | false | `portfolioOrganizations.ts:60-67` nomme le PNUE et inclut explicitement ce slug. Aucun champ contractuel distinct n'existe. | CO2, forêt, eau renouvelable, PM2.5, aires protégées |

### Conclusion de la vérification de provenance

- Le type `ProjectConfig` ne possède actuellement aucun champ `organization`, `commissioningOrganization`, `funder`, `financingSource` ou `isUcpo`.
- `CALL_FOR_OFFERS_ORGANIZATIONS[].projectSlugs` est la seule association organisme-projet du dépôt. L'historique Git montre qu'elle a été introduite par le commit `8ccc259` (« Group bid projects by reference organization »), sans note de provenance contractuelle.
- Une source statistique (Banque mondiale, HCR, FP2030, DHS, etc.) ne doit pas être confondue avec le commanditaire ou le financeur.
- Classification opérationnelle retenue : **1 UCPO confirmé, 11 non-UCPO**, dont 3 commanditaires encore non confirmés (santé maternelle, nutrition, éducation).
- À l'étape A, aucun thème OMS, UNFPA, MSAS, PAM, UN Women, AFD ou UE ne sera attribué aux trois projets non confirmés sans nouvelle preuve. Un thème neutre et institutionnel pourra être utilisé provisoirement, explicitement marqué `organizationKey: "unverified"`.

### Statut et dates

- Le libellé public du portfolio qualifie les 12 éléments de « projets réalisés dans le cadre d'appels d'offres » ; leur statut de présentation est donc **réalisé**.
- Aucun fichier ne précise le résultat administratif `gagné / soumis / en cours`, la date de soumission, la date d'attribution ou la période contractuelle. Ces champs restent **non documentés** pour les 12 projets.
- La période **2010–2024** est la période des données analytiques, pas celle des mandats. Les fichiers ont été extraits le **21 juin 2026** depuis Banque mondiale Open Data.

## État actuel par dashboard

Tous les projets ont un sélecteur de pays, un sélecteur d'année, 4 KPI, un tableau maître, une recherche textuelle, un tri par colonne, un export CSV et un panneau de sources. Le tableau ci-dessous recense les différences utiles.

| Projet | Visuels analytiques actuels | Carte | Fiche détaillée | Filtres | Storytelling actuel |
|---|---:|:---:|:---:|:---:|---|
| Santé maternelle | 3 + 4 mini-séries | Oui, clic vers fiche | Oui, 1 vue / 4 indicateurs | Pays + année + pays fiche | 1 insight dans le dashboard + 3 enseignements exécutifs |
| Nutrition enfant | 5 (classement, carte, scatter, small multiples, jauges) | Oui | Non | Indicateur + pays + année | 1 insight + 3 enseignements |
| Éducation & genre | 5 | Oui | Non | Pays + année | 1 insight + 3 enseignements |
| Santé reproductive | 5 | Oui, sans drill-down | Non | Pays + année | 1 insight + 3 enseignements |
| WASH | 5, dont carte à bulles | Oui | Non | Pays + année | 1 insight + 3 enseignements |
| Populations déplacées | 4 + 3 mini-séries | Oui, clic vers fiche | Oui, 1 vue / 3 indicateurs | Indicateur + pays + année + pays fiche | 1 insight + 3 enseignements |
| Économie & croissance | 5 | Oui | Non | Pays + année | 1 insight + 3 enseignements |
| Agriculture | 3 + 5 mini-séries | Oui, clic vers fiche | Oui, 1 vue / 5 indicateurs | Indicateur + pays + année + pays fiche | 1 insight + 3 enseignements |
| Énergie | 5 | Oui | Non | Pays + année | 1 insight + 3 enseignements |
| Inclusion numérique | 5 | Oui | Non | Pays + année | 1 insight + 3 enseignements |
| Emploi & jeunesse | 5 | Oui | Non | Pays + année | 1 insight + 3 enseignements |
| Climat & environnement | 3 + 5 mini-séries | Oui, clic vers fiche | Oui, 1 vue / 5 indicateurs | Indicateur + pays + année + pays fiche | 1 insight + 3 enseignements |

## Écarts communs par rapport au nouveau cahier

### Présents ou largement présents

- séries 2010–2024 lorsque la source publie des observations ;
- cartes choroplèthes SVG interactives avec zoom et info-bulles ;
- classements, comparaisons, nuages de points et quelques visuels spécialisés ;
- tableaux maîtres triables et filtrables ;
- export CSV ;
- sources Banque mondiale visibles dans un panneau dédié ;
- valeurs manquantes affichées `n.d.` sans estimation ;
- lazy loading des corps de dashboards au niveau de chaque projet.

### Absents ou insuffisants sur les 12 projets

- aucun `sum-band` complet avec 3 à 5 KPI transverses en plus de la rangée KPI ;
- pas de bloc de financement structuré (budget, part domestique, dépendance PTF, exposition aux chocs) ;
- pas de module impact/ROI avec données sourcées ;
- pas de timeline d'actualités ou d'événements structurants ;
- moins de trois callouts interprétatifs structurés par page ;
- pas de répartition de type doughnut/stacked 100 % sur la majorité des pages ;
- pas de matrice de risque dédiée ;
- fiches détaillées absentes sur 8 projets et limitées à une seule vue sur les 4 autres ;
- sources visibles dans le panneau final, mais pas encore sous chaque graphique/tableau sous forme de badge ;
- export Excel absent ; CSV présent ;
- métadonnées projet incomplètes : budget, bénéficiaires, statut administratif AO, date contractuelle et méthodologie propre au mandat ne sont pas stockés ;
- aucun fichier de dictionnaire FR/EN ;
- aucune preuve Lighthouse ni jeu de captures desktop/mobile dans le dépôt.

## Écarts spécifiques UCPO

- ces écarts concernent désormais uniquement `sante-reproductive-fecondite` ;
- la palette de base UCPO est partiellement injectée, mais les tokens ne sont pas encore centralisés dans un thème typé et scoping complet ;
- Chelsea Market et Montserrat ainsi que plusieurs styles UCPO existent déjà dans la couche CSS en cours de refonte, mais leur application reste à vérifier composant par composant ;
- absence de sidebar UCPO 260 px et de drawer responsive dédié ;
- absence des neuf fiches pays UCPO (BJ, BF, CI, GN, ML, MR, NE, SN, TG) à six onglets ;
- absence du Motion Tracker mCPR depuis 2011 ;
- absence du module crise (INFORM, ACLED, PDI, ruptures de stock) ;
- la carte du projet Santé reproductive couvre actuellement 16 pays et utilise les indicateurs Banque mondiale du projet, pas un dataset UCPO enrichi limité aux neuf pays du PO ;
- aucun dataset Track20, DHS, MICS, PMA, FP2030, ACLED, IDMC, INFORM ou financement n'est présent.

## Écarts spécifiques non-UCPO

- des variables locales différencient déjà BAD, UNICEF et PNUE, mais il n'existe pas encore de fichier de thèmes dédié et typé ; les trois projets retirés du groupe UCPO n'ont pas encore de commanditaire confirmé ;
- les composants graphiques partagent encore une palette globale : l'absence totale des couleurs UCPO dans les projets non-UCPO n'est donc pas garantie ;
- les pages reprennent la même architecture générique sans modules financement/impact/risque propres au mandat de chaque organisme.

## Données nécessaires avant enrichissement factuel

Pour respecter l'interdiction d'inventer des chiffres, les nouveaux modules suivants exigent soit des sources réelles supplémentaires, soit des données fictives explicitement labellisées :

1. statut `gagné / en cours / soumis`, date, budget et bénéficiaires de chaque mandat ;
2. financements domestiques/PTF et exposition USAID ;
3. impacts attribuables (grossesses/décès évités, coût par bénéficiaire, ROI social) ;
4. séries UCPO/Track20/FP2030 et mix de méthodes ;
5. données de crise INFORM, ACLED, IDMC/PDI et ruptures de stock ;
6. événements datés pour les timelines ;
7. lots, régions ou trimestres propres aux mandats lorsque le drill-down par pays n'est pas pertinent.

## Règles validées pour les données illustratives

En l'absence de données publiques réelles gratuites et sans clé, les composants seront construits avec des valeurs plausibles mais explicitement démonstratives :

1. chaque dataset concerné portera `illustrative: true` dans son manifest typé ;
2. chaque KPI, tableau ou graphique alimenté par ces valeurs affichera un badge visible **« Données illustratives »**, sur fond orange clair, en 8–9 px, à côté du badge source ;
3. chaque page concernée affichera en pied : « Données présentées à des fins de démonstration. Les valeurs réelles issues des enquêtes citées seront intégrées à la livraison finale. » ;
4. les ordres de grandeur seront cohérents avec la littérature sectorielle citée, sans présenter ces valeurs comme des observations ;
5. les sources publiques réelles (Track20/FP2030, Banque mondiale, UNFPA World Population Dashboard et autres sources gratuites sans clé) seront prioritaires ;
6. les observations réelles et illustratives seront séparées dans les manifests afin d'empêcher un mélange silencieux dans un même indicateur.

## Recommandation technique issue de l'audit

Conserver **Recharts**, déjà utilisé par tous les dashboards. Les patterns visuels UCPO seront traduits vers son API : courbes monotones avec rendu visuel proche d'une tension 0,35, coins de barres de 3–4 px, palette UCPO isolée, points masqués hors survol et info-bulles en français. L'architecture proposée dans le brief doit être adaptée à React/Vite (`src/components`, `src/themes`, `src/data`) sans migration Next.js.
