# Sources des données

Toutes les données de ce portfolio sont **réelles**, récupérées via l'API ouverte de la
**Banque mondiale (World Bank Open Data)** — licence **CC BY-4.0**.

- API : `https://api.worldbank.org/v2/country/{ISO3}/indicator/{CODE}`
- Pays couverts (16, Afrique de l'Ouest et Centrale) : Bénin, Burkina Faso, Cameroun, Côte
  d'Ivoire, Ghana, Guinée, Libéria, Mali, Mauritanie, Niger, Nigéria, RDC, Sénégal, Sierra Leone,
  Tchad, Togo.
- Période demandée : 2010–2024. Selon l'indicateur, les données réellement disponibles peuvent
  s'arrêter avant 2024 (décalage de publication propre à chaque série statistique).
- **Aucune valeur manquante n'est fabriquée.** Lorsqu'un pays n'a pas de donnée pour une année
  donnée, le champ correspondant est `null` et l'interface affiche « n.d. ».
- Récupération via [`scripts/fetch_data.mjs`](../../scripts/fetch_data.mjs) — régénérable avec
  `npm run fetch-data`. Les fichiers JSON produits dans `src/data/` sont committés pour permettre
  un fonctionnement hors-ligne en build statique.

Chaque page projet affiche un encart **« Sources & méthodologie »** listant, pour les indicateurs
qu'elle utilise, le libellé, le code Banque mondiale, la plage d'années réellement disponible, la
couverture pays et un lien vers `data.worldbank.org`.

---

## Santé maternelle & néonatale (`project1.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Ratio de mortalité maternelle | `SH.STA.MMRT` | pour 100 000 naissances vivantes | 2010–2023 | 16/16 pays |
| Femmes enceintes ayant reçu des soins prénatals | `SH.STA.ANVC.ZS` | % | 2010–2021 | 16/16 pays |
| Accouchements assistés par personnel qualifié | `SH.STA.BRTC.ZS` | % | 2010–2022 | 16/16 pays |
| Mortalité néonatale | `SH.DYN.NMRT` | pour 1 000 naissances vivantes | 2010–2024 | 16/16 pays |

## Nutrition & survie de l'enfant (`project2.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Retard de croissance (stunting) | `SH.STA.STNT.ZS` | % | 2010–2024 | 16/16 pays |
| Émaciation (wasting) | `SH.STA.WAST.ZS` | % | 2010–2024 | 16/16 pays |
| Insuffisance pondérale | `SH.STA.MALN.ZS` | % | 2010–2024 | 16/16 pays |
| Couverture vaccinale DTC3 | `SH.IMM.IDPT` | % | 2010–2024 | 16/16 pays |
| Couverture vaccinale rougeole | `SH.IMM.MEAS` | % | 2010–2024 | 16/16 pays |
| PIB par habitant | `NY.GDP.PCAP.CD` | USD courants | 2010–2024 | 16/16 pays |

## Éducation des filles & égalité de genre (`project3.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Scolarisation primaire, filles (brut) | `SE.PRM.ENRR.FE` | % | 2010–2024 | 16/16 pays |
| Scolarisation primaire, garçons (brut) | `SE.PRM.ENRR.MA` | % | 2010–2024 | 16/16 pays |
| Scolarisation secondaire, filles (brut) | `SE.SEC.ENRR.FE` | % | 2010–2024 | 16/16 pays |
| Scolarisation secondaire, garçons (brut) | `SE.SEC.ENRR.MA` | % | 2010–2024 | 16/16 pays |
| Indice de parité filles/garçons, primaire (GPI) | `SE.ENR.PRIM.FM.ZS` | ratio | 2010–2022 | 16/16 pays |
| Alphabétisation des jeunes femmes (15-24 ans) | `SE.ADT.1524.LT.FE.ZS` | % | 2010–2024 | 16/16 pays |

> **Indicateur dérivé.** L'« index de progrès » affiché sur cette page est calculé par ce
> portfolio (non officiel) : moyenne de deux sous-scores normalisés (0–100) — la parité de
> scolarisation secondaire (filles ÷ garçons) et l'alphabétisation des jeunes femmes — à partir
> des dernières valeurs réelles disponibles pour chaque indicateur.

## Santé reproductive & fécondité (`project4.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Prévalence contraceptive, toutes méthodes | `SP.DYN.CONU.ZS` | % des femmes mariées 15-49 ans | 2010–2023 | 16/16 pays |
| Prévalence contraceptive, méthodes modernes | `SP.DYN.CONM.ZS` | % des femmes mariées 15-49 ans | 2010–2023 | 16/16 pays |
| Indice synthétique de fécondité | `SP.DYN.TFRT.IN` | naissances par femme | 2010–2024 | 16/16 pays |
| Demande satisfaite par méthodes modernes | `SH.FPL.SATM.ZS` | % des femmes mariées avec demande | 2010–2023 | 16/16 pays |

> Ce projet a été recadré pour rester 100 % réel : la Banque mondiale ne publie pas de
> répartition détaillée par méthode contraceptive (method-mix). Les indicateurs ci-dessus
> (prévalence, fécondité, demande satisfaite) sont tous des séries officielles.

## Eau, assainissement & hygiène — WASH (`project5.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Accès à l'eau potable de base | `SH.H2O.BASW.ZS` | % | 2010–2024 | 16/16 pays |
| Accès à l'assainissement de base | `SH.STA.BASS.ZS` | % | 2010–2024 | 16/16 pays |
| Défécation à l'air libre | `SH.STA.ODFC.ZS` | % | 2010–2024 | 16/16 pays |
| Mortalité des moins de 5 ans | `SH.DYN.MORT` | pour 1 000 naissances vivantes | 2010–2024 | 16/16 pays |
| Population totale | `SP.POP.TOTL` | millions d'habitants | 2010–2024 | 16/16 pays |

## Populations déplacées & accès aux services (`project6.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Réfugiés sous mandat HCR (pays d'asile) | `SM.POP.RHCR.EA` | personnes | 2010–2024 | 16/16 pays |
| Réfugiés sous mandat HCR (pays d'origine) | `SM.POP.RHCR.EO` | personnes | 2010–2024 | 16/16 pays |
| Déplacés internes (pays d'asile/origine) | `SM.POP.IDPC` | personnes | 2010–2024 | 14/16 pays |
| Population totale | `SP.POP.TOTL` | millions d'habitants | 2010–2024 | 16/16 pays |

> **Note technique.** Les codes initialement prévus par le cahier des charges
> (`SM.POP.REFG`, `SM.POP.REFG.OR`, `VC.IDP.TOCV`) sont **archivés côté API** (statut « WDI
> Database Archives » : la métadonnée existe mais aucune donnée n'est plus servie). Ils ont été
> remplacés par leurs équivalents actifs ci-dessus, vérifiés avant intégration (voir
> `scripts/fetch_data.mjs`).

## Économie & croissance (`project7.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Croissance du PIB | `NY.GDP.MKTP.KD.ZG` | % annuel | 2010–2024 | 16/16 pays |
| PIB par habitant | `NY.GDP.PCAP.CD` | USD courants | 2010–2024 | 16/16 pays |
| Inflation des prix à la consommation | `FP.CPI.TOTL.ZG` | % annuel | 2010–2024 | 16/16 pays |
| Investissements directs étrangers, entrants | `BX.KLT.DINV.WD.GD.ZS` | % du PIB | 2010–2024 | 16/16 pays |

> **Indicateur exclu.** `GC.DOD.TOTL.GD.ZS` (dette publique/PIB), prévu au cahier des charges,
> n'est renseigné que pour 2/16 pays sur la période (7 observations) : trop épars pour être
> exploitable, il a été retiré du dashboard.

## Agriculture & sécurité alimentaire (`project8.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Valeur ajoutée agricole | `NV.AGR.TOTL.ZS` | % du PIB | 2010–2024 | 16/16 pays |
| Rendement céréalier | `AG.YLD.CREL.KG` | kg/hectare | 2010–2024 | 16/16 pays |
| Indice de production alimentaire | `AG.PRD.FOOD.XD` | indice (2014-2016=100) | 2010–2022 | 16/16 pays |
| Prévalence de la sous-alimentation | `SN.ITK.DEFC.ZS` | % de la population | 2010–2023 | 16/16 pays |
| Terres arables | `AG.LND.ARBL.ZS` | % de la surface du territoire | 2010–2023 | 16/16 pays |

## Énergie & accès à l'électricité (`project9.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Accès à l'électricité | `EG.ELC.ACCS.ZS` | % de la population | 2010–2023 | 16/16 pays |
| Accès à des combustibles de cuisson propres | `EG.CFT.ACCS.ZS` | % de la population | 2010–2023 | 16/16 pays |
| Part des énergies renouvelables | `EG.FEC.RNEW.ZS` | % de la consommation finale | 2010–2022 | 16/16 pays |
| Consommation d'électricité par habitant | `EG.USE.ELEC.KH.PC` | kWh/habitant | 2010–2023 | 11/16 pays |

> **Couverture partielle.** `EG.USE.ELEC.KH.PC` n'est disponible que pour 11 des 16 pays : les
> pays sans donnée affichent « n.d. », jamais de valeur estimée.

## Inclusion numérique & financière (`project10.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Abonnements à la téléphonie mobile | `IT.CEL.SETS.P2` | pour 100 habitants | 2010–2024 | 16/16 pays |
| Utilisateurs d'Internet | `IT.NET.USER.ZS` | % de la population | 2010–2024 | 16/16 pays |
| Détention d'un compte (banque/mobile money) | `FX.OWN.TOTL.ZS` | % des 15 ans et plus | 2011–2024 | 16/16 pays |
| Abonnements au haut débit fixe | `IT.NET.BBND.P2` | pour 100 habitants | 2010–2024 | 16/16 pays |

> **Enquêtes Findex.** `FX.OWN.TOTL.ZS` provient des enquêtes Global Findex, publiées par vagues
> espacées (2011, 2014, 2017, 2021, 2022...) plutôt qu'annuellement : les années intermédiaires
> sans enquête affichent « n.d. ».

## Emploi & jeunesse (`project11.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Chômage des jeunes (15-24 ans) | `SL.UEM.1524.ZS` | % de la pop. active 15-24 ans | 2010–2024 | 16/16 pays |
| Chômage total | `SL.UEM.TOTL.ZS` | % de la population active | 2010–2024 | 16/16 pays |
| Emploi vulnérable | `SL.EMP.VULN.ZS` | % de l'emploi total | 2010–2024 | 16/16 pays |
| Taux d'activité | `SL.TLF.CACT.ZS` | % de la population 15+ | 2010–2024 | 16/16 pays |

## Climat & environnement (`project12.json`)

| Indicateur | Code | Unité | Années disponibles | Couverture |
|---|---|---|---|---|
| Émissions de CO₂ par habitant | `EN.GHG.CO2.PC.CE.AR5` | tonnes/habitant | 2010–2024 | 16/16 pays |
| Couvert forestier | `AG.LND.FRST.ZS` | % de la surface du territoire | 2010–2023 | 16/16 pays |
| Ressources en eau douce renouvelables / hab. | `ER.H2O.INTR.PC` | m³/habitant | 2010–2022 | 16/16 pays |
| Exposition aux particules fines (PM2.5) | `EN.ATM.PM25.MC.M3` | µg/m³ | 2010–2020 | 16/16 pays |
| Aires terrestres protégées | `ER.LND.PTLD.ZS` | % de la surface du territoire | 2013–2024 | 16/16 pays |

> **Note technique.** Le code initialement prévu par le cahier des charges (`EN.ATM.CO2E.PC`) est
> **archivé côté API** (statut « WDI Database Archives »). Il a été remplacé par son équivalent
> actif `EN.GHG.CO2.PC.CE.AR5`, vérifié avant intégration (voir `scripts/fetch_data.mjs`).

---

## Reproductibilité

```bash
npm run fetch-data
```

Régénère les 12 fichiers JSON de `src/data/` à partir de l'API Banque mondiale en direct, et
affiche dans la console la couverture obtenue (pays/années) pour chaque indicateur.
