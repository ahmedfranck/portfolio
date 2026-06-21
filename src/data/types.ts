export interface MaternalRow {
  iso3: string;
  year: number;
  maternalMortalityRatio: number | null; // pour 100 000 naissances vivantes — SH.STA.MMRT
  prenatalCareCoverage: number | null; // % femmes enceintes, au moins 1 visite — SH.STA.ANVC.ZS
  skilledBirthAttendance: number | null; // % accouchements assistés — SH.STA.BRTC.ZS
  neonatalMortality: number | null; // pour 1 000 naissances vivantes — SH.DYN.NMRT
}

export interface NutritionRow {
  iso3: string;
  year: number;
  stunting: number | null; // % retard de croissance — SH.STA.STNT.ZS
  wasting: number | null; // % émaciation — SH.STA.WAST.ZS
  underweight: number | null; // % insuffisance pondérale — SH.STA.MALN.ZS
  dtp3Coverage: number | null; // % couverture DTC3 — SH.IMM.IDPT
  measlesCoverage: number | null; // % couverture rougeole — SH.IMM.MEAS
  gdpPerCapita: number | null; // USD courants — NY.GDP.PCAP.CD
}

export interface EducationRow {
  iso3: string;
  year: number;
  enrollmentPrimaryFemale: number | null; // SE.PRM.ENRR.FE
  enrollmentPrimaryMale: number | null; // SE.PRM.ENRR.MA
  enrollmentSecondaryFemale: number | null; // SE.SEC.ENRR.FE
  enrollmentSecondaryMale: number | null; // SE.SEC.ENRR.MA
  parityPrimary: number | null; // indice de parité (GPI) — SE.ENR.PRIM.FM.ZS
  literacyYoungWomen: number | null; // % femmes 15-24 ans — SE.ADT.1524.LT.FE.ZS
}

export interface ReproductiveHealthRow {
  iso3: string;
  year: number;
  mcpr: number | null; // % prévalence contraceptive toutes méthodes — SP.DYN.CONU.ZS
  mcprModern: number | null; // % prévalence contraceptive méthodes modernes — SP.DYN.CONM.ZS
  tfr: number | null; // naissances par femme — SP.DYN.TFRT.IN
  demandSatisfied: number | null; // % demande satisfaite par méthodes modernes — SH.FPL.SATM.ZS
}

export interface WashRow {
  iso3: string;
  year: number;
  waterAccess: number | null; // % — SH.H2O.BASW.ZS
  basicSanitation: number | null; // % — SH.STA.BASS.ZS
  openDefecation: number | null; // % — SH.STA.ODFC.ZS
  childMortality: number | null; // pour 1 000 naissances vivantes (-5 ans) — SH.DYN.MORT
  populationMillions: number | null; // SP.POP.TOTL / 1e6
}

export interface DisplacementRow {
  iso3: string;
  year: number;
  refugees: number | null; // pays d'asile — SM.POP.REFG
  refugeesOrigin: number | null; // pays d'origine — SM.POP.REFG.OR
  idp: number | null; // déplacés internes (conflits) — VC.IDP.TOCV
  populationMillions: number | null; // SP.POP.TOTL / 1e6
}

export interface EconomyRow {
  iso3: string;
  year: number;
  gdpGrowth: number | null; // % annuel — NY.GDP.MKTP.KD.ZG
  gdpPerCapita: number | null; // USD courants — NY.GDP.PCAP.CD
  inflation: number | null; // % annuel — FP.CPI.TOTL.ZG
  fdiPercentGdp: number | null; // % du PIB — BX.KLT.DINV.WD.GD.ZS
}

export interface AgricultureRow {
  iso3: string;
  year: number;
  agricultureValueAdded: number | null; // % du PIB — NV.AGR.TOTL.ZS
  cerealYield: number | null; // kg/hectare — AG.YLD.CREL.KG
  foodProductionIndex: number | null; // indice 2014-2016=100 — AG.PRD.FOOD.XD
  undernourishment: number | null; // % population — SN.ITK.DEFC.ZS
  arableLand: number | null; // % surface du territoire — AG.LND.ARBL.ZS
}

export interface EnergyRow {
  iso3: string;
  year: number;
  electricityAccess: number | null; // % population — EG.ELC.ACCS.ZS
  cleanCookingAccess: number | null; // % population — EG.CFT.ACCS.ZS
  renewableShare: number | null; // % consommation finale — EG.FEC.RNEW.ZS
  electricityConsumption: number | null; // kWh/habitant — EG.USE.ELEC.KH.PC
}

export interface DigitalInclusionRow {
  iso3: string;
  year: number;
  mobileSubscriptions: number | null; // pour 100 habitants — IT.CEL.SETS.P2
  internetUsers: number | null; // % population — IT.NET.USER.ZS
  accountOwnership: number | null; // % 15 ans et plus (Findex) — FX.OWN.TOTL.ZS
  fixedBroadband: number | null; // pour 100 habitants — IT.NET.BBND.P2
}

export interface EmploymentRow {
  iso3: string;
  year: number;
  youthUnemployment: number | null; // % pop. active 15-24 ans — SL.UEM.1524.ZS
  totalUnemployment: number | null; // % pop. active — SL.UEM.TOTL.ZS
  vulnerableEmployment: number | null; // % emploi total — SL.EMP.VULN.ZS
  laborForceParticipation: number | null; // % pop. 15+ — SL.TLF.CACT.ZS
}

export interface ClimateRow {
  iso3: string;
  year: number;
  co2PerCapita: number | null; // tonnes/habitant — EN.GHG.CO2.PC.CE.AR5 (substitut actif de EN.ATM.CO2E.PC, archivé)
  forestArea: number | null; // % surface du territoire — AG.LND.FRST.ZS
  renewableWaterPerCapita: number | null; // m³/habitant — ER.H2O.INTR.PC
  pm25Exposure: number | null; // µg/m³ — EN.ATM.PM25.MC.M3
  protectedAreas: number | null; // % surface du territoire — ER.LND.PTLD.ZS
}
