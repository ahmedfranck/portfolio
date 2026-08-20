import { UCPO_COUNTRIES, UCPO_DATASETS } from "../../../../data/projects/ao-ucpo-observatoire";
import { BubbleScatter } from "../charts";
import { DataTable, Panel, RankList } from "../shared";

const crisisPoints = UCPO_COUNTRIES.map((country) => ({
  id: country.iso3,
  name: country.name,
  x: country.informRisk,
  y: country.stockoutRate,
  z: Math.max(20, country.displacedThousands),
  group: country.informRisk >= 7 ? "Risque élevé" : country.informRisk >= 5 ? "Risque moyen" : "Risque contenu",
}));

export default function CrisisModule() {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(270px,.8fr)]">
        <Panel title="Pression de crise sur la continuité contraceptive" subtitle="Taille des bulles : personnes déplacées internes (milliers).">
          <BubbleScatter
            data={crisisPoints}
            xLabel="Indice INFORM"
            yLabel="Ruptures de stock"
            zLabel="PDI"
            yUnit=" %"
            zUnit=" k"
            ariaLabel="Relation entre risque INFORM, ruptures de stock et déplacements internes"
            source={UCPO_DATASETS.crisis.source}
            illustrative
          />
        </Panel>
        <Panel title="Pays à surveiller" subtitle="Classement composite par niveau de risque INFORM.">
          <RankList
            items={UCPO_COUNTRIES.map((country) => ({ id: country.iso3, label: country.name, value: country.informRisk, displayValue: country.informRisk.toFixed(1), detail: `${country.stockoutRate} % ruptures` }))}
            source={UCPO_DATASETS.crisis.source}
            illustrative
          />
        </Panel>
      </div>

      <Panel title="Registre de continuité des services" subtitle="Lecture opérationnelle par pays.">
        <DataTable
          rows={UCPO_COUNTRIES}
          rowKey={(row) => row.iso3}
          columns={[
            { id: "country", header: "Pays", accessor: (row) => row.name },
            { id: "inform", header: "INFORM", accessor: (row) => row.informRisk, align: "right" },
            { id: "pdi", header: "PDI (milliers)", accessor: (row) => row.displacedThousands, align: "right" },
            { id: "stock", header: "Ruptures", accessor: (row) => row.stockoutRate, render: (value) => `${value} %`, align: "right" },
            { id: "response", header: "Réponse prioritaire", accessor: (row) => row.informRisk >= 7 ? "Prépositionnement + cliniques mobiles" : row.stockoutRate >= 20 ? "Réallocation des stocks" : "Surveillance renforcée" },
          ]}
          source={UCPO_DATASETS.crisis.source}
          illustrative
          exportFilename="ucpo-contexte-crise"
        />
      </Panel>

    </div>
  );
}
