import { UCPO_COUNTRIES, UCPO_DATASETS, UCPO_MOTION_SERIES, type UcpoCountryCode } from "../../../../data/projects/ao-ucpo-observatoire";
import { LineTrend } from "../charts";

interface MotionTrackerProps {
  readonly countries: readonly UcpoCountryCode[];
}

export default function MotionTracker({ countries }: MotionTrackerProps) {
  const series = countries.map((iso3) => {
    const country = UCPO_COUNTRIES.find((item) => item.iso3 === iso3)!;
    return { dataKey: iso3, name: country.name, unit: " %" };
  });

  return (
    <LineTrend
      data={UCPO_MOTION_SERIES}
      xKey="year"
      xLabel="Année"
      series={series}
      yLabel="mCPR (%)"
      height={360}
      ariaLabel="Motion Tracker de la prévalence contraceptive moderne dans les pays UCPO"
      source={UCPO_DATASETS.motion.source}
      illustrative={UCPO_DATASETS.motion.illustrative}
      formatValue={(value) => `${value.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} %`}
    />
  );
}
