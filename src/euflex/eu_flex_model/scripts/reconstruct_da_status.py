import logging
from pathlib import Path

import pandas as pd

from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    snakemake = get_snakemake(globals_scope=globals(), rule="reconstruct_da_status")
    configure_logging(snakemake)

    centroid_paths = snakemake.input.centroid_statuses
    if isinstance(centroid_paths, str):
        centroid_paths = [centroid_paths]

    csv_paths = [p for p in centroid_paths if str(p).endswith(".csv")]

    logger.info(f"Concatenating {len(csv_paths)} centroid status CSV files.")

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame(
            columns=[
                "horizon_index",
                "snapshot_start",
                "snapshot_end",
                "status",
                "condition",
            ]
        )

    output_path = Path(snakemake.output.status_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Wrote reconstructed DA status to {output_path}")
