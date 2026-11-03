# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
#
# SPDX-License-Identifier: MIT

import logging

import pypsa
from snakemake.script import snakemake

from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)

logger = logging.getLogger(__name__)

configure_logging(snakemake)

n = pypsa.Network(snakemake.input.network)  # ty: ignore[possibly-missing-attribute]
resolution = snakemake.params.time_resolution


def average_every_nhours(n: pypsa.Network, offset: str) -> pypsa.Network:
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(snapshots=[])

    # Use closed="right" for consistent binning
    snapshot_weightings = n.snapshot_weightings.resample(
        offset, closed="right", label="right"
    ).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.components:
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.dynamic.items():
            if not df.empty:
                # Use the same closed="right" parameter as snapshot_weightings
                resampled = df.resample(offset, closed="right", label="right").mean()

                # Reindex to match the new snapshots exactly (handles any misalignment)
                resampled = resampled.reindex(m.snapshots)

                # Fill any remaining NaN values that can occur at boundaries
                if resampled.isna().any().any():
                    logger.debug(
                        f"Found NaN values in {c.list_name}.{k} after resampling. "
                        f"Filling with forward-fill then backward-fill."
                    )
                    # Forward-fill first (handles start boundary NaNs)
                    resampled = resampled.ffill()
                    # Backward-fill to handle end boundary NaNs
                    resampled = resampled.bfill()

                    # Check if any NaNs remain
                    if resampled.isna().any().any():
                        logger.warning(
                            f"Still have NaN values in {c.list_name}.{k} after filling. "
                            f"This may indicate missing data in the original timeseries."
                        )

                pnl[k] = resampled

    return m


m = average_every_nhours(n, resolution)

m.export_to_netcdf(
    snakemake.output.network, compression=snakemake.config["compression"]
)  # ty: ignore[possibly-missing-attribute]
