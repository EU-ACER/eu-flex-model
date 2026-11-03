import logging

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def slice_network_to_centroid(
    n: pypsa.Network,
    week_clusters: pd.DataFrame,
    cluster_id: int,
) -> pypsa.Network:
    """
    Slice a full-year network to the snapshots of a single centroid chunk.

    Parameters
    ----------
    n:
        Full-year base PyPSA network.
    week_clusters:
        DataFrame from week_clusters_{ty}_{it}.csv (output of cluster_weeks.py).
        Must contain columns: chunk_id, chunk_start, chunk_end, cluster_id,
        centroid_flag, weight.
    cluster_id:
        Which cluster's medoid to extract.

    Returns
    -------
    pypsa.Network
        Copy of n restricted to snapshots in [chunk_start, chunk_end], with
        snapshot_weightings["objective"] set to cluster_weight throughout.
    """
    centroid_row = week_clusters[
        (week_clusters["cluster_id"] == cluster_id) & week_clusters["centroid_flag"]
    ]
    if centroid_row.empty:
        raise ValueError(
            f"No centroid found for cluster_id={cluster_id} in week_clusters."
        )

    chunk_id = centroid_row["chunk_id"].iloc[0]
    cluster_weight = float(centroid_row["weight"].iloc[0])
    chunk_start = pd.Timestamp(centroid_row["chunk_start"].iloc[0])
    chunk_end = pd.Timestamp(centroid_row["chunk_end"].iloc[0])

    centroid_snapshots = n.snapshots[
        (n.snapshots >= chunk_start) & (n.snapshots <= chunk_end)
    ]

    if len(centroid_snapshots) == 0:
        raise ValueError(
            f"No snapshots found in network for chunk_id={chunk_id} "
            f"({chunk_start} to {chunk_end})."
        )

    m = n.copy(snapshots=centroid_snapshots)
    # Scale weight so that sum(w * n_snapshots) across all centroids = total_snapshots_in_year.
    # cluster_weight is a fraction (sums to 1.0 globally); multiplying by the full-year
    # snapshot count converts it to hour-equivalents so PyPSA statistics.revenue() integrates
    # correctly over the full year.
    n_snapshots_full_year = len(n.snapshots)
    n_snapshots_chunk = len(centroid_snapshots)
    scaled_weight = cluster_weight * n_snapshots_full_year / n_snapshots_chunk
    # Set all three weighting columns: statistics.revenue() uses "generators" for
    # Generator components and "stores" for StorageUnit/Store -- not "objective".
    for col in m.snapshot_weightings.columns:
        m.snapshot_weightings[col] = scaled_weight
    m.meta["cluster_id"] = cluster_id
    m.meta["cluster_weight"] = cluster_weight  # fractional weight (sums to 1.0)
    m.meta["cluster_weight_scaled"] = (
        scaled_weight  # hour-equivalent weight for statistics
    )
    m.meta["chunk_id"] = chunk_id

    logger.info(
        f"Sliced centroid network: cluster_id={cluster_id}, chunk_id={chunk_id}, "
        f"weight={cluster_weight:.6f}, snapshots={len(m.snapshots)}"
    )
    return m


if __name__ == "__main__":
    from snakemake.script import snakemake

    from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
        configure_logging,
    )

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)  # ty: ignore[possibly-missing-attribute]
    week_clusters = pd.read_csv(
        snakemake.input.week_clusters
    )  # ty: ignore[possibly-missing-attribute]
    cluster_id = int(
        snakemake.wildcards.cluster_id
    )  # ty: ignore[possibly-missing-attribute]

    m = slice_network_to_centroid(n, week_clusters, cluster_id)

    # Use update() to preserve cluster_weight_scaled and chunk_id set by the function above.
    m.meta.update(dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards))))
    m.meta["cluster_id"] = cluster_id  # ensure int, not wildcard string

    m.export_to_netcdf(
        snakemake.output.network
    )  # ty: ignore[possibly-missing-attribute]
    logger.info(
        f"Wrote {snakemake.output.network}"
    )  # ty: ignore[possibly-missing-attribute]
