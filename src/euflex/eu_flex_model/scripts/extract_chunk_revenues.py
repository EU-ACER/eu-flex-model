import logging

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def _iter_nday_chunks(snapshots: pd.DatetimeIndex, horizon_days: int):
    """
    Yield (chunk_label, chunk_snapshots) for fixed-length N-day windows.

    Slides from the first snapshot in steps of horizon_days*24 hours.
    Partial final window is discarded.
    """
    expected = horizon_days * 24
    chunk_num = 1
    for start in range(0, len(snapshots), expected):
        chunk = snapshots[start : start + expected]
        if len(chunk) < expected:
            logger.debug(f"Dropping partial chunk {chunk_num} ({len(chunk)} snapshots)")
            continue
        yield f"chunk_{chunk_num:03d}", chunk
        chunk_num += 1


def extract_chunk_revenues(
    n: pypsa.Network,
    ty: str,
    cy: str,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """
    Slice a solved full-year network into horizon chunks and return per-chunk revenue.

    Parameters
    ----------
    n:
        A solved full-year day-ahead PyPSA network.
    ty:
        Target year string (e.g. "2030"), used to label output rows.
    cy:
        Climate year string (e.g. "1980"), used to label output rows.
    horizon_days:
        Chunk length in days. Must be >= 1. Default is 7 (one week).

    Returns
    -------
    pd.DataFrame
        Index: chunk_id strings of the form "{ty}_{cy}_chunk_{NNN}".
        Columns: BZ x carrier revenue features + chunk_start + chunk_end.
        Values: total energy revenue in EUR for that chunk.
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}.")

    snapshots = n.snapshots

    def _bus_and_carrier(n, c, port="", **kwargs):
        """Group statistics by bidding zone and carrier, yielding '{bus}_{carrier}' labels."""
        bus_col = f"bus{port}" if port else "bus"
        return n.df(c)[bus_col] + "_" + n.df(c)["carrier"]

    records = {}
    chunk_starts = {}
    chunk_ends = {}

    for chunk_label, chunk_snapshots in _iter_nday_chunks(snapshots, horizon_days):
        chunk_id = f"{ty}_{cy}_{chunk_label}"
        chunk_net = n.copy(snapshots=chunk_snapshots)
        revenue_series = chunk_net.statistics.revenue(
            comps=["Generator", "StorageUnit"],
            groupby=_bus_and_carrier,
        )
        records[chunk_id] = revenue_series
        chunk_starts[chunk_id] = chunk_snapshots[0]
        chunk_ends[chunk_id] = chunk_snapshots[-1]

    if not records:
        raise ValueError(
            f"No complete {horizon_days}-day chunks found in network for ty={ty}, cy={cy}. "
            f"Snapshot range: {snapshots[0]} to {snapshots[-1]}"
        )

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "chunk_id"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.insert(0, "chunk_start", pd.Series(chunk_starts))
    df.insert(1, "chunk_end", pd.Series(chunk_ends))

    logger.info(
        f"Extracted {len(df)} {horizon_days}-day chunks for ty={ty}, cy={cy} "
        f"({df.shape[1] - 2} BZ×carrier features)"
    )
    return df


if __name__ == "__main__":
    from snakemake.script import snakemake

    from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
        configure_logging,
    )

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)  # ty: ignore[possibly-missing-attribute]

    wildcards = n.meta.get("wildcards", {})
    ty = wildcards.get(
        "ty", snakemake.wildcards.ty
    )  # ty: ignore[possibly-missing-attribute]
    cy = wildcards.get(
        "cy", snakemake.wildcards.cy
    )  # ty: ignore[possibly-missing-attribute]
    horizon_days = snakemake.config["eva"]["advanced_clustering"]["horizon_days"]

    df = extract_chunk_revenues(n, ty=ty, cy=cy, horizon_days=horizon_days)

    df.to_csv(snakemake.output.chunk_revenues)  # ty: ignore[possibly-missing-attribute]
    logger.info(
        f"Wrote {snakemake.output.chunk_revenues}"
    )  # ty: ignore[possibly-missing-attribute]
