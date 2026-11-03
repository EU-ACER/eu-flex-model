import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from euflex.eu_flex_model.scripts._fasterpam import fasterpam

logger = logging.getLogger(__name__)


def _parse_cy_from_chunk_id(chunk_id: str) -> str:
    """Extract climate year from a chunk_id label like '2030_1980_week_03'."""
    return chunk_id.split("_")[1]


def cluster_weeks(
    revenue_csv_paths: list[str],
    n_clusters: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Run a single joint K-medoids across all CYs' chunk revenues.

    Parameters
    ----------
    revenue_csv_paths:
        Paths to chunk_revenues CSVs produced by extract_chunk_revenues,
        one per CY. Each CSV has rows labelled '{ty}_{cy}_{chunk_label}',
        columns chunk_start, chunk_end, and one column per carrier.
    n_clusters:
        Total number of K-medoids clusters (k). All k centroids are chosen
        from the joint pool of all CYs -- multiple centroids may land in one CY.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: chunk_id, cy, chunk_start, chunk_end, cluster_id,
        centroid_flag, weight.
        One row per chunk across all CYs.
        Weights sum to 1.0 globally (weight = cluster_size / total_chunks).
        Cluster IDs in range [0, k).

    Notes
    -----
    Clustering runs on Day-Ahead (forecast) revenues only. The resulting week
    mapping is applied identically to both forecast and actual base networks.
    This is mandatory: the Intraday (actual) solve is sequentially coupled to
    the Day-Ahead solve of the same week -- it inherits unit commitments and
    SOC state. Clustering DA and ID separately would assign different calendar
    weeks to the same cluster index, making the DA->ID handoff impossible.
    """
    frames = []
    for path in sorted(revenue_csv_paths):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "chunk_id"
        frames.append(df)

    combined = pd.concat(frames, axis=0).fillna(0.0)
    n_chunks = len(combined)

    if n_clusters > n_chunks:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds number of chunks={n_chunks}. "
            f"Reduce n_clusters or provide more weekly revenue data."
        )

    chunk_ids = combined.index.tolist()
    chunk_starts = combined["chunk_start"].values
    chunk_ends = combined["chunk_end"].values
    revenue_matrix = combined.drop(columns=["chunk_start", "chunk_end"])

    scaler = StandardScaler()
    X = scaler.fit_transform(revenue_matrix.values)

    D = pairwise_distances(X, metric="euclidean")
    result = fasterpam(D, n_clusters=n_clusters, random_state=random_seed)
    labels = np.asarray(result.labels)

    cluster_sizes = np.bincount(labels, minlength=n_clusters)

    centroid_flags = np.zeros(n_chunks, dtype=bool)
    centroid_flags[result.medoids] = True

    # Weight = cluster_size / total_chunks so weights sum to 1.0 globally.
    weights = cluster_sizes[labels] / n_chunks

    df_out = pd.DataFrame(
        {
            "chunk_id": chunk_ids,
            "cy": [_parse_cy_from_chunk_id(c) for c in chunk_ids],
            "chunk_start": chunk_starts,
            "chunk_end": chunk_ends,
            "cluster_id": labels,
            "centroid_flag": centroid_flags,
            "weight": weights,
        }
    )

    centroid_cys = df_out.loc[df_out["centroid_flag"], "cy"].tolist()
    logger.info(
        f"Joint K-medoids: {n_chunks} chunks -> {n_clusters} clusters. "
        f"Centroid CYs: {centroid_cys}"
    )

    return df_out


def write_clustering_summary(
    output_path: Path,
    ty: str,
    it: int,
    mode: str,
    df_clusters: pd.DataFrame,
) -> None:
    """Write a one-row summary CSV alongside the week_clusters file."""
    centroids = df_clusters[df_clusters["centroid_flag"]]
    centroid_chunk_ids = ";".join(centroids["chunk_id"].tolist())
    centroid_cys = ";".join(sorted(centroids["cy"].astype(str).unique().tolist()))

    summary = pd.DataFrame(
        [
            {
                "ty": ty,
                "it": it,
                "mode": mode,
                "n_centroids": int(centroids["centroid_flag"].sum()),
                "n_total_chunks": len(df_clusters),
                "centroid_chunk_ids": centroid_chunk_ids,
                "centroid_cys": centroid_cys,
            }
        ]
    )
    summary_path = output_path.parent / f"clustering_summary_{ty}_{it}.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Wrote clustering summary to {summary_path}")


def _is_full_run(
    current_it: int,
    freq: int,
    max_it: int = 0,
    prev_was_clustered_converged: bool = False,
) -> bool:
    """Return True if current_it should run K-medoids rather than freeze clusters."""
    return (
        (current_it == 0)
        or (freq > 0 and current_it % freq == 0)
        or (max_it > 0 and current_it >= max_it - 1)
        or prev_was_clustered_converged
    )


def _find_source_full_run_it(current_it: int, full_run_frequency: int) -> int:
    """
    Return the most recent full-run iteration index before current_it.

    Full-run iterations are: 0, freq, 2*freq, ...
    For any clustered it, the nearest preceding full run is:
        floor((it - 1) / freq) * freq
    """
    if full_run_frequency <= 0:
        return 0
    return ((current_it - 1) // full_run_frequency) * full_run_frequency


if __name__ == "__main__":
    import shutil

    from snakemake.script import snakemake

    from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
        configure_logging,
    )

    configure_logging(snakemake)

    current_it = int(snakemake.wildcards.it)  # ty: ignore[possibly-missing-attribute]
    output_path = Path(
        snakemake.output.week_clusters
    )  # ty: ignore[possibly-missing-attribute]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    freq = snakemake.config["eva"]["advanced_clustering"]["full_run_frequency"]
    max_it = snakemake.config["eva"]["max_it"]
    ty = snakemake.wildcards.ty  # ty: ignore[possibly-missing-attribute]

    prev_was_clustered_converged = False
    if current_it > 0:
        run_name = snakemake.config["run"].get("name", "")
        # Check both relative path formats to ensure compatibility with all directory layouts
        paths_to_check = [
            Path(f"resources/convergence/convergence_{current_it - 1}.txt"),
            Path(f"resources/{run_name}/convergence/convergence_{current_it - 1}.txt"),
        ]
        for p in paths_to_check:
            if p.exists() and p.read_text().strip() == "ClusteredConverged":
                prev_was_clustered_converged = True
                break

    if not _is_full_run(current_it, freq, max_it, prev_was_clustered_converged):
        source_it = _find_source_full_run_it(current_it, freq)
        source_path = output_path.parent / f"week_clusters_{ty}_{source_it}.csv"

        if not source_path.exists():
            raise FileNotFoundError(
                f"Cluster freeze: source clusters file not found: {source_path}. "
                f"Ensure iteration {source_it} (full run) completed before it={current_it}."
            )

        shutil.copy(str(source_path), str(output_path))
        logger.info(
            f"Cluster freeze (it={current_it}): copied clusters from it={source_it} "
            f"({source_path} -> {output_path})"
        )
        df_frozen = pd.read_csv(str(output_path))
        write_clustering_summary(
            output_path, ty=ty, it=current_it, mode="freeze", df_clusters=df_frozen
        )
    else:
        paths = list(
            snakemake.input.chunk_revenues
        )  # ty: ignore[possibly-missing-attribute]
        k = snakemake.config["eva"]["advanced_clustering"]["n_clusters"]
        seed = snakemake.config["eva"]["advanced_clustering"]["random_seed"]

        df = cluster_weeks(revenue_csv_paths=paths, n_clusters=k, random_seed=seed)
        df.to_csv(output_path, index=False)
        logger.info(f"Wrote {output_path}")
        write_clustering_summary(
            output_path, ty=ty, it=current_it, mode="full", df_clusters=df
        )
