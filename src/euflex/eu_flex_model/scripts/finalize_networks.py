#!/usr/bin/env python3
"""
Finalize networks for the EVA process.

This script aggregates forced outage samples and copies the final iteration networks
from the resources directory to the results directory, removing the iteration,
forecast type, and forced outage wildcards.

For intraday networks with multiple forced outage samples, timeseries data is aggregated
by computing the mean across all samples.

Inputs:
- Intraday networks from the last iteration (multiple {fo} samples per ty/cy)
- Day-ahead networks from the last iteration (one per ty/cy)

Outputs:
- Final aggregated intraday networks in the results directory (one per ty/cy)
- Final day-ahead networks in the results directory (one per ty/cy)
"""

import logging
import pathlib
from collections import defaultdict

import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

COMPRESSION = {"zlib": True, "complevel": 4}


def aggregate_intraday_networks(
    network_paths: list[pathlib.Path], output_path: pathlib.Path
) -> None:
    """
    Aggregate multiple intraday networks from different forced outage samples.

    Parameters
    ----------
    network_paths : list of paths
        Paths to intraday network files for different forced outage samples
    output_path : path
        Path to save the aggregated network

    Returns
    -------
    None
    """
    if len(network_paths) == 1:
        # Only one forced outage sample, just copy it
        logger.info(f"Single forced outage sample, copying to {output_path}")
        network = pypsa.Network(network_paths[0])
        network.meta = {} if not hasattr(network, "meta") else network.meta

        # Get iteration from network metadata if available
        if "wildcards" in network.meta and "it" in network.meta["wildcards"]:
            network.meta["eva_final_iteration"] = int(network.meta["wildcards"]["it"])
        else:
            logger.warning(f"No iteration metadata found in {network_paths[0]}")

        network.meta["eva_network_type"] = "intraday"
        network.meta["forced_outage_samples"] = 1
        network.export_to_netcdf(output_path, compression=COMPRESSION)
        return

    logger.info(
        f"Aggregating {len(network_paths)} forced outage samples into {output_path}"
    )

    # Load all networks
    networks = [pypsa.Network(path) for path in network_paths]

    # Use the first network as the base
    aggregated = networks[0].copy()

    # Aggregate timeseries data by taking the mean across forced outage samples
    # This preserves the stochastic nature while providing a representative result

    # List of components with timeseries data
    timeseries_components = [
        "Generator",
        "Load",
        "StorageUnit",
        "Store",
        "Line",
        "Link",
        "Bus",
    ]

    for component in timeseries_components:
        # Check if the component exists in the network
        if component not in aggregated.components.keys():
            continue

        # Get the component's timeseries attributes
        comp_t = aggregated.dynamic(component)

        for attr in comp_t:
            # Get timeseries dataframes from all networks
            dfs = []
            for n in networks:
                df = n.dynamic(component)[attr]
                if not df.empty:
                    dfs.append(df)

            if not dfs:
                continue

            # Compute mean across all forced outage samples
            # Handle both Series and DataFrames
            if len(dfs) > 1:
                # Stack all dataframes and compute mean
                stacked = pd.concat(dfs, keys=range(len(dfs)))
                aggregated_df = stacked.groupby(level=1).mean()

                # Update the aggregated network's timeseries
                setattr(comp_t, attr, aggregated_df)

    # Update metadata
    if "wildcards" in aggregated.meta and "it" in aggregated.meta["wildcards"]:
        aggregated.meta["eva_final_iteration"] = int(aggregated.meta["wildcards"]["it"])
    else:
        logger.warning("No iteration metadata found in aggregated network")

    aggregated.meta["eva_network_type"] = "intraday"
    aggregated.meta["forced_outage_samples"] = len(network_paths)
    aggregated.meta["aggregation_method"] = "mean"

    # Save the aggregated network
    aggregated.export_to_netcdf(output_path, compression=COMPRESSION)
    logger.info(f"Saved aggregated network to {output_path}")


def group_networks_by_ty_cy(
    input_paths: list[pathlib.Path],
) -> dict[tuple[str, str], list[pathlib.Path]]:
    """
    Group network input paths by (ty, cy) using metadata from each network file.

    Parameters
    ----------
    input_paths : list of str
        Paths to network files.

    Returns
    -------
    dict[tuple[str, str], list[str]]
        Mapping from (ty, cy) to list of matching input paths.
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for input_path in input_paths:
        n = pypsa.Network(input_path)
        wildcards = n.meta["wildcards"]
        ty = wildcards.get("ty")
        cy = wildcards.get("cy")
        if ty and cy:
            groups[(ty, cy)].append(input_path)
            logger.debug(f"Grouped {input_path} under ty={ty}, cy={cy}")
        else:
            logger.warning(f"Missing ty or cy in metadata for {input_path}")
    return groups


def process_aggregated_networks(
    input_paths: list[pathlib.Path], output_paths: list[pathlib.Path], label: str = ""
) -> None:
    """
    Group inputs by (ty, cy), then aggregate and write each output network.

    Parameters
    ----------
    input_paths : list of pathlib.Path
        Input network paths (may include multiple forced outage samples per ty/cy).
    output_paths : list of pathlib.Path
        Output paths to write aggregated networks to.
    label : str, optional
        Label used in log messages to identify the network type.
    """
    groups = group_networks_by_ty_cy(input_paths)

    for output_path in output_paths:
        matched = False
        for (ty, cy), network_paths in groups.items():
            if network_paths and (
                f"_{ty}_{cy}" in str(output_path)
                or str(output_path).endswith(f"_{ty}_{cy}.nc")
            ):
                aggregate_intraday_networks(network_paths, output_path)
                logger.info(f"Processed {label}ty={ty}, cy={cy} -> {output_path}")
                matched = True
                break
        if not matched:
            logger.warning(f"No matching {label}networks found for {output_path}")


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="finalize_networks",
    )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    perfect_foresight = snakemake.config["eva"]["perfect_foresight"]["enable"]

    # Get no-ramping inputs/outputs if available (common to both modes)
    intraday_no_ramping_inputs = getattr(
        snakemake.input, "intraday_networks_without_ramping", []
    )
    intraday_no_ramping_outputs = getattr(
        snakemake.output, "intraday_networks_without_ramping", []
    )
    market_no_ramping_inputs = getattr(
        snakemake.input, "market_networks_without_ramping", []
    )
    market_no_ramping_outputs = getattr(
        snakemake.output, "market_networks_without_ramping", []
    )

    if perfect_foresight:
        # Perfect foresight mode: only market networks
        market_inputs = (
            snakemake.input.market_networks  # ty: ignore[possibly-missing-attribute]
        )
        market_outputs = (
            snakemake.output.market_networks  # ty: ignore[possibly-missing-attribute]
        )

        process_aggregated_networks(market_inputs, market_outputs, label="market ")
        logger.info("Perfect foresight mode: Finalized market networks")

        if market_no_ramping_inputs and market_no_ramping_outputs:
            process_aggregated_networks(
                market_no_ramping_inputs,
                market_no_ramping_outputs,
                label="no-ramping market ",
            )
            logger.info(
                "Perfect foresight mode: Finalized without-ramping market networks"
            )

    else:
        # Standard EVA mode: both intraday and day-ahead networks
        intraday_inputs = (
            snakemake.input.intraday_networks  # ty: ignore[possibly-missing-attribute]
        )
        day_ahead_inputs = (
            snakemake.input.day_ahead_networks  # ty: ignore[possibly-missing-attribute]
        )
        intraday_outputs = (
            snakemake.output.intraday_networks  # ty: ignore[possibly-missing-attribute]
        )
        day_ahead_outputs = (
            snakemake.output.day_ahead_networks  # ty: ignore[possibly-missing-attribute]
        )

        process_aggregated_networks(
            intraday_inputs, intraday_outputs, label="intraday "
        )

        # Process day-ahead networks (no aggregation needed)
        for input_path, output_path in zip(day_ahead_inputs, day_ahead_outputs):
            network = pypsa.Network(input_path)
            network.meta["eva_final_iteration"] = int(network.meta["wildcards"]["it"])
            network.meta["eva_network_type"] = "day_ahead"
            network.export_to_netcdf(output_path, compression=COMPRESSION)
            logger.info(f"Saved day-ahead network to {output_path}")

        if intraday_no_ramping_inputs and intraday_no_ramping_outputs:
            process_aggregated_networks(
                intraday_no_ramping_inputs,
                intraday_no_ramping_outputs,
                label="no-ramping intraday ",
            )
