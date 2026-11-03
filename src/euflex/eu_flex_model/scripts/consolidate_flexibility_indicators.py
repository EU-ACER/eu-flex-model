#!/usr/bin/env python3
"""
Consolidate flexibility indicators from multiple scenarios.

This script consolidates flexibility indicators calculated for different
scenarios (target years, climate years) into a single dataset.

Inputs:
    - flexibility_indicators: Multiple xarray datasets for each scenario

Outputs:
    - consolidated_flexibility_indicators: Single xarray dataset with all scenarios
"""

import logging

import numpy as np
import xarray as xr

from euflex.eu_flex_model.scripts.calculate_model_outputs import (
    SHORT_TERM_NEED_PERCENTILES,
)
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

# Raw signed hourly short-term need, pooled across all weather scenarios (CY).
# The direction-zeroed upward/downward series must not be used here.
SHORT_TERM_NEED_HOURLY_VAR = "short_term_need_hourly"

# Which agg keys are pooled per tail. The percentile values themselves live in
# SHORT_TERM_NEED_PERCENTILES (calculate_model_outputs.py) - single source
# of truth for the fixed FNA percentile pairs (99.9/99 and 0.1/1).
SHORT_TERM_NEED_POOLED_AGG_KEYS = {
    "upward": ["p99.9", "p99"],
    "downward": ["p0.1", "p1"],
}


def add_aggregated_short_term_need_percentiles(ds: xr.Dataset) -> xr.Dataset:
    """
    Add short-term need percentile indicators pooled across all weather
    scenarios (climate years).

    Pooling the raw signed hourly values across CY before taking the
    percentile is not equivalent to averaging each CY's own percentile, so
    this is computed here as a dedicated indicator rather than relying on the
    dashboard's generic per-CY mean. Uses nanpercentile since the merge's
    outer join NaN-pads the extra hour-of-year values that only leap-year
    CYs have.

    Parameters
    ----------
    ds : xr.Dataset
        Consolidated "base" dataset, potentially with a "CY" dimension and
        the signed short_term_need_hourly variable.

    Returns
    -------
    xr.Dataset
        Dataset with the aggregated percentile variables added, if the
        signed hourly short-term-need variable is present.
    """
    hourly_var = SHORT_TERM_NEED_HOURLY_VAR
    if "CY" not in ds.dims or hourly_var not in ds.data_vars:
        return ds

    pooled_dims = [d for d in ("CY", "hour") if d in ds[hourly_var].dims]
    for direction, agg_keys in SHORT_TERM_NEED_POOLED_AGG_KEYS.items():
        for agg_key in agg_keys:
            percentile = SHORT_TERM_NEED_PERCENTILES[agg_key]
            suffix = agg_key.replace(".", "_")
            agg_name = f"aggregated_short_term_need_{direction}_{suffix}"
            ds[agg_name] = ds[hourly_var].reduce(
                np.nanpercentile, dim=pooled_dims, q=percentile
            )
            ds[agg_name].attrs["units"] = ds[hourly_var].attrs["units"]
            ds[agg_name].attrs["kind"] = "flexibility_indicator"
            ds[agg_name].attrs["description"] = (
                f"{percentile}th percentile of the signed short term need, "
                "pooled across all weather scenarios"
            )

    return ds


# TODO: this is a hacky solution to treat same carriers with different nice_names and colors among networks
#  Normally, with the fix of add_placeholder_components, this can be removed once pre-builts are updated
#  (or once we make sure that same carriers have same names everywhere)
def align_carrier_metadata(datasets):
    """
    Align carrier metadata (_color and _nice_name) across multiple datasets.
    When different scenarios have conflicting metadata for the same carrier
    (e.g., different years built with different config), this function picks
    the first non-empty value encountered and applies it consistently across
    all datasets.

    Parameters
    ----------
    datasets : list of xr.Dataset
        List of datasets that may have conflicting carrier metadata

    Returns
    -------
    list of xr.Dataset
        Datasets with aligned carrier metadata
    """
    # Collect all carriers and their metadata from all datasets
    carrier_colors = {}
    carrier_nice_names = {}

    for ds in datasets:
        if "_color" in ds.data_vars and "carrier" in ds.dims:
            # Load data if it's dask array
            colors = (
                ds._color.compute() if hasattr(ds._color.data, "compute") else ds._color
            )
            nice_names = None
            if "_nice_name" in ds.data_vars:
                nice_names = (
                    ds._nice_name.compute()
                    if hasattr(ds._nice_name.data, "compute")
                    else ds._nice_name
                )

            for carrier in ds.carrier.values:
                # Store first non-empty value for each carrier
                if carrier not in carrier_colors:
                    # Select carrier, then get first value if there are other dimensions
                    color_data = colors.sel(carrier=carrier)
                    if color_data.ndim > 0:
                        # If there are other dimensions (e.g., mh, ct), get the first value
                        color_val = color_data.values.flat[0]
                    else:
                        color_val = color_data.values.item()

                    # Only store if not empty/None
                    if color_val and str(color_val).strip():
                        carrier_colors[carrier] = color_val

                if nice_names is not None and carrier not in carrier_nice_names:
                    # Select carrier, then get first value if there are other dimensions
                    name_data = nice_names.sel(carrier=carrier)
                    if name_data.ndim > 0:
                        # If there are other dimensions (e.g., mh, ct), get the first value
                        name_val = name_data.values.flat[0]
                    else:
                        name_val = name_data.values.item()

                    # Only store if not empty/None
                    if name_val and str(name_val).strip():
                        carrier_nice_names[carrier] = name_val

    logger.info(f"Aligned metadata for {len(carrier_colors)} carriers")

    # Apply aligned metadata to all datasets
    aligned_datasets = []
    for ds in datasets:
        if "_color" in ds.data_vars and "carrier" in ds.dims:
            # Update colors with aligned values
            for carrier in ds.carrier.values:
                if carrier in carrier_colors:
                    # Use .loc with just the carrier dimension, let other dims broadcast
                    ds._color.loc[dict(carrier=carrier)] = carrier_colors[carrier]

            # Update nice_names with aligned values
            if "_nice_name" in ds.data_vars:
                for carrier in ds.carrier.values:
                    if carrier in carrier_nice_names:
                        ds._nice_name.loc[dict(carrier=carrier)] = carrier_nice_names[
                            carrier
                        ]

        aligned_datasets.append(ds)

    return aligned_datasets


# Carriers whose capacity is demand-driven rather than an investment decision
# (e.g. DSR_implicit), but which still end up in commissioned_capacity/
# decommissioned_capacity because that calculation diffs raw network capacity
# snapshots instead of being sourced from unit_states.csv (which already
# excludes non-expandable/non-decommissionable units). Their capacity can
# genuinely differ across climate-year files sharing the same target year,
# which breaks merging.
# TODO: remove once calculate_model_outputs.py sources commissioned/
#  decommissioned_capacity from unit_states.csv instead of diffing capacity
#  snapshots - then these carriers will never appear in these variables.
NON_EXPANDABLE_CAPACITY_CARRIERS = ("DSR_implicit",)
NON_EXPANDABLE_CAPACITY_VARIABLES = ("commissioned_capacity", "decommissioned_capacity")


def zero_non_expandable_carrier_capacity(datasets: list) -> list:
    """
    Zero out NON_EXPANDABLE_CAPACITY_CARRIERS in NON_EXPANDABLE_CAPACITY_VARIABLES
    across all datasets, in place, so merging with compat="no_conflicts" doesn't
    raise on carriers whose capacity isn't really a commission/decommission event.

    Parameters
    ----------
    datasets : list
        List of x-arrays to be merged

    Returns
    -------
    list
        Corrected x-arrays

    """
    for ds in datasets:
        vars_to_zero = [
            var
            for var in NON_EXPANDABLE_CAPACITY_VARIABLES
            if var in ds.data_vars and "carrier" in ds[var].dims
        ]

        if not vars_to_zero or "carrier" not in ds.coords:
            continue

        carriers = ds.carrier.values
        carriers_to_zero = [
            carrier
            for carrier in carriers
            if carrier in NON_EXPANDABLE_CAPACITY_CARRIERS
        ]

        if carriers_to_zero:
            ds[vars_to_zero].loc[{"carrier": carriers_to_zero}] = 0.0

    return datasets


def consolidate_shadow_prices(input_files: list) -> xr.Dataset:
    """
    Consolidate shadow prices from multiple files into a single dataset.

    Shadow prices are stored as plain xr.Dataset (not DataTree like model_outputs).

    Parameters
    ----------
    input_files : list
        List of paths to shadow prices files

    Returns
    -------
    xr.Dataset
        Consolidated dataset containing all shadow prices
    """
    logger.info(f"Consolidating {len(input_files)} shadow prices files...")

    datasets = []
    for file in input_files:
        logger.debug(f"Loading {file}")
        with xr.open_dataset(
            file,
            chunks={"snapshot": 1000, "BZ": 10, "CY": 1, "TY": 1},
            decode_timedelta=True,
        ) as ds:
            # Cast float64 to float32 to save space
            for var in ds.data_vars:
                if ds[var].dtype == "float64":
                    logger.debug(f"Casting variable {var} to float32")
                    ds = ds.assign({var: ds[var].astype("float32")})

            # Load data into memory and close file handle before appending
            ds_loaded = ds.load()
            datasets.append(ds_loaded)

    # Merge all datasets
    if len(datasets) == 1:
        consolidated = datasets[0]
    else:
        consolidated = xr.merge(datasets, join="outer", compat="no_conflicts")

    logger.info(f"Successfully consolidated {len(datasets)} shadow prices datasets")
    logger.info(f"Consolidated dataset has variables: {list(consolidated.data_vars)}")

    return consolidated


def consolidate_flexibility_indicators(input_files: list) -> xr.DataTree:
    """
    Consolidate flexibility indicators from multiple files into a single dataset.

    Parameters
    ----------
    input_files : list
        List of paths to flexibility indicator files

    Returns
    -------
    xr.DataTree
        Consolidated dataset containing all flexibility indicators
    """
    logger.info(f"Consolidating {len(input_files)} flexibility indicator files...")

    # Open all datasets
    data = {"base": [], "dummy_storage_optimization": []}
    for file in input_files:
        logger.debug(f"Loading {file}")
        with xr.open_datatree(
            file,
            chunks={"snapshot": 1000, "BZ": 10, "CY": 1, "TY": 1},
            decode_timedelta=True,
        ) as dt:
            for key, store in data.items():
                for var in dt[key].data_vars:  # cast only float64 to float 32
                    if dt[key].data_vars[var].dtype == "float64":
                        logger.debug(f"Casting variable {var} to float32")
                        dt[key] = dt[key].assign(
                            {var: dt[key].data_vars[var].astype("float32")}
                        )

                # Load data into memory and close file handle before appending
                ds_loaded = dt[key].dataset.load()
                store.append(ds_loaded)

    # Align carrier metadata before merging to avoid conflicts
    logger.info("Aligning carrier metadata across base datasets...")
    data["base"] = align_carrier_metadata(data["base"])

    logger.info(
        "Aligning carrier metadata across dummy storage optimization datasets..."
    )
    data["dummy_storage_optimization"] = align_carrier_metadata(
        data["dummy_storage_optimization"]
    )

    # Zero out non-expandable carriers' commissioned/decommissioned_capacity before
    # merging - see NON_EXPANDABLE_CAPACITY_CARRIERS for why this is needed.
    data["base"] = zero_non_expandable_carrier_capacity(data["base"])

    # Merge datasets with aligned metadata
    data["base"] = xr.merge(data["base"], join="outer", compat="no_conflicts")
    data["dummy_storage_optimization"] = xr.merge(
        data["dummy_storage_optimization"], join="outer", compat="no_conflicts"
    )

    data["base"] = add_aggregated_short_term_need_percentiles(data["base"])

    converted_data = {}
    for key, value in data.items():
        if isinstance(value, list):
            # Convert first list item to Dataset if possible
            converted_data[key] = value[0] if value else None
        else:
            converted_data[key] = value

    logger.info(f"Successfully consolidated {len(data['base'])} datasets")
    return xr.DataTree.from_dict(converted_data)


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(), rule="consolidate_flexibility_indicators"
    )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Get list of input files from snakemake
    input_files = snakemake.input

    # Ensure input is a list
    if not isinstance(input_files, list):
        input_files = [input_files]

    # Detect if we're consolidating shadow prices or flexibility indicators
    # based on the rule name or output filename
    output_file = snakemake.output.fn  # ty: ignore[possibly-missing-attribute]
    is_shadow_prices = "shadow_prices" in str(output_file)

    if is_shadow_prices:
        # Shadow prices are plain Datasets
        consolidated = consolidate_shadow_prices(input_files)
    else:
        # Model outputs are DataTrees
        consolidated = consolidate_flexibility_indicators(input_files)

    # Save consolidated dataset to NetCDF file
    logger.info(f"Saving consolidated data to {output_file}")
    consolidated.to_netcdf(output_file)
