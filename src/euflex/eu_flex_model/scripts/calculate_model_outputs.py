#!/usr/bin/env python3
"""
Calculate comprehensive model outputs including flexibility indicators and market outputs.

This unified script combines the functionality of calculate_flexibility_indicators.py
and extract_market_outputs.py to create a single comprehensive dataset with all
model outputs. Each output is tagged with a "kind" attribute to distinguish
between "flexibility_indicator" and "market_output".

Inputs:
    - intraday_network: PyPSA network resulting from intraday dispatch
    - day_ahead_network: PyPSA network resulting from day-ahead dispatch
    - pre_optimization_storage: Pre-optimization storage plan (optional)

Outputs:
    - model_outputs: xarray Dataset with all calculated indicators and market outputs
"""

import functools
import hashlib
import logging
import pathlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from euflex.eu_flex_model.scripts.helpers.date_helpers import (
    load_co2_targets,
    load_necp_targets,
)
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.pypsa_helpers import (
    as_dense,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

PRICE_UNITS = "€/MWh"
INVALID_AGGREGATION_MESSAGE = "Invalid aggregation operation"
# Solver output carries floating-point noise (~1e-15 MW) even where load
# shedding is truly zero; a strict `> 0` check flags nearly every snapshot as
# a loss-of-load hour. This is orders of magnitude below any real shedding
# event, which is at least in the kW-MW range.
LOAD_SHEDDING_TOLERANCE_MW = 1e-3

# EVA outputs representing per-TY investment decisions (capacity additions/
# removals/mothballing, NPV) that don't vary with climate year - see
# calculate_capacity_changes, calculate_mothballed_capacity and
# load_npv_data_from_csv. Kept off the CY axis when assembling the final
# dataset instead of being broadcast across every CY with identical values.
TY_ONLY_EVA_OUTPUT_VARS = {
    "commissioned_capacity",
    "decommissioned_capacity",
    "mothballed_capacity",
    "npv_per_mw",
}

# Mapping from NECP target categories (variable2) to model carriers.
# Covers both Generator and StorageUnit components that produce electricity.
# NOTE: Pumping/charging carriers (e.g., "...Pumping") are excluded as they
# represent consumption, not generation.
NECP_TO_MODEL_CARRIERS = {
    "Wind": ["Wind Onshore", "Wind Offshore"],
    "Solar": ["Solar (PV)", "Solar (Thermal)"],
    "Hydro": [
        # Generator carriers
        "Hydro - Run of River (Turbine)",
        # StorageUnit carriers (produce generation when discharging)
        "Hydro - Reservoir",
        "Hydro - Pondage",
        # Pump storage turbine mode (generation)
        "Hydro - Pump Storage Open Loop (Turbine)",
        "Hydro - Pump Storage Closed Loop (Turbine)",
    ],
    "Bio": ["Biofuel", "Small biomass"],
    "Other Renewable": ["Geothermal", "Marine"],
}


# =============================================================================
# Shadow Price Extraction Functions
# =============================================================================

# Component type to _t attribute mapping for shadow price extraction
COMPONENT_T_MAPPING = {
    "Bus": "buses_t",
    "Generator": "generators_t",
    "StorageUnit": "storage_units_t",
    "Link": "links_t",
    "Line": "lines_t",
    "Store": "stores_t",
}


def _infer_shadow_price_units(constraint_name: str) -> str:
    """
    Infer units based on constraint type.

    Parameters
    ----------
    constraint_name : str
        Name of the constraint

    Returns
    -------
    str
        Inferred units string
    """
    name_lower = constraint_name.lower()
    if "nodal_balance" in name_lower or "power_balance" in name_lower:
        return "EUR/MWh"
    elif "reserve" in name_lower:
        return "EUR/MW"
    elif "co2" in name_lower or "emission" in name_lower:
        return "EUR/tCO2"
    elif (
        "energy" in name_lower or "soc" in name_lower or "state_of_charge" in name_lower
    ):
        return "EUR/MWh"
    elif "ramp" in name_lower or "fix_p" in name_lower:
        return "EUR/MW"
    return "N/A"


def _infer_shadow_price_description(constraint_name: str) -> str:
    """
    Generate human-readable description from constraint name.

    Parameters
    ----------
    constraint_name : str
        Name of the constraint

    Returns
    -------
    str
        Human-readable description
    """
    parts = re.split(r"[-_]", constraint_name)
    return " ".join(p.capitalize() for p in parts if p)


def _get_component_dimension_name(component_type: str) -> str:
    """
    Get the appropriate dimension name for a component type.

    Parameters
    ----------
    component_type : str
        The component type (Bus, Generator, StorageUnit, Link, etc.)

    Returns
    -------
    str
        Dimension name for the xarray DataArray
    """
    # Bus components use BZ (bidding zone) dimension
    if component_type == "Bus":
        return "BZ"
    # Other components use their own identifier dimension
    return component_type.lower()


def _extract_shadow_price_for_pattern(
    component_t,
    component_t_keys: list[str],
    component_type: str,
    constraint_suffix: str,
    full_pattern: str,
) -> xr.DataArray | None:
    """
    Extract shadow price for a specific constraint pattern from a component's _t attribute.

    Parameters
    ----------
    component_t
        The component's time-varying attribute (e.g., network.buses_t)
    component_t_keys : list[str]
        Available keys in the component_t
    component_type : str
        Component type (Bus, Generator, StorageUnit, Link, etc.)
    constraint_suffix : str
        The constraint name suffix (e.g., "nodal_balance", "fix_p_ramp_limit_up")
    full_pattern : str
        Full pattern string for logging

    Returns
    -------
    xr.DataArray | None
        DataArray with shadow prices, or None if not found
    """
    # Special case: Bus-nodal_balance maps to marginal_price
    if component_type == "Bus" and constraint_suffix == "nodal_balance":
        if "marginal_price" in component_t_keys:
            mp = component_t.marginal_price
            if not mp.empty:
                da = xr.DataArray(
                    mp.values,
                    dims=["snapshot", "BZ"],
                    coords={
                        "snapshot": mp.index,
                        "BZ": mp.columns.tolist(),
                    },
                )
                da.attrs["constraint_name"] = full_pattern
                da.attrs["kind"] = "shadow_price"
                return da
        else:
            logger.warning(f"No marginal_price found for pattern: {full_pattern}")
        return None

    # Special case: Bus-Reserve_requirements_* patterns
    if component_type == "Bus" and constraint_suffix.startswith(
        "Reserve_requirements_"
    ):
        mu_key = f"mu_{constraint_suffix}"
        if mu_key in component_t_keys:
            df = component_t[mu_key]
            if not df.empty:
                da = xr.DataArray(
                    df.values,
                    dims=["snapshot", "BZ"],
                    coords={
                        "snapshot": df.index,
                        "BZ": df.columns.tolist(),
                    },
                )
                da.attrs["constraint_name"] = full_pattern
                da.attrs["kind"] = "shadow_price"
                return da
        logger.warning(f"No dual variable found for pattern: {full_pattern}")
        return None

    # General case: look for mu_* key in component_t
    mu_key = f"mu_{constraint_suffix}"

    if mu_key in component_t_keys:
        df = component_t[mu_key]
        if not df.empty:
            # Determine dimension name based on component type
            component_dim = _get_component_dimension_name(component_type)

            da = xr.DataArray(
                df.values,
                dims=["snapshot", component_dim],
                coords={
                    "snapshot": df.index,
                    component_dim: df.columns.tolist(),
                },
            )
            da.attrs["constraint_name"] = full_pattern
            da.attrs["kind"] = "shadow_price"
            return da
    else:
        # Try pattern matching for wildcards
        regex_pattern = mu_key.replace("*", ".*")
        matching_keys = [k for k in component_t_keys if re.match(regex_pattern, k)]

        if matching_keys:
            # Return first match
            key = matching_keys[0]
            df = component_t[key]
            if not df.empty:
                component_dim = _get_component_dimension_name(component_type)
                clean_name = key.replace("mu_", "")

                da = xr.DataArray(
                    df.values,
                    dims=["snapshot", component_dim],
                    coords={
                        "snapshot": df.index,
                        component_dim: df.columns.tolist(),
                    },
                )
                da.attrs["constraint_name"] = f"{component_type}-{clean_name}"
                da.attrs["kind"] = "shadow_price"
                return da
        else:
            logger.warning(f"No dual variable found for pattern: {full_pattern}")

    return None


def extract_shadow_prices(
    network: pypsa.Network,
    constraint_patterns: list[str],
) -> xr.Dataset:
    """
    Extract shadow prices (dual variables) from a solved PyPSA network.

    Shadow prices represent the marginal cost of relaxing a constraint by one unit.
    They are extracted from the dual variables stored by PyPSA when assign_all_duals=True.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network with dual variables
    constraint_patterns : list[str]
        Constraint patterns to extract in format "ComponentType-constraint_name":
        - "Bus-nodal_balance" -> marginal electricity prices
        - "Bus-Reserve_requirements_FCR_up" -> FCR upward reserve prices
        - "Generator-fix_p_ramp_limit_up" -> Generator ramp-up limit duals
        - "StorageUnit-soc_min_constraint" -> Storage SOC minimum duals
        - "Link-fix_p_lower" -> Link lower power limit duals

    Returns
    -------
    xr.Dataset
        Dataset with shadow prices for all specified constraints
    """
    shadow_prices: dict[str, xr.DataArray] = {}

    for pattern in constraint_patterns:
        # Parse pattern into component type and constraint name
        if "-" in pattern:
            component_type, constraint_suffix = pattern.split("-", 1)
        else:
            # Assume Bus for backward compatibility
            component_type = "Bus"
            constraint_suffix = pattern

        # Get the appropriate _t attribute for this component type
        t_attr_name = COMPONENT_T_MAPPING.get(component_type)
        if t_attr_name is None:
            logger.warning(
                f"Unknown component type '{component_type}' in pattern: {pattern}"
            )
            continue

        component_t = getattr(network, t_attr_name)
        component_t_keys = list(component_t.keys())
        logger.debug(f"Available {t_attr_name} keys: {component_t_keys}")

        # Extract the shadow price based on component type and constraint
        da = _extract_shadow_price_for_pattern(
            component_t, component_t_keys, component_type, constraint_suffix, pattern
        )

        if da is not None:
            # Use mu_ prefix as per PyPSA convention. Include component type to
            # avoid collisions (e.g., multiple components have "upper"/"lower").
            if component_type == "Bus":
                var_name = f"mu_{constraint_suffix}"
            else:
                var_name = f"{component_type}_mu_{constraint_suffix}"
            shadow_prices[var_name] = da
            logger.info(f"Extracted {var_name} shadow prices")

    # Build dataset from extracted shadow prices
    if not shadow_prices:
        logger.warning("No shadow prices extracted from network")
        return xr.Dataset()

    ds = xr.Dataset(data_vars=shadow_prices)

    # Note: TY/CY coordinates are added by the caller when combining datasets
    # from multiple networks (day-ahead and intraday)

    logger.info(f"Created shadow prices dataset with {len(ds.data_vars)} variables")
    return ds


def _normalise_label(text: str) -> str:
    """
    Convert a text string to a human-friendly label format.

    Transforms underscores and hyphens to spaces, capitalizes words appropriately,
    and preserves acronyms (all-uppercase tokens) and digits. Useful for converting
    variable names to readable plot labels and titles.

    Parameters
    ----------
    text : str
        The input text to normalize. Can contain underscores, hyphens, or spaces
        as word separators.

    Returns
    -------
    str
        A normalized label string with:
        - Underscores and hyphens replaced by spaces
        - Words capitalized (except all-uppercase acronyms)
        - All-uppercase tokens preserved (e.g., "CO2", "LOL")
        - Tokens containing digits preserved as-is
        - Multiple consecutive separators collapsed to single spaces
    """
    cleaned = text.replace("_", " ").replace("-", " ")
    tokens = cleaned.split()
    normalised: list[str] = []
    for token in tokens:
        if token.isupper() or any(char.isdigit() for char in token):
            normalised.append(token.upper())
        else:
            normalised.append(token.capitalize())
    return " ".join(normalised)


def _attach_plot_labels(ds: xr.Dataset) -> xr.Dataset:
    """Ensure each data variable exposes a "label" attribute for plotting."""

    for name, data_array in ds.data_vars.items():
        description = data_array.attrs.get("description")
        label = description or _normalise_label(str(name))
        data_array.attrs["label"] = label
        data_array.attrs.setdefault("units", "")
    return ds


def _shift_time_columns_to_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Shift DatetimeIndex-like columns to a specified year while preserving month and day.

    Creates a copy of the input DataFrame with all datetime-like column labels
    shifted to the target year. This is useful for normalizing time series data
    from different years to a common reference year for comparison or analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex-like column labels that support the
        `replace()` method (e.g., pandas Timestamp objects).
    year : int
        The target year to shift all column datetime values to.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with datetime columns shifted to the
        target year. Month and day values are preserved. The data values
        remain unchanged.
    """
    df = df.copy()
    df.columns = df.columns.map(lambda ts: ts.replace(year=int(year)))
    return df


def _shift_time_rows_to_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Return a copy with DatetimeIndex-like index shifted to the given year.
    """
    df = df.copy()
    df.index = df.index.map(lambda ts: ts.replace(year=int(year)))
    return df


def _assign_coordinates_and_attrs(
    da: xr.DataArray, name: str, units: str = "MWh"
) -> xr.DataArray:
    """
    Assign standard coordinates and attributes to a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to assign coordinates and attributes to
    name : str
        Name for the description attribute
    units : str, optional
        Units for the units attribute, by default "MWh"

    Returns
    -------
    xr.DataArray
        DataArray with coordinates and attributes assigned
    """
    # Rename "bus" or "Bus" dimension to "BZ" if it exists
    if "bus" in da.dims:
        da = da.rename({"bus": "BZ"})
    if "Bus" in da.dims:
        da = da.rename({"Bus": "BZ"})

    if "snapshot" in da.dims:
        hour_of_year = (
            (da.snapshot.dt.dayofyear - 1) * 24 + da.snapshot.dt.hour + 1
        ).values
        da = da.assign_coords(hour=("snapshot", hour_of_year)).swap_dims(
            {"snapshot": "hour"}
        )
    da.attrs["units"] = units
    da.attrs["description"] = name
    return da


# =============================================================================
# UNIFIED SUPPLY/GENERATION FUNCTIONS
# =============================================================================


def calculate_supply(
    n: pypsa.Network, aggregate_time: bool = False, kind: str = "flexibility_indicator"
) -> xr.DataArray:
    """
    Calculate total supply per bidding zone and timestep.

    This unified function serves both flexibility indicators and market outputs,
    replacing the separate extract_total_generation_by_technology function.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    aggregate_time : bool, optional
        Whether to aggregate over time, by default False
    kind : str, optional
        Kind of output: "flexibility_indicator" or "market_output", by default "flexibility_indicator"

    Returns
    -------
    xr.DataArray
        Total supply with dimensions varying based on aggregate_time
    """
    supply = n.statistics.supply(
        comps=["Generator", "StorageUnit"],
        groupby=["carrier", "bus"],
        nice_names=False,
        aggregate_time=aggregate_time,
    ).droplevel("component")

    da = xr.DataArray(supply).unstack("dim_0")
    da = _assign_coordinates_and_attrs(da, "Total supply")
    da.attrs["kind"] = kind
    return da


def calculate_withdrawal(
    n: pypsa.Network, aggregate_time: bool = False, kind: str = "flexibility_indicator"
) -> xr.DataArray:
    """
    Calculate total withdrawal per bidding zone and timestep.

    This unified function mirrors calculate_supply but for energy withdrawal,
    serving both flexibility indicators and market outputs.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    aggregate_time : bool, optional
        Whether to aggregate over time, by default False
    kind : str, optional
        Kind of output: "flexibility_indicator" or "market_output", by default "flexibility_indicator"

    Returns
    -------
    xr.DataArray
        Total withdrawal with dimensions varying based on aggregate_time
    """
    withdrawal = n.statistics.withdrawal(
        comps=["Generator", "StorageUnit", "Load"],
        groupby=["carrier", "bus"],
        nice_names=False,
        aggregate_time=aggregate_time,
    ).droplevel("component")

    da = xr.DataArray(withdrawal).unstack("dim_0")
    da = _assign_coordinates_and_attrs(da, "Total withdrawal")
    da.attrs["kind"] = kind
    return da


def calculate_daily_supply(
    n: pypsa.Network, market_type: str = "intraday"
) -> xr.DataArray:
    """
    Calculate daily supply aggregated by technology and bidding zone.

    This provides the market output equivalent using the existing supply calculation.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    market_type : str, optional
        Market type for metadata, by default "intraday"

    Returns
    -------
    xr.DataArray
        Daily supply with dimensions (day, technology, bidding_zone)
    """
    # Get hourly supply data
    supply_hourly = calculate_supply(n)

    supply_daily = (
        supply_hourly.groupby("snapshot.dayofyear").sum("hour").rename(dayofyear="day")
    )

    supply_daily.attrs["description"] = (
        f"Total generation by technology per day ({market_type})"
    )
    supply_daily.attrs["data_type"] = "Raw"
    supply_daily.attrs["time_series"] = True
    supply_daily.attrs["market_model"] = market_type
    supply_daily.attrs["workflow_step"] = market_type.replace("_", "-")
    supply_daily.attrs["kind"] = "market_output"

    logger.info(
        f"Extracted {market_type} daily generation: {supply_daily.shape} (days x technologies x bidding_zones)"
    )
    return supply_daily


# =============================================================================
# FLEXIBILITY INDICATOR FUNCTIONS
# =============================================================================


def calculate_production_capacities(n: pypsa.Network) -> xr.DataArray:
    """
    Calculate installed capacities per bidding zone.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object

    Returns
    -------
    xr.DataArray
        Installed capacities with dimensions (TY, BZ)
    """
    # Group generators by bidding zone and technology
    installed_cap = n.statistics.installed_capacity(
        groupby=["carrier", "bus"],
        nice_names=False,
    ).droplevel("component")
    if "load shedding" in installed_cap.index:
        installed_cap = installed_cap.drop(
            "load shedding", level="carrier", errors="ignore"
        )
        installed_cap = installed_cap.drop(
            "load spillage", level="carrier", errors="ignore"
        )
    da = xr.DataArray(installed_cap).unstack("dim_0")
    da = _assign_coordinates_and_attrs(da, "Installed generation capacities", "MW")
    da.attrs["kind"] = "eva_output"
    return da


def calculate_storage_capacities(n: pypsa.Network) -> xr.DataArray:
    """
    Calculate installed capacities per bidding zone.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object

    Returns
    -------
    xr.DataArray
        Installed capacities with dimensions (TY, BZ)
    """
    # Group generators by bidding zone and technology
    installed_cap = n.statistics.installed_capacity(
        groupby=["carrier", "bus"],
        nice_names=False,
        storage=True,
    ).droplevel("component")
    if installed_cap.empty:
        # With no storage units/stores, statistics returns a flat, empty Series that
        # has no "dim_0" MultiIndex to unstack. Return an empty DataArray with the
        # expected (carrier, bus) dimensions so downstream consumers still get a
        # well-formed variable instead of crashing.
        da = xr.DataArray(
            np.empty((0, 0)),
            dims=("carrier", "bus"),
            coords={
                "carrier": pd.Index([], name="carrier"),
                "bus": pd.Index([], name="bus"),
            },
        )
    else:
        da = xr.DataArray(installed_cap).unstack("dim_0")
    da = _assign_coordinates_and_attrs(da, "Installed storage capacities")
    da.attrs["kind"] = "eva_output"
    return da


def calculate_energy_not_served(
    n: pypsa.Network, agg: Literal["mean", "sum", "max"] | None = None
) -> xr.DataArray:
    """
    Calculate load shedding per bidding zone and timestep.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    agg : str | None, optional
        Aggregation operation, by default None

    Returns
    -------
    xr.DataArray
        Energy not served. If agg is None, returns timeseries with dimensions (TY, CY, time, BZ).
        If agg is specified, returns aggregated values with dimensions (TY, CY, BZ).
    """
    # Get loss of load from network as timeseries
    if "load shedding" in n.carriers.index:
        supply = n.statistics.supply(
            comps="Generator",
            groupby=["carrier", "bus"],
            aggregate_time=False,
            nice_names=False,
        )
        lol = supply.loc["load shedding"]
        # Group by bus and sum across all generators with load shedding in each bus
        lol = lol.groupby(level="bus").sum()
    else:
        # Create empty DataFrame with correct structure
        lol = pd.DataFrame(0, columns=n.snapshots, index=n.buses.index).rename_axis(
            "bus", axis=0
        )

    # Shift timestamps from climate year to target year
    target_year = n.meta["wildcards"]["ty"]
    lol = _shift_time_columns_to_year(lol, int(target_year))

    da = xr.DataArray(lol)
    da = _assign_coordinates_and_attrs(da, "Energy not served")
    da.attrs["kind"] = "adequacy_indicator"

    if agg == "sum":
        da = da.sum("hour")
        da = _assign_coordinates_and_attrs(da, "Total energy not served")
        da.attrs["kind"] = "adequacy_indicator"
    elif agg == "max":
        da = da.max("hour")
        da = _assign_coordinates_and_attrs(da, "Maximum energy not served")
        da.attrs["kind"] = "adequacy_indicator"
    elif agg == "mean":
        da = da.mean("hour")
        da = _assign_coordinates_and_attrs(da, "Average energy not served")
        da.attrs["kind"] = "adequacy_indicator"
    elif agg is not None:
        raise ValueError(INVALID_AGGREGATION_MESSAGE)

    return da


def calculate_loss_of_load(n: pypsa.Network) -> xr.DataArray:
    """
    Calculate loss of load (hours with load shedding) per bidding zone.

    This function counts the number of hours in which energy not served is
    strictly positive, as defined in the modelling specification (equation 14.3):
    LOL_{n,sy} = Sum_t [ Sum_{s in S^LoL_n} g_{n,s,t,sy} > 0 ]

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object

    Returns
    -------
    xr.DataArray
        Loss of load count (hours) with dimensions (TY, CY, BZ).
    """
    # Get load shedding from network as timeseries
    if "load shedding" in n.carriers.index:
        supply = n.statistics.supply(
            comps="Generator",
            groupby=["carrier", "bus"],
            aggregate_time=False,
            nice_names=False,
        )
        lol = supply.loc["load shedding"]
        # Group by bus and sum across all generators with load shedding in each bus
        lol = lol.groupby(level="bus").sum()
    else:
        # Create empty DataFrame with correct structure
        lol = pd.DataFrame(0, columns=n.snapshots, index=n.buses.index).rename_axis(
            "bus", axis=0
        )

    # Count hours where load shedding > 0 (equation 14.3). A strict `> 0`
    # comparison would also catch floating-point solver noise (~1e-15 MW)
    # present in almost every snapshot, so compare against a tolerance well
    # below any real shedding event instead.
    lol_count = (lol > LOAD_SHEDDING_TOLERANCE_MW).sum(axis=1)

    da = xr.DataArray(lol_count)
    # Rename bus dimension to BZ
    if "bus" in da.dims:
        da = da.rename({"bus": "BZ"})

    # NOT "h": xarray's CF coder auto-encodes any variable whose `units`
    # matches a recognized time-duration string as a timedelta64 on to_netcdf,
    # even for a plain integer count -- this round-trips through
    # decode_cf_timedelta on read and can overflow for large counts.
    da.attrs["units"] = "h (count)"
    da.attrs["description"] = "Loss of load"
    da.attrs["kind"] = "adequacy_indicator"

    return da


def calculate_curtailment(
    n: pypsa.Network, renewable_carriers: list[str] | None = None
) -> xr.DataArray:
    """
    Calculate curtailment per bidding zone and timestep.

    Curtailment is defined as the difference between the potential renewable
    generation (p_max_pu * p_nom) and the actual dispatch (p). When
    ``p_max_pu_original`` is stored on the network (e.g. after reserve-blocking
    has reduced ``p_max_pu``), it is used as the availability baseline so that
    curtailment reflects true renewable potential rather than the
    reserve-reduced capacity.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    renewable_carriers: list, optional
        List of renewable carriers

    Returns
    -------
    xr.DataArray
        Curtailment with dimensions (TY, CY, time, BZ)
    """
    # Extract target year from network wildcards
    target_year = n.meta["wildcards"]["ty"]

    if renewable_carriers is None:
        renewable_carriers = RENEWABLE_CARRIERS

    snapshots = n.snapshots

    # Filter to renewable generators that exist in the network
    renewable_gens = n.generators[n.generators.carrier.isin(renewable_carriers)].index

    if len(renewable_gens) > 0:
        # Use p_max_pu_original if available (before reserve-blocking modifications)
        # to ensure curtailment reflects true renewable potential.
        if "p_max_pu_original" in n.generators_t:
            p_max_pu = n.generators_t.p_max_pu_original.reindex(
                index=snapshots, columns=renewable_gens, fill_value=0.0
            )
            logger.debug("Using p_max_pu_original for curtailment calculation")
        else:
            p_max_pu = as_dense(n, "Generator", "p_max_pu", renewable_gens)

        # Potential generation = p_max_pu * p_nom
        potential = p_max_pu.multiply(
            n.generators.loc[renewable_gens, "p_nom"], axis=1
        ).fillna(0.0)

        # Actual dispatch
        actual = n.generators_t.p.reindex(
            index=snapshots, columns=renewable_gens, fill_value=0.0
        )

        # Curtailment = potential - actual (clipped to zero to avoid negatives)
        curtailment_gen = (potential - actual).clip(lower=0.0)

        # Aggregate by (carrier, bus)
        gen_meta = n.generators.loc[renewable_gens, ["carrier", "bus"]]
        curtailment_gen.columns = pd.MultiIndex.from_frame(gen_meta)
        curtailment_by_carrier_bus = curtailment_gen.T.groupby(
            level=["carrier", "bus"]
        ).sum()

        # Filter to renewable carriers that actually have data
        existing_renewable_carriers = [
            c
            for c in renewable_carriers
            if c in curtailment_by_carrier_bus.index.get_level_values("carrier")
        ]
        if existing_renewable_carriers:
            curtailment = curtailment_by_carrier_bus.loc[existing_renewable_carriers]
        else:
            curtailment = curtailment_by_carrier_bus.iloc[0:0]
    else:
        # No renewable generators — return empty frame with correct structure
        curtailment = pd.DataFrame(dtype=float)

    # Shift timestamps from climate year to target year
    curtailment = _shift_time_columns_to_year(curtailment, int(target_year))

    da = xr.DataArray(curtailment).unstack("dim_0")
    da = _assign_coordinates_and_attrs(da, "Curtailment of renewable generation")
    da.attrs["kind"] = "flexibility_indicator"
    return da


def network_cache(func: Callable) -> Callable:
    """
    A custom caching decorator for functions that accept a PyPSA network.

    It creates a cache key by hashing the network's name and data for
    key components, making it suitable for caching results of functions
    that operate on non-hashable pypsa.Network objects.
    """
    cache = {}
    # Define a maximum cache size to prevent memory issues
    max_cache_size = 128

    @functools.wraps(func)
    def wrapper(pypsa_network: pypsa.Network, *args, **kwargs):
        # Create a hashable key from non-hashable arguments
        # Using network name and a hash of key components' data
        component_hashes = []
        for comp in sorted(pypsa_network.all_components):
            df = pypsa_network.df(comp)
            if not df.empty:
                # Using pandas' hashing is faster than converting to bytes
                component_hashes.append(str(pd.util.hash_pandas_object(df).sum()))

        # Combine hashes and other arguments to form the key
        hash_str = "".join(component_hashes)
        m = hashlib.sha256()
        m.update(pypsa_network.name.encode())
        m.update(hash_str.encode())
        m.update(str(args).encode())
        m.update(str(sorted(kwargs.items())).encode())
        key = m.hexdigest()

        if key in cache:
            return cache[key]
        else:
            # Evict the oldest entry if the cache is full
            if len(cache) >= max_cache_size:
                # This removes the first-inserted item
                cache.pop(next(iter(cache)))

            result = func(pypsa_network, *args, **kwargs)
            cache[key] = result
            return result

    return wrapper


def calculate_residual_load_df(
    n: pypsa.Network,
    renewable_carriers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate residual load per bidding zone and timestep.

    Residual load is calculated as demand minus potential renewable supply
    minus must-run supply (from generators with is_must_run=True).

    Potential renewable supply is calculated from p_max_pu * p_nom, representing
    the maximum available renewable generation capacity at each timestep.

    In perfect foresight mode, availability is not considered for must-run supply
    to maintain consistency with planning assumptions.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    renewable_carriers : None | list
        renewable carriers

    Returns
    -------
    pd.DataFrame
    """
    # Create a cache key based on network metadata and parameters

    # In perfect foresight mode, don't consider availability for residual load
    # to maintain consistency with planning assumptions
    perfect_foresight = (
        n.meta.get("eva", {}).get("perfect_foresight", {}).get("enable", False)
    )
    consider_availability = not perfect_foresight

    buses = n.buses.index
    snapshots = n.snapshots

    # Calculate potential renewable supply from p_max_pu * p_nom
    # Filter for renewable generators that exist in the network
    renewable_gens = n.generators[n.generators.carrier.isin(renewable_carriers)].index

    if len(renewable_gens) > 0:
        # Calculate potential supply = p_max_pu * p_nom
        # Use p_max_pu_original if available (for consistent forecast error calculation)
        # This ensures residual load calculations use the same baseline renewable availability
        # before any reserve blocking modifications
        if "p_max_pu_original" in n.generators_t:
            availability_res = n.generators_t.p_max_pu_original.reindex(
                index=snapshots, columns=renewable_gens, fill_value=0.0
            )
            logger.debug("Using p_max_pu_original for residual load calculation")
        else:
            availability_res = as_dense(n, "Generator", "p_max_pu", renewable_gens)

        renewable_potential = availability_res.multiply(
            n.generators.loc[renewable_gens, "p_nom"], axis=1
        ).fillna(0.0)

        renewable_potential_per_bus = (
            renewable_potential.T.groupby(n.generators.loc[renewable_gens, "bus"])
            .sum()
            .T.reindex(columns=buses, fill_value=0.0)
        )

    else:
        renewable_potential_per_bus = pd.DataFrame(
            0.0, index=n.snapshots, columns=buses
        )

    # Calculate demand
    load_per_bus = (
        n.loads_t.p_set.T.groupby(n.loads.bus)
        .sum()
        .T.reindex(columns=buses, fill_value=0.0)
    )

    must_run_supply_per_bus = pd.DataFrame(0.0, index=snapshots, columns=buses)
    if "is_must_run" in n.generators.columns:
        must_run_gens = n.generators.index[n.generators["is_must_run"] == True]
        if len(must_run_gens) > 0:
            # Get must-run capacity and weighted ratio
            if "must-run capacity" in n.generators.columns:
                must_run_capacity_static = n.generators.loc[
                    must_run_gens, "must-run capacity"
                ].fillna(0.0)

                # Get weighted average must-run ratio (defaults to 1.0)
                if "weighted average must-run ratio" in n.generators.columns:
                    weighted_ratios = n.generators.loc[
                        must_run_gens, "weighted average must-run ratio"
                    ].fillna(1.0)
                else:
                    weighted_ratios = pd.Series(1.0, index=must_run_gens)

                # Calculate must-run requirement: capacity * weighted_ratio
                must_run_requirement = must_run_capacity_static * weighted_ratios

                # Create time-series (constant over snapshots)
                must_run_capacity_df = pd.DataFrame(
                    index=snapshots, columns=must_run_gens, dtype=float
                )
                for gen in must_run_gens:
                    must_run_capacity_df[gen] = must_run_requirement[gen]

                # Apply availability if requested (default behavior)
                if consider_availability:
                    # Get availability timeseries (accounts for forced and planned outages)
                    if (
                        "availability" in n.generators_t
                        and not n.generators_t.availability.empty
                    ):
                        availability = n.generators_t.availability[
                            must_run_gens
                        ].reindex(snapshots, fill_value=1.0)
                        # Multiply must-run capacity by availability
                        must_run_capacity_df = must_run_capacity_df * availability
                    # If availability data doesn't exist, use full capacity (availability = 1.0)

                # Aggregate by bus
                gen_buses = n.generators.loc[must_run_gens, "bus"]
                must_run_supply_per_bus = (
                    must_run_capacity_df.T.groupby(gen_buses)
                    .sum()
                    .T.reindex(index=snapshots, columns=buses, fill_value=0.0)
                )

    residual_load_df = (
        load_per_bus - renewable_potential_per_bus - must_run_supply_per_bus
    )

    return residual_load_df


@network_cache
def calculate_residual_load(
    n: pypsa.Network,
    renewable_carriers: list[str] | None = None,
) -> xr.DataArray:
    """
    Calculate residual load per bidding zone and timestep.

    Residual load is calculated as demand minus potential renewable supply
    minus must-run supply (from generators with is_must_run=True).

    Potential renewable supply is calculated from p_max_pu * p_nom, representing
    the maximum available renewable generation capacity at each timestep.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    renewable_carriers : None | list
        renewable carriers

    Returns
    -------
    xr.DataArray
        Residual load with dimensions (TY, CY, time, BZ)
    """
    # Create a cache key based on network metadata and parameters

    logger.debug("Calculating residual load...")

    # Extract target year from network wildcards
    if "wildcards" in n.meta and "ty" in n.meta["wildcards"]:
        target_year = n.meta["wildcards"]["ty"]
    else:
        # Fallback: extract year from first snapshot if wildcards not available
        target_year = n.snapshots[0].year

    # If renewable_carriers is not provided, use default values
    if renewable_carriers is None:
        renewable_carriers = RENEWABLE_CARRIERS

    residual_load_df = calculate_residual_load_df(n, renewable_carriers)

    residual_load_df = residual_load_df.T

    # Shift timestamps from climate year to target year
    residual_load = _shift_time_columns_to_year(residual_load_df, int(target_year))

    # Ensure index and columns have proper names for xarray conversion
    residual_load.index.name = "bus"
    residual_load.columns.name = "snapshot"

    da = xr.DataArray(residual_load)
    da = _assign_coordinates_and_attrs(
        da, "Residual load (demand - potential renewable supply - must-run supply)"
    )
    da.attrs["kind"] = "flexibility_indicator"

    return da


def calculate_residual_load_deviations(
    n: pypsa.Network,
    renewable_carriers: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """
    Calculate residual load deviations per bidding zone for different time periods.

    Deviations are computed at hourly resolution to ensure proper energy units (MWh)
    when summed. Each hour's deviation is computed against the appropriate timescale
    average (daily, weekly, or yearly). Seasonal deviations use monthly averages.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    renewable_carriers : list, optional
        List of renewable carriers, defaults to RENEWABLE_CARRIERS

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary with deviations from average residual load:
        - "daily": Hourly deviations from daily mean, dims (TY, CY, hour, BZ)
        - "weekly": Hourly deviations from weekly mean, dims (TY, CY, hour, BZ)
        - "seasonal": Each hour's monthly mean deviation from the yearly mean,
          dims (TY, CY, hour, BZ)
    """
    # Get residual load (hourly MW values)
    rl = calculate_residual_load(n, renewable_carriers)

    # Daily deviations: each hour compared to its day's average (unchanged)
    daily = rl.groupby("snapshot.dayofyear") - rl.groupby("snapshot.dayofyear").mean(
        "hour"
    )
    daily = daily.rename(dayofyear="day")

    # Weekly deviations: each hour gets its day's deviation from weekly mean
    # Step 1: Compute daily average for each day
    rl_daily_avg = rl.groupby("snapshot.date").mean("hour")

    # Step 2: Add week coordinate to daily averages
    rl_daily_avg = rl_daily_avg.assign_coords(
        week=("date", pd.to_datetime(rl_daily_avg.date).isocalendar().week.to_numpy())
    )

    # Step 3: Compute weekly mean of daily averages
    weekly_mean = rl_daily_avg.groupby("week").mean("date")

    # Step 4: Compute daily deviation from weekly mean (for each date)
    daily_deviation_from_weekly = rl_daily_avg.groupby("week") - weekly_mean

    # Step 5: Broadcast back to hourly resolution
    # Create mapping from each hour to its date's deviation
    hour_dates = pd.to_datetime(rl.snapshot.values).date
    weekly = daily_deviation_from_weekly.sel(date=hour_dates)
    # Rename date dimension to hour to match rl dimensions
    weekly = weekly.rename(date="hour")
    weekly = weekly.assign_coords(
        hour=rl.hour.values,
        week=("hour", pd.to_datetime(rl.snapshot.values).isocalendar().week.to_numpy()),
    )

    # Seasonal deviations: each hour gets its month's deviation from yearly mean
    # Step 1: Add month coordinate to daily averages
    rl_daily_avg = rl_daily_avg.assign_coords(
        month=("date", pd.to_datetime(rl_daily_avg.date).month.to_numpy())
    )

    # Step 2: Compute monthly averages of daily averages
    monthly_avg = rl_daily_avg.groupby("month").mean("date")

    # Step 3: Compute yearly mean (using daily averages for robustness with unequal month lengths)
    yearly_mean = rl_daily_avg.mean("date")

    # Step 4: Compute monthly deviation from yearly mean
    monthly_deviation_from_yearly = monthly_avg - yearly_mean

    # Step 5: Broadcast back to hourly resolution
    hour_months = pd.to_datetime(rl.snapshot.values).month.to_numpy()
    seasonal = monthly_deviation_from_yearly.sel(month=hour_months)
    # Rename month dimension to hour to match rl dimensions
    seasonal = seasonal.rename(month="hour")
    seasonal = seasonal.assign_coords(
        hour=rl.hour.values,
        month=("hour", hour_months),
    )

    return {
        "daily": daily,
        "weekly": weekly,
        "seasonal": seasonal,
    }


def calculate_flexibility_needs(
    n: pypsa.Network,
    renewable_carriers: None | list = None,
) -> dict[str, xr.DataArray]:
    """
    Calculate flexibility needs per bidding zone for different time horizons.

    Flexibility needs are calculated as the sum of absolute deviations from
    average residual load over daily, weekly, and seasonal time periods. These
    indicate the magnitude of flexibility requirements at different time scales.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    renewable_carriers : list, optional
        List of renewable carriers, defaults to RENEWABLE_CARRIERS

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary containing flexibility needs with keys:
        - "daily": Daily flexibility needs with dimensions (TY, CY, day, BZ)
        - "weekly": Weekly flexibility needs with dimensions (TY, CY, week, BZ)
        - "seasonal": Seasonal flexibility needs with dimensions (TY, CY, BZ)
    """
    # Get residual load and aggregations
    deviations = calculate_residual_load_deviations(n, renewable_carriers)

    # daily flexibility need
    fn_daily = abs(deviations["daily"]).groupby("day").sum("hour")
    fn_daily.attrs["units"] = "MWh"
    fn_daily.attrs["description"] = (
        "Daily flexibility need (sum of absolute deviations from daily mean residual load)"
    )
    fn_daily.attrs["kind"] = "flexibility_indicator"

    # weekly flexibility need (deviations are now at hourly resolution)
    fn_weekly = abs(deviations["weekly"]).groupby("week").sum("hour")
    fn_weekly.attrs["units"] = "MWh"
    fn_weekly.attrs["description"] = (
        "Weekly flexibility need (sum of absolute hourly deviations from weekly mean residual load)"
    )
    fn_weekly.attrs["kind"] = "flexibility_indicator"

    # seasonal flexibility need (deviations are now at hourly resolution)
    fn_seasonal = abs(deviations["seasonal"]).sum("hour")
    fn_seasonal.attrs["units"] = "MWh"
    fn_seasonal.attrs["description"] = (
        "Seasonal flexibility need (sum of absolute hourly deviations from seasonal mean residual load)"
    )
    fn_seasonal.attrs["kind"] = "flexibility_indicator"

    return {
        "daily": fn_daily,
        "weekly": fn_weekly,
        "seasonal": fn_seasonal,
    }


def calculate_maximum_ramping_requirement(
    n: pypsa.Network, renewable_carriers: None | list = None
) -> xr.DataArray:
    """
    Calculate maximum ramping requirement per bidding zone.

    Maximum ramping requirement is the largest hour-to-hour change in
    residual load, indicating peak flexibility needs.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object
    renewable_carriers : list, optional
        List of renewable carriers, defaults to RENEWABLE_CARRIERS

    Returns
    -------
    xr.DataArray
        Maximum ramping requirement with dimensions (TY, CY, BZ)
    """
    # Get residual load data
    residual_load = calculate_residual_load(n, renewable_carriers)

    # Calculate hour-to-hour changes (ramps)
    ramps = abs(residual_load.diff("hour"))

    # Find maximum ramp for each bidding zone
    max_ramps = ramps.max("hour")

    max_ramps.attrs["units"] = "MWh"
    max_ramps.attrs["description"] = (
        "Maximum ramping requirement (largest hour-to-hour change in residual load)"
    )
    max_ramps.attrs["kind"] = "flexibility_indicator"

    return max_ramps


def calculate_ramping_shortfall_indicator(
    n_with_ramping: pypsa.Network,
    n_no_ramping: pypsa.Network,
    renewable_carriers: None | list = None,
) -> dict[str, xr.DataArray]:
    """
    Calculate ramping shortfall indicators by comparing dispatch with and without ramping constraints.

    The ramping shortfall is separated into upward (curtailment-related) and downward (ENS-related)
    components per ACER requirements. These are two separate indicators, not a ratio.

    Implementation follows ACER + JRC agreement:
    - Run ID with ramping constraints and DA commitments (regular run)
    - Run ID without ramping constraints but keeping DA commitments
    - Calculate separate upward and downward components

    The difference is kept signed as the model without ramping can lead to higher ENS/curtailment in
    certain rolling horizons, so we evaluate the net impact here.

    Parameters
    ----------
    n_with_ramping : pypsa.Network
        Intraday network with ramping constraints (regular run)
    n_no_ramping : pypsa.Network
        Intraday network without ramping constraints
    renewable_carriers : list, optional
        List of renewable carriers, defaults to RENEWABLE_CARRIERS

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary with keys:
        - "upward": signed curtailment differences
        - "downward": signed ENS differences
        Both with dimensions (hour, BZ) and units MWh
    """
    # Calculate curtailment for both networks
    curtailment_with_ramping = calculate_curtailment(n_with_ramping, renewable_carriers)
    curtailment_no_ramping = calculate_curtailment(n_no_ramping, renewable_carriers)

    # Calculate ENS for both networks
    ens_with_ramping = calculate_energy_not_served(n_with_ramping)
    ens_no_ramping = calculate_energy_not_served(n_no_ramping)

    # Calculate differences: curtailment and ENS should be higher WITH ramping constraints
    # because ramping limits reduce system flexibility
    # Sum over carrier dimension in curtailment to match ENS dimensions
    curtailment_diff = curtailment_with_ramping.sum(
        "carrier"
    ) - curtailment_no_ramping.sum("carrier")
    ens_diff = ens_with_ramping - ens_no_ramping

    # Separate into upward and downward components. Signed on purpose -- see the
    # docstring: a per-cell clip here is what turns zone-to-zone and
    # hour-to-hour reshuffling into a fictitious shortfall.
    ramping_shortfall_upward = curtailment_diff
    ramping_shortfall_downward = ens_diff

    ramping_shortfall_upward.attrs["units"] = "MWh"
    ramping_shortfall_upward.attrs["description"] = (
        "Ramping shortfall upward (signed curtailment difference due to ramping constraints)"
    )
    ramping_shortfall_upward.attrs["kind"] = "flexibility_indicator"

    ramping_shortfall_downward.attrs["units"] = "MWh"
    ramping_shortfall_downward.attrs["description"] = (
        "Ramping shortfall downward (signed ENS difference due to ramping constraints)"
    )
    ramping_shortfall_downward.attrs["kind"] = "flexibility_indicator"

    logger.info(
        "Ramping shortfall: net upward %.1f MWh, net downward %.1f MWh",
        float(ramping_shortfall_upward.sum()),
        float(ramping_shortfall_downward.sum()),
    )

    return {
        "upward": ramping_shortfall_upward,
        "downward": ramping_shortfall_downward,
    }


def calculate_ramping_contribution(
    n: pypsa.Network,
    n_worst_events: list[int],
    renewable_carriers: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """
    Calculate technology contributions to ramping requirements.

    Implements formula 13.16 without flooring to calculate each technology's
    fractional contribution to meeting the N worst ramping events.

    Notes
    -----
    - Individual carrier fractions can be **negative** (e.g. storage absorbing
      during a ramp-up event) or **greater than 1.0** (when a technology
      over-contributes to compensate for counter-ramping technologies).
    - Per-BZ carrier sums should ideally equal 1.0, meaning all ramping is
      accounted for. Deviations can occur in small test networks where the
      residual load calculation (using potential renewable supply) diverges
      from the actual dispatch balance.
    """
    target_year = n.meta["wildcards"]["ty"]

    # 1. Residual Load Diff
    rl = calculate_residual_load(n, renewable_carriers)
    rl_diff = rl.diff("hour")
    rl_diff_abs = abs(rl_diff)
    rl_diff_sign = np.sign(rl_diff)

    # 2. Dispatch Diff (Generators & Storage)
    dispatch_df = n.statistics.energy_balance(
        comps=["Generator", "StorageUnit"],
        groupby=["carrier", "bus"],
        aggregate_time=False,
        nice_names=False,
    ).droplevel("component")

    dispatch_df = _shift_time_columns_to_year(dispatch_df, int(target_year))
    dispatch = xr.DataArray(dispatch_df).unstack("dim_0")
    dispatch = _assign_coordinates_and_attrs(dispatch, "Dispatch")
    dispatch = dispatch.fillna(0.0)
    dispatch_diff = dispatch.diff("hour")

    # 3. Cross-border flows Diff
    # Follow cross-border contribution convention: K matrix approach
    K = _calculate_incidence_matrix(n)
    link_flows = n.links_t.p0.copy()
    link_flows = _shift_time_rows_to_year(link_flows, int(target_year))
    link_flows_da = xr.DataArray(link_flows, dims=["snapshot", "link"])
    K_da = xr.DataArray(K, dims=["BZ", "link"])

    # Net injection into each BZ from all links
    cb_injection = (link_flows_da * K_da).sum("link")
    cb_injection = cb_injection.fillna(0.0)
    # Rename snapshot to hour to match dispatch
    cb_injection = cb_injection.rename({"snapshot": "hour"})
    cb_injection.coords["hour"] = rl.hour.values
    cb_diff = cb_injection.diff("hour")

    # Combine Interconnectors into dispatch_diff
    cb_diff_expanded = cb_diff.expand_dims(carrier=["Interconnectors"])
    total_dispatch_diff = xr.concat(
        [dispatch_diff, cb_diff_expanded], dim="carrier", join="outer"
    )

    results = {}

    # 4. Calculate contribution for each N in n_worst_events
    for n_worst in n_worst_events:
        # Calculate numerator: (g_t - g_{t-1}) * sign(RL_t - RL_{t-1})
        numerator_all = total_dispatch_diff * rl_diff_sign

        # Initialize empty list to store per-BZ arrays
        bz_results = []

        for bz in rl.BZ.values:
            # Get the indices (hours) of the largest n_worst ramps for this BZ
            try:
                rl_diff_abs_bz = rl_diff_abs.sel(BZ=bz)
            except KeyError:
                continue

            k_events = min(n_worst, len(rl_diff_abs_bz.hour))
            if k_events == 0:
                continue

            arr = rl_diff_abs_bz.to_numpy()
            # Handle NaNs: convert NaN to -1.0 so they go to the bottom
            arr_no_nan = np.nan_to_num(arr, nan=-1.0)

            top_k_indices = np.argsort(arr_no_nan)[-k_events:]
            top_k_hours = rl_diff_abs_bz.hour.values[top_k_indices]

            # Sum numerator and denominator over these specific hours
            numerator_bz = numerator_all.sel(BZ=bz, hour=top_k_hours).sum("hour")
            denominator_bz = rl_diff_abs_bz.sel(hour=top_k_hours).sum("hour")

            # Contribution fraction for this BZ (protect against div by zero)
            rc_bz = xr.where(denominator_bz > 0, numerator_bz / denominator_bz, 0.0)

            # Reattach BZ dimension
            rc_bz = rc_bz.expand_dims(BZ=[bz])
            bz_results.append(rc_bz)

        if bz_results:
            rc_da = xr.concat(bz_results, dim="BZ")

            rc_da.attrs["units"] = "fraction"
            rc_da.attrs["description"] = (
                f"Technology contributions to ramping requirements (worst {n_worst} events)"
            )
            rc_da.attrs["kind"] = "flexibility_indicator"

            results[f"n{n_worst}"] = rc_da

    return results


def _calculate_incidence_matrix(n: pypsa.Network) -> pd.DataFrame:
    """
    Calculate incidence matrix K_{n,l} for interconnectors.

    The incidence matrix indicates how each link affects each bidding zone:
    - K_{n,l} = +1 if link l ends at zone n (import)
    - K_{n,l} = -1 if link l starts from zone n (export)
    - K_{n,l} = 0 if link l doesn't connect to zone n

    Note that this is the negative of K_{n,l} as defined by equation (1.2) of the
    modelling specification, which is export-positive. The leading minus sign of
    specification equations (13.11)-(13.13) is therefore absorbed here, and the
    cross-border contributions multiply flows by +K. Do not add it back.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object

    Returns
    -------
    pd.DataFrame
        Incidence matrix with bidding zones as rows and links as columns
    """
    buses = n.buses.index
    links = n.links.index

    # Build K as a plain NumPy array and wrap it once at the end;
    # writing through K.values isn't guaranteed to update K
    K = np.zeros((len(buses), len(links)), dtype=int)

    # Get bus0 and bus1 for all links
    bus0_series = n.links.loc[links, "bus0"]
    bus1_series = n.links.loc[links, "bus1"]

    # Positional assignment on the array: pandas .loc with two list-likes
    # selects the cross-product of rows and columns, which would set every listed
    # link for every listed bus. Pairwise (row, column) indexing is required here.
    bus_pos = pd.Series(np.arange(len(buses)), index=buses)
    link_pos = pd.Series(np.arange(len(links)), index=links)

    # For bus0 connections (export = -1)
    valid_bus0 = bus0_series[bus0_series.isin(buses)]
    if not valid_bus0.empty:
        K[bus_pos[valid_bus0.values].values, link_pos[valid_bus0.index].values] = -1

    # For bus1 connections (import = +1)
    valid_bus1 = bus1_series[bus1_series.isin(buses)]
    if not valid_bus1.empty:
        K[bus_pos[valid_bus1.values].values, link_pos[valid_bus1.index].values] = 1

    return pd.DataFrame(K, index=buses, columns=links)


def calculate_technology_contributions(
    n: pypsa.Network, renewable_carriers: None | list = None
) -> dict[str, xr.DataArray]:
    """
    Calculate technology contributions to flexibility needs.

    Implements equations 12.8-12.13 from the modelling specification to
    calculate how much each technology (generators, storage, DSR) contributes
    to meeting daily, weekly, and seasonal flexibility needs.

    The contribution is calculated by multiplying the technology dispatch
    with the sign of the residual load deviation for each time period.
    This determines whether the technology helps meet flexibility needs
    (positive contribution) or increases them (negative contribution).

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object containing solved dispatch results
    renewable_carriers : list, optional
        List of renewable carriers for residual load calculation.
        If None, uses default renewable carriers from network metadata.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary containing technology contributions with keys:
        - "daily": Daily contributions aggregated over all days
            Dimensions: (carrier, BZ)
            Units: MWh
        - "weekly": Weekly contributions aggregated over all weeks
            Dimensions: (carrier, BZ)
            Units: MWh
        - "seasonal": Seasonal contributions aggregated over the year
            Dimensions: (carrier, BZ)
            Units: MWh
    """
    # Extract target year from network wildcards
    target_year = n.meta["wildcards"]["ty"]

    deviations = calculate_residual_load_deviations(n, renewable_carriers)

    # Get technology dispatch using energy_balance (supply - withdrawal)
    dispatch_df = n.statistics.energy_balance(
        comps=["Generator", "StorageUnit"],
        groupby=["carrier", "bus"],
        aggregate_time=False,
        nice_names=False,
    ).droplevel("component")

    # Shift timestamps to target year
    dispatch_df = _shift_time_columns_to_year(dispatch_df, int(target_year))

    # Convert to DataArray and reshape
    dispatch = xr.DataArray(dispatch_df).unstack("dim_0")
    # energy_balance leaves NaN where a (carrier, bus) pair has no dispatch in an hour.
    # These are zeros, not missing data: without filling them the period averages below
    # would only span the hours a technology was running and its off-hours would drop
    # out of the sums.
    dispatch = dispatch.fillna(0.0)
    # Add day, week and month coordinates before assigning other coordinates
    dispatch = dispatch.assign_coords(
        day=("snapshot", dispatch.snapshot.dt.dayofyear.values),
        week=("snapshot", dispatch.snapshot.dt.isocalendar().week.to_numpy()),
        month=("snapshot", dispatch.snapshot.dt.month.values),
    )
    dispatch = _assign_coordinates_and_attrs(dispatch, "Dispatch")

    sign_daily = np.sign(deviations["daily"])
    # Use dispatch variations: (g - g_mean_daily) instead of just g
    dispatch_variation_daily = dispatch.groupby("day") - dispatch.groupby("day").mean(
        "hour"
    )
    cb_daily = (
        (dispatch_variation_daily * sign_daily).groupby("day").sum("hour").sum("day")
    )

    # Add attributes to daily contributions
    cb_daily.attrs["units"] = "MWh"
    cb_daily.attrs["description"] = "Daily technology contributions to flexibility"
    cb_daily.attrs["kind"] = "flexibility_indicator"

    # Weekly contributions: deviations are now at hourly resolution
    # Broadcast dispatch daily variations to hourly resolution
    sign_weekly = np.sign(deviations["weekly"])
    dispatch_daily = dispatch.groupby("snapshot.date").mean("hour")
    # Add week coordinate to dispatch_daily
    date_as_datetime = pd.to_datetime(dispatch_daily.date.values)
    dispatch_daily = dispatch_daily.assign_coords(
        week=("date", date_as_datetime.isocalendar().week.to_numpy())
    )
    # Compute daily deviation from weekly mean
    dispatch_variation_weekly = dispatch_daily.groupby("week") - dispatch_daily.groupby(
        "week"
    ).mean("date")
    # Broadcast to hourly resolution (each hour gets its day's variation)
    hour_dates = pd.to_datetime(dispatch.snapshot.values).date
    dispatch_variation_weekly_hourly = dispatch_variation_weekly.sel(date=hour_dates)
    dispatch_variation_weekly_hourly = dispatch_variation_weekly_hourly.rename(
        date="hour"
    )
    dispatch_variation_weekly_hourly = dispatch_variation_weekly_hourly.assign_coords(
        hour=dispatch.hour.values,
        week=(
            "hour",
            pd.to_datetime(dispatch.snapshot.values).isocalendar().week.to_numpy(),
        ),
    )
    cb_weekly = (
        (dispatch_variation_weekly_hourly * sign_weekly)
        .groupby("week")
        .sum("hour")
        .sum("week")
    )

    # Add attributes to weekly contributions
    cb_weekly.attrs["units"] = "MWh"
    cb_weekly.attrs["description"] = "Weekly technology contributions to flexibility"
    cb_weekly.attrs["kind"] = "flexibility_indicator"

    # Seasonal contributions: deviations are now at hourly resolution
    # Broadcast dispatch monthly variations to hourly resolution
    sign_seasonal = np.sign(deviations["seasonal"])
    dispatch_monthly = dispatch.groupby("month").mean("hour")
    # Compute monthly deviation from yearly mean (using daily mean for robustness with unequal month lengths)
    dispatch_variation_seasonal = dispatch_monthly - dispatch_daily.mean("date")
    # Broadcast to hourly resolution (each hour gets its month's variation)
    hour_months = pd.to_datetime(dispatch.snapshot.values).month.to_numpy()
    dispatch_variation_seasonal_hourly = dispatch_variation_seasonal.sel(
        month=hour_months
    )
    dispatch_variation_seasonal_hourly = dispatch_variation_seasonal_hourly.rename(
        month="hour"
    )
    dispatch_variation_seasonal_hourly = (
        dispatch_variation_seasonal_hourly.assign_coords(
            hour=dispatch.hour.values,
            month=("hour", hour_months),
        )
    )
    cb_seasonal = (dispatch_variation_seasonal_hourly * sign_seasonal).sum("hour")

    # Add attributes to seasonal contributions
    cb_seasonal.attrs["units"] = "MWh"
    cb_seasonal.attrs["description"] = (
        "Seasonal technology contributions to flexibility"
    )
    cb_seasonal.attrs["kind"] = "flexibility_indicator"

    return {
        "daily": cb_daily,
        "weekly": cb_weekly,
        "seasonal": cb_seasonal,
    }


def calculate_cross_border_contributions(
    n: pypsa.Network, renewable_carriers: None | list = None
) -> dict[str, xr.DataArray]:
    """
    Calculate cross-border (interconnector) contributions to flexibility needs.

    Implements equations 12.13-12.15 from the modelling specification to
    calculate how much each interconnector contributes to meeting daily,
    weekly, and seasonal flexibility needs of each bidding zone.

    Cross-border contributions are calculated by multiplying link flows
    with the incidence matrix and the sign of residual load deviations.
    This determines whether interconnectors help meet flexibility needs
    (positive contribution) or increase them (negative contribution).

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object containing solved dispatch results
    renewable_carriers : list, optional
        List of renewable carriers for residual load calculation.
        If None, uses default renewable carriers from network metadata.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary containing cross-border contributions with keys:
        - "daily": Daily contributions aggregated over all days
            Dimensions: (link, BZ)
            Units: MWh
        - "weekly": Weekly contributions aggregated over all weeks
            Dimensions: (link, BZ)
            Units: MWh
        - "seasonal": Seasonal contributions aggregated over the year
            Dimensions: (link, BZ)
            Units: MWh
    """
    # Extract target year from network wildcards
    target_year = n.meta["wildcards"]["ty"]

    # Get residual load deviations
    deviations = calculate_residual_load_deviations(n, renewable_carriers)

    # Calcukate flows
    link_flows = n.links_t.p0.copy()
    link_flows = _shift_time_rows_to_year(link_flows, int(target_year))

    # Calculate incidence matrix K_{n,l}
    K = _calculate_incidence_matrix(n)

    # Convert link flows to DataArray
    link_flows_da = xr.DataArray(link_flows, dims=["snapshot", "link"])
    K_da = xr.DataArray(K, dims=["BZ", "link"])
    # K already carries the direction (+1 where the link ends in the zone, i.e. an
    # import, -1 where it starts), so K * p0 is the net injection into each zone.
    flows = link_flows_da * K_da
    # Add day, week and month coordinates before assigning other coordinates
    flows = flows.assign_coords(
        day=("snapshot", flows.snapshot.dt.dayofyear.values),
        week=("snapshot", flows.snapshot.dt.isocalendar().week.to_numpy()),
        month=("snapshot", flows.snapshot.dt.month.values),
    )
    flows = _assign_coordinates_and_attrs(flows, "Cross-border Flow")

    sign_daily = np.sign(deviations["daily"])
    # Use flow variations: (f - f_mean_daily) instead of just f
    flows_variation_daily = flows.groupby("day") - flows.groupby("day").mean("hour")
    cb_daily = (
        (flows_variation_daily * sign_daily)
        .groupby("day")
        .sum("hour")
        .sum("day")
        .sum("link")
    )

    # Add attributes to daily contributions
    cb_daily.attrs["units"] = "MWh"
    cb_daily.attrs["description"] = (
        "Daily technology contributions to crossborder flows"
    )
    cb_daily.attrs["kind"] = "flexibility_indicator"

    # Weekly contributions: deviations are now at hourly resolution
    # Broadcast flow daily variations to hourly resolution
    sign_weekly = np.sign(deviations["weekly"])
    flows_daily = flows.groupby("snapshot.date").mean("hour")
    # Add week coordinate to flows_daily based on date dimension
    flows_daily = flows_daily.assign_coords(
        week=(
            "date",
            pd.to_datetime(flows_daily.date.values).isocalendar().week.to_numpy(),
        )
    )
    # Compute daily deviation from weekly mean
    flows_variation_weekly = flows_daily.groupby("week") - flows_daily.groupby(
        "week"
    ).mean("date")
    # Broadcast to hourly resolution (each hour gets its day's variation)
    hour_dates = pd.to_datetime(flows.snapshot.values).date
    flows_variation_weekly_hourly = flows_variation_weekly.sel(date=hour_dates)
    flows_variation_weekly_hourly = flows_variation_weekly_hourly.rename(date="hour")
    flows_variation_weekly_hourly = flows_variation_weekly_hourly.assign_coords(
        hour=flows.hour.values,
        week=(
            "hour",
            pd.to_datetime(flows.snapshot.values).isocalendar().week.to_numpy(),
        ),
    )
    cb_weekly = (
        (flows_variation_weekly_hourly * sign_weekly)
        .groupby("week")
        .sum("hour")
        .sum("week")
        .sum("link")
    )

    # Add attributes to weekly contributions
    cb_weekly.attrs["units"] = "MWh"
    cb_weekly.attrs["description"] = (
        "Weekly technology contributions to crossborder flows"
    )
    cb_weekly.attrs["kind"] = "flexibility_indicator"

    # Seasonal contributions: deviations are now at hourly resolution
    # Broadcast flow monthly variations to hourly resolution
    sign_seasonal = np.sign(deviations["seasonal"])
    flows_monthly = flows.groupby("month").mean("hour")
    # Compute monthly deviation from yearly mean (using daily mean for robustness with unequal month lengths)
    flows_variation_seasonal = flows_monthly - flows_daily.mean("date")
    # Broadcast to hourly resolution (each hour gets its month's variation)
    hour_months = pd.to_datetime(flows.snapshot.values).month.to_numpy()
    flows_variation_seasonal_hourly = flows_variation_seasonal.sel(month=hour_months)
    flows_variation_seasonal_hourly = flows_variation_seasonal_hourly.rename(
        month="hour"
    )
    flows_variation_seasonal_hourly = flows_variation_seasonal_hourly.assign_coords(
        hour=flows.hour.values,
        month=("hour", hour_months),
    )
    cb_seasonal = (
        (flows_variation_seasonal_hourly * sign_seasonal).sum("hour").sum("link")
    )

    # Add attributes to seasonal contributions
    cb_seasonal.attrs["units"] = "MWh"
    cb_seasonal.attrs["description"] = (
        "Seasonal technology contributions to crossborder flows"
    )
    cb_seasonal.attrs["kind"] = "flexibility_indicator"

    return {
        "daily": cb_daily,
        "weekly": cb_weekly,
        "seasonal": cb_seasonal,
    }


def calculate_short_term_need_deviations(
    n: pypsa.Network,
    n_da: pypsa.Network,
    renewable_carriers: None | list = None,
) -> dict[str, xr.DataArray]:
    """
    Calculate short-term need deviations per bidding zone for different time periods.

    Short-term need (forecast error) is defined as the difference in residual load between intraday
    and day-ahead markets, representing the deviation from forecast. This function
    calculates deviations from average short-term need over daily, weekly, and
    seasonal time periods.

    Deviations are computed at hourly resolution to ensure proper energy units (MWh)
    when summed.

    Parameters
    ----------
    n : pypsa.Network
        The intraday PyPSA network object
    n_da : pypsa.Network
        The day-ahead PyPSA network object
    renewable_carriers : list, optional
        List of renewable carriers, defaults to RENEWABLE_CARRIERS

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary with deviations from average short-term need:
        - "hourly": Raw short-term need (forecast error), dims (TY, CY, hour, BZ)
        - "daily": Hourly deviations from daily mean, dims (TY, CY, hour, BZ)
        - "weekly": Hourly deviations from weekly mean, dims (TY, CY, hour, BZ)
        - "seasonal": Each hour's monthly mean deviation from the yearly mean,
          dims (TY, CY, hour, BZ)
    """
    # Calculate short-term need as difference in residual load
    rl_id = calculate_residual_load(n, renewable_carriers)
    rl_da = calculate_residual_load(n_da, renewable_carriers)
    short_term_need = rl_id - rl_da

    # Daily deviations: deviation from daily mean short-term need (unchanged)
    daily = short_term_need.groupby("snapshot.dayofyear") - short_term_need.groupby(
        "snapshot.dayofyear"
    ).mean("hour")
    daily = daily.rename(dayofyear="day")

    # Weekly deviations: each hour gets its day's deviation from weekly mean
    # Step 1: Compute daily average for each day
    stn_daily_avg = short_term_need.groupby("snapshot.date").mean("hour")

    # Step 2: Add week coordinate to daily averages
    stn_daily_avg = stn_daily_avg.assign_coords(
        week=("date", pd.to_datetime(stn_daily_avg.date).isocalendar().week.to_numpy())
    )

    # Step 3: Compute weekly mean of daily averages
    weekly_mean = stn_daily_avg.groupby("week").mean("date")

    # Step 4: Compute daily deviation from weekly mean (for each date)
    daily_deviation_from_weekly = stn_daily_avg.groupby("week") - weekly_mean

    # Step 5: Broadcast back to hourly resolution
    hour_dates = pd.to_datetime(short_term_need.snapshot.values).date
    weekly = daily_deviation_from_weekly.sel(date=hour_dates)
    weekly = weekly.rename(date="hour")
    weekly = weekly.assign_coords(
        hour=short_term_need.hour.values,
        week=(
            "hour",
            pd.to_datetime(short_term_need.snapshot.values)
            .isocalendar()
            .week.to_numpy(),
        ),
    )

    # Seasonal deviations: each hour gets its month's deviation from yearly mean
    # Step 1: Add month coordinate to daily averages
    stn_daily_avg = stn_daily_avg.assign_coords(
        month=("date", pd.to_datetime(stn_daily_avg.date).month.to_numpy())
    )

    # Step 2: Compute monthly averages of daily averages
    monthly_avg = stn_daily_avg.groupby("month").mean("date")

    # Step 3: Compute yearly mean (using daily averages for robustness with unequal month lengths)
    yearly_mean = stn_daily_avg.mean("date")

    # Step 4: Compute monthly deviation from yearly mean
    monthly_deviation_from_yearly = monthly_avg - yearly_mean

    # Step 5: Broadcast back to hourly resolution
    hour_months = pd.to_datetime(short_term_need.snapshot.values).month.to_numpy()
    seasonal = monthly_deviation_from_yearly.sel(month=hour_months)
    seasonal = seasonal.rename(month="hour")
    seasonal = seasonal.assign_coords(
        hour=short_term_need.hour.values,
        month=("hour", hour_months),
    )

    return {
        "hourly": short_term_need,
        "daily": daily,
        "weekly": weekly,
        "seasonal": seasonal,
    }


def calculate_short_term_technology_contributions(
    n: pypsa.Network,
    n_da: pypsa.Network,
    renewable_carriers: None | list = None,
) -> dict[str, xr.DataArray]:
    """
    Calculate technology contributions to short-term flexibility needs.

    The contribution is (g_ID - g_DA) * sign(short_term_need), i.e. how much
    each technology adjusted its dispatch from the DA plan in the direction of
    the short-term need (forecast error). The yearly mean is not subtracted
    because the short-term need itself is not bias-corrected.

    Parameters
    ----------
    n : pypsa.Network
        The intraday PyPSA network object containing solved dispatch results
    n_da : pypsa.Network
        The day-ahead PyPSA network object
    renewable_carriers : list, optional
        List of renewable carriers for residual load calculation.
        If None, uses default renewable carriers from network metadata.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary containing short-term technology contributions with key:
        - "hourly": Hourly contributions summed over all hours
            Dimensions: (carrier, BZ)
            Units: MWh
    """
    target_year = n.meta["wildcards"]["ty"]

    deviations = calculate_short_term_need_deviations(n, n_da, renewable_carriers)

    # ID dispatch
    dispatch_id_df = n.statistics.energy_balance(
        comps=["Generator", "StorageUnit"],
        groupby=["carrier", "bus"],
        aggregate_time=False,
        nice_names=False,
    ).droplevel("component")
    dispatch_id_df = _shift_time_columns_to_year(dispatch_id_df, int(target_year))
    dispatch_id = xr.DataArray(dispatch_id_df).unstack("dim_0").fillna(0.0)
    dispatch_id = _assign_coordinates_and_attrs(dispatch_id, "ID Dispatch")

    # DA dispatch
    dispatch_da_df = n_da.statistics.energy_balance(
        comps=["Generator", "StorageUnit"],
        groupby=["carrier", "bus"],
        aggregate_time=False,
        nice_names=False,
    ).droplevel("component")
    dispatch_da_df = _shift_time_columns_to_year(dispatch_da_df, int(target_year))
    dispatch_da = xr.DataArray(dispatch_da_df).unstack("dim_0").fillna(0.0)
    dispatch_da = _assign_coordinates_and_attrs(dispatch_da, "DA Dispatch")

    # Adjustment = ID dispatch - DA dispatch
    dispatch_adjustment = dispatch_id - dispatch_da

    sign_hourly = np.sign(deviations["hourly"])
    cb_hourly = (dispatch_adjustment * sign_hourly).sum("hour")

    cb_hourly.attrs["units"] = "MWh"
    cb_hourly.attrs["kind"] = "flexibility_indicator"
    cb_hourly.attrs["description"] = (
        "Technology contributions to hourly short-term needs"
    )

    return {"hourly": cb_hourly}


def calculate_short_term_cross_border_contributions(
    n: pypsa.Network,
    n_da: pypsa.Network,
    renewable_carriers: None | list = None,
) -> dict[str, xr.DataArray]:
    """
    Calculate cross-border contributions to short-term flexibility needs.

    The contribution is (f_ID - f_DA) * sign(short_term_need), i.e. how much
    each interconnector adjusted its flow from the DA plan in the direction of
    the short-term need (forecast error). The yearly mean is not subtracted
    because the short-term need itself is not bias-corrected.

    Parameters
    ----------
    n : pypsa.Network
        The intraday PyPSA network object containing solved dispatch results
    n_da : pypsa.Network
        The day-ahead PyPSA network object
    renewable_carriers : list, optional
        List of renewable carriers for residual load calculation.
        If None, uses default renewable carriers from network metadata.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary containing short-term cross-border contributions with key:
        - "hourly": Hourly contributions summed over all hours and links
            Dimensions: (BZ)
            Units: MWh
    """
    target_year = n.meta["wildcards"]["ty"]

    deviations = calculate_short_term_need_deviations(n, n_da, renewable_carriers)

    # Flow adjustment = ID flows - DA flows
    link_flows_actual = n.links_t.p0.copy()
    link_flows_forecasted = n_da.links_t.p0.copy()
    link_flows = link_flows_actual - link_flows_forecasted
    link_flows = _shift_time_rows_to_year(link_flows, int(target_year))

    K = _calculate_incidence_matrix(n)

    link_flows_da = xr.DataArray(link_flows, dims=["snapshot", "link"])
    K_da = xr.DataArray(K, dims=["BZ", "link"])
    # K already carries the direction (+1 where the link ends in the zone, i.e. an
    # import, -1 where it starts), so K * p0 is the net injection into each zone.
    flows = link_flows_da * K_da
    flows = _assign_coordinates_and_attrs(flows, "Cross-border Flow")

    sign_hourly = np.sign(deviations["hourly"])
    cb_hourly = (flows * sign_hourly).sum("hour").sum("link")

    cb_hourly.attrs["units"] = "MWh"
    cb_hourly.attrs["description"] = (
        "Cross-border contributions to hourly short-term needs"
    )
    cb_hourly.attrs["kind"] = "flexibility_indicator"

    return {"hourly": cb_hourly}


SHORT_TERM_NEED_PERCENTILES = {
    "p99.9": 99.9,
    "p99": 99,
    "p1": 1,
    "p0.1": 0.1,
}


def calculate_short_term_need(
    n: pypsa.Network,
    n_da: pypsa.Network,
    agg: Literal["mean", "sum", "p99.9", "p99", "p1", "p0.1"] | None = None,
    direction: Literal["upward", "downward", "both"] = "both",
) -> xr.DataArray:
    """
    Calculate the short-term need based on the residual load difference
    between intraday and day-ahead networks.

    Parameters
    ----------
    n : pypsa.Network
        The intraday PyPSA network object
    n_da : pypsa.Network
        The day-ahead PyPSA network object
    agg : str | None, optional
        Aggregation operation, by default None.
        Options: "sum", "mean", "p99.9", "p99", "p1", "p0.1"
        The "pXX.X" options compute the given percentile over the "hour" dim
        of the signed series (FNA methodology, Article 10), fixed at
        99.9/0.1 and 99/1.
    direction : str, optional
        Direction filter for forecast errors, by default "both"
        - "upward": only positive values (intraday > day-ahead)
        - "downward": only negative values (intraday < day-ahead)
        - "both": all values (no filtering)
        Ignored by the percentile aggregations, which use the signed series.

    Returns
    -------
    xr.DataArray
        DataArray containing short-term need.
    """
    rl = calculate_residual_load(n)
    rl_da = calculate_residual_load(n_da)

    signed_need = rl - rl_da
    short_term_need = signed_need

    # Apply directional filter
    if direction == "upward":
        short_term_need = short_term_need.where(short_term_need > 0, 0)
        direction_label = " (upward)"
    elif direction == "downward":
        short_term_need = short_term_need.where(short_term_need < 0, 0)
        direction_label = " (downward)"
    else:
        direction_label = ""

    short_term_need.attrs["description"] = f"Short term needs{direction_label}"

    if agg == "sum":
        short_term_need = short_term_need.sum("hour")
        short_term_need.attrs["description"] = (
            f"Total short term needs{direction_label}"
        )
    elif agg in SHORT_TERM_NEED_PERCENTILES:
        percentile = SHORT_TERM_NEED_PERCENTILES[agg]
        short_term_need = signed_need.reduce(np.percentile, dim="hour", q=percentile)
        short_term_need.attrs["description"] = (
            f"{percentile}th percentile of the signed short term need"
        )
    elif agg == "mean":
        short_term_need = short_term_need.mean("hour")
        short_term_need.attrs["description"] = (
            f"Average short term need{direction_label}"
        )
    elif agg is not None:
        raise ValueError(INVALID_AGGREGATION_MESSAGE)

    short_term_need.attrs["units"] = "MWh"
    short_term_need.attrs["kind"] = "flexibility_indicator"

    return short_term_need


# =============================================================================
# MARKET OUTPUT FUNCTIONS
# =============================================================================


def extract_market_clearing_prices(n: pypsa.Network, market_type: str) -> xr.DataArray:
    """
    Extract market clearing prices per bidding zone from marginal prices.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network
    market_type : str
        "day_ahead" or "intraday"

    Returns
    -------
    xr.DataArray
        Market clearing prices with dimensions (hour, bidding_zone)
    """
    if "marginal_price" not in n.buses_t:
        raise ValueError(f"No marginal prices found in {market_type} network")

    prices_df = n.buses_t.marginal_price.copy()
    da = xr.DataArray(prices_df, dims=["snapshot", "bidding_zone"])

    # Use the same coordinate assignment as flexibility indicators
    da = _assign_coordinates_and_attrs(
        da,
        f"Market clearing prices ({market_type})",
        units="€/MWh",
    )
    da.attrs["kind"] = "market_output"

    # Add additional metadata
    da.attrs["data_type"] = "Calculated"
    da.attrs["time_series"] = True
    da.attrs["market_model"] = market_type

    logger.info(f"Extracted {market_type} prices: {da.shape} (hours x bidding_zones)")
    return da


def calculate_average_prices(
    n: pypsa.Network, market_type: str, prices: xr.DataArray | None = None
) -> xr.DataArray:
    """
    Calculate average market clearing prices per bidding zone.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network
    market_type : str
        "day_ahead" or "intraday"
    prices : xr.DataArray | None, optional
        Pre-extracted prices. If None, prices will be extracted from the network.

    Returns
    -------
    xr.DataArray
        Average prices with dimensions (bidding_zone,)
    """
    # Extract prices if not provided
    if prices is None:
        prices = extract_market_clearing_prices(n, market_type)

    # Calculate average over time dimension (hour)
    avg_prices = prices.mean(dim="hour")

    # Add metadata
    avg_prices.attrs["units"] = PRICE_UNITS
    avg_prices.attrs["description"] = f"Average market clearing prices ({market_type})"
    avg_prices.attrs["data_type"] = "Calculated"
    avg_prices.attrs["time_series"] = False
    avg_prices.attrs["market_model"] = market_type
    avg_prices.attrs["kind"] = "market_output"

    logger.info(
        f"Calculated {market_type} average prices: {len(avg_prices.bidding_zone)} bidding zones"
    )
    return avg_prices


def calculate_zero_negative_price_hours(
    n: pypsa.Network, market_type: str, prices: xr.DataArray | None = None
) -> xr.DataArray:
    """
    Calculate number of hours with zero or negative market clearing prices per bidding zone.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network
    market_type : str
        "day_ahead" or "intraday"
    prices : xr.DataArray | None, optional
        Pre-extracted prices. If None, prices will be extracted from the network.

    Returns
    -------
    xr.DataArray
        Number of hours with zero or negative prices with dimensions (bidding_zone,)
    """
    # Extract prices if not provided
    if prices is None:
        prices = extract_market_clearing_prices(n, market_type)

    # Count hours where price <= 0
    zero_negative_hours = (prices <= 0).sum(dim="hour")

    # Add metadata
    # NOT "hours": xarray's CF coder auto-encodes any variable whose `units`
    # matches a recognized time-duration string as a timedelta64 on to_netcdf,
    # even for a plain integer count -- this round-trips through
    # decode_cf_timedelta on read and can overflow for large counts.
    zero_negative_hours.attrs["units"] = "hours (count)"
    zero_negative_hours.attrs["description"] = (
        f"Number of hours with zero or negative prices ({market_type})"
    )
    zero_negative_hours.attrs["data_type"] = "Calculated"
    zero_negative_hours.attrs["time_series"] = False
    zero_negative_hours.attrs["market_model"] = market_type
    zero_negative_hours.attrs["kind"] = "market_output"

    logger.info(
        f"Calculated {market_type} zero/negative price hours: {len(zero_negative_hours.bidding_zone)} bidding zones"
    )
    return zero_negative_hours


def calculate_price_volatility(
    n: pypsa.Network, market_type: str, prices: xr.DataArray | None = None
) -> xr.DataArray:
    """
    Calculate price volatility (standard deviation) per bidding zone.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network
    market_type : str
        "day_ahead" or "intraday"
    prices : xr.DataArray | None, optional
        Pre-extracted prices. If None, prices will be extracted from the network.

    Returns
    -------
    xr.DataArray
        Price volatility (standard deviation) with dimensions (bidding_zone,)
    """
    # Extract prices if not provided
    if prices is None:
        prices = extract_market_clearing_prices(n, market_type)

    # Calculate standard deviation over time dimension (hour)
    price_volatility = prices.std(dim="hour")

    # Add metadata
    price_volatility.attrs["units"] = PRICE_UNITS
    price_volatility.attrs["description"] = (
        f"Price volatility (std dev) ({market_type})"
    )
    price_volatility.attrs["data_type"] = "Calculated"
    price_volatility.attrs["time_series"] = False
    price_volatility.attrs["market_model"] = market_type
    price_volatility.attrs["kind"] = "market_output"

    logger.info(
        f"Calculated {market_type} price volatility: {len(price_volatility.bidding_zone)} bidding zones"
    )
    return price_volatility


def extract_total_co2_emissions(n: pypsa.Network, market_type: str) -> xr.DataArray:
    """
    Extract total CO2 emissions per bidding zone.

    Calculates annual CO2 emissions based on generation dispatch and emission factors.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network
    market_type : str
        "day_ahead" or "intraday"

    Returns
    -------
    xr.DataArray
        Total CO2 emissions with dimensions (bidding_zone,)
    """
    if "co2_emissions" not in n.carriers.columns:
        raise ValueError("No CO2 emission factors found in carriers")

    # Get data
    emission_factors = n.carriers["co2_emissions"]

    # Map emission factors to generators and calculate emissions by bus
    em = n.generators.carrier.map(emission_factors).fillna(0)
    emissions_by_bus = n.generators_t.p.sum().mul(em).groupby(n.generators.bus).sum()

    if emissions_by_bus.empty:
        raise ValueError(f"No emissions calculated for {market_type}")

    # Convert to sorted lists for xarray
    buses = sorted(emissions_by_bus.index)
    emissions = [emissions_by_bus[bus] for bus in buses]

    da = xr.DataArray(emissions, dims=["bidding_zone"], coords={"bidding_zone": buses})

    # Add metadata
    da.attrs["units"] = "tonnes CO2"
    da.attrs["description"] = f"Total CO2 emissions ({market_type})"
    da.attrs["data_type"] = "Calculated"
    da.attrs["time_series"] = False
    da.attrs["market_model"] = market_type
    da.attrs["kind"] = "market_output"

    logger.info(f"Extracted {market_type} CO2 emissions: {len(buses)} bidding zones")
    return da


def extract_net_power_transfer(n: pypsa.Network, market_type: str) -> xr.DataArray:
    """
    Extract net power transfer between bidding zones.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network
    market_type : str
        "day_ahead" or "intraday"

    Returns
    -------
    xr.DataArray
        Net power transfer with dimensions (hour, interconnection)
    """
    # Get link flows and create DataArray
    flows = n.links_t.p0.copy()
    da = xr.DataArray(flows, dims=["snapshot", "interconnection"])

    # Ensure interconnection coordinate has string dtype even when empty
    # This prevents dtype mismatch when merging datasets with empty and non-empty interconnections
    if da.interconnection.size == 0:
        da = da.assign_coords(interconnection=da.interconnection.astype(str))

    # Use the same coordinate assignment as flexibility indicators
    da = _assign_coordinates_and_attrs(
        da,
        f"Net power transfer between bidding zones ({market_type})",
        units="MWh",
    )
    da.attrs["kind"] = "market_output"

    # Add additional metadata
    da.attrs["data_type"] = "Raw"
    da.attrs["time_series"] = True
    da.attrs["market_model"] = market_type

    logger.info(
        f"Extracted {market_type} power transfers: {da.shape} (hours x interconnections)"
    )
    return da


def extract_reserve_capacity_per_generator(
    n: pypsa.Network, market_type: str
) -> dict[str, xr.DataArray]:
    """
    Extract reserve capacity per generator for all reserve types.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network
    market_type : str
        "day_ahead" or "intraday"

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary with reserve types as keys and DataArrays as values
    """
    reserve_data = {}

    # Find all reserve variables in generators_t
    reserve_vars = [key for key in n.generators_t.keys() if key.startswith("r_")]

    if not reserve_vars:
        logger.warning(f"No reserve data found in {market_type} network")
        return reserve_data

    for reserve_var in reserve_vars:
        reserve_df = n.generators_t[reserve_var].copy()

        # Only include generators that have non-zero reserve capacity
        non_zero_cols = reserve_df.columns[(reserve_df > 0).any()]
        if non_zero_cols.empty:
            logger.info(f"No generators with {reserve_var} reserves in {market_type}")
            continue

        reserve_df = reserve_df[non_zero_cols]
        da = xr.DataArray(reserve_df, dims=["snapshot", "generator"])

        # Use the same coordinate assignment as flexibility indicators
        da = _assign_coordinates_and_attrs(
            da,
            f"Reserve capacity {reserve_var} ({market_type})",
            units="MW",
        )
        da.attrs["kind"] = "market_output"

        # Add additional metadata
        da.attrs["data_type"] = "Calculated"
        da.attrs["time_series"] = True
        da.attrs["market_model"] = market_type

        reserve_data[reserve_var] = da
        logger.info(
            f"Extracted {market_type} {reserve_var}: {da.shape} (hours x generators)"
        )

    return reserve_data


def load_reservoir_storage_levels(
    storage_plan_path: str | list[str],
) -> xr.DataArray | None:
    """
    Load reservoir storage levels from pre-optimization storage plan.

    Parameters
    ----------
    storage_plan_path : str or list[str]
        Path(s) to storage plan CSV file(s). If a list is provided (multiple forced outage samples),
        the first path is used since storage planning is independent of forced outages.

    Returns
    -------
    xr.DataArray or None
        Reservoir storage levels with dimensions (hour, storage_unit)
    """
    # Handle list of paths (multiple forced outage samples)
    if isinstance(storage_plan_path, (list | tuple)):
        if not storage_plan_path:
            logger.warning("Empty list of storage plan paths provided")
            return None
        # Use the first path - storage plans should be identical across forced outage samples
        # since storage planning happens before forced outages are applied
        n_samples = len(storage_plan_path)
        storage_plan_path = storage_plan_path[0]
        logger.info(f"Using first storage plan from {n_samples} forced outage samples")

    if not Path(storage_plan_path).exists():
        logger.warning(f"Pre-optimization storage plan not found: {storage_plan_path}")
        return None

    try:
        # Read storage plan CSV
        storage_df = pd.read_csv(storage_plan_path, index_col=0, parse_dates=True)

        if storage_df.empty:
            logger.warning("Empty storage plan file")
            return None

        # Create DataArray with proper hour coordinates
        da = xr.DataArray(storage_df, dims=["snapshot", "storage_unit"])

        # Use the same coordinate assignment as flexibility indicators
        da = _assign_coordinates_and_attrs(
            da,
            "Reservoir storage levels (pre-optimization)",
            units="MWh",
        )
        da.attrs["kind"] = "market_output"

        # Add additional metadata
        da.attrs["data_type"] = "Raw"
        da.attrs["time_series"] = True
        da.attrs["pre_optimisation"] = True

        logger.info(
            f"Loaded reservoir storage levels: {da.shape} (hours x storage_units)"
        )
        return da

    except Exception as e:
        logger.error(f"Error loading storage plan: {e}")
        return None


def get_meta_data(n: pypsa.Network) -> xr.Dataset:
    """
    Get metadata from the network and return as an xarray dataset.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object

    Returns
    -------
    xr.Dataset
        Dataset containing metadata
    """
    ds = (
        n.carriers[["color", "nice_name"]]
        .rename(columns=lambda s: "_" + s)
        .rename_axis(index="carrier")
        .to_xarray()
    )

    return ds


def calculate_short_term_shortfall_curtailment(
    n: pypsa.Network,
    n_da: pypsa.Network,
    renewable_carriers: None | list = None,
    agg: Literal["mean", "sum", "max"] | None = None,
) -> xr.DataArray:
    """
    Calculate curtailment shortfall ID as compared to DA.

    Parameters
    ----------
    n : pypsa.Network
        Intraday PyPSA network object
    n_da : pypsa.Network
        Day-ahead PyPSA network object
    renewable_carriers : list, optional
        List of renewable carriers, defaults to RENEWABLE_CARRIERS
    agg : str | None, optional
        Aggregation operation, by default None

    Returns
    -------
    xr.DataArray
        Curtailment shortfall. If agg is None, returns timeseries with dimensions (TY, CY, time, BZ).
        If agg is specified, returns aggregated values with dimensions (TY, CY, BZ).
    """
    # Calculate curtailment for intraday and day-ahead networks
    curtailment_intraday = calculate_curtailment(n, renewable_carriers)
    curtailment_day_ahead = calculate_curtailment(n_da, renewable_carriers)

    # Calculate shortfall curtailment
    shortfall_curtailment = curtailment_intraday - curtailment_day_ahead

    # Sum over carriers to get total shortfall curtailment
    shortfall_curtailment = shortfall_curtailment.sum(dim="carrier")

    # Add attributes
    shortfall_curtailment.attrs["units"] = "MWh"
    shortfall_curtailment.attrs["description"] = "Short-term shortfall curtailment"
    shortfall_curtailment.attrs["kind"] = "flexibility_indicator"

    if agg == "sum":
        shortfall_curtailment = shortfall_curtailment.sum("hour")
        shortfall_curtailment = _assign_coordinates_and_attrs(
            shortfall_curtailment, "Total short-term shortfall curtailment"
        )
        shortfall_curtailment.attrs["kind"] = "flexibility_indicator"
    elif agg == "max":
        shortfall_curtailment = shortfall_curtailment.max("hour")
        shortfall_curtailment = _assign_coordinates_and_attrs(
            shortfall_curtailment, "Maximum short-term shortfall curtailment"
        )
        shortfall_curtailment.attrs["kind"] = "flexibility_indicator"
    elif agg == "mean":
        shortfall_curtailment = shortfall_curtailment.mean("hour")
        shortfall_curtailment = _assign_coordinates_and_attrs(
            shortfall_curtailment, "Average short-term shortfall curtailment"
        )
        shortfall_curtailment.attrs["kind"] = "flexibility_indicator"
    elif agg is not None:
        raise ValueError(INVALID_AGGREGATION_MESSAGE)

    return shortfall_curtailment


def calculate_short_term_shortfall_ens(
    n: pypsa.Network,
    n_da: pypsa.Network,
    agg: Literal["mean", "sum", "max"] | None = None,
) -> xr.DataArray:
    """
    Calculate ENS shortfall ID as compared to DA.

    Parameters
    ----------
    n : pypsa.Network
        Intraday PyPSA network object
    n_da : pypsa.Network
        Day-ahead PyPSA network object
    agg : str | None, optional
        Aggregation operation, by default None

    Returns
    -------
    xr.DataArray
        ENS shortfall. If agg is None, returns timeseries with dimensions (TY, CY, time, BZ).
        If agg is specified, returns aggregated values with dimensions (TY, CY, BZ).
    """
    # Calculate ENS for intraday and day-ahead networks
    ens_intraday = calculate_energy_not_served(n)
    ens_day_ahead = calculate_energy_not_served(n_da)

    # Calculate shortfall ENS
    shortfall_ens = ens_intraday - ens_day_ahead

    # Add attributes
    shortfall_ens.attrs["units"] = "MWh"
    shortfall_ens.attrs["description"] = "Short-term shortfall energy not served"
    shortfall_ens.attrs["kind"] = "flexibility_indicator"

    if agg == "sum":
        shortfall_ens = shortfall_ens.sum("hour")
        shortfall_ens = _assign_coordinates_and_attrs(
            shortfall_ens, "Total short-term shortfall energy not served"
        )
        shortfall_ens.attrs["kind"] = "flexibility_indicator"
    elif agg == "max":
        shortfall_ens = shortfall_ens.max("hour")
        shortfall_ens = _assign_coordinates_and_attrs(
            shortfall_ens, "Maximum short-term shortfall energy not served"
        )
        shortfall_ens.attrs["kind"] = "flexibility_indicator"
    elif agg == "mean":
        shortfall_ens = shortfall_ens.mean("hour")
        shortfall_ens = _assign_coordinates_and_attrs(
            shortfall_ens, "Average short-term shortfall energy not served"
        )
        shortfall_ens.attrs["kind"] = "flexibility_indicator"
    elif agg is not None:
        raise ValueError(INVALID_AGGREGATION_MESSAGE)

    return shortfall_ens


# =============================================================================
# PROBABILISTIC SHORT-TERM REQUIREMENTS FUNCTIONS
# =============================================================================


def assign_system_conditions(
    timestamps: pd.DatetimeIndex,
    condition_months: dict[str, list[int]],
    condition_hours: dict[str, list[int]],
) -> pd.Series:
    """
    Assign a system condition label to each timestamp based on month and hour groupings.

    System conditions are identified as a combination of a season label (e.g. "1", "2")
    and a time-of-day label (e.g. "a", "b").  For example, winter daytime = "1a".

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Timestamps to classify.
    condition_months : dict[str, list[int]]
        Mapping of season label -> list of months belonging to that season.
        Example: {"1": [11, 12, 1, 2, 3], "2": [4, 5, 6, 7, 8, 9, 10]}.
    condition_hours : dict[str, list[int]]
        Mapping of time-of-day label -> list of hours belonging to that period.
        Example: {"a": [7, ..., 21], "b": [22, 23, 0, ..., 6]}.

    Returns
    -------
    pd.Series
        String condition label for each timestamp (e.g. "1a", "2b").
        Timestamps that do not match any condition return NaN.
    """
    months = timestamps.month  # ty: ignore[unresolved-attribute]
    hours = timestamps.hour  # ty: ignore[unresolved-attribute]

    season_labels = pd.Series(index=timestamps, dtype=object)
    for label, month_list in condition_months.items():
        mask = months.isin(month_list)
        season_labels[mask] = label

    tod_labels = pd.Series(index=timestamps, dtype=object)
    for label, hour_list in condition_hours.items():
        mask = hours.isin(hour_list)
        tod_labels[mask] = label

    combined = season_labels + tod_labels
    return combined


def compute_forecast_error_distribution(
    network_pairs: list[tuple[pypsa.Network, pypsa.Network]],
    config: dict,
    renewable_carriers: list[str] | None = None,
) -> xr.Dataset:
    """
    Compute the residual load forecast error probability distribution across all CYs.

    The forecast error is defined as ``RL_id - RL_da`` (positive = upward need,
    negative = downward need).  Errors are pooled across all provided (ID, DA)
    network pairs (i.e. all climate years in the run) and characterised separately
    for each bidding zone and system condition.

    System conditions are formed by combining season (month-based) and time-of-day
    (hour-based) labels, yielding e.g. "1a" (winter day), "1b" (winter night),
    "2a" (summer day), "2b" (summer night).

    Parameters
    ----------
    network_pairs : list[tuple[pypsa.Network, pypsa.Network]]
        List of (intraday_network, day_ahead_network) pairs, one per climate year.
    config : dict
        Configuration dictionary containing eva.short_term_requirements settings.
    renewable_carriers : list[str] | None, optional
        Renewable carrier names, defaults to module-level ``RENEWABLE_CARRIERS``.

    Returns
    -------
    xr.Dataset
        Dataset with variables:

        - ``forecast_error_p_low``  – (condition, BZ): *percentile_low* of errors [MWh]
        - ``forecast_error_p_high`` – (condition, BZ): *percentile_high* of errors [MWh]

        Negative values of ``forecast_error_p_low`` indicate downward flex need;
        positive values of ``forecast_error_p_high`` indicate upward flex need.
    """
    if renewable_carriers is None:
        renewable_carriers = RENEWABLE_CARRIERS

    # Parse short-term requirements config
    st_cfg = config["eva"]["short_term_requirements"]

    condition_months: dict[str, list[int]] = {}
    condition_hours: dict[str, list[int]] = {}

    for key, val in st_cfg.items():
        if key.startswith("condition_") and key.endswith("_months"):
            label = key[len("condition_") : -len("_months")]
            condition_months[label] = list(val)
        elif key.startswith("condition_") and key.endswith("_hours"):
            label = key[len("condition_") : -len("_hours")]
            condition_hours[label] = list(val)

    percentile_low = st_cfg["percentile_low"]
    percentile_high = st_cfg["percentile_high"]

    all_errors: list[pd.DataFrame] = []
    all_conditions: list[pd.Series] = []

    for n_id, n_da in network_pairs:
        rl_id = calculate_residual_load_df(n_id, renewable_carriers)
        rl_da = calculate_residual_load_df(n_da, renewable_carriers)

        # Align on common index (handles clock shifts between DA and ID)
        common_idx = rl_id.index.intersection(rl_da.index)
        errors = rl_id.loc[common_idx] - rl_da.loc[common_idx]

        conditions = assign_system_conditions(
            common_idx, condition_months, condition_hours
        )

        all_errors.append(errors)
        all_conditions.append(conditions)

    errors_df = pd.concat(all_errors)
    conditions_series = pd.concat(all_conditions)

    # Determine all condition labels present in data
    all_cond_labels = sorted(conditions_series.dropna().unique())
    bz_labels = errors_df.columns.tolist()

    p_low_data = np.full((len(all_cond_labels), len(bz_labels)), np.nan)
    p_high_data = np.full((len(all_cond_labels), len(bz_labels)), np.nan)

    for i, cond in enumerate(all_cond_labels):
        mask = conditions_series == cond
        subset = errors_df.loc[mask]
        if len(subset) == 0:
            continue
        p_low_data[i, :] = np.percentile(subset.values, percentile_low, axis=0)
        p_high_data[i, :] = np.percentile(subset.values, percentile_high, axis=0)

    p_low = xr.DataArray(
        p_low_data,
        dims=["condition", "BZ"],
        coords={"condition": all_cond_labels, "BZ": bz_labels},
        attrs={
            "description": f"Forecast error p{percentile_low} per system condition",
            "units": "MWh",
            "kind": "flexibility_indicator",
        },
    )
    p_high = xr.DataArray(
        p_high_data,
        dims=["condition", "BZ"],
        coords={"condition": all_cond_labels, "BZ": bz_labels},
        attrs={
            "description": f"Forecast error p{percentile_high} per system condition",
            "units": "MWh",
            "kind": "flexibility_indicator",
        },
    )

    return xr.Dataset({"forecast_error_p_low": p_low, "forecast_error_p_high": p_high})


def compute_residual_flex_capacity(
    n_da: pypsa.Network,
    renewable_carriers: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """
    Compute residual upward and downward flexibility capacity available in the DA schedule.

    The residual flex capacity is the headroom (UP) or footroom (DOWN) that dispatchable
    generators can provide from their DA dispatch point, after accounting for:

    - Generator capacity bounds (p_nom, p_min_pu, p_max_pu)
    - Ramping limits (ramp_limit_up, ramp_limit_down) applied to the DA dispatch
    - FCR reserves fully blocked (both directions)
    - FRR reserves blocked in the direction of the forecast error (imbalance method):
      only the upward FRR is blocked for downward flex and vice-versa

    Renewable generators are excluded as they set output at maximum available.
    Storage units (hydro reservoirs, pumped hydro, batteries, implicit DSR) are excluded.

    Parameters
    ----------
    n_da : pypsa.Network
        Solved day-ahead PyPSA network.
    renewable_carriers : list[str] | None, optional
        Renewable carrier names, defaults to module-level ``RENEWABLE_CARRIERS``.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary with keys:

        - ``"up"``:   (hour, BZ) upward flex capacity [MWh]
        - ``"down"``: (hour, BZ) downward flex capacity [MWh]

        Both DataArrays have ``kind = "flexibility_indicator"`` and ``units = "MWh"``.
    """
    if renewable_carriers is None:
        renewable_carriers = RENEWABLE_CARRIERS

    gens = n_da.generators
    gens_t = n_da.generators_t
    snapshots = n_da.snapshots
    buses = n_da.buses.index

    # Carriers to exclude from flex capacity calculation
    excluded_carriers = [
        "load shedding",
        "load spillage",
        "reserve shedding",
        *renewable_carriers,
    ]

    # Identify dispatchable (non-renewable, non-shedding) generators
    disp_gens = gens[~gens.carrier.isin(excluded_carriers)].index

    if len(disp_gens) == 0:
        zero = pd.DataFrame(0.0, index=snapshots, columns=buses)
        zero_da = _assign_coordinates_and_attrs(
            xr.DataArray(zero), "Zero flex capacity"
        )
        zero_da.attrs["kind"] = "flexibility_indicator"
        return {"up": zero_da, "down": zero_da.copy()}

    # DA dispatch
    dispatch = gens_t.p.reindex(index=snapshots, columns=disp_gens, fill_value=0.0)

    # Maximum capacity (p_max_pu * p_nom)
    p_nom = gens.loc[disp_gens, "p_nom"].fillna(0.0)
    p_min_pu = as_dense(n_da, "Generator", "p_min_pu", disp_gens).reindex(
        index=snapshots
    )
    p_max_pu = as_dense(n_da, "Generator", "p_max_pu", disp_gens).reindex(
        index=snapshots
    )
    p_max = p_max_pu.multiply(p_nom)
    p_min = p_min_pu.multiply(p_nom)

    # Ramp limits (fraction of p_nom per hour, NaN means no limit)
    ramp_up_rate = gens.loc[disp_gens, "ramp_limit_up"].fillna(1.0)
    ramp_dn_rate = gens.loc[disp_gens, "ramp_limit_down"].fillna(1.0)
    ramp_up_mw = (ramp_up_rate * p_nom).clip(lower=0.0)
    ramp_dn_mw = (ramp_dn_rate * p_nom).clip(lower=0.0)

    # Upper bound considering ramp up from current dispatch
    p_max_ramp = dispatch.add(ramp_up_mw, axis=1).clip(upper=p_max)
    # Lower bound considering ramp down from current dispatch
    p_min_ramp = dispatch.subtract(ramp_dn_mw, axis=1).clip(lower=p_min)

    flex_up_gen = (p_max_ramp - dispatch).clip(lower=0.0)
    flex_dn_gen = (dispatch - p_min_ramp).clip(lower=0.0)

    # Subtract FCR reserves (fully blocked in both directions)
    fcr_up_col = "r_FCR_up"
    fcr_dn_col = "r_FCR_down"
    if fcr_up_col in gens_t:
        fcr_up = gens_t[fcr_up_col].reindex(
            index=snapshots, columns=disp_gens, fill_value=0.0
        )
        flex_up_gen = (flex_up_gen - fcr_up).clip(lower=0.0)
    if fcr_dn_col in gens_t:
        fcr_dn = gens_t[fcr_dn_col].reindex(
            index=snapshots, columns=disp_gens, fill_value=0.0
        )
        flex_dn_gen = (flex_dn_gen - fcr_dn).clip(lower=0.0)

    # Subtract FRR reserves using imbalance method:
    # FRR_up reduces upward flex (it is committed to cover downward deviations)
    # FRR_down reduces downward flex (it is committed to cover upward deviations)
    frr_up_col = "r_FRR_up"
    frr_dn_col = "r_FRR_down"
    if frr_up_col in gens_t:
        frr_up = gens_t[frr_up_col].reindex(
            index=snapshots, columns=disp_gens, fill_value=0.0
        )
        flex_up_gen = (flex_up_gen - frr_up).clip(lower=0.0)
    if frr_dn_col in gens_t:
        frr_dn = gens_t[frr_dn_col].reindex(
            index=snapshots, columns=disp_gens, fill_value=0.0
        )
        flex_dn_gen = (flex_dn_gen - frr_dn).clip(lower=0.0)

    # Aggregate by bus (bidding zone)
    gen_buses = gens.loc[disp_gens, "bus"]
    flex_up_bz = (
        flex_up_gen.T.groupby(gen_buses).sum().T.reindex(columns=buses, fill_value=0.0)
    )
    flex_dn_bz = (
        flex_dn_gen.T.groupby(gen_buses).sum().T.reindex(columns=buses, fill_value=0.0)
    )

    target_year = n_da.meta["wildcards"]["ty"]
    flex_up_bz = _shift_time_rows_to_year(flex_up_bz, int(target_year))
    flex_dn_bz = _shift_time_rows_to_year(flex_dn_bz, int(target_year))

    flex_up_da = _assign_coordinates_and_attrs(
        xr.DataArray(flex_up_bz),
        "Residual upward flex capacity (DA, reserve-adjusted)",
    )
    flex_up_da.attrs["kind"] = "flexibility_indicator"

    flex_dn_da = _assign_coordinates_and_attrs(
        xr.DataArray(flex_dn_bz),
        "Residual downward flex capacity (DA, reserve-adjusted)",
    )
    flex_dn_da.attrs["kind"] = "flexibility_indicator"

    return {"up": flex_up_da, "down": flex_dn_da}


def calculate_probabilistic_short_term_requirements(
    n_id: pypsa.Network,
    n_da: pypsa.Network,
    error_distribution: xr.Dataset,
    config: dict | None,
    renewable_carriers: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """
    Compute probabilistic short-term requirements for each timestep in a network.

    For each hour the function:

    1. Identifies the system condition (season × time-of-day).
    2. Looks up the forecast error percentiles for that condition and BZ from
       ``error_distribution``.
    3. Computes upward/downward uncovered need as the excess of the tail error
       beyond the available residual flex capacity.
    4. Computes expected ENS and expected curtailment as ``uncovered_need × tail_probability``.

    Parameters
    ----------
    n_id : pypsa.Network
        Solved intraday PyPSA network (used only for timestamp index and metadata).
    n_da : pypsa.Network
        Solved day-ahead PyPSA network (used for flex capacity computation).
    error_distribution : xr.Dataset
        Output of :func:`compute_forecast_error_distribution`.  Must contain
        ``forecast_error_p_low`` and ``forecast_error_p_high`` variables.
    config : dict
        Configuration dictionary containing eva.short_term_requirements settings.
    renewable_carriers : list[str] | None, optional
        Renewable carrier names, defaults to module-level ``RENEWABLE_CARRIERS``.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary with keys:

        - ``"uncovered_need_up"``      – (hour, BZ) MW uncovered upward need
        - ``"uncovered_need_down"``    – (hour, BZ) MW uncovered downward need
        - ``"expected_ens"``           – (hour, BZ) expected ENS [MW * tail_prob]
        - ``"expected_curtailment"``   – (hour, BZ) expected curtailment [MW * tail_prob]

        All arrays have ``units = "MWh"`` and ``kind = "flexibility_indicator"``.
    """
    # Parse short-term requirements config
    if config is None:
        raise ValueError(
            "config must be provided when using probabilistic short-term requirements method"
        )
    st_cfg = config["eva"]["short_term_requirements"]

    condition_months: dict[str, list[int]] = {}
    condition_hours: dict[str, list[int]] = {}

    for key, val in st_cfg.items():
        if key.startswith("condition_") and key.endswith("_months"):
            label = key[len("condition_") : -len("_months")]
            condition_months[label] = list(val)
        elif key.startswith("condition_") and key.endswith("_hours"):
            label = key[len("condition_") : -len("_hours")]
            condition_hours[label] = list(val)

    percentile_low = st_cfg["percentile_low"]
    percentile_high = st_cfg["percentile_high"]

    target_year = n_id.meta["wildcards"]["ty"]
    snapshots = n_id.snapshots
    shifted_index = pd.DatetimeIndex(
        [ts.replace(year=int(target_year)) for ts in snapshots]
    )

    # Hour-of-year integer coordinate (matches the convention used throughout the module)
    hour_of_year = (
        (shifted_index.dayofyear - 1) * 24  # ty:ignore[unresolved-attribute]
        + shifted_index.hour  # ty:ignore[unresolved-attribute]
        + 1
    )

    # Residual flex capacity from DA (indexed by integer hour-of-year and BZ)
    flex = compute_residual_flex_capacity(n_da, renewable_carriers)
    flex_up = flex["up"]  # dims: (BZ, hour) with integer hour-of-year
    flex_dn = flex["down"]  # same

    buses = n_da.buses.index.tolist()

    # Assign system conditions to each snapshot using shifted timestamps
    conditions = assign_system_conditions(
        shifted_index, condition_months, condition_hours
    )

    # Tail probabilities (fraction of distribution beyond the percentile)
    prob_up = (100.0 - percentile_high) / 100.0  # prob of error > p_high
    prob_dn = percentile_low / 100.0  # prob of error < p_low

    n_t = len(shifted_index)
    n_bz = len(buses)
    uncov_up = np.zeros((n_t, n_bz))
    uncov_dn = np.zeros((n_t, n_bz))

    # Vectorised per-condition block: group timesteps by condition, then apply
    # the same percentile values to all hours in that group simultaneously.
    unique_conds = conditions.dropna().unique()
    dist_cond_vals = error_distribution.condition.values.tolist()

    for cond in unique_conds:
        if cond not in dist_cond_vals:
            logger.warning(
                "Condition %s not found in error distribution, skipping", cond
            )
            continue

        cond_mask = (conditions == cond).values  # bool array over time
        t_indices = np.nonzero(cond_mask)[0]
        if len(t_indices) == 0:
            continue

        # Percentile values for this condition, shape (n_bz,)
        p_high_vals = (
            error_distribution["forecast_error_p_high"]
            .sel(condition=cond)
            .sel(BZ=buses)
            .values  # (n_bz,)
        )
        p_low_vals = (
            error_distribution["forecast_error_p_low"]
            .sel(condition=cond)
            .sel(BZ=buses)
            .values  # (n_bz,)
        )

        # Flex capacity for the selected hours, shape (n_t_cond, n_bz)
        hoy_cond = hour_of_year[t_indices]
        # flex_up / flex_dn are indexed as (BZ, hour) in xarray
        fu_block = flex_up.sel(
            hour=hoy_cond
        ).values  # (n_t_cond, n_bz) or (n_bz, n_t_cond)
        fd_block = flex_dn.sel(hour=hoy_cond).values

        # Ensure shape is (n_t_cond, n_bz)
        if fu_block.ndim == 2 and fu_block.shape[0] == n_bz:
            fu_block = fu_block.T
            fd_block = fd_block.T

        # Broadcast percentile values to (n_t_cond, n_bz)
        p_high_block = np.broadcast_to(p_high_vals, fu_block.shape)
        p_low_block = np.broadcast_to(p_low_vals, fd_block.shape)

        unc_up_block = np.maximum(0.0, p_high_block - fu_block)
        unc_dn_block = np.maximum(0.0, np.abs(np.minimum(0.0, p_low_block)) - fd_block)

        uncov_up[np.ix_(t_indices, np.arange(n_bz))] = unc_up_block
        uncov_dn[np.ix_(t_indices, np.arange(n_bz))] = unc_dn_block

    exp_ens = uncov_up * prob_up
    exp_curt = uncov_dn * prob_dn

    coords = {"hour": hour_of_year, "BZ": buses}

    # Create DataArrays inline like other flexibility indicators
    uncovered_need_up = xr.DataArray(uncov_up, dims=["hour", "BZ"], coords=coords)
    uncovered_need_up.attrs.update(
        description="Probabilistic uncovered upward need",
        units="MWh",
        kind="flexibility_indicator",
    )

    uncovered_need_down = xr.DataArray(uncov_dn, dims=["hour", "BZ"], coords=coords)
    uncovered_need_down.attrs.update(
        description="Probabilistic uncovered downward need",
        units="MWh",
        kind="flexibility_indicator",
    )

    expected_ens = xr.DataArray(exp_ens, dims=["hour", "BZ"], coords=coords)
    expected_ens.attrs.update(
        description="Probabilistic expected energy not served",
        units="MWh",
        kind="flexibility_indicator",
    )

    expected_curtailment = xr.DataArray(exp_curt, dims=["hour", "BZ"], coords=coords)
    expected_curtailment.attrs.update(
        description="Probabilistic expected curtailment",
        units="MWh",
        kind="flexibility_indicator",
    )

    return {
        "uncovered_need_up": uncovered_need_up,
        "uncovered_need_down": uncovered_need_down,
        "expected_ens": expected_ens,
        "expected_curtailment": expected_curtailment,
    }


# =============================================================================
# NECP TARGET COMPARISON FUNCTIONS
# =============================================================================


def _aggregate_to_country(data: xr.DataArray, bz_dim: str = "BZ") -> xr.DataArray:
    """
    Aggregate bidding zone data to country level.

    Country code is derived from the first 2 characters of the BZ code.
    E.g., "DE00" -> "DE", "AT00" -> "AT", "NL00" -> "NL".

    Parameters
    ----------
    data : xr.DataArray
        Data with a bidding zone dimension.
    bz_dim : str
        Name of the bidding zone dimension.

    Returns
    -------
    xr.DataArray
        Data aggregated to country level with "country" dimension.
    """
    bz_values = data[bz_dim].values
    countries = [str(bz)[:2] for bz in bz_values]
    data_with_country = data.assign_coords(country=(bz_dim, countries))
    return data_with_country.groupby("country").sum(bz_dim)


def calculate_res_generation_by_country(
    n: pypsa.Network,
    renewable_carriers: list[str] | None = None,
) -> xr.DataArray:
    """
    Calculate RES generation in TWh by country and carrier category (Wind/Solar/Hydro).

    1. Get supply by carrier and BZ
    2. Map model carriers to NECP categories (Wind, Solar, Hydro)
    3. Aggregate BZ to country
    4. Convert MWh to TWh

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network.
    renewable_carriers : list[str] | None
        List of renewable carrier names. If None, uses module default.

    Returns
    -------
    xr.DataArray
        RES generation with dimensions (country, carrier_category).
        Units: TWh.
    """
    # Get supply by carrier and BZ
    supply = n.statistics.supply(
        comps=["Generator", "StorageUnit"],
        groupby=["carrier", "bus"],
        nice_names=False,
        aggregate_time="sum",
    ).droplevel("component")

    # Convert to DataArray
    da = xr.DataArray(supply).unstack("dim_0")
    if "bus" in da.dims:
        da = da.rename({"bus": "BZ"})

    # Filter to only include carriers in NECP mapping
    necp_carriers = []
    for category, carriers in NECP_TO_MODEL_CARRIERS.items():
        necp_carriers.extend(carriers)

    # Filter to NECP-relevant carriers
    available_carriers = [c for c in necp_carriers if c in da.carrier.values]
    if not available_carriers:
        logger.warning("No NECP-relevant renewable carriers found in supply data")
        # Return empty DataArray with expected structure
        return xr.DataArray(
            np.empty((0, 0)),
            dims=["country", "carrier_category"],
            coords={"country": [], "carrier_category": []},
            attrs={
                "units": "TWh",
                "description": "RES generation by country and carrier category",
                "kind": "necp_comparison",
            },
        )

    da_filtered = da.sel(carrier=available_carriers)

    # Map carriers to NECP categories
    carrier_to_category = {}
    for category, carriers in NECP_TO_MODEL_CARRIERS.items():
        for carrier in carriers:
            carrier_to_category[carrier] = category

    # Create category coordinate and group by it
    categories = [
        carrier_to_category.get(str(c), "Other") for c in da_filtered.carrier.values
    ]
    da_with_cat = da_filtered.assign_coords(carrier_category=("carrier", categories))
    da_by_category = da_with_cat.groupby("carrier_category").sum("carrier")

    # Aggregate BZ to country
    da_by_country = _aggregate_to_country(da_by_category, "BZ")

    # Convert MWh to TWh
    da_twh = da_by_country / 1_000_000

    da_twh.attrs["units"] = "TWh"
    da_twh.attrs["description"] = "RES generation by country and carrier category"
    da_twh.attrs["kind"] = "necp_comparison"

    return da_twh


def calculate_necp_res_generation_comparison(
    n: pypsa.Network,
    target_year: int,
    config: dict[str, Any],
    renewable_carriers: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """
    Compare model RES generation against NECP targets.

    Parameters
    ----------
    n : pypsa.Network
        Solved PyPSA network.
    target_year : int
        Target year (2030, 2040, 2050).
    config : dict
        Configuration dictionary.
    renewable_carriers : list[str] | None
        List of renewable carrier names.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary with keys:
        - res_generation_model: Model RES generation (TWh) by country & carrier
        - res_generation_target: NECP targets (TWh) by country & carrier
        - res_generation_diff_pct: Percentage difference ((model-target)/target * 100)

        Positive diff = model exceeds target
        Negative diff = model below target
    """
    # Get model RES generation by country
    model_gen = calculate_res_generation_by_country(n, renewable_carriers)

    # Load NECP targets from CSV
    necp_config = config["costs"]["necp_targets"]
    csv_fallback = necp_config["file"]
    targets_df = load_necp_targets(csv_fallback, target_year)

    if targets_df.empty:
        logger.warning(f"No NECP targets found for target year {target_year}")
        # Return model data with NaN-filled targets and diffs matching model shape
        return {
            "res_generation_model": model_gen,
            "res_generation_target": xr.full_like(
                model_gen, np.nan, dtype=float
            ).assign_attrs(
                {
                    "units": "TWh",
                    "description": f"NECP RES generation targets for {target_year}",
                    "kind": "necp_comparison",
                }
            ),
            "res_generation_diff_pct": xr.full_like(
                model_gen, np.nan, dtype=float
            ).assign_attrs(
                {
                    "units": "%",
                    "description": "Percentage difference: (model - target) / target * 100",
                    "kind": "necp_comparison",
                }
            ),
        }

    # Pivot targets to (country, carrier_category) structure
    # Rename variable2 to carrier_category for consistency
    targets_df = targets_df.rename(columns={"variable2": "carrier_category"})
    targets_pivot = targets_df.pivot_table(
        index="country_short",
        columns="carrier_category",
        values="value",
        aggfunc="sum",
    )

    # Convert to xarray DataArray
    target_da = xr.DataArray(
        targets_pivot.values,
        dims=["country", "carrier_category"],
        coords={
            "country": targets_pivot.index.tolist(),
            "carrier_category": targets_pivot.columns.tolist(),
        },
    )
    target_da.attrs["units"] = "TWh"
    target_da.attrs["description"] = f"NECP RES generation targets for {target_year}"
    target_da.attrs["kind"] = "necp_comparison"

    # Align model and target data for comparison
    # Get common countries and carriers
    model_countries = (
        set(model_gen.country.values) if "country" in model_gen.dims else set()
    )
    target_countries = (
        set(target_da.country.values) if "country" in target_da.dims else set()
    )
    common_countries = sorted(model_countries & target_countries)

    model_carriers = (
        set(model_gen.carrier_category.values)
        if "carrier_category" in model_gen.dims
        else set()
    )
    target_carriers = (
        set(target_da.carrier_category.values)
        if "carrier_category" in target_da.dims
        else set()
    )
    common_carriers = sorted(model_carriers & target_carriers)

    if not common_countries or not common_carriers:
        logger.warning(
            f"No overlap between model and NECP targets. "
            f"Model countries: {model_countries}, Target countries: {target_countries}. "
            f"Model carriers: {model_carriers}, Target carriers: {target_carriers}."
        )
        return {
            "res_generation_model": model_gen,
            "res_generation_target": target_da,
            "res_generation_diff_pct": xr.full_like(
                model_gen, np.nan, dtype=float
            ).assign_attrs(
                {
                    "units": "%",
                    "description": "Percentage difference: (model - target) / target * 100",
                    "kind": "necp_comparison",
                }
            ),
        }

    # Select common subset for comparison
    model_aligned = model_gen.sel(
        country=common_countries, carrier_category=common_carriers
    )
    target_aligned = target_da.sel(
        country=common_countries, carrier_category=common_carriers
    )

    # Calculate percentage difference: (model - target) / target * 100
    # Handle division by zero: where target is 0, result is NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_pct = (model_aligned - target_aligned) / target_aligned * 100
        # Replace inf with NaN (from division by zero)
        diff_pct = xr.where(np.isinf(diff_pct), np.nan, diff_pct)

    diff_pct.attrs["units"] = "%"
    diff_pct.attrs["description"] = (
        "Percentage difference: (model - target) / target * 100"
    )
    diff_pct.attrs["kind"] = "necp_comparison"

    logger.info(
        f"Calculated NECP comparison for {len(common_countries)} countries, "
        f"{len(common_carriers)} carrier categories"
    )

    return {
        "res_generation_model": model_aligned,
        "res_generation_target": target_aligned,
        "res_generation_diff_pct": diff_pct,
    }


# =============================================================================
# CO2 TARGET COMPARISON FUNCTIONS
# =============================================================================


def calculate_co2_target_comparison(
    n: pypsa.Network,
    target_year: int,
    config: dict,
) -> dict[str, xr.DataArray]:
    """
    Compare modelled electricity-sector CO₂ emissions against EU member state targets.

    Steps:

    1. Extract intraday CO₂ emissions per bidding zone (tonnes CO₂) using the
       already-solved network.
    2. Aggregate bidding zones to countries (first 2 chars of BZ code).
    3. Convert from tonnes CO₂ to MtCO₂ (÷ 1,000,000).
    4. Load CO₂ targets from the CSV fallback file via
       :mod:`~euflex.eu_flex_model.scripts.data_helpers`.
    5. Align model and target on common countries.
    6. Calculate percentage difference: ``(model − target) / target × 100``.

    Parameters
    ----------
    n : pypsa.Network
        Solved intraday PyPSA network.
    target_year : int
        Target year (e.g. 2026, 2030, 2035).
    config : dict
        Workflow configuration dictionary.

    Returns
    -------
    dict[str, xr.DataArray]
        Keys:

        - ``"co2_emissions_model"`` – modelled emissions per country (MtCO₂)
        - ``"co2_emissions_target"`` – policy target per country (MtCO₂eq)
        - ``"co2_emissions_diff_pct"`` – percentage difference
          ``(model − target) / target × 100`` (positive = above target)
    """
    # ------------------------------------------------------------------
    # Steps 1-3: model emissions → per country in MtCO₂
    # ------------------------------------------------------------------
    co2_bz = extract_total_co2_emissions(
        n, "intraday"
    )  # dim: bidding_zone, unit: tonnes

    co2_country = _aggregate_to_country(
        co2_bz, "bidding_zone"
    )  # dim: country, unit: tonnes

    co2_country_mt = co2_country / 1_000_000  # → MtCO₂
    co2_country_mt.attrs["units"] = "MtCO2"
    co2_country_mt.attrs["description"] = (
        f"Modelled electricity-sector CO\u2082 emissions by country ({target_year})"
    )
    co2_country_mt.attrs["kind"] = "co2_comparison"

    # ------------------------------------------------------------------
    # Step 4: load targets from CSV
    # ------------------------------------------------------------------
    co2_config = config["costs"]["co2_targets"]
    csv_fallback = co2_config["file"]

    targets_df = load_co2_targets(csv_fallback, target_year)

    if targets_df.empty:
        logger.warning(f"No CO\u2082 targets found for target year {target_year}")
        empty_attrs_target = {
            "units": "MtCO2eq",
            "description": f"CO\u2082 emission targets by country ({target_year})",
            "kind": "co2_comparison",
        }
        empty_attrs_diff = {
            "units": "%",
            "description": "Percentage difference: (model \u2212 target) / target \u00d7 100",
            "kind": "co2_comparison",
        }
        return {
            "co2_emissions_model": co2_country_mt,
            "co2_emissions_target": xr.full_like(
                co2_country_mt, np.nan, dtype=float
            ).assign_attrs(empty_attrs_target),
            "co2_emissions_diff_pct": xr.full_like(
                co2_country_mt, np.nan, dtype=float
            ).assign_attrs(empty_attrs_diff),
        }

    # ------------------------------------------------------------------
    # Step 5: build target DataArray aligned to countries
    # ------------------------------------------------------------------
    targets_df = targets_df.set_index("country_short")["value"]
    target_da = xr.DataArray(
        targets_df.values,
        dims=["country"],
        coords={"country": targets_df.index.tolist()},
    )
    target_da.attrs["units"] = "MtCO2eq"
    target_da.attrs["description"] = (
        f"CO\u2082 emission targets by country ({target_year})"
    )
    target_da.attrs["kind"] = "co2_comparison"

    # ------------------------------------------------------------------
    # Step 6: align on common countries and compute diff %
    # ------------------------------------------------------------------
    model_countries = set(co2_country_mt.country.values)
    target_countries = set(target_da.country.values)
    common_countries = sorted(model_countries & target_countries)

    if not common_countries:
        logger.warning(
            f"No country overlap between model ({sorted(model_countries)}) "
            f"and CO\u2082 targets ({sorted(target_countries)}). Skipping diff."
        )
        return {
            "co2_emissions_model": co2_country_mt,
            "co2_emissions_target": target_da,
            "co2_emissions_diff_pct": xr.full_like(
                co2_country_mt, np.nan, dtype=float
            ).assign_attrs(
                {
                    "units": "%",
                    "description": "Percentage difference: (model \u2212 target) / target \u00d7 100",
                    "kind": "co2_comparison",
                }
            ),
        }

    model_aligned = co2_country_mt.sel(country=common_countries)
    target_aligned = target_da.sel(country=common_countries)

    with np.errstate(divide="ignore", invalid="ignore"):
        diff_pct = (model_aligned - target_aligned) / target_aligned * 100
        diff_pct = xr.where(np.isinf(diff_pct), np.nan, diff_pct)

    diff_pct.attrs["units"] = "%"
    diff_pct.attrs["description"] = (
        "Percentage difference: (model \u2212 target) / target \u00d7 100"
    )
    diff_pct.attrs["kind"] = "co2_comparison"

    logger.info(
        f"Calculated CO\u2082 target comparison for {len(common_countries)} countries"
    )

    return {
        "co2_emissions_model": co2_country_mt,
        "co2_emissions_target": target_da,
        "co2_emissions_diff_pct": diff_pct,
    }


# =============================================================================
# EVA OUTPUT FUNCTIONS
# =============================================================================


def get_capacity_by_technology(n: pypsa.Network) -> pd.Series:
    """
    Extract installed capacity by technology and bidding zone using PyPSA statistics.
    """
    installed_cap = n.statistics.installed_capacity(
        groupby=["carrier", "bus"],
        nice_names=False,
    ).droplevel("component")

    # Remove load shedding and spillage if present
    if "load shedding" in installed_cap.index:
        installed_cap = installed_cap.drop(
            "load shedding", level="carrier", errors="ignore"
        )
        installed_cap = installed_cap.drop(
            "load spillage", level="carrier", errors="ignore"
        )

    return pd.Series(installed_cap)


def calculate_capacity_changes(
    current_capacity: pd.Series,
    previous_capacity: pd.Series,
    iteration: int,
    target_year: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate commissioned and decommissioned capacity between iterations.

    Note: Capacity changes are per TY (target year) only, not per CY (climate year),
    as they represent investment decisions that don't vary with weather scenarios.
    """
    # Align indices and fill missing values with 0
    all_indices = current_capacity.index.union(previous_capacity.index)
    current_aligned = current_capacity.reindex(all_indices, fill_value=0)
    previous_aligned = previous_capacity.reindex(all_indices, fill_value=0)

    # Calculate capacity difference
    capacity_diff = current_aligned - previous_aligned

    # Commissioned capacity (positive changes only)
    commissioned = capacity_diff[capacity_diff > 0]
    commissioned_da = (
        xr.DataArray(commissioned).unstack("dim_0")
        if not commissioned.empty
        else xr.DataArray([])
    )
    if not commissioned.empty:
        commissioned_da = _assign_coordinates_and_attrs(
            commissioned_da,
            f"Commissioned capacity (iteration {iteration}, TY {target_year})",
            "MW",
        )
        # Add TY as scalar coordinate (not CY, as capacity changes are per TY)
        commissioned_da = commissioned_da.assign_coords({"TY": str(target_year)})
        commissioned_da.attrs["eva_iteration"] = iteration
        commissioned_da.attrs["target_year"] = target_year
        commissioned_da.attrs["kind"] = "eva_output"

    # Decommissioned capacity (negative changes, converted to positive)
    decommissioned = -capacity_diff[capacity_diff < 0]
    decommissioned_da = (
        xr.DataArray(decommissioned).unstack("dim_0")
        if not decommissioned.empty
        else xr.DataArray([])
    )
    if not decommissioned.empty:
        decommissioned_da = _assign_coordinates_and_attrs(
            decommissioned_da,
            f"Decommissioned capacity (iteration {iteration}, TY {target_year})",
            "MW",
        )
        # Add TY as scalar coordinate (not CY, as capacity changes are per TY)
        decommissioned_da = decommissioned_da.assign_coords({"TY": str(target_year)})
        decommissioned_da.attrs["eva_iteration"] = iteration
        decommissioned_da.attrs["target_year"] = target_year
        decommissioned_da.attrs["kind"] = "eva_output"

    return commissioned_da, decommissioned_da


def load_npv_data_from_csv(
    npv_csv_path: str, unit_states_path: str, iteration: int, target_year: str
) -> xr.DataArray:
    """
    Load NPV per MW data for one EVA iteration using CSV files only.

    Parameters
    ----------
    npv_csv_path : str
        Path to the npv_{it}.csv file for this iteration.
    unit_states_path : str
        Path to the unit_states_{it}.csv file containing carrier and bus per unit.
    iteration : int
        EVA iteration number.
    target_year : str
        Target year string (e.g. "2026").

    Returns
    -------
    xr.DataArray
        DataArray with asset dimension and carrier/bus coordinates.
    """
    npv_data = pd.read_csv(npv_csv_path, index_col=[0, 1])
    npv_data.index.names = ["component", "name"]

    # Load unit_states to filter to only capacity-expansion-participating assets
    unit_states = pd.read_csv(unit_states_path)
    expandable_assets = set(zip(unit_states["component"], unit_states["unit"]))

    # Rename npv_data index level 'name' to 'unit' for consistency
    npv_data.index = npv_data.index.set_names(["component", "unit"])

    # Filter to only expandable assets (inner join logic)
    valid_mask = npv_data.index.isin(expandable_assets)
    npv_data = npv_data[valid_mask]

    # Extract all values at once (vectorized)
    asset_names = npv_data.index.get_level_values("unit").tolist()
    carriers = npv_data["carrier"].tolist()
    buses = npv_data["bus"].tolist()
    # NPV value for the specific target year
    # CSV columns: "carrier", "bus", then TY columns like "2028", "2035", etc.
    if target_year not in npv_data.columns:
        raise KeyError(
            f"Target year {target_year} not found in NPV CSV columns {npv_data.columns.tolist()}"
        )
    npv_values = npv_data[target_year].tolist()

    npv_da = xr.DataArray(
        npv_values,
        dims=["asset"],
        coords={
            "asset": asset_names,
            "TY": target_year,
            # Use asset_carrier and asset_BZ to avoid name conflict with dimensional coordinates
            # carrier and BZ used by aggregated variables (e.g., production_capacities)
            "asset_carrier": ("asset", carriers),
            "asset_BZ": ("asset", buses),
        },
    )
    npv_da.attrs["units"] = "EUR/MW"
    npv_da.attrs["description"] = "NPV per MW for each asset (per TY, not per CY)"
    npv_da.attrs["source"] = "Computed from EVA iteration results"
    npv_da.attrs["kind"] = "eva_output"
    return npv_da


def load_capacity_from_csv(capacity_csv_path: str, target_year: str) -> pd.Series:
    """
    Load installed capacity by (carrier, bus) for a given target year from capacity_{it}.csv.

    Parameters
    ----------
    capacity_csv_path : str
        Path to capacity_{it}.csv written by update_capacity.py.
    target_year : str
        Target year column to extract (e.g. "2026").

    Returns
    -------
    pd.Series
        Series indexed by (carrier, bus) with MW values.
    """
    df = pd.read_csv(capacity_csv_path, index_col=[0, 1, 2])
    # Drop the top-level "component" index level to match get_capacity_by_technology output
    df.index = df.index.droplevel(0)
    df.index.names = ["carrier", "bus"]

    exclude_carriers = ["load shedding", "load spillage"]
    df = df[~df.index.get_level_values("carrier").str.lower().isin(exclude_carriers)]

    if target_year in df.columns:
        return df[target_year].rename(target_year)
    # Fall back to last available column
    logger.warning(
        f"Target year {target_year} not found in {capacity_csv_path}, using last column"
    )
    return df.iloc[:, -1]


def calculate_mothballed_capacity(
    unit_states_csv_path: str, iteration: int, target_year: int
) -> xr.DataArray:
    """
    Calculate mothballed capacity by (carrier, bus) for one EVA iteration.

    Sourced from unit_states_{it}.csv (written by update_capacity.py), which
    records the post-processed "state" of every decommissionable unit per TY,
    including "mothballed" for units temporarily taken offline.

    Note: Mothballed capacity is per TY (target year) only, not per CY (climate
    year), as it represents an investment/dispatch-planning decision that
    doesn't vary with weather scenarios - same convention as
    calculate_capacity_changes.
    """
    unit_states = pd.read_csv(unit_states_csv_path)
    mothballed = unit_states[
        (unit_states["ty"] == target_year) & (unit_states["state"] == "mothballed")
    ]

    mothballed_capacity = mothballed.groupby(["carrier", "bus"])["p_nom"].sum()
    mothballed_da = (
        xr.DataArray(mothballed_capacity).unstack("dim_0")
        if not mothballed_capacity.empty
        else xr.DataArray([])
    )
    mothballed_da = _assign_coordinates_and_attrs(
        mothballed_da,
        f"Mothballed capacity (iteration {iteration}, TY {target_year})",
        "MW",
    )
    # Add TY as scalar coordinate (not CY, as mothballed capacity is per TY)
    mothballed_da = mothballed_da.assign_coords({"TY": str(target_year)})
    mothballed_da.attrs["eva_iteration"] = iteration
    mothballed_da.attrs["target_year"] = target_year
    mothballed_da.attrs["kind"] = "eva_output"

    return mothballed_da


def calculate_eva_outputs(
    target_year: str,
    capacity_csv_paths: list[str] | None = None,
    unit_states_csv_paths: list[str] | None = None,
    base_network_path: str | None = None,
    npv_csv_path: str | None = None,
) -> list[xr.DataArray]:
    """
    Calculate all EVA outputs and return as a list of DataArrays.

    Parameters
    ----------
    target_year : str
        Target year for this calculate_model_outputs rule invocation (i.e. the
        current network's own wildcards.ty). capacity_{it}.csv/npv_{it}.csv are
        single global files shared by every {ty}_{cy} invocation, so this must
        be passed in explicitly rather than inferred from the CSV's column
        order - otherwise every TY except whichever happens to be listed first
        in the CSV would have its "current" capacity read from the wrong TY's
        column, mixed against this invocation's own (correct) TY baseline.
    capacity_csv_paths : list[str] | None
        Paths to capacity_{it}.csv files (one per iteration), written by update_capacity.
    unit_states_csv_paths : list[str] | None
        Paths to unit_states_{it}.csv files, one per iteration and aligned
        1:1 with capacity_csv_paths (same convention the calculate_model_outputs
        Snakemake rule uses to build both lists) - required whenever
        capacity_csv_paths is non-empty, since mothballed capacity is read
        from these files. Must include carrier and bus columns. update_capacity
        always writes this file (mothballing enabled or not), so it's never
        legitimately missing when capacity_csv_paths is provided.
    base_network_path : str | None
        Path to base network (iteration 0) for baseline capacity before EVA started.
    npv_csv_path : str | None
        Path to any npv_{it}.csv (used to locate the npv directory).

    Returns
    -------
    list[xr.DataArray]
        List of EVA output DataArrays.
    """
    eva_outputs = []

    if not capacity_csv_paths:
        logger.info("No capacity CSVs provided - skipping EVA outputs calculation")
        return eva_outputs

    logger.info(
        f"Calculating EVA outputs from {len(capacity_csv_paths)} capacity CSVs..."
    )

    # Sanity-check that target_year actually has a column in the capacity CSV
    first_cap_df = pd.read_csv(capacity_csv_paths[0], index_col=[0, 1, 2])
    ty_columns = [c for c in first_cap_df.columns if c.isdigit()]
    if not ty_columns:
        logger.warning("Could not determine target year from capacity CSV columns")
        return eva_outputs
    if target_year not in ty_columns:
        logger.warning(
            f"target_year {target_year} not found in capacity CSV columns "
            f"{ty_columns} - skipping EVA outputs calculation"
        )
        return eva_outputs

    previous_capacity = None

    for i, cap_csv in enumerate(capacity_csv_paths):
        logger.info(
            f"Processing EVA iteration {i + 1}/{len(capacity_csv_paths)}: {cap_csv}"
        )

        current_capacity = load_capacity_from_csv(cap_csv, target_year)
        iteration = i

        if previous_capacity is None:
            if base_network_path and Path(base_network_path).exists():
                logger.info(
                    f"Loading baseline capacity from base network: {base_network_path}"
                )
                base_network = pypsa.Network(base_network_path)
                previous_capacity = get_capacity_by_technology(base_network)
            else:
                logger.warning(
                    "Base network not provided or not found - using zeros as baseline for iteration 0"
                )
                previous_capacity = pd.Series(0, index=current_capacity.index)

        commissioned, decommissioned = calculate_capacity_changes(
            current_capacity, previous_capacity, iteration, int(target_year)
        )

        if commissioned.size > 0:
            eva_outputs.append(commissioned)
        if decommissioned.size > 0:
            eva_outputs.append(decommissioned)

        mothballed = calculate_mothballed_capacity(
            unit_states_csv_paths[i], iteration, int(target_year)
        )
        if mothballed.size > 0:
            eva_outputs.append(mothballed)

        logger.info(
            f"Iteration {iteration}: {commissioned.size} commissioned, "
            f"{decommissioned.size} decommissioned, {mothballed.size} mothballed"
        )

        previous_capacity = current_capacity

    # Load NPV data from all iterations
    if npv_csv_path and unit_states_csv_paths:
        npv_dir = Path(npv_csv_path).parent
        for i, unit_states_path in enumerate(unit_states_csv_paths):
            npv_iter_file = npv_dir / f"npv_{i}.csv"
            if not npv_iter_file.exists():
                logger.debug(f"NPV file not found for iteration {i}: {npv_iter_file}")
                continue
            try:
                npv_data = load_npv_data_from_csv(
                    str(npv_iter_file), unit_states_path, i, target_year
                )
                npv_data.attrs["eva_iteration"] = i
                eva_outputs.append(npv_data)
                logger.info(
                    f"Loaded NPV data for iteration {i}: {npv_data.size} entries"
                )
            except FileNotFoundError:
                logger.warning(f"NPV file not found: {npv_iter_file}")
            except Exception as e:
                logger.warning(f"Could not load NPV data for iteration {i}: {e}")

    logger.info(f"Created {len(eva_outputs)} EVA output variables")
    return eva_outputs


# =============================================================================
# MAIN CALCULATION FUNCTION
# =============================================================================


def calculate_model_outputs(
    n: pypsa.Network,
    n_da: pypsa.Network | None = None,
    n_no_ramping: pypsa.Network | None = None,
    pre_optimization_storage_path: str | None = None,
    capacity_csv_paths: list[str] | None = None,
    unit_states_csv_paths: list[str] | None = None,
    base_network_path: str | None = None,
    npv_csv_path: str | None = None,
    renewable_carriers: list[str] | None = None,
    short_term_method: str = "simulation",
    error_distribution: xr.Dataset | None = None,
    config: dict | None = None,
) -> xr.Dataset:
    """
    Calculate all model outputs (flexibility indicators + market outputs + EVA outputs) and return as a dataset.

    Parameters
    ----------
    n : pypsa.Network
        The intraday PyPSA network object
    n_da : pypsa.Network | None, optional
        The day-ahead PyPSA network object, by default None
    n_no_ramping : pypsa.Network | None, optional
        The intraday network without ramping constraints for ramping shortfall calculation, by default None
    pre_optimization_storage_path : str | None, optional
        Path to pre-optimization storage plan, by default None
    capacity_csv_paths : list[str] | None, optional
        Paths to capacity_{it}.csv files (one per EVA iteration), by default None
    unit_states_csv_paths : list[str] | None, optional
        Paths to unit_states_{it}.csv files (one per EVA iteration), by default None
    base_network_path : str | None, optional
        Path to base network for iteration 0 baseline capacity, by default None
    npv_csv_path : str | None, optional
        Path to NPV data file from last iteration, by default None
    renewable_carriers : list[str] | None, optional
        List of renewable carriers for flexibility calculations, by default None
    short_term_method : str, optional
        Method for short-term requirements: "simulation" or "probabilistic". Default: "simulation"
    error_distribution : xr.Dataset | None, optional
        Pre-computed forecast error distribution for probabilistic method (output of
        :func:`compute_forecast_error_distribution`). Required when ``short_term_method``
        is "probabilistic".
    config : dict | None, optional
        Configuration dictionary containing eva.short_term_requirements settings. Required
        when ``short_term_method`` is "probabilistic".

    Returns
    -------
    xr.Dataset
        Dataset containing all model outputs with "kind" attribute distinguishing
        between "flexibility_indicator", "market_output", and "eva_output"
    """
    logger.info("Calculating flexibility indicators...")

    # Calculate flexibility indicators
    production_cap = calculate_production_capacities(n)
    storage_cap = calculate_storage_capacities(n)
    supply = calculate_supply(n, kind="market_output")  # Use unified supply function
    withdrawal = calculate_withdrawal(
        n, kind="market_output"
    )  # Use unified withdrawal function
    energy_not_served = calculate_energy_not_served(n)
    total_energy_not_served = calculate_energy_not_served(n, agg="sum")
    loss_of_load = calculate_loss_of_load(n)
    curtailment = calculate_curtailment(n, renewable_carriers)
    residual_load = calculate_residual_load(n, renewable_carriers)
    flexibility_needs = calculate_flexibility_needs(n, renewable_carriers)
    maximum_ramping_requirement = calculate_maximum_ramping_requirement(
        n, renewable_carriers
    )

    # Calculate ramping shortfall only if no-ramping network is provided
    if n_no_ramping is not None:
        ramping_shortfall_indicators = calculate_ramping_shortfall_indicator(
            n, n_no_ramping, renewable_carriers
        )
        ramping_shortfall_upward = ramping_shortfall_indicators["upward"]
        ramping_shortfall_downward = ramping_shortfall_indicators["downward"]
    else:
        # Create zero-filled ramping shortfall if calculation is disabled
        # Use same structure as energy_not_served for consistency
        target_year = n.meta["wildcards"]["ty"]
        ramping_shortfall_df = pd.DataFrame(
            0.0, index=n.snapshots, columns=n.buses.index
        )
        ramping_shortfall_df = _shift_time_rows_to_year(
            ramping_shortfall_df, int(target_year)
        )

        ramping_shortfall_upward = xr.DataArray(ramping_shortfall_df)
        ramping_shortfall_upward = _assign_coordinates_and_attrs(
            ramping_shortfall_upward, "Ramping shortfall upward (disabled)"
        )
        ramping_shortfall_upward.attrs["kind"] = "flexibility_indicator"

        ramping_shortfall_downward = xr.DataArray(ramping_shortfall_df)
        ramping_shortfall_downward = _assign_coordinates_and_attrs(
            ramping_shortfall_downward, "Ramping shortfall downward (disabled)"
        )
        ramping_shortfall_downward.attrs["kind"] = "flexibility_indicator"

    technology_contributions = calculate_technology_contributions(n, renewable_carriers)

    # Calculate ramping contributions
    try:
        n_worst_events = n.meta["eva"]["ramping_contribution"]["n_worst_events"]
    except KeyError:
        n_worst_events = [1, 10, 25]

    ramping_contributions = calculate_ramping_contribution(
        n, n_worst_events, renewable_carriers
    )

    # Create aggregated flexibility needs (sum over day/week dimensions)
    daily_flexibility_need_aggregated = flexibility_needs["daily"].sum("day")
    daily_flexibility_need_aggregated.attrs["units"] = "MWh"
    daily_flexibility_need_aggregated.attrs["description"] = (
        "Total daily flexibility need (sum across all days)"
    )
    daily_flexibility_need_aggregated.attrs["kind"] = "flexibility_indicator"

    weekly_flexibility_need_aggregated = flexibility_needs["weekly"].sum("week")
    weekly_flexibility_need_aggregated.attrs["units"] = "MWh"
    weekly_flexibility_need_aggregated.attrs["description"] = (
        "Total weekly flexibility need (sum across all weeks)"
    )
    weekly_flexibility_need_aggregated.attrs["kind"] = "flexibility_indicator"

    # Create aggregated ramping shortfall (sum over hour dimension).
    # The signed hourly values are summed, so hour-to-hour reshuffling cancels
    # instead of accumulating.
    ramping_shortfall_upward_aggregated = ramping_shortfall_upward.sum("hour")
    ramping_shortfall_upward_aggregated.attrs["units"] = "MWh"
    ramping_shortfall_upward_aggregated.attrs["description"] = (
        "Net ramping shortfall upward (signed sum across all hours)"
    )
    ramping_shortfall_upward_aggregated.attrs["kind"] = "flexibility_indicator"

    ramping_shortfall_downward_aggregated = ramping_shortfall_downward.sum("hour")
    ramping_shortfall_downward_aggregated.attrs["units"] = "MWh"
    ramping_shortfall_downward_aggregated.attrs["description"] = (
        "Net ramping shortfall downward (signed sum across all hours)"
    )
    ramping_shortfall_downward_aggregated.attrs["kind"] = "flexibility_indicator"

    # Create aggregated curtailment (sum over hour and carrier dimensions, keep carrier dim)
    curtailment_aggregated = curtailment.sum("hour")
    curtailment_aggregated.attrs["units"] = "MWh"
    curtailment_aggregated.attrs["description"] = (
        "Total curtailment by renewable carrier and bidding zone"
    )
    curtailment_aggregated.attrs["kind"] = "flexibility_indicator"

    data_vars = {
        "production_capacities": production_cap,
        "storage_capacities": storage_cap,
        "supply": supply,
        "withdrawal": withdrawal,
        "energy_not_served": energy_not_served,
        "total_energy_not_served": total_energy_not_served,
        "loss_of_load": loss_of_load,
        "curtailment": curtailment,
        "curtailment_aggregated": curtailment_aggregated,
        "residual_load": residual_load,
        "daily_flexibility_need": flexibility_needs["daily"],
        "daily_flexibility_need_aggregated": daily_flexibility_need_aggregated,
        "weekly_flexibility_need": flexibility_needs["weekly"],
        "weekly_flexibility_need_aggregated": weekly_flexibility_need_aggregated,
        "seasonal_flexibility_need": flexibility_needs["seasonal"],
        "maximum_ramping_requirement": maximum_ramping_requirement,
        "ramping_shortfall_upward": ramping_shortfall_upward,
        "ramping_shortfall_upward_aggregated": ramping_shortfall_upward_aggregated,
        "ramping_shortfall_downward": ramping_shortfall_downward,
        "ramping_shortfall_downward_aggregated": ramping_shortfall_downward_aggregated,
        "technology_contributions_daily": technology_contributions["daily"],
        "technology_contributions_weekly": technology_contributions["weekly"],
        "technology_contributions_seasonal": technology_contributions["seasonal"],
    }

    # Add ramping contributions to dataset
    for key, val in ramping_contributions.items():
        data_vars[f"ramping_contribution_{key}"] = val

    # Only add cross-border contributions if there are links in the network
    if n.links.size:
        cbc = calculate_cross_border_contributions(n, renewable_carriers)
        data_vars["cross_border_contributions_daily"] = cbc["daily"]
        data_vars["cross_border_contributions_weekly"] = cbc["weekly"]
        data_vars["cross_border_contributions_seasonal"] = cbc["seasonal"]

    # Add short-term needs if day-ahead network is provided
    if n_da is not None:
        logger.info("Calculating short-term needs (method=%s)...", short_term_method)

        if short_term_method == "probabilistic":
            # ----------------------------------------------------------------
            # Probabilistic method: use pre-computed error distribution and
            # DA residual flex capacity to determine uncovered needs and
            # expected shortfalls.
            # ----------------------------------------------------------------
            if error_distribution is None:
                raise ValueError(
                    "error_distribution must be provided when short_term_method='probabilistic'"
                )
            prob_results = calculate_probabilistic_short_term_requirements(
                n_id=n,
                n_da=n_da,
                error_distribution=error_distribution,
                config=config,
                renewable_carriers=renewable_carriers,
            )
            data_vars["probabilistic_uncovered_need_up"] = prob_results[
                "uncovered_need_up"
            ]
            data_vars["probabilistic_uncovered_need_down"] = prob_results[
                "uncovered_need_down"
            ]
            data_vars["probabilistic_expected_ens"] = prob_results["expected_ens"]
            data_vars["probabilistic_expected_curtailment"] = prob_results[
                "expected_curtailment"
            ]

            # Also store the forecast error distribution in the dataset
            data_vars["forecast_error_p_low"] = error_distribution[
                "forecast_error_p_low"
            ]
            data_vars["forecast_error_p_high"] = error_distribution[
                "forecast_error_p_high"
            ]

            # Residual flex capacity is informative – include in outputs
            flex_cap = compute_residual_flex_capacity(n_da, renewable_carriers)
            data_vars["residual_flex_capacity_up"] = flex_cap["up"]
            data_vars["residual_flex_capacity_down"] = flex_cap["down"]

        else:
            # ----------------------------------------------------------------
            # Simulation method (default): compare ID vs DA residual load
            # ----------------------------------------------------------------
            # Raw signed hourly forecast error (source for the FNA percentiles)
            data_vars["short_term_need_hourly"] = calculate_short_term_need(
                n, n_da, agg=None
            )
            # Hourly forecast error timeseries separated by direction
            data_vars["short_term_need_upward_hourly"] = calculate_short_term_need(
                n, n_da, agg=None, direction="upward"
            )
            data_vars["short_term_need_downward_hourly"] = calculate_short_term_need(
                n, n_da, agg=None, direction="downward"
            )
            # Total forecast error separated by direction
            data_vars["total_short_term_need_upward"] = calculate_short_term_need(
                n, n_da, agg="sum", direction="upward"
            )
            data_vars["total_short_term_need_downward"] = calculate_short_term_need(
                n, n_da, agg="sum", direction="downward"
            )
            # Percentiles of the signed forecast error per the FNA short-term need
            # methodology (fixed pairs: 99.9/0.1 and 99/1); upward/downward name the tail
            data_vars["short_term_need_upward_p99_9"] = calculate_short_term_need(
                n, n_da, agg="p99.9"
            )
            data_vars["short_term_need_downward_p0_1"] = calculate_short_term_need(
                n, n_da, agg="p0.1"
            )
            data_vars["short_term_need_upward_p99"] = calculate_short_term_need(
                n, n_da, agg="p99"
            )
            data_vars["short_term_need_downward_p1"] = calculate_short_term_need(
                n, n_da, agg="p1"
            )
            data_vars["short_term_shortfall_curtailment"] = (
                calculate_short_term_shortfall_curtailment(n, n_da, renewable_carriers)
            )
            data_vars["short_term_shortfall_curtailment_total"] = (
                calculate_short_term_shortfall_curtailment(
                    n, n_da, renewable_carriers, agg="sum"
                )
            )
            data_vars["short_term_shortfall_curtailment_max"] = (
                calculate_short_term_shortfall_curtailment(
                    n, n_da, renewable_carriers, agg="max"
                )
            )
            data_vars["short_term_shortfall_ens"] = calculate_short_term_shortfall_ens(
                n, n_da
            )
            data_vars["short_term_shortfall_ens_total"] = (
                calculate_short_term_shortfall_ens(n, n_da, agg="sum")
            )
            data_vars["short_term_shortfall_ens_max"] = (
                calculate_short_term_shortfall_ens(n, n_da, agg="max")
            )

            st_tech_contributions = calculate_short_term_technology_contributions(
                n, n_da, renewable_carriers
            )
            # Only keep hourly contributions (daily, weekly, seasonal removed per ACER requirements)
            data_vars["short_term_technology_contributions_hourly"] = (
                st_tech_contributions["hourly"]
            )

            # Only add short-term cross-border contributions if there are links
            if n.links.size:
                st_cb_contributions = calculate_short_term_cross_border_contributions(
                    n, n_da, renewable_carriers
                )
                # Only keep hourly contributions (daily, weekly, seasonal removed per ACER requirements)
                data_vars["short_term_cross_border_contributions_hourly"] = (
                    st_cb_contributions["hourly"]
                )

    # Calculate market outputs
    logger.info("Extracting market outputs...")

    # Market clearing prices
    if n_da is not None:
        prices_da = extract_market_clearing_prices(n_da, "day_ahead")
        data_vars["market_prices_day_ahead"] = prices_da

    prices_id = extract_market_clearing_prices(n, "intraday")
    data_vars["market_prices_intraday"] = prices_id

    # Calculate price statistics for intraday (reuse extracted prices)
    data_vars["average_prices"] = calculate_average_prices(n, "intraday", prices_id)
    data_vars["zero_negative_price_hours"] = calculate_zero_negative_price_hours(
        n, "intraday", prices_id
    )
    data_vars["price_volatility"] = calculate_price_volatility(n, "intraday", prices_id)

    # Total generation by technology per day using unified supply function
    if n_da is not None:
        gen_da = calculate_daily_supply(n_da, "day_ahead")
        data_vars["total_generation_day_ahead"] = gen_da

    gen_id = calculate_daily_supply(n, "intraday")
    data_vars["total_generation_intraday"] = gen_id

    # Total CO2 emissions
    if n_da is not None:
        co2_da = extract_total_co2_emissions(n_da, "day_ahead")
        data_vars["total_co2_emissions_day_ahead"] = co2_da

    co2_id = extract_total_co2_emissions(n, "intraday")
    data_vars["total_co2_emissions_intraday"] = co2_id

    # NECP RES Generation Target Comparison (if enabled)
    necp_enabled = (
        config
        and "costs" in config
        and "necp_targets" in config["costs"]
        and config["costs"]["necp_targets"]["enable"]
    )
    if necp_enabled and config:
        logger.info("Calculating NECP RES generation target comparison...")
        target_year = int(n.meta["wildcards"]["ty"])

        necp_comparison = calculate_necp_res_generation_comparison(
            n, target_year, config, renewable_carriers
        )
        for key, value in necp_comparison.items():
            data_vars[f"necp_{key}"] = value
        logger.info(f"Added NECP comparison for {len(necp_comparison)} variables")

    # CO2 target comparison (if enabled)
    if config and config["costs"]["co2_targets"]["enable"]:
        logger.info("Calculating CO\u2082 target comparison...")
        target_year = int(n.meta["wildcards"]["ty"])
        co2_comparison = calculate_co2_target_comparison(n, target_year, config)
        data_vars.update(co2_comparison)
        logger.info(f"Added CO\u2082 comparison outputs: {list(co2_comparison.keys())}")

    # Net power transfer
    if n_da is not None and not n_da.links.empty:
        transfer_da = extract_net_power_transfer(n_da, "day_ahead")
        data_vars["net_power_transfer_day_ahead"] = transfer_da

    if not n.links.empty:
        transfer_id = extract_net_power_transfer(n, "intraday")
        data_vars["net_power_transfer_intraday"] = transfer_id

    # Reserve capacity per generator (only day-ahead market has reserves)
    if n_da is not None:
        reserves_da = extract_reserve_capacity_per_generator(n_da, "day_ahead")
        for reserve_type, reserve_data in reserves_da.items():
            data_vars[f"reserve_capacity_day_ahead_{reserve_type}"] = reserve_data

    # Reservoir storage levels
    if pre_optimization_storage_path:
        storage_levels = load_reservoir_storage_levels(pre_optimization_storage_path)
        if storage_levels is not None:
            data_vars["reservoir_storage_levels"] = storage_levels

    # Calculate EVA outputs
    if capacity_csv_paths or npv_csv_path:
        logger.info("Calculating EVA outputs...")
        eva_outputs = calculate_eva_outputs(
            n.meta["wildcards"]["ty"],
            capacity_csv_paths,
            unit_states_csv_paths,
            base_network_path,
            npv_csv_path,
        )

        # Group EVA outputs by type and collect them for concatenation
        commissioned_outputs = []
        decommissioned_outputs = []
        mothballed_outputs = []
        npv_outputs = []

        for i, eva_output in enumerate(eva_outputs):
            description = eva_output.attrs.get("description", "").lower()
            iteration = eva_output.attrs.get("eva_iteration", i)

            # Check for decommissioned FIRST to avoid matching "commissioned" substring
            if "decommissioned" in description:
                decommissioned_outputs.append(
                    eva_output.expand_dims(iteration=[iteration])
                )
            elif "commissioned" in description:
                commissioned_outputs.append(
                    eva_output.expand_dims(iteration=[iteration])
                )
            elif "mothballed" in description:
                mothballed_outputs.append(eva_output.expand_dims(iteration=[iteration]))
            elif "npv" in description:
                # Collect NPV outputs from all iterations for concatenation
                npv_outputs.append(eva_output.expand_dims(iteration=[iteration]))
            else:
                data_vars[f"eva_output_{i}"] = eva_output

        # Concatenate along iteration dimension
        # Use fill_value=0 to replace NaN with 0 (missing capacity change = no change)
        if commissioned_outputs:
            data_vars["commissioned_capacity"] = xr.concat(
                commissioned_outputs, dim="iteration", fill_value=0
            )
            logger.info(
                f"Concatenated commissioned capacity from {len(commissioned_outputs)} iterations"
            )
        if decommissioned_outputs:
            data_vars["decommissioned_capacity"] = xr.concat(
                decommissioned_outputs, dim="iteration", fill_value=0
            )
            logger.info(
                f"Concatenated decommissioned capacity from {len(decommissioned_outputs)} iterations"
            )
        elif commissioned_outputs:
            # If we have commissioned but no decommissioned, create empty decommissioned with same structure
            # This ensures consistency in output structure
            logger.info(
                "No decommissioned capacity found - this is expected when all capacity changes are increases"
            )
        if mothballed_outputs:
            data_vars["mothballed_capacity"] = xr.concat(
                mothballed_outputs, dim="iteration", fill_value=0
            )

        # Concatenate NPV outputs along iteration dimension
        if npv_outputs:
            data_vars["npv_per_mw"] = xr.concat(npv_outputs, dim="iteration")
            logger.info(
                f"Concatenated NPV data from {len(npv_outputs)} iterations: "
                f"{data_vars['npv_per_mw'].shape}"
            )

    # Network needs timeseries (from dummy storage optimization)
    if "network_needs" in n.buses_t:
        network_needs = n.buses_t["network_needs"]
        # Convert DataFrame to DataArray - use the actual index and column names
        # The columns are likely named with 'Bus' dimension, not 'bus'
        da = xr.DataArray(
            network_needs.values,
            dims=["snapshot", "Bus"],  # Use capital B to match PyPSA convention
            coords={"snapshot": network_needs.index, "Bus": network_needs.columns},
        )
        da = _assign_coordinates_and_attrs(da, "Network needs timeseries", "MW")
        da.attrs["kind"] = "flexibility_indicator"
        da.attrs["description"] = (
            "Mock network needs generated for dummy storage optimization"
        )
        data_vars["network_needs"] = da

    # Original curtailment timeseries (from dummy storage optimization)
    if "original_curtailment_timeseries" in n.buses_t:
        original_curtailment = n.buses_t["original_curtailment_timeseries"]
        # Convert DataFrame to DataArray - extract the "total" column if it exists
        if "total" in original_curtailment.columns:
            da = xr.DataArray(
                original_curtailment["total"].values,
                dims=["snapshot"],
                coords={"snapshot": original_curtailment.index},
            )
        else:
            # If structure is different, adapt accordingly
            da = xr.DataArray(
                original_curtailment.values,
                dims=["snapshot", "Bus"],
                coords={
                    "snapshot": original_curtailment.index,
                    "Bus": original_curtailment.columns,
                },
            )
        da = _assign_coordinates_and_attrs(da, "Original curtailment timeseries", "MW")
        da.attrs["kind"] = "flexibility_indicator"
        da.attrs["description"] = "Original curtailment before network needs adaptation"
        data_vars["original_curtailment_timeseries"] = da

    # Curtailment targets timeseries (from dummy storage optimization)
    if "curtailment_targets" in n.buses_t:
        curtailment_targets = n.buses_t["curtailment_targets"]
        # Convert DataFrame to DataArray - use the actual index and column names
        da = xr.DataArray(
            curtailment_targets.values,
            dims=["snapshot", "Bus"],  # Use capital B to match PyPSA convention
            coords={
                "snapshot": curtailment_targets.index,
                "Bus": curtailment_targets.columns,
            },
        )
        da = _assign_coordinates_and_attrs(
            da, "Curtailment targets timeseries", "per-unit"
        )
        da.attrs["kind"] = "flexibility_indicator"
        da.attrs["description"] = (
            "Effective curtailment targets (max of base target and network needs) used in dummy storage optimization"
        )
        data_vars["curtailment_targets"] = da

    # Adjusted curtailment timeseries (from dummy storage optimization)
    if "adjusted_curtailment_timeseries" in n.buses_t:
        adjusted_curtailment = n.buses_t["adjusted_curtailment_timeseries"]
        # Convert DataFrame to DataArray - extract the "total" column if it exists
        if "total" in adjusted_curtailment.columns:
            da = xr.DataArray(
                adjusted_curtailment["total"].values,
                dims=["snapshot"],
                coords={"snapshot": adjusted_curtailment.index},
            )
        else:
            # If structure is different, adapt accordingly
            da = xr.DataArray(
                adjusted_curtailment.values,
                dims=["snapshot", "Bus"],
                coords={
                    "snapshot": adjusted_curtailment.index,
                    "Bus": adjusted_curtailment.columns,
                },
            )
        da = _assign_coordinates_and_attrs(da, "Adjusted curtailment timeseries", "MW")
        da.attrs["kind"] = "flexibility_indicator"
        da.attrs["description"] = "Adjusted curtailment after network needs adaptation"
        data_vars["adjusted_curtailment_timeseries"] = da

    # Demand timeseries
    if hasattr(n, "loads_t") and "p_set" in n.loads_t and not n.loads_t.p_set.empty:
        demand_data = n.loads_t.p_set
        # Convert DataFrame to DataArray - use Load dimension
        da = xr.DataArray(
            demand_data.values,
            dims=["snapshot", "Load"],
            coords={"snapshot": demand_data.index, "Load": demand_data.columns},
        )
        da = _assign_coordinates_and_attrs(da, "Electrical demand timeseries", "MW")
        da.attrs["kind"] = "market_output"
        da.attrs["description"] = "Base electrical load demand"
        data_vars["demand"] = da

    # Storage units charging (negative dispatch = charging)
    if (
        hasattr(n, "storage_units_t")
        and "p" in n.storage_units_t
        and not n.storage_units_t.p.empty
    ):
        storage_dispatch = n.storage_units_t.p
        # Charging is negative dispatch
        storage_charging = -storage_dispatch.where(storage_dispatch < 0, 0)
        da = xr.DataArray(
            storage_charging.values,
            dims=["snapshot", "StorageUnit"],
            coords={
                "snapshot": storage_charging.index,
                "StorageUnit": storage_charging.columns,
            },
        )
        da = _assign_coordinates_and_attrs(
            da, "Storage units charging timeseries", "MW"
        )
        da.attrs["kind"] = "market_output"
        da.attrs["description"] = "Energy consumed by storage units during charging"
        data_vars["storage_units_charging"] = da

    # Load spillage (from generators with 'load spillage' carrier)
    if (
        hasattr(n, "generators_t")
        and "p" in n.generators_t
        and not n.generators_t.p.empty
    ):
        # Find load spillage generators
        load_spillage_gens = n.generators[n.generators.carrier == "load spillage"].index
        if len(load_spillage_gens) > 0:
            load_spillage_dispatch = n.generators_t.p[load_spillage_gens]
            # Load spillage should be negative (consumption), take absolute value
            load_spillage_consumption = -load_spillage_dispatch.where(
                load_spillage_dispatch < 0, 0
            )

            da = xr.DataArray(
                load_spillage_consumption.values,
                dims=["snapshot", "Generator"],
                coords={
                    "snapshot": load_spillage_consumption.index,
                    "Generator": load_spillage_consumption.columns,
                },
            )
            da = _assign_coordinates_and_attrs(
                da, "Load spillage consumption timeseries", "MW"
            )
            da.attrs["kind"] = "market_output"
            da.attrs["description"] = (
                "Energy consumed by load spillage generators (demand reduction)"
            )
            data_vars["load_spillage"] = da

    # Clean up snapshot coordinates
    for k, v in data_vars.items():
        # Drop coordinates that can't be serialized to NetCDF
        coords_to_drop = []
        if "snapshot" in v.coords:
            coords_to_drop.append("snapshot")
        if "date" in v.coords:
            coords_to_drop.append("date")
        if coords_to_drop:
            data_vars[k] = v.drop_vars(coords_to_drop)

    # Create dataset
    ds = xr.Dataset(data_vars=data_vars)

    # TY_ONLY_EVA_OUTPUT_VARS carry a scalar TY coordinate already (see e.g.
    # calculate_capacity_changes) but no CY - expand them on TY alone so the
    # dataset doesn't gain a spurious CY dimension that would duplicate
    # identical values across every CY once per-(ty,cy) outputs are merged
    # into model_outputs.nc.
    ty_only_vars = [v for v in TY_ONLY_EVA_OUTPUT_VARS if v in ds.data_vars]
    ty = n.meta["wildcards"]["ty"]
    cy = n.meta["wildcards"]["cy"]

    ds_cy = (
        ds.drop_vars(ty_only_vars)
        .assign_coords({"TY": ty, "CY": cy})
        .expand_dims(["TY", "CY"])
    )
    if ty_only_vars:
        ds_ty = ds[ty_only_vars].assign_coords({"TY": ty}).expand_dims(["TY"])
        ds = xr.merge([ds_cy, ds_ty], join="outer", compat="no_conflicts")
    else:
        ds = ds_cy

    # Add global attributes
    ds.attrs["title"] = "Comprehensive Model Outputs"
    ds.attrs["description"] = (
        "Combined flexibility indicators and market outputs from day-ahead and intraday optimization"
    )
    ds.attrs["target_year"] = n.meta["wildcards"]["ty"]
    ds.attrs["climate_year"] = n.meta["wildcards"]["cy"]
    ds.attrs["created_by"] = "calculate_model_outputs.py"
    ds.attrs["data_source"] = "Solved PyPSA networks from EVA workflow"

    ds = _attach_plot_labels(ds)

    # Add metadata
    ds = ds.merge(get_meta_data(n), join="outer", compat="no_conflicts")

    logger.info(
        f"Created comprehensive model outputs dataset with {len(data_vars)} variables"
    )

    # NetCDF4 does not support boolean attributes; convert them to strings so
    # the returned dataset is always serialisation-safe.
    convert_bool_attrs(ds)

    return ds


def convert_bool_attrs(node: xr.Dataset | xr.DataTree | xr.DataArray) -> None:
    """
    Convert all boolean attributes to strings in-place.

    Works on both :class:`xr.Dataset` (direct fix) and :class:`xr.DataTree`
    (fix current node's dataset then recurse into children).  NetCDF4 does not
    support the boolean dtype for attributes, so this must be called before
    writing to disk.

    Parameters
    ----------
    node : xr.Dataset | xr.DataTree | xr.DataArray
        Dataset or DataTree to fix in-place.

    Returns
    -------
    None
        This function modifies attributes in-place and does not return a value.
    """
    ds = node if isinstance(node, xr.Dataset) else node.ds
    for var_name in ds.data_vars:
        for attr_name, attr_value in list(ds[var_name].attrs.items()):
            if isinstance(attr_value, bool):
                ds[var_name].attrs[attr_name] = str(attr_value)
    for attr_name, attr_value in list(ds.attrs.items()):
        if isinstance(attr_value, bool):
            ds.attrs[attr_name] = str(attr_value)
    if isinstance(node, xr.DataTree):
        for child in node.children:
            convert_bool_attrs(node[child])


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="calculate_model_outputs",
        configfiles=[
            pathlib.Path("config", "test", "config.eva-quick-test.yaml"),
        ],
        run="eva-quick-test",
        cy=2013,
        ty=2030,
    )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    RENEWABLE_CARRIERS = snakemake.params.renewable_carriers

    # Load networks
    n = pypsa.Network(
        snakemake.input.intraday_network  # ty: ignore [possibly-missing-attribute]
    )
    n_da = (
        pypsa.Network(snakemake.input.day_ahead_network)
        if hasattr(snakemake.input, "day_ahead_network")
        else None
    )

    # Load no-ramping network if available (for ramping shortfall calculation)
    intraday_without_ramping_paths = getattr(
        snakemake.input, "intraday_network_without_ramping", []
    )
    n_no_ramping = None

    if (
        intraday_without_ramping_paths
        and snakemake.config["eva"]["ramping_shortfall"]["enable"]
    ):
        for path in intraday_without_ramping_paths:
            logger.info(f"Loading intraday network without ramping from {path}")
            n_temp = pypsa.Network(path)
            cy = n_temp.meta["wildcards"]["cy"]
            ty = n_temp.meta["wildcards"]["ty"]
            if (
                cy == snakemake.wildcards.cy  # ty: ignore[possibly-missing-attribute]
                and ty
                == snakemake.wildcards.ty  # ty: ignore[possibly-missing-attribute]
            ):
                n_no_ramping = n_temp
                break

    # Get pre-optimization storage path if available
    pre_opt_storage = getattr(snakemake.input, "pre_optimization_storage", None)

    # Get EVA capacity CSVs, unit states CSVs, and NPV data if available
    capacity_csv_paths = getattr(snakemake.input, "capacity_csv_paths", []) or []
    unit_states_csv_paths = getattr(snakemake.input, "unit_states_csv_paths", []) or []
    base_network = getattr(snakemake.input, "base_network", None)
    npv_data_path = getattr(snakemake.input, "npv_data", None)

    # Resolve short-term requirements method and probabilistic config
    st_cfg = snakemake.config["eva"]["short_term_requirements"]
    short_term_method = st_cfg["method"]

    error_distribution = None

    if short_term_method == "probabilistic" and n_da is not None:
        logger.info(
            "Building forecast error distribution for probabilistic ST method..."
        )

        # Collect all (ID, DA) network pairs for the same TY across all CYs
        # The snakemake input may contain paths for all CY combinations
        all_intraday_paths = getattr(snakemake.input, "all_intraday_networks", [])
        all_da_paths = getattr(snakemake.input, "all_day_ahead_networks", [])

        if all_intraday_paths and all_da_paths:
            network_pairs: list[tuple[pypsa.Network, pypsa.Network]] = []
            for id_path, da_path in zip(all_intraday_paths, all_da_paths):
                n_id_cy = pypsa.Network(id_path)
                n_da_cy = pypsa.Network(da_path)
                network_pairs.append((n_id_cy, n_da_cy))
        else:
            # Fallback: use only the current CY pair (single-CY run)
            logger.warning(
                "all_intraday_networks / all_day_ahead_networks not provided; "
                "computing error distribution from single CY only."
            )
            network_pairs = [(n, n_da)]

        error_distribution = compute_forecast_error_distribution(
            network_pairs=network_pairs,
            config=snakemake.config,
            renewable_carriers=RENEWABLE_CARRIERS,
        )

    # Calculate base model outputs
    base = calculate_model_outputs(
        n,
        n_da,
        n_no_ramping,
        pre_opt_storage,
        capacity_csv_paths,
        unit_states_csv_paths,
        base_network,
        npv_data_path,
        RENEWABLE_CARRIERS,
        short_term_method=short_term_method,
        error_distribution=error_distribution,
        config=snakemake.config,
    )

    # Handle dummy storage optimizations if available
    paths = getattr(snakemake.input, "dummy_storage_optimizations", [])

    if paths:
        indicators = []
        for path in paths:
            logger.info(f"Loading network from {path}")
            n_dummy = pypsa.Network(path)
            mh = n_dummy.meta["wildcards"]["mh"]
            ct = n_dummy.meta["wildcards"]["ct"]
            cy = n_dummy.meta["wildcards"]["cy"]
            ty = n_dummy.meta["wildcards"]["ty"]
            if (
                cy == snakemake.wildcards.cy  # ty: ignore[possibly-missing-attribute]
                and ty
                == snakemake.wildcards.ty  # ty: ignore[possibly-missing-attribute]
            ):
                indic = calculate_model_outputs(
                    n_dummy, renewable_carriers=RENEWABLE_CARRIERS
                )
                indic = indic.expand_dims({"mh": [mh], "ct": [ct]})
                indicators.append(indic)

        logger.info("Concatenating model outputs")
        # Each case holds one (mh, ct) pair. xr.merge would first align every
        # case to the full grid, so memory grows with the square of the number
        # of cases; combining by coordinates gives the same result in linear
        # memory.
        indic = xr.combine_by_coords(
            indicators, join="outer", compat="no_conflicts", combine_attrs="override"
        )
    else:
        logger.info(
            "No dummy storage optimizations to process (dummy storage is disabled)"
        )
        # Create empty dataset with appropriate dimensions
        indic = xr.Dataset()

    # Create final output structure
    dt = xr.DataTree.from_dict({"base": base, "dummy_storage_optimization": indic})

    # Convert boolean attributes to strings to avoid h5netcdf compatibility issues
    convert_bool_attrs(dt)
    dt.to_netcdf(snakemake.output.fn)  # ty: ignore[possibly-missing-attribute]

    # Extract and save shadow prices if enabled
    shadow_prices_enabled = getattr(snakemake.params, "shadow_prices_enabled", False)
    shadow_prices_constraints = getattr(
        snakemake.params, "shadow_prices_constraints", []
    )
    shadow_prices_output = getattr(snakemake.output, "shadow_prices", None)

    if shadow_prices_enabled and shadow_prices_output:
        logger.info(
            f"Extracting shadow prices for constraints: {shadow_prices_constraints}"
        )
        # Extract shadow prices from both day-ahead and intraday networks
        all_shadow_prices: dict[str, xr.DataArray] = {}

        # Extract from day-ahead network (contains reserve requirement duals)
        if n_da is not None:
            da_ds = extract_shadow_prices(n_da, shadow_prices_constraints)
            for var_name, var in da_ds.data_vars.items():
                all_shadow_prices[f"da_{var_name}"] = var
                logger.info(f"Extracted da_{var_name} from day-ahead network")

        # Extract from intraday network
        id_ds = extract_shadow_prices(n, shadow_prices_constraints)
        for var_name, var in id_ds.data_vars.items():
            all_shadow_prices[f"id_{var_name}"] = var
            logger.info(f"Extracted id_{var_name} from intraday network")

        if all_shadow_prices:
            # Build combined dataset
            shadow_prices_ds = xr.Dataset(data_vars=all_shadow_prices)

            # Add scenario coordinates
            if hasattr(n, "meta") and "wildcards" in n.meta:
                ty = n.meta["wildcards"].get("ty", "unknown")
                cy = n.meta["wildcards"].get("cy", "unknown")
            else:
                ty = "unknown"
                cy = "unknown"
            shadow_prices_ds = shadow_prices_ds.assign_coords(
                {"TY": ty, "CY": cy}
            ).expand_dims(["TY", "CY"])
            shadow_prices_ds.attrs["title"] = "Shadow Prices"

            # Convert boolean attributes for compatibility
            for var_name, var in shadow_prices_ds.data_vars.items():
                for attr_name, attr_value in var.attrs.items():
                    if isinstance(attr_value, bool):
                        var.attrs[attr_name] = str(attr_value)

            shadow_prices_ds.to_netcdf(shadow_prices_output)
            logger.info(f"Saved shadow prices to {shadow_prices_output}")
        else:
            # Create minimal empty file so Snakemake doesn't fail
            empty_ds = xr.Dataset(
                attrs={
                    "title": "Shadow Prices (Empty)",
                    "description": "No shadow prices extracted - constraints not found in network",
                }
            )
            empty_ds.to_netcdf(shadow_prices_output)
            logger.warning(
                f"No shadow prices extracted, created empty file at {shadow_prices_output}"
            )
