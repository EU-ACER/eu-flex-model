# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System

import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.data_patchings import (
    patch_expansion_limit_inconsistencies,
    patch_p_min_pu_inconsistencies,
    patch_p_set_p_min_max_inconsistencies,
    patch_reserve_amounts_exceed_pu_bounds,
    patch_reserve_providers,
)
from euflex.eu_flex_model.scripts.fbmc_ntc_helpers import (
    adapt_network_for_market_coupling,
)
from euflex.eu_flex_model.scripts.helpers.date_helpers import (
    decommissionable_by_carrier,
    existing_capacity_bounds,
)
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.pypsa_helpers import (
    as_dense,
    reduce_network_size,
    sanitize_carriers,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
)
from euflex.eu_flex_model.scripts.validate_dataframes import (
    GeneratorSchema,
    StorageUnitSchema,
)

logger = logging.getLogger(__name__)

COMPRESSION = {"zlib": True, "complevel": 4}


def spread_overlapping_bus_coordinates(n: pypsa.Network, offset: float = 2.0) -> int:
    """
    Spread out buses that have identical coordinates by applying small offsets.

    This function identifies groups of buses that share the same x,y coordinates
    (typically buses for the same country) and spreads them out in a circular
    pattern around the original location to prevent visual overlap in network plots.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to modify
    offset : float, optional
        Maximum radius offset for spreading buses (default: 0.5 degrees)

    Returns
    -------
    int
        Number of buses that were repositioned
    """
    buses = n.buses.copy()
    if "x" not in buses.columns or "y" not in buses.columns:
        logger.warning(
            "Bus coordinates (x, y) not found. Skipping coordinate spreading."
        )
        return 0

    # Group buses by their original coordinates
    coord_groups = buses.groupby(["x", "y"])
    repositioned_count = 0

    for (orig_x, orig_y), group in coord_groups:  # ty: ignore[not-iterable]
        # Skip groups with only one bus (no overlap)
        bus_indices = group.index
        if len(bus_indices) <= 1:
            continue

        logger.debug(
            f"Found {len(bus_indices)} overlapping buses at ({orig_x:.4f}, {orig_y:.4f}): {list(bus_indices)}"
        )

        # Spread buses in a circular pattern around the original coordinates
        for i, bus_idx in enumerate(bus_indices):
            if i == 0:
                # Keep the first bus at original location
                continue

            # Calculate angle for circular distribution
            angle = 2 * np.pi * i / len(bus_indices)

            # Calculate new coordinates with small offset
            new_x = orig_x + offset * np.cos(angle)
            new_y = orig_y + offset * np.sin(angle)

            # Update bus coordinates
            n.buses.loc[bus_idx, "x"] = new_x
            n.buses.loc[bus_idx, "y"] = new_y
            repositioned_count += 1

            logger.debug(f"  Moved bus {bus_idx} to ({new_x:.4f}, {new_y:.4f})")

    if repositioned_count > 0:
        logger.info(
            f"Repositioned {repositioned_count} buses to avoid coordinate overlap"
        )
    else:
        logger.debug("No overlapping bus coordinates found")

    return repositioned_count


def clean_up_gen_names(n: pypsa.Network) -> None:
    """
    Clean up generator names by removing extra '0' in naming convention.

    Converts names like "BE00 0 solar" to "BE00 solar" and updates both
    n.generators DataFrame and all n.generators_t timeseries DataFrames.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object to modify.

    Returns
    -------
    None
    """

    # Pattern to match generators with extra 0: "XX00 0 carrier"
    pattern = re.compile(r"^([A-Z]{2}\d{2}) 0 (.+)$")

    # Build mapping of old names to new names
    rename_mapping = {}
    conflicts = []

    for gen_name in n.generators.index:
        match = pattern.match(gen_name)
        if match:
            bus_code, carrier = match.groups()
            new_name = f"{bus_code} {carrier}"

            # Check for naming conflicts
            if new_name in n.generators.index and new_name != gen_name:
                conflicts.append(
                    f"'{gen_name}' -> '{new_name}' (conflicts with existing)"
                )
            else:
                rename_mapping[gen_name] = new_name

    if conflicts:
        logger.warning("Naming conflicts found, skipping these generators:")
        for conflict in conflicts:
            logger.warning(f"  {conflict}")

    if not rename_mapping:
        logger.info("No generator names need cleaning")
        return

    logger.info(f"Cleaning up {len(rename_mapping)} generator names:")
    for old_name, new_name in list(rename_mapping.items())[:5]:  # Show first 5
        logger.info(f"  '{old_name}' -> '{new_name}'")
    if len(rename_mapping) > 5:
        logger.info(f"  ... and {len(rename_mapping) - 5} more")

    # Update n.generators DataFrame index
    n.generators = n.generators.rename(index=rename_mapping)

    # Update all generator timeseries DataFrames

    timeseries_attrs = list(n.generators_t)

    for attr in timeseries_attrs:
        if hasattr(n.generators_t, attr):
            df = getattr(n.generators_t, attr)
            if not df.empty:
                # Only rename columns that exist in the DataFrame
                existing_renames = {
                    old: new for old, new in rename_mapping.items() if old in df.columns
                }
                if existing_renames:
                    df = df.rename(columns=existing_renames)
                    setattr(n.generators_t, attr, df)
                    logger.debug(
                        f"Updated {len(existing_renames)} columns in generators_t.{attr}"
                    )

    logger.info(f"Successfully cleaned up {len(rename_mapping)} generator names")


def add_load_shedding(
    n: pypsa.Network, ty: int, ft: str, config: dict
) -> pypsa.Network:
    """
    Adds load shedding components to a PyPSA Network.

    Load shedding generators are added for each bus in the network with a
    specified high marginal cost. The carrier is also defined with a
    predefined color and name.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA Network object where load shedding components are to be added.
    ty: int
        Target year
    ft: str
        Forecast
    config: dict
        Define the load shedding price

    Returns
    -------
    pypsa.Network
    """
    # Includes load shedding to avoid infeasibility
    year_index = config["scenario"]["ty"].index(ty)
    if ft == "forecast":
        price_cap = config["eva"]["price_cap"]["day_ahead_values"][year_index]
    else:
        price_cap = config["eva"]["price_cap"]["intraday_values"][year_index]
    buses = n.buses.index
    n.add(
        "Generator",
        name=list(buses),
        suffix=" load shedding",
        bus=buses,
        carrier="load shedding",
        marginal_cost=price_cap,
        p_nom=10e6,
        reserve_FRR=False,
        reserve_FCR=False,
        lifetime=20,
        build_year=ty,
        decommissioning_year=ty + 20,
    )
    n.add("Carrier", name="load shedding", color="#dd2e23", nice_name="Load shedding")

    return n


def add_load_spillage(
    n: pypsa.Network, ty: int, ft: str, config: dict
) -> pypsa.Network:
    """
    Adds load spillage components to a PyPSA Network.

    Load spillage generators are added for each bus in the network with a
    specified low marginal cost. The carrier is also defined with a
    predefined color and name.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA Network object where load spillage components are to be added.
    ty: int
        Target year
    ft: str
        Forecast
    config: dict
        Define the load spillage price


    Returns
    -------
    pypsa.Network
    """
    # Includes load spillage to avoid infeasibility
    year_index = config["scenario"]["ty"].index(ty)
    if ft == "forecast":
        price_cap = config["eva"]["price_cap"]["day_ahead_values"][year_index]
    else:
        price_cap = config["eva"]["price_cap"]["intraday_values"][year_index]
    buses = n.buses.index
    n.add(
        "Generator",
        name=list(buses),
        suffix=" load spillage",
        bus=buses,
        carrier="load spillage",
        marginal_cost=-price_cap,
        p_min_pu=-1,
        p_max_pu=0,
        p_nom=10e6,
        reserve_FRR=False,
        reserve_FCR=False,
        lifetime=20,
        build_year=ty,
        decommissioning_year=ty + 20,
    )
    n.add("Carrier", name="load spillage", color="#df8e23", nice_name="Load spillage")

    return n


def add_reserve_shedding(
    n: pypsa.Network, ty: int, ft: str, config: dict
) -> pypsa.Network:
    """
    Adds reserve shedding components to a PyPSA Network.

    Reserve shedding generators are added for each bus in the network with a
    high marginal cost for energy and very high reservation cost for reserves.
    This acts as an emergency valve when reserve requirements cannot be met.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA Network object where reserve shedding components are to be added.
    ty: int
        Target year
    ft: str
        Forecast
    config: dict
        Define the reserve shedding price

    Returns
    -------
    pypsa.Network
    """
    # Includes reserve shedding to avoid reserve infeasibility
    year_index = config["scenario"]["ty"].index(ty)
    if ft == "forecast":
        price_cap = config["eva"]["price_cap"]["day_ahead_values"][year_index]
    else:
        price_cap = config["eva"]["price_cap"]["intraday_values"][year_index]
    buses = n.buses.index
    n.add(
        "Generator",
        name=list(buses),
        suffix=" reserve shedding",
        bus=buses,
        carrier="reserve shedding",
        marginal_cost=price_cap,
        p_nom=10e6,
        reserve_FRR=True,
        reserve_FCR=True,
        lifetime=20,
        build_year=ty,
        decommissioning_year=ty + 20,
    )
    n.add(
        "Carrier",
        name="reserve shedding",
        color="#ff00ff",
        nice_name="Reserve shedding",
    )

    return n


def group_electrical_links_by_connection(n: pypsa.Network) -> None:
    """
    Groups electrical links in the network by their connections.

    This function takes all electrical links in a PyPSA network and groups them
    based on their connections (i.e., the buses they connect). It also aggregates
    time-varying parameters such as p_min_pu and p_max_pu using p_nom-weighted means.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA Network object where links will be grouped.

    Returns
    -------
    None
    """

    electrical_links = n.links[n.links.carrier.isin(["AC", "DC"])]

    if electrical_links.empty:
        logging.info("No electrical links to group.")
        return

    links = n.links.loc[electrical_links.index]
    connection = links.apply(lambda x: "-".join(sorted([x["bus0"], x["bus1"]])), axis=1)

    # Base aggregation for static attributes
    agg_dict = {
        "p_nom": "sum",
        "p_nom_min": "sum",
        "p_nom_max": "sum",
        "length": "first",
        "capital_cost": "first",
        "bus0": "first",
        "bus1": "first",
        "carrier": "first",
    }

    if "Forced Outage Rate [%]" in links.columns:
        agg_dict["Forced Outage Rate [%]"] = "mean"
    if "Number of poles [#]" in links.columns:
        agg_dict["Number of poles [#]"] = "sum"
    if "FOR direction" in links.columns:
        agg_dict["FOR direction"] = "first"

    grouped_links = links.groupby(connection).agg(agg_dict)

    # Sort bus0 and bus1
    grouped_links[["bus0", "bus1"]] = np.sort(grouped_links[["bus0", "bus1"]], axis=1)

    # Aggregate timeseries using capacity-weighted averaging (PyPSA pattern)
    aggregated_timeseries = {}
    for attr_name in ["p_min_pu", "p_max_pu", "p_set"]:
        # Get both static and dynamic values using switchable_as_dense
        full_ts = as_dense(n, "Link", attr_name, links.index)

        if full_ts.empty:
            continue

        # Calculate normalized capacity weights per connection group
        capacity_weights = (
            links["p_nom"].groupby(connection).transform(lambda x: x / x.sum())
        )

        # Apply capacity weighting and aggregate by connection
        weighted_ts = full_ts * capacity_weights
        aggregated_ts = weighted_ts.T.groupby(connection).sum().T

        aggregated_timeseries[attr_name] = aggregated_ts
        logging.info(
            f"Aggregated {attr_name} for {len(aggregated_ts.columns)} grouped links "
            f"using capacity-weighted averaging"
        )

    # Remove the original links
    n.remove("Link", links.index)

    # Add grouped links with static parameters first (PyPSA pattern)
    n.add("Link", grouped_links.index, **grouped_links.to_dict("series"))

    # Then add aggregated timeseries for links that have them
    for attr_name, attr_df in aggregated_timeseries.items():
        if not attr_df.empty:
            n.links_t[attr_name] = attr_df
            logging.info(
                f"Added {attr_name} timeseries for {len(attr_df.columns)} grouped links"
            )


def generate_temporal_noise(
    timesteps: int, autocorr: float, seed: int | None = None
) -> np.ndarray:
    """
    Generate temporally correlated noise using an autoregressive process.

    This function creates a sequence of noise values with a specified temporal autocorrelation,
    following an AR(1) (first-order autoregressive) process.

    Parameters
    ----------
    timesteps : int
        The number of time steps to generate noise for.
    autocorr : float
        The autocorrelation coefficient, which controls the temporal correlation
        of the noise. Must be between -1 and 1.
        - A value close to 0 produces nearly independent noise
        - A value close to 1 creates highly correlated noise
        - A value close to -1 creates anti-correlated noise
    seed : int, optional
        Random seed for reproducibility. If None, uses random state.

    Returns
    -------
    np.ndarray
        An array of generated noise values with length equal to `timesteps`.

    Notes
    -----
    The noise generation follows the formula:
    noise[t] = autocorr * noise[t-1] + sqrt(1 - autocorr^2) * standard_normal_random_variable
    """
    rng = np.random.default_rng(seed)
    noise = np.zeros(timesteps)
    noise[0] = rng.normal(0, 1)
    for t in range(1, timesteps):
        noise[t] = autocorr * noise[t - 1] + np.sqrt(1 - autocorr**2) * rng.normal(0, 1)
    return noise


def apply_uncertainty_to_timeseries(
    df: pd.DataFrame,
    index: list[str],
    unc: dict[str, float | int],
    clip_min: float = 0,
    clip_max: float | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Apply uncertainty and noise to specified columns in a time series DataFrame.

    This function introduces temporal and spatial noise to the specified columns,
    simulating uncertainty in time series data through a combination of common
    and spatial noise components.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series data.
    index : List[str]
        List of column names to apply uncertainty to.
    unc : Dict[str, Union[float, int]]
        Dictionary containing uncertainty parameters:
        - 'autocorr': Autocorrelation coefficient for temporal noise
        - 'bias': Systematic bias to be applied
        - 'noise_std': Standard deviation of the noise
    clip_min : float, optional
        Minimum value to clip the modified time series. Default is 0.
    clip_max : float, optional
        Maximum value to clip the modified time series. Default is None (no upper clipping).
    seed : int, optional
        Random seed for reproducibility. If None, uses random state.

    Returns
    -------
    None

    Notes
    -----
    The uncertainty is applied through a two-component noise model:
    1. Common temporal noise generated using an autoregressive process
    2. Spatial noise with a fixed standard deviation

    The final modification follows the formula:
    modified_value = original_value * (1 + bias + noise_std * total_noise)
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    timesteps = df.shape[0]
    common_noise = generate_temporal_noise(timesteps, unc["autocorr"], seed=seed)

    for col in index:
        if col not in df.columns:
            continue

        spatial_noise = 0.3 * rng.normal(0, 1, timesteps)
        total_noise = common_noise + spatial_noise

        forecast = df[col] * (1 + unc["bias"] + unc["noise_std"] * total_noise)
        df[col] = forecast.clip(lower=clip_min, upper=clip_max)


def apply_renewable_uncertainty(
    n: pypsa.Network, uncertainties: dict, seed: int | None = None
) -> None:
    # Storage unit carriers that have inflow uncertainty
    hydro_storage_carriers = [
        "Hydro - Reservoir (Generator)",
        "Hydro - Reservoir",
        "Hydro - Pondage",
        "Hydro - Pump Storage Open Loop (Pumping)",
        "Hydro - Pump Storage Closed Loop (Pumping)",
    ]

    gen_df = n.generators_t["p_max_pu"]
    rng = np.random.default_rng(seed)
    for carrier, unc_params in uncertainties.items():
        # Skip load and hydro storage carriers
        if carrier in ["load"] + hydro_storage_carriers:
            continue
        gens = n.generators[n.generators.carrier == carrier].index
        if len(gens) > 0:
            apply_uncertainty_to_timeseries(
                gen_df, gens, unc_params, 0, 1, seed=seed, rng=rng
            )

    # STORAGE UNITS: hydro with inflow (apply to inflow)
    storage_df = n.storage_units_t["inflow"]
    for carrier in hydro_storage_carriers:
        if carrier in uncertainties:
            units = n.storage_units[n.storage_units.carrier == carrier].index
            if len(units) > 0:
                apply_uncertainty_to_timeseries(
                    storage_df,
                    units,
                    uncertainties[carrier],
                    0,
                    None,
                    seed=seed,
                    rng=rng,
                )


def apply_load_uncertainty(
    n: pypsa.Network, uncertainties: dict, seed: int | None = None
) -> None:
    load_df = n.loads_t["p_set"]
    timesteps = load_df.shape[0]
    unc = uncertainties["load"]
    common_noise = generate_temporal_noise(timesteps, unc["autocorr"], seed=seed)

    for col in load_df.columns:
        forecast = load_df[col] * (1 + unc["bias"] + unc["noise_std"] * common_noise)
        load_df[col] = forecast.clip(lower=0)


def scale_network_timeseries(n: pypsa.Network, config: dict, ft: str) -> None:
    """
    Apply DA/ID timeseries multipliers to renewable generation and load.

    Multiplies ``n.generators_t.p_max_pu`` for renewable carriers and
    ``n.loads_t.p_set`` for all loads by the configured per-market multipliers.
    For forecast (DA) networks the ``da_res_ts_multiplier`` and
    ``da_load_ts_multiplier`` are used; for actual (ID) networks the
    ``id_res_ts_multiplier`` and ``id_load_ts_multiplier`` are used.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify in place.
    config : dict
        Full snakemake configuration dictionary.
    ft : str
        Forecast type wildcard — ``"forecast"`` for DA, ``"actual"`` for ID.
    """
    renewable_carriers = config["eva"]["base_network"]["renewable_carriers"]
    res_gens = n.generators.index[n.generators.carrier.isin(renewable_carriers)]
    res_gens_in_ts = res_gens.intersection(n.generators_t.p_max_pu.columns)

    scale_cfg = config["eva"]["build_network"]["scale_network"]
    if ft == "forecast":
        res_multiplier = scale_cfg["da_res_ts_multiplier"]
        load_multiplier = scale_cfg["da_load_ts_multiplier"]
    elif ft == "actual":
        res_multiplier = scale_cfg["id_res_ts_multiplier"]
        load_multiplier = scale_cfg["id_load_ts_multiplier"]
    else:
        return

    if len(res_gens_in_ts) > 0:
        n.generators_t.p_max_pu[res_gens_in_ts] = (
            n.generators_t.p_max_pu[res_gens_in_ts] * res_multiplier
        ).clip(upper=1.0, lower=0.0)
    n.loads_t.p_set = n.loads_t.p_set.mul(load_multiplier)


def add_forecast_uncertainty(
    n: pypsa.Network,
    forecast_timeseries_source: str,
    noise_std_multiplier: float,
    config: dict,
    target_year: int | None = None,
    climate_year: int | None = None,
    forecast_p_max_pu_path: str | None = None,
    forecast_p_set_path: str | None = None,
) -> None:
    """
    Add forecast uncertainty to load and renewable generation time series.

    When ``forecast_timeseries_source`` is ``"noise"``, applies synthetic
    forecast errors (bias + autocorrelated noise) to load and renewable
    generators. When it is ``"database"``, replaces timeseries with pre-queried
    forecast CSVs written by ``build_network_from_database``. When it is
    ``"csv"``, replaces timeseries with user-supplied forecast CSVs (in the
    same format as ``generators-p_max_pu.csv``/``loads-p_set.csv`` exported by
    ``prepare-sensitivity``).

    Parameters
    ----------
    n : pypsa.Network
        Network to which forecast uncertainty will be added.
    forecast_timeseries_source : str
        ``"noise"`` to generate synthetic errors, ``"database"`` or ``"csv"``
        to load forecast timeseries from CSV files.
    noise_std_multiplier : float
        Global multiplier for all noise_std and bias values (``"noise"`` mode only).
        Values > 1.0 increase forecast errors, values < 1.0 decrease them.
        Setting to 0.0 produces DA timeseries identical to ID (no forecast error).
    config : dict
        Full snakemake configuration dictionary.
    target_year : int, optional
        Unused; kept for API compatibility.
    climate_year : int, optional
        Unused; kept for API compatibility.
    forecast_p_max_pu_path : str, optional
        Path to the forecast generators p_max_pu CSV. Required when
        ``forecast_timeseries_source`` is ``"database"`` or ``"csv"``.
    forecast_p_set_path : str, optional
        Path to the forecast loads p_set CSV. Required when
        ``forecast_timeseries_source`` is ``"database"`` or ``"csv"``.

    Returns
    -------
    None
    """

    if forecast_timeseries_source == "noise":
        # Set random seed for reproducibility
        np.random.seed(4)

        # Define base uncertainty parameters for different components
        base_uncertainties = {
            "load": {
                "bias": 0.01,  # 1% systematic overestimation
                "noise_std": 0.03,  # 3% random noise standard deviation
                "autocorr": 0.9,  # Temporal autocorrelation coefficient
            },
            "Solar (PV)": {
                "bias": -0.02,  # 2% systematic underestimation
                "noise_std": 0.08,  # 8% random noise
                "autocorr": 0.7,
                "cap_effect": True,  # Cap effect to ensure forecast doesn't exceed max output
            },
            "Wind Onshore": {
                "bias": 0.0,  # no bias
                "noise_std": 0.12,  # 12% random noise
                "autocorr": 0.8,
            },
            "Wind Offshore": {
                "bias": 0.0,  # no bias
                "noise_std": 0.12,  # 12% random noise
                "autocorr": 0.8,
            },
            "Hydro - Reservoir (Generator)": {
                "bias": 0.0,
                "noise_std": 0.05,  # 5% random noise
                "autocorr": 0.95,  # High autocorrelation
            },
            "Hydro - Reservoir": {
                "bias": 0.0,
                "noise_std": 0.05,  # 5% random noise
                "autocorr": 0.95,  # High autocorrelation
            },
            "Hydro - Pondage": {
                "bias": 0.0,
                "noise_std": 0.05,  # 5% random noise
                "autocorr": 0.95,  # High autocorrelation
            },
            "Hydro - Pump Storage Open Loop (Pumping)": {
                "bias": 0.0,
                "noise_std": 0.05,  # 5% random noise
                "autocorr": 0.95,  # High autocorrelation
            },
            "Hydro - Pump Storage Closed Loop (Pumping)": {
                "bias": 0.0,
                "noise_std": 0.05,  # 5% random noise
                "autocorr": 0.95,  # High autocorrelation
            },
            "Hydro - Run of River (Turbine)": {
                "bias": 0.0,
                "noise_std": 0.05,  # 5% random noise
                "autocorr": 0.95,  # High autocorrelation
            },
        }

        # Apply the multiplier to all noise_std and bias values so that
        # noise_std_multiplier=0 produces DA timeseries identical to ID.
        uncertainties = {}
        for key, params in base_uncertainties.items():
            uncertainties[key] = params.copy()
            uncertainties[key]["noise_std"] = params["noise_std"] * noise_std_multiplier
            uncertainties[key]["bias"] = params["bias"] * noise_std_multiplier

        apply_load_uncertainty(n, uncertainties, seed=4)
        apply_renewable_uncertainty(n, uncertainties, seed=4)
    else:
        if forecast_p_max_pu_path is None or forecast_p_set_path is None:
            raise ValueError(
                "forecast_p_max_pu_path and forecast_p_set_path must be provided when "
                "forecast_timeseries_source is 'database' or 'csv'"
            )

        # Forecast CSVs are matched to the network by row position (not by
        # index value): both the database-query export and the
        # prepare-sensitivity export write one row per snapshot in
        # chronological order, but the sensitivity export's index column is
        # a plain row number rather than a timestamp, so aligning by value
        # would silently produce all-NaN timeseries.
        gen_p_max_pu = pd.read_csv(forecast_p_max_pu_path, index_col=0)
        if len(gen_p_max_pu) != len(n.snapshots):
            raise ValueError(
                f"Forecast p_max_pu CSV at '{forecast_p_max_pu_path}' has "
                f"{len(gen_p_max_pu)} rows, but the network has {len(n.snapshots)} "
                "snapshots. The forecast CSV must have exactly one row per network "
                "snapshot, in the same order."
            )
        gen_p_max_pu.index = n.snapshots
        valid_cols = [c for c in gen_p_max_pu.columns if c in n.generators.index]
        n.generators_t.p_max_pu = gen_p_max_pu[valid_cols]

        p_set = pd.read_csv(forecast_p_set_path, index_col=0)
        if len(p_set) != len(n.snapshots):
            raise ValueError(
                f"Forecast p_set CSV at '{forecast_p_set_path}' has {len(p_set)} "
                f"rows, but the network has {len(n.snapshots)} snapshots. The "
                "forecast CSV must have exactly one row per network snapshot, in "
                "the same order."
            )
        p_set.index = n.snapshots
        valid_load_cols = [c for c in p_set.columns if c in n.loads_t.p_set.columns]
        n.loads_t.p_set = p_set[valid_load_cols]


def derive_decommissionable(n: pypsa.Network, config: dict) -> None:
    """
    Derive ``p_nom_decommissionable`` for networks that do not provide it.

    Applies the aggregated database rule to Generator and StorageUnit tables
    lacking the column, e.g. older pre-built networks or sensitivity CSVs.
    Existing (non-extendable) units with a carrier in
    ``eva.base_network.decommissioned_carriers`` become decommissionable with
    ``p_nom_min = 0`` and ``p_nom_max = p_nom``; other existing units are fixed
    at ``p_nom``. Extendable units are not decommissionable and keep their
    bounds. Tables that already carry the column are left untouched.
    """
    for c, table in (("Generator", n.generators), ("StorageUnit", n.storage_units)):
        if "p_nom_decommissionable" in table.columns:
            continue
        existing = ~table["p_nom_extendable"].astype(bool)
        decommissionable = decommissionable_by_carrier(
            table.loc[existing, "carrier"], config, c
        )
        bounds = existing_capacity_bounds(
            table.loc[existing, "p_nom"], decommissionable
        )
        table["p_nom_decommissionable"] = False
        table.loc[existing, bounds.columns] = bounds
        logger.info(
            f"Derived p_nom_decommissionable for {c}s: "
            f"{decommissionable.sum()} of {existing.sum()} existing units decommissionable"
        )


def cap_decommissionable_capacity(n: pypsa.Network) -> None:
    """
    Cap decommissionable, non-extendable units at their installed capacity.

    Existing units may be reduced but never grow, so ``p_nom_max`` is set to
    ``p_nom``. Applied to user-edited sensitivity CSVs, where ``p_nom_max`` is
    typically not exported and defaults to infinity.
    """
    for table in (n.generators, n.storage_units):
        capped = table["p_nom_decommissionable"].eq(True) & ~table[
            "p_nom_extendable"
        ].astype(bool)
        table.loc[capped, "p_nom_max"] = table.loc[capped, "p_nom"]


def update_initial_capacities(n: pypsa.Network, config: dict) -> None:
    """
    Update initial capacities and clip minimums for extendable and decommissionable components.
    """
    upscale_capacities = config["enable"]
    capacity_scaling_factor = config["capacity_scaling_factor"]
    min_capacity = config["min_capacity"]

    for c in ["generators", "stores", "storage_units"]:
        if c not in n.components.keys():
            continue

        component_map = {
            "generators": n.generators,
            "stores": n.stores,
            "storage_units": n.storage_units,
        }

        component_table = component_map[c]
        attr = "e" if c == "stores" else "p"

        extendable_col = f"{attr}_nom_extendable"
        decommissionable_col = f"{attr}_nom_decommissionable"
        nom_col = f"{attr}_nom"
        nom_min_col = f"{attr}_nom_min"

        # Handle extendable components
        for col in [extendable_col, decommissionable_col]:
            if col in component_table.columns:
                # Ensure boolean dtype (NetCDF may load as object dtype)
                mask = component_table[col].fillna(False).astype(bool)
                extendables = component_table.index[mask]
                if not extendables.empty:
                    # Clip the minimum column to min_capacity
                    if nom_min_col in component_table.columns:
                        component_table.loc[extendables, nom_min_col] = (
                            component_table.loc[extendables, nom_min_col].clip(
                                lower=min_capacity
                            )
                        )

                    # Ensure actual capacities respect the minimum
                    component_table.loc[extendables, nom_col] = component_table.loc[
                        extendables, nom_col
                    ].clip(lower=min_capacity)

    if upscale_capacities:
        # Update the initial capacities of extendable generators by 100%
        # to avoid infeasibility in the first iteration (due to high reserve requirements)
        n.generators.p_nom = n.generators.p_nom.mul(capacity_scaling_factor).clip(
            lower=min_capacity
        )
        # Exclude implicit DSR storage units from capacity scaling (their p_nom is calculated from demand data)
        non_dsr_mask = n.storage_units.carrier != "DSR_implicit"
        n.storage_units.loc[non_dsr_mask, "p_nom"] = (
            n.storage_units.loc[non_dsr_mask, "p_nom"]
            .mul(capacity_scaling_factor)
            .clip(lower=min_capacity)
        )


def remove_p_nom_max(n: pypsa.Network) -> None:
    """
    Make the p_nom_max of extendable components np.inf
    """

    # Generators
    n.generators.loc[n.generators.p_nom_extendable, "p_nom_max"] = np.inf

    # Storage units
    n.storage_units.loc[n.storage_units.p_nom_extendable, "p_nom_max"] = np.inf


def make_must_run_pu(n: pypsa.Network, min_capacity: float) -> None:
    """
    Adapt the must-run capacity to p_nom * weighted average must-run ratio
    Remove p_nom_min
    """

    # Select must run generators
    if "is_must_run" in n.generators.columns:
        must_run_mask = n.generators["is_must_run"].fillna(False).astype(bool)
    else:
        must_run_mask = pd.Series(False, index=n.generators.index)
    must_run_gens = n.generators.index[must_run_mask]

    # Adapt their must-run capacity
    n.generators.loc[must_run_gens, "must-run capacity"] = (
        n.generators.loc[must_run_gens, "p_nom"]
        * n.generators.loc[must_run_gens, "weighted average must-run ratio"]
    )

    # Remove p_nom_min of the extendable generators
    must_run_gens_extendable = must_run_gens[
        n.generators.loc[must_run_gens, "p_nom_extendable"]
    ]
    n.generators.loc[must_run_gens_extendable, "p_nom_min"] = min_capacity


def remove_cyclic_soc(n: pypsa.Network) -> None:
    """
    Make cyclic_state_of_charge_parameter false for all storage units to ensure preopt-market interaction works as intended.
    """
    # Add a mask for implicit DSR units (they are always cyclic)
    n.storage_units["cyclic_state_of_charge"] = False


def initialize_soc_initial(n: pypsa.Network) -> None:
    """
    Initialize state_of_charge_initial for hydro storage units.

    Uses min/max SOC timeseries constraints when available, falling back to
    capacity-based defaults. Mirrors the logic in add_storage_constraints in
    build_network_from_database.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to modify.
    """
    has_min = (
        "state_of_charge_min" in n.storage_units_t
        and not n.storage_units_t.state_of_charge_min.empty
    )
    has_max = (
        "state_of_charge_max" in n.storage_units_t
        and not n.storage_units_t.state_of_charge_max.empty
    )

    hydro_units = n.storage_units[
        n.storage_units.carrier.str.contains("Hydro", na=False)
    ].index

    for unit in hydro_units:
        min_soc = 0.0
        max_soc = None

        if has_min and unit in n.storage_units_t.state_of_charge_min.columns:
            unit_min_series = n.storage_units_t.state_of_charge_min[unit].dropna()
            if not unit_min_series.empty:
                min_soc = unit_min_series.iloc[0]

        if has_max and unit in n.storage_units_t.state_of_charge_max.columns:
            unit_max_series = n.storage_units_t.state_of_charge_max[unit].dropna()
            if not unit_max_series.empty:
                max_soc = unit_max_series.max()

        if max_soc is not None and max_soc > min_soc:
            n.storage_units.loc[unit, "state_of_charge_initial"] = (
                min_soc + max_soc
            ) / 2.0
        elif min_soc == max_soc:
            n.storage_units.loc[unit, "state_of_charge_initial"] = min_soc
        elif min_soc > 0:
            n.storage_units.loc[unit, "state_of_charge_initial"] = min_soc * 1.01
        else:
            max_hours = n.storage_units.loc[unit, "max_hours"]
            if np.isinf(max_hours):
                max_hours = 8760
            n.storage_units.loc[unit, "state_of_charge_initial"] = (
                n.storage_units.loc[unit, "p_nom"] * max_hours
            ) * 0.5

    logger.info(
        f"Initialized state_of_charge_initial for {len(hydro_units)} hydro units"
    )


def set_committable_status(n: pypsa.Network, config: dict) -> int:
    """
    Set committable status for generators based on carrier types in config.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to modify
    config : dict
        Configuration dictionary containing committable_carriers list

    Returns
    -------
    int
        Number of generators set as committable
    """
    committable_carriers = config["eva"]["base_network"]["committable_carriers"]

    if not committable_carriers:
        logger.warning("No committable carriers specified in config")
        return 0

    # Set generators with matching carriers as committable
    committable_mask = n.generators.carrier.isin(committable_carriers)
    n.generators.loc[committable_mask, "committable"] = True

    num_committable = committable_mask.sum()
    logger.info(
        f"Set {num_committable} generators as committable for carriers: {committable_carriers}"
    )

    return num_committable


def remove_ramp_su_sd(n: pypsa.Network) -> int:
    """
    Remove ramp_limit_start_up and ramp_limit_shut_down for all committable generators.

    Sets both attributes to NaN, removing the startup/shutdown ramp constraints
    from the LP. This prevents the cascade infeasibility caused by
    ``ramp_limit_start_up << p_min_pu`` (common for nuclear units): tiny fractional
    status values produced by the LP relaxation propagate through the ramp formula
    and create a negative RHS in the next rolling-horizon window.

    The slow-startup/shutdown physics is already captured by ``min_up_time`` and
    ``min_down_time``; ``ramp_limit_up``/``ramp_limit_down`` remain active for
    intra-operational ramping.

    Parameters
    ----------
    n : pypsa.Network
        The network to modify in place.

    Returns
    -------
    int
        Number of committable generators modified.
    """
    if "committable" not in n.generators.columns:
        logger.debug("remove_ramp_su_sd: no committable column found, skipping")
        return 0

    committable_mask = n.generators["committable"].fillna(False).astype(bool)
    if not committable_mask.any():
        logger.debug("remove_ramp_su_sd: no committable generators found, skipping")
        return 0

    n.generators.loc[committable_mask, "ramp_limit_start_up"] = np.nan
    n.generators.loc[committable_mask, "ramp_limit_shut_down"] = np.nan

    n_modified = int(committable_mask.sum())
    logger.info(
        f"remove_ramp_su_sd: cleared ramp_limit_start_up and ramp_limit_shut_down "
        f"for {n_modified} committable generator(s)"
    )
    return n_modified


def adapt_costs_for_lin_uc(n: pypsa.Network, config: dict) -> int | None:
    """
    Adapt start-up and shut-down costs for linearized unit commitment.

    When convex hull tightening is enabled (use_convex_hull_tightening=True):
        PyPSA's tight convex hull relaxation (Hua et al. 2018) requires
        start_up_cost == shut_down_cost for all committable generators.
        This function computes symmetric costs as:
            sup_symmetric = sdn_symmetric = (start_up_cost + shut_down_cost) / 2

    When convex hull tightening is disabled (use_convex_hull_tightening=False):
        Ensures costs are NOT symmetric to prevent PyPSA from automatically
        applying tightening constraints. A small asymmetry (1% of total cost)
        is introduced by adding to start_up_cost and subtracting from shut_down_cost.

    Only applies when config["solving"]["options"]["linearized_unit_commitment"]
    is True and there are committable generators.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to modify
    config : dict
        Configuration dictionary containing solving options

    Returns
    -------
    int | None
        Number of generators with modified costs, or None if not applicable
    """
    # Check if linearized UC is enabled
    linearized_uc = config["solving"]["options"]["linearized_unit_commitment"]
    if not linearized_uc:
        logger.debug("Linearized UC disabled, skipping cost adaptation")
        return

    use_convex_hull = config["solving"]["options"]["use_convex_hull_tightening"]
    logger.debug(
        f"adapt_costs_for_lin_uc: use_convex_hull_tightening = {use_convex_hull}"
    )

    # Check if committable column exists
    if "committable" not in n.generators.columns:
        logger.debug("No committable column found in generators")
        return

    # Get committable generators
    committable_mask = n.generators["committable"].fillna(False).astype(bool)
    committable_gens = n.generators.index[committable_mask]

    if committable_gens.empty:
        logger.debug("No committable generators found")
        return

    logger.debug(
        f"adapt_costs_for_lin_uc: Found {len(committable_gens)} committable generators"
    )

    # Ensure cost columns exist with default values
    if "start_up_cost" not in n.generators.columns:
        n.generators["start_up_cost"] = 0.0
    if "shut_down_cost" not in n.generators.columns:
        n.generators["shut_down_cost"] = 0.0

    # Get current costs for committable generators
    start_up = n.generators.loc[committable_gens, "start_up_cost"].fillna(0.0)
    shut_down = n.generators.loc[committable_gens, "shut_down_cost"].fillna(0.0)

    if use_convex_hull:
        # CONVEX-HULL METHOD: Force symmetric costs for all committable generators
        # Compute symmetric cost: (start_up_cost + shut_down_cost) / 2
        symmetric_cost = (start_up + shut_down) / 2.0

        # Apply symmetric costs to both start-up and shut-down
        n.generators.loc[committable_gens, "start_up_cost"] = symmetric_cost
        n.generators.loc[committable_gens, "shut_down_cost"] = symmetric_cost

        # Count generators that had asymmetric costs
        asymmetric_count = (start_up != shut_down).sum()

        logger.info(
            f"Convex hull method: Symmetrized costs for {len(committable_gens)} committable "
            f"generators ({asymmetric_count} had asymmetric costs). "
            f"Mean symmetric cost: {symmetric_cost.mean():.2f} EUR"
        )
    else:
        # SIMPLE METHOD: Ensure costs are NOT symmetric to prevent tightening
        # Identify generators that currently have symmetric costs
        symmetric_mask = start_up == shut_down
        symmetric_gens = committable_gens[symmetric_mask]

        logger.debug(
            f"adapt_costs_for_lin_uc SIMPLE METHOD: Found {len(symmetric_gens)} generators with symmetric costs out of {len(committable_gens)} committable"
        )

        if len(symmetric_gens) > 0:
            # Calculate total cost for symmetric generators
            total_cost = start_up.loc[symmetric_gens] + shut_down.loc[symmetric_gens]

            # Introduce 1% asymmetry to prevent PyPSA convex hull tightening
            # Add 0.5% to start_up, subtract 0.5% from shut_down
            asymmetry = 0.005 * total_cost  # 0.5% of total cost

            # Ensure minimum asymmetry of 0.01 EUR to handle zero-cost generators
            asymmetry = asymmetry.clip(lower=0.01)

            new_start_up = start_up.loc[symmetric_gens] + asymmetry
            new_shut_down = (shut_down.loc[symmetric_gens] - asymmetry).clip(lower=0.0)

            n.generators.loc[symmetric_gens, "start_up_cost"] = new_start_up
            n.generators.loc[symmetric_gens, "shut_down_cost"] = new_shut_down

            logger.info(
                f"Simple method: Introduced asymmetry for {len(symmetric_gens)} committable "
                f"generators with symmetric costs (1% total cost adjustment) to disable "
                f"convex hull tightening. {len(committable_gens) - len(symmetric_gens)} "
                f"generators already had asymmetric costs."
            )
        else:
            logger.info(
                f"Simple method: All {len(committable_gens)} committable generators "
                f"already have asymmetric costs. No modification needed."
            )

    return len(committable_gens)


def remove_dsr(n: pypsa.Network, dsr_cfg: dict) -> None:
    # Determine which DSR carriers to remove
    carrier_to_remove = []

    if not dsr_cfg["enabled"]:
        carrier_to_remove = ["DSR_implicit", "DSR_explicit", "DSR"]
    else:
        if not dsr_cfg["explicit_dsr"]["enabled"]:
            carrier_to_remove.append("DSR_explicit")
        if not dsr_cfg["implicit_dsr"]["enabled"]:
            carrier_to_remove.append("DSR_implicit")

    if carrier_to_remove:
        # Vectorized removal from StorageUnits
        mask_storage = n.storage_units["carrier"].isin(carrier_to_remove)
        n.remove("StorageUnit", n.storage_units.index[mask_storage])
        logger.info(
            f"Removed {mask_storage.sum()} StorageUnit(s) with carriers: {carrier_to_remove}"
        )

        # Vectorized removal from Generators
        mask_gen = n.generators["carrier"].isin(carrier_to_remove)
        n.remove("Generator", n.generators.index[mask_gen])
        logger.info(
            f"Removed {mask_gen.sum()} Generator(s) with carriers: {carrier_to_remove}"
        )


def override_reserve_id_blocked(n: pypsa.Network, config: dict) -> None:
    """
    Override reserve id_blocked and reserve_blocking_method parameters in bus table based on config values.

    This function updates the reserve_FCR_id_blocked, reserve_FRR_id_blocked,
    and reserve_FRR_reserve_blocking_method columns in the bus table using values from the config.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to modify
    config : dict
        Configuration dictionary containing eva.reserves settings

    Returns
    -------
    None
    """
    reserves_cfg = config["eva"]["reserves"]

    # Override FCR id_blocked if configured
    if "fcr" in reserves_cfg and "id_blocked" in reserves_cfg["fcr"]:
        fcr_blocked = reserves_cfg["fcr"]["id_blocked"]
        if "reserve_FCR_id_blocked" in n.buses.columns:
            n.buses["reserve_FCR_id_blocked"] = fcr_blocked
            logger.info(
                f"Set reserve_FCR_id_blocked to {fcr_blocked} for all buses from config"
            )
        else:
            logger.warning(
                "reserve_FCR_id_blocked column not found in buses table, skipping FCR override"
            )

    # Override FRR id_blocked if configured
    if "frr" in reserves_cfg and "id_blocked" in reserves_cfg["frr"]:
        frr_blocked = reserves_cfg["frr"]["id_blocked"]
        if "reserve_FRR_id_blocked" in n.buses.columns:
            n.buses["reserve_FRR_id_blocked"] = frr_blocked
            logger.info(
                f"Set reserve_FRR_id_blocked to {frr_blocked} for all buses from config"
            )
        else:
            logger.warning(
                "reserve_FRR_id_blocked column not found in buses table, skipping FRR override"
            )

    # Override FRR reserve_blocking_method if configured
    if "frr" in reserves_cfg and "reserve_blocking_method" in reserves_cfg["frr"]:
        frr_blocking_method = reserves_cfg["frr"]["reserve_blocking_method"]
        # Add column if it doesn't exist
        if "reserve_FRR_reserve_blocking_method" not in n.buses.columns:
            n.buses["reserve_FRR_reserve_blocking_method"] = "fraction"  # Default
            logger.info(
                "Created reserve_FRR_reserve_blocking_method column with default 'fraction'"
            )
        n.buses["reserve_FRR_reserve_blocking_method"] = frr_blocking_method
        logger.info(
            f"Set reserve_FRR_reserve_blocking_method to '{frr_blocking_method}' for all buses from config"
        )


def apply_renewable_negative_pricing(n: pypsa.Network, config: dict) -> None:
    """
    Create separate _NEG generators with capacity bidding at negative prices.

    This function implements the T1.3 Option 3 feature by splitting renewable generators
    into two parts:
    - Original generator: keeps (1 - share) of capacity at normal marginal cost
    - New _NEG generator: has (share) of capacity at negative marginal cost

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA Network object to modify
    config : dict
        Configuration dictionary containing negative_pricing settings under
        eva.base_network.negative_pricing with keys:
        - enable: bool - Whether to apply negative pricing
        - share: float - Default capacity share for negative pricing (0-1)
        - value: float - Default negative marginal cost value in EUR/MWh
        - carriers: dict - Per-carrier configuration with share/value overrides

    Returns
    -------
    None
        Modifies n.generators in place by adding _NEG generators and reducing
        original generator capacities

    Notes
    -----
    - Only applies to Solar (PV), Wind Onshore, Wind Offshore, and
      Hydro - Run of River (Turbine) carriers
    - Creates new generators with suffix "_NEG" (e.g., "BE00 Solar (PV)_NEG")
    - The _NEG generators share the same p_max_pu timeseries as the original
    - Original generator capacity is reduced by (1 - share) * p_nom
    """
    neg_pricing_config = config["eva"]["base_network"]["negative_pricing"]

    # Defaults and per-carrier config (config validated by schema)
    default_share = neg_pricing_config["share"]
    default_value = neg_pricing_config["value"]
    carriers_config = neg_pricing_config.get("carriers", {})

    # Target carriers for negative pricing
    target_carriers = [
        "Solar (PV)",
        "Wind Onshore",
        "Wind Offshore",
        "Hydro - Run of River (Turbine)",
    ]

    def get_carrier_config(carrier: str) -> tuple[float, float]:
        """Get share and value for a specific carrier."""
        carrier_cfg = carriers_config.get(carrier, {})
        return carrier_cfg.get("share", default_share), carrier_cfg.get(
            "value", default_value
        )

    # Find renewable generators
    renewable_gens = n.generators[n.generators.carrier.isin(target_carriers)].copy()

    if len(renewable_gens) == 0:
        logger.warning("No renewable generators found for negative pricing. Skipping.")
        return

    logger.info(f"Applying negative pricing to {len(renewable_gens)} generators")

    # VECTORIZED: Precompute shares and values for all carriers
    carrier_configs = renewable_gens["carrier"].apply(get_carrier_config)
    shares = pd.Series([c[0] for c in carrier_configs], index=renewable_gens.index)
    neg_values = pd.Series([c[1] for c in carrier_configs], index=renewable_gens.index)

    # Filter to only generators with positive share
    valid_mask = shares > 0
    gens_to_split = renewable_gens[valid_mask].copy()
    shares_valid = shares[valid_mask]
    neg_values_valid = neg_values[valid_mask]

    if len(gens_to_split) == 0:
        neg_gen_count = 0
    else:
        # CRITICAL: Save original p_nom BEFORE any modification
        original_p_noms = gens_to_split["p_nom"].copy()

        # Vectorized capacity reduction for original generators
        n.generators.loc[gens_to_split.index, "p_nom"] *= 1 - shares_valid

        # Prepare data for bulk NEG generator addition
        neg_gen_names = [f"{idx}_NEG" for idx in gens_to_split.index]
        # Use ORIGINAL p_nom values (not the reduced ones)
        neg_p_noms = original_p_noms * shares_valid

        # Use madd() for bulk addition
        # IMPORTANT: Copy all attributes needed for NPV calculation and network consistency
        n.madd(
            "Generator",
            names=neg_gen_names,
            bus=gens_to_split["bus"].values,
            carrier=gens_to_split["carrier"].values,
            p_nom=neg_p_noms.values,
            marginal_cost=neg_values_valid.values,
            p_nom_extendable=gens_to_split.get(
                "p_nom_extendable", pd.Series(False, index=gens_to_split.index)
            ).values,
            efficiency=gens_to_split.get(
                "efficiency", pd.Series(1.0, index=gens_to_split.index)
            ).values,
            # NPV-critical attributes (must be copied from original generators)
            lifetime=gens_to_split.get(
                "lifetime", pd.Series(pd.NA, index=gens_to_split.index)
            ).values,
            build_year=gens_to_split.get(
                "build_year", pd.Series(pd.NA, index=gens_to_split.index)
            ).values,
            decommissioning_year=gens_to_split.get(
                "decommissioning_year", pd.Series(pd.NA, index=gens_to_split.index)
            ).values,
            # Other important attributes that should match the original
            capital_cost=gens_to_split.get(
                "capital_cost", pd.Series(0.0, index=gens_to_split.index)
            ).values,
            p_nom_decommissionable=gens_to_split.get(
                "p_nom_decommissionable", pd.Series(False, index=gens_to_split.index)
            ).values,
        )

        # Copy p_max_pu timeseries for NEG generators
        for orig_idx, neg_name in zip(gens_to_split.index, neg_gen_names):
            if orig_idx in n.generators_t.p_max_pu.columns:
                n.generators_t.p_max_pu[neg_name] = n.generators_t.p_max_pu[
                    orig_idx
                ].copy()

        neg_gen_count = len(gens_to_split)

        # Debug logging (sample first few)
        for gen_idx, share in list(zip(gens_to_split.index, shares_valid))[:5]:
            original_p_nom = original_p_noms[gen_idx]
            logger.debug(
                f"Split {gen_idx}: original={original_p_nom * (1 - share):.1f}MW, "
                f"_NEG={original_p_nom * share:.1f}MW @ {neg_values_valid[gen_idx]} EUR/MWh"
            )

    # Summary logging
    for carrier in target_carriers:
        share, value = get_carrier_config(carrier)
        neg_gens = [
            g
            for g in n.generators.index
            if g.endswith("_NEG") and n.generators.loc[g, "carrier"] == carrier
        ]
        if neg_gens:
            total_cap = n.generators.loc[neg_gens, "p_nom"].sum()
            logger.info(
                f"  {carrier}: {len(neg_gens)} _NEG generators, "
                f"{total_cap:.1f}MW @ {value} EUR/MWh"
            )

    logger.info(f"Created {neg_gen_count} _NEG generators with negative marginal costs")


def _scale_generator_costs(
    c: str,
    unit: str,
    component_table: pd.DataFrame,
    current_cap: float,
    updated_cap: float,
) -> None:
    if c == "Generator" and current_cap > 0:
        scaling_ratio = updated_cap / current_cap
        for cost_attr in ["start_up_cost", "shut_down_cost", "stand_by_cost"]:
            if cost_attr in component_table.columns:
                component_table.loc[unit, cost_attr] *= scaling_ratio


def apply_capacities_from_csv(
    n: pypsa.Network,
    unit_states_path: str,
    ty: int,
) -> None:
    """
    Apply per-unit capacity updates from a unit_states CSV to a network in place.

    Reads ``unit_states_{it}.csv`` written by ``update_capacity`` and sets ``p_nom``
    on every Generator / StorageUnit whose updated capacity is recorded there.
    Only rows matching ``ty`` are applied.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify in place.
    unit_states_path : str
        Path to the ``unit_states_{it}.csv`` file produced by ``update_capacity``.
    ty : int
        Target year whose rows should be applied.
    """
    df = pd.read_csv(unit_states_path)
    df = df[df["ty"] == ty]

    component_map = {"Generator": n.generators, "StorageUnit": n.storage_units}

    # VECTORIZED: Group by component for batch updates
    for comp, group in df.groupby("component"):
        if comp not in component_map:
            continue
        table = component_map[comp]

        # Filter to valid units that exist in the table
        valid_units = group[group["unit"].isin(table.index)]["unit"]
        if len(valid_units) == 0:
            continue

        # Get original capacities before update
        original_caps = table.loc[valid_units, "p_nom"].copy()

        # Vectorized capacity update
        new_caps_series = group.set_index("unit").loc[valid_units, "p_nom"]
        table.loc[valid_units, "p_nom"] = new_caps_series.values

        # Vectorized cost scaling for generators
        if comp == "Generator":
            scaling_ratios = new_caps_series.values / original_caps.replace(0, 1).values
            for cost_attr in ["start_up_cost", "shut_down_cost", "stand_by_cost"]:
                if cost_attr in table.columns:
                    table.loc[valid_units, cost_attr] *= scaling_ratios


if __name__ == "__main__":
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    snakemake = get_snakemake(globals_scope=globals(), rule="build_base_network")

    configure_logging(snakemake)

    # Sensitivity Mode
    if (
        snakemake.config["eva"]["sensitivity"]
        and snakemake.wildcards.it == "0"  # ty: ignore[possibly-missing-attribute]
    ):
        logging.info(
            "Running in sensitivity mode. Building network from CSVs instead of the normal routine."
        )

        csv_folder_path = (
            Path(
                snakemake.input.base_network  # ty: ignore[possibly-missing-attribute]
            )
            .resolve()
            .as_posix()
        )

        logging.info(
            f"Validating CSV files in {csv_folder_path} (sensitivity mode - strict validation)..."
        )
        validation_errors = []

        # Validate generators.csv
        # In sensitivity mode, use strict validation (validate_component)
        # User-edited CSVs must have all required fields including build_year
        try:
            gen_df = pd.read_csv(f"{csv_folder_path}/generators.csv", index_col=0)
            GeneratorSchema.validate_component(gen_df)
            logging.info("✓ generators.csv passed validation")
        except FileNotFoundError:
            logging.warning(f"generators.csv not found in {csv_folder_path}")
        except Exception as e:
            validation_errors.append(f"generators.csv: {e}")
            logging.error(f"✗ generators.csv validation failed: {e}")

        # Validate storage_units.csv
        try:
            storage_df = pd.read_csv(
                f"{csv_folder_path}/storage_units.csv", index_col=0
            )
            StorageUnitSchema.validate_component(storage_df)
            logging.info("✓ storage_units.csv passed validation")
        except FileNotFoundError:
            logging.warning(f"storage_units.csv not found in {csv_folder_path}")
        except Exception as e:
            validation_errors.append(f"storage_units.csv: {e}")
            logging.error(f"✗ storage_units.csv validation failed: {e}")

        if validation_errors:
            error_msg = "\n".join(validation_errors)
            raise ValueError(
                f"CSV validation failed with {len(validation_errors)} error(s):\n{error_msg}"
            )

        # Load the network. PyPSA's CSV format numbers the rows of the
        # snapshot and time-series files and keeps the timestamps in a
        # separate column, so its ``parse_dates=True`` read warns once per
        # file about the row numbers; the snapshots are still read correctly.
        n = pypsa.Network()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Could not infer format", category=UserWarning
            )
            n.import_from_csv_folder(csv_folder_path)
    else:
        # Normal mode: load the network produced upstream
        n = pypsa.Network(snakemake.input["base_network"])

        # Fill in missing fields such as build_year. validate_with_defaults
        # validates strictly first and only applies defaults where that fails,
        # so a network that already carries them (one built from the database,
        # which validates at build time) passes through untouched.
        target_year = int(
            snakemake.wildcards.ty
        )  # ty: ignore[possibly-missing-attribute]
        earliest_ty = (
            min(snakemake.config["scenario"]["ty"])
            if snakemake.config["scenario"]["ty"]
            else target_year
        )
        logging.info(f"Validating {len(n.generators)} generators from input network")
        n.generators = GeneratorSchema.validate_with_defaults(
            n.generators,
            target_year=target_year,
            earliest_ty=earliest_ty,
            carrier_efficiency_defaults=snakemake.config["eva"]["base_network"][
                "efficiency"
            ],
        )
        logging.info(
            f"Validating {len(n.storage_units)} storage units from input network"
        )
        n.storage_units = StorageUnitSchema.validate_with_defaults(
            n.storage_units,
            target_year=target_year,
            earliest_ty=earliest_ty,
            carrier_efficiency_defaults=snakemake.config["eva"]["base_network"][
                "efficiency"
            ],
        )

    derive_decommissionable(n, snakemake.config)
    if (
        snakemake.config["eva"]["sensitivity"]
        and snakemake.wildcards.it == "0"  # ty: ignore[possibly-missing-attribute]
    ):
        cap_decommissionable_capacity(n)

    # In restart mode the iteration 0 input is an evolved network from a previous
    # run, which already carries the one-off transformations below baked in from
    # that run's own iteration 0. Re-applying the multiplicative and noise-adding
    # ones would compound them, so they are skipped. The idempotent iteration 0
    # steps still run, which is what lets the new run's config take effect.
    is_warm_started = (
        int(snakemake.wildcards.it) == 0  # ty: ignore[possibly-missing-attribute]
        and snakemake.config["eva"]["restart_from"]["enable"]
    )
    if is_warm_started:
        logging.info(
            "Warm start from run %s: skipping capacity/load/timeseries scaling and "
            "forecast noise, which are already applied in the parent network.",
            snakemake.config["eva"]["restart_from"]["parent_run"],
        )

    # For it > 0, apply the per-unit capacity updates from the previous iteration's
    # unit_states CSV instead of reading from an updated_network file.
    if int(snakemake.wildcards.it) > 0:  # ty: ignore[possibly-missing-attribute]
        apply_capacities_from_csv(
            n,
            unit_states_path=snakemake.input["unit_states_csv"],
            ty=int(snakemake.wildcards.ty),  # ty: ignore[possibly-missing-attribute]
        )

    # Clean up generator names
    clean_up_gen_names(n)

    # Override reserve id_blocked parameters from config (only in first iteration)
    if snakemake.wildcards.it == "0":  # ty: ignore[possibly-missing-attribute]
        override_reserve_id_blocked(n, snakemake.config)

    if snakemake.wildcards.it == "0":  # ty: ignore[possibly-missing-attribute]
        group_electrical_links_by_connection(n)

        if not is_warm_started:
            load_scaling_factor = snakemake.config["eva"]["build_network"][
                "scale_network"
            ]["load_scaling_factor"]

            n.loads_t.p_set = n.loads_t.p_set.mul(load_scaling_factor)

            update_initial_capacities(n, snakemake.params.scale_network)

        # Fix any inconsistencies where p_nom_max < p_nom_min
        patch_expansion_limit_inconsistencies(n)

    # Determine ft (forecast/actual) for shedding and spillage functions
    # In perfect foresight mode, there is no ft wildcard, so use 'actual'
    if snakemake.config["eva"]["perfect_foresight"]["enable"]:
        ft = "actual"
    else:
        ft = snakemake.wildcards.ft  # ty: ignore[possibly-missing-attribute]

    if (
        snakemake.config["solving"]["options"]["load_shedding"]
        and "load shedding" not in n.carriers.index
    ):
        ty = int(snakemake.wildcards.ty)  # ty: ignore[possibly-missing-attribute]
        add_load_shedding(n, ty, ft, snakemake.config)

    if (
        snakemake.config["solving"]["options"]["reserve_shedding"]
        and "reserve shedding" not in n.carriers.index
    ):
        ty = int(snakemake.wildcards.ty)  # ty: ignore[possibly-missing-attribute]
        add_reserve_shedding(n, ty, ft, snakemake.config)

    if (
        snakemake.config["solving"]["options"]["load_spill"]
        and "load spillage" not in n.carriers.index
    ):
        ty = int(snakemake.wildcards.ty)  # ty: ignore[possibly-missing-attribute]
        add_load_spillage(n, ty, ft, snakemake.config)

    # Adapt link limits based on NTC data if using NTC or FBMC market coupling
    if snakemake.wildcards.it == "0":  # ty: ignore[possibly-missing-attribute]
        market_interconnection = snakemake.config["eva"]["market_interconnection"]
        if market_interconnection == "fbmc" or market_interconnection == "ntc":
            eraa_year = snakemake.config["eva"]["eraa_year"]
            n = adapt_network_for_market_coupling(
                n, market_interconnection, int(eraa_year)
            )
        #
        # Spread overlapping bus coordinates for better visualization (only in first iteration)
        spread_overlapping_bus_coordinates(n)

    # Add forecast uncertainty if this is a forecast network
    # Only do this for the first iteration of the EVA
    if not snakemake.config["eva"]["perfect_foresight"]["enable"]:
        if (
            snakemake.wildcards.ft  # ty: ignore[possibly-missing-attribute]
            == "forecast"
            and int(snakemake.wildcards.it)  # ty: ignore[possibly-missing-attribute]
            == 0
            and not is_warm_started
        ):
            # Get noise_std_multiplier from config if provided, otherwise use default (1.0)
            noise_std_multiplier = snakemake.config["eva"]["base_network"][
                "forecast_noise_std"
            ]
            forecast_timeseries_source = snakemake.config["eva"]["build_network"][
                "forecast_timeseries_source"
            ]

            add_forecast_uncertainty(
                n,
                forecast_timeseries_source=forecast_timeseries_source,
                noise_std_multiplier=noise_std_multiplier,
                config=snakemake.config,
                forecast_p_max_pu_path=snakemake.input["forecast_p_max_pu"]
                if forecast_timeseries_source in ("database", "csv")
                else None,  # ty: ignore[possibly-missing-attribute]
                forecast_p_set_path=snakemake.input["forecast_p_set"]
                if forecast_timeseries_source in ("database", "csv")
                else None,  # ty: ignore[possibly-missing-attribute]
            )

        if int(snakemake.wildcards.it) == 0 and not is_warm_started:
            scale_network_timeseries(
                n,
                snakemake.config,
                snakemake.wildcards.ft,  # ty: ignore[possibly-missing-attribute],
            )

    # Remove p_nom_max from extendable technologies if option is chosen
    if (
        not snakemake.config["eva"]["base_network"]["consider_p_nom_max"]
        and int(snakemake.wildcards.it) == 0  # ty: ignore[possibly-missing-attribute]
    ):
        remove_p_nom_max(n)

    # Adapt must-run capacity if pu_must_run option is chosen
    if snakemake.config["eva"]["base_network"]["pu_must_run"]:
        make_must_run_pu(
            n,
            snakemake.config["eva"]["build_network"]["scale_network"]["min_capacity"],
        )

    if not snakemake.config["eva"]["base_network"]["consider_cyclic_soc"]:
        remove_cyclic_soc(n)

    initialize_soc_initial(n)

    if int(snakemake.wildcards.it) == 0:  # ty: ignore[possibly-missing-attribute]
        # Set committable status based on carrier configuration
        set_committable_status(n, snakemake.config)
        # Adapt start-up/shut-down costs for linearized UC (if enabled)
        adapt_costs_for_lin_uc(n, snakemake.config)
        # Remove startup/shutdown ramp limits if enabled (prevents LP cascade infeasibilities)
        if snakemake.config["eva"]["base_network"]["remove_ramp_su_sd"]:
            remove_ramp_su_sd(n)

    # Remove DSR if they are not enabled
    if int(snakemake.wildcards.it) == 0:  # ty: ignore[possibly-missing-attribute]
        remove_dsr(n, dsr_cfg=snakemake.config["eva"]["dsr"])

    # Apply negative pricing to renewables if enabled (only in first iteration)
    if int(snakemake.wildcards.it) == 0:  # ty: ignore[possibly-missing-attribute]
        if snakemake.config["eva"]["base_network"]["negative_pricing"]["enable"]:
            apply_renewable_negative_pricing(n, snakemake.config)

    run_eva = snakemake.config["eva"]["enable"]
    if run_eva:
        logging.info(
            "Running EU-Flex with EVA turned on. Asset capacities will be updated."
        )
    else:
        n.generators.p_nom_extendable = False
        n.storage_units.p_nom_extendable = False
        n.generators["p_nom_decommissionable"] = False
        n.storage_units["p_nom_decommissionable"] = False
        logging.info(
            "Running EU-Flex with EVA turned off. Asset capacities will not be updated."
        )

    n.loads["carrier"] = n.loads.carrier.where(n.loads.carrier.ne(""), "load")
    # TODO: remove this reserve providers patching function once the full scale EVA pre-built has been updated
    n = patch_reserve_providers(n)
    # TODO: remove this patching of p_min_pu once the DB has been fixed (only needed for unit level data for full scale EVA)
    n = patch_p_min_pu_inconsistencies(n)
    n = patch_p_set_p_min_max_inconsistencies(n)
    if (
        snakemake.config["eva"]["reserves"]["option"] == "derated"
        and snakemake.wildcards.it == "0"
    ):  # ty: ignore[possibly-missing-attribute]
        # TODO: remove once prebuilt networks carry headroom-aware up amounts
        # n = patch_reserve_amounts_exceed_must_run_headroom(n)
        n = patch_reserve_amounts_exceed_pu_bounds(n)
    sanitize_carriers(n, snakemake.config)

    reduce_network_size(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(
        snakemake.output.network,  # ty: ignore[possibly-missing-attribute]
        compression=COMPRESSION,
    )
