"""
Optimize the expansions of dummy storages for analysing flexibility needs.

This script runs a dummy storage optimization with the final layout resulting from the EVA.
It follows to the approach of running a market simulation with minimum requirements for renewable techs
and maximum limits for conventional tech, both given from the last EVA market session. At the same time
the network has to respect total curtailment targets.

"""

import logging
import pathlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from euflex.eu_flex_model.scripts.exceptions import (
    InfeasibilityError,
    maybe_print_solver_infeasibilities,
)
from euflex.eu_flex_model.scripts.extensions import (
    prepare_solver_instrumentation,
    run_optimization,
)
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.pypsa_helpers import (
    as_dense,
    must_run_requirement_mw,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    curtailment_targets_csv,
    get_snakemake,
    set_scenario_config,
)
from euflex.eu_flex_model.scripts.helpers.solver_helpers import (
    with_gurobi_exports,
)
from euflex.eu_flex_model.scripts.market_functions import (
    add_fbmc_constraint,
    get_must_run_generators,
    register_market_interconnection,
    write_model_to_lp_file,
)
from euflex.eu_flex_model.scripts.paths import get_path_config

logger = logging.getLogger(__name__)

COMPRESSION = {"zlib": True, "complevel": 4}
DISPATCH_TOLERANCE = 1e-6
EXCLUDED_CARRIERS = {
    "storage",
    "load spillage",
    "load shedding",
    "reserve shedding",
    "Electrolyser",
}


def classify_generator_carriers(
    carriers: pd.Series | list[str],
    renewable_carriers: list[str],
) -> tuple[set[str], set[str]]:
    """
    Classify generator carriers into renewable and conventional carriers.

    Carriers that are neither renewable nor in ``EXCLUDED_CARRIERS`` are
    conventional. The clustered carriers ``renewables`` and ``conventionals``
    are always included.

    Parameters
    ----------
    carriers : pd.Series | list[str]
        Generator carriers of the network.
    renewable_carriers : list[str]
        Renewable carrier names.

    Returns
    -------
    tuple[set[str], set[str]]
        Renewable and conventional carriers.
    """
    renewable = set(renewable_carriers) | {"renewables"}
    conventional = (set(carriers) - renewable - EXCLUDED_CARRIERS) | {"conventionals"}
    return renewable, conventional


def extract_clustered_generators(
    n: pypsa.Network,
    carriers: set[str],
    generator_type: str,
    dispatch: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate generators by bus for creating a fresh network.

    Parameters
    ----------
    n : pypsa.Network
        The input network with generators to cluster
    carriers : set[str]
        Carrier types to cluster.
    generator_type : str
        Type label for clustered generators (e.g., 'renewables', 'conventionals')
    dispatch : pd.DataFrame
        Generator dispatch with generators as columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Static data (``bus``, summed ``p_nom``) indexed by clustered name, and
        the capacity-weighted ``p_max_pu`` and summed dispatch with clustered
        names as columns.
    """
    gens = n.generators[n.generators.carrier.isin(list(carriers))]
    names = gens.bus + f" {generator_type}"
    static = gens.groupby(names, sort=False).agg(
        bus=("bus", "first"), p_nom=("p_nom", "sum")
    )
    p_max_pu = as_dense(n, "Generator", "p_max_pu", gens.index)
    capacity = p_max_pu.mul(gens.p_nom).T.groupby(names, sort=False).sum().T
    clustered_p_max_pu = capacity.div(static.p_nom)
    clustered_p_max_pu.loc[:, static.p_nom <= 0] = 0.0
    clustered_dispatch = dispatch[gens.index].T.groupby(names, sort=False).sum().T
    return static, clustered_p_max_pu, clustered_dispatch


def extract_clustered_storage_as_generators(
    n: pypsa.Network,
) -> tuple[list[str], list[float], pd.DataFrame]:
    """
    Extract and aggregate storage units as generators for creating a fresh network.

    Parameters
    ----------
    n : pypsa.Network
        The input network with storage units

    Returns
    -------
    tuple
        (storage_names, storage_p_nom_values, storage_dispatch)
    """
    if n.storage_units.empty:
        return [], [], pd.DataFrame()

    # n.storage_units_t.p (net dispatch = p_dispatch - p_store) may be missing from the
    # netCDF file when all net values equal the PyPSA default (0.0), because
    # export_to_netcdf skips columns that are identical to their default. Reconstruct it.
    if n.storage_units_t.p.empty:
        p_net = n.storage_units_t.p_dispatch - n.storage_units_t.p_store
    else:
        p_net = n.storage_units_t.p

    if p_net.empty:
        return [], [], pd.DataFrame()

    storage_buses = n.storage_units.bus.unique()
    dispatch = (
        p_net.T.groupby(n.storage_units.bus)
        .sum()
        .T.reindex(columns=storage_buses, fill_value=0.0)
    )

    storage_names = [f"{bus} Storage" for bus in storage_buses]
    storage_p_nom_values = dispatch.abs().max(axis=0).tolist()

    return storage_names, storage_p_nom_values, dispatch


def compute_conventional_floor(
    n: pypsa.Network,
    gens: pd.Index,
    technical_minimum_gens: pd.Index,
    must_run_gens: pd.Index,
    dispatch: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the conventional floor per bus from the input network.

    Per generator, the floor is the technical minimum
    ``p_nom * p_min_pu * status`` for generators in
    ``technical_minimum_gens`` and 0 otherwise, raised to the must-run
    requirement and capped at the generator's dispatch. The status is the
    solved commitment status for committable generators and 1 otherwise,
    since the day-ahead commitment of the other generators is already
    embedded in ``p_min_pu``.

    Parameters
    ----------
    n : pypsa.Network
        The solved input network.
    gens : pd.Index
        Conventional generators contributing to the floor.
    technical_minimum_gens : pd.Index
        Generators whose technical minimum contributes to the floor.
    must_run_gens : pd.Index
        Generators with a must-run requirement.
    dispatch : pd.DataFrame
        Non-negative generator dispatch with generators as columns.

    Returns
    -------
    pd.DataFrame
        Floor in MW, indexed by snapshot with buses as columns.

    Raises
    ------
    ValueError
        If a generator has a negative ``p_min_pu`` or must-run requirement.
    """
    p_min_pu = as_dense(n, "Generator", "p_min_pu", gens)
    p_min_pu.loc[:, gens.difference(technical_minimum_gens)] = 0.0
    must_run = must_run_requirement_mw(
        n, n.snapshots, must_run_gens.intersection(gens)
    ).reindex(columns=gens, fill_value=0.0)
    if (p_min_pu < 0).any().any() or (must_run < 0).any().any():
        raise ValueError(
            "Conventional floor requires non-negative p_min_pu and must-run "
            "requirements"
        )
    committable = gens[n.generators.loc[gens, "committable"]]
    status = n.generators_t.status.reindex(
        index=n.snapshots, columns=committable, fill_value=1.0
    ).reindex(columns=gens, fill_value=1.0)
    technical_minimum = p_min_pu.mul(n.generators.loc[gens, "p_nom"]).mul(status)
    floor = np.minimum(np.maximum(technical_minimum, must_run), dispatch[gens])
    return floor.T.groupby(n.generators.loc[gens, "bus"]).sum().T


def create_fresh_network_from_input(
    n_input: pypsa.Network,
    renewable_carriers: list,
    conventional_floor_carriers: list[str] | None = None,
) -> pypsa.Network:
    """
    Create a fresh network with only clustered components from input network.

    This function extracts buses, loads, links, and aggregated/clustered generators
    from the input network and builds a clean network for dummy storage optimization.
    Storage/hydro generators and load shedding keep their historic dispatch
    fixed (via p_set) so it can't be substituted by renewables or dummy
    storage; other conventionals only get
    an upper bound from historical levels, remainining free to be displaced by renewables/curtailment
    up to the conventional floor (technical minimum and must-run requirement).

    Parameters
    ----------
    n_input : pypsa.Network
        The input network (with fixed optimal capacities)
    renewable_carriers : list
        Renewable carrier names list
    conventional_floor_carriers : list[str] | None, default None
        Conventional carriers whose technical minimum contributes to the
        conventional floor. None selects all conventional carriers, an empty
        list leaves only the must-run requirement, which always applies.

    Returns
    -------
    pypsa.Network
        Fresh network with clustered components
    """
    n = pypsa.Network()
    n.set_snapshots(list(n_input.snapshots))
    n.snapshot_weightings = n_input.snapshot_weightings.copy()

    # Add buses (AC only) with all attributes
    ac_buses = n_input.buses[n_input.buses.carrier == "AC"]
    n.add("Bus", ac_buses.index, **ac_buses)

    # Get base loads
    base_load_p_set = n_input.get_switchable_as_dense("Load", "p_set")

    # Add Electrolyser consumption to loads (Electrolyser p is negative when consuming)
    # Electrolysers are flexible loads, not generators, so their consumption should be
    # added to demand rather than clustered with conventional generators
    electrolyser_gens = n_input.generators[n_input.generators.carrier == "Electrolyser"]
    if not electrolyser_gens.empty:
        # Get electrolyser dispatch and group by bus
        electrolyser_dispatch = n_input.generators_t.p[electrolyser_gens.index]
        # Electrolyser p is negative when consuming, so negate it to get positive consumption
        electrolyser_consumption_by_bus = (
            -electrolyser_dispatch.T.groupby(electrolyser_gens.bus).sum().T
        )

        # Map bus names to load names (loads indexed by bus have load index matching the bus)
        # base_load_p_set columns are load indices (e.g., "AT00 load")
        # electrolyser_consumption_by_bus columns are bus names (e.g., "AT00")
        # Need to map bus -> load index
        bus_to_load = n_input.loads.bus.to_dict()  # {load_index: bus_name}
        load_to_bus = {v: k for k, v in bus_to_load.items()}  # {bus_name: load_index}

        # Rename electrolyser columns from bus names to load names
        electrolyser_consumption_by_load = electrolyser_consumption_by_bus.rename(
            columns=load_to_bus
        )

        # Add to loads (pandas will align by column names automatically)
        base_load_p_set = base_load_p_set.add(
            electrolyser_consumption_by_load, fill_value=0
        )

        total_consumption = electrolyser_consumption_by_bus.sum().sum()
        logging.info(
            f"Added Electrolyser consumption to loads "
            f"(total: {total_consumption:.2f} MWh across all buses)"
        )

    n.add(
        "Load",
        list(n_input.loads.index),
        **n_input.loads.drop(columns="p_set"),
        p_set=base_load_p_set,
    )
    # Prepare link attributes with time-varying data taking precedence
    link_static = n_input.links.copy()
    # Reindex p0/p1 columns to match links.index order: PyPSA's n.add() uses
    # pd.Index.equals() which is order-sensitive, while optimization results are
    # stored alphabetically (diverging when FBMC virtual links are appended late).
    link_time_varying = {
        "p0": n_input.links_t.p0[n_input.links.index],
        "p1": n_input.links_t.p1[n_input.links.index],
    }

    # Use get_switchable_as_dense for attributes that may be time-varying
    for attr in ["p_min_pu", "p_max_pu"]:
        if not getattr(n_input.links_t, attr).empty:
            link_time_varying[attr] = n_input.get_switchable_as_dense("Link", attr)
            link_static = link_static.drop(columns=[attr])

    n.add("Link", list(n_input.links.index), **link_static, **link_time_varying)

    added_carriers = (
        set(n.links.carrier)
        | set(n.buses.carrier)
        | set(n.loads.carrier)
        | set(n.generators.carrier)
    )
    added_carriers = list(added_carriers - {""})
    n.add("Carrier", added_carriers, **n_input.carriers.loc[added_carriers])

    # Conventional generators (e.g. nuclear) are aggregated per bus below, but
    # their technical minimum and must-run requirement are summed per bus and
    # applied as a p_min_pu floor on the aggregated cluster, so renewables
    # can't substitute for them under a tight curtailment target.
    must_run_gens = get_must_run_generators(n_input).intersection(
        n_input.generators.index
    )

    renewable_carriers_set, conventional_carriers = classify_generator_carriers(
        n_input.generators.carrier, renewable_carriers
    )

    conv_gens = n_input.generators.index[
        n_input.generators.carrier.isin(conventional_carriers)
    ]
    technical_minimum_gens = conv_gens
    if conventional_floor_carriers is not None:
        invalid_carriers = set(conventional_floor_carriers) - set(
            n_input.generators.carrier[conv_gens]
        )
        if invalid_carriers:
            raise ValueError(
                "conventional_floor_carriers contains carriers that are not "
                "conventional carriers of the input network: "
                f"{sorted(invalid_carriers)}"
            )
        technical_minimum_gens = conv_gens[
            n_input.generators.carrier[conv_gens].isin(conventional_floor_carriers)
        ]

    ren_static, ren_p_max_pu, ren_dispatch = extract_clustered_generators(
        n_input, renewable_carriers_set, "renewables", n_input.generators_t.p
    )
    if not ren_static.empty:
        n.add(
            "Generator",
            ren_static.index,
            **ren_static,
            p_max_pu=ren_p_max_pu,
            p_min_pu=0,
            marginal_cost=0,
            carrier="renewables",
        )
        n.generators_t.p[ren_static.index] = ren_dispatch

    # Extract and add clustered conventionals, including floored generators
    # (e.g. nuclear) - not split into their own cluster, just floored below.
    dispatch = n_input.generators_t.p.copy()
    if (dispatch[conv_gens] < -DISPATCH_TOLERANCE).any().any():
        raise ValueError("Conventional dispatch must be non-negative")
    dispatch[conv_gens] = dispatch[conv_gens].clip(lower=0.0)
    conv_static, conv_p_max_pu, conv_dispatch = extract_clustered_generators(
        n_input, conventional_carriers, "conventionals", dispatch
    )
    if not conv_static.empty:
        n.add(
            "Generator",
            conv_static.index,
            **conv_static,
            p_max_pu=conv_p_max_pu,
            p_min_pu=0,
            marginal_cost=0,
            carrier="conventionals",
        )
        n.generators_t.p[conv_static.index] = conv_dispatch
        bus_to_name = pd.Series(conv_static.index, index=conv_static.bus)
        floor = compute_conventional_floor(
            n_input, conv_gens, technical_minimum_gens, must_run_gens, dispatch
        ).rename(columns=bus_to_name)
        p_nom = conv_static.p_nom[floor.columns]
        floored = floor.columns[p_nom > 0]
        n.generators_t.p_min_pu[floored] = floor[floored] / p_nom[floored]

    # Extract and add storage as generators
    storage_names, storage_p_nom, storage_dispatch = (
        extract_clustered_storage_as_generators(n_input)
    )

    if storage_names:
        storage_buses = [name.replace(" Storage", "") for name in storage_names]
        n.add(
            "Generator",
            storage_names,
            p_nom=storage_p_nom,
            p_max_pu=1,
            p_min_pu=-1,
            marginal_cost=0,
            carrier="storage",
            bus=storage_buses,
        )
        n.generators_t.p_set[storage_names] = storage_dispatch

    # Carry over the original load-shedding generators (one per bus already,
    # so no clustering needed), fixed to their historic dispatch so that
    # dummy storage cannot replace load shedding.
    load_shedding = n_input.generators[n_input.generators.carrier == "load shedding"]
    if not load_shedding.empty:
        n.add(
            "Generator",
            load_shedding.index,
            bus=load_shedding.bus,
            carrier="load shedding",
            marginal_cost=load_shedding.marginal_cost,
            p_nom=load_shedding.p_nom,
        )
        n.generators_t.p_set[load_shedding.index] = n_input.generators_t.p.reindex(
            index=n_input.snapshots, columns=load_shedding.index, fill_value=0.0
        )

    # Add missing carriers for clustered components
    for carrier, color, nice_name in [
        ("renewables", "#00AA00", "Var. Renewables"),
        ("conventionals", "#8B7B9B", "Conventionals"),
        ("storage", "#647585", "Storage"),
        ("load shedding", "#dd2e23", "Load shedding"),
    ]:
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier, color=color, nice_name=nice_name)

    # Copy metadata
    n.meta = n_input.meta.copy() if hasattr(n_input, "meta") else {}

    return n


def mock_network_needs(
    n: pypsa.Network,
    renewable_carriers: list[str],
    max_percentage: float = 0.3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly calculate network needs to include in the dummy storage RES availability.
    This mock the TSO/DSO curtailment data for network needs. The needs of a bus
    are capped at its RES curtailment plus the headroom of its conventionals, so
    that ``adapt_eva_outputs_from_network_needs`` can always compensate them.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object
    renewable_carriers : list
        List of renewable carrier names
    max_percentage : float
        Maximum percentage of RES availability to use (default 30%)
    random_seed : int
        Random seed for reproducible results (default 42)

    Returns
    -------
    pd.DataFrame
        DataFrame with network needs per bidding zone
    """
    logger.warning(
        "Network needs are random mock data (up to "
        f"{max_percentage:.0%} of the RES availability, capped at the local "
        "conventional headroom), not TSO/DSO input data. "
        "Dummy storage results are not meaningful for network needs."
    )

    # Get renewable generators
    renewables = n.generators[n.generators.carrier.isin(renewable_carriers)]

    # Calculate RES availability per bidding zone
    # RES availability = p_nom * p_max_pu for each generator
    renewable_buses = renewables.bus.unique()
    p_max_pu = as_dense(n, "Generator", "p_max_pu", renewables.index)
    res_availability = (
        renewables.p_nom.mul(p_max_pu)
        .T.groupby(renewables.bus)
        .sum()
        .loc[renewable_buses]
        .T
    )

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # Create random percentage timeseries for each bidding zone
    # Each BZ gets its own random timeseries between 0 and max_percentage
    n_hours = len(res_availability)
    n_zones = len(res_availability.columns)

    # Generate random percentages (0 to max_percentage) for each hour and BZ
    random_percentages = rng.uniform(0, max_percentage, size=(n_hours, n_zones))

    # Create DataFrame with same structure as res_availability
    percentage_df = pd.DataFrame(
        random_percentages,
        index=res_availability.index,
        columns=res_availability.columns,
    )

    res_dispatch = (
        n.generators_t.p.reindex(columns=renewables.index, fill_value=0.0)
        .T.groupby(renewables.bus)
        .sum()
        .loc[renewable_buses]
        .T
    )
    bus_headroom = (
        conventional_headroom(n).T.groupby(n.generators.bus).sum().T
    ).reindex(columns=renewable_buses, fill_value=0.0)
    cap = (res_availability - res_dispatch + bus_headroom).clip(lower=0.0)
    network_needs = np.floor(np.minimum(res_availability * percentage_df, cap))

    # Reset index to make hour as a column
    network_needs = network_needs.reset_index()
    network_needs.index.name = None

    # Rename the time column to match the CSV format (just index numbers)
    network_needs = network_needs.drop("snapshot", axis=1)
    network_needs.index = range(len(network_needs))

    return network_needs


def create_per_bz_curtailment_targets(
    global_curtailment_value: float,
    renewable_buses: pd.Index,
    curtailment_targets_per_bz: bool,
    curtailment_target: float,
) -> xr.DataArray:
    """
    Create per-bidding-zone curtailment targets using xarray broadcasting.

    This function implements the simplified approach suggested in PR review:
    1. Extract per bus targets from CSV if available
    2. Broadcast the scalar curtailment_target to contain BZ coordinates
    3. Update/overwrite values with per-BZ targets where available

    Parameters
    ----------
    global_curtailment_value : float
        The global/default curtailment value
    renewable_buses : pd.Index
        Renewable buses (bidding zones) index
    curtailment_targets_per_bz : bool
        Whether to use per-BZ targets from CSV
    curtailment_target : str
        The curtailment target variable name

    Returns
    -------
    xr.DataArray
        DataArray with curtailment targets per bidding zone
    """
    # Step 1: Create broadcast curtailment target with BZ coordinates
    curtailment_target_array = xr.DataArray(
        global_curtailment_value, coords=[renewable_buses], dims=["bus"]
    )

    # Step 2: Update with per-BZ targets if enabled
    if curtailment_targets_per_bz:
        logging.info("Per-BZ curtailment targets enabled - reading from CSV")
        csv_path = curtailment_targets_csv()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Curtailment targets CSV file not found: {csv_path}"
            )

        ct_df = pd.read_csv(csv_path)
        available_targets = sorted(ct_df["variable"].unique())
        if curtailment_target not in ct_df["variable"].values:
            raise ValueError(
                f"\n\nCurtailment target variable '{curtailment_target}' not found in CSV file '{csv_path}'\n"
                f"Available curtailment targets in CSV: {available_targets}\n\n"
                f"This error should have been caught earlier by Snakemake validation. "
                f"If you're seeing this, please report it as a bug.\n"
            )

        # Extract per-BZ targets
        ct_data = (
            ct_df[ct_df["variable"] == curtailment_target]
            .set_index("BZ")["value"]
            .to_dict()
        )
        logging.info(f"Extracted per-BZ targets: {ct_data}")

        # Update array values where per-BZ targets are available
        for bz in renewable_buses:
            if bz in ct_data:
                new_value = float(ct_data[bz])
                curtailment_target_array.loc[dict(bus=bz)] = new_value
            else:
                logging.warning(
                    f"No per-BZ target found for {bz}, using global value {global_curtailment_value}"
                )
    else:
        logging.info(
            f"Using global curtailment target: {global_curtailment_value} for all buses"
        )

    return curtailment_target_array


def extra_functionality(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Define additional functionality to be executed during the optimization process.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    sns : pd.Index
        The snapshots to be processed.
    """

    if hasattr(n, "_market_interconnection"):
        if (
            n._market_interconnection == "fbmc"
            and snakemake.params.shared_across_countries
        ):
            add_fbmc_constraint(n, sns)
        else:
            logger.info("FBMC constraints are not added")
    else:
        logger.info(
            "Network has not attribute _market_interconnection. FBMC constraints not added."
        )


def conventional_headroom(n: pypsa.Network) -> pd.DataFrame:
    """Headroom ``p_nom * p_max_pu - p`` of the clustered conventional generators."""
    conventionals = n.generators[n.generators.carrier == "conventionals"]
    return (
        as_dense(n, "Generator", "p_max_pu", conventionals.index) * conventionals.p_nom
        - n.generators_t.p[conventionals.index]
    ).clip(lower=0.0)


def adapt_eva_outputs_from_network_needs(
    n: pypsa.Network, renewable_carriers: list[str]
) -> pypsa.Network:
    """
    Adapt EVA outputs based on network needs for clustered renewables.
    This function adjusts renewable generation outputs based on network needs,
    ensuring that curtailment meets at least the network needs, and compensates
    the additional curtailment with the clustered conventional generators of the
    same bus, proportionally to their headroom ``p_nom * p_max_pu - p``.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object containing solved dispatch results

    renewable_carriers : list[str]
        List of renewable carrier names

    Returns
    -------
    pypsa.Network
        Modified network with adjusted generator outputs

    Raises
    ------
    ValueError
        If the conventional headroom of a bus cannot compensate its additional
        curtailment.
    """
    network_needs = n.buses_t["network_needs"]
    renewables = n.generators[
        n.generators.carrier.isin(renewable_carriers)
        & n.generators.bus.isin(network_needs.columns)
    ]
    availability = (
        as_dense(n, "Generator", "p_max_pu", renewables.index) * renewables.p_nom
    )
    dispatch = n.generators_t.p.reindex(columns=renewables.index, fill_value=0.0)
    curtailment = availability - dispatch
    needs = pd.DataFrame(
        network_needs[renewables.bus].values,
        index=n.snapshots,
        columns=renewables.index,
    )
    adjusted_curtailment = np.maximum(curtailment, needs)
    n.generators_t.p[renewables.index] = availability - adjusted_curtailment
    n.buses_t["original_curtailment_timeseries"] = (
        curtailment.T.groupby(renewables.bus).sum().T
    )
    n.buses_t["adjusted_curtailment_timeseries"] = (
        adjusted_curtailment.T.groupby(renewables.bus).sum().T
    )
    additional = (adjusted_curtailment - curtailment).T.groupby(renewables.bus).sum().T

    conventionals = n.generators[n.generators.carrier == "conventionals"]
    conventional_dispatch = n.generators_t.p[conventionals.index]
    headroom = conventional_headroom(n)
    bus_headroom = headroom.T.groupby(conventionals.bus).sum().T
    shortfall = additional - bus_headroom.reindex(
        columns=additional.columns, fill_value=0.0
    )
    short_buses = shortfall.columns[(shortfall > DISPATCH_TOLERANCE).any()]
    if not short_buses.empty:
        raise ValueError(
            "Conventional headroom cannot compensate the network needs curtailment "
            f"at buses {short_buses.tolist()}, "
            f"maximum shortfall {shortfall.max().max():.2f} MW"
        )
    share = headroom.div(bus_headroom[conventionals.bus].values).where(
        headroom > 0, 0.0
    )
    n.generators_t.p[conventionals.index] = conventional_dispatch + share.mul(
        additional.reindex(columns=conventionals.bus, fill_value=0.0).values
    )

    return n


def net_dummy_storage_flows(n: pypsa.Network, dummies: pd.Index) -> None:
    """
    Net the charging and discharging of solved dummy storages per snapshot.

    Without losses, simultaneous charging and discharging is degenerate.
    Netting removes it without changing the state of charge or the net
    dispatch.

    Parameters
    ----------
    n : pypsa.Network
        Solved network.
    dummies : pd.Index
        Dummy storage units.

    Raises
    ------
    ValueError
        If a dummy storage has efficiencies other than 1 or a standing loss.
    """
    for attr, lossless_value in [
        ("efficiency_store", 1.0),
        ("efficiency_dispatch", 1.0),
        ("standing_loss", 0.0),
    ]:
        values = as_dense(n, "StorageUnit", attr, dummies)
        if not (values == lossless_value).all().all():
            raise ValueError(
                f"Netting dummy storage flows requires {attr} == {lossless_value}"
            )
    p = n.storage_units_t.p_dispatch[dummies] - n.storage_units_t.p_store[dummies]
    n.storage_units_t.p_dispatch[dummies] = p.clip(lower=0.0)
    n.storage_units_t.p_store[dummies] = (-p).clip(lower=0.0)


def add_dummy_storage_hull_constraints(
    n: pypsa.Network, dummies: pd.Index, max_hours: float
) -> None:
    """
    Add the convex hull constraints of cyclic dummy storages.

    The constraints follow the H-representation of Pozo (Model 3). They tighten
    the linear relaxation against simultaneous charging and discharging.

    Parameters
    ----------
    n : pypsa.Network
        Network with a created optimization model.
    dummies : pd.Index
        Dummy storage units.
    max_hours : float
        Energy-to-power ratio of the dummy storages.
    """
    m = n.model
    charge = m.variables["StorageUnit-p_store"].loc[:, dummies]
    discharge = m.variables["StorageUnit-p_dispatch"].loc[:, dummies]
    soc_before = (
        m.variables["StorageUnit-state_of_charge"].loc[:, dummies].roll(snapshot=1)
    )
    capacity = (
        m.variables["StorageUnit-p_nom"]
        .loc[dummies]
        .rename({"StorageUnit-ext": "StorageUnit"})
    )
    dt = n.snapshot_weightings.stores.to_xarray()
    if max_hours <= 0 or bool((dt <= 0).any()) or bool((dt > max_hours).any()):
        raise ValueError(
            "Dummy storage hull requires 0 < snapshot duration <= max_hours, "
            f"got max_hours={max_hours}"
        )
    m.add_constraints(
        charge + discharge <= capacity, name="StorageUnit-dummy_hull_power"
    )
    m.add_constraints(
        soc_before + charge * dt <= max_hours * capacity,
        name="StorageUnit-dummy_hull_charge",
    )
    m.add_constraints(
        soc_before - discharge * dt >= 0, name="StorageUnit-dummy_hull_discharge"
    )


def get_dummy_storage_solver_options(
    solving: dict[str, Any],
    ds_solver_options: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """
    Select the solver options for the dummy storage optimization.

    The options key mapped to the configured solver in ``ds_solver_options``
    is used, independent of the configured options key, so that the solution
    is a vertex solution. The thread count of the configured options is kept.
    Solvers without a mapping use the configured options.

    Parameters
    ----------
    solving : dict[str, Any]
        The ``solving`` section of the configuration.
    ds_solver_options : dict[str, str]
        Mapping of solver name to options key in ``solving.solver_options``.

    Returns
    -------
    tuple[str, dict[str, Any]]
        Solver name and solver options.

    Raises
    ------
    KeyError
        If the mapped options key does not exist in ``solving.solver_options``.
    """
    solver_name = str(solving["solver"]["name"])
    option_key = solving["solver"]["options"]
    user_options = solving["solver_options"][option_key]
    if solver_name not in ds_solver_options:
        logger.info(f"Using solver options '{option_key}' for dummy storage")
        return solver_name, user_options.copy()
    ds_key = ds_solver_options[solver_name]
    if ds_key not in solving["solver_options"]:
        raise KeyError(
            f"Dummy storage solver options '{ds_key}' for solver '{solver_name}' "
            "not found in solving.solver_options"
        )
    options = solving["solver_options"][ds_key].copy()
    if "threads" in user_options:
        options["threads"] = user_options["threads"]
    logger.info(f"Using solver options '{ds_key}' for dummy storage")
    return solver_name, options


def optimize_dummy_storage(
    network: pypsa.Network,
    max_hours: float,
    curtailment_target: float,
    renewable_carriers: list[str],
    target_year: int,
    climate_year: int,
    eraa_year: int,
    config: dict[str, Any],
    shared_across_countries: bool = True,
    minimize_system_costs: bool = True,
    include_network_needs: bool = False,
    curtailment_targets_per_bz: bool = False,
    enable_load_spillage: bool = True,
    market_interconnection: str = "ntc",
    conventional_floor_carriers: list[str] | None = None,
    log_fn: str | None = None,
) -> tuple[pypsa.Network, str, str]:
    """
    Optimize dummy storage capacities for flexibility assessment.

    Parameters
    ----------
    network : pypsa.Network
        The network to optimize
    max_hours : float
        Maximum storage duration in hours
    curtailment_target : float
        Curtailment target (0-1, where 0.2 means 20% curtailment allowed)
    renewable_carriers : list
        Renewable carrier names list
    climate_year : int
        The climate year for which the flow-based data is relevant.
    target_year : int
        The target year for which the flow-based data is relevant.
    eraa_year: int
        ERAA data year
    shared_across_countries : bool, default True
        Whether to allow cross-border sharing
    minimize_system_costs : bool, default True
        Whether to minimize system costs or storage capacity
    include_network_needs : bool, default False
        Whether to include (mock) network needs in the optimization
    curtailment_targets_per_bz : bool, default False
        Whether to use per-bidding-zone curtailment targets from CSV file instead of global target
    enable_load_spillage : bool, default True
        Whether to include load spillage
    market_interconnection : str, default ntc
        Market interconnection type to determine constraint setting mechanism
    conventional_floor_carriers : list[str] | None, default None
        Conventional carriers whose technical minimum contributes to the
        conventional floor. None selects all conventional carriers, an empty
        list leaves only the must-run requirement, which always applies.
    log_fn : str | None, default None
        Path of the solver log file. Not used for Gurobi if Gurobi logs are
        exported.
    config : dict
        Configuration

    Returns
    -------
    tuple
        (optimized_network, status, condition)
    """
    # Fix optimal capacities on input network
    n_input = network.copy()
    n_input.optimize.fix_optimal_capacities()

    # Create fresh network with clustered components
    # Note: The fresh network inherits all market coupling topology (CORE/NORDIC buses,
    # link attributes, NTC limits) from the input network that was already processed
    # by build_base_network, so we don't need to call adapt_network_for_market_coupling
    logging.info("Creating fresh network with clustered generators and storage")
    n = create_fresh_network_from_input(
        n_input,
        renewable_carriers,
        conventional_floor_carriers=conventional_floor_carriers,
    )
    renewable_carriers_set, conventional_carriers_set = classify_generator_carriers(
        n.generators.carrier, renewable_carriers
    )

    # Register market interconnection (FBMC flow-based data if needed)
    logging.info(f"Registering market interconnection: {market_interconnection}")
    register_market_interconnection(
        n,
        market_interconnection,
        int(climate_year),
        int(target_year),
        config,
        int(eraa_year),
    )

    # Generate and add network needs
    if include_network_needs:
        logging.info("Generating network needs")
        network_needs = mock_network_needs(
            n, list(renewable_carriers_set), max_percentage=0.8
        )
        n.buses_t.network_needs = network_needs

        # Adapt generator dispatch for network needs
        logging.info("Adapting generator dispatch for network needs")
        n = adapt_eva_outputs_from_network_needs(
            n, renewable_carriers=list(renewable_carriers_set)
        )

    # Add dummy storages
    logging.info("Adding dummy storage units")
    buses = n.buses[n.buses.carrier == "AC"].index
    dummies = n.add(
        "StorageUnit",
        name=buses,
        suffix=" dummy storage",
        bus=buses,
        p_nom_extendable=True,
        max_hours=float(max_hours),
        capital_cost=1000 * len(n.snapshots) / 8760,
        carrier="Dummy Storage",
        cyclic_state_of_charge=True,
    )
    n.add("Carrier", "Dummy Storage", color="#005fb8", nice_name="Dummy Storage")

    # Add load spillage (optional)
    if enable_load_spillage and "load spillage" not in n.carriers.index:
        buses_all = n.buses.index
        n.add(
            "Generator",
            name=list(buses_all),
            suffix=" load spillage",
            bus=buses_all,
            carrier="load spillage",
            marginal_cost=-4500,
            p_min_pu=-1,
            p_max_pu=0,
            p_nom=10e6,
            overwrite=True,
        )
        n.add(
            "Carrier", name="load spillage", color="#df8e23", nice_name="Load spillage"
        )

    # Create optimization model
    logging.info("Creating optimization model")
    n.optimize.create_model()  # ty: ignore[missing-argument]
    add_dummy_storage_hull_constraints(n, dummies, float(max_hours))

    # Define generator sets and calculate availability
    m = n.model
    vrs = m.variables
    gens = n.generators

    renewables = gens[gens.carrier.isin(list(renewable_carriers_set))]
    conventionals = gens[gens.carrier.isin(list(conventional_carriers_set))]
    renewable_buses = renewables.bus.unique()

    p_max_pu = as_dense(n, "Generator", "p_max_pu", renewables.index)
    availability = (
        renewables.p_nom.mul(p_max_pu)
        .T.groupby(renewables.bus)
        .sum()
        .loc[renewable_buses]
        .T
    )
    renewable_supply = (
        vrs["Generator-p"]
        .loc[:, renewables.index]
        .groupby(renewables.bus)
        .sum()
        .loc[renewable_buses]
    )

    # Parse curtailment target
    if isinstance(curtailment_target, str) and curtailment_target.startswith("ct"):
        ct_index = int(curtailment_target[2:])
        curtailment_targets_list = config["eva"]["dummy_storage_optimization"][
            "curtailment_targets"
        ]
        if ct_index > len(curtailment_targets_list):
            raise ValueError(
                f"Curtailment target index {ct_index} exceeds available targets in config"
            )
        global_curtailment_value = curtailment_targets_list[ct_index - 1]
    else:
        global_curtailment_value = float(curtailment_target)

    curtailment_target_array = create_per_bz_curtailment_targets(
        global_curtailment_value,
        renewable_buses,
        curtailment_targets_per_bz,
        curtailment_target,
    )

    # Add constraints
    logging.info("Adding optimization constraints")

    # Curtailment constraint
    required_generation = (1 - curtailment_target_array) * availability.sum()
    if include_network_needs:
        # Network needs force mandatory curtailment that shouldn't count against voluntary curtailment budget
        additional_curtailment_per_bus = (
            n.buses_t.adjusted_curtailment_timeseries.sum()
            - n.buses_t.original_curtailment_timeseries.sum()
        ).loc[renewable_buses]
        required_generation = (
            required_generation
            - xr.DataArray(
                additional_curtailment_per_bus.values,
                coords=[renewable_buses],
                dims=["bus"],
            )
        ).clip(min=0)

    m.add_constraints(
        renewable_supply.sum("snapshot") >= required_generation,
        name="Bus-total-curtailment",
    )

    # Renewable lower bound constraints
    m.add_constraints(
        vrs["Generator-p"].loc[:, renewables.index]
        >= n.generators_t.p[renewables.index],
        name="Generator-renewable_lower_bound",
    )

    # Conventional upper bound constraints
    m.add_constraints(
        vrs["Generator-p"].loc[:, conventionals.index]
        <= n.generators_t.p[conventionals.index],
        name="Generator-conventional_upper_bound",
    )

    # Fixed link dispatch if not shared across countries
    if not shared_across_countries and len(n.links) > 0:
        m.add_constraints(vrs["Link-p"] == n.links_t.p0, name="Link-fixed_dispatch")

    # Set objective function
    if not minimize_system_costs:
        # Overwriting the objective drops PyPSA's default marginal-cost terms, so
        # load spillage (and any other slack with a marginal cost) would otherwise
        # be free and preferred over installing dummy storage. Re-add its cost here.
        objective = (
            vrs["StorageUnit-p_nom"]
            .loc[dummies]  # ty: ignore[invalid-argument-type]
            .sum()
        )

        weighting = n.snapshot_weightings.objective

        load_spillage_gens = gens.index[gens.carrier == "load spillage"]
        if not load_spillage_gens.empty:
            load_spillage_penalty = config["eva"]["dummy_storage_optimization"][
                "load_spillage_penalty"
            ]
            weighted_penalty = pd.DataFrame(
                float(load_spillage_penalty),
                index=n.snapshots,
                columns=load_spillage_gens,
            ).mul(weighting, axis=0)
            # Load-spillage generators only ever dispatch negatively (they consume
            # energy), so negate to get the spilled MWh the penalty applies to.
            objective = (
                objective
                + (
                    -vrs["Generator-p"].loc[:, load_spillage_gens] * weighted_penalty
                ).sum()
            )

        n.model.objective = objective

    # Solve
    solver_name, solver_options = get_dummy_storage_solver_options(
        config["solving"],
        config["eva"]["dummy_storage_optimization"]["solver_options"],
    )

    if solver_name == "gurobi":
        solver_options = with_gurobi_exports(
            solver_name=solver_name,
            solver_options=solver_options,
            export_dir="resources/solver_exports",
            stem=(
                f"dummy_storage_{target_year}_{climate_year}_"
                f"{max_hours}_{curtailment_target}"
            ),
            export_log=config["eva"]["export_gurobi_logs"],
            export_model=config["eva"]["export_gurobi_mps"],
        )
        if config["eva"]["export_gurobi_logs"]:
            log_fn = None

    logging.info(f"Solving with {solver_name}")

    status, condition = run_optimization(
        optimize_callable=n.optimize.solve_model,
        optimize_kwargs=dict(
            solver_name=solver_name,
            solver_options=solver_options,
            extra_functionality=extra_functionality,
            log_fn=log_fn,
        ),
        network=n,
        module="dummy_storage",
    )

    if status == "ok":
        net_dummy_storage_flows(n, dummies)

    return n, status, condition


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="run_dummy_storage_optimization",
        configfiles=[
            pathlib.Path("config", "test", "config.eva-quick-test.yaml"),
            pathlib.Path("config", "config.eva-debug-infeasibility.yaml"),
        ],
        run="eva-quick-test",
        ty=2030,
        cy=2010,
        ct="ct2",
        mh=2,
    )
    configure_logging(snakemake)

    # Suppress verbose matplotlib font manager debug logs unless logging level is DEBUG
    if logging.getLogger().level != logging.DEBUG:
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    set_scenario_config(snakemake)

    config = snakemake.config

    # Check if dummy storage optimization is enabled
    if not snakemake.config["eva"]["dummy_storage_optimization"]["enable"]:
        n = pypsa.Network(
            snakemake.input.network  # ty: ignore[possibly-missing-attribute]
        )
        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(
            snakemake.output.network,  # ty: ignore[possibly-missing-attribute]
            compression=COMPRESSION,
        )

        # Create empty LP file if needed
        with (
            open(
                snakemake.output.mathematical_model,  # ty: ignore[possibly-missing-attribute]
                "w",
            ) as f
        ):
            f.write("# Dummy storage optimization is disabled\n")

        logging.info(
            "Dummy storage optimization is disabled. Skipping optimization and returning original network."
        )
        sys.exit(0)

    network_input = pypsa.Network(
        snakemake.input.network,  # ty: ignore[possibly-missing-attribute]
    )

    run_name = snakemake.config["run"]["name"]
    path_config = get_path_config()
    run_dir = Path(path_config.working_dir, run_name)

    # Build infeasibility LP path so it is available if the solve fails
    wc = snakemake.wildcards
    target_year = wc.ty  # ty: ignore[possibly-missing-attribute]
    climate_year = wc.cy  # ty: ignore[possibly-missing-attribute]
    energy_power_ration = wc.mh  # ty: ignore[possibly-missing-attribute]
    curtailment_target = wc.ct  # ty: ignore[possibly-missing-attribute]
    infeasibility_lp_dir = run_dir / "resources" / "market_models"
    infeasibility_lp_dir.mkdir(parents=True, exist_ok=True)
    infeasibility_lp_file = (
        infeasibility_lp_dir
        / f"{snakemake.rule}_{target_year}_{climate_year}_{energy_power_ration}_{curtailment_target}_infeasible.lp"
    )

    prepare_solver_instrumentation(network_input, snakemake, run_dir / "benchmarks")

    n, status, condition = optimize_dummy_storage(
        network=network_input,
        max_hours=snakemake.wildcards.mh,  # ty: ignore[possibly-missing-attribute]
        curtailment_target=snakemake.wildcards.ct,  # ty: ignore[possibly-missing-attribute]
        renewable_carriers=snakemake.params.renewable_carriers,
        target_year=snakemake.wildcards.ty,  # ty: ignore[possibly-missing-attribute]
        climate_year=snakemake.wildcards.cy,  # ty: ignore[possibly-missing-attribute]
        eraa_year=snakemake.params.eraa_year,
        config=config,
        shared_across_countries=snakemake.params.shared_across_countries,
        minimize_system_costs=snakemake.params.minimize_system_costs,
        include_network_needs=snakemake.params.add_network_needs,
        curtailment_targets_per_bz=snakemake.params.curtailment_targets_per_bz,
        enable_load_spillage=snakemake.params.enable_load_spillage,
        market_interconnection=snakemake.params.market_interconnection,
        conventional_floor_carriers=snakemake.params.conventional_floor_carriers,
        log_fn=snakemake.log.solver,
    )
    if status != "ok":
        solver_name = config["solving"]["solver"]["name"]
        maybe_print_solver_infeasibilities(
            getattr(n, "model", None), solver_name, logger
        )
        logger.info("Writing .lp file for the failing model...")
        write_model_to_lp_file(n, infeasibility_lp_file.as_posix(), True, False)
        logger.info(f".lp file ready in {infeasibility_lp_file}")
        logger.error(
            "INFEASIBILITY DETECTED during dummy storage optimisation (status=%s, condition=%s)",
            status,
            condition,
        )
        raise InfeasibilityError(
            message=(
                "Dummy storage optimisation failed: "
                f"status={status}, condition={condition}."
            ),
            source="run_dummy_storage_optimization",
            details={
                "status": status,
                "condition": str(condition),
                "solver_name": solver_name,
            },
        )

    export_models = (
        snakemake.config["eva"]["export_lp_models"] or snakemake.params.export_models
    )

    if export_models:
        write_model_to_lp_file(
            n,
            snakemake.output.mathematical_model,
            generate_lp=True,
        )
        logger.info(
            f"Dummy storage model exported to {snakemake.output.mathematical_model}"
        )
    else:
        with open(snakemake.output.mathematical_model, "w") as f:
            f.write(
                "! LP export not requested (set eva.export_lp_models or "
                "eva.dummy_storage_optimization.export_models: true)\n"
            )

    n.optimize.fix_optimal_capacities()  # Fix optimal dummy storage capacities after optimization
    existing_meta = getattr(n, "meta", {}) or {}

    existing_meta["config"] = snakemake.config
    existing_meta["wildcards"] = dict(snakemake.wildcards)

    n.meta = existing_meta
    n.export_to_netcdf(
        snakemake.output.network,  # ty: ignore[possibly-missing-attribute]
        compression=COMPRESSION,
    )
