import contextlib
import gc
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import filelock
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from linopy.expressions import merge
from pypsa import Network

# Import helpers to apply the meshed_threshold patch before any linopy.Model is created
# (The patch is applied automatically when any helpers submodule is imported via helpers/__init__.py)
from euflex.eu_flex_model.scripts.calculate_model_outputs import (
    calculate_residual_load_df,
)
from euflex.eu_flex_model.scripts.exceptions import (
    InfeasibilityError,
    maybe_print_solver_infeasibilities,
)
from euflex.eu_flex_model.scripts.extensions import run_optimization
from euflex.eu_flex_model.scripts.fbmc_ntc_helpers import (
    register_market_interconnection,
)
from euflex.eu_flex_model.scripts.helpers.pypsa_helpers import (
    as_dense,
    must_run_requirement_mw,
)
from euflex.eu_flex_model.scripts.helpers.solver_helpers import (
    sanitize_filename,
    with_gurobi_exports,
)
from euflex.eu_flex_model.scripts.uc_helpers import (
    fix_uc_min_uptime_downtime_constraints_at_rolling_horizon_boundary,
    fix_uc_transition_constraints_at_rolling_horizon_boundary,
)

logger = logging.getLogger(__name__)

NO_RESERVES_DEBUG_MESSAGE = "No reserve parameters found in day-ahead network buses"

# Slack variable absorbing must-run capacity that the dispatch cannot serve
MUST_RUN_SLACK_VAR = "Generator-must_run_slack"

#: Suffixes marking a bus column as a reserve *parameter* rather than a reserve
#: requirement. Order matters: :func:`reserve_type_from_column` strips the first
#: one it matches.
RESERVE_PARAMETER_SUFFIXES: tuple[str, ...] = (
    "_full_activation_time",
    "_min_duration",
    "_id_blocked",
    "_penalty_cost",
    "_reserve_blocking_method",
)


def reserve_type_from_column(column: str) -> str:
    """
    Derive the reserve type from a reserve parameter bus column.

    Parameters
    ----------
    column : str
        Bus column name, e.g. ``reserve_FCR_min_duration``.

    Returns
    -------
    str
        The reserve type, e.g. ``FCR``.
    """
    stem = column.replace("reserve_", "")
    for suffix in RESERVE_PARAMETER_SUFFIXES:
        if suffix in stem:
            return stem.replace(suffix, "")

    return stem


def update_storage_plan(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Update storage plan constraints in a PyPSA network based on the given snapshots.

    This function updates the storage constraints for the specified snapshots within
    a PyPSA network. It ensures that the state of charge of each storage unit at the
    last snapshot in the provided snapshot index meets or exceeds the target state
    of charge defined in the storage plan. The constraints are added to the network's
    optimization model.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the storage units, their operational
        data, and the optimization model.
    sns : pd.Index
        A pandas index object representing the snapshots for which the storage
        constraints are to be updated. If the index is empty, the function returns
        without applying any constraints.

    """
    if sns.empty or n.storage_units.empty:
        logger.debug(
            "Skipping update_storage_plan: no snapshots or storage units in network."
        )
        return

    last_snapshot = sns[-1]

    # Access target SoC from network's storage_units_t for the last snapshot
    soc_targets = n.storage_units_t["storage_plan"].loc[last_snapshot]

    # Return early if soc_targets is empty or contains only NaNs
    if soc_targets.empty or soc_targets.isna().all():
        logger.debug("Skipping update_storage_plan: no storage targets in network.")
        return

    # Access state-of-charge variables for all units at the last snapshot
    soc_var = n.model.variables["StorageUnit-state_of_charge"].loc[
        last_snapshot, soc_targets.index
    ]

    # Slack for storage targets
    # Named dimension required by linopy
    slack_var = n.model.add_variables(
        coords={"StorageUnit": soc_targets.index},
        name="StorageUnit-slack-soc-target",
        dims=("StorageUnit",),
        lower=0,
        upper=float("inf"),
    )

    # Add all constraints in one go
    n.model.add_constraints(
        soc_var + slack_var >= soc_targets, name="StorageUnit_soc_min_constraint"
    )


def initialize_soc_from_storage_plan(
    n: pypsa.Network,
    storage_plan: pd.DataFrame,
    preopt_targeted_carriers: Sequence[str],
) -> pd.Index:
    """
    Seed ``state_of_charge_initial`` from the pre-optimisation storage plan at t=0.

    When the storage planning pre-optimisation enforces a cyclic state of charge
    (``eva.storage_planning.use_cyclic_soc``), the plan's SOC at its first
    timestep is also its SOC at the end of the year. Starting the market models
    from that same level keeps the first rolling horizon consistent with the
    plan the later horizons are steered towards, instead of the generic
    midpoint initialisation applied when the base network is built.

    Only storage units whose carrier is covered by the pre-optimisation are
    touched; batteries, DSR and any other unit outside its scope keep the
    initialisation from ``build_base_network``. Every covered unit must carry a
    value in the plan at t=0 -- a partial plan is an error, not a reason to seed
    some units and leave the rest on the generic initialisation.

    Parameters
    ----------
    n : pypsa.Network
        Market network whose ``storage_units.state_of_charge_initial`` is updated
        in place.
    storage_plan : pd.DataFrame
        Storage plan produced by ``plan_storage``, indexed by snapshot with one
        column per planned storage unit.
    preopt_targeted_carriers : Sequence[str]
        Storage unit carriers covered by the pre-optimisation, from
        ``eva.storage_planning.preopt_targeted_carriers``.

    Returns
    -------
    pd.Index
        Index of the storage units whose ``state_of_charge_initial`` was set.

    Raises
    ------
    KeyError
        If the network has units covered by the pre-optimisation but the plan does
        not cover the network's first snapshot, or has no value there for some of
        those units. Both mean the plan and the market network were not built from
        the same base network.
    """
    targeted_units = n.storage_units.index[
        n.storage_units.carrier.isin(preopt_targeted_carriers)
    ]
    if targeted_units.empty:
        # The only legitimate no-op: nothing in this network is pre-optimised.
        logger.debug(
            "Skipping SOC initialization from storage plan: no storage unit carries "
            f"one of the pre-optimised carriers {list(preopt_targeted_carriers)}."
        )
        return pd.Index([], name="StorageUnit")

    # Past this point the plan must cover every targeted unit at t=0. plan_storage
    # writes exactly the columns selected by `carrier.isin(preopt_targeted_carriers)`
    # and reindexes them onto the base network's hourly snapshots; every market
    # network -- full year or centroid slice, which keeps the original timestamps --
    # draws its snapshots from that same base network. Any gap therefore means the
    # plan and the network came from different runs, and seeding some units while
    # silently leaving others on the generic initialization would desynchronise the
    # market models from the plan they are steered towards.
    first_snapshot = n.snapshots[0]
    if first_snapshot not in storage_plan.index:
        span = (
            f"spans {storage_plan.index[0]} to {storage_plan.index[-1]}"
            if len(storage_plan.index)
            else "is empty"
        )
        raise KeyError(
            f"Storage plan does not cover the first market snapshot {first_snapshot}; "
            f"the plan {span}. The plan and the market network were not built from "
            "the same base network."
        )

    planned_soc = storage_plan.loc[first_snapshot].reindex(targeted_units).astype(float)
    missing = planned_soc.index[planned_soc.isna()]
    if not missing.empty:
        raise KeyError(
            f"Storage plan has no value at {first_snapshot} for "
            f"{len(missing)} of {len(targeted_units)} storage unit(s) covered by the "
            f"pre-optimisation: {list(missing)}. plan_storage writes every unit whose "
            f"carrier is in {list(preopt_targeted_carriers)}, so the plan and the "
            "network disagree on which units exist."
        )

    n.storage_units.loc[targeted_units, "state_of_charge_initial"] = planned_soc
    logger.info(
        "Cyclic pre-optimisation SOC: initialized state_of_charge_initial for "
        f"{len(targeted_units)} storage unit(s) from the storage plan at "
        f"{first_snapshot}."
    )

    return targeted_units


def add_storage_soc_slack_penalty_to_objective(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Add penalty terms for slack variable to the objective function.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    if sns.empty or n.storage_units.empty:
        logger.debug(
            "Skipping add_storage_soc_slack_penalty_to_objective: no snapshots or storage units in network."
        )
        return

    # Access slack variables at the last snapshot
    if "StorageUnit-slack-soc-target" not in n.model.variables:
        logger.debug(
            "Skipping add_storage_soc_slack_penalty_to_objective: no StorageUnit-slack-soc-target variable in network."
        )
        return

    slack_var = n.model.variables["StorageUnit-slack-soc-target"]

    # Penalty is 1.05x the max marginal cost of any generator, excluding load shedding and DSR
    exclude_carriers = {"load shedding", "DSR_explicit", "DSR_implicit"}
    regular_gens = n.generators[~n.generators.carrier.isin(exclude_carriers)]
    penalty_cost = 1.05 * regular_gens["marginal_cost"].max()

    # Add to objective function
    n.model.objective += penalty_cost * slack_var.sum()


def update_technical_constraints(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Update hydro technical constraints (min/max reservoir levels) in a PyPSA network.
    """
    if sns.empty or n.storage_units.empty:
        return

    # Apply constraints to model
    if not hasattr(n, "model") or n.model is None:
        return
    if "StorageUnit-state_of_charge" not in n.model.variables:
        return

    soc_var = n.model.variables["StorageUnit-state_of_charge"]

    for constraint_type, op in [
        ("state_of_charge_min", ">="),
        ("state_of_charge_max", "<="),
    ]:
        # Time-varying constraints
        if constraint_type in n.storage_units_t:
            if not (all_c := n.storage_units_t[constraint_type]).empty:
                if len(overlap := all_c.index.intersection(sns)) > 0:
                    c_vals = all_c.loc[overlap]

                    # Fill NaNs with non-binding values
                    # - For min constraints (>=): NaN → 0 (non-binding, allows SOC to go to 0)
                    # - For max constraints (<=): NaN → inf (non-binding, no upper limit)
                    if op == ">=":
                        c_vals = c_vals.fillna(0)
                    else:  # op == "<="
                        c_vals = c_vals.fillna(float("inf"))

                    # Apply constraints to all units that exist in both c_vals and the model
                    units = c_vals.columns.intersection(
                        soc_var.coords[soc_var.dims[1]].values
                    )

                    logger.info(f"Applied {constraint_type} to {len(units)} units")

                    if len(units) > 0:
                        constraint_values = c_vals[units].values

                        soc_subset = soc_var.sel(
                            snapshot=overlap, **{str(soc_var.dims[1]): units}
                        )
                        n.model.add_constraints(
                            soc_subset,
                            op,
                            constraint_values,
                            name=f"StorageUnit_technical_{constraint_type.split('_')[-1]}",
                        )
                        continue
        # # Static constraints fallback
        if constraint_type in n.storage_units.columns:
            units = n.storage_units.index[
                n.storage_units[constraint_type].notna()
                & (n.storage_units[constraint_type] > 0)
            ]

            if valid := [
                u for u in units if u in soc_var.coords[soc_var.dims[1]].values
            ]:
                for unit in valid:
                    val = n.storage_units.loc[unit, constraint_type]
                    unit_soc = soc_var.sel(snapshot=sns, **{str(soc_var.dims[1]): unit})
                    n.model.add_constraints(
                        unit_soc, op, val, name=f"StorageUnit_{unit}_{constraint_type}"
                    )

    # Apply MIN_PUMP constraints (minimum pumping power)
    if "min_pump_mw" in n.storage_units_t:
        if not (min_pump_df := n.storage_units_t["min_pump_mw"]).empty:
            if len(overlap := min_pump_df.index.intersection(sns)) > 0:
                min_pump_vals = min_pump_df.loc[overlap]

                # Get the p_store variable (positive when charging/pumping in PyPSA)
                if "StorageUnit-p_store" in n.model.variables:
                    p_store_var = n.model.variables["StorageUnit-p_store"]

                    # Apply constraints to all units that exist in both min_pump_vals and the model
                    units = min_pump_vals.columns.intersection(
                        p_store_var.coords[p_store_var.dims[1]].values
                    )

                    if len(units) > 0:
                        # Filter to only units with actual constraints (drop all-NaN columns)
                        units_with_constraints = [
                            u for u in units if not min_pump_vals[u].isna().all()
                        ]

                        if len(units_with_constraints) > 0:
                            # Get constraint values, fill NaN with 0 (no minimum when constraint not active)
                            constraint_values = (
                                min_pump_vals[units_with_constraints].fillna(0).values
                            )

                            # Get the p_store subset for these units
                            p_store_subset = p_store_var.sel(
                                snapshot=overlap,
                                **{str(p_store_var.dims[1]): units_with_constraints},
                            )

                            # Add constraint: p_store >= min_pump_mw
                            # p_store is positive when pumping, min_pump_mw already abs() converted to positive
                            n.model.add_constraints(
                                p_store_subset,
                                ">=",
                                constraint_values,
                                name="StorageUnit_min_pump",
                            )

                            logger.info(
                                f"Applied min_pump_mw constraints to {len(units_with_constraints)} storage units"
                            )


def turn_off_extendable(n: pypsa.Network) -> tuple[pypsa.Network, dict[str, Any]]:
    """
    Sets the "extendable" attribute to False for all applicable components in the given pypsa.Network.

    This function modifies the given pypsa.Network object to disable the "extendable"
    attribute for generators, storage units, stores, links, and lines, ensuring that their
    respective nominal capacities or energy limits are fixed and not adjustable during
    optimization or other operations.

    Parameters
    ----------
    n : pypsa.Network
        A pypsa.Network instance containing components such as generators, storage units,
        stores, links, and lines. The "extendable" attribute for these components will be
        set to False if present.

    Returns
    -------
    pypsa.Network
        The modified pypsa.Network instance with all applicable components' "extendable"
        attributes set to False.
    """
    # Save original p_nom_extendable values
    original_extendable = {
        "generators": n.generators["p_nom_extendable"].copy(),
        "storage_units": n.storage_units["p_nom_extendable"].copy(),
        "stores": n.stores["e_nom_extendable"].copy(),
        "links": n.links["p_nom_extendable"].copy(),
        "lines": n.lines["s_nom_extendable"].copy(),
    }

    # Temporarily disable capacity expansion
    n.generators.p_nom_extendable = False
    n.storage_units.p_nom_extendable = False
    n.stores.e_nom_extendable = False
    n.links.p_nom_extendable = False
    n.lines.s_nom_extendable = False

    return n, original_extendable


def set_all_generators_not_committable(
    n: pypsa.Network,
    n_da: pypsa.Network,
    set_p_min_pu_zero: bool = True,
    id_free_commitment_carriers: list[str] | None = None,
) -> pypsa.Network:
    """
    Make all generators but the recommitable  as not committable
    """
    if id_free_commitment_carriers is None:
        id_free_commitment_carriers = ["DSR_explicit"]

    com_i = n_da.get_committable_i("Generator")
    free_gen = n_da.generators.index[
        n_da.generators.carrier.isin(id_free_commitment_carriers)
    ]
    com_i = com_i.difference(free_gen)
    n.generators.loc[com_i, "committable"] = False
    if set_p_min_pu_zero:
        n.generators.loc[com_i, "p_min_pu"] = 0
        n.generators_t["p_min_pu"] = pd.DataFrame(index=n.snapshots)
    return n


def add_fbmc_constraint(n, sns):
    if n.links.empty:
        logger.debug("No links in the network. Skipping FBMC constraint.")
        return

    ptdf_fb_region = n._ptdf_fb_region.sel(snapshot=sns)
    ptdf_ahc_directed = n._ptdf_ahc_directed.sel(snapshot=sns)
    ptdf_dc = n._ptdf_dc.sel(snapshot=sns)
    ram = n._ram.sel(snapshot=sns)

    p = n.model.variables["Link-p"]

    # Core/nordic contribution
    flows_to_fb_region = p.sel(Link=ptdf_fb_region.indexes["Link"]).where(
        ptdf_fb_region.notnull()
    )

    # Evolved flow-based contribution
    flow_evolved = p.sel(Link=ptdf_dc.indexes["Link"]).where(ptdf_dc.notnull())

    # Advanced hybrid coupling contribution
    external_flow = p.sel(Link=ptdf_ahc_directed.indexes["Link"]).where(
        ptdf_ahc_directed.notnull()
    )

    lhs = (
        (ptdf_ahc_directed * external_flow).sum("Link")
        + (ptdf_fb_region * flows_to_fb_region).sum("Link")
        + (ptdf_dc * flow_evolved).sum("Link")
    )

    con = lhs <= ram
    n.model.add_constraints(con, name="FBMC_PTDF_RAM")


def add_fixed_commitment_constraints(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Add constraints to fix generator commitment status to integer solution results.

    This function checks if the network has stored integer unit commitment results
    and adds equality constraints to fix the status variables to the integer solution values.
    The binary variables are automatically relaxed by PyPSA when linearized_unit_commitment=False
    is used in the second optimization pass.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """

    # Check if integer status data exists in the network
    if not hasattr(n, "_var_status") or n._var_status is None:
        return

    # Get committable generators
    committable_gens = n.generators.index[n.generators.committable]

    if committable_gens.empty:
        return

    # Get the status variables from the model
    if "Generator-status" not in n.model.variables:
        return

    status_vars = n.model.variables["Generator-status"]
    integer_status = n._var_status

    # Filter to current snapshots and committable generators
    sns_filtered = sns.intersection(
        integer_status.index  # ty: ignore[unresolved-attribute]
    )
    gens_filtered = committable_gens.intersection(
        integer_status.columns  # ty: ignore[unresolved-attribute]
    )

    if sns_filtered.empty or gens_filtered.empty:
        return

    # Select relevant status variables and values for current snapshots and generators
    try:
        status_vars_filtered = status_vars.sel(snapshot=sns_filtered).sel(
            **{str(status_vars.dims[1]): gens_filtered}
        )
        integer_status_filtered = (
            integer_status.loc[  # ty: ignore[unresolved-attribute]
                sns_filtered, gens_filtered
            ]
        )

        # Round status values to ensure they are exactly 0 or 1 (handle numerical precision issues)
        # Status variables should be binary but solvers might return values like 0.9999999
        integer_status_rounded = integer_status_filtered.round().astype(int)

        # Add equality constraints to fix status variables to integer solution values
        # This ensures the LP solution matches the integer solution exactly
        lhs = status_vars_filtered
        rhs = integer_status_rounded

        constraints = lhs == rhs
        n.model.add_constraints(constraints, name="fixed_commitment_status")
        logger.info(
            f"Added fixed commitment constraints for {len(gens_filtered)} generators across {len(sns_filtered)} snapshots"
        )
    except (KeyError, AttributeError, ValueError) as e:
        logger.warning(f"Could not add fixed commitment constraints: {e}")
        return


def add_reserves_requirements(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Add reserves requirements constraints to the optimization model.

    Creates reserve variables for generators and storage units that can provide reserves,
    and adds constraints to ensure reserve requirements are met at each bus.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    # Check if reserves are configured
    reserves_parameters = get_reserves_parameters(n)
    if not reserves_parameters:
        logger.debug("No reserves parameters found. Skipping reserves requirements.")
        return

    # Get available reserve types from parameters
    reserve_types = list(reserves_parameters.keys())

    # Get all bus reserve requirement columns
    bus_reserve_cols = [col for col in n.buses.columns if col.startswith("reserve_")]

    if not bus_reserve_cols:
        logger.debug(
            "No reserve requirements found in buses. Skipping reserves constraints."
        )
        return

    for reserve_type in reserve_types:
        reserve_col = f"reserve_{reserve_type}"

        # Skip if this reserve type is not in bus requirements
        if reserve_col not in n.buses.columns:
            continue

        # Get buses with reserve requirements for this type
        buses_with_requirements = n.buses.index[
            n.buses[reserve_col].notna() & (n.buses[reserve_col] > 0)
        ]

        if buses_with_requirements.empty:
            continue

        # CREATE RESERVE VARIABLES

        # Get generators that can provide this reserve type
        if reserve_col in n.generators.columns:
            reserve_generators = n.generators.index[n.generators[reserve_col]]
        else:
            reserve_generators = pd.Index([])

        directions: list[str] = n.meta["eva"]["reserves"]["direction"]

        # Create reserve variables for generators
        if not reserve_generators.empty:
            # Reserves for generators
            for sign in directions:
                gen_var_name = f"Generator-r_{reserve_type}_{sign}"
                n.model.add_variables(
                    coords=[sns, reserve_generators],
                    name=gen_var_name,
                    lower=0,
                    upper=float("inf"),
                )

        # Get storage units that can provide this reserve type
        if reserve_col in n.storage_units.columns:
            reserve_storage = n.storage_units.index[n.storage_units[reserve_col]]
        else:
            reserve_storage = pd.Index([])

        # Create reserve variables for storage units
        if not reserve_storage.empty:
            for sign in directions:
                # Reserves for storage units
                storage_var_name = f"StorageUnit-r_{reserve_type}_{sign}"
                n.model.add_variables(
                    coords=[sns, reserve_storage],
                    name=storage_var_name,
                    lower=0,
                    upper=float("inf"),
                )

        # CREATE RESERVE REQUIREMENTS CONSTRAINTS: Equation (5.1) of the modeling guide
        # \sum_{n \in S} r_{n,t} >= R_{n,t})

        # Build vectorized constraints for each direction (up/down)
        for direction in directions:
            exprs = []
            var_suffix = f"r_{reserve_type}_{direction}"

            # Add generator expressions
            if not reserve_generators.empty:
                gen_var = n.model.variables[f"Generator-{var_suffix}"]
                gen_buses = (
                    n.generators.loc[reserve_generators, "bus"]
                    .rename("Bus")
                    .to_xarray()
                )
                gen_expr = (
                    gen_var.sel(Generator=reserve_generators).groupby(gen_buses).sum()
                )
                exprs.append(gen_expr)

            # Add storage unit expressions
            if not reserve_storage.empty:
                storage_var_name = f"StorageUnit-{var_suffix}"
                if storage_var_name in n.model.variables:
                    storage_var = n.model.variables[storage_var_name]
                    # Only use units present in the variable
                    available_storage = storage_var.coords["StorageUnit"].values
                    valid_storage = reserve_storage.intersection(available_storage)

                    if not valid_storage.empty:
                        storage_buses = (
                            n.storage_units.loc[valid_storage, "bus"]
                            .rename("Bus")
                            .to_xarray()
                        )
                        storage_expr = (
                            storage_var.sel(StorageUnit=valid_storage)
                            .groupby(storage_buses)
                            .sum()
                        )
                        exprs.append(storage_expr)

            # Create constraint if we have any expressions
            if exprs:
                # Merge all expressions and reindex to required buses
                lhs = merge(exprs, join="outer").reindex(Bus=buses_with_requirements)
                rhs = xr.DataArray(n.buses.loc[buses_with_requirements, reserve_col])

                # Add constraint with proper naming
                n.model.add_constraints(
                    lhs,
                    ">=",
                    rhs,
                    name=f"Bus-Reserve_requirements_{reserve_type}_{direction}",
                )

    logger.info(
        f"Added reserves requirements constraints for {len(reserve_types)} reserve types"
    )


def adapt_generator_constraints_with_reserves(
    n: pypsa.Network, sns: pd.Index, constraint_type: str = "power"
) -> None:
    """
    Adapt generator constraints to account for reserves provision.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    constraint_type : str, default "power"
        Type of constraint to adapt. Either "power" or "ramp".

        For power constraints:
        - Adapt Generator-fix-p-lower: subtract all reserves down that the generator can provide
        - Adapt Generator-fix-p-upper: add all reserves up that the generator can provide
        - Adapt Generator-com-p-lower: subtract all reserves down that the generator can provide
        - Adapt Generator-com-p-upper: add all reserves up that the generator can provide

        For ramp constraints:
        - Adapt Generator-fix-p-ramp_limit_down: subtract all reserves down that the generator can provide
        - Adapt Generator-fix-p-ramp_limit_up: add all reserves up that the generator can provide
        - Adapt Generator-com-p-ramp_limit_down: subtract all reserves down that the generator can provide
        - Adapt Generator-com-p-ramp_limit_up: add all reserves up that the generator can provide
    """
    # Check if reserves are configured
    reserves_parameters = get_reserves_parameters(n)
    if not reserves_parameters:
        return

    # Get available reserve types from parameters
    reserve_types = list(reserves_parameters.keys())

    directions: list[str] = n.meta["eva"]["reserves"]["direction"]

    # Process each constraint type
    for direction in directions:
        for type_gen in ["fix", "com"]:
            # Get the generator coordinate name from the constraint
            var_name = [
                f"Generator-r_{reserve}_{direction}" for reserve in reserve_types
            ]

            # Build constraint name based on type
            if constraint_type == "power":
                if direction == "down":
                    constraint_name = f"Generator-{type_gen}-p-lower"
                else:
                    constraint_name = f"Generator-{type_gen}-p-upper"
            elif constraint_type == "ramp":
                constraint_name = f"Generator-{type_gen}-p-ramp_limit_{direction}"
            else:
                raise ValueError(f"Unknown constraint_type: {constraint_type}")

            # Check if constraint exists in the model
            if constraint_name not in n.model.constraints:
                continue

            constraint = n.model.constraints[constraint_name]

            # Get the generator indices based on committable status
            if type_gen == "fix":
                gens_list = n.generators.index[~n.generators.committable]
            else:
                gens_list = n.generators.index[n.generators.committable]

            # For ramp constraints, restrict to generators already in the constraint
            # (i.e. those PyPSA built a ramp constraint for because they have a
            # non-NaN ramp limit). Generators with NaN ramp limits must not be
            # pulled in via xarray coordinate broadcasting.
            if constraint_type == "ramp":
                gen_coord = f"Generator-{type_gen}"
                gens_in_constraint = pd.Index(
                    constraint.data.coords[gen_coord].values
                    if gen_coord in constraint.data.coords
                    else []
                )
                gens_list = gens_list.intersection(gens_in_constraint)

            # Build vars_gen expression
            vars_gen = sum(
                n.model.variables[name]
                .sel(
                    Generator=gens_list.intersection(
                        n.model.variables[name].coords["Generator"].values
                    )
                )
                .rename(
                    {"Generator": f"Generator-{type_gen}"}
                )  # rename dim for alignment
                for name in var_name
                if name in n.model.variables
                and len(
                    gens_list.intersection(
                        n.model.variables[name].coords["Generator"].values
                    )
                )
                > 0
            )

            # Modify constraint
            if direction == "up":
                constraint.lhs += vars_gen
            else:
                constraint.lhs -= vars_gen

    constraint_desc = "power" if constraint_type == "power" else "ramp limit"
    logger.info(
        f"Completed adaptation of generator {constraint_desc} constraints with reserves"
    )


def adapt_generator_power_constraints_with_reserves(
    n: pypsa.Network, sns: pd.Index
) -> None:
    """
    Adapt generator power constraints to account for reserves provision.

    For non-committable generators:
    - Adapt Generator-fix-p-lower: subtract all reserves down that the generator can provide
    - Adapt Generator-fix-p-upper: add all reserves up that the generator can provide

    For committable generators:
    - Adapt Generator-com-p-lower: subtract all reserves down that the generator can provide
    - Adapt Generator-com-p-upper: add all reserves up that the generator can provide

    Constraints (5.2) to (5.5) of the modeling guide.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    adapt_generator_constraints_with_reserves(n, sns, constraint_type="power")


def add_storage_reserves_limit_constraints(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Add storage reserves limit constraints.

    Constraints (5.6) and (5.7) of the modeling guide.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    # Check if reserves are configured
    reserves_parameters = get_reserves_parameters(n)
    if not reserves_parameters:
        return

    # Get available reserve types from parameters
    reserve_types = list(reserves_parameters.keys())

    # Filter variable to only include storage units that can provide reserves
    reserve_columns = [f"reserve_{reserve}" for reserve in reserve_types]

    # Find storage units that have at least one reserve type set to True
    storage_providers = n.storage_units.loc[
        n.storage_units[reserve_columns].any(axis=1)
    ].index

    # Skip if no storage providers
    if storage_providers.empty:
        return

    directions: list[str] = n.meta["eva"]["reserves"]["direction"]

    for direction in directions:
        # Create left-hand-side of the equation
        lhs = sum(
            n.model.variables[f"StorageUnit-r_{reserve}_{direction}"]
            for reserve in reserve_types
            if f"StorageUnit-r_{reserve}_{direction}" in n.model.variables
        )

        # Get storage units dispatch and store variables
        p_dispatch_var = n.model.variables["StorageUnit-p_dispatch"].sel(
            StorageUnit=storage_providers
        )
        p_store_var = n.model.variables["StorageUnit-p_store"].sel(
            StorageUnit=storage_providers
        )

        # Create right-hand-side of the equation
        if direction == "up":
            # For upward reserves: p_dispatch - p_store + sum(upward reserves) <= max*p_nom
            p_nom = n.storage_units.p_nom.loc[storage_providers]
            p_max_pu = n.storage_units.p_max_pu.loc[storage_providers].fillna(1)
            rhs = p_nom * p_max_pu

            # Add dispatch and store variables to the left-hand side
            lhs += p_dispatch_var - p_store_var
        else:
            # For downward reserves: p_dispatch - p_store - sum(downward reserves) >= min*p_nom
            p_nom = n.storage_units.p_nom.loc[storage_providers]
            p_min_pu = n.storage_units.p_min_pu.loc[storage_providers].fillna(-1)
            rhs = p_nom * p_min_pu

            # Same for downward
            lhs = p_dispatch_var - p_store_var - lhs

        # Create the constraint name
        constraint_name = f"StorageUnit-r_{direction}-limit"

        # Add constraint to model
        if direction == "up":
            n.model.add_constraints(lhs, "<=", rhs, name=constraint_name)
        else:
            n.model.add_constraints(lhs, ">=", rhs, name=constraint_name)


def add_storage_reserves_limit_constraints_intraday_market(
    n: pypsa.Network, sns: pd.Index
) -> None:
    """
    Add storage reserves limit constraints for intraday markets. Take into account results from the day-ahead market.

    Same constraints (5.6) and (5.7) of the modeling guide, but with the results from the reserves instead of variables.

    Allows for some slack in the constraints to avoid numerical infeasibilities.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    if (
        "total_reserve_up" not in n.storage_units_t
        or "total_reserve_down" not in n.storage_units_t
    ):
        logger.debug(
            "No total_reserve_up/down in storage_units_t. Skipping intraday reserves limit constraints."
        )
        return

    # Find storage units that have at least one reserve type set to True
    storage_providers = n.storage_units.loc[
        n.storage_units_t["total_reserve_up"].columns
    ].index

    # Skip if no storage providers
    if storage_providers.empty:
        return

    directions: list[str] = n.meta["eva"]["reserves"]["direction"]

    for direction in directions:
        # Create left-hand-side of the equation
        eps = 1e-4
        add_slack = -eps

        # Convert pandas DataFrame to xarray DataArray with correct coordinate names
        lhs_df = n.storage_units_t[f"total_reserve_{direction}"].loc[
            sns, storage_providers
        ]
        lhs = xr.DataArray(
            lhs_df.values + add_slack,
            dims=["snapshot", "StorageUnit"],
            coords={"snapshot": sns, "StorageUnit": storage_providers},
        )

        # Get storage units dispatch and store variables
        p_dispatch_var = n.model.variables["StorageUnit-p_dispatch"].sel(
            StorageUnit=storage_providers
        )
        p_store_var = n.model.variables["StorageUnit-p_store"].sel(
            StorageUnit=storage_providers
        )

        # Create right-hand-side of the equation
        if direction == "up":
            # For upward reserves: p_dispatch - p_store + sum(upward reserves) <= max*p_nom
            p_nom = n.storage_units.p_nom.loc[storage_providers]
            p_max_pu = n.storage_units.p_max_pu.loc[storage_providers].fillna(1)
            rhs = p_nom * p_max_pu

            # Add dispatch and store variables to the left-hand side
            # Start with linopy expression and add DataArray to it
            lhs = (p_dispatch_var - p_store_var) + lhs
        else:
            # For downward reserves: p_dispatch - p_store - sum(downward reserves) >= min*p_nom
            p_nom = n.storage_units.p_nom.loc[storage_providers]
            p_min_pu = n.storage_units.p_min_pu.loc[storage_providers].fillna(-1)
            rhs = p_nom * p_min_pu

            # Same for downward - start with linopy expression
            lhs = (p_dispatch_var - p_store_var) - lhs

        # Create the constraint name
        constraint_name = f"StorageUnit-r_{direction}-limit"

        # Add constraint to model
        if direction == "up":
            n.model.add_constraints(lhs, "<=", rhs, name=constraint_name)
        else:
            n.model.add_constraints(lhs, ">=", rhs, name=constraint_name)


def add_implicit_dsr_fixing_constraints(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Commit a fraction of the day-ahead implicit-DSR (storage) dispatch intraday.

    Floors the intraday ``p_dispatch`` and ``p_store`` variables at ``f`` times
    their day-ahead realised values, where ``f`` is
    ``n.meta["eva"]["dsr"]["implicit_dsr"]["da_dispatch_fixed_fraction"]``.

    This is the counterpart to the per-unit *ceilings* set in
    :func:`fix_dsr_dispatch_from_da`. Ceilings alone cannot commit a storage unit:
    PyPSA floors both ``p_dispatch`` and ``p_store`` at 0, so the feasible interval
    always contains "do nothing" and the solver may abandon the day-ahead schedule.
    The floors added here force delivery of at least fraction ``f`` of the day-ahead
    action. At ``f = 1`` the floor meets the ceiling and dispatch is pinned to the
    day-ahead result; at ``f = 0`` the floors are no-ops (fully flexible intraday).

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    # The day-ahead references are stashed by fix_dsr_dispatch_from_da, which only
    # runs when preparing an intraday network. Absent them, there is nothing to fix.
    if (
        "da_dispatch_fix" not in n.storage_units_t
        or "da_store_fix" not in n.storage_units_t
    ):
        return

    f = n.meta["eva"]["dsr"]["implicit_dsr"]["da_dispatch_fixed_fraction"]
    if f <= 0.0:
        return

    dsr_storage = n.storage_units_t["da_dispatch_fix"].columns
    if len(dsr_storage) == 0:
        return

    # Small slack so floating-point round-trip (MW -> pu -> MW in the ceilings)
    # cannot push the floor marginally above the ceiling at f = 1.
    eps = 1e-4

    p_dispatch_var = n.model.variables["StorageUnit-p_dispatch"].sel(
        StorageUnit=dsr_storage
    )
    p_store_var = n.model.variables["StorageUnit-p_store"].sel(StorageUnit=dsr_storage)

    dispatch_floor = (
        f * n.storage_units_t["da_dispatch_fix"].loc[sns, dsr_storage] - eps
    ).clip(lower=0.0)
    store_floor = (
        f * n.storage_units_t["da_store_fix"].loc[sns, dsr_storage] - eps
    ).clip(lower=0.0)

    dispatch_floor_da = xr.DataArray(
        dispatch_floor.values,
        dims=["snapshot", "StorageUnit"],
        coords={"snapshot": sns, "StorageUnit": dsr_storage},
    )
    store_floor_da = xr.DataArray(
        store_floor.values,
        dims=["snapshot", "StorageUnit"],
        coords={"snapshot": sns, "StorageUnit": dsr_storage},
    )

    n.model.add_constraints(
        p_dispatch_var, ">=", dispatch_floor_da, name="DSR_implicit-p_dispatch-floor"
    )
    n.model.add_constraints(
        p_store_var, ">=", store_floor_da, name="DSR_implicit-p_store-floor"
    )

    logger.info(
        f"Applied implicit-DSR dispatch floors at {f:.1%} of day-ahead for "
        f"{len(dsr_storage)} storage unit(s)"
    )


def add_generators_reserves_ramping_constraint(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Add generator reserves ramping constraints.

    Constraints (5.8) and (5.9) of the modeling guide.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    # Check if reserves are configured
    reserves_parameters = get_reserves_parameters(n)
    if not reserves_parameters:
        return

    # Get available reserve types from parameters
    reserve_types = list(reserves_parameters.keys())

    directions: list[str] = n.meta["eva"]["reserves"]["direction"]

    for reserve in reserve_types:
        for direction in directions:
            # Get reserve variables for generators
            var_name = f"Generator-r_{reserve}_{direction}"
            if var_name not in n.model.variables:
                continue

            lhs = n.model.variables[var_name]

            # Get the generators that have reserve variables (from the variable coordinates)
            # The reserve variables are only created for generators with reserve_{reserve}=True
            reserve_generators = lhs.coords["Generator"].values

            # Get the max provision parameter for this reserve type and direction
            # For FRR, use separate up/down columns; for FCR, use symmetric column
            if reserve == "FRR":
                max_provision_col = f"max_{reserve.lower()}_provision_{direction}"
            else:  # FCR
                max_provision_col = f"max_{reserve.lower()}_provision"

            if max_provision_col in n.generators.columns:
                # max provision is in per-unit, need to multiply by p_nom to get MW
                # Filter to only include generators that have reserve variables
                rhs = (
                    n.generators.loc[reserve_generators, max_provision_col]
                    * n.generators.loc[reserve_generators, "p_nom"]
                )
            else:
                # TODO: remove this once pre-builts have been updated to include the new columns
                rhs = 0.05 * n.generators.loc[reserve_generators, "p_nom"]
                logger.warning(
                    f"Column {max_provision_col} not found in generators. Using a default of 0.05 for all generators "
                )

            # Only add constraint for generators with valid (non-NaN) max_provision values.
            # Generators with NaN max_provision (like reserve shedding) have no ramping limit
            # for reserves, allowing them to provide unlimited reserves.
            valid_mask = rhs.notna()
            if valid_mask.any():
                # Filter both lhs and rhs to only include generators with valid limits
                constrained_gens = rhs[valid_mask].index
                lhs_filtered = lhs.sel(Generator=constrained_gens)
                rhs_filtered = rhs[valid_mask]

                # Create the constraint name
                constraint_name = f"Generator-r_{reserve}_{direction}-ramping"

                # Add constraint to model
                n.model.add_constraints(
                    lhs_filtered, "<=", rhs_filtered, name=constraint_name
                )


def add_storage_reserves_soc_constraint(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Add storage reserves soc constraints.

    Constraints (5.10) and (5.11) of the modeling guide.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    # Check if reserves are configured
    reserves_parameters = get_reserves_parameters(n)
    if not reserves_parameters:
        return

    # Get available reserve types from parameters
    reserve_types = list(reserves_parameters.keys())

    directions: list[str] = n.meta["eva"]["reserves"]["direction"]

    for direction in directions:
        # Create left-hand-side of the equation
        reserve_vars = [
            reserves_parameters[reserve]["min_duration"]
            * n.model.variables[f"StorageUnit-r_{reserve}_{direction}"]
            for reserve in reserve_types
            if f"StorageUnit-r_{reserve}_{direction}" in n.model.variables
        ]

        # Skip if no reserve variables exist for this direction
        if not reserve_vars:
            logger.info(
                f"No reserve variables found for {direction} direction, skipping soc constraint"
            )
            continue

        lhs = sum(reserve_vars)

        # Filter variable to only include storage units that can provide reserves
        reserve_columns = [f"reserve_{reserve}" for reserve in reserve_types]

        # Find storage units that have at least one reserve type set to True
        storage_providers = n.storage_units.loc[
            n.storage_units[reserve_columns].any(axis=1)
        ].index

        # Skip if no storage providers exist
        if len(storage_providers) == 0:
            logger.info(
                f"No storage providers found for {direction} direction, skipping soc constraint"
            )
            continue

        # Get storage units state-of-charge (soc) variable
        var_name = "StorageUnit-state_of_charge"
        soc_var = n.model.variables[var_name].sel(StorageUnit=storage_providers)

        # Create right-hand-side of the equation
        if direction == "up":
            # For upward reserves, we use the state-of-charge (soc)
            rhs = 0

            # Add soc variables to the left-hand side
            lhs -= soc_var

        else:
            # For downward reserves, we use the max soc - soc
            p_nom = n.storage_units.p_nom.loc[storage_providers]
            max_hours = n.storage_units.max_hours.loc[storage_providers].fillna(1)
            max_soc = p_nom * max_hours

            # Now we can subtract the soc_var from the expanded max_soc values
            rhs = max_soc

            # Add soc variables to the left-hand side
            lhs += soc_var

        # Create the constraint name
        constraint_name = f"StorageUnit-r_{direction}-soc"

        # Add constraint to model
        n.model.add_constraints(lhs, "<=", rhs, name=constraint_name)

        logger.info(
            "Completed adaptation of storage unit soc constraints with reserves"
        )


def add_reserves_penalty_to_objective(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Add penalty terms for reserves to the objective function.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    if not hasattr(n, "model") or n.model is None:
        return

    # Check if reserves are configured
    reserves_parameters = get_reserves_parameters(n)
    if not reserves_parameters:
        logger.debug(
            "No reserves parameters found. Skipping adaptation of objective function."
        )
        return

    # Get available reserve types from parameters
    reserve_types = list(reserves_parameters.keys())

    directions: list[str] = n.meta["eva"]["reserves"]["direction"]

    for reserve_type in reserve_types:
        # Get reserve variable names
        for direction in directions:
            for component in ["Generator", "StorageUnit"]:
                var_name = f"{component}-r_{reserve_type}_{direction}"
                if var_name not in n.model.variables:
                    continue

                reserve_var = n.model.variables[var_name]

                # Get component dataframe
                comp_attr = (
                    "storage_units"
                    if component == "StorageUnit"
                    else f"{component.lower()}s"
                )
                comp_df = getattr(n, comp_attr)

                # Get base penalty cost for this reserve type
                base_penalty_cost = reserves_parameters[reserve_type].get(
                    "penalty_cost", 0.0
                )

                # Vectorised penalty: build a cost weight per component and do a
                # single objective += instead of one per component.  The per-component
                # loop was O(N²) because each linopy += re-concatenates the full
                # growing expression via xr.concat.
                valid_indices = reserve_var.coords[component].values
                if not len(valid_indices):
                    continue

                carriers = comp_df.reindex(valid_indices)["carrier"].values
                costs = np.where(
                    carriers == "reserve shedding",
                    base_penalty_cost * 1e5,
                    base_penalty_cost,
                )
                costs_da = xr.DataArray(
                    costs, coords={component: valid_indices}, dims=[component]
                )
                n.model.objective += (costs_da * reserve_var).sum()

                logger.info(
                    f"Added reserves penalty for {reserve_type} ({direction}) in {component} with cost {base_penalty_cost:.2f} (reserve_shedding: {base_penalty_cost * 1e6:.2f})"
                )


def adapt_generator_ramp_constraints_with_reserves(
    n: pypsa.Network, sns: pd.Index
) -> None:
    """
    Adapt generator ramp constraints to account for reserves provision.

    For non-committable generators:
    - Adapt Generator-fix-p-ramp_limit_down: subtract all reserves down that the generator can provide
    - Adapt Generator-fix-p-ramp_limit_up: add all reserves up that the generator can provide

    For committable generators:
    - Adapt Generator-com-p-ramp_limit_down: subtract all reserves down that the generator can provide
    - Adapt Generator-com-p-ramp_limit_up: add all reserves up that the generator can provide

    Constraints (5.12) to (5.15) of the modeling guide.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    adapt_generator_constraints_with_reserves(n, sns, constraint_type="ramp")


def add_explicit_dsr_constraints(n: pypsa.Network, sns: pd.Index) -> None:
    """Add explicit DSR constraints based on type"""
    explicit_dsrs = n.generators[n.generators.carrier == "DSR_explicit"]
    if explicit_dsrs.empty:
        logger.debug("No explicit DSR generators found")
        return

    p_vars = n.model.variables["Generator-p"]

    # Load shifting constraints - applies to load shifting DSR only
    valid_load_shift_dsrs = explicit_dsrs[explicit_dsrs["dsr_type"] == "load_shifting"]
    if not valid_load_shift_dsrs.empty:
        # Group DSR generators by their shifting_time_window
        grouped = valid_load_shift_dsrs.groupby("shifting_time_window", dropna=True)

        for time_window_hours, dsr_group in grouped:
            time_window_hours = int(time_window_hours)
            dsr_names = dsr_group.index.values

            # Get variables for this group of DSR generators
            load_shift_vars = p_vars.sel(Generator=dsr_names, snapshot=sns)

            # Create window index for all snapshots
            num_windows = (len(sns) + time_window_hours - 1) // time_window_hours
            window_indices = np.repeat(np.arange(num_windows), time_window_hours)[
                : len(sns)
            ]

            # Add window dimension to snapshots
            window_coord = xr.DataArray(
                window_indices, dims=("snapshot",), coords={"snapshot": sns}
            )

            # Sum over snapshots within each window for all generators at once
            # This creates constraints: sum(p[gen, t] for t in window) == 0 for each gen and window
            window_sums = load_shift_vars.groupby(window_coord).sum("snapshot")

            # Add all constraints at once (vectorized across generators and windows)
            n.model.add_constraints(
                window_sums == 0, name=f"DSR_load_shift_{time_window_hours}h"
            )

            logger.info(
                f"Applied {time_window_hours}h energy balance constraints for {len(dsr_names)} load shifting DSR generators ({num_windows} windows)"
            )

    # Peak shaving constraints (daily hours) - applies to committable DSR generators
    committable_dsrs = explicit_dsrs[explicit_dsrs.committable]
    if not committable_dsrs.empty:
        # Maximum daily hours per generator
        max_hours = xr.DataArray(
            committable_dsrs["max_daily_hours"].values,
            dims=("Generator-com",),
            coords={"Generator-com": committable_dsrs.index.values},
        )

        # Status variables for those generators
        status_var = n.model.variables["Generator-status"]
        status_var_dsr = status_var.sel({"Generator-com": committable_dsrs.index})

        # Group by day and sum
        daily_status = status_var_dsr.groupby("snapshot.date").sum("snapshot")
        # dims: (date, Generator-com)

        # Broadcast max_hours across days so shapes match
        max_hours_daily = max_hours.expand_dims({"date": daily_status.coords["date"]})

        # Add constraint
        n.model.add_constraints(
            daily_status, "<=", max_hours_daily, name="DSR_peak_shaving_max_hours"
        )

        logger.info(
            f"Applied peak shaving constraints using status variables for {len(committable_dsrs)} committable DSR generators"
        )


def get_must_run_generators(n: pypsa.Network) -> pd.Index:
    """
    Return generators with static or timeseries must-run capacity > 0.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.

    Returns
    -------
    pd.Index
        Generators with a non-zero must-run requirement.
    """
    must_run_gens = []

    if (
        "must-run capacity" in n.generators.columns
        and "is_must_run" in n.generators.columns
    ):
        static_must_run = n.generators.index[
            (n.generators["must-run capacity"] > 0) & n.generators["is_must_run"]
        ]
        must_run_gens.extend(static_must_run.tolist())

    if (
        "must_run_capacity" in n.generators_t
        and not n.generators_t.must_run_capacity.empty
    ):
        must_run_gens.extend(n.generators_t.must_run_capacity.columns.tolist())

    # dict.fromkeys rather than set(): this index fixes the order in which the
    # must-run constraints enter the model, and a hash-ordered set makes the LP
    # differ between runs. See get_reserves_parameters for the same reasoning.
    return pd.Index(dict.fromkeys(must_run_gens))


def add_must_run_constraints(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Add must-run constraints for generators with must-run capacity > 0.

    The constraint enforces ``p + slack >= must_run_requirement`` when
    ``eva.must_run.enable_slack`` is true (the default), or a hard
    ``p >= must_run_requirement`` when it is false. ``must_run_requirement``
    is computed differently depending on the source:

    For time-varying capacity (from n.generators_t.must_run_capacity):
    - must_run_requirement = must_run_capacity_t * availability
    - Note: timeseries already incorporates the weighted_average_must_run_ratio

    For static capacity (from n.generators["must-run capacity"]):
    - must_run_constraint = must_run_capacity * weighted_average_must_run_ratio
    - must_run_requirement = must_run_constraint * availability
    - The weighted_average_must_run_ratio is from "weighted average must-run ratio"
      column (defaults to 1.0 if not present)

    The per-unit approach with weighted ratio is used by default for static capacity.
    The constraint uses rolling minimum availability to avoid conflicts with ramp
    constraints.

    The requirement is applied unconditionally to every must-run generator,
    committable or not. It is deliberately NOT multiplied by the
    ``Generator-status`` variable: scaling by status lets the solver dodge the
    requirement simply by keeping a unit offline, which serves less must-run
    capacity than the input data asks for.

    The requirement is never clipped either. Whatever cannot be served — because
    unit commitment keeps the unit offline (minimum up/down times, a day-ahead
    commitment locked into the intraday market), because reserve blocking,
    maintenance or forced outages ate the headroom, or because the must-run data
    is inconsistent with the capacity — is absorbed by a non-negative slack
    variable (``Generator-must_run_slack``) priced in the objective at
    ``eva.must_run.slack_penalty``; see
    :func:`add_must_run_slack_penalty_to_objective`. Unserved must-run capacity
    is therefore always priced rather than silently dropped or infeasible, and
    its size is readable from the slack solution.

    Setting ``eva.must_run.enable_slack`` to false drops the slack variable and
    falls back to the hard constraint the slack replaced: the model will be
    infeasible whenever the requirement cannot be met (e.g. a ramp-limited
    generator coming off a zero-availability hour cannot reach a must-run floor
    set for the next snapshot).

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots to be processed.
    """
    # Find generators with must-run capacity (either static or timeseries)
    must_run_gens = get_must_run_generators(n)

    if must_run_gens.empty:
        return

    # Get generator power variables
    if "Generator-p" not in n.model.variables:
        return

    p_vars = n.model.variables["Generator-p"]
    gen_dim = str(p_vars.dims[1])

    # Filter to must-run generators that exist in the model
    valid_must_run_gens = must_run_gens.intersection(p_vars.coords[gen_dim].values)

    if valid_must_run_gens.empty:
        return

    # Select power variables
    must_run_p_vars = p_vars.sel(snapshot=sns).sel(**{gen_dim: valid_must_run_gens})

    # Availability-derated must-run requirement (rolling-min availability applied)
    must_run_requirement = must_run_requirement_mw(n, sns, valid_must_run_gens)

    requirement = xr.DataArray(
        must_run_requirement.reindex(index=sns, columns=valid_must_run_gens).to_numpy(),
        coords={"snapshot": sns, gen_dim: valid_must_run_gens},
        dims=("snapshot", gen_dim),
    )

    enable_slack = n.meta.get("eva", {}).get("must_run", {}).get("enable_slack", True)

    if not enable_slack:
        n.model.add_constraints(
            must_run_p_vars >= requirement,
            name="Generator_must_run_capacity",
        )
        return

    # Slack for the must-run requirement, on exactly the same (snapshot, generator)
    # dimensions as Generator-p so the sum below stays element-wise instead of
    # broadcasting into a cross product.
    slack_var = n.model.add_variables(
        coords={"snapshot": sns, gen_dim: valid_must_run_gens},
        dims=("snapshot", gen_dim),
        name=MUST_RUN_SLACK_VAR,
        lower=0,
        upper=float("inf"),
    )

    n.model.add_constraints(
        must_run_p_vars + slack_var >= requirement,
        name="Generator_must_run_capacity",
    )


def add_must_run_slack_penalty_to_objective(n: pypsa.Network) -> None:
    """
    Penalise unserved must-run capacity in the objective function.

    The penalty per MWh of unserved must-run capacity is read from
    ``n.meta["eva"]["must_run"]["slack_penalty"]``. The slack variable is summed
    over all of its dimensions before being added to the objective, so no
    dimension of the slack broadcasts into the existing objective terms.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    """
    if MUST_RUN_SLACK_VAR not in n.model.variables:
        logger.debug(
            f"Skipping add_must_run_slack_penalty_to_objective: no {MUST_RUN_SLACK_VAR} variable in model."
        )
        return

    slack_var = n.model.variables[MUST_RUN_SLACK_VAR]
    penalty_cost = n.meta["eva"]["must_run"]["slack_penalty"]

    n.model.objective += penalty_cost * slack_var.sum()


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
    add_storage = (
        n.meta.get("eva", {})
        .get("storage_planning", {})
        .get("storage_constraints_enabled", True)
    )
    if add_storage:
        logger.info("Applying technical constraints in market model...")
        update_technical_constraints(n, sns)

    update_storage_plan(n, sns)

    add_storage_soc_slack_penalty_to_objective(n, sns)

    # DSR explicit constraints
    add_explicit_dsr_constraints(n, sns)

    # DSR implicit constraints: floor intraday storage dispatch at fraction f of
    # the day-ahead schedule (self-guards; only active on intraday networks).
    add_implicit_dsr_fixing_constraints(n, sns)

    # Must-run constraints
    add_must_run_constraints(n, sns)

    add_must_run_slack_penalty_to_objective(n)

    market_interconnection = getattr(n, "_market_interconnection", None)
    if market_interconnection == "fbmc":
        add_fbmc_constraint(n, sns)
    elif market_interconnection == "physical" or market_interconnection == "ntc":
        # No additional constraints for physical or ntc interconnection
        pass
    else:
        logger.warning(f"Unknown market interconnection type: {market_interconnection}")

    # Add fixed commitment constraints if integer status data is available
    add_fixed_commitment_constraints(n, sns)

    # Fix rolling horizon UC constraints for linearized UC
    # Only applies when using linearized UC at rolling horizon boundaries
    linearized_uc = n.meta["solving"]["options"]["linearized_unit_commitment"]
    if linearized_uc:
        fix_uc_transition_constraints_at_rolling_horizon_boundary(n, sns)
        fix_uc_min_uptime_downtime_constraints_at_rolling_horizon_boundary(n, sns)

    # Check if ramping constraints should be disabled for ramping shortfall calculation
    disable_ramping = n.meta.get("disable_ramping_for_shortfall", False)

    is_day_ahead = n.meta["is_day_ahead"]
    reserves_option = n.meta["eva"]["reserves"]["option"]
    add_reserves = reserves_option == "co-optimised"

    if not is_day_ahead and add_reserves:
        logger.debug("Intra-day network. Skipping reserves constraints.")
        add_storage_reserves_limit_constraints_intraday_market(n, sns)
    if is_day_ahead and add_reserves:
        logger.info("Applying reserves constraints to day-ahead market")
        # Add reserves requirements variables and constraints (5.1)
        add_reserves_requirements(n, sns)

        # Adapt generator power constraints with reserves (5.2) to (5.5)
        adapt_generator_power_constraints_with_reserves(n, sns)

        # Add storage reserves limit constraints (5.6) and (5.7)
        add_storage_reserves_limit_constraints(n, sns)

        # Add reserves ramping constraints (5.8) and (5.9) - skip if ramping disabled
        if not disable_ramping:
            add_generators_reserves_ramping_constraint(n, sns)
        else:
            logger.info(
                "Skipping reserves ramping constraints (disabled for ramping shortfall)"
            )

        # Add storage constraints for soc (5.10) and (5.11)
        add_storage_reserves_soc_constraint(n, sns)

        # Adapt ramping constraints of generators to consider reserves (5.12) to (5.15) - skip if ramping disabled
        if not disable_ramping:
            adapt_generator_ramp_constraints_with_reserves(n, sns)
        else:
            logger.info(
                "Skipping generator ramp constraints adaptation (disabled for ramping shortfall)"
            )

        # Adapt objective function to include reserve procurement costs
        add_reserves_penalty_to_objective(n, sns)


def apply_derated_reserves(n: pypsa.Network) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Adjust p_max_pu and p_min_pu of generators and storage units by their
    pre-assigned static reserve amounts (derated mode).

    Any column matching *_up_amount reduces p_max_pu by value / p_nom.
    Any column matching *_down_amount raises p_min_pu by value / p_nom.
    Adjustments are expressed in per-unit (divided by p_nom).

    The static amounts are sized against full availability
    (patch_reserve_amounts_exceed_pu_bounds at network build time ensures
    combined up+down derating fits min(p_max_pu - p_min_pu) there). Maintenance
    and forced outages applied afterwards shrink the per-snapshot headroom, so
    on snapshots where the combined derating no longer fits, both directions
    are scaled down proportionally to the remaining headroom. This guarantees
    new p_max_pu >= new p_min_pu at every snapshot while preserving the
    up/down ratio.

    Parameters
    ----------
    n : pypsa.Network

    Returns
    -------
    dict
        Keys "generators" and "storage_units", each a dict
        {"p_max_pu": original_df, "p_min_pu": original_df} for restoration.
    """
    originals: dict[str, dict[str, pd.DataFrame]] = {}

    for component_name, c in [
        ("generators", "Generator"),
        ("storage_units", "StorageUnit"),
    ]:
        static = n.static(c)
        if static.empty:
            originals[component_name] = {}
            continue

        p_nom = static["p_nom"]
        p_max_pu = n.get_switchable_as_dense(c, "p_max_pu").astype(float)
        p_min_pu = n.get_switchable_as_dense(c, "p_min_pu").astype(float)

        originals[component_name] = {
            "p_max_pu": p_max_pu.copy(),
            "p_min_pu": p_min_pu.copy(),
        }

        directions: list[str] = n.meta["eva"]["reserves"]["direction"]

        up_cols = (
            [col for col in static.columns if col.endswith("_up_amount")]
            if "up" in directions
            else []
        )
        down_cols = (
            [col for col in static.columns if col.endswith("_down_amount")]
            if "down" in directions
            else []
        )

        zero = pd.Series(0.0, index=static.index)
        total_up = sum(
            (static[col].fillna(0.0) for col in up_cols),
            zero,
        )
        total_down = sum(
            (static[col].fillna(0.0) for col in down_cols),
            zero,
        )

        up_pu = total_up.div(p_nom).fillna(0.0)
        down_pu = total_down.div(p_nom).fillna(0.0)

        if c == "StorageUnit":
            # Unlike Generator (where p_max_pu and p_min_pu bound the same
            # dispatch variable), StorageUnit's p_max_pu bounds p_dispatch
            # (>= 0) and p_min_pu bounds p_store (<= 0) independently via
            # PyPSA's get_bounds_pu (p_store upper = -p_min_pu * p_nom). Each
            # direction must be scaled against its own bound crossing zero,
            # not against the combined interval -- otherwise down-derating can
            # push p_min_pu positive (p_store upper negative) even while
            # p_max_pu >= p_min_pu still holds, producing an infeasible fixed
            # p_store bound.
            scale_up = (
                p_max_pu.div(up_pu.where(up_pu > 0.0), axis=1)
                .clip(lower=0.0, upper=1.0)
                .fillna(1.0)
            )
            scale_down = (
                (-p_min_pu)
                .div(down_pu.where(down_pu > 0.0), axis=1)
                .clip(lower=0.0, upper=1.0)
                .fillna(1.0)
            )
            scaled = (scale_up < 1.0) | (scale_down < 1.0)
            if scaled.any().any():
                scaled_units = scale_up.columns[scaled.any()].tolist()
                logger.warning(
                    f"apply_derated_reserves: reserve derating exceeds the "
                    f"per-snapshot p_max_pu/p_min_pu headroom for "
                    f"{int(scaled.sum().sum())} {component_name}-snapshot "
                    f"pairs; scaling amounts down proportionally to keep "
                    f"p_max_pu >= 0 and p_min_pu <= 0. "
                    f"Affected units: {scaled_units}"
                )
            new_max = p_max_pu - scale_up.mul(up_pu, axis=1)
            new_min = p_min_pu + scale_down.mul(down_pu, axis=1)
        else:
            # Scale both directions down proportionally on snapshots where
            # reduced availability (maintenance, forced outages) leaves less
            # headroom than the combined static derating, so that
            # new_max >= new_min everywhere.
            combined_pu = up_pu + down_pu
            headroom = (p_max_pu - p_min_pu).clip(lower=0.0)
            scale = (
                headroom.div(combined_pu.where(combined_pu > 0.0), axis=1)
                .clip(upper=1.0)
                .fillna(1.0)
            )
            scaled = scale < 1.0
            if scaled.any().any():
                scaled_units = scale.columns[scaled.any()].tolist()
                logger.warning(
                    f"apply_derated_reserves: combined reserve derating exceeds the "
                    f"per-snapshot pu-bound headroom for {int(scaled.sum().sum())} "
                    f"{component_name}-snapshot pairs; scaling amounts down "
                    f"proportionally to the remaining headroom. "
                    f"Affected units: {scaled_units}"
                )

            new_max = p_max_pu - scale.mul(up_pu, axis=1)
            new_min = p_min_pu + scale.mul(down_pu, axis=1)

        dynamic = n.dynamic(c)
        dynamic["p_max_pu"] = new_max
        dynamic["p_min_pu"] = new_min

    return originals


def add_fixed_reserve_timeseries(n: pypsa.Network) -> None:
    """
    Broadcast *_amount columns as constant timeseries into generators_t / storage_units_t.

    For every column named ``{rt}_{direction}_amount`` present on a component's
    static DataFrame, a constant timeseries ``r_{rt}_{direction}`` is written into
    the corresponding dynamic store.  Used in derated mode so that downstream
    intra-day blocking has the same data format as co-optimised mode.

    Reserve types are determined dynamically from the columns present — no
    timeseries are created if no ``*_amount`` columns exist.

    Parameters
    ----------
    n : pypsa.Network
    """
    for c in ["Generator", "StorageUnit"]:
        static = n.static(c)
        dynamic = n.dynamic(c)
        if static.empty:
            continue
        directions: list[str] = n.meta["eva"]["reserves"]["direction"]
        amount_cols = [
            col
            for col in static.columns
            if col.endswith("_up_amount")
            and "up" in directions
            or col.endswith("_down_amount")
            and "down" in directions
        ]
        for col_name in amount_cols:
            stem = col_name[: -len("_amount")]  # e.g. "FCR_up"
            ts_name = f"r_{stem}"  # e.g. "r_FCR_up"
            values = static[col_name].fillna(0.0)
            dynamic[ts_name] = pd.DataFrame(
                {asset: values[asset] for asset in static.index},
                index=n.snapshots,
            )


def apply_generators_reserve_blocking_id(n, n_da):
    """
    Adjust p_max_pu and p_min_pu in the intraday network `n` to reflect
    a fraction of the capacity reserved in the day-ahead network `n_da`
    for different reserve types using their specific id_blocked fractions.

    Supports two blocking methods:
    - "fraction": Static id_blocked fraction (original behavior)
    - "imbalance": Dynamic blocking based on forecast errors (FRR only)

    Only applied to generators. Storage units have their own constraints.

    Parameters
    ----------
    n : pypsa.Network
        Intraday network to modify.
    n_da : pypsa.Network
        Day-ahead network containing reserve bookings and reserve parameters.
    """

    # Get reserve parameters from day-ahead network buses
    reserves_parameters = get_reserves_parameters(n_da)
    if not reserves_parameters:
        logger.debug(NO_RESERVES_DEBUG_MESSAGE)
        return n

    # Get reserve types from parameters (already extracted from bus columns)
    reserve_types = list(reserves_parameters.keys())

    if not reserve_types:
        logger.warning(
            "No valid reserve types found in both bus columns and parameters. Skipping reserve blocking."
        )
        return n

    # Check if FRR uses dynamic blocking method
    dynamic_frr_blocking = {}
    frr_blocking_method = None
    if "FRR" in reserve_types:
        frr_blocking_method = reserves_parameters["FRR"]["reserve_blocking_method"]

    if frr_blocking_method == "imbalance":
        logger.info("Using dynamic imbalance-based blocking for FRR reserves")
        # Get or calculate dynamic per-bus, per-snapshot blocking fractions
        dynamic_frr_blocking["up"], dynamic_frr_blocking["down"] = (
            get_or_calculate_frr_blocking(n_da, n)
        )

    c = "Generator"

    static = n.static(c)
    dynamic = n.dynamic(c)
    dynamic_da = n_da.dynamic(c)

    directions: list[str] = n_da.meta["eva"]["reserves"]["direction"]

    for dir in directions:
        bound = "p_max_pu" if dir == "up" else "p_min_pu"

        idx_list = []
        for rtype in reserve_types:
            # If infeasibilty was encountered, no reserves values are available
            if f"r_{rtype}_{dir}" not in dynamic_da:
                dynamic_da[f"r_{rtype}_{dir}"] = pd.DataFrame()
            idx_list.append(dynamic_da[f"r_{rtype}_{dir}"].columns)
        # Union all indices from the list
        idx = pd.Index([])
        for index in idx_list:
            idx = idx.union(index)

        if idx.empty:
            continue

        # Compute total reserved capacity with individual fractions
        # For FRR with imbalance method: use dynamic per-bus, per-snapshot fractions
        # For other reserves (FCR, or FRR with fraction method): use static fractions
        total_reserved = 0
        for rtype in reserve_types:
            # Get reserves from DA network and align with ID network snapshots
            da_reserves = dynamic_da[f"r_{rtype}_{dir}"].reindex(
                index=n.snapshots, columns=idx, fill_value=0.0
            )

            # Determine blocking fraction
            if rtype == "FRR" and frr_blocking_method == "imbalance":
                # Use dynamic per-bus blocking fractions
                # Map generator buses to blocking fractions - VECTORIZED
                gen_buses = static.loc[idx, "bus"]
                # Vectorized approach: use apply on the bus column to map to blocking fractions
                blocking_fractions = pd.DataFrame(
                    {gen: dynamic_frr_blocking[dir][gen_buses[gen]] for gen in idx},
                    index=n.snapshots,
                )
                reserve_contribution = da_reserves * blocking_fractions
            else:
                # Use static fraction from config (original behavior)
                id_blocked_fraction = reserves_parameters[rtype]["id_blocked"]
                reserve_contribution = da_reserves * id_blocked_fraction

            total_reserved += reserve_contribution

        # Apply the original clipping
        reserved = total_reserved.clip(  # ty: ignore[possibly-missing-attribute]
            lower=0.0
        )

        p_nom = static.p_nom[idx]
        pu_adjustment = reserved.div(p_nom).fillna(0.0)
        if (pu_adjustment >= 1.0).any().any() >= 1.0:
            logger.info(
                "Reserve allocation in DA was greater than nominal capacity. "
                "Clipping to max allowable to prevent infeasibilities"
            )
            pu_adjustment = pu_adjustment.clip(lower=0.0, upper=1.0)

        # Ensure the time-dependent bound exists
        dynamic[bound] = n.get_switchable_as_dense(c, bound).astype(float)
        current = dynamic[bound]

        # Create timeseries for reserve generators with original values as base
        reserve_current = current[idx].astype(float)

        # Get the opposite bound values for cross-bound constraints
        opposite_bound = "p_min_pu" if dir == "up" else "p_max_pu"
        opposite_values = as_dense(n, c, opposite_bound, idx).astype(float)
        if dir == "up":
            # For up direction: subtract adjustment from original p_max_pu
            # Ensure updated p_max_pu doesn't go below p_min_pu, or below
            # the availability-derated must-run floor: a DA-booked up
            # reserve must not be blocked out of a must-run unit's own
            # must-run headroom, or add_must_run_constraints later hits a
            # hard infeasibility against this same p_max_pu.
            must_run_gens = get_must_run_generators(n).intersection(idx)
            must_run_pu_floor = pd.DataFrame(-1.0, index=n.snapshots, columns=idx)
            if len(must_run_gens) > 0:
                must_run_mw = must_run_requirement_mw(n, n.snapshots, must_run_gens)
                must_run_pu_floor[must_run_gens] = must_run_mw.div(p_nom[must_run_gens])

            floor = opposite_values.clip(lower=must_run_pu_floor)
            binding = (must_run_pu_floor > opposite_values) & (
                reserve_current.subtract(pu_adjustment, axis=1) < must_run_pu_floor
            )
            if binding.any().any():
                n_binding = int(binding.sum().sum())
                logger.warning(
                    f"Must-run floor limited ID reserve blocking for {n_binding} "
                    f"generator-timestep pair(s): blocking would have pushed "
                    f"p_max_pu below the must-run requirement, so the reserve "
                    f"amount was left unblocked on those pairs instead."
                )

            updated = reserve_current.subtract(pu_adjustment, axis=1)
            updated = updated.clip(lower=floor, upper=1.0)
        else:
            # For down direction: add adjustment to original p_min_pu
            # Ensure updated p_min_pu doesn't exceed p_max_pu
            updated = reserve_current.add(pu_adjustment, axis=1)
            updated = updated.clip(lower=-1.0, upper=opposite_values)

        updated = updated.astype(float)
        dynamic[bound].loc[updated.index, updated.columns] = updated

    logger.info(
        f"Applied Generators reserve dispatch blocking for {len(reserve_types)} reserve types: {', '.join(reserve_types)}"
    )
    return n


def _correct_ramp_feasibility(
    modified: pd.DataFrame,
    p_min_mod: pd.DataFrame,
    p_max_mod: pd.DataFrame,
    direction: str,
    tol_ramp: float,
) -> pd.DataFrame:
    """
    Enforce the LP feasibility invariant for ramp constraints in the ID model.

    When the DA produces near-zero status values (LP cascade artefacts), the resulting
    p_min / p_max bounds in the ID model can create an infeasible triple with the ramp
    limit.  This function increases the ramp limit by the minimum amount needed to restore
    feasibility, but only when the required adjustment is within ``tol_ramp`` (p.u. of
    p_nom).  Larger adjustments indicate a genuine model inconsistency and are skipped
    with a warning.

    Parameters
    ----------
    modified : pd.DataFrame
        Time-varying ramp limit DataFrame (index = snapshots, columns = generators).
    p_min_mod : pd.DataFrame
        ``p_min_pu * status`` for all committable generators.
    p_max_mod : pd.DataFrame
        ``p_max_pu * status`` for all committable generators.
    direction : str
        ``"down"`` or ``"up"``.
    tol_ramp : float
        Maximum adjustment (p.u. of p_nom) applied silently.

    Returns
    -------
    pd.DataFrame
        Corrected ramp limit DataFrame.
    """
    if direction == "down":
        # ramp_down[t] must be >= p_min_mod[t-1] - p_max_mod[t]
        min_feasible = (p_min_mod.shift(1).fillna(0) - p_max_mod).clip(lower=0.0)
        attr_name = "ramp_limit_down"
    else:
        # ramp_up[t] must be >= p_min_mod[t] - p_max_mod[t-1]
        min_feasible = (p_min_mod - p_max_mod.shift(1).fillna(0)).clip(lower=0.0)
        attr_name = "ramp_limit_up"

    adjustment_needed = (min_feasible - modified).clip(lower=0.0)
    within_tolerance = adjustment_needed <= tol_ramp

    n_adjusted = int((within_tolerance & (adjustment_needed > 0)).values.sum())
    n_skipped = int((~within_tolerance & (adjustment_needed > 0)).values.sum())

    if n_adjusted > 0:
        max_adj_pu = float(
            adjustment_needed[within_tolerance & (adjustment_needed > 0)].max().max()
        )
        logger.info(
            f"align_ramp_limits: increased {attr_name} for {n_adjusted} "
            f"generator-timestep pair(s) to enforce ID feasibility invariant "
            f"(max adjustment {max_adj_pu:.2e} p.u., within {tol_ramp:.0e} p_nom tolerance)"
        )
    if n_skipped > 0:
        worst = adjustment_needed[~within_tolerance].stack().nlargest(5).to_string()
        logger.warning(
            f"align_ramp_limits: {n_skipped} generator-timestep pair(s) require "
            f"a {attr_name} adjustment exceeding {tol_ramp:.0e} p_nom — possible "
            f"model inconsistency, no adjustment applied. Largest cases:\n{worst}"
        )

    adjusted = modified.clip(lower=min_feasible)
    return modified.where(~within_tolerance, adjusted)


def align_ramp_limits(
    n: pypsa.Network,
    n_da: pypsa.Network,
    tol_ramp: float = 1e-6,
    id_free_commitment_carriers: list[str] | None = None,
):
    """
    Update the p_min_pu, p_max_pu, ramp_limit_up and ramp_limit_down based on DA
    commitment results

    Generators belonging to ``id_free_commitment_carriers`` are excluded, so that
    they remain free to (re)commit in the intra-day stage rather than being locked
    to their day-ahead commitment status.

    Parameters
    ----------
    n : pypsa.Network
        Intraday network to modify.
    n_da : pypsa.Network
        Day-ahead network containing reserve bookings and reserve parameters.
    tol_ramp : float, optional
        Tolerance used when correcting ramp feasibility.
    id_free_commitment_carriers : list[str] | None, optional
        Carriers whose committable generators are free to (re)commit in the
        intra-day stage and are therefore excluded from commitment locking. When
        ``None``, only ``DSR_explicit`` generators are excluded (legacy behaviour).
    """
    if id_free_commitment_carriers is None:
        id_free_commitment_carriers = ["DSR_explicit"]

    com_i = n_da.get_committable_i("Generator")
    free_gen = n_da.generators.index[
        n_da.generators.carrier.isin(id_free_commitment_carriers)
    ]
    com_i = com_i.difference(free_gen)
    status = n_da.generators_t.status.reindex(columns=com_i, fill_value=1).clip(lower=0)
    status_t_minus_1 = status.shift(1).fillna(0)

    def dense(attr: str) -> pd.DataFrame:
        return as_dense(n, "Generator", attr, com_i)

    p_min_mod = status * dense("p_min_pu")
    p_max_mod = status * dense("p_max_pu")

    for attr in ["p_min_pu", "p_max_pu", "ramp_limit_up", "ramp_limit_down"]:
        modified = status * dense(attr)
        if attr == "ramp_limit_up":
            ramp_limit_start_up = dense("ramp_limit_start_up")
            modified = status_t_minus_1 * dense(attr)
            modified = modified + status.diff() * ramp_limit_start_up
            modified = _correct_ramp_feasibility(
                modified, p_min_mod, p_max_mod, "up", tol_ramp
            )
        if attr == "ramp_limit_down":
            ramp_limit_shut_down = dense("ramp_limit_shut_down")
            modified = modified - status.diff() * ramp_limit_shut_down
            modified = _correct_ramp_feasibility(
                modified, p_min_mod, p_max_mod, "down", tol_ramp
            )
        n.generators_t[attr].loc[:, com_i] = modified


def adjust_ramp_limits_for_reserves(n: pypsa.Network, n_da: pypsa.Network):
    """
    Adjust ramp limits in the intraday network to account for reserved capacity.

    Supports two blocking methods:
    - "fraction": Static id_blocked fraction (original behavior)
    - "imbalance": Dynamic blocking based on forecast errors (FRR only)

    Reduces both ramp_limit_up and ramp_limit_down by the amount reserved
    multiplied by id_fraction divided by p_nom for each unit.

    Parameters
    ----------
    n : pypsa.Network
        Intraday network to modify.
    n_da : pypsa.Network
        Day-ahead network containing reserve bookings and reserve parameters.
    """
    # Get reserve parameters from day-ahead network buses
    reserves_parameters = get_reserves_parameters(n_da)
    if not reserves_parameters:
        logger.debug(NO_RESERVES_DEBUG_MESSAGE)
        return n

    # Get reserve types from parameters (already extracted from bus columns)
    valid_reserve_types = list(reserves_parameters.keys())

    if not valid_reserve_types:
        return n

    # Check if FRR uses dynamic blocking method
    dynamic_frr_blocking = {}
    frr_blocking_method = None
    if "FRR" in valid_reserve_types:
        frr_blocking_method = reserves_parameters["FRR"]["reserve_blocking_method"]

    if frr_blocking_method == "imbalance":
        logger.info(
            "Using dynamic imbalance-based blocking for FRR reserves in ramp limits"
        )
        dynamic_frr_blocking["up"], dynamic_frr_blocking["down"] = (
            get_or_calculate_frr_blocking(n_da, n)
        )

    all_gens = n.generators.index

    for ramp_attr in ["ramp_limit_up", "ramp_limit_down"]:
        direction = "up" if "up" in ramp_attr else "down"

        # Calculate total reserved capacity reduction per generator.
        # Only generators that actually appear in at least one reserve variable
        # will receive a non-zero reduction; others are left untouched.
        total_reduction = pd.DataFrame(0.0, index=n.snapshots, columns=all_gens)

        for reserve_type in valid_reserve_types:
            reserve_var = f"r_{reserve_type}_{direction}"

            # Check if this reserve type exists in DA network
            if reserve_var in n_da.generators_t:
                # Get reserves from DA network
                da_reserves = n_da.generators_t[reserve_var]

                # Find generators that have reserves
                reserve_gens = da_reserves.columns.intersection(all_gens)

                if not reserve_gens.empty:
                    # Get p_nom for normalization
                    p_nom = n_da.generators.p_nom[reserve_gens]

                    # Determine blocking fraction
                    if reserve_type == "FRR" and frr_blocking_method == "imbalance":
                        # Use dynamic per-bus blocking fractions - VECTORIZED
                        gen_buses = n.generators.loc[reserve_gens, "bus"]
                        # Vectorized approach: dictionary comprehension
                        blocking_fractions = pd.DataFrame(
                            {
                                gen: dynamic_frr_blocking[direction][gen_buses[gen]]
                                for gen in reserve_gens
                            },
                            index=n.snapshots,
                        )

                        # Calculate per-unit reduction with dynamic fractions
                        reserve_pu = (
                            da_reserves[reserve_gens]
                            .reindex(index=n.snapshots, fill_value=0.0)
                            .divide(p_nom.replace(0.0, np.nan), axis=1)
                            * blocking_fractions
                        ).fillna(0.0)
                    else:
                        # Use static fraction from config (original behavior)
                        id_fraction = reserves_parameters[reserve_type]["id_blocked"]
                        reserve_pu = (
                            da_reserves[reserve_gens]
                            .reindex(index=n.snapshots, fill_value=0.0)
                            .divide(p_nom.replace(0.0, np.nan), axis=1)
                            * id_fraction
                        ).fillna(0.0)

                    # Add to total reduction
                    total_reduction[reserve_gens] += reserve_pu

        # Only adapt generators that actually carry reserves; leave all others untouched.
        gens_with_reserves = total_reduction.columns[(total_reduction != 0).any()]
        if gens_with_reserves.empty:
            continue

        # Get current ramp limits for reserve-carrying generators only.
        current_ramp = as_dense(n, "Generator", ramp_attr, gens_with_reserves)

        # Only adjust generators that have existing ramp limits (not NaN).
        # Generators without ramp limits remain unconstrained in ramping speed,
        # even if they provide reserves. Reserves only constrain capacity (via p_min_pu/p_max_pu),
        # not ramping speed. This preserves the flexibility of fast-ramping resources like
        # batteries, DSR, and other units without physical ramping constraints.
        gens_with_ramp_limits = current_ramp.columns[current_ramp.notna().any()]
        if gens_with_ramp_limits.empty:
            continue

        current_ramp = current_ramp[gens_with_ramp_limits]
        modified = (current_ramp - total_reduction[gens_with_ramp_limits]).clip(
            lower=0.0
        )

        # Write back only the affected generators; use .loc to avoid clobbering
        # generators without reserves (mirrors the pattern in align_ramp_limits).
        if ramp_attr not in n.generators_t or n.generators_t[ramp_attr].empty:
            n.generators_t[ramp_attr] = modified
        else:
            n.generators_t[ramp_attr].loc[:, modified.columns] = modified

    return n


def add_storage_reserves_to_be_blocked(n: pypsa.Network, n_da: pypsa.Network) -> None:
    """
    Add total_reserve_up and total_reserve_down columns to storage_units and storage_units_t
    in the intraday network `n` based on the reserves booked in the day-ahead network `n_da`.

    Supports two blocking methods:
    - "fraction": Static id_blocked fraction (original behavior)
    - "imbalance": Dynamic blocking based on forecast errors (FRR only)

    Parameters
    ----------
    n : pypsa.Network
        Intraday network to modify.
    n_da : pypsa.Network
        Day-ahead network containing reserve bookings and reserve parameters.
    """
    # Get reserve parameters from day-ahead network buses
    reserves_parameters = get_reserves_parameters(n_da)
    if not reserves_parameters:
        logger.debug(NO_RESERVES_DEBUG_MESSAGE)

    # Get reserve types from parameters (already extracted from bus columns)
    reserve_types = list(reserves_parameters.keys())

    if not reserve_types:
        logger.warning(
            "No valid reserve types found in both bus columns and parameters. Skipping storage reserve blocking."
        )
        return

    # Check if FRR uses dynamic blocking method
    dynamic_frr_blocking = {}
    frr_blocking_method = None
    if "FRR" in reserve_types:
        frr_blocking_method = reserves_parameters["FRR"]["reserve_blocking_method"]

    if frr_blocking_method == "imbalance":
        # Reuse the same dynamic blocking fractions calculated for generators
        # (cached on network from first call, avoiding recalculation)
        logger.info(
            "Using dynamic imbalance-based blocking for FRR reserves in storage"
        )
        dynamic_frr_blocking["up"], dynamic_frr_blocking["down"] = (
            get_or_calculate_frr_blocking(n_da, n)
        )

    c = "StorageUnit"

    static = n.static(c)
    dynamic_da = n_da.dynamic(c)

    # Initialize total_reserve_up and total_reserve_down DataFrames
    total_reserve_up = pd.DataFrame(0.0, index=n.snapshots, columns=static.index)
    total_reserve_down = pd.DataFrame(0.0, index=n.snapshots, columns=static.index)

    for rtype in reserve_types:
        # If infeasibilty was encountered, no reserves values are available
        if f"r_{rtype}_up" not in dynamic_da:
            dynamic_da[f"r_{rtype}_up"] = pd.DataFrame()
        if f"r_{rtype}_down" not in dynamic_da:
            dynamic_da[f"r_{rtype}_down"] = pd.DataFrame()

        # Get reserves from DA network and align with ID network snapshots
        da_reserve_up = dynamic_da[f"r_{rtype}_up"].reindex(
            index=n.snapshots, columns=static.index, fill_value=0.0
        )
        da_reserve_down = dynamic_da[f"r_{rtype}_down"].reindex(
            index=n.snapshots, columns=static.index, fill_value=0.0
        )

        # Determine blocking fraction
        if rtype == "FRR" and frr_blocking_method == "imbalance":
            # Use dynamic per-bus blocking fractions
            # Map storage unit buses to blocking fractions
            storage_buses = static["bus"]
            # Create time-varying blocking fraction DataFrames aligned with storage units - VECTORIZED
            blocking_fractions_up = pd.DataFrame(
                {
                    stor: dynamic_frr_blocking["up"][storage_buses[stor]]
                    for stor in static.index
                },
                index=n.snapshots,
            )
            blocking_fractions_down = pd.DataFrame(
                {
                    stor: dynamic_frr_blocking["down"][storage_buses[stor]]
                    for stor in static.index
                },
                index=n.snapshots,
            )

            # Calculate the blocked reserves contribution for this type
            blocked_up = da_reserve_up * blocking_fractions_up
            blocked_down = da_reserve_down * blocking_fractions_down
        else:
            # Use static fraction from config (original behavior)
            id_blocked_fraction = reserves_parameters[rtype]["id_blocked"]
            blocked_up = da_reserve_up * id_blocked_fraction
            blocked_down = da_reserve_down * id_blocked_fraction

        # Accumulate into total reserves
        total_reserve_up += blocked_up
        total_reserve_down += blocked_down

    # Assign to network
    n.storage_units_t["total_reserve_up"] = total_reserve_up
    n.storage_units_t["total_reserve_down"] = total_reserve_down


def get_or_calculate_frr_blocking(
    n_da: pypsa.Network,
    n_id: pypsa.Network,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get cached FRR blocking fractions from network or calculate if not present.

    This function checks if the blocking fractions have already been calculated and
    attached to n_id.buses_t. If they exist, it returns them directly. Otherwise,
    it calls adjust_frr_blocked_from_forecast_error to calculate them.

    Parameters
    ----------
    n_da : pypsa.Network
        Day-ahead network with forecast data and allocated FRR reserves
    n_id : pypsa.Network
        Intraday network with updated forecast data

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - id_blocked_up: Per-bus, per-snapshot blocking fractions for upward reserves (0-1)
        - id_blocked_down: Per-bus, per-snapshot blocking fractions for downward reserves (0-1)
    """
    # Check if blocking fractions already exist on the network
    if "frr_blocked_up" in n_id.buses_t and "frr_blocked_down" in n_id.buses_t:
        logger.debug("Reusing cached FRR blocking fractions from network")
        return n_id.buses_t["frr_blocked_up"], n_id.buses_t["frr_blocked_down"]

    # Calculate and cache on network
    return adjust_frr_blocked_from_forecast_error(n_da, n_id)


def adjust_frr_blocked_from_forecast_error(
    n_da: pypsa.Network,
    n_id: pypsa.Network,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate dynamic FRR blocking fractions based on forecast errors between DA and ID.

    Uses residual load to compute forecast error at each bus. Residual load captures
    the forecast uncertainty by measuring: demand - renewable_generation_potential - must_run_supply.
    The blocking fraction represents how much of the day-ahead FRR reserves should be
    unavailable for intraday dispatch based on the forecast uncertainty.

    Parameters
    ----------
    n_da : pypsa.Network
        Day-ahead network with forecast data and allocated FRR reserves
    n_id : pypsa.Network
        Intraday network with updated forecast data

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - id_blocked_up: Per-bus, per-snapshot blocking fractions for upward reserves (0-1)
        - id_blocked_down: Per-bus, per-snapshot blocking fractions for downward reserves (0-1)

    Side Effects
    ------------
    Attaches the blocking fractions to n_id.buses_t:
        - frr_blocked_up: Dynamic blocking fractions for upward FRR reserves
        - frr_blocked_down: Dynamic blocking fractions for downward FRR reserves

    Notes
    -----
    Formula implementation:
        residual_load = demand - renewable_generation_potential - must_run_supply
        forecast_error[bus, t] = residual_load_id[bus, t] - residual_load_da[bus, t]

        id_blocked_up[bus, t] = min(1, max(0, 1 - forecast_error / available_FRR_up))
        id_blocked_down[bus, t] = min(1, max(0, 1 + forecast_error / available_FRR_down))

    When forecast_error > 0 (ID shows higher residual load than DA):
        - Less blocking of FRR_up as it can be used to cover upward imbalance in the market.
        - All FRR_down is blocked from being used in the market

    When forecast_error < 0 (ID shows lower residual load than DA):
        - All FRR_up is blocked from being used in the market.
        - Less blocking of FRR_down as it can be used to cover downward imbalance in the market.

    Uses vectorized operations for performance.
    """
    buses = n_id.buses.index
    snapshots = n_id.snapshots
    if "eva" in n_da.meta and "base_network" in n_da.meta["eva"]:
        renewable_carriers = n_da.meta["eva"]["base_network"]["renewable_carriers"]
    else:
        renewable_carriers = [
            "Solar (PV)",
            "Hydro - Run of River (Turbine)",
            "Wind Onshore",
            "Wind Offshore",
            "renewables",
        ]

    # Calculate residual load: demand - renewable_generation - must_run_supply
    residual_load_da = calculate_residual_load_df(n_da, renewable_carriers)
    residual_load_id = calculate_residual_load_df(n_id, renewable_carriers)

    # Calculate forecast error per bus (positive = higher residual load in ID)
    forecast_error = residual_load_id - residual_load_da

    # Calculate available FRR reserves from DA market (generators + storage)
    available_frr_up = pd.DataFrame(0.0, index=snapshots, columns=buses)
    available_frr_down = pd.DataFrame(0.0, index=snapshots, columns=buses)

    # Add generator FRR reserves
    if "r_FRR_up" in n_da.generators_t:
        gen_frr_up = n_da.generators_t["r_FRR_up"].reindex(
            index=snapshots, fill_value=0.0
        )
        gen_frr_up_per_bus = (
            gen_frr_up.T.groupby(n_da.generators.bus)
            .sum()
            .T.reindex(columns=buses, fill_value=0.0)
        )
        available_frr_up += gen_frr_up_per_bus

    if "r_FRR_down" in n_da.generators_t:
        gen_frr_down = n_da.generators_t["r_FRR_down"].reindex(
            index=snapshots, fill_value=0.0
        )
        gen_frr_down_per_bus = (
            gen_frr_down.T.groupby(n_da.generators.bus)
            .sum()
            .T.reindex(columns=buses, fill_value=0.0)
        )
        available_frr_down += gen_frr_down_per_bus

    # Add storage FRR reserves
    if "r_FRR_up" in n_da.storage_units_t:
        stor_frr_up = n_da.storage_units_t["r_FRR_up"].reindex(
            index=snapshots, fill_value=0.0
        )
        stor_frr_up_per_bus = (
            stor_frr_up.T.groupby(n_da.storage_units.bus)
            .sum()
            .T.reindex(columns=buses, fill_value=0.0)
        )
        available_frr_up += stor_frr_up_per_bus

    if "r_FRR_down" in n_da.storage_units_t:
        stor_frr_down = n_da.storage_units_t["r_FRR_down"].reindex(
            index=snapshots, fill_value=0.0
        )
        stor_frr_down_per_bus = (
            stor_frr_down.T.groupby(n_da.storage_units.bus)
            .sum()
            .T.reindex(columns=buses, fill_value=0.0)
        )
        available_frr_down += stor_frr_down_per_bus

    # Calculate blocking fractions with safety against division by zero
    # Replace zeros with NaN to avoid division errors, then fill with 1.0 (full blocking)
    available_frr_up_safe = available_frr_up.replace(0.0, np.nan)
    available_frr_down_safe = available_frr_down.replace(0.0, np.nan)

    # Formula: id_blocked = min(1, max(0, 1 - forecast_error / available_FRR))
    id_blocked_up = (1.0 - (forecast_error / available_frr_up_safe)).clip(
        lower=0.0, upper=1.0
    )
    id_blocked_down = (1.0 + (forecast_error / available_frr_down_safe)).clip(
        lower=0.0, upper=1.0
    )

    # Fill NaN (where FRR was zero) with 1.0 (full blocking - conservative approach)
    id_blocked_up = id_blocked_up.fillna(1.0)
    id_blocked_down = id_blocked_down.fillna(1.0)

    logger.info(
        f"Calculated dynamic FRR blocking fractions: "
        f"up mean={id_blocked_up.mean().mean():.3f}, down mean={id_blocked_down.mean().mean():.3f}"
    )

    # Attach blocking fractions to ID network for debugging and analysis
    n_id.buses_t["frr_blocked_up"] = id_blocked_up
    n_id.buses_t["frr_blocked_down"] = id_blocked_down

    return id_blocked_up, id_blocked_down


def fix_dsr_dispatch_from_da(
    n: pypsa.Network,
    n_da: pypsa.Network,
) -> None:
    """
    Fix a fraction of day-ahead DSR dispatch in the intraday market.

    This reflects DSR that only responds to day-ahead prices or requires
    sufficient lead times to be activated.

    For explicit DSR (generators):
    - Sets p_min_pu based on DA dispatch
    - Reduces p_max_pu to account for fixed dispatch

    For implicit DSR (storage units):
    - Adjusts charging and discharging capacity based on DA dispatch

    The fractions are read from:
    - n.meta["eva"]["dsr"]["explicit_dsr"]["da_dispatch_fixed_fraction"]
    - n.meta["eva"]["dsr"]["implicit_dsr"]["da_dispatch_fixed_fraction"]

    Parameters
    ----------
    n : pypsa.Network
        Intraday network to be constrained
    n_da : pypsa.Network
        Day-ahead network with solved dispatch
    """
    # Handle explicit DSR (generators)
    explicit_fixed_fraction = n.meta["eva"]["dsr"]["explicit_dsr"][
        "da_dispatch_fixed_fraction"
    ]

    dsr_gens = n.generators.index[n.generators.carrier == "DSR_explicit"]

    if not dsr_gens.empty:
        # Get day-ahead dispatch for DSR generators
        da_dispatch = n_da.generators_t.p[dsr_gens]
        p_nom = n.generators.loc[dsr_gens, "p_nom"]

        # Calculate per-unit DA dispatch
        da_dispatch_pu = da_dispatch.div(p_nom, axis=1).fillna(0.0)

        # Set p_min_pu: lock in DA-responsive portion that was activated
        p_min_pu = da_dispatch_pu * explicit_fixed_fraction
        n.generators_t.p_min_pu[dsr_gens] = p_min_pu.clip(upper=1.0).fillna(0.0)

        # Set p_max_pu: remove DA-responsive portion of unused capacity
        # Get current p_max_pu (default to 1.0 if not set)
        current_p_max_pu = as_dense(n, "Generator", "p_max_pu", dsr_gens)

        new_p_max_pu = (
            current_p_max_pu - (1 - da_dispatch_pu) * explicit_fixed_fraction
        ).clip(lower=0.0)

        n.generators_t.p_max_pu[dsr_gens] = new_p_max_pu

        logger.info(
            f"Fixed {explicit_fixed_fraction:.1%} of day-ahead dispatch for {len(dsr_gens)} explicit DSR generators"
        )
    else:
        logger.debug("No explicit DSR generators found, skipping explicit DSR fixing")

    # Handle implicit DSR (storage units)
    implicit_fixed_fraction = n.meta["eva"]["dsr"]["implicit_dsr"][
        "da_dispatch_fixed_fraction"
    ]

    dsr_storage = n.storage_units.index[n.storage_units.carrier == "DSR_implicit"]

    if not dsr_storage.empty:
        # Get day-ahead dispatch for DSR storage units (positive = discharging, negative = charging)
        # Ensure alignment with intraday network snapshots
        da_dispatch = n_da.storage_units_t.p_dispatch[dsr_storage].reindex(
            index=n.snapshots, fill_value=0.0
        )
        da_store = n_da.storage_units_t.p_store[dsr_storage].reindex(
            index=n.snapshots, fill_value=0.0
        )

        p_nom = n.storage_units.loc[dsr_storage, "p_nom"]

        # Get current time-series limits using get_switchable_as_dense
        # This returns time series with proper defaults: p_min_pu=-1.0, p_max_pu=1.0 for storage
        current_p_min_pu = as_dense(n, "StorageUnit", "p_min_pu", dsr_storage)
        current_p_max_pu = as_dense(n, "StorageUnit", "p_max_pu", dsr_storage)

        # Convert DA dispatch to per-unit (time series)
        da_dispatch_pu = da_dispatch.div(p_nom, axis=1).fillna(0.0)
        da_store_pu = da_store.div(p_nom, axis=1).fillna(0.0)

        # For discharging (p_max_pu): lock in what was dispatched, remove DA-only portion of unused capacity
        # new_p_max_pu = da_dispatch_pu + (p_max_pu - da_dispatch_pu) * (1 - f)
        #              = p_max_pu - (p_max_pu - da_dispatch_pu) * f
        new_p_max_pu = (
            current_p_max_pu
            - (current_p_max_pu - da_dispatch_pu) * implicit_fixed_fraction
        )

        # For charging (p_min_pu): lock in what was stored, remove DA-only portion of unused charging capacity
        # Note: charging is negative, so we adjust upward (less negative)
        # new_p_min_pu = -da_store_pu - (abs(p_min_pu) - da_store_pu) * (1 - f)
        #              = p_min_pu + (abs(p_min_pu) - da_store_pu) * f
        new_p_min_pu = (
            current_p_min_pu
            + (-current_p_min_pu - da_store_pu) * implicit_fixed_fraction
        )

        # Assign time series back to network with proper clipping.
        # These are *ceilings* only: they cap how far the unit may push in each
        # direction. Because PyPSA floors both p_dispatch and p_store at 0, the
        # ceilings alone cannot commit the unit - the solver can always back off
        # to zero. The matching *floors* are imposed in
        # add_implicit_dsr_fixing_constraints (extra_functionality).
        n.storage_units_t.p_min_pu[dsr_storage] = new_p_min_pu.clip(
            lower=-1.0, upper=1.0
        )
        n.storage_units_t.p_max_pu[dsr_storage] = new_p_max_pu.clip(
            lower=-1.0, upper=1.0
        )

        # Stash day-ahead realised dispatch/charging (MW) so the intraday model
        # can floor the storage variables at ``f`` times these values.
        n.storage_units_t["da_dispatch_fix"] = da_dispatch
        n.storage_units_t["da_store_fix"] = da_store

        logger.info(
            f"Fixed {implicit_fixed_fraction:.1%} of day-ahead dispatch for {len(dsr_storage)} implicit DSR storage units"
        )
    else:
        logger.debug(
            "No implicit DSR storage units found, skipping implicit DSR fixing"
        )


def prepare_intraday_network(
    n: pypsa.Network,
    n_da: pypsa.Network,
    tol_ramp: float = 1e-6,
    id_free_commitment_carriers: list[str] | None = None,
) -> pypsa.Network:
    # Store original p_max_pu BEFORE any modifications for consistent residual load calculations
    if "p_max_pu_original" not in n.generators_t:
        n.generators_t["p_max_pu_original"] = n.get_switchable_as_dense(
            "Generator", "p_max_pu"
        ).copy()
        logger.debug("Stored original p_max_pu values for forecast error calculation")

    # Lock p_max_pu/p_min_pu to the DA commitment status BEFORE blocking
    # reserves out of them: reserve blocking subtracts an absolute MW amount
    # (not scaled by status), so doing it after the status lock keeps the ID
    # bound DA-consistent (status*p_max*p_nom - r). Doing it before, as this
    # used to, let the status multiplication scale down the already-blocked
    # reserve too, releasing (1-status)*r MW back for energy dispatch.
    align_ramp_limits(
        n,
        n_da,
        tol_ramp=tol_ramp,
        id_free_commitment_carriers=id_free_commitment_carriers,
    )
    apply_generators_reserve_blocking_id(n, n_da)
    add_storage_reserves_to_be_blocked(n, n_da)
    # Only block ramp capability for reserves when the DA did too: reserve-aware
    # ramp adaptation exists only in co-optimised mode, so in derated mode the
    # DA schedule was made with full ramps and subtracting reserves here would
    # make the ID stricter than the DA plan it is locked to (e.g. a committed
    # start-up whose ramp is consumed by the blocked reserve).
    if n.meta["eva"]["reserves"]["option"] == "co-optimised":
        adjust_ramp_limits_for_reserves(n, n_da)
    set_all_generators_not_committable(
        n=n,
        n_da=n_da,
        set_p_min_pu_zero=False,
        id_free_commitment_carriers=id_free_commitment_carriers,
    )
    fix_dsr_dispatch_from_da(n, n_da)

    return n


def _initialize_storage_state(n, storage_plan, i, sns, fallback_soc):
    """Initialize storage state of charge for rolling horizon window."""
    fallback_soc_units = []
    if i == 0 or fallback_soc is None:
        soc_plan_row = (
            storage_plan.loc[sns[0]]
            if sns[0] in storage_plan.index
            else pd.Series(dtype=float)
        )
        soc_init = pd.Series(index=n.storage_units.index, dtype=float)
        for unit in n.storage_units.index:
            val = soc_plan_row.get(unit, None)
            if pd.notna(val):
                soc_init[unit] = val
            else:
                soc_init[unit] = 0.0
                fallback_soc_units.append(unit)
    else:
        soc_init = fallback_soc

    if not n.storage_units.empty:
        if "state_of_charge_initial" not in n.storage_units.columns:
            # Column doesn't exist, create it with soc_init values
            n.storage_units["state_of_charge_initial"] = soc_init
        else:
            # Fill NaN values in existing column with values from soc_init
            mask = n.storage_units.state_of_charge_initial.isna()
            if mask.any():
                n.storage_units.loc[mask, "state_of_charge_initial"] = soc_init[mask]
                logger.info(
                    f"Initialized state_of_charge_initial from storage plan for {mask.sum()} units"
                )
            else:
                logger.info(
                    "All units already have state_of_charge_initial; no override needed."
                )

    return fallback_soc_units


def optimize_with_rolling_horizon(
    n: pypsa.Network,
    status_log_csv: Path,
    storage_plans_csv: Path,
    config: dict[str, Any],
    snapshots: Sequence | None = None,
    horizon: int = 100,
    overlap: int = 0,
    fail_on_infeasibility: bool = False,
    **kwargs,
) -> Network:
    """
    Optimizes the network in a rolling horizon fashion.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to optimize.
    status_log_csv : Path
        Path to CSV file where rolling horizon status logs will be stored.
    storage_plans_csv : Path
        Path to CSV with planned storage unit state of charge (indexed by time).
    snapshots : list-like, optional
        Set of snapshots to consider in the optimization. The default is None.
    horizon : int
        Number of snapshots to consider in each iteration.
    overlap : int
        Number of snapshots to overlap between two iterations.
    fail_on_infeasibility : bool, optional
        Whether to fail the optimization if infeasibility is detected. The default is False.
    **kwargs:
        Keyword arguments passed to `n.optimize`, such as `solver_name`, etc.
    """
    if snapshots is None:
        snapshots = list(n.snapshots)

    if horizon <= overlap:
        raise ValueError("overlap must be smaller than horizon")

    # Read and reindex the storage plan
    if storage_plans_csv:
        full_storage_plan = pd.read_csv(
            storage_plans_csv, index_col=0, parse_dates=True
        )
        storage_plan = full_storage_plan.reindex(n.snapshots)
    else:
        full_storage_plan = pd.DataFrame(dtype=float)
        storage_plan = pd.DataFrame(index=n.snapshots, columns=n.storage_units.index)

    starting_points = range(0, len(snapshots), horizon - overlap)
    status_log = []
    fallback_soc = None  # Will be updated after each successful solve

    solver_export_prefix = str(
        kwargs.pop("solver_export_prefix", Path(status_log_csv).stem)
    )

    # NOTE: Do NOT create a persistent gurobipy.Env() here.  Linopy's Gurobi
    # solver backend (solvers.py) creates its own Env inside each
    # model.solve() call via a context-manager.  Creating a second Env here
    # conflicts with single-use WLS licences because two Envs cannot coexist.
    # If solver_options["env"] is passed by the caller, strip it out so
    # linopy doesn't receive a stale/conflicting env object.
    solver_opts = kwargs.get("solver_options", {})
    if "env" in solver_opts:
        solver_opts.pop("env")
        logger.debug(
            "Removed caller-supplied Gurobi env to avoid single-use licence conflict."
        )

    for i, start in enumerate(starting_points):
        end = min(len(snapshots), start + horizon)
        sns = snapshots[start:end]
        logger.info(
            "Optimizing network for snapshot horizon [%s:%s] (%d/%d).",
            sns[0],
            sns[-1],
            i + 1,
            len(starting_points),
        )

        # Initialize storage state following PyPSA's approach
        # For i==0: initialize from the storage plan timestamp just before the window starts.
        # For i>0: use previous snapshot's SOC, with fallback to storage plan if previous failed
        if (
            not i
            and n.meta.get("cluster_id") is not None
            and not n.storage_units.empty
            and not full_storage_plan.empty
        ):
            first_ts = sns[0]
            plan_index = full_storage_plan.index
            pos = plan_index.searchsorted(first_ts)
            if pos > 0:
                ref_ts = plan_index[pos - 1]
                plan_row = full_storage_plan.iloc[pos - 1].reindex(
                    n.storage_units.index
                )
            else:
                ref_ts = plan_index[0]
                plan_row = full_storage_plan.iloc[0].reindex(n.storage_units.index)
            # Only override units present in the storage plan (hydro reservoirs).
            # Units not in the plan (batteries, DSR, etc.) keep their existing
            # state_of_charge_initial set by build_base_network.
            planned_units = plan_row.index[plan_row.notna()]
            if planned_units.empty:
                logger.warning(
                    f"Window 1 (i=0): no storage plan values found at {ref_ts}; "
                    "SOC initialization unchanged."
                )
            else:
                n.storage_units.loc[planned_units, "state_of_charge_initial"] = (
                    plan_row[planned_units]
                )
                logger.info(
                    f"Window 1 (i=0): initialized SOC for {len(planned_units)} units "
                    f"from storage plan at {ref_ts} (last timestamp before centroid start {first_ts})."
                )

        if i:
            if not n.stores.empty:
                n.stores.e_initial = n.stores_t.e.loc[snapshots[start - 1]]

            if not n.storage_units.empty:
                # If previous window solved successfully, use its SOC results
                if fallback_soc is not None:
                    n.storage_units.state_of_charge_initial = (
                        n.storage_units_t.state_of_charge.loc[snapshots[start - 1]]
                    )
                # If previous solve failed, fall back to storage plan
                elif storage_plans_csv and sns[0] in storage_plan.index:
                    logger.warning(
                        f"Window {i + 1}: Previous optimization failed, using storage plan for initialization"
                    )
                    n.storage_units.state_of_charge_initial = storage_plan.loc[
                        sns[0]
                    ].fillna(0.0)
                else:
                    logger.warning(
                        f"Window {i + 1}: Previous optimization failed and no storage plan available, using zero initial SOC"
                    )
                    n.storage_units.state_of_charge_initial = pd.Series(
                        0.0, index=n.storage_units.index
                    )

        # Solve
        if kwargs.get("solver_options") == {}:
            kwargs.pop("solver_options")

        solver_name = kwargs.get("solver_name", "")
        local_kwargs = dict(kwargs)

        if solver_name == "gurobi":
            export_gurobi_logs = config["eva"]["export_gurobi_logs"]
            export_gurobi_mps = config["eva"]["export_gurobi_mps"]

            # Gurobi LogFile and PyPSA/Linopy log_fn should not be used together.
            if export_gurobi_logs:
                local_kwargs.pop("log_fn", None)

            stem = (
                f"{solver_export_prefix}_window_{i:03d}_"
                f"{sanitize_filename(sns[0])}_{sanitize_filename(sns[-1])}"
            )
            local_kwargs["solver_options"] = with_gurobi_exports(
                solver_name=solver_name,
                solver_options=local_kwargs.get("solver_options", {}),
                export_dir="resources/solver_exports",
                stem=stem,
                export_log=export_gurobi_logs,
                export_model=export_gurobi_mps,
            )

        # Determine if we should use a global lock to prevent Gurobi WLS collisions.
        if "EUFLEX_GUROBI_LOCKING" not in os.environ:
            logger.warning(
                "EUFLEX_GUROBI_LOCKING not set, defaulting to 'false' "
                "(no Gurobi locking). Set this variable explicitly in your "
                "config to suppress this warning."
            )

        gurobi_locking = os.environ.get("EUFLEX_GUROBI_LOCKING", "false")

        if solver_name == "gurobi" and gurobi_locking.strip().casefold() == "true":
            if "EUFLEX_GUROBI_LOCK_PATH" not in os.environ:
                logger.warning(
                    "EUFLEX_GUROBI_LOCK_PATH not set, defaulting to "
                    "'/var/lock/euflex/gurobi_wls.lock'. Set this variable "
                    "explicitly in your config to suppress this warning."
                )
            lock_path = os.environ.get(
                "EUFLEX_GUROBI_LOCK_PATH", "/var/lock/euflex/gurobi_wls.lock"
            )
            lock_manager = filelock.FileLock(lock_path, timeout=3600)
        else:
            lock_manager = contextlib.nullcontext()

        with lock_manager:
            status, condition = run_optimization(
                optimize_callable=n.optimize,
                optimize_kwargs=dict(
                    snapshots=sns,
                    assign_all_duals=True,
                    **local_kwargs,
                ),
                network=n,
                module=f"rolling_horizon_window_{i}",
            )

        # Persist objective value while the model is still available.
        if hasattr(n, "model"):
            try:
                n.meta["objective_value"] = float(n.model.objective.value)
            except Exception:
                pass

        # Release the linopy model's Gurobi env so the next window can acquire
        # the single-use WLS licence.  Linopy stores the raw gurobipy.Model as
        # n.model.solver_model; that object holds a reference to the gurobipy.Env.
        # We must explicitly dispose the gurobipy model and then delete the
        # linopy model, otherwise gurobipy defers the env free ("environment
        # still referenced") and the next Env() call fails.
        # Only do this for Gurobi -- other solvers (HiGHS, etc.) do not need it
        # and callers may still need to inspect n.model after the solve.
        # Skip the dispose when we are about to diagnose an infeasibility: the
        # IIS computation and .lp export below need the live gurobipy model,
        # and that path raises anyway so the licence is not reused.
        about_to_diagnose = status != "ok" and fail_on_infeasibility
        if solver_name == "gurobi" and hasattr(n, "model") and not about_to_diagnose:
            solver_model = getattr(n.model, "solver_model", None)
            if solver_model is not None:
                try:
                    solver_model.dispose()
                except Exception:
                    pass
                n.model.solver_model = None
            del n.model
        gc.collect()

        # Track optimization success for storage plan fallback in next window
        if status == "ok":
            logger.info(
                f"Rolling horizon window {i + 1}/{len(starting_points)}: PyPSA automatically stores duals with assign_all_duals=True"
            )
            # PyPSA automatically stores dual variables in buses_t, no manual collection needed
            # Mark this window as successful so next window uses its results
            fallback_soc = True  # Any non-None value indicates success
        else:
            logger.warning(
                "Optimization failed with status %s and condition %s. "
                "Next window will use storage plan or zero initial SOC.",
                status,
                condition,
            )
            # Mark as failed so next window uses storage plan fallback
            fallback_soc = None
            if fail_on_infeasibility:
                solver_name = kwargs.get("solver_name")
                maybe_print_solver_infeasibilities(
                    getattr(n, "model", None), solver_name, logger
                )
                filename = Path(n.meta["infeasibility_lp_file"] + f"_{i}.lp")
                logger.info("Writing .lp file for the failing model...")
                write_model_to_lp_file(n, filename.as_posix(), True, False)
                logger.info(f".lp file ready in {filename}")
                logger.error(
                    "INFEASIBILITY DETECTED during rolling horizon window "
                    "[%s:%s] (status=%s, condition=%s)",
                    sns[0],
                    sns[-1],
                    status,
                    condition,
                )
                raise InfeasibilityError(
                    message=(
                        "Rolling horizon optimisation failed: "
                        f"status={status}, condition={condition}."
                    ),
                    source="market_functions.optimize_with_rolling_horizon",
                    details={
                        "status": status,
                        "condition": str(condition),
                        "horizon_index": i,
                        "snapshot_start": str(sns[0]),
                        "snapshot_end": str(sns[-1]),
                        "solver_name": kwargs.get("solver_name"),
                    },
                )

        # Append status log
        status_log.append(
            {
                "horizon_index": i,
                "snapshot_start": str(sns[0]),
                "snapshot_end": str(sns[-1]),
                "status": status,
                "condition": str(condition),
                # "fallback_soc_units": ";".join(fallback_soc_units),
            }
        )

    # Write status log
    Path(status_log_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(status_log).to_csv(status_log_csv, index=False)

    return n


def run_market_model(
    n: pypsa.Network,
    storage_plans_csv: Path,
    status_log_csv: Path,
    log: Path,
    market_interconnection: str,
    climate_year: int,
    target_year: int,
    config: dict[str, Any],
    is_day_ahead: bool = False,
    **optimize_kwargs,
) -> pypsa.Network | tuple[pypsa.Network, pypsa.Network]:
    """
    Runs a market model optimization on a PyPSA network using a rolling horizon
    approach with storage plans and configuration settings.

    Parameters
    ----------
    n : pypsa.Network
        The input PyPSA network object to be optimized. This network serves as
        the base model, and its components are modified during the optimization
        process. It turns the extendable attribute of generators, storage units,
        stores, links, and lines off for the market optimization then
        back to their original values after the optimization
    storage_plans_csv : Path
        File path to the CSV containing the storage plans. The CSV file should
        use the snapshots of the network as its index. Parsed storage plans from
        the file are added to the network for the optimization process. When
        ``eva.storage_planning.use_cyclic_soc`` is enabled, the plan value at
        t=0 also seeds ``state_of_charge_initial`` for the storage units covered
        by the pre-optimisation (see
        :func:`initialize_soc_from_storage_plan`).
    status_log_csv : Path
        File to include the status of each optimization model
    log : Path
        Path to the log file where the optimization process outputs are written.
        Primarily used for debugging and record-keeping.
    market_interconnection : str
        The type of market interconnection model to be used. Supported values
        include "fbmc" for flow-based market coupling (with integrated NTCs)
        and "physical" for physical interconnection.
    climate_year : int
        The climate year for which the flow-based data is relevant.
    target_year : int
        The target year for which the flow-based data is relevant.
    config : dict[str, Any]
        A dictionary specifying configuration parameters for the rolling horizon
        optimization. Supported keys include:
            - 'horizon' : int
                The time horizon in hours for each iteration of the rolling
                horizon.
            - 'overlap' : int
                The number of overlapping hours between consecutive iterations of
                the optimization.
            - 'linearized_unit_commitment' : bool
                Determines whether linearized unit commitment constraints are used
                during optimization.
        It must also carry ``eva.storage_planning`` with the keys
        ``use_cyclic_soc`` and ``preopt_targeted_carriers``.
    is_day_ahead : bool, optional
        A flag indicating whether the optimization is for a day-ahead market.

    Returns
    -------
    pypsa.Network | tuple[pypsa.Network, pypsa.Network]
        When linearized_unit_commitment is True or is_day_ahead is False:
            Returns a single optimized PyPSA network object.
        When is_day_ahead is True and linearized_unit_commitment is False:
            Returns a tuple of two networks:
            - First network: The final optimized network with market prices
            - Second network: The integer unit commitment network from the first step
    """

    # Read storage plans and add to network
    storage_plan = pd.read_csv(storage_plans_csv, index_col=0, parse_dates=True)
    n.storage_units_t["storage_plan"] = storage_plan.reindex(n.snapshots)

    # When the pre-optimisation enforced a cyclic state of charge, start the
    # market model from the planned SOC at t=0 so the first rolling horizon is
    # consistent with the plan the later horizons are steered towards.
    storage_planning_cfg = config["eva"]["storage_planning"]
    if storage_planning_cfg["use_cyclic_soc"]:
        initialize_soc_from_storage_plan(
            n,
            storage_plan,
            storage_planning_cfg["preopt_targeted_carriers"],
        )

    # Turn off extendable for all components in the network
    n, original_extendable = turn_off_extendable(n=n)

    # Apply derated reserves if requested
    reserves_option = config["eva"]["reserves"]["option"]
    original_pu_bounds: dict | None = None
    if reserves_option == "derated" and is_day_ahead:
        original_pu_bounds = apply_derated_reserves(n)
        logger.info("Derated mode: p_max_pu/p_min_pu adjusted by reserve amounts.")

    # Get eraa year (default 2023 if not specified)
    eraa_year = config["eva"]["eraa_year"]

    # Register market interconnection
    register_market_interconnection(
        n,
        market_interconnection,
        int(climate_year),
        int(target_year),
        config,
        int(eraa_year),
    )

    # Get config parameters
    config_options = config["solving"]["options"]
    horizon = config_options["horizon"]
    overlap = config_options["overlap"]
    linearized_unit_commitment = config_options["linearized_unit_commitment"]

    fail_on_infeasibility = config["eva"]["fail_on_infeasibility"]
    n.meta["is_day_ahead"] = is_day_ahead

    # Make a copy of the original network to solve the second step of the integer resolution if needed
    if is_day_ahead and not linearized_unit_commitment:
        n_step_two = n.copy()
    else:
        n_step_two = None

    # For intraday, always use linearized UC since commitment decisions are fixed from day-ahead.
    # Integer UC (linearized_unit_commitment=False) is only meaningful for day-ahead.
    # This keeps intraday as a pure LP consistent with the "Fixed Commitment LP" methodology.
    effective_linearized_uc = linearized_unit_commitment if is_day_ahead else True

    # Allow callers to override model_kwargs (e.g. meshed_threshold in tests).
    # Default keeps meshed_threshold high to avoid the force_dim_names error on
    # networks where many generators share a single bus.
    model_kwargs = optimize_kwargs.pop("model_kwargs", {"meshed_threshold": 10000})

    optimize_with_rolling_horizon(
        n,
        status_log_csv=status_log_csv,
        storage_plans_csv=storage_plans_csv,
        config=config,
        horizon=horizon,
        overlap=overlap,
        fail_on_infeasibility=fail_on_infeasibility,
        linearized_unit_commitment=effective_linearized_uc,
        log_fn=log,
        extra_functionality=extra_functionality,
        solver_name=config["solving"]["solver"]["name"],
        solver_options=config["solving"]["solver_options"][
            config["solving"]["solver"]["options"]
        ],
        model_kwargs=model_kwargs,
        io_api="lp-polars",
        **optimize_kwargs,
    )

    tol_status = config["eva"]["patching_tol_status"]
    n = fix_small_status_variables(n, tol_status=tol_status)

    if is_day_ahead and not linearized_unit_commitment:
        # Two-step integer UC process:
        # Step 1 (already done above): Solve with binary UC to get commitment decisions
        # Step 2 (below): Re-solve as LP with fixed commitments to get market prices

        # Type guard: n_step_two is guaranteed to be not None here
        # because it was created in the same condition above (line 2595-2596)
        assert n_step_two is not None, (
            "n_step_two should not be None when is_day_ahead and not linearized_unit_commitment"
        )

        logger.info(
            "Re-solving day-ahead model with fixed commitment status variables to calculate price "
            "(second step when applying integer)"
        )

        # Store integer status results in network for constraint fixing
        # The add_fixed_commitment_constraints function will use this to fix status vars
        if not n.generators_t.status.empty:
            committable_gens = n.generators.index[n.generators.committable]
            if not committable_gens.empty:
                n_step_two._var_status = (
                    n.generators_t.status[  # ty: ignore[unresolved-attribute]
                        committable_gens
                    ]
                )
                logger.info(
                    f"Stored integer status for {len(committable_gens)} committable generators"
                )

        # Align ramp limits based on integer solution
        tol_ramp = config["eva"]["patching_tol_ramp"]
        align_ramp_limits(
            n_step_two,
            n,
            tol_ramp=tol_ramp,
            id_free_commitment_carriers=config["eva"]["base_network"][
                "id_free_commitment_carriers"
            ],
        )

        # Second solve: LP with linearized UC and fixed commitments to get market prices
        # The add_fixed_commitment_constraints in extra_functionality will fix status
        # variables to the integer solution values stored in n._var_status
        optimize_with_rolling_horizon(
            n_step_two,
            status_log_csv=status_log_csv,
            storage_plans_csv=storage_plans_csv,
            config=config,
            horizon=horizon,
            overlap=overlap,
            fail_on_infeasibility=fail_on_infeasibility,
            linearized_unit_commitment=True,  # Use LP with continuous status for duals
            log_fn=log,
            extra_functionality=extra_functionality,
            solver_name=config["solving"]["solver"]["name"],
            solver_options=config["solving"]["solver_options"][
                config["solving"]["solver"]["options"]
            ],
            model_kwargs=model_kwargs,
            io_api="lp-polars",
            solver_export_prefix=f"{Path(status_log_csv).stem}_price_step",
        )

        # Copy status values from integer network to final network for consistency
        if not n.generators_t.status.empty:
            n_step_two.generators_t["status"] = n.generators_t.status.copy()

        # Clean up temporary attribute
        if hasattr(n_step_two, "_var_status"):
            del n_step_two._var_status

    # Restore p_max_pu/p_min_pu and add fixed reserve timeseries in derated mode
    if reserves_option == "derated" and is_day_ahead:
        for component_name, c in [
            ("generators", "Generator"),
            ("storage_units", "StorageUnit"),
        ]:
            bounds = original_pu_bounds.get(component_name, {})
            dynamic = n.dynamic(c)
            for bound in ("p_max_pu", "p_min_pu"):
                if bound in bounds:
                    dynamic[bound] = bounds[bound]
        add_fixed_reserve_timeseries(n)
        logger.info(
            "Derated mode: p_max_pu/p_min_pu restored; fixed reserve timeseries added."
        )

    # Restore original extendability settings
    n.generators.p_nom_extendable = original_extendable["generators"]
    n.storage_units.p_nom_extendable = original_extendable["storage_units"]
    n.stores.e_nom_extendable = original_extendable["stores"]
    n.links.p_nom_extendable = original_extendable["links"]
    n.lines.s_nom_extendable = original_extendable["lines"]

    if n_step_two is not None:
        # Restore original extendability settings for integer network too
        n_step_two.generators.p_nom_extendable = original_extendable["generators"]
        n_step_two.storage_units.p_nom_extendable = original_extendable["storage_units"]
        n_step_two.stores.e_nom_extendable = original_extendable["stores"]
        n_step_two.links.p_nom_extendable = original_extendable["links"]
        n_step_two.lines.s_nom_extendable = original_extendable["lines"]

        # In this case, n_step_two is the one with the final solution (including prices)
        # n contains the integer result from first step
        return n_step_two, n

    return n


def fix_small_status_variables(
    n: pypsa.Network, tol_status: float = 1e-6
) -> pypsa.Network:
    """
    Round numerically insignificant LP status residuals to 0 or 1.

    Values with ``abs(status) <= tol_status`` are set to 0.
    Values with ``abs(status - 1) <= tol_status`` are set to 1.
    This prevents solver-tolerance artefacts from propagating into the ID model
    as hard dispatch bounds.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object to process.
    tol_status : float
        Absolute tolerance applied to both near-zero and near-one rounding.

    Returns
    -------
    pypsa.Network
        The network with cleaned status variables.
    """
    if not hasattr(n, "generators_t") or "status" not in n.generators_t:
        logger.debug("No status variables found in network generators_t")
        return n

    status_df = n.generators_t.status

    if status_df.empty:
        logger.debug("Status dataframe is empty")
        return n

    near_zero_mask = status_df.abs() <= tol_status
    near_one_mask = (status_df - 1.0).abs() <= tol_status

    n_zero = int(near_zero_mask.sum().sum())
    n_one = int(near_one_mask.sum().sum())

    if n_zero > 0:
        logger.info(
            f"fix_small_status_variables: rounding {n_zero} near-zero status value(s) to 0 (tol={tol_status:.0e})"
        )
        status_df = status_df.where(~near_zero_mask, 0.0)
    if n_one > 0:
        logger.info(
            f"fix_small_status_variables: rounding {n_one} near-one status value(s) to 1 (tol={tol_status:.0e})"
        )
        status_df = status_df.where(~near_one_mask, 1.0)

    n.generators_t.status = status_df
    return n


def write_model_to_lp_file(
    n: pypsa.Network,
    output_filename: str,
    generate_lp: bool = False,
    explicit_coordinate_names: bool = True,
):
    # Model export is best-effort only. The Linopy model may already be unavailable
    # after solving or cleanup, so this function must not stop the workflow.
    model = getattr(n, "model", None)

    if model is None:
        logger.warning(
            "Skipping model export to %s because the network has no attached model.",
            output_filename,
        )

        with open(output_filename, "w") as f:
            f.write("! Linopy model unavailable after solve\n")
            f.write("! No LP representation could be exported.\n")

        return

    if generate_lp:
        try:
            model.to_file(
                output_filename, explicit_coordinate_names=explicit_coordinate_names
            )
        except Exception as e:
            logger.warning("Could not write LP file %s: %s", output_filename, e)

            with open(output_filename, "w") as f:
                f.write(f"! Error writing LP file: {e}\n")
                f.write("! No LP representation could be exported.\n")

            return
    else:
        try:
            with open(output_filename, "w") as f:
                f.write(str(model))
        except Exception as e:
            logger.warning("Could not write model string to %s: %s", output_filename, e)

            with open(output_filename, "w") as f:
                f.write(f"! Error writing model string: {e}\n")
                f.write("! No model string representation could be exported.\n")

            return


def get_reserves_parameters(
    network: pypsa.Network,
) -> dict[str, dict[str, float | str]]:
    """
    Get reserves parameters from network buses.

    Extracts reserve parameters from bus columns and returns them in the same format
    as they were stored in n.meta["reserves_parameters"].

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.

    Returns
    -------
    dict[str, dict[str, float | str]]
        Dictionary mapping reserve types to their parameters.
        Example: {"FCR": {"full_activation_time": 0.5, "min_duration": 1.0,
                         "id_blocked": 0.5, "reserve_blocking_method": "fraction"}}
    """
    # Find all reserve parameter columns in buses (exclude requirement columns)
    # Parameter columns end with _full_activation_time, _min_duration, _id_blocked, or _reserve_blocking_method
    reserve_cols = [
        col
        for col in network.buses.columns
        if col.startswith("reserve_")
        and any(col.endswith(suffix) for suffix in RESERVE_PARAMETER_SUFFIXES)
    ]

    # Extract unique reserve types from parameter columns only.
    #
    # Sorted, and deliberately not in bus-column order: this dictionary decides
    # the order in which reserve variables and constraints are added to the
    # optimisation model, and PyPSA does not preserve DataFrame column order
    # when it reads a network back from netCDF -- two processes reading the
    # same file can see n.buses.columns in different orders. Without a total
    # order here the LP differs between runs of an otherwise identical
    # configuration, and the solver lands on a different one of the many
    # solutions that share the optimal cost.
    reserve_types: list[str] = sorted(
        {reserve_type_from_column(col) for col in reserve_cols}
    )

    if not reserve_types:
        return {}

    # Build reserves parameters dict
    reserves_params = {}
    for reserve_type in reserve_types:
        col_activation = f"reserve_{reserve_type}_full_activation_time"
        col_duration = f"reserve_{reserve_type}_min_duration"
        col_blocked = f"reserve_{reserve_type}_id_blocked"
        col_penalty_cost = f"reserve_{reserve_type}_penalty_cost"
        col_blocking_method = f"reserve_{reserve_type}_reserve_blocking_method"

        # Get value from first bus (they should all be the same)
        if col_activation in network.buses.columns:
            # TODO: the if in the penalty_cost is a temporary fix for missing columns
            #  because we cannot update pre-builts.
            reserves_params[reserve_type] = {
                "full_activation_time": network.buses[col_activation].iloc[0],
                "min_duration": network.buses[col_duration].iloc[0],
                "id_blocked": network.buses[col_blocked].iloc[0],
                "penalty_cost": network.buses[col_penalty_cost].iloc[0]
                if col_penalty_cost in network.buses.columns
                else network.meta["eva"]["reserves"][reserve_type.lower()][
                    "penalty_cost"
                ],
                "reserve_blocking_method": network.buses[col_blocking_method].iloc[0]
                if col_blocking_method in network.buses.columns
                else "fraction",  # Default to "fraction" for backward compatibility
            }

    return reserves_params


def sanitize_dual_names(network: pypsa.Network) -> pypsa.Network:
    """
    Sanitize and convert PyPSA automatic dual variable names to expected format.

    This function converts PyPSA's automatically stored dual variables (with 'mu_' prefix)
    from reserve constraints to the expected naming format for downstream calculations
    like NPV computations.

    Parameters
    ----------
    network : pypsa.Network
        Network with optimization model containing dual variables

    Returns
    -------
    pypsa.Network
        Network with reserve dual variables sanitized to expected format in buses_t
    """
    # PyPSA automatically stores dual variables in buses_t with 'mu_' prefix when assign_all_duals=True
    # Look for automatically stored reserve dual variables in buses_t
    buses_t_keys = list(network.buses_t.keys())
    reserve_mu_keys = [
        key for key in buses_t_keys if key.startswith("mu_Reserve_requirements_")
    ]

    if not reserve_mu_keys:
        return network

    # Convert PyPSA's automatically stored duals to our expected format
    for mu_key in reserve_mu_keys:
        # Extract reserve type and direction from mu_Bus-Reserve_requirements_FCR_up format
        if mu_key.startswith("mu_Reserve_requirements_"):
            suffix = mu_key[
                len("mu_Reserve_requirements_") :
            ]  # Get FCR_up, FCR_down, etc.
            ts_name = f"r_{suffix}_price"  # Convert to r_FCR_up_price format

            # Get the dual variable data and ensure proper data types
            prices_df = network.buses_t[mu_key].astype(float)

            # Store in the expected format for backwards compatibility
            network.buses_t[ts_name] = prices_df

    logger.info(
        f"Converted {len(reserve_mu_keys)} reserve dual variables from PyPSA automatic storage"
    )

    return network
