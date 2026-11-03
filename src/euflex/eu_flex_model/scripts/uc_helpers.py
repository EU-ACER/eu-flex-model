"""
Unit Commitment Constraint Fixes for PyPSA Rolling Horizon with Linearized UC.

This module contains workarounds for PyPSA limitations when using linearized unit
commitment (fractional status variables) with rolling horizon optimization.

Background
----------
PyPSA's unit commitment constraints at rolling horizon boundaries assume binary
status values (0 or 1). When using linearized UC, status variables are fractional
(e.g., 0.57, 0.82), causing several issues:

1. PyPSA hardcodes boundary constraint RHS values to -1/0/+1
2. The up_time_before/down_time_before calculation breaks with fractional values
3. This leads to incorrect constraints and artificial inflation of startup/shutdown
   variables, causing significant cost errors (€100k+ per generator)

Affected Constraints
-------------------
At rolling horizon boundaries, the following PyPSA constraints are incorrect for
linearized UC:

1. Generator-com-transition-start-up (startup transition)
2. Generator-com-transition-shut-down (shutdown transition)
3. Generator-com-min-up-time (minimum up-time enforcement)
4. Generator-com-min-down-time (minimum down-time enforcement)
5. Convex hull tightening constraints (when start_up_cost == shut_down_cost)

Solution
--------
This module provides post-constraint-creation fixes that update the constraint RHS
values at the first snapshot of each rolling horizon window to use actual fractional
status values from the previous window instead of binary assumptions.

When to Remove
--------------
This workaround can be removed when:
- PyPSA natively supports linearized UC + rolling horizon (check versions > 1.3.0)
- OR you switch to integer UC (linearized_unit_commitment = False)
- OR you stop using rolling horizon optimization

Related PyPSA Code
------------------
See PyPSA's `pypsa/optimization/constraints.py` function
`define_operational_constraints_for_committables` (circa line 173-350 in v0.33.2)
for the constraint creation logic that this module fixes.

References:
----------
Issue discovered: 2026-08-31
PyPSA version at discovery: 0.33.2 (issue still present in 1.3.0)

Note:
----
Ramp constraints involving status.shift() at boundaries are already properly implemented,
and a regression test "test/test_ut_linearized_uc.py::TestLinearizedUC::
test_rolling_horizon_ramp_constraints_regression" is available

Convex hull tightening constraints are not fixed here - they exist to improve the relaxation quality,
but they don't enforce fundamental operational rules like min up/downtime and stratup/shutdown
transition constraints.
"""

import logging

import pandas as pd
import pypsa
import xarray as xr

logger = logging.getLogger(__name__)


def fix_uc_min_uptime_downtime_constraints_at_rolling_horizon_boundary(
    n: pypsa.Network, sns: pd.Index
) -> None:
    """
    Fix min up-time and min down-time constraints at rolling horizon boundary.

    PyPSA's constraints enforce:
    - Generator-com-up-time: status[t] >= rolling_sum(start_up over last min_up_time)
    - Generator-com-down-time: status[t] + rolling_sum(shut_down over last min_down_time) <= 1

    At rolling horizon boundaries, the rolling sums can only see start_up/shut_down
    from the current window, missing those from the previous window.

    For snapshots t=0 to t=min_time-1:
    - At t=0: PyPSA skips this (uses sns[1:]), need to ADD constraint
    - At t=1 to min_time-1: Need to MODIFY RHS to include previous window values

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots being optimized in this window.

    Returns
    -------
    None
    """
    # Only apply at rolling horizon boundaries (not the first window)
    if sns[0] == n.snapshots[0]:
        return

    # Get committable generators
    committable_gens = n.generators.index[n.generators.committable]
    if committable_gens.empty:
        return

    # Get previous window index
    t_minus_1_idx = n.snapshots.get_loc(sns[0]) - 1

    # Check if start_up and shut_down are available
    if "start_up" not in n.generators_t or "shut_down" not in n.generators_t:
        logger.warning(
            "fix_uc_min_uptime_downtime: start_up/shut_down not found, skipping"
        )
        return

    # Process both up_time and down_time constraints
    constraints_to_fix = [
        {
            "name": "Generator-com-up-time",
            "variable_name": "start_up",
            "min_time_attr": "min_up_time",
            "lhs_sign": 1,  # Coefficient for variable in LHS
        },
        {
            "name": "Generator-com-down-time",
            "variable_name": "shut_down",
            "min_time_attr": "min_down_time",
            "lhs_sign": 1,  # Coefficient for variable in LHS
        },
    ]

    for constraint_info in constraints_to_fix:
        constraint_name = constraint_info["name"]
        variable_name = constraint_info["variable_name"]
        min_time_attr = constraint_info["min_time_attr"]
        lhs_sign = constraint_info["lhs_sign"]

        # Filter generators with this min_time > 0
        min_time_series = n.generators.loc[committable_gens, min_time_attr]
        gens_with_min_time = min_time_series[min_time_series > 0].index
        if gens_with_min_time.empty:
            continue

        # Get the constraint
        if constraint_name not in n.model.constraints:
            logger.debug(f"{constraint_name} constraint not found in model")
            continue

        constraint = n.model.constraints[constraint_name]

        # Filter generators that exist in constraint coordinates
        gens_in_constraint = [
            gen
            for gen in gens_with_min_time
            if gen in constraint.data.rhs.coords.get("Generator-com", [])
        ]

        if not gens_in_constraint:
            continue

        # Build snapshot x generator DataFrame of missing_sum adjustments
        # Initialize with zeros for all combinations
        max_snapshots_to_fix = max(
            int(n.generators.loc[gen, min_time_attr]) for gen in gens_in_constraint
        )
        num_snaps_to_fix = min(max_snapshots_to_fix, len(sns))

        # Create DataFrame for RHS adjustments (snapshots x generators)
        rhs_adjustments = pd.DataFrame(
            0.0, index=sns[:num_snaps_to_fix], columns=gens_in_constraint
        )

        # Track t=0 constraint data for vectorized addition
        t0_gens = []
        t0_rhs_values = []

        # Calculate adjustments for each generator
        for gen in gens_in_constraint:
            min_time = int(n.generators.loc[gen, min_time_attr])

            # Get values from previous window
            prev_window_end = t_minus_1_idx + 1
            prev_window_start = max(0, prev_window_end - min_time)
            prev_values = n.generators_t[variable_name].loc[
                n.snapshots[prev_window_start:prev_window_end], gen
            ]

            # Calculate adjustments for snapshots where rolling sum is incomplete
            for t_idx in range(min(min_time, len(sns))):
                snap = sns[t_idx]

                # How many snapshots does rolling sum see from current window?
                snapshots_visible = t_idx + 1

                # How many missing from previous window?
                snapshots_missing = min_time - snapshots_visible

                if snapshots_missing <= 0:
                    continue  # Full window visible, no fix needed

                # Sum of missing values from previous window
                missing_sum = prev_values.iloc[-snapshots_missing:].sum()

                if t_idx == 0:
                    # Store data for vectorized t=0 constraint addition
                    if constraint_name == "Generator-com-up-time":
                        rhs_value = -prev_values.iloc[-(min_time - 1) :].sum()
                    else:  # down_time
                        rhs_value = 1 - prev_values.iloc[-(min_time - 1) :].sum()

                    t0_gens.append(gen)
                    t0_rhs_values.append(rhs_value)
                else:
                    # Store adjustment for vectorized RHS update
                    rhs_adjustments.loc[snap, gen] = -missing_sum

        # Vectorized RHS update for t > 0
        # Only update rows/cols that have non-zero adjustments
        non_zero_mask = rhs_adjustments != 0
        updated_count = 0

        if non_zero_mask.any().any():
            snaps_to_update = rhs_adjustments.index[non_zero_mask.any(axis=1)]
            gens_to_update = rhs_adjustments.columns[non_zero_mask.any(axis=0)].tolist()

            if snaps_to_update.size > 0 and len(gens_to_update) > 0:
                # Get current RHS values
                current_rhs = constraint.data.rhs.loc[
                    {"snapshot": snaps_to_update, "Generator-com": gens_to_update}
                ]

                # Apply adjustments - convert to xarray DataArray for proper indexing
                adjustments_da = xr.DataArray(
                    rhs_adjustments.loc[snaps_to_update, gens_to_update].values,
                    coords={
                        "snapshot": snaps_to_update,
                        "Generator-com": gens_to_update,
                    },
                    dims=["snapshot", "Generator-com"],
                )

                # Update RHS in one operation
                constraint.data.rhs.loc[
                    {"snapshot": snaps_to_update, "Generator-com": gens_to_update}
                ] = current_rhs + adjustments_da

                updated_count = non_zero_mask.sum().sum()

        # Vectorized t=0 constraint addition (all generators at once)
        if t0_gens:
            snap_0 = sns[0]
            status_var = n.model["Generator-status"]
            variable_var = n.model[f"Generator-{variable_name}"]

            # Create RHS as DataArray indexed by Generator-com
            rhs_da = xr.DataArray(
                t0_rhs_values, coords={"Generator-com": t0_gens}, dims=["Generator-com"]
            )

            # Build LHS with Generator-com dimension
            if constraint_name == "Generator-com-up-time":
                lhs_tuple = (
                    (
                        -1,
                        status_var.loc[{"snapshot": snap_0, "Generator-com": t0_gens}],
                    ),
                    (
                        lhs_sign,
                        variable_var.loc[
                            {"snapshot": snap_0, "Generator-com": t0_gens}
                        ],
                    ),
                )
            else:  # down_time
                lhs_tuple = (
                    (1, status_var.loc[{"snapshot": snap_0, "Generator-com": t0_gens}]),
                    (
                        lhs_sign,
                        variable_var.loc[
                            {"snapshot": snap_0, "Generator-com": t0_gens}
                        ],
                    ),
                )

            # Add all t=0 constraints in one call
            n.model.add_constraints(
                lhs_tuple,
                "<=",
                rhs_da,
                name=f"{constraint_name}-t0-fix",
            )
            updated_count += len(t0_gens)

        if updated_count > 0:
            logger.info(
                f"Fixed {constraint_name} at boundary {sns[0]}: "
                f"updated {updated_count} constraints for {len(gens_with_min_time)} generators"
            )


def fix_uc_transition_constraints_at_rolling_horizon_boundary(
    n: pypsa.Network, sns: pd.Index
) -> None:
    """
    Fix startup and shutdown transition constraints at rolling horizon boundary.

    PyPSA's transition constraints at the first snapshot of a rolling horizon
    window use hardcoded RHS values assuming binary status (0 or 1).

    For linearized UC with fractional status values, the correct RHS should use
    the actual fractional status from the previous window:

    1. Generator-com-transition-start-up: su_t - u_t >= RHS
       RHS should be -status[t-1]

    2. Generator-com-transition-shut-down: sd_t + u_t >= RHS
       RHS should be +status[t-1]

    This function updates both constraints at the first snapshot only.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the optimization model.
    sns : pd.Index
        The snapshots being optimized in this window.

    Returns
    -------
    None
    """
    # Only apply at rolling horizon boundaries (not the first window)
    if sns[0] == n.snapshots[0]:
        return

    # Get committable generators
    committable_gens = n.generators.index[n.generators.committable]
    if committable_gens.empty:
        return

    # Get previous snapshot (t-1)
    t_minus_1_idx = n.snapshots.get_loc(sns[0]) - 1
    t_minus_1 = n.snapshots[t_minus_1_idx]

    status_t_minus_1 = n.generators_t.status.loc[t_minus_1, committable_gens]
    first_snap = sns[0]

    # Fix both startup and shutdown constraints
    constraints_to_fix = [
        ("Generator-com-transition-start-up", -1),  # RHS = -status[t-1]
        ("Generator-com-transition-shut-down", +1),  # RHS = +status[t-1]
    ]

    for constraint_name, sign in constraints_to_fix:
        if constraint_name not in n.model.constraints:
            logger.debug(f"{constraint_name} constraint not found in model")
            continue

        constraint = n.model.constraints[constraint_name]
        rhs = constraint.data.rhs
        updated_count = 0

        for gen in committable_gens:
            if gen in rhs.coords.get("Generator-com", []):
                new_rhs = sign * status_t_minus_1[gen]
                rhs.loc[{"snapshot": first_snap, "Generator-com": gen}] = new_rhs
                updated_count += 1

        if updated_count > 0:
            logger.info(
                f"Fixed {constraint_name} at boundary {first_snap}: "
                f"updated RHS for {updated_count} generators"
            )
