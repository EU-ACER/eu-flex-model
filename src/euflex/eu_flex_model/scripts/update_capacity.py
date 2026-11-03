# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypsa
from snakemake.script import snakemake as _snakemake

from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
)

snakemake: Any = _snakemake  # make snakemake a dynamic object for the type checker


class UnitState(Enum):
    """
    Enum representing the state of an EXISTING unit in the EVA.

    Note: Only applies to decommissionable (existing) units, not extendable (new) units.
    The state tracks what action was taken in the current iteration.

    Unit states are saved in two columns:
    - 'state': Post-processed state for ERAA compliance (MOTHBALLED -> CAPACITY_REMOVED
      if unit never demothballs within the iteration)
    - 'raw_state': Original state before post-processing, preserved for cross-iteration
      mothballing cost calculation
    """

    CAPACITY_ADDED = "capacity_added"  # Capacity added/re-added this iteration
    MOTHBALLED = "mothballed"  # Temporarily offline (can be reactivated with costs)
    CAPACITY_REMOVED = "capacity_removed"  # Capacity reduced (was: RETIRED)
    EXTENDED = "extended"  # Operating beyond original lifetime
    EXTENSION_REJECTED = (
        "extension_rejected"  # Past lifetime end, extension evaluated and rejected
    )


def calculate_capacity_delta_extendable(
    npv_ty: float,
    current_capacity: float,
    previous_ty_capacity: float | None,
    increment: float,
    threshold: float,
    ignore_monotonicity: bool,
) -> float:
    """
    Calculate capacity delta for extendable (new) units.

    Rules for new units (p_nom_extendable=True):
    - NPV > threshold: ADD capacity
    - NPV < -threshold: REMOVE only if result >= previous ty capacity
    - |NPV| <= threshold: No change
    - First ty (previous_ty_capacity is None): No constraints

    Parameters
    ----------
    npv_ty : float
        NPV value for the unit at this target year.
    current_capacity : float
        Current p_nom capacity of the unit.
    previous_ty_capacity : float | None
        Capacity of the unit in the previous target year (None for first ty).
    increment : float
        Capacity change increment (MW).
    threshold : float
        NPV threshold for triggering changes.
    ignore_monotonicity : bool
        Ignore the monotonicity condition for that unit

    Returns
    -------
    float
        Capacity delta to apply (positive for addition, negative for removal).
    """
    if npv_ty > threshold:
        return increment
    elif npv_ty < -threshold:
        if previous_ty_capacity is None:
            # First ty: no constraints
            return -increment
        proposed = current_capacity - increment
        if proposed >= previous_ty_capacity or ignore_monotonicity:
            return -increment
        else:
            # Increment overshoots the floor: remove down to it instead of blocking.
            return min(0.0, previous_ty_capacity - current_capacity)
    return 0.0


def calculate_capacity_delta_decommissionable_with_mothballing(
    npv_ty: float,
    current_capacity: float,
    previous_ty_capacity: float | None,
    previous_state: str,
    mothballing_fom: float,
    mothballing_cost: float,
    demothballing_cost: float,
    increment: float,
    threshold: float,
    ignore_monotonicity: bool,
    mothballing_enabled: bool,
    hurdle_rate: float,
    mothballed_capacity_prev_iter: float = 0.0,
    years_mothballed: int = 0,
) -> tuple[float, str, float]:
    """
    Calculate capacity delta for decommissionable units with mothballing support.

    Following ERAA methodology: costs are only calculated at demothballing time
    (when capacity would exceed previous TY capacity). The cost is calculated
    based on PREVIOUS ITERATION state, not current iteration.

    Cost formula (per MW): sum_years(mothballing_fom / (1+r)^year) + demothballing_cost + mothballing_cost
    All costs are in EUR/MW to match NPV units (per MW). The costs were converted
    from EUR/kW to EUR/MW in load_sql_data.py.

    Parameters
    ----------
    npv_ty : float
        NPV value for the unit at this target year.
    current_capacity : float
        Current p_nom capacity of the unit.
    previous_ty_capacity : float | None
        Capacity of the unit in the previous target year (None for first ty).
    previous_state : str
        State of the unit from PREVIOUS ITERATION ("active", "mothballed", etc.).
    mothballing_fom : float
        Annual fixed O&M cost while mothballed (EUR/MW/year).
    mothballing_cost : float
        Capital cost for mothballing (EUR/MW). Currently 0.0 as placeholder -
        ERAA methodology (Table 14) includes this but DB doesn't have it yet.
    demothballing_cost : float
        One-time cost to reactivate a mothballed unit (EUR/MW).
    increment : float
        Capacity change increment (MW).
    threshold : float
        NPV threshold for triggering changes.
    ignore_monotonicity : bool
        Ignore the monotonicity condition for that unit.
    mothballing_enabled : bool
        Whether mothballing is enabled in configuration.
    hurdle_rate : float
        Discount rate for present value calculation.
        Should be the demothballing-specific hurdle rate if available.
    mothballed_capacity_prev_iter : float
        Amount of capacity that was mothballed in PREVIOUS ITERATION (MW).
        This is used to calculate demothballing costs.
    years_mothballed : int
        Number of years the capacity has been mothballed (for cost calculation).

    Returns
    -------
    tuple[float, str, float]
        Tuple of (capacity_delta, new_state, npv_adjusted) where:
        - capacity_delta: Capacity change to apply (positive/negative for add/remove)
        - new_state: New state of the unit ("active", "mothballed", "capacity_removed")
        - npv_adjusted: NPV after any cost adjustments (for logging)
    """
    npv_adjusted = npv_ty

    # Step 1: Evaluate NPV signal - negative NPV means remove capacity
    if npv_ty < -threshold:
        # Remove capacity (mothball if enabled, else capacity_removed)
        new_state = (
            UnitState.MOTHBALLED.value
            if mothballing_enabled
            else UnitState.CAPACITY_REMOVED.value
        )
        return -increment, new_state, npv_adjusted

    # Step 2: Positive NPV - try to add capacity
    elif npv_ty > threshold:
        if previous_ty_capacity is None:
            # First TY: no monotonicity constraints
            return increment, UnitState.CAPACITY_ADDED.value, npv_adjusted

        proposed = current_capacity + increment

        # Check monotonicity constraint (existing units should not exceed previous TY)
        if proposed <= previous_ty_capacity or ignore_monotonicity:
            # Can add capacity within monotonicity constraint
            return increment, UnitState.CAPACITY_ADDED.value, npv_adjusted

        # Would exceed previous TY capacity - this is a demothballing scenario
        elif mothballing_enabled and mothballed_capacity_prev_iter > 0:
            # Calculate demothballing cost per ERAA methodology:
            # Cost per MW = (years × mothballing_fom) + demothballing_cost + mothballing_cost
            # All costs are in EUR/MW to match NPV units (per MW)

            # Calculate mothballing period cost with present value discount
            # Sum of FOM costs over years, discounted to present value
            # Costs are already in EUR/MW (converted in load_sql_data.py)
            mothball_period_cost = 0.0
            for year in range(1, years_mothballed + 1):
                annual_cost = mothballing_fom  # EUR/MW/year (not * capacity)
                # Discount to present value: cost / (1 + rate)^year
                mothball_period_cost += annual_cost / ((1 + hurdle_rate) ** year)

            # Add one-time demothballing cost (already in present value)
            demothball_one_time_cost = demothballing_cost  # EUR/MW (not * capacity)

            # Add mothballing CAPEX if available (placeholder for future DB addition)
            mothball_capex_cost = mothballing_cost  # EUR/MW (not * capacity)

            total_demothball_cost = (
                mothball_period_cost + demothball_one_time_cost + mothball_capex_cost
            )

            # Check if NPV still positive after demothballing costs
            # Both npv_ty and total_demothball_cost are in EUR/MW
            npv_adjusted = npv_ty - total_demothball_cost
            if npv_adjusted >= 0:
                # Demothballing justified - allow capacity increase
                return increment, UnitState.CAPACITY_ADDED.value, npv_adjusted
            # Demothballing not cost-justified - maintain current state
            return 0.0, previous_state, npv_adjusted

        # Increment overshoots the ceiling: add up to it instead of blocking.
        partial_delta = max(0.0, previous_ty_capacity - current_capacity)
        if partial_delta > 0:
            return partial_delta, UnitState.CAPACITY_ADDED.value, npv_adjusted
        return 0.0, previous_state, npv_adjusted

    # Within threshold - no change
    return 0.0, previous_state, npv_adjusted


def evaluate_lifetime_extension(
    target_year: int,
    build_year: int,
    original_lifetime: float,
    npv_ty: float,
    npv_ty_last: float,
    lifetime_extension_years: float,
    lifetime_extension_capex: float,
    lifetime_extension_fom: float,
    capacity: float,
    threshold: float,
    all_tys: list[int],
    extension_enabled: bool,
    hurdle_rate: float,
) -> tuple[bool, float, float]:
    """
    Evaluate whether lifetime extension is justified for an existing unit.

    Lifetime extension is only evaluated for target years after the original
    lifetime has ended. Following ERAA approach:
    - If extension TY <= last modeled TY: Adjust NPV with extension costs (CAPEX + FOM)
    - If extension TY > last modeled TY: Compute NPV_TY5

    Parameters
    ----------
    target_year : int
        Current target year being evaluated.
    build_year : int
        Year when the unit was built/commissioned.
    original_lifetime : float
        Original economic lifetime of the unit (years).
    npv_ty : float
        NPV value for the unit at this target year.
    npv_ty_last : float
        NPV value at the last modeled target year.
    lifetime_extension_years : float
        Available years of lifetime extension.
    lifetime_extension_capex : float
        Capital cost for lifetime extension (EUR/MW).
    lifetime_extension_fom : float
        Annual fixed O&M cost for lifetime extension (EUR/MW/year).
    capacity : float
        Current capacity of the unit (MW).
    threshold : float
        NPV threshold for triggering changes.
    all_tys : list[int]
        List of all modeled target years.
    extension_enabled : bool
        Whether lifetime extension is enabled in configuration.
    hurdle_rate : float
        Discount rate for present value calculation of extension costs.
        Should be the lifetime-extension-specific hurdle rate if available.

    Returns
    -------
    tuple[bool, float, float]
        Tuple of (should_extend, extended_lifetime, adjusted_npv) where:
        - should_extend: Whether lifetime extension is justified
        - extended_lifetime: New lifetime if extended, 0 otherwise
        - adjusted_npv: NPV after subtracting extension costs (equals npv_ty when not applicable)
    """
    if not extension_enabled:
        return False, 0.0, npv_ty

    # Calculate when original lifetime ends
    original_lifetime_end = build_year + int(original_lifetime)

    # Check if we're in the extension period
    if target_year <= original_lifetime_end:
        return False, 0.0, npv_ty  # Not yet in extension period

    # Check if extension years are available
    if lifetime_extension_years <= 0 or pd.isna(lifetime_extension_years):
        return False, 0.0, npv_ty  # No extension available

    last_modeled_ty = max(all_tys)

    if target_year <= last_modeled_ty:
        # Extension TY is within modeled period
        # Adjust NPV with lifetime extension costs (CAPEX + FOM, discounted)
        # Costs are already in EUR/MW (converted in load_sql_data.py)
        extension_cost = 0.0

        # Annualize CAPEX over extension period using hurdle rate
        # Annuity factor = r / (1 - (1 + r)^-n)
        # Note: All costs are in EUR/MW to match NPV units (per MW)
        if lifetime_extension_capex > 0 and lifetime_extension_years > 0:
            annuity_factor = hurdle_rate / (
                1 - (1 + hurdle_rate) ** (-lifetime_extension_years)
            )
            capex_annualized = lifetime_extension_capex * annuity_factor
            extension_cost += capex_annualized

        # Discount FOM costs over extension years to present value
        for year in range(1, int(lifetime_extension_years) + 1):
            annual_fom = lifetime_extension_fom
            extension_cost += annual_fom / ((1 + hurdle_rate) ** year)

        adjusted_npv = npv_ty - extension_cost

        if adjusted_npv > threshold:
            # Extension justified
            new_lifetime = original_lifetime + lifetime_extension_years
            return True, new_lifetime, adjusted_npv
        elif adjusted_npv < -threshold:
            # Extension not justified - keep original lifetime
            return False, original_lifetime, adjusted_npv
        else:
            # Within threshold - no change, keep original lifetime
            return False, original_lifetime, adjusted_npv
    else:
        # Extension TY is beyond modeled period (NPV_TY5 calculation)
        years_beyond = target_year - last_modeled_ty
        # Estimate extra NPV using last TY's NPV as proxy for annual profit
        # This is a simplified estimate
        extra_npv = npv_ty_last * years_beyond

        # Calculate extension costs over the years beyond
        extension_costs = 0.0

        # Annualize CAPEX (same as above)
        # Note: All costs are in EUR/MW to match NPV units (per MW)
        if lifetime_extension_capex > 0 and years_beyond > 0:
            annuity_factor = hurdle_rate / (1 - (1 + hurdle_rate) ** (-years_beyond))
            capex_annualized = lifetime_extension_capex * annuity_factor
            extension_costs += capex_annualized

        # Discount FOM costs
        for year in range(1, years_beyond + 1):
            annual_fom = lifetime_extension_fom
            extension_costs += annual_fom / ((1 + hurdle_rate) ** year)

        npv_ty5 = npv_ty_last + extra_npv - extension_costs

        if npv_ty5 > threshold:
            new_lifetime = original_lifetime + lifetime_extension_years
            return True, new_lifetime, npv_ty5
        else:
            # NPV_ty5 below threshold - keep original lifetime
            return False, original_lifetime, npv_ty5


def extract_eva_cost_params(
    networks_by_ty: dict[int, list[pypsa.Network]],
    components: list[str],
) -> dict[int, pd.DataFrame]:
    """
    Extract EVA cost parameters (mothballing, lifetime extension) from networks.

    Extracts parameters from each target year's network separately to support
    TY-varying costs if they are added in the future.

    Note: Mothballing parameters only apply to generators, not storage units.
    Storage units don't have mothballing data in the DB.

    Parameters
    ----------
    networks_by_ty : dict[int, list[pypsa.Network]]
        Dictionary mapping target years to lists of PyPSA networks.
    components : list[str]
        List of component types to extract (e.g., ["Generator", "StorageUnit"]).

    Returns
    -------
    dict[int, pd.DataFrame]
        Dictionary mapping target years to DataFrames with EVA cost parameters.
        Each DataFrame has:
        Index: MultiIndex (component, unit)
        Columns (all components):
            - lifetime_extension_capex: CAPEX for lifetime extension (EUR/MW)
            - lifetime_extension_years: Available years of extension
            - Lifetime_Extension_FOM: Annual FOM during extension (EUR/MW/year)
            - lifetime: Original unit lifetime (years)
            - build_year: Year unit was built/commissioned
            - hurdle_rate: Standard hurdle rate
            - hurdle_rate_demothballing: Hurdle rate for demothballing decisions
            - hurdle_rate_lifetime_extension: Hurdle rate for extension decisions
        Columns (generators only, 0.0 for storage units):
            - Mothballing_FOM: Annual FOM while mothballed (EUR/MW/year)
            - Mothballing_Cost: CAPEX for mothballing (EUR/MW)
            - Demothballing_Cost: One-time cost to demothball (EUR/MW)
    """
    # Base cost columns for all components
    base_cost_columns = [
        "lifetime_extension_capex",  # This is extension CAPEX
        "lifetime_extension_years",  # Extension years from Economic_Lifetime
        "Lifetime_Extension_FOM",
        "lifetime",
        "build_year",
        "hurdle_rate",
        "hurdle_rate_lifetime_extension",
        "lifetime_extendable",
    ]

    # Mothballing columns only apply to generators (not storage units)
    mothballing_columns = [
        "Mothballing_FOM",
        "Mothballing_Cost",
        "Demothballing_Cost",
        "hurdle_rate_demothballing",
    ]

    result: dict[int, pd.DataFrame] = {}

    for ty, networks in networks_by_ty.items():
        # Use first network for this TY as reference
        n = networks[0]

        all_data = []

        component_map = {
            "Generator": n.generators,
            "StorageUnit": n.storage_units,
        }

        for c in components:
            if c not in component_map:
                continue
            component_table = component_map[c]

            # Determine which columns to extract based on component type
            # Mothballing only applies to generators, not storage units
            if c == "Generator":
                cost_columns = base_cost_columns + mothballing_columns
            else:
                cost_columns = base_cost_columns

            for unit in component_table.index:
                row = {"component": c, "unit": unit}

                for col in cost_columns:
                    if col in component_table.columns:
                        val = component_table.loc[unit, col]
                        row[col] = val if not pd.isna(val) else 0.0
                    else:
                        row[col] = 0.0

                all_data.append(row)

        if all_data:
            df = pd.DataFrame(all_data)
            df = df.set_index(["component", "unit"])
            result[ty] = df
        else:
            result[ty] = pd.DataFrame()

    return result


def load_previous_unit_states(
    unit_states_path: str | None,
) -> dict[tuple[str, str, int], str]:
    """
    Load unit states from previous iteration's CSV file.

    Uses 'raw_state' column if available (for cross-iteration mothballing detection),
    otherwise falls back to 'state' column for backward compatibility.

    The raw_state preserves the original state before post-processing, which is
    needed to correctly detect mothballing and calculate demothballing costs
    across iterations.

    Parameters
    ----------
    unit_states_path : str or None
        Path to the previous iteration's unit_states CSV file.
        If None or empty, returns an empty dict.

    Returns
    -------
    dict[tuple[str, str, int], str]
        Dictionary mapping (component, unit, ty) -> state string.
        States are: "capacity_added", "mothballed", "capacity_removed", "extended".
    """
    if not unit_states_path:
        return {}

    path = Path(unit_states_path)
    if not path.exists():
        logging.warning(
            f"Previous unit states file not found: {unit_states_path}. "
            "Starting with empty state tracker."
        )
        return {}

    try:
        df = pd.read_csv(path)
        states = {}
        # Use raw_state if available (preserves MOTHBALLED before post-processing)
        # Fall back to state for backward compatibility
        state_column = "raw_state" if "raw_state" in df.columns else "state"
        if state_column == "raw_state":
            logging.debug("Using raw_state column for cross-iteration state tracking")
        for _, row in df.iterrows():
            key = (row["component"], row["unit"], int(row["ty"]))
            states[key] = row[state_column]
        logging.info(
            f"Loaded {len(states)} unit states from previous iteration: {unit_states_path}"
        )
        return states
    except Exception as e:
        logging.warning(
            f"Error loading previous unit states from {unit_states_path}: {e}. "
            "Starting with empty state tracker."
        )
        return {}


def scale_generator_costs(
    c: str,
    unit: str,
    component_table: pd.DataFrame,
    current_cap: float,
    updated_cap: float,
) -> None:
    """
    Scale generator cost attributes based on capacity change.

    Parameters
    ----------
    c : str
        Component type (should be "Generator" to apply scaling).
    unit : str
        Name of the unit.
    component_table : pd.DataFrame
        DataFrame containing component attributes.
    current_cap : float
        Current capacity of the unit.
    updated_cap : float
        Updated capacity of the unit.
    """
    if c == "Generator" and current_cap > 0:
        scaling_ratio = updated_cap / current_cap
        cost_attrs = [
            "start_up_cost",
            "shut_down_cost",
            "stand_by_cost",
        ]
        for cost_attr in cost_attrs:
            if cost_attr in component_table.columns:
                component_table.loc[unit, cost_attr] *= scaling_ratio


def verify_monotonicity(
    networks_by_ty: dict[int, list[pypsa.Network]],
) -> dict[int, list[pypsa.Network]]:
    """
    Verify monotonicity of initial capacities across target years.

    For existing units (``is_existing=True``), capacity should be
    non-increasing across TYs.  For new units (``is_existing=False``),
    capacity should be non-decreasing.  When the initial data violates
    these constraints, ``ignore_monotonicity`` is set to ``True`` for that
    unit at that target year so the capacity-update algorithm will not
    enforce the monotonicity rule.

    The column is written into the ``generators`` and ``storage_units``
    DataFrames of every network.

    Parameters
    ----------
    networks_by_ty : dict[int, list[pypsa.Network]]
        Dictionary mapping target years to lists of PyPSA networks.

    Returns
    -------
    dict[int, list[pypsa.Network]]
        Same dictionary with ``ignore_monotonicity`` column added to the
        ``generators`` and ``storage_units`` DataFrames of each network.
    """
    sorted_tys = sorted(networks_by_ty.keys())

    for component_attr in ["generators", "storage_units"]:
        prev_capacities: pd.Series | None = None

        for ty_idx, ty in enumerate(sorted_tys):
            ref_table = getattr(networks_by_ty[ty][0], component_attr)

            if ref_table.empty:
                prev_capacities = None
                continue

            current_capacities = ref_table["p_nom"].copy()

            if ty_idx == 0 or prev_capacities is None:
                ignore_flags = pd.Series(False, index=ref_table.index)
            else:
                ignore_flags = pd.Series(False, index=ref_table.index)

                for unit in ref_table.index:
                    if unit not in prev_capacities.index:
                        continue

                    # TODO, replace patching function below by this once new pre-builts or read from db are available
                    # unit_is_existing = bool(ref_table.loc[unit, "is_existing"])
                    unit_is_existing = (
                        bool(ref_table.loc[unit, "is_existing"])
                        if "is_existing" in ref_table.columns
                        else True
                    )

                    cap_ty = current_capacities[unit]
                    cap_prev = prev_capacities[unit]

                    if unit_is_existing:
                        ignore_flags[unit] = cap_ty > cap_prev
                    else:  # new unit
                        ignore_flags[unit] = cap_ty < cap_prev

            # Apply to all networks for this ty
            for network in networks_by_ty[ty]:
                comp_table = getattr(network, component_attr)
                comp_table["ignore_monotonicity"] = ignore_flags

            prev_capacities = current_capacities

    return networks_by_ty


def extract_capacities_from_networks(
    networks_by_ty: dict[int, list[pypsa.Network]],
    components: list[str],
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame
]:
    """
    Extract capacity data from networks into DataFrames.

    Parameters
    ----------
    networks_by_ty : dict[int, list[pypsa.Network]]
        Dictionary mapping target years to lists of PyPSA networks.
    components : list[str]
        List of component types to extract (e.g., ["Generator", "StorageUnit"]).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]
        - capacities: p_nom values, MultiIndex (component, unit) x TY columns
        - p_nom_max: max capacity, same structure
        - p_nom_min: min capacity, same structure
        - is_extendable: boolean Series by (component, unit)
        - is_decommissionable: boolean Series by (component, unit)
        - ignore_monotonicity: boolean, MultiIndex (component, unit) x TY columns
    """
    sorted_tys = sorted(networks_by_ty.keys())

    # Collect data from first network of each TY
    # (assuming all networks for same TY have same units)
    capacities_data: dict[int, dict[tuple[str, str], float]] = {}
    p_nom_max_data: dict[int, dict[tuple[str, str], float]] = {}
    p_nom_min_data: dict[int, dict[tuple[str, str], float]] = {}
    is_extendable_data: dict[tuple[str, str], bool] = {}
    is_decommissionable_data: dict[tuple[str, str], bool] = {}
    ignore_monotonicity_data: dict[int, dict[tuple[str, str], bool]] = {}

    for ty in sorted_tys:
        # Use first network of each TY for structure
        n = networks_by_ty[ty][0]
        capacities_data[ty] = {}
        p_nom_max_data[ty] = {}
        p_nom_min_data[ty] = {}
        ignore_monotonicity_data[ty] = {}

        component_map = {
            "Generator": n.generators,
            "StorageUnit": n.storage_units,
        }

        for c in components:
            if c not in component_map:
                continue
            component_table = component_map[c]
            attr = "e" if c == "Store" else "p"

            for unit in component_table.index:
                key = (c, unit)
                capacities_data[ty][key] = component_table.loc[unit, f"{attr}_nom"]

                max_val = component_table.loc[unit, f"{attr}_nom_max"]
                p_nom_max_data[ty][key] = max_val if not pd.isna(max_val) else np.inf

                min_val = component_table.loc[unit, f"{attr}_nom_min"]
                p_nom_min_data[ty][key] = min_val if not pd.isna(min_val) else 0.0

                # Extract ignore_monotonicity flag (per unit per TY)
                if "ignore_monotonicity" in component_table.columns:
                    ignore_monotonicity_data[ty][key] = bool(
                        component_table.loc[unit, "ignore_monotonicity"]
                    )
                else:
                    ignore_monotonicity_data[ty][key] = False

                # Extract extendable/decommissionable flags (same across TYs)
                if key not in is_extendable_data:
                    is_extendable_data[key] = bool(
                        component_table.loc[unit, f"{attr}_nom_extendable"]
                    )
                    decommissionable_val = component_table.loc[
                        unit, f"{attr}_nom_decommissionable"
                    ]
                    # bool(float("nan")) is True, so a unit whose
                    # p_nom_decommissionable was never explicitly set (NaN,
                    # e.g. implicit DSR storage units) would otherwise be
                    # incorrectly treated as decommissionable and enter the
                    # capacity-change loop despite p_nom_extendable=False.
                    is_decommissionable_data[key] = (
                        bool(decommissionable_val)
                        if pd.notna(decommissionable_val)
                        else False
                    )

    # Build DataFrames
    all_units = list(is_extendable_data.keys())
    index = pd.MultiIndex.from_tuples(all_units, names=["component", "unit"])

    capacities = pd.DataFrame(
        {ty: [capacities_data[ty].get(u, 0.0) for u in all_units] for ty in sorted_tys},
        index=index,
    )
    p_nom_max = pd.DataFrame(
        {
            ty: [p_nom_max_data[ty].get(u, np.inf) for u in all_units]
            for ty in sorted_tys
        },
        index=index,
    )
    p_nom_min = pd.DataFrame(
        {ty: [p_nom_min_data[ty].get(u, 0.0) for u in all_units] for ty in sorted_tys},
        index=index,
    )
    is_extendable = pd.Series(is_extendable_data, name="is_extendable")
    is_extendable.index = pd.MultiIndex.from_tuples(
        is_extendable.index, names=["component", "unit"]
    )
    is_decommissionable = pd.Series(
        is_decommissionable_data, name="is_decommissionable"
    )
    is_decommissionable.index = pd.MultiIndex.from_tuples(
        is_decommissionable.index, names=["component", "unit"]
    )
    ignore_monotonicity = pd.DataFrame(
        {
            ty: [ignore_monotonicity_data[ty].get(u, False) for u in all_units]
            for ty in sorted_tys
        },
        index=index,
    )

    return (
        capacities,
        p_nom_max,
        p_nom_min,
        is_extendable,
        is_decommissionable,
        ignore_monotonicity,
    )


def update_capacities(
    capacities: pd.DataFrame,
    p_nom_max: pd.DataFrame,
    p_nom_min: pd.DataFrame,
    is_extendable: pd.Series,
    is_decommissionable: pd.Series,
    ignore_monotonicity: pd.DataFrame,
    npv: pd.DataFrame,
    threshold: float,
    increment: float,
    eva_cost_params: dict[int, pd.DataFrame] | None = None,
    mothballing_enabled: bool = False,
    lifetime_extension_enabled: bool = False,
    previous_iteration_states: dict[tuple[str, str, int], str] | None = None,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    Update capacities based on NPV signals.

    Pure function that calculates updated capacities without modifying networks.
    Changes are propagated to all future TYs after each decision.

    Monotonicity constraints (unless initial monotonicity was not respected):
    - New units (extendable): capacity must be non-decreasing across TYs
    - Existing units (decommissionable): capacity must be non-increasing across TYs

    Parameters
    ----------
    capacities : pd.DataFrame
        Current p_nom values. MultiIndex (component, unit), columns are TYs.
    p_nom_max : pd.DataFrame
        Maximum capacity bounds, same structure as capacities.
    p_nom_min : pd.DataFrame
        Minimum capacity bounds, same structure as capacities.
    is_extendable : pd.Series
        Boolean indicating if unit is extendable (new). Index: (component, unit).
    is_decommissionable : pd.Series
        Boolean indicating if unit is decommissionable (existing).
        Index: (component, unit).
    ignore_monotonicity : pd.DataFrame
        Boolean flags per (component, unit) and TY. When True, the
        monotonicity constraint is not enforced for that unit at that TY.
    npv : pd.DataFrame
        NPV values. MultiIndex (component, name), columns are TYs (as strings).
    threshold : float
        NPV threshold for triggering capacity changes.
    increment : float
        Capacity change increment (MW).
    eva_cost_params : dict[int, pd.DataFrame] | None
        EVA cost parameters (mothballing, lifetime extension) extracted from networks,
        keyed by target year. Each DataFrame has Index: MultiIndex (component, unit).
        Columns: Mothballing_FOM, Demothballing_Cost, lifetime_extension_capex,
        Lifetime_Extension_FOM, lifetime, build_year.
    mothballing_enabled : bool
        Enable mothballing for existing units. When True, units can be
        temporarily offline and reactivated later with associated costs.
    lifetime_extension_enabled : bool
        Enable lifetime extension evaluation for existing units beyond
        original lifetime with associated costs.
    previous_iteration_states : dict | None
        Unit states from the previous iteration. Used to persist state across
        EVA iterations. Dict mapping (component, unit, ty) -> state string.

    Returns
    -------
    tuple[pd.DataFrame, float, pd.DataFrame]
        - updated_capacities: DataFrame with same structure as input capacities
        - max_abs_npv: Maximum absolute NPV among units with capacity changes
        - unit_states: DataFrame tracking unit states per TY with columns:
          component, unit, ty, state, p_nom, npv_raw, npv_adjusted, lifetime_extended
    """
    updated = capacities.copy()
    sorted_tys = list(capacities.columns)
    max_abs_npv = 0.0

    # Track capacity for monotonicity constraints
    # Structure: {(component, unit): capacity_after_previous_ty}
    capacity_tracker: dict[tuple[str, str], float] = {}

    # Track unit states across TYs
    # Structure: {(component, unit, ty): state}
    # Initialize with previous iteration's states if available (for state persistence)
    if previous_iteration_states:
        state_tracker = previous_iteration_states.copy()
        logging.info(
            f"Initialized state tracker with {len(state_tracker)} states from previous iteration"
        )
    else:
        state_tracker = {}

    # Track mothballed capacity per unit from PREVIOUS ITERATION
    # Structure: {(component, unit, ty): mothballed_capacity}
    # This is needed to calculate demothballing costs
    prev_iter_mothballed: dict[tuple[str, str, int], float] = {}

    # Track when mothballing started for each unit (from PREVIOUS ITERATION)
    # Structure: {(component, unit): first_ty_when_mothballed}
    mothball_start_ty: dict[tuple[str, str], int | None] = {}

    # Calculate mothballed capacity from previous iteration states
    if previous_iteration_states:
        for (comp, unit_name, ty), state in previous_iteration_states.items():
            if state == UnitState.MOTHBALLED.value:
                # Get the capacity that was mothballed (capacity before mothballing - capacity after)
                # For now, approximate as the current capacity at that TY
                if (comp, unit_name) in capacities.index and ty in capacities.columns:
                    cap_at_ty = capacities.loc[(comp, unit_name), ty]
                    prev_iter_mothballed[(comp, unit_name, ty)] = cap_at_ty

                # Track when mothballing started
                if (comp, unit_name) not in mothball_start_ty:
                    mothball_start_ty[(comp, unit_name)] = ty

    # Track lifetime extension decisions per unit for consistency both within this
    # iteration (across TYs) and carried over from the previous iteration.
    # Value: (granted: bool, new_lifetime: float, adjusted_npv: float)
    # new_lifetime is 0.0 and adjusted_npv is 0.0 when not yet known (carry-in from prior iter).
    # EXTENDED takes priority over EXTENSION_REJECTED if both appear across TYs.
    extension_decision_tracker: dict[tuple[str, str], tuple[bool, float, float]] = {}
    if previous_iteration_states:
        for (comp, unit_name, _ty), state in previous_iteration_states.items():
            key = (comp, unit_name)
            if state == UnitState.EXTENDED.value:
                extension_decision_tracker[key] = (True, 0.0, 0.0)
            elif (
                state == UnitState.EXTENSION_REJECTED.value
                and key not in extension_decision_tracker
            ):
                extension_decision_tracker[key] = (False, 0.0, 0.0)

    # Track unit states for output
    unit_states_records: list[dict] = []

    for ty_idx, ty in enumerate(sorted_tys):
        ty_str = str(ty)

        # Get EVA cost params for this TY (costs may vary by TY)
        eva_cost_params_ty = (
            eva_cost_params.get(ty) if eva_cost_params is not None else None
        )

        for idx in updated.index:
            component, unit = idx

            # Check if unit has NPV data
            if component not in npv.index.get_level_values(0):
                continue
            npv_c = npv.loc[component]
            if unit not in npv_c.index:
                continue
            if ty_str not in npv_c.columns:
                continue

            npv_ty_val = npv_c.loc[unit, ty_str]
            current_cap = updated.loc[idx, ty]
            prev_cap = capacity_tracker.get(idx)

            # Determine if extendable or decommissionable
            unit_is_extendable = is_extendable.get(idx, False)
            unit_is_decommissionable = is_decommissionable.get(idx, False)

            # Skip if neither extendable nor decommissionable
            if not unit_is_extendable and not unit_is_decommissionable:
                capacity_tracker[idx] = current_cap
                continue

            # Look up ignore_monotonicity for this unit at this TY
            unit_ignore_mono = bool(ignore_monotonicity.loc[idx, ty])

            # Get previous state (from previous TY or default to active)
            if ty_idx > 0:
                prev_ty = sorted_tys[ty_idx - 1]
                previous_state = state_tracker.get(
                    (component, unit, prev_ty), UnitState.CAPACITY_ADDED.value
                )
            else:
                previous_state = UnitState.CAPACITY_ADDED.value

            # Initialize state tracking variables
            new_state = previous_state
            npv_adjusted = npv_ty_val
            lifetime_extended = False
            new_lifetime = 0.0  # Track new lifetime if extended
            delta = 0.0

            # Calculate delta based on unit type
            if unit_is_extendable:
                # New units: no mothballing, no lifetime extension
                delta = calculate_capacity_delta_extendable(
                    npv_ty_val,
                    current_cap,
                    prev_cap,
                    increment,
                    threshold,
                    unit_ignore_mono,
                )
                new_state = UnitState.CAPACITY_ADDED.value
            else:  # decommissionable (existing units)
                # Mothballing only applies to generators, not storage units
                # Storage units don't have mothballing data in the DB
                unit_mothballing_enabled = (
                    mothballing_enabled and component == "Generator"
                )

                # Initialize cost parameters with defaults
                mothballing_fom = 0.0
                mothballing_cost = 0.0
                demothballing_cost = 0.0
                unit_hurdle_rate = snakemake.config["costs"]["discount_rate"]
                unit_hurdle_rate_demothballing = unit_hurdle_rate
                unit_hurdle_rate_lifetime_extension = unit_hurdle_rate
                demoth_hr_explicitly_set = False

                if eva_cost_params_ty is not None and idx in eva_cost_params_ty.index:
                    # Get mothballing cost parameters (generators only)
                    # Storage units don't have these columns in the DataFrame
                    if unit_mothballing_enabled:
                        mothballing_fom = eva_cost_params_ty.loc[idx, "Mothballing_FOM"]
                        mothballing_cost = eva_cost_params_ty.loc[
                            idx, "Mothballing_Cost"
                        ]
                        demothballing_cost = eva_cost_params_ty.loc[
                            idx, "Demothballing_Cost"
                        ]

                        # Demothballing-specific hurdle rate (generators only)
                        hr_demoth = eva_cost_params_ty.loc[
                            idx, "hurdle_rate_demothballing"
                        ]
                        if not pd.isna(hr_demoth) and hr_demoth > 0:
                            unit_hurdle_rate_demothballing = hr_demoth
                            demoth_hr_explicitly_set = True

                    # Use unit-specific hurdle rates if available (all components)
                    hr = eva_cost_params_ty.loc[idx, "hurdle_rate"]
                    if not pd.isna(hr) and hr > 0:
                        unit_hurdle_rate = hr

                    # Update demothballing hurdle rate default if not set above
                    if not demoth_hr_explicitly_set:
                        unit_hurdle_rate_demothballing = unit_hurdle_rate

                    # Lifetime extension-specific hurdle rate (all components)
                    hr_ext = eva_cost_params_ty.loc[
                        idx, "hurdle_rate_lifetime_extension"
                    ]
                    if not pd.isna(hr_ext) and hr_ext > 0:
                        unit_hurdle_rate_lifetime_extension = hr_ext
                    else:
                        unit_hurdle_rate_lifetime_extension = unit_hurdle_rate

                # Calculate mothballed capacity and years from PREVIOUS ITERATION
                # This is used for demothballing cost calculation
                mothballed_capacity = 0.0
                years_mothballed = 0

                if unit_mothballing_enabled:
                    # Get mothballed capacity from previous iteration
                    # Sum up all mothballed capacity for this unit across TYs
                    for prev_ty in sorted_tys[: ty_idx + 1]:
                        key = (component, unit, prev_ty)
                        if key in prev_iter_mothballed:
                            mothballed_capacity = max(
                                mothballed_capacity, prev_iter_mothballed[key]
                            )

                    # Calculate years since mothballing started
                    start_ty = mothball_start_ty.get((component, unit))
                    if start_ty is not None and start_ty < ty:
                        years_mothballed = ty - start_ty

                # Use mothballing-aware capacity delta calculation for all decommissionable units
                # When unit_mothballing_enabled=False, this behaves like the standard logic
                delta, new_state, npv_adjusted = (
                    calculate_capacity_delta_decommissionable_with_mothballing(
                        npv_ty=npv_ty_val,
                        current_capacity=current_cap,
                        previous_ty_capacity=prev_cap,
                        previous_state=previous_state,
                        mothballing_fom=mothballing_fom,
                        mothballing_cost=mothballing_cost,
                        demothballing_cost=demothballing_cost,
                        increment=increment,
                        threshold=threshold,
                        ignore_monotonicity=unit_ignore_mono,
                        mothballing_enabled=unit_mothballing_enabled,
                        mothballed_capacity_prev_iter=mothballed_capacity,
                        years_mothballed=years_mothballed,
                        hurdle_rate=unit_hurdle_rate_demothballing,
                    )
                )

                # Lifetime extension evaluation for decommissionable (existing) units
                if lifetime_extension_enabled and eva_cost_params_ty is not None:
                    if idx in eva_cost_params_ty.index:
                        # Per-asset extendability check (from lifetime_extendable column in DB)
                        unit_lifetime_extendable = (
                            bool(eva_cost_params_ty.loc[idx, "lifetime_extendable"])
                            if "lifetime_extendable" in eva_cost_params_ty.columns
                            else True  # Default True for backward compatibility
                        )

                        if unit_lifetime_extendable:
                            build_year = eva_cost_params_ty.loc[idx, "build_year"]
                            original_lifetime = eva_cost_params_ty.loc[idx, "lifetime"]

                            ext_years_raw = eva_cost_params_ty.loc[
                                idx, "lifetime_extension_years"
                            ]
                            ext_capex_raw = eva_cost_params_ty.loc[
                                idx, "lifetime_extension_capex"
                            ]
                            ext_fom_raw = eva_cost_params_ty.loc[
                                idx, "Lifetime_Extension_FOM"
                            ]

                            ext_years = (
                                ext_years_raw if not pd.isna(ext_years_raw) else 0.0
                            )
                            ext_capex = (
                                ext_capex_raw if not pd.isna(ext_capex_raw) else 0.0
                            )
                            ext_fom = ext_fom_raw if not pd.isna(ext_fom_raw) else 0.0

                            if (
                                not pd.isna(build_year)
                                and not pd.isna(original_lifetime)
                                and original_lifetime > 0
                            ):
                                original_lifetime_end = int(build_year) + int(
                                    original_lifetime
                                )

                                # Only act when we are past the original lifetime end
                                if ty > original_lifetime_end:
                                    prior = extension_decision_tracker.get(idx)

                                    if prior is not None:
                                        # Decision carried from an earlier TY or iteration
                                        prior_decision, prior_new_lifetime, _ = prior
                                        if prior_decision:
                                            # Previously granted — maintain capacity
                                            if (
                                                prev_cap is not None
                                                and current_cap < prev_cap
                                            ):
                                                delta = prev_cap - current_cap
                                            new_state = UnitState.EXTENDED.value
                                            lifetime_extended = True
                                            new_lifetime = prior_new_lifetime
                                            logging.debug(
                                                f"Extension carry-forward for "
                                                f"{component} {unit} at TY {ty}"
                                            )
                                        else:
                                            # Previously rejected — keep at minimum and always mark state
                                            min_cap = p_nom_min.loc[idx, ty]
                                            new_state = (
                                                UnitState.EXTENSION_REJECTED.value
                                            )
                                            if current_cap > min_cap:
                                                delta = min_cap - current_cap
                                    else:
                                        # First evaluation in the extension window
                                        last_ty = sorted_tys[-1]
                                        last_ty_str = str(last_ty)
                                        npv_ty_last = 0.0
                                        if last_ty_str in npv_c.columns:
                                            npv_ty_last = npv_c.loc[unit, last_ty_str]

                                        (
                                            should_extend,
                                            extended_lifetime,
                                            ext_adjusted_npv,
                                        ) = evaluate_lifetime_extension(
                                            target_year=ty,
                                            build_year=int(build_year),
                                            original_lifetime=original_lifetime,
                                            npv_ty=npv_ty_val,
                                            npv_ty_last=npv_ty_last,
                                            lifetime_extension_years=ext_years,
                                            lifetime_extension_capex=ext_capex,
                                            lifetime_extension_fom=ext_fom,
                                            capacity=current_cap,
                                            threshold=threshold,
                                            all_tys=sorted_tys,
                                            extension_enabled=lifetime_extension_enabled,
                                            hurdle_rate=unit_hurdle_rate_lifetime_extension,
                                        )
                                        npv_adjusted = ext_adjusted_npv

                                        extension_decision_tracker[idx] = (
                                            should_extend,
                                            extended_lifetime if should_extend else 0.0,
                                            ext_adjusted_npv,
                                        )

                                        if should_extend:
                                            lifetime_extended = True
                                            new_lifetime = extended_lifetime
                                            new_state = UnitState.EXTENDED.value
                                            if (
                                                prev_cap is not None
                                                and current_cap < prev_cap
                                            ):
                                                delta = prev_cap - current_cap
                                            logging.debug(
                                                f"Lifetime extension granted for "
                                                f"{component} {unit} at TY {ty}: "
                                                f"new_lifetime={extended_lifetime}"
                                            )
                                        else:
                                            min_cap = p_nom_min.loc[idx, ty]
                                            if current_cap > min_cap:
                                                delta = min_cap - current_cap
                                                new_state = (
                                                    UnitState.EXTENSION_REJECTED.value
                                                )
                                            logging.debug(
                                                f"Lifetime extension rejected for "
                                                f"{component} {unit} at TY {ty}: "
                                                "setting capacity to minimum"
                                            )

            # Apply delta to current TY and propagate to future TYs
            if delta != 0:
                if abs(npv_ty_val) > max_abs_npv:
                    max_abs_npv = abs(npv_ty_val)

                for future_ty in sorted_tys[ty_idx:]:
                    future_cap = updated.loc[idx, future_ty]
                    future_max = p_nom_max.loc[idx, future_ty]
                    future_min = p_nom_min.loc[idx, future_ty]
                    new_cap = np.clip(future_cap + delta, future_min, future_max)
                    updated.loc[idx, future_ty] = new_cap

            # Update trackers
            capacity_tracker[idx] = updated.loc[idx, ty]
            state_tracker[(component, unit, ty)] = new_state

            # Record unit state for output
            # raw_state preserves the original state before post-processing
            # state may be modified by post-processing (e.g., MOTHBALLED -> CAPACITY_REMOVED)
            # raw_state is used for cross-iteration cost calculation
            unit_states_records.append(
                {
                    "component": component,
                    "unit": unit,
                    "ty": ty,
                    "state": new_state,
                    "raw_state": new_state,  # Preserved for cross-iteration mothballing detection
                    "p_nom": updated.loc[idx, ty],
                    "npv_raw": npv_ty_val,
                    "npv_adjusted": npv_adjusted,
                    "lifetime_extended": lifetime_extended,
                    "new_lifetime": new_lifetime,  # Pass to apply_lifetime_extensions
                }
            )

    # TY5 Evaluation: Check for lifetime extension BEYOND last modeled TY
    # Per ERAA methodology: NPV_ty5 = NPV_ty4 + extra_npv - extension_costs
    # TY5 changes do NOT contribute to max_abs_npv (doesn't block convergence)
    if lifetime_extension_enabled and eva_cost_params is not None:
        last_ty = sorted_tys[-1]
        last_ty_str = str(last_ty)

        # Use last TY's cost params for TY5 evaluation (extrapolation beyond modeled years)
        eva_cost_params_last_ty = eva_cost_params.get(last_ty)
        if eva_cost_params_last_ty is None or eva_cost_params_last_ty.empty:
            eva_cost_params_last_ty = None

        for idx in updated.index:
            component, unit = idx

            # Only evaluate decommissionable (existing) units
            if not is_decommissionable.get(idx, False):
                continue

            # Skip if no EVA cost params
            if (
                eva_cost_params_last_ty is None
                or idx not in eva_cost_params_last_ty.index
            ):
                continue

            # Skip if unit is not eligible for lifetime extension
            if "lifetime_extendable" in eva_cost_params_last_ty.columns:
                if not bool(eva_cost_params_last_ty.loc[idx, "lifetime_extendable"]):
                    continue

            build_year = eva_cost_params_last_ty.loc[idx, "build_year"]
            original_lifetime = eva_cost_params_last_ty.loc[idx, "lifetime"]
            # Get extension parameters (with NaN handling)
            ext_years_raw = eva_cost_params_last_ty.loc[idx, "lifetime_extension_years"]
            ext_capex_raw = eva_cost_params_last_ty.loc[idx, "lifetime_extension_capex"]
            ext_fom_raw = eva_cost_params_last_ty.loc[idx, "Lifetime_Extension_FOM"]
            # Get unit-specific hurdle rate for lifetime extension
            ty5_hurdle_rate = eva_cost_params_last_ty.loc[
                idx, "hurdle_rate_lifetime_extension"
            ]
            if pd.isna(ty5_hurdle_rate) or ty5_hurdle_rate <= 0:
                # Fallback to standard hurdle rate, then global discount_rate
                ty5_hurdle_rate = eva_cost_params_last_ty.loc[idx, "hurdle_rate"]
                if pd.isna(ty5_hurdle_rate) or ty5_hurdle_rate <= 0:
                    ty5_hurdle_rate = snakemake.config["costs"]["discount_rate"]

            # Convert NaN to defaults
            ext_years = ext_years_raw if not pd.isna(ext_years_raw) else 0.0
            ext_capex = ext_capex_raw if not pd.isna(ext_capex_raw) else 0.0
            ext_fom = ext_fom_raw if not pd.isna(ext_fom_raw) else 0.0

            # Skip if data not available
            if (
                pd.isna(build_year)
                or pd.isna(original_lifetime)
                or original_lifetime <= 0
                or ext_years <= 0
            ):
                continue

            # Calculate when original lifetime ends
            original_lifetime_end = int(build_year) + int(original_lifetime)

            # TY5 evaluation: when lifetime ends within modeled period
            # but extension would go BEYOND last modeled TY
            # e.g., lifetime ends at 2038, last_ty=2040, extension to 2048
            if original_lifetime_end <= last_ty:
                # Already handled in main loop (within modeled period)
                continue

            # TY5 case: lifetime ends AFTER last modeled TY
            # We need to evaluate if extending beyond the last modeled TY is profitable
            # Virtual TY5 = original_lifetime_end (when lifetime actually ends)
            ty5 = original_lifetime_end

            # Get NPV at last modeled TY as proxy
            if component not in npv.index.get_level_values(0):
                continue
            npv_c = npv.loc[component]
            if unit not in npv_c.index:
                continue
            if last_ty_str not in npv_c.columns:
                continue

            npv_ty_last = npv_c.loc[unit, last_ty_str]
            current_cap = updated.loc[idx, last_ty]

            # Check if decision already made in a previous iteration
            ty5_prior = extension_decision_tracker.get(idx)
            if ty5_prior is not None:
                ty5_prior_decision, ty5_prior_lifetime, ty5_prior_npv_adjusted = (
                    ty5_prior
                )
                if ty5_prior_decision:
                    unit_states_records.append(
                        {
                            "component": component,
                            "unit": unit,
                            "ty": ty5,
                            "state": UnitState.EXTENDED.value,
                            "raw_state": UnitState.EXTENDED.value,
                            "p_nom": current_cap,
                            "npv_raw": npv_ty_last,
                            "npv_adjusted": ty5_prior_npv_adjusted,
                            "lifetime_extended": True,
                            "new_lifetime": ty5_prior_lifetime,
                        }
                    )
                else:
                    unit_states_records.append(
                        {
                            "component": component,
                            "unit": unit,
                            "ty": ty5,
                            "state": UnitState.EXTENSION_REJECTED.value,
                            "raw_state": UnitState.EXTENSION_REJECTED.value,
                            "p_nom": current_cap,
                            "npv_raw": npv_ty_last,
                            "npv_adjusted": ty5_prior_npv_adjusted,
                            "lifetime_extended": False,
                            "new_lifetime": 0.0,
                        }
                    )
                continue

            # Fresh TY5 evaluation (does NOT affect max_abs_npv — does not block convergence)
            should_extend, extended_lifetime, ty5_adjusted_npv = (
                evaluate_lifetime_extension(
                    target_year=ty5,
                    build_year=int(build_year),
                    original_lifetime=original_lifetime,
                    npv_ty=npv_ty_last,  # Use last TY's NPV
                    npv_ty_last=npv_ty_last,
                    lifetime_extension_years=ext_years,
                    lifetime_extension_capex=ext_capex,
                    lifetime_extension_fom=ext_fom,
                    capacity=current_cap,
                    threshold=threshold,
                    all_tys=sorted_tys,
                    extension_enabled=True,
                    hurdle_rate=ty5_hurdle_rate,
                )
            )

            if should_extend:
                unit_states_records.append(
                    {
                        "component": component,
                        "unit": unit,
                        "ty": ty5,
                        "state": UnitState.EXTENDED.value,
                        "raw_state": UnitState.EXTENDED.value,
                        "p_nom": current_cap,
                        "npv_raw": npv_ty_last,
                        "npv_adjusted": ty5_adjusted_npv,
                        "lifetime_extended": True,
                        "new_lifetime": extended_lifetime,
                    }
                )
                logging.debug(
                    f"TY5 lifetime extension granted for {component} {unit}: "
                    f"original_lifetime_end={original_lifetime_end}, new_lifetime={extended_lifetime}"
                )
            else:
                unit_states_records.append(
                    {
                        "component": component,
                        "unit": unit,
                        "ty": ty5,
                        "state": UnitState.EXTENSION_REJECTED.value,
                        "raw_state": UnitState.EXTENSION_REJECTED.value,
                        "p_nom": current_cap,
                        "npv_raw": npv_ty_last,
                        "npv_adjusted": ty5_adjusted_npv,
                        "lifetime_extended": False,
                        "new_lifetime": 0.0,
                    }
                )
                logging.debug(
                    f"TY5 lifetime extension rejected for {component} {unit}: "
                    f"original_lifetime_end={original_lifetime_end}"
                )

    # Post-processing: Finalize unit states after all TYs evaluated
    # Per ERAA methodology: you can only determine if a unit is "mothballed" (temporary)
    # vs "capacity_removed" (permanent) AFTER seeing the full trajectory.
    # If a unit is marked as MOTHBALLED but never comes back online (demothballing),
    # it should be reclassified as CAPACITY_REMOVED.
    if mothballing_enabled and unit_states_records:
        # Group records by (component, unit) to analyze trajectories
        unit_records: dict[tuple[str, str], list[dict]] = {}
        for record in unit_states_records:
            key = (record["component"], record["unit"])
            if key not in unit_records:
                unit_records[key] = []
            unit_records[key].append(record)

        for key, records in unit_records.items():
            # Sort records by TY
            records.sort(key=lambda r: r["ty"])

            # Check if unit ever demothballs (goes from MOTHBALLED back to active)
            # Demothballing is detected when:
            # 1. Unit was MOTHBALLED in a previous TY
            # 2. Unit has CAPACITY_ADDED state in a later TY (came back online)
            has_mothball = False
            has_demothball = False

            for i, record in enumerate(records):
                if record["state"] == UnitState.MOTHBALLED.value:
                    has_mothball = True
                elif has_mothball and record["state"] == UnitState.CAPACITY_ADDED.value:
                    # Unit came back online after being mothballed
                    has_demothball = True
                    break

            # If unit was mothballed but never demothballed, change MOTHBALLED to CAPACITY_REMOVED
            if has_mothball and not has_demothball:
                for record in records:
                    if record["state"] == UnitState.MOTHBALLED.value:
                        record["state"] = UnitState.CAPACITY_REMOVED.value
                        # Also update state_tracker for consistency
                        state_tracker[
                            (record["component"], record["unit"], record["ty"])
                        ] = UnitState.CAPACITY_REMOVED.value
                logging.debug(
                    f"Unit {key[0]} {key[1]} mothballed in all TYs without demothballing - "
                    f"reclassified as CAPACITY_REMOVED"
                )

    # Create unit states DataFrame
    unit_states = pd.DataFrame(unit_states_records)

    return updated, max_abs_npv, unit_states


def apply_capacities_to_networks(
    networks_by_ty: dict[int, list[pypsa.Network]],
    original_capacities: pd.DataFrame,
    updated_capacities: pd.DataFrame,
) -> None:
    """
    Write updated capacities back to networks and scale costs.

    Parameters
    ----------
    networks_by_ty : dict[int, list[pypsa.Network]]
        Dictionary mapping target years to lists of PyPSA networks.
        Networks are modified in place.
    original_capacities : pd.DataFrame
        Original p_nom values before update. MultiIndex (component, unit), columns TYs.
    updated_capacities : pd.DataFrame
        Updated p_nom values. Same structure as original_capacities.
    """
    for ty, networks in networks_by_ty.items():
        for n in networks:
            component_map = {
                "Generator": n.generators,
                "StorageUnit": n.storage_units,
            }

            for idx in updated_capacities.index:
                component, unit = idx
                if component not in component_map:
                    continue

                component_table = component_map[component]
                if unit not in component_table.index:
                    continue

                attr = "e" if component == "Store" else "p"
                original_cap = original_capacities.loc[idx, ty]
                updated_cap = updated_capacities.loc[idx, ty]

                # Update capacity
                component_table.loc[unit, f"{attr}_nom"] = updated_cap

                # Scale costs if capacity changed
                if updated_cap != original_cap:
                    scale_generator_costs(
                        component, unit, component_table, original_cap, updated_cap
                    )


def apply_lifetime_extensions_to_networks(
    networks_by_ty: dict[int, list[pypsa.Network]],
    unit_states: pd.DataFrame,
    eva_cost_params: dict[int, pd.DataFrame] | None,
) -> None:
    """
    Update lifetime attribute in networks for units granted lifetime extension.

    When a unit has lifetime_extended=True in unit_states, this function
    updates the lifetime column in the network's generators/storage_units
    DataFrame to reflect the extended lifetime.

    Parameters
    ----------
    networks_by_ty : dict[int, list[pypsa.Network]]
        Dictionary mapping target years to lists of PyPSA networks.
        Networks are modified in place.
    unit_states : pd.DataFrame
        DataFrame with unit states. Must have columns:
        component, unit, ty, lifetime_extended, new_lifetime.
        If new_lifetime is provided and > 0, it is used directly.
        Otherwise, new_lifetime is calculated from eva_cost_params.
    eva_cost_params : dict[int, pd.DataFrame] | None
        EVA cost parameters with lifetime extension data (fallback), keyed by TY.
        Each DataFrame has Index: MultiIndex (component, unit).
        Required columns: lifetime, lifetime_extension_years.
    """
    if unit_states.empty or eva_cost_params is None:
        return

    # Find units with lifetime extension granted
    extended_units = unit_states[unit_states["lifetime_extended"] == True]

    if extended_units.empty:
        return

    logging.info(
        f"Applying lifetime extensions to {len(extended_units)} unit-TY combinations"
    )

    for _, row in extended_units.iterrows():
        component = row["component"]
        unit = row["unit"]
        ty = row["ty"]
        idx = (component, unit)

        # Get eva_cost_params for this TY
        eva_cost_params_ty = eva_cost_params.get(ty)

        # Use new_lifetime from unit_states if available (Comment #16 optimization)
        # This avoids recalculating and ensures consistency with evaluate_lifetime_extension
        if "new_lifetime" in row and row["new_lifetime"] > 0:
            new_lifetime = row["new_lifetime"]
            # Get original_lifetime for logging (needed outside else branch)
            if eva_cost_params_ty is not None and idx in eva_cost_params_ty.index:
                original_lifetime = eva_cost_params_ty.loc[idx, "lifetime"]
            else:
                original_lifetime = "N/A"
        else:
            # Fallback: recalculate from eva_cost_params
            if eva_cost_params_ty is None or idx not in eva_cost_params_ty.index:
                continue

            original_lifetime = eva_cost_params_ty.loc[idx, "lifetime"]
            # Use lifetime_extension_years (not Lifetime_Extension which is CAPEX)
            extension_years = eva_cost_params_ty.loc[idx, "lifetime_extension_years"]

            if pd.isna(original_lifetime) or pd.isna(extension_years):
                continue

            new_lifetime = original_lifetime + extension_years

        # Update lifetime in all networks for all TYs
        for ty, networks in networks_by_ty.items():
            for n in networks:
                component_map = {
                    "Generator": n.generators,
                    "StorageUnit": n.storage_units,
                }

                if component not in component_map:
                    continue

                comp_table = component_map[component]
                if unit not in comp_table.index:
                    continue

                if "lifetime" in comp_table.columns:
                    comp_table.loc[unit, "lifetime"] = new_lifetime

        logging.debug(
            f"Updated lifetime for {component} {unit}: "
            f"{original_lifetime} -> {new_lifetime}"
        )


def compute_linear_decay_increment(
    base_increment: float,
    iteration: int,
    max_it: int,
    decay_iterations: int | None = None,
    min_step: float | None = None,
) -> int:
    """
    Compute the capacity update increment with linear decay applied.

    Linearly interpolates the increment from ``base_increment`` down to a
    minimum step size over a given number of iterations. After the decay period
    the increment is capped at the minimum step size.

    Parameters
    ----------
    base_increment : float
        Starting capacity change increment (MW).
    iteration : int
        Current EVA iteration index (0-based).
    max_it : int
        Maximum number of EVA iterations (used when ``decay_iterations`` is
        ``None``).
    decay_iterations : int or None
        Number of iterations over which to spread the decay. If ``None``,
        ``max_it`` is used.
    min_step : float or None
        Target minimum step size in MW. If not provided, no decay is applied.

    Returns
    -------
    int
        Rounded capacity increment for the current iteration.
    """

    # Determine number of decay iterations
    n_iters = decay_iterations if decay_iterations is not None else max_it
    # Determine minimum step
    min_step = min_step if min_step is not None else base_increment * 0.01

    # Compute decay target as fraction of base increment
    decay_target = min_step / base_increment

    # Linearly interpolate; clamp to decay_target once the period ends
    progress = min(iteration / n_iters, 1.0)
    decay_factor = 1.0 - (1.0 - decay_target) * progress

    return round(base_increment * decay_factor)


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="update_capacity",
    )
    configure_logging(snakemake)
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    # Retrieves the input network from snakemake
    npv = pd.read_csv(
        snakemake.input.npv,  # ty: ignore[possibly-missing-attribute]
        index_col=[0, 1],
        header=0,
    )
    npv.index.set_names(["component", "name"], inplace=True)

    # Handle case when no extendable assets exist (empty NPV file)
    if npv.empty:
        logging.info(
            "No extendable assets found in NPV file. Skipping capacity updates."
        )
        # Create empty capacity file
        empty_capacity = pd.DataFrame(
            columns=pd.Index(["component", "name", "p_nom", "iteration"])
        )
        empty_capacity.to_csv(
            snakemake.output.capacities,  # ty: ignore[possibly-missing-attribute]
            index=False,
        )

        # Create empty unit states file
        empty_unit_states = pd.DataFrame(
            columns=pd.Series(
                [
                    "component",
                    "unit",
                    "ty",
                    "state",
                    "p_nom",
                    "npv_raw",
                    "npv_adjusted",
                    "lifetime_extended",
                    "new_lifetime",
                ]
            )
        )
        empty_unit_states.to_csv(
            snakemake.output.unit_states,  # ty: ignore[possibly-missing-attribute]
            index=False,
        )

        # Write convergence status
        with open(
            snakemake.output.convergence_check,  # ty: ignore[possibly-missing-attribute]
            "w",
        ) as f:
            f.write("Converged")
        exit()

    comps = npv.index.get_level_values("component").unique()
    threshold = snakemake.config["eva"]["capacity_update"]["threshold"]
    decay_enabled = snakemake.config["eva"]["capacity_update"]["linear_decay"]
    decay_iterations = snakemake.config["eva"]["capacity_update"][
        "linear_decay_iterations"
    ]
    decay_min_step = snakemake.config["eva"]["capacity_update"]["linear_decay_min_step"]
    max_it = snakemake.config["eva"]["max_it"]
    max_abs_npv = 0
    capacities = {}

    # Load networks for capacity extraction — no output networks written
    # (updated capacities are saved to unit_states CSV; build_base_network applies them)
    perfect_foresight = snakemake.config["eva"]["perfect_foresight"]["enable"]

    if perfect_foresight:
        networks_in = list(
            snakemake.input.networks_actual  # ty: ignore[possibly-missing-attribute]
        )
    else:
        networks_in = list(
            snakemake.input.networks_forecast  # ty: ignore[possibly-missing-attribute]
        ) + list(
            snakemake.input.networks_actual  # ty: ignore[possibly-missing-attribute]
        )

    # Calculate increment with decay
    base_increment = snakemake.config["eva"]["capacity_update"]["increment"]
    iteration = int(
        snakemake.wildcards.it  # ty: ignore[possibly-missing-attribute]
    )
    if decay_enabled:
        increment = compute_linear_decay_increment(
            base_increment=base_increment,
            iteration=iteration,
            max_it=max_it,
            decay_iterations=decay_iterations,
            min_step=decay_min_step,
        )
    else:
        increment = base_increment

    # Step 1: Load all networks and organize by target year
    networks_by_ty: dict[int, list[pypsa.Network]] = {}
    for path_in in networks_in:
        n = pypsa.Network(path_in)
        ty = int(n.meta["wildcards"]["ty"])
        if ty not in networks_by_ty:
            networks_by_ty[ty] = []
        networks_by_ty[ty].append(n)

    # Step 2 : Identify if initial capacities are monotonic
    # If iteration = 0, identify if the initial
    if iteration == 0:
        # Update networks_by_ty to include column "ignore_monotonicity"
        # in case it is not respected in the starting networks for a certain TY
        networks_by_ty = verify_monotonicity(networks_by_ty)

    # Step 3: Extract capacities from networks
    (
        original_capacities,
        p_nom_max,
        p_nom_min,
        is_extendable,
        is_decommissionable,
        ignore_monotonicity,
    ) = extract_capacities_from_networks(networks_by_ty, list(comps))

    # Step 3.5: Extract EVA cost parameters (mothballing, lifetime extension)
    eva_cost_params = extract_eva_cost_params(networks_by_ty, list(comps))

    # Get mothballing and lifetime extension configuration
    # Get mothballing and lifetime extension flags (simplified config)
    mothballing_enabled = snakemake.config["eva"].get("enable_mothballing", False)
    lifetime_extension_enabled = snakemake.config["eva"].get(
        "enable_lifetime_extension", False
    )

    logging.info(f"Mothballing: enabled={mothballing_enabled}")
    logging.info(f"Lifetime extension: enabled={lifetime_extension_enabled}")

    # Step 3.6: Load previous iteration's unit states for state persistence
    previous_unit_states_path = getattr(snakemake.input, "previous_unit_states", None)
    # Handle case where previous_unit_states is a list (snakemake returns [] for it=0)
    if isinstance(previous_unit_states_path, list):
        previous_unit_states_path = (
            previous_unit_states_path[0] if previous_unit_states_path else None
        )
    previous_iteration_states = load_previous_unit_states(previous_unit_states_path)

    # Step 4: Update capacities based on NPV signals
    updated_capacities, max_abs_npv, unit_states = update_capacities(
        capacities=original_capacities,
        p_nom_max=p_nom_max,
        p_nom_min=p_nom_min,
        is_extendable=is_extendable,
        is_decommissionable=is_decommissionable,
        ignore_monotonicity=ignore_monotonicity,
        npv=npv,
        threshold=threshold,
        increment=increment,
        eva_cost_params=eva_cost_params,
        mothballing_enabled=mothballing_enabled,
        lifetime_extension_enabled=lifetime_extension_enabled,
        previous_iteration_states=previous_iteration_states,
    )

    has_converged = max_abs_npv <= threshold
    iteration = int(snakemake.wildcards.it)  # ty: ignore[possibly-missing-attribute]
    max_iterations = snakemake.config["eva"]["max_it"]

    clustering_enabled = snakemake.config["eva"]["advanced_clustering"]["enable"]
    freq = snakemake.config["eva"]["advanced_clustering"].get("full_run_frequency", 1)

    prev_status = ""
    prev_conv_input = getattr(snakemake.input, "prev_convergence_check", None)
    if isinstance(prev_conv_input, list):
        prev_conv_input = prev_conv_input[0] if prev_conv_input else None
    if prev_conv_input:
        prev_conv_path = Path(str(prev_conv_input))
        if prev_conv_path.exists():
            prev_status = prev_conv_path.read_text().strip()

    is_full = (
        not clustering_enabled
        or iteration == 0
        or iteration % freq == 0
        or iteration >= max_iterations - 1
        or prev_status == "ClusteredConverged"
    )
    logging.info(
        f"Iteration {iteration}: is_full={is_full}, clustering_enabled={clustering_enabled}, "
        f"prev_status='{prev_status}'"
    )

    is_final_iteration = has_converged or (iteration >= max_iterations - 1)

    if lifetime_extension_enabled:
        logging.info(
            f"Lifetime extensions tracked; is_final_iteration={is_final_iteration}. "
            "Extensions will be applied in build_base_network via unit_states CSV."
        )

    # Step 6: Collect installed capacities (networks are no longer written — updated
    # capacities are applied in the next iteration's build_base_network via unit_states CSV)
    sorted_tys = sorted(networks_by_ty.keys())
    for ty in sorted_tys:
        for n in networks_by_ty[ty]:
            if ty not in capacities:
                capacities[ty] = n.statistics.installed_capacity(
                    comps=list(comps), groupby=["carrier", "bus"], nice_names=False
                )

    capacity_df = pd.concat(capacities, axis=1)

    exclude_carriers = ["load shedding", "load spillage"]
    if "carrier" in capacity_df.index.names:
        capacity_df = capacity_df[
            ~capacity_df.index.get_level_values("carrier")
            .str.lower()
            .isin(exclude_carriers)
        ]
    capacity_df.to_csv(
        snakemake.output.capacities  # ty: ignore[possibly-missing-attribute]
    )

    # Step 7: Export unit states (always required output)
    # Build per-unit carrier/bus lookup from all networks across TYs
    # Note: Carrier/bus are static for existing units, but extendable units may only exist in later TYs
    unit_carrier_bus: dict[tuple[str, str], tuple[str, str]] = {}
    for _ty, nets in networks_by_ty.items():
        for net in nets:
            for comp_attr, comp_name in [
                ("generators", "Generator"),
                ("storage_units", "StorageUnit"),
            ]:
                table = getattr(net, comp_attr, None)
                if table is not None and not table.empty:
                    for uname, row in table.iterrows():
                        key = (comp_name, uname)
                        if key not in unit_carrier_bus:
                            unit_carrier_bus[key] = (
                                row.get("carrier", ""),
                                row.get("bus", ""),
                            )

    unit_states["carrier"] = unit_states.apply(
        lambda r: unit_carrier_bus.get((r["component"], r["unit"]), ("", ""))[0], axis=1
    )
    unit_states["bus"] = unit_states.apply(
        lambda r: unit_carrier_bus.get((r["component"], r["unit"]), ("", ""))[1], axis=1
    )

    unit_states.to_csv(
        snakemake.output.unit_states,  # ty: ignore[possibly-missing-attribute]
        index=False,
    )
    logging.info(
        f"Unit states exported to {snakemake.output.unit_states}"  # ty: ignore[possibly-missing-attribute]
    )

    # Note: has_converged, iteration, max_iterations already defined above
    # for lifetime extension application

    # Special case for max_it = 0: immediate convergence
    if max_iterations == 0:
        status = "No EVA"
        logging.info(f"Iteration {iteration}: max_it=0, forcing immediate stoppage.")
    elif has_converged:
        if is_full:
            status = "Converged"
            logging.info(
                f"Iteration {iteration}: Convergence reached on full run. "
                f"Max absolute NPV ({max_abs_npv:.2e}) <= threshold ({threshold:.2e})."
            )
        else:
            status = "ClusteredConverged"
            logging.info(
                f"Iteration {iteration}: Convergence reached on clustered run -- "
                f"flagging ClusteredConverged to trigger validation full run next. "
                f"Max absolute NPV ({max_abs_npv:.2e}) <= threshold ({threshold:.2e})."
            )
    elif iteration >= max_iterations - 1:
        status = "Aborted"
        logging.warning(
            f"Iteration {iteration}: Maximum iterations ({max_iterations}) reached without convergence. "
            f"Max absolute NPV: {max_abs_npv:.2e}, Threshold: {threshold:.2e}."
        )
    else:
        status = "Running"
        logging.info(
            f"Iteration {iteration}: Not converged yet. Max absolute NPV: {max_abs_npv:.2e}, Threshold: {threshold:.2e}."
        )

    # Write status to file (as before, but now also logged)
    with open(
        snakemake.output.convergence_check,  # ty: ignore[possibly-missing-attribute]
        "w",
    ) as f:
        f.write(status)
    logging.info(
        f"Status '{status}' written to {snakemake.output.convergence_check}"  # ty: ignore[possibly-missing-attribute]
    )
