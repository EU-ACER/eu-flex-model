# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
import logging
import multiprocessing
import sys
from collections.abc import Callable
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.calculate_model_outputs import (
    calculate_residual_load_df,
)
from euflex.eu_flex_model.scripts.helpers.date_helpers import drop_feb_29
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.pypsa_helpers import (
    as_dense,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
)
from euflex.eu_flex_model.scripts.paths import resolve_input_path

logger = logging.getLogger(__name__)

# Set multiprocessing start method to 'spawn' to avoid fork() deadlock warnings
# with multi-threaded libraries (NumPy, PyPSA, etc.)
# This must be called before creating any pools
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    # Already set, which is fine
    pass

# Constants
HOURS_PER_DAY = 24
SA_CALIBRATION_SAMPLES = 100
WARNING_NO_PLANNED_OUTAGE = (
    "No planned_outage_rate found in generators. Returning an empty dataframe."
)


def _get_generators_with_maintenance(
    n: pypsa.Network, filter_type: str = "all"
) -> pd.DataFrame:
    """
    Return generators with planned_outage_rate > 0.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object.
    filter_type : str, optional
        Filter type: "all" (default), "existing", or "new".
        - "all": Return all generators with planned_outage_rate > 0
        - "existing": Return only existing generators (is_existing=True)
        - "new": Return only new generators (is_existing=False)

    Returns
    -------
    pd.DataFrame
        Generators with planned_outage_rate > 0, filtered by is_existing if requested.
    """
    if "planned_outage_rate" not in n.generators.columns:
        logger.warning(WARNING_NO_PLANNED_OUTAGE)
        return pd.DataFrame()

    mask = n.generators["planned_outage_rate"].notna() & (
        n.generators["planned_outage_rate"] > 0
    )

    if filter_type in ("existing", "new") and "is_existing" in n.generators.columns:
        target_value = filter_type == "existing"
        mask = mask & (n.generators["is_existing"] == target_value)  # noqa: E712

    return n.generators[mask].copy()


def _place_maintenance_windows(
    rng: np.random.Generator,
    total_periods: int,
    window_sizes: list[int],
    time_range: pd.DatetimeIndex,
) -> list[int]:
    """
    Place non-overlapping maintenance windows over a planning horizon.

    The function creates a binary maintenance vector of length total_periods,
    where 1 indicates a period under maintenance and 0 indicates full
    availability. Windows are placed such that they do not overlap and are
    probabilistically biased toward summer months (April to September) based on
    time_range.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used for sampling window start indices.
    total_periods : int
        Total number of periods in the planning horizon.
    window_sizes : list[int]
        Lengths of maintenance windows to place, expressed in number of periods.
    time_range : pandas.DatetimeIndex
        DateTime index covering the same horizon as total_periods. Used to
        bias placement toward summer months.

    Returns
    -------
    list[int]
        List of length total_periods with 1 for periods in maintenance
        windows and 0 otherwise.

    Notes
    -----
    If a window size exceeds total_periods or a non-overlapping position
    cannot be found after a fixed number of attempts, that window is skipped and
    a warning is logged.
    """
    maintenance_flags = [0] * total_periods
    placed_windows: list[tuple[int, int]] = []

    for window_size in window_sizes:
        max_start = total_periods - window_size
        if max_start < 0:
            logger.warning(
                f"Window size {window_size} exceeds total periods {total_periods}"
            )
            continue

        start_idx = _find_non_overlapping_start(
            rng, max_start, window_size, placed_windows, time_range
        )
        if start_idx is not None:
            maintenance_flags[start_idx : start_idx + window_size] = [1] * window_size
            placed_windows.append((start_idx, start_idx + window_size))

    return maintenance_flags


def _find_non_overlapping_start(
    rng: np.random.Generator,
    max_start: int,
    window_size: int,
    existing_windows: list[tuple[int, int]],
    time_range: pd.DatetimeIndex,
    max_attempts: int = 100,
) -> int | None:
    """
    Sample a non-overlapping maintenance window start, biased toward summer.

    The function randomly samples candidate start indices for a window of
    length window_size within [0, max_start]. With 70% probability,
    sampling is restricted to indices whose timestamps fall in April--September
    (inclusive), provided such indices exist. Each candidate is checked
    against existing_windows to avoid overlap. After at most
    max_attempts trials, the function either returns a feasible start
    index or None.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used for sampling candidate start indices.
    max_start : int
        Maximum valid start index (inclusive) such that
        start + window_size <= total_periods.
    window_size : int
        Length of the window in number of periods.
    existing_windows : list of tuple of int
        List of already placed windows as (start, end) index pairs,
        where end is exclusive. The new window must not overlap any of
        these intervals.
    time_range : pandas.DatetimeIndex
        Datetime index covering at least indices [0, max_start]. Used to
        identify "summer" indices (months 4--9) for biased sampling.
    max_attempts : int, optional
        Maximum number of sampling attempts before giving up. Default is 100.

    Returns
    -------
    int or None
        Start index of a non-overlapping window if found, otherwise None.

    Notes
    -----
    A warning is logged if no non-overlapping position can be found after
    max_attempts trials.
    """
    summer_indices = [
        i for i, t in enumerate(time_range) if 4 <= t.month <= 9 and i <= max_start
    ]

    for _ in range(max_attempts):
        if rng.random() < 0.7 and summer_indices:
            start_idx = int(rng.choice(summer_indices))
        else:
            start_idx = int(rng.integers(0, max_start + 1))

        end_idx = start_idx + window_size
        if all(end_idx <= s or start_idx >= e for s, e in existing_windows):
            return start_idx

    logger.warning("Could not place window without overlap after 100 attempts")
    return None


def _apply_shift_move(
    rng: np.random.Generator,
    maintenance: np.ndarray,
    windows: list[tuple[int, int]],
    n_periods: int,
    max_shift: int = 30,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Shift a randomly selected maintenance window to a nearby position.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for reproducibility.
    maintenance : np.ndarray
        Binary array where 1 indicates maintenance, 0 indicates normal operation.
    windows : list[tuple[int, int]]
        List of (start, end) maintenance windows. Modified in-place.
    n_periods : int
        Total number of time periods.
    max_shift : int, optional
        Maximum shift distance in periods (default: 30 days).

    Returns
    -------
    tuple[np.ndarray, list[tuple[int, int]]]
        Updated maintenance array and windows list.
    """
    idx = rng.integers(0, len(windows))
    start, end = windows[idx]
    wsize = end - start

    # Shift by random amount within [-max_shift, +max_shift]
    shift_amount = int(rng.integers(-max_shift, max_shift + 1))
    new_start = start + shift_amount

    # Clamp to valid bounds
    new_start = max(0, min(new_start, n_periods - wsize))

    maintenance[start:end] = 0
    maintenance[new_start : new_start + wsize] = 1
    windows[idx] = (new_start, new_start + wsize)
    return maintenance, windows


def _apply_resize_move(
    rng: np.random.Generator,
    maintenance: np.ndarray,
    windows: list[tuple[int, int]],
    n_periods: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Resize a randomly selected maintenance window by ±1 period.

    Randomly selects a window and adjusts its duration by one period while
    keeping its start position fixed. The window size is clamped to valid
    bounds [1, n_periods].

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random number generator for reproducible randomness.
    maintenance : np.ndarray
        Binary array of shape (n_periods,) marking maintenance periods.
    windows : list[tuple[int, int]]
        List of (start, end) maintenance windows. Modified in-place.
    n_periods : int
        Total number of time periods.

    Returns
    -------
    tuple[np.ndarray, list[tuple[int, int]]]
        Updated maintenance array and windows list with resized window.

    Notes
    -----
    Window size is bounded to [1, n_periods] to ensure validity.
    """
    idx = rng.integers(0, len(windows))
    start, end = windows[idx]
    wsize = max(1, end - start + rng.choice([-1, 1]))
    wsize = min(wsize, n_periods)
    maintenance[start:end] = 0
    maintenance[start : start + wsize] = 1
    windows[idx] = (start, start + wsize)
    return maintenance, windows


def _apply_swap_move(
    rng: np.random.Generator,
    maintenance: np.ndarray,
    windows: list[tuple[int, int]],
    n_periods: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Swap two randomly selected maintenance windows.

    Exchanges the positions of two windows. If windows have equal length,
    they swap positions directly. Otherwise, each window is placed at the
    other's position if it fits; otherwise, it reverts to its original position.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random number generator for reproducible randomness.
    maintenance : np.ndarray
        Binary array of shape (n_periods,) marking maintenance periods.
    windows : list[tuple[int, int]]
        List of (start, end) maintenance windows. Modified in-place.
    n_periods : int
        Total number of time periods.

    Returns
    -------
    tuple[np.ndarray, list[tuple[int, int]]]
        Updated maintenance array and windows list with swapped windows.

    Notes
    -----
    For unequal-length windows, a swap is only completed if both windows
    fit at each other's positions. Failed placements revert to original positions.
    """
    idx1, idx2 = rng.choice(len(windows), size=2, replace=False)
    s1, e1 = windows[idx1]
    s2, e2 = windows[idx2]
    len1, len2 = e1 - s1, e2 - s2
    temp1, temp2 = maintenance[s1:e1].copy(), maintenance[s2:e2].copy()

    maintenance[s1:e1] = 0
    maintenance[s2:e2] = 0

    if len1 == len2:
        maintenance[s1:e1] = temp2
        maintenance[s2:e2] = temp1
        windows[idx1] = (s1, e1)
        windows[idx2] = (s2, e2)
    else:
        if s1 + len2 <= n_periods:
            maintenance[s1 : s1 + len2] = temp2
            windows[idx1] = (s1, s1 + len2)
        else:
            maintenance[s1:e1] = temp1
            windows[idx1] = (s1, e1)

        if s2 + len1 <= n_periods:
            maintenance[s2 : s2 + len1] = temp1
            windows[idx2] = (s2, s2 + len1)
        else:
            maintenance[s2:e2] = temp2
            windows[idx2] = (s2, e2)

    return maintenance, windows


def normalize_timestamps_to_year_agnostic(
    df: pd.DataFrame, target_year: int | None = None
) -> pd.DataFrame:
    """
    Normalize timestamps in DataFrame index to be year-agnostic.

    Replaces the year component with a target year (or removes it entirely),
    keeping only month-day-hour information. This makes maintenance schedules
    applicable across all climate years.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index.
    target_year : int, optional
        Target year to use for normalized timestamps. If None, uses 2000 as default.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized timestamps (year replaced with target_year).

    Notes
    -----
    Rows timestamped Feb 29 are dropped before the year is replaced, since
    ``target_year`` may not itself be a leap year (e.g. a leap climate year's
    snapshots being normalized onto a non-leap target year), which would
    otherwise raise ``ValueError: day is out of range for month``.
    """
    feb29 = (df.index.month == 2) & (df.index.day == 29)
    if feb29.any():
        logger.info(
            f"Dropping {int(feb29.sum())} Feb 29 row(s) while normalizing timestamps "
            f"to target year {target_year} (Feb 29 is always excluded)"
        )
        df = df.loc[~feb29]

    # Replace year in timestamps while keeping month-day-hour
    normalized_index = df.index.map(
        lambda dt: dt.replace(year=target_year) if not pd.isna(dt) else dt
    )

    df_normalized = df.copy()
    df_normalized.index = normalized_index

    logger.info(
        f"Normalized timestamps to target year {target_year} (year-agnostic schedule)"
    )

    return df_normalized


def is_full_year_network(n: pypsa.Network, tolerance_days: int = 7) -> bool:
    """
    Check if network snapshots span a full year.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object.
    tolerance_days : int, optional
        Tolerance in days for considering a year "full" (default: 7 days).

    Returns
    -------
    bool
        True if snapshots span approximately a full year, False otherwise.
    """
    if len(n.snapshots) == 0:
        return False

    time_span = n.snapshots[-1] - n.snapshots[0]
    days_span = time_span.days + time_span.seconds / 86400.0

    # Check if span is approximately 365 days (within tolerance)
    is_full = abs(days_span - 365.0) <= tolerance_days

    logger.info(f"Network spans {days_span:.1f} days (full year: {is_full})")
    return is_full


def apply_maintenance_schedules(
    n: pypsa.Network,
    maintenance_plan: pd.DataFrame,
) -> pypsa.Network:
    """
    Apply planned maintenance schedules to network generators.

    This function follows the same logic as apply_forced_outages from generate_forced_outages.py.
    It modifies p_max_pu and p_min_pu based on the unavailability fraction from planned maintenance.

    Note: A Generator represents an aggregated fleet of units, so partial maintenance
    (availability > 0) still leaves some units active and able to experience a forced
    outage independently in the same snapshot; only a fleet fully down for maintenance
    (availability == 0) has no units left that could also be forced out.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network with generators.
    maintenance_plan : pd.DataFrame
        Maintenance plan with Time as index and generators as columns.
        Values are unavailability fractions (0 = fully available, 1 = fully unavailable).
        The index should be year-agnostic (e.g., all with target year like 2030),
        and will be applied to any climate year network by matching month-day-hour.

    Returns
    -------
    pypsa.Network
        Modified network with updated p_max_pu and p_min_pu

    Notes
    -----
    - Modifies the network in-place by updating n.generators_t attributes
    - Uses get_switchable_as_dense to properly handle static and timeseries p_max_pu
    - Works on top of any existing p_max_pu modifications (e.g., from forced outages)
    - Handles year-agnostic maintenance plans by normalizing network snapshots to plan year
      before alignment, then applying to actual network snapshots
    """
    logger.info("Applying planned maintenance schedules to network generators")

    # Get generators that are in the maintenance plan
    planned_gens = maintenance_plan.columns.intersection(n.generators.index)

    all_gens = n.generators.index

    # Get base p_max_pu values (handles both static and timeseries)
    base_p_max_pu = as_dense(n, "Generator", "p_max_pu", all_gens)
    base_p_min_pu = as_dense(n, "Generator", "p_min_pu", all_gens)

    # Create availability matrix from maintenance plan (availability = 1 - unavailability)
    # For generators not in plan, assume fully available (availability = 1.0)
    availability = pd.DataFrame(1.0, index=n.snapshots, columns=all_gens)

    # Handle year-agnostic maintenance plans (same month-day-hour pattern across all climate years)
    # Convert network snapshots to same year as maintenance plan for alignment
    if isinstance(maintenance_plan.index, pd.DatetimeIndex):
        # Get the year from maintenance plan (typically target year like 2030)
        plan_year = maintenance_plan.index[0].year

        # Create aligned index by replacing year in network snapshots
        network_snapshots_normalized = n.snapshots.map(
            lambda dt: dt.replace(year=plan_year) if not pd.isna(dt) else dt
        )

        # Reindex maintenance plan using normalized network snapshots
        maintenance_aligned = maintenance_plan.reindex(
            index=network_snapshots_normalized, method="ffill"
        ).fillna(0.0)

        # Update availability for generators in the plan
        for gen in planned_gens:
            availability[gen] = 1.0 - maintenance_aligned[gen].values
    else:
        # Fallback: direct alignment (if not datetime index)
        maintenance_aligned = maintenance_plan.reindex(
            index=n.snapshots, method="ffill"
        ).fillna(0.0)

        for gen in planned_gens:
            availability[gen] = 1.0 - maintenance_aligned[gen]

    n.generators_t.availability = availability
    # Multiply base p_max_pu by availability to get final p_max_pu
    final_p_max_pu = base_p_max_pu * availability

    # Adjust p_min_pu: multiply by availability, then clip to final_p_max_pu
    # This ensures: p_min_pu = 0 during full maintenance, and p_min_pu <= p_max_pu always
    final_p_min_pu = (base_p_min_pu * availability).clip(upper=final_p_max_pu)

    # Update the network
    n.generators_t.p_max_pu = final_p_max_pu
    n.generators_t.p_min_pu = final_p_min_pu

    # Logging statistics
    n_maintenance_applied = (availability < 1.0).sum().sum()
    total_gen_snapshots = len(n.snapshots) * len(all_gens)
    maintenance_impact_pct = (
        (n_maintenance_applied / total_gen_snapshots) * 100
        if total_gen_snapshots > 0
        else 0
    )

    logger.info(
        f"Applied planned maintenance: {n_maintenance_applied} generator-snapshot combinations affected "
        f"({maintenance_impact_pct:.2f}% of total)"
    )

    return n


def compute_hourly_thermal_capacity(
    n: pypsa.Network,
    config: dict,
    maintenance_schedule: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute hourly available thermal committable capacity per bidding zone.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object with generator data.
    config : dict
        Configuration dictionary.
    maintenance_schedule : pd.DataFrame
        Unit-level maintenance schedule with columns ["Generator", "Unit", "Time", "Maintenance"],
        OR fleet-level schedule with ["Generator", "Time", "Unavailability_Fraction"].

    Returns
    -------
    pd.DataFrame
        Available thermal capacity with columns ["bus", "Time", "capacity_mw"].
        Capacity accounts for planned outages and maintenance schedules.
    """
    # Get generators with planned maintenance requirements
    # (filtering is done at data loading time to only include thermal/committable)
    generators_with_maintenance = _get_generators_with_maintenance(n)

    if generators_with_maintenance.empty:
        logger.warning(WARNING_NO_PLANNED_OUTAGE)
        return pd.DataFrame(columns=pd.Series(["bus", "Time", "capacity_mw"]))

    # Check if schedule is unit-level or fleet-level
    if "Unit" in maintenance_schedule.columns:
        # Aggregate to fleet level first
        fleet_schedule = aggregate_unit_schedule_to_fleet(
            maintenance_schedule, n, config
        )
    elif "Unavailability_Fraction" in maintenance_schedule.columns:
        # Already fleet-level
        fleet_schedule = maintenance_schedule
    else:
        raise ValueError(
            "maintenance_schedule must have either 'Unit' column (unit-level) "
            "or 'Unavailability_Fraction' column (fleet-level)"
        )

    # Optimized: Merge fleet schedule with generator data first, then compute capacity
    # This avoids creating a large MultiIndex product

    # Start with fleet schedule (already filtered to relevant generator-time pairs)
    df = fleet_schedule.merge(
        generators_with_maintenance[["bus", "p_nom"]].reset_index(),
        on="Generator",
        how="left",
    )

    # Fill missing unavailability fractions with 0
    df["Unavailability_Fraction"] = df["Unavailability_Fraction"].fillna(0)

    # Compute effective capacity (accounting for maintenance schedule)
    # Available capacity = p_nom * (1 - unavailability_fraction)
    df["capacity_mw"] = df["p_nom"] * (1 - df["Unavailability_Fraction"])

    # Aggregate by bus and time
    result = df.groupby(["bus", "Time"], as_index=False, observed=True)[
        "capacity_mw"
    ].sum()

    return result


def compute_capacity_margin(
    n: pypsa.Network,
    config: dict,
    maintenance_schedule: pd.DataFrame,
    renewable_carriers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute capacity margin per bidding zone and time.

    Capacity margin = available thermal capacity - residual load

    A positive margin indicates surplus capacity, while a negative margin
    indicates insufficient thermal capacity to meet residual demand.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object.
    config : dict
        Configuration dictionary.
    maintenance_schedule : pd.DataFrame
        Maintenance schedule with columns ["Generator", "Time", "Maintenance"].
    renewable_carriers : list[str] | None, optional
        List of renewable carrier names for residual load calculation.

    Returns
    -------
    pd.DataFrame
        Capacity margin with Time as index and bidding zones as columns [MW].
    """
    # Compute residual load
    if renewable_carriers is None:
        renewable_carriers = config["eva"]["base_network"]["renewable_carriers"]

    residual_load = calculate_residual_load_df(n, renewable_carriers)

    # Compute thermal capacity
    thermal_capacity = compute_hourly_thermal_capacity(n, config, maintenance_schedule)

    # Pivot thermal capacity to have time as index and buses as columns
    thermal_capacity_pivot = thermal_capacity.pivot(
        index="Time", columns="bus", values="capacity_mw"
    ).fillna(0)

    # Align indices and columns
    common_times = residual_load.index.intersection(thermal_capacity_pivot.index)
    common_buses = residual_load.columns.intersection(thermal_capacity_pivot.columns)

    residual_load_aligned = residual_load.loc[common_times, common_buses]
    thermal_capacity_aligned = thermal_capacity_pivot.loc[common_times, common_buses]

    # Calculate capacity margin
    capacity_margin = thermal_capacity_aligned - residual_load_aligned

    return capacity_margin


def generate_maintenance_schedule_for_new_units(
    n: pypsa.Network | list[pypsa.Network],
    config: dict,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Generate maintenance schedule for new units based on residual load.

    For new units (is_existing=False), maintenance is scheduled based on residual
    load - higher derating during periods of low residual load (high renewable
    generation). Each hour is weighted by how far its residual load sits below the
    bus's yearly maximum, so the weighting stays bounded even when residual load
    approaches or crosses zero (routine at high-RES buses). The total maintenance
    applied over the year equals the planned_outage_annual_rate.

    Parameters
    ----------
    n : pypsa.Network | list[pypsa.Network]
        PyPSA network object(s). If list, residual load is averaged across networks.
    config : dict
        Configuration dictionary with renewable carriers.
    start_time : pd.Timestamp | None, optional
        Start of the planning period. If None, uses first snapshot.
    end_time : pd.Timestamp | None, optional
        End of the planning period. If None, uses last snapshot.

    Returns
    -------
    pd.DataFrame
        Maintenance schedule with Time as index and generators as columns.
        Values are unavailability fractions (0 = fully available, 1 = fully unavailable).
    """
    # Handle single network or list
    if isinstance(n, list):
        networks = n
        n_ref = networks[0]
    else:
        networks = [n]
        n_ref = n

    if start_time is None:
        start_time = n_ref.snapshots[0]
    if end_time is None:
        end_time = n_ref.snapshots[-1]

    # Get new generators (is_existing=False) with maintenance requirements
    if "is_existing" not in n_ref.generators.columns:
        logger.warning(
            "is_existing column not found - cannot identify new units. Returning empty schedule."
        )
        return pd.DataFrame()

    new_generators = _get_generators_with_maintenance(n_ref, filter_type="new")

    if new_generators.empty:
        logger.info("No new generators with maintenance requirements found")
        return pd.DataFrame()

    logger.info(
        f"Generating maintenance schedules for {len(new_generators)} new generators based on inverse residual load"
    )

    # Calculate average residual load across all networks/climate years
    renewable_carriers = config["eva"]["base_network"]["renewable_carriers"]
    residual_loads = []

    # Get reference year from the reference network
    ref_year = n_ref.snapshots[0].year

    for net in networks:
        if isinstance(net, pypsa.Network):
            rl = calculate_residual_load_df(net, renewable_carriers)
            # Normalize timestamps to reference year for alignment across climate years
            rl.index = rl.index.map(lambda x: x.replace(year=ref_year))
            residual_loads.append(rl)

    # Average residual load across climate years
    # After normalization, all timestamps align, so groupby reduces to single year
    avg_residual_load = pd.concat(residual_loads).groupby(level=0).mean()

    # Create maintenance schedule for each new generator
    maintenance_schedule = pd.DataFrame(index=n_ref.snapshots)

    for gen_name, gen_data in new_generators.iterrows():
        bus = gen_data["bus"]
        planned_outage_rate = gen_data["planned_outage_rate"]

        if bus not in avg_residual_load.columns:
            logger.warning(
                f"Bus {bus} not found in residual load data for generator {gen_name}. Skipping."
            )
            continue

        # Get residual load for this bus
        bus_residual_load = avg_residual_load[bus]

        # Weight each hour by how far its residual load sits below the bus's yearly
        # maximum (higher weight during low residual load / high renewable-surplus
        # periods). Unlike a literal reciprocal (1/residual_load), this stays bounded
        # and never blows up as residual load approaches or crosses zero -- which
        # happens routinely at high-RES buses and previously caused the weighting to
        # collapse (either to an all-zero schedule, or to a single dominant hour).
        residual_load_weight = bus_residual_load.max() - bus_residual_load

        # Convert planned_outage_rate from days to hours
        planned_outage_hours = planned_outage_rate * HOURS_PER_DAY

        # Normalize weights to distribute planned outage hours
        # Each hour gets an unavailability fraction (0 to 1) proportional to its weight
        # Sum of all fractions across all hours = planned_outage_hours
        sum_weight = residual_load_weight.sum()
        if sum_weight > 0:
            # Distribute planned_outage_hours proportionally based on residual load weight
            # unavailability_fraction[hour] = fraction of capacity unavailable at that hour
            unavailability_fraction = (
                residual_load_weight / sum_weight
            ) * planned_outage_hours

            # Clip to [0, 1] to ensure valid unavailability fractions
            unavailability_fraction = unavailability_fraction.clip(0.0, 1.0)

            # Renormalize to maintain the total planned_outage_hours
            # This redistributes the "clipped" excess back to all hours proportionally
            current_sum = unavailability_fraction.sum()
            if current_sum > 0 and abs(current_sum - planned_outage_hours) > 0.1:
                # Scale to match planned total
                unavailability_fraction = (
                    unavailability_fraction / current_sum
                ) * planned_outage_hours
                # Clip again in case renormalization pushed some values > 1.0
                unavailability_fraction = unavailability_fraction.clip(0.0, 1.0)
        else:
            # Residual load is flat all year (no low/high-load hours to prefer) --
            # spread the planned outage evenly rather than silently dropping it.
            uniform_fraction = min(1.0, planned_outage_hours / len(n_ref.snapshots))
            unavailability_fraction = pd.Series(uniform_fraction, index=n_ref.snapshots)

        # Verify that total unavailability equals planned_outage_rate (approximately)
        # Sum of fractions = total hours of unavailability
        total_unavailability_hours = unavailability_fraction.sum()
        total_unavailability_days = total_unavailability_hours / HOURS_PER_DAY

        logger.info(
            f"Generator {gen_name}: planned={planned_outage_rate:.2f} days, "
            f"actual={total_unavailability_days:.2f} days, "
            f"error={(total_unavailability_days - planned_outage_rate):.2f} days "
            f"({((total_unavailability_days - planned_outage_rate) / planned_outage_rate * 100):.1f}%)"
        )

        # Add to schedule
        maintenance_schedule[gen_name] = unavailability_fraction.values

    logger.info(
        f"Generated maintenance schedules for {len(maintenance_schedule.columns)} new generators"
    )

    return maintenance_schedule


def generate_initial_maintenance_schedule(
    n: pypsa.Network,
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
    resolution: str = "D",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate an initial maintenance schedule for individual thermal generator units.

    Each generator fleet is modeled as n_units independent units (from network data).
    Maintenance is scheduled for individual units, then aggregated to get fleet-level unavailability.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object with generator data (must have n_units column).
    start_time : pd.Timestamp | None, optional
        Start of the planning period. If None, uses first snapshot.
    end_time : pd.Timestamp | None, optional
        End of the planning period. If None, uses last snapshot.
    resolution : str, optional
        Time resolution for maintenance scheduling ('D' for daily, 'H' for hourly).
        Default is 'D' (daily).
    seed : int, optional
        Random seed for reproducibility of maintenance schedule generation.

    Returns
    -------
    pd.DataFrame
        Maintenance schedule with columns ["Generator", "Unit", "Time", "Maintenance"]
        where Maintenance is 1 if unit is under maintenance, else 0.
    """

    rng = np.random.default_rng(seed)

    if start_time is None:
        start_time = n.snapshots[0]
    if end_time is None:
        end_time = n.snapshots[-1]

    # Create time range at specified resolution
    time_range = pd.date_range(start=start_time, end=end_time, freq=resolution)

    # Filter for EXISTING generators that have planned maintenance requirements
    # New generators (is_existing=False) are handled separately via inverse residual load
    generators_with_maintenance = _get_generators_with_maintenance(
        n, filter_type="existing"
    )

    if generators_with_maintenance.empty:
        logger.warning(WARNING_NO_PLANNED_OUTAGE)
        return pd.DataFrame(
            columns=pd.Series(["Generator", "Unit", "Time", "Maintenance"])
        )

    # Note: planned_outage_min_hours and planned_outage_annual_windows should already
    # be set in the network (from load_sql_data or applied in main script before optimization)

    # Determine number of units per generator (from network data)
    if "n_units" in generators_with_maintenance.columns:
        generators_with_maintenance["n_units_effective"] = (
            generators_with_maintenance["n_units"].fillna(1).astype(int)
        )
    else:
        logger.warning(
            "n_units column not found in network, defaulting to 1 unit per generator"
        )
        generators_with_maintenance["n_units_effective"] = 1

    maintenance_records = []

    for gen_name, gen_data in generators_with_maintenance.iterrows():
        maintenance_rate = gen_data["planned_outage_rate"]
        num_units = int(gen_data["n_units_effective"])
        min_hours = gen_data.get("planned_outage_min_hours", HOURS_PER_DAY)
        num_windows = int(gen_data.get("planned_outage_annual_windows", 1))

        # Convert minimum hours to days (round up)
        if resolution == "D":
            min_duration = int(np.ceil(min_hours / HOURS_PER_DAY))
        else:
            min_duration = int(min_hours)

        total_periods = len(time_range)
        # maintenance_rate is in days per year, scale to actual time range
        # For daily resolution: maintenance_rate days over 365 days
        # For hourly resolution: maintenance_rate days over 365 days (same formula)
        maintenance_periods = int(maintenance_rate * total_periods / 365.0)

        # For each unit in the fleet
        for unit_idx in range(num_units):
            # Allocate maintenance into num_windows blocks
            if maintenance_periods > 0 and maintenance_periods <= total_periods:
                # Ensure each window meets minimum duration
                if maintenance_periods < num_windows * min_duration:
                    # Not enough maintenance periods to satisfy constraints
                    logger.warning(
                        f"{gen_name} unit {unit_idx}: maintenance_periods={maintenance_periods} < "
                        f"num_windows={num_windows} * min_duration={min_duration}. "
                        f"Reducing windows or adjusting duration."
                    )
                    # Use maximum possible windows given the constraint
                    num_windows = max(1, maintenance_periods // min_duration)

                # Distribute maintenance periods across windows
                periods_per_window = maintenance_periods // num_windows
                remaining_periods = maintenance_periods % num_windows

                # Find all window sizes
                window_sizes = [
                    max(
                        periods_per_window + (1 if i < remaining_periods else 0),
                        min_duration,
                    )
                    for i in range(num_windows)
                ]

                maintenance_flags = _place_maintenance_windows(
                    rng, total_periods, window_sizes, time_range
                )

                # Create records (vectorized - avoid loop + append)
                unit_records = pd.DataFrame(
                    {
                        "Generator": gen_name,
                        "Unit": unit_idx,
                        "Time": time_range,
                        "Maintenance": maintenance_flags,
                    }
                )
                maintenance_records.append(unit_records)
            else:
                # If no maintenance needed, all periods are available (vectorized)
                unit_records = pd.DataFrame(
                    {
                        "Generator": gen_name,
                        "Unit": unit_idx,
                        "Time": time_range,
                        "Maintenance": 0,
                    }
                )
                maintenance_records.append(unit_records)

    # Concatenate all unit records into single DataFrame
    if maintenance_records:
        return pd.concat(maintenance_records, ignore_index=True)
    else:
        return pd.DataFrame(
            columns=pd.Index(["Generator", "Unit", "Time", "Maintenance"])
        )


def aggregate_unit_schedule_to_fleet(
    schedule: pd.DataFrame, n: pypsa.Network, config: dict
) -> pd.DataFrame:
    """
    Aggregate unit-level maintenance schedule to fleet-level unavailability fraction.

    Parameters
    ----------
    schedule : pd.DataFrame
        Unit-level maintenance schedule with columns ["Generator", "Unit", "Time", "Maintenance"].
    n : pypsa.Network
        PyPSA network object to get generator fleet sizes.
    config : dict
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Fleet-level schedule with columns ["Generator", "Time", "Unavailability_Fraction"]
        where Unavailability_Fraction is the fraction of fleet capacity unavailable (0 to 1).
    """
    # Optimized version: avoid creating full MultiIndex product
    # Instead, directly compute from maintenance schedule

    # Get total units per generator
    total_units_per_gen = schedule.groupby("Generator")["Unit"].max() + 1

    # Count units under maintenance per generator per time
    # Only process records where Maintenance == 1 for efficiency
    maintenance_count = (
        schedule[schedule["Maintenance"] == 1]
        .groupby(["Generator", "Time"], observed=True)
        .size()
    )

    # Convert to DataFrame and add total units
    result = maintenance_count.reset_index(name="units_in_maintenance")
    result = result.merge(
        total_units_per_gen.rename("total_units").reset_index(),
        on="Generator",
        how="left",
    )

    # Calculate unavailability fraction
    result["Unavailability_Fraction"] = (
        result["units_in_maintenance"] / result["total_units"]
    )

    # For times/generators with no maintenance, we need to fill in zeros
    # Get all unique generator-time combinations from original schedule
    all_gen_times = schedule[["Generator", "Time"]].drop_duplicates()

    # Merge back to ensure all combinations are present
    result = all_gen_times.merge(
        result[["Generator", "Time", "Unavailability_Fraction"]],
        on=["Generator", "Time"],
        how="left",
    )
    result["Unavailability_Fraction"] = result["Unavailability_Fraction"].fillna(0)

    return result[["Generator", "Time", "Unavailability_Fraction"]]


def expand_schedule_to_hourly(
    schedule: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Expand a daily unit-level maintenance schedule to hourly resolution.

    Each day's maintenance flag is applied to all hours in that day.

    Parameters
    ----------
    schedule : pd.DataFrame
        Daily maintenance schedule with columns ["Generator", "Unit", "Time", "Maintenance"].
    start_time : pd.Timestamp
        Start time for hourly expansion.
    end_time : pd.Timestamp
        End time for hourly expansion.

    Returns
    -------
    pd.DataFrame
        Hourly maintenance schedule with columns ["Generator", "Unit", "Time", "Maintenance"].
    """
    # start_time/end_time are a network's own (leap-day-free) first/last
    # snapshot, but date_range between them regenerates Feb 29 for a leap
    # climate year regardless, so it must be dropped again here to keep the
    # leap-day-free invariant that network snapshots are built with.
    hourly_range = drop_feb_29(pd.date_range(start=start_time, end=end_time, freq="h"))
    schedule = schedule.copy()
    schedule["Time"] = pd.to_datetime(schedule["Time"])

    logger.info(f"Expanding schedule with shape: {schedule.shape}")

    # Check if schedule has Unit column (unit-level) or not (fleet-level)
    if "Unit" not in schedule.columns:
        # Fleet-level schedule - add dummy Unit column
        logger.warning(
            "Schedule does not have 'Unit' column, treating as fleet-level with single unit per generator"
        )
        schedule["Unit"] = 0

    # VECTORIZED APPROACH: Use pandas merge instead of nested loops
    # Create hourly time index with date mapping
    hourly_df = pd.DataFrame({"Time": hourly_range})
    hourly_df["Date"] = hourly_df["Time"].dt.date

    # Add date column to daily schedule for merging
    schedule["Date"] = schedule["Time"].dt.date

    # Get unique generator-unit combinations
    gen_units = schedule[["Generator", "Unit"]].drop_duplicates()

    # Cross join: all generator-units with all hourly times
    # This is much faster than nested Python loops
    hourly_expanded = gen_units.merge(hourly_df, how="cross")

    # Merge with daily schedule on generator, unit, and date
    result = hourly_expanded.merge(
        schedule[["Generator", "Unit", "Date", "Maintenance"]],
        on=["Generator", "Unit", "Date"],
        how="left",
    )

    # Fill missing maintenance flags with 0
    result["Maintenance"] = result["Maintenance"].fillna(0).astype(int)

    # Drop the temporary Date column and reorder
    result = result[["Generator", "Unit", "Time", "Maintenance"]].sort_values(
        ["Generator", "Unit", "Time"]
    )

    logger.info(f"Expanded to hourly schedule with shape: {result.shape}")

    return result


def compute_objective(
    n: pypsa.Network | list[pypsa.Network],
    config: dict,
    maintenance_schedule: pd.DataFrame,
    penalty_weight: float = 1000.0,
    use_hourly: bool = True,
    normalize: bool = True,
    _residual_load_cache: dict | None = None,
    buses_to_include: list[str] | None = None,
) -> float:
    """
    Compute objective function for maintenance optimization.

    The objective minimizes capacity margin variability while penalizing
    deviations from planned outage rates at the unit level.

    Parameters
    ----------
    n : pypsa.Network | list[pypsa.Network]
        PyPSA network object(s). If a list is provided, capacity margins are computed
        for each network and averaged (for robustness across climate years).
    config : dict
        Configuration dictionary.
    maintenance_schedule : pd.DataFrame
        Unit-level maintenance schedule (daily or hourly resolution).
    penalty_weight : float, optional
        Weight for the penalty term that enforces planned outage constraints.
    use_hourly : bool, optional
        If True, expand schedule to hourly before computing capacity margin.
    normalize : bool, optional
        If True, normalize both objective terms to similar scales (recommended).
        This prevents one term from dominating due to different units/magnitudes.
    _residual_load_cache : dict | None, optional
        Internal cache for residual load calculations (computed once, reused).
    buses_to_include : list[str] | None, optional
        If provided, only compute capacity margin objective for these buses.
        Used for per-bus optimization to avoid penalizing other buses.

    Returns
    -------
    float
        Objective value (lower is better).
    """
    # Handle single network or list of networks
    if isinstance(n, list):
        networks = n
        n_ref = networks[0]  # Use first network for reference
    else:
        networks = [n]
        n_ref = n

    # Expand to hourly if needed
    if use_hourly and not maintenance_schedule.empty:
        hourly_schedule = expand_schedule_to_hourly(
            maintenance_schedule,
            start_time=n_ref.snapshots[0],
            end_time=n_ref.snapshots[-1],
        )
    else:
        hourly_schedule = maintenance_schedule

    # Compute capacity margin for each network and average
    # Use cached residual load if available (residual load doesn't change with maintenance schedule)
    capacity_margins = []
    for idx, network in enumerate(networks):
        # Check cache first
        if _residual_load_cache is not None and idx in _residual_load_cache:
            residual_load = _residual_load_cache[idx]
        else:
            # Compute residual load (expensive, but only once per network)
            renewable_carriers = config["eva"]["base_network"]["renewable_carriers"]
            residual_load = calculate_residual_load_df(network, renewable_carriers)
            # A daily schedule only has 00:00 timestamps, so compare it against
            # each day's peak residual load rather than the midnight hour alone
            if not use_hourly:
                residual_load = residual_load.resample("D").max()
            if _residual_load_cache is not None:
                _residual_load_cache[idx] = residual_load

        # Compute thermal capacity (this changes with maintenance schedule)
        thermal_capacity = compute_hourly_thermal_capacity(
            network,
            config,
            hourly_schedule,
        )

        # Pivot and compute margin
        thermal_capacity_pivot = thermal_capacity.pivot(
            index="Time", columns="bus", values="capacity_mw"
        ).fillna(0)

        # Align and compute margin
        common_times = residual_load.index.intersection(thermal_capacity_pivot.index)
        common_buses = residual_load.columns.intersection(
            thermal_capacity_pivot.columns
        )

        residual_load_aligned = residual_load.loc[common_times, common_buses]
        thermal_capacity_aligned = thermal_capacity_pivot.loc[
            common_times, common_buses
        ]

        capacity_margin = thermal_capacity_aligned - residual_load_aligned
        capacity_margins.append(capacity_margin)

    # Average capacity margin across all networks (climate years)
    capacity_margin = pd.concat(capacity_margins).groupby(level=0).mean()

    # Ensure index is DatetimeIndex after groupby (can become plain Index during pickling)
    if not isinstance(capacity_margin.index, pd.DatetimeIndex):
        capacity_margin.index = pd.DatetimeIndex(capacity_margin.index)

    # Filter to only relevant buses (for per-bus optimization)
    if buses_to_include is not None:
        # Only compute objective for buses being optimized
        available_buses = [b for b in buses_to_include if b in capacity_margin.columns]
        if available_buses:
            capacity_margin = capacity_margin[available_buses]
        else:
            logger.warning(
                f"No buses from buses_to_include={buses_to_include} found in capacity_margin. "
                f"Available buses: {list(capacity_margin.columns)}"
            )

    # Aggregate to daily resolution if maintenance schedule is daily (use_hourly=False)
    if not use_hourly:
        # Resample to daily by taking the mean of each day
        # This ensures the objective matches the daily decision resolution
        capacity_margin = capacity_margin.resample("D").mean()

    # Capacity margin objective: minimize sum of squared capacity margins
    # This penalizes extreme tail ends much more heavily than variance/std
    # Goal: push down high capacity margin values aggressively
    if normalize:
        # Normalize by number of time periods and buses for fair comparison with penalty term
        capacity_margin_penalty = (capacity_margin**2).sum().sum() / (
            len(capacity_margin) * len(capacity_margin.columns)
        )
    else:
        # Sum of squared capacity margins across all time periods and buses
        capacity_margin_penalty = (capacity_margin**2).sum().sum()

    # Penalty for violating planned outage rates (checked at unit level)
    generators_with_maintenance = _get_generators_with_maintenance(n_ref)

    if generators_with_maintenance.empty:
        outage_penalty = 0.0
        num_units = 0
    else:
        total_hours = len(
            pd.date_range(start=n_ref.snapshots[0], end=n_ref.snapshots[-1], freq="h")
        )

        periods_per_day = HOURS_PER_DAY if use_hourly else 1

        # VECTORIZED APPROACH: Calculate penalties for all units at once
        # Group by generator and unit to get actual maintenance periods
        actual_periods_per_unit = (
            hourly_schedule.groupby(["Generator", "Unit"], observed=True)["Maintenance"]
            .sum()
            .reset_index(name="actual_periods")
        )

        # Filter to only generators with maintenance requirements
        actual_periods_per_unit = actual_periods_per_unit[
            actual_periods_per_unit["Generator"].isin(generators_with_maintenance.index)
        ]

        # Merge with planned outage rates
        actual_periods_per_unit = actual_periods_per_unit.merge(
            generators_with_maintenance[["planned_outage_rate"]].reset_index(),
            on="Generator",
            how="left",
        )

        # Calculate planned periods for each unit (vectorized)
        actual_periods_per_unit["planned_periods"] = (
            actual_periods_per_unit["planned_outage_rate"]
            * periods_per_day
            * (total_hours / (365.0 * HOURS_PER_DAY))
        )

        # Calculate absolute deviations (vectorized)
        actual_periods_per_unit["deviation"] = abs(
            actual_periods_per_unit["actual_periods"]
            - actual_periods_per_unit["planned_periods"]
        )

        # Sum up penalties
        outage_penalty = actual_periods_per_unit["deviation"].sum()
        num_units = len(actual_periods_per_unit)

    # Normalize outage penalty by number of units for fair comparison
    if normalize and num_units > 0:
        outage_penalty = outage_penalty / num_units

    # Log decomposition for debugging
    if normalize:
        logger.debug(
            f"Objective breakdown: capacity_margin={capacity_margin_penalty:.2e}, "
            f"outage_penalty={outage_penalty:.2e} (weight={penalty_weight}), "
            f"total={capacity_margin_penalty + penalty_weight * outage_penalty:.2e}"
        )

    total_objective = capacity_margin_penalty + penalty_weight * outage_penalty

    return total_objective


def generate_neighbor_schedule(
    schedule: pd.DataFrame,
    n,
    num_units_to_modify: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a neighbor solution by modifying maintenance schedules.

    Implements shift, resize, and swap moves on existing maintenance windows
    without regenerating the entire schedule from scratch.

    Parameters
    ----------
    schedule : pd.DataFrame
        Unit-level maintenance schedule with columns ["Generator","Unit","Time","Maintenance"].
    n : pypsa.Network
        PyPSA network object to get generator constraints.
    num_units_to_modify : int
        Number of subunits to modify per neighbor generation.
    seed : int, optional
        Random seed for reproducibility of maintenance schedule generation.

    Returns
    -------
    pd.DataFrame
        New maintenance schedule.
    """
    rng = np.random.default_rng(seed)
    new_schedule = schedule.copy()

    generators_with_maintenance = _get_generators_with_maintenance(n)
    if generators_with_maintenance.empty:
        return new_schedule

    # All unique subunits
    units = [
        tuple(x)
        for x in new_schedule[["Generator", "Unit"]].drop_duplicates().to_numpy()
    ]
    if not units:
        return new_schedule

    # Randomly select subunits by index for robustness
    num_to_choose = min(num_units_to_modify, len(units))
    chosen_indices = rng.choice(len(units), size=num_to_choose, replace=False)
    chosen_units = [units[i] for i in chosen_indices]

    for generator, unit in chosen_units:
        if generator not in generators_with_maintenance.index:
            continue

        # Get current subunit schedule
        mask = (new_schedule["Generator"] == generator) & (new_schedule["Unit"] == unit)
        unit_schedule = new_schedule.loc[mask].sort_values("Time").copy()
        n_periods = len(unit_schedule)
        if n_periods == 0:
            continue

        # Get current maintenance array (don't regenerate!)
        new_maintenance = unit_schedule["Maintenance"].values.copy().astype(int)
        original_total = new_maintenance.sum()  # Preserve total maintenance days

        # Find existing windows in current schedule
        windows: list[tuple[int, int]] = []
        in_window = False
        start_idx = 0
        for i, val in enumerate(new_maintenance):
            if val == 1 and not in_window:
                start_idx = i
                in_window = True
            elif val == 0 and in_window:
                windows.append((start_idx, i))
                in_window = False
        # Handle case where maintenance window extends to end
        if in_window:
            windows.append((start_idx, len(new_maintenance)))

        # Only apply move operations if windows exist
        if windows:
            # Choose appropriate move type based on number of windows
            # Only use shift/swap moves to preserve total maintenance days
            # Resize changes the total, which violates the constraint
            if len(windows) == 1:
                # Only shift possible with 1 window (resize would change total)
                move = "shift"
            else:
                # Shift or swap (both preserve total maintenance days)
                move = rng.choice(["shift", "swap"])

            if move == "shift":
                new_maintenance, windows = _apply_shift_move(
                    rng, new_maintenance, windows, n_periods
                )
            elif move == "swap":
                new_maintenance, windows = _apply_swap_move(
                    rng, new_maintenance, windows, n_periods
                )

        # Verify that total maintenance days didn't change
        assert new_maintenance.sum() == original_total, (
            f"Maintenance total changed: {original_total} -> {new_maintenance.sum()}"
        )

        # Update schedule with modified maintenance
        new_schedule.loc[mask, "Maintenance"] = new_maintenance

    return new_schedule


def calibrate_initial_temperature(
    objective: Callable[[pd.DataFrame], float],
    schedule: pd.DataFrame,
    n: pypsa.Network,
    num_units_per_move: int,
    initial_acceptance: float,
    seed: int,
) -> float:
    """
    Calibrate the SA start temperature from sampled neighbour moves.

    The temperature is set such that the median worsening move of
    ``SA_CALIBRATION_SAMPLES`` neighbours of ``schedule`` is accepted with
    probability ``initial_acceptance``: ``T0 = -median(delta > 0) / ln(p0)``.

    Parameters
    ----------
    objective : Callable[[pd.DataFrame], float]
        Objective function evaluated on a maintenance schedule.
    schedule : pd.DataFrame
        Schedule around which neighbour moves are sampled.
    n : pypsa.Network
        Reference network for neighbour generation.
    num_units_per_move : int
        Number of units modified per neighbour move.
    initial_acceptance : float
        Target acceptance probability of the median worsening move, in (0, 1).
    seed : int
        Base seed; sample ``i`` uses ``seed + i``.

    Returns
    -------
    float
        Start temperature. 0.0 (greedy search) if no sampled move worsens the objective.
    """
    if not (0.0 < initial_acceptance < 1.0):
        raise ValueError(
            f"initial_acceptance must be in (0, 1), got {initial_acceptance}"
        )
    base = objective(schedule)
    deltas = np.array(
        [
            objective(
                generate_neighbor_schedule(
                    schedule, n, num_units_per_move, seed=seed + i
                )
            )
            - base
            for i in range(SA_CALIBRATION_SAMPLES)
        ]
    )
    worse = deltas[deltas > 0]
    if worse.size == 0:
        logger.warning(
            f"No worsening move among {SA_CALIBRATION_SAMPLES} sampled neighbours, "
            "running greedy search (T0=0)"
        )
        return 0.0
    return float(-np.median(worse) / np.log(initial_acceptance))


def _sa_worker(args):
    """Worker function for parallel SA runs. Returns (schedule, objective, run_id, history, T0)."""
    (
        networks,
        config,
        initial_acceptance,
        alpha,
        max_iterations,
        penalty_weight,
        max_block_length,
        num_units_per_move,
        run_id,
        seed,
        print_every,
        buses_to_include,
    ) = args

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Use first network as reference for schedule generation
    n_ref = networks[0] if isinstance(networks, list) else networks

    # Generate initial schedule
    initial_schedule = generate_initial_maintenance_schedule(
        n_ref,
        resolution="D",
        start_time=n_ref.snapshots[0],
        end_time=n_ref.snapshots[-1],
        seed=seed,
    )

    # Create residual load cache (computed once, reused for all iterations)
    residual_load_cache: dict = {}
    objective = partial(
        compute_objective,
        networks,
        config,
        penalty_weight=penalty_weight,
        use_hourly=False,
        normalize=True,
        _residual_load_cache=residual_load_cache,
        buses_to_include=buses_to_include,
    )

    # Run SA (pass all networks for objective computation)
    current_schedule = initial_schedule.copy()
    current_obj = objective(current_schedule)

    best_schedule = current_schedule.copy()
    best_obj = current_obj
    initial_obj = current_obj
    T = calibrate_initial_temperature(
        objective,
        current_schedule,
        n_ref,
        num_units_per_move,
        initial_acceptance,
        seed + max_iterations,
    )
    initial_temp = T

    # Track progress
    progress_log = []

    for i in range(max_iterations):
        # Generate neighbor with iteration-based seed for variety
        new_schedule = generate_neighbor_schedule(
            current_schedule, n_ref, num_units_per_move, seed=seed + i
        )
        new_obj = objective(new_schedule)

        delta = new_obj - current_obj
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_schedule = new_schedule.copy()
            current_obj = new_obj

            if current_obj < best_obj:
                best_schedule = current_schedule.copy()
                best_obj = current_obj

        T *= alpha

        # Log progress periodically
        if print_every > 0 and i % print_every == 0:
            improvement = (
                (initial_obj - best_obj) / initial_obj * 100 if initial_obj > 0 else 0
            )
            progress_log.append(
                {
                    "run_id": run_id,
                    "iteration": i,
                    "temperature": T,
                    "current_obj": current_obj,
                    "best_obj": best_obj,
                    "improvement": improvement,
                }
            )
            # Print to stdout so it appears in logs even with multiprocessing
            print(  # noqa: T201
                f"Chain {run_id} | Iteration {i}: T={T:.2e}, Current={current_obj:.2e}, Best={best_obj:.2e}, Improvement={improvement:.2f}%",
                flush=True,
            )

    return best_schedule, best_obj, run_id, progress_log, initial_temp


def _init_pool_worker() -> None:
    """
    Register PyPSA component properties in a spawned worker process.

    PyPSA attaches ``generators``, ``generators_t``, ... to the ``Network`` class
    when the first ``Network`` is instantiated. Unpickling skips ``__init__``, so
    networks passed to a fresh worker lack these properties until then.
    """
    pypsa.Network()


def _copy_network_data(net: pypsa.Network) -> pypsa.Network:
    """
    Create a lightweight copy of network for multiprocessing.

    PyPSA Network objects don't pickle reliably, so we copy only the
    essential DataFrames needed for maintenance optimization.
    """
    net_copy = pypsa.Network()
    net_copy.snapshots = net.snapshots.copy()  # ty: ignore[invalid-assignment]
    net_copy.generators = net.generators.copy()
    net_copy.buses = net.buses.copy()
    net_copy.loads = net.loads.copy()

    # Copy time-varying data
    if hasattr(net, "generators_t") and hasattr(net.generators_t, "p_max_pu"):
        net_copy.generators_t.p_max_pu = net.generators_t.p_max_pu.copy()
    if hasattr(net, "generators_t") and hasattr(net.generators_t, "availability"):
        net_copy.generators_t.availability = net.generators_t.availability.copy()
    if hasattr(net, "loads_t") and hasattr(net.loads_t, "p"):
        net_copy.loads_t.p = net.loads_t.p.copy()
    # Residual load is built from p_set, so the copy needs it to see demand
    net_copy.loads_t.p_set = net.loads_t.p_set.copy()

    return net_copy


def _optimize_single_bus(args):
    """Worker function for parallel bus optimization."""
    (
        bus,
        networks,
        config,
        generators_with_maintenance,
        kwargs,
    ) = args

    # Make copies of networks to avoid pickling issues
    # PyPSA Networks don't pickle reliably, so we copy the essential data
    networks = [_copy_network_data(net) for net in networks]

    # Get generators on this bus
    gens_on_bus = generators_with_maintenance[
        generators_with_maintenance["bus"] == bus
    ].index

    logger.info(f"Bus {bus}: Optimizing {len(gens_on_bus)} generators independently")

    # Create filtered network(s) with only generators on this bus
    filtered_networks = []
    for net in networks:
        # Temporarily zero out planned outage rate for other generators
        if "planned_outage_rate" in net.generators.columns:
            net.generators["planned_outage_rate_original"] = net.generators[
                "planned_outage_rate"
            ].copy()
            net.generators.loc[
                ~net.generators.index.isin(gens_on_bus), "planned_outage_rate"
            ] = 0.0
        else:
            logger.warning(
                f"Bus {bus}: planned_outage_rate column missing from generators"
            )
        filtered_networks.append(net)

    # Optimize for this bus
    # Add bus filter to kwargs to only optimize for this specific bus
    kwargs_with_bus = kwargs.copy()
    kwargs_with_bus["buses_to_include"] = [bus]

    schedule_bus, objective_bus, history_bus = optimize_maintenance_schedule(
        filtered_networks if len(filtered_networks) > 1 else filtered_networks[0],
        config,
        **kwargs_with_bus,
    )

    # Calculate improvement (history[0] is initial objective)
    if history_bus:
        initial_obj = history_bus[0]
        improvement_pct = (
            (initial_obj - objective_bus) / initial_obj * 100
            if initial_obj > 0
            else 0.0
        )
        logger.info(
            f"Bus {bus}: Initial={initial_obj:.2e}, Final={objective_bus:.2e}, "
            f"Improvement={improvement_pct:.2f}%"
        )
    else:
        logger.info(f"Bus {bus}: Final Objective = {objective_bus:.2e}")

    return bus, schedule_bus, objective_bus, history_bus


def optimize_maintenance_schedule_per_bus(
    n: pypsa.Network | list[pypsa.Network],
    config: dict,
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, tuple[float, list[float]]]]:
    """
    Optimize maintenance schedules independently for each bus.

    This runs separate SA optimizations for each bus. If n_parallel_runs is specified:
    - Multiple buses: Parallelizes across buses (one bus per worker). In this case, it is not possible to launch multiple parallel runs per bus, you can only launch up to N indidependent runs.
    - Single bus: Uses parallel SA runs (embarrassingly parallel SA). The embarassingly parallel SA is only available for single bus mode, where several parallel SA workers are launched to look to improve solution quality.

    Parameters
    ----------
    n : pypsa.Network | list[pypsa.Network]
        Network(s) to optimize.
    config : dict
        Configuration dictionary.
    **kwargs
        Parameters passed to optimize_maintenance_schedule.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - Combined schedule for all buses
        - Dictionary mapping bus -> (objective, history)
    """
    # Handle single network or list
    if isinstance(n, list):
        networks = n
        n_ref = networks[0]
    else:
        networks = [n]
        n_ref = n

    # Only optimize EXISTING generators (new generators handled separately)
    generators_with_maintenance = _get_generators_with_maintenance(
        n_ref, filter_type="existing"
    )

    if generators_with_maintenance.empty:
        logger.info(
            "No existing generators with maintenance requirements found - skipping optimization"
        )
        return pd.DataFrame(), {}

    # Group generators by bus
    buses = generators_with_maintenance["bus"].unique()
    n_buses = len(buses)
    logger.info(
        f"Optimizing maintenance for {n_buses} buses with existing generators independently"
    )

    # Extract n_parallel_runs from
    n_parallel_runs = kwargs.get("n_parallel_runs", 1)

    all_schedules = []
    bus_results = {}

    # Strategy: Use parallelization at the bus level if we have multiple buses
    # Otherwise, fall back to parallel SA runs within a single bus
    if n_buses > 1 and n_parallel_runs > 1:
        # Multiple buses: parallelize across buses (embarrassingly parallel)
        # Force single SA run per bus (parallelism is at bus level)
        logger.info(
            f"Parallelizing across {n_buses} buses with {min(n_parallel_runs, n_buses)} workers "
            f"(forcing n_parallel_runs=1 per bus)"
        )
        kwargs_single_run = kwargs.copy()
        kwargs_single_run["n_parallel_runs"] = 1

        # Prepare arguments for parallel bus optimization
        # Networks will be pickled by Python's multiprocessing (works reliably with maxtasksperchild=1)
        bus_args = [
            (bus, networks, config, generators_with_maintenance, kwargs_single_run)
            for bus in buses
        ]

        # Run buses in parallel with fresh workers to avoid state corruption
        # maxtasksperchild=1 ensures each task gets a clean worker process
        with Pool(
            processes=min(n_parallel_runs, n_buses, cpu_count()),
            initializer=_init_pool_worker,
            maxtasksperchild=1,
        ) as pool:
            results = list(pool.imap(_optimize_single_bus, bus_args, chunksize=1))

        # Collect results
        for bus, schedule_bus, objective_bus, history_bus in results:
            if not schedule_bus.empty:
                all_schedules.append(schedule_bus)
                bus_results[bus] = (objective_bus, history_bus)
            else:
                logger.warning(f"Bus {bus}: No schedule generated")

    else:
        # Single bus OR no parallelization: use sequential bus optimization
        # This will allow parallel SA runs within each bus if n_parallel_runs > 1
        if n_buses == 1 and n_parallel_runs > 1:
            logger.info(
                f"Single bus optimization with {n_parallel_runs} parallel SA chains"
            )
        else:
            logger.info(
                f"Sequential bus optimization (n_parallel_runs={n_parallel_runs})"
            )

        for bus in buses:
            bus_name, schedule_bus, objective_bus, history_bus = _optimize_single_bus(
                (bus, networks, config, generators_with_maintenance, kwargs)
            )

            if not schedule_bus.empty:
                all_schedules.append(schedule_bus)
                bus_results[bus_name] = (objective_bus, history_bus)
            else:
                logger.warning(f"Bus {bus_name}: No schedule generated")

    # Combine all schedules
    if all_schedules:
        combined_schedule = pd.concat(all_schedules, ignore_index=True)
        logger.info(
            f"Combined schedule: {len(combined_schedule)} records across {n_buses} buses"
        )
    else:
        combined_schedule = pd.DataFrame()
        logger.warning("No schedules generated for any bus")

    return combined_schedule, bus_results


def optimize_maintenance_schedule(
    n: pypsa.Network | list[pypsa.Network],
    config: dict,
    initial_schedule: pd.DataFrame | None = None,
    initial_acceptance: float = 0.8,
    alpha: float = 0.99,
    max_iterations: int = 10000,
    penalty_weight: float = 1000.0,
    max_block_length: int = 5,
    num_units_per_move: int = 3,
    print_every: int = 500,
    n_parallel_runs: int = 1,
    random_seed: int = 42,
    buses_to_include: list[str] | None = None,
) -> tuple[pd.DataFrame, float, list[float]]:
    """
    Optimize unit-level maintenance schedule using simulated annealing.

    The optimization minimizes capacity margin variability while respecting
    planned outage rate constraints at the unit level.

    Parameters
    ----------
    n : pypsa.Network | list[pypsa.Network]
        PyPSA network object(s) with n_units data. If a list is provided,
        capacity margins are averaged across all networks (climate years).
    config : dict
        Configuration dictionary.
    initial_schedule : pd.DataFrame | None, optional
        Initial unit-level maintenance schedule (daily resolution). If None, generates random schedule.
    initial_acceptance : float, optional
        Probability of accepting the median worsening move at the start. The
        start temperature is calibrated per run from sampled neighbour moves.
    alpha : float, optional
        Cooling rate (temperature multiplier per iteration).
    max_iterations : int, optional
        Maximum number of iterations.
    penalty_weight : float, optional
        Weight for planned outage constraint penalty.
    max_block_length : int, optional
        Maximum consecutive periods to modify in neighbor generation.
    num_units_per_move : int, optional
        Number of units to modify in each neighbor move.
        Higher values = more exploration but potentially lower acceptance rate.
    print_every : int, optional
        Print progress every N iterations. If 0, no progress is printed.
    n_parallel_runs : int, optional
        Number of parallel SA runs to execute. Best result is returned.
        Set to 1 for sequential, > 1 for parallel runs.
    random_seed : int, optional
        Random seed for reproducibility of maintenance schedule generation.
    buses_to_include : list[str] | None, optional
        If provided, only optimize capacity margin objective for these buses.
        Used for per-bus optimization.

    Returns
    -------
    tuple[pd.DataFrame, float, list[float]]
        - Best unit-level maintenance schedule found (daily resolution)
        - Best objective value
        - List of objective values over iterations (only from best run)
    """
    # Handle single network or list of networks
    if isinstance(n, list):
        networks = n
        n_ref = networks[0]
    else:
        networks = [n]
        n_ref = n

    if n_parallel_runs > 1:
        logger.info(
            f"Running {n_parallel_runs} parallel SA chains with {max_iterations} iterations each"
        )
        logger.info(
            f"SA parameters: initial_acceptance={initial_acceptance}, alpha={alpha}, penalty_weight={penalty_weight}"
        )

        # Prepare arguments for each worker
        seeds = [random_seed + i for i in range(n_parallel_runs)]
        worker_args = [
            (
                networks,
                config,
                initial_acceptance,
                alpha,
                max_iterations,
                penalty_weight,
                max_block_length,
                num_units_per_move,
                i,
                seeds[i],
                print_every,
                buses_to_include,
            )
            for i in range(n_parallel_runs)
        ]

        # Run in parallel
        logger.info(f"Starting {min(n_parallel_runs, cpu_count())} worker processes...")
        with Pool(
            processes=min(n_parallel_runs, cpu_count()), initializer=_init_pool_worker
        ) as pool:
            results = pool.map(_sa_worker, worker_args)

        # Unpack results
        all_schedules = [r[0] for r in results]
        all_objectives = [r[1] for r in results]
        all_run_ids = [r[2] for r in results]
        all_progress = [r[3] for r in results]
        all_initial_temps = [r[4] for r in results]

        # Log progress from each chain
        logger.info("\n" + "=" * 80)
        logger.info("PARALLEL SA CHAINS SUMMARY")
        logger.info("=" * 80)
        for run_id in range(n_parallel_runs):
            logger.info(f"Chain {run_id}: T0={all_initial_temps[run_id]:.2e}")
            run_progress = all_progress[run_id]
            if run_progress:
                initial = (
                    run_progress[0]["best_obj"]
                    if run_progress
                    else all_objectives[run_id]
                )
                final = (
                    run_progress[-1]["best_obj"]
                    if run_progress
                    else all_objectives[run_id]
                )
                improvement = (initial - final) / initial * 100 if initial > 0 else 0
                logger.info(
                    f"Chain {run_id}: Initial={initial:.2e}, Final={final:.2e}, "
                    f"Improvement={improvement:.2f}%"
                )
        logger.info("=" * 80 + "\n")

        # Find best result
        best_idx = all_objectives.index(min(all_objectives))
        best_schedule = all_schedules[best_idx]
        best_obj = all_objectives[best_idx]
        best_run = all_run_ids[best_idx]

        logger.info(f"Best result from chain {best_run} with objective {best_obj:.2e}")

        return best_schedule, best_obj, []

    # Single run (original implementation)
    # Generate initial schedule if not provided
    rng = np.random.default_rng(random_seed)
    if initial_schedule is None:
        logger.info(
            "Generating initial unit-level maintenance schedule from network data"
        )
        initial_schedule = generate_initial_maintenance_schedule(
            n_ref,
            resolution="D",
            start_time=n_ref.snapshots[0],
            end_time=n_ref.snapshots[-1],
            seed=random_seed,
        )

    # Log schedule statistics
    total_units = initial_schedule[["Generator", "Unit"]].drop_duplicates().shape[0]
    total_periods = initial_schedule["Time"].nunique()
    logger.info(
        f"Schedule dimensions: {total_units} units × {total_periods} time periods = {len(initial_schedule)} records"
    )
    if len(networks) > 1:
        logger.info(
            f"Optimizing across {len(networks)} climate year networks (averaging capacity margins)"
        )

    # Create residual load cache (computed once, reused for all iterations)
    # This dramatically speeds up objective computation by avoiding redundant calculations
    residual_load_cache: dict = {}
    # Use daily resolution for faster optimization
    # IMPORTANT: Use normalize=True to ensure objective terms are comparable
    objective = partial(
        compute_objective,
        networks,
        config,
        penalty_weight=penalty_weight,
        use_hourly=False,
        normalize=True,
        _residual_load_cache=residual_load_cache,
        buses_to_include=buses_to_include,
    )

    current_schedule = initial_schedule.copy()
    logger.info(
        "Computing initial objective (this may take a while for large problems)..."
    )
    current_obj = objective(current_schedule)

    best_schedule = current_schedule.copy()
    best_obj = current_obj
    initial_obj = current_obj

    T = calibrate_initial_temperature(
        objective,
        current_schedule,
        n_ref,
        num_units_per_move,
        initial_acceptance,
        random_seed + max_iterations,
    )
    objective_history = [current_obj]

    logger.info(
        f"Starting simulated annealing optimization: T={T:.2e}, "
        f"alpha={alpha}, max_iter={max_iterations}, units_per_move={num_units_per_move}"
    )
    if buses_to_include:
        logger.info(f"Optimizing for buses: {buses_to_include}")
    logger.info(f"Initial objective: {current_obj:.2e}")
    logger.info(
        f"Residual load cache populated with {len(residual_load_cache)} network(s) "
        f"- will reuse for all {max_iterations} iterations"
    )

    # TODO: remove this extra debug logging once maintenance plannig successfully executed in TEST
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    _ = objective(current_schedule)
    logger.setLevel(old_level)

    for i in range(max_iterations):
        # Generate neighbor with iteration-based seed for variety
        new_schedule = generate_neighbor_schedule(
            current_schedule, n_ref, num_units_per_move, seed=random_seed + i
        )
        new_obj = objective(new_schedule)

        # Accept or reject
        delta = new_obj - current_obj
        accepted = False
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_schedule = new_schedule.copy()
            current_obj = new_obj
            objective_history.append(current_obj)
            accepted = True

            # Update best solution
            if current_obj < best_obj:
                best_schedule = current_schedule.copy()
                best_obj = current_obj

        # Cool down
        T *= alpha

        # Enhanced progress logging - every 5 iterations for close monitoring
        if i % 500 == 0:
            improvement_pct = (
                (initial_obj - best_obj) / initial_obj * 100 if initial_obj > 0 else 0
            )
            acceptance_status = "✓ ACCEPTED" if accepted else "✗ rejected"
            logger.info(
                f"Iter {i:5d}/{max_iterations}: T={T:.2e}, Current={current_obj:.2e}, "
                f"Best={best_obj:.2e}, Δ={delta:+.2e}, Improvement={improvement_pct:6.2f}% [{acceptance_status}]"
            )
        # Also keep the original print_every logging
        elif print_every > 0 and i % print_every == 0:
            improvement_pct = (
                (initial_obj - best_obj) / initial_obj * 100 if initial_obj > 0 else 0
            )
            logger.info(
                f"Iteration {i}/{max_iterations}: T={T:.2e}, Current={current_obj:.2e}, "
                f"Best={best_obj:.2e}, Improvement={improvement_pct:.2f}%"
            )

    logger.info(f"Optimization complete. Final best objective: {best_obj:.2f}")
    if initial_obj > 0:
        improvement_pct = (initial_obj - best_obj) / initial_obj * 100
        logger.info(f"Improvement: {improvement_pct:.1f}%")

    return best_schedule, best_obj, objective_history


def load_maintenance_plan_from_csv(
    csv_path: str, target_year: int, reference_network: pypsa.Network
) -> pd.DataFrame:
    """
    Read a pre-computed maintenance plan from a CSV file and validate it.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file. Expected format: wide-format with a datetime index
        and generator names as columns; values are unavailability fractions (0–1).
    target_year : int
        Year to normalise timestamps to (year-agnostic schedule).
    reference_network : pypsa.Network
        Network used to validate that CSV columns match known generator names
        and that the schedule covers a full year of timestamps.

    Returns
    -------
    pd.DataFrame
        Maintenance schedule with timestamps normalised to ``target_year``.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If the CSV fails structural validation: non-datetime index, insufficient
        time coverage, or out-of-range values. Generator columns not present in
        the reference network's fleet are dropped with a warning rather than
        raising, since the fleet in ``reference_network`` can legitimately
        diverge from the one the plan was generated against (e.g. a unit
        retired/renamed, or a different DB snapshot) even for the same target
        year.
    """
    schedule = pd.read_csv(
        csv_path, index_col=0, parse_dates=True, date_format="ISO8601"
    )

    # 1. Index must be a DatetimeIndex
    if not isinstance(schedule.index, pd.DatetimeIndex):
        raise ValueError(
            f"Maintenance plan CSV '{csv_path}' has a non-datetime index. "
            "The first column must contain parseable timestamps."
        )

    # 2. Must cover a full year (≥98 % of 8760 rows)
    min_rows = int(8760 * 0.98)
    if len(schedule) < min_rows:
        raise ValueError(
            f"Maintenance plan CSV '{csv_path}' has only {len(schedule)} rows; "
            f"expected at least {min_rows} (a full year at hourly resolution)."
        )

    # 3. Drop columns for generators not present in this network's fleet. Even
    # though reference_network is built for the same target_year as csv_path,
    # the fleets can still diverge (a unit retired/renamed, or a different DB
    # snapshot between plan generation and this run), so this is expected
    # rather than a sign the wrong file was passed.
    expected_generators = set(_get_generators_with_maintenance(reference_network).index)
    csv_generators = set(schedule.columns)
    unknown = csv_generators - expected_generators
    if unknown:
        logger.warning(
            f"Maintenance plan CSV '{csv_path}' contains {len(unknown)} generator "
            f"column(s) not found in this network's fleet, dropping them: "
            f"{sorted(unknown)}."
        )
        schedule = schedule.drop(columns=sorted(unknown))

    # 4. Values must be in [0, 1]
    if (schedule < 0).any().any() or (schedule > 1).any().any():
        raise ValueError(
            f"Maintenance plan CSV '{csv_path}' contains values outside [0, 1]. "
            "Unavailability fractions must be between 0 and 1."
        )

    return normalize_timestamps_to_year_agnostic(schedule, target_year=target_year)


if __name__ == "__main__":
    snakemake = get_snakemake(globals_scope=globals(), rule="plan_maintenance")

    configure_logging(snakemake)

    logger.info("=" * 80)
    logger.info("Starting maintenance planning optimization")

    output_path = (
        snakemake.output.maintenance_plans  # ty: ignore[possibly-missing-attribute]
    )
    target_year = int(
        snakemake.wildcards.ty  # ty: ignore[possibly-missing-attribute]
    )

    # Check if maintenance planning is enabled
    maintenance_enabled = snakemake.config["solving"]["maintenance_planning"]["enable"]

    input_network = snakemake.input.network  # ty: ignore[possibly-missing-attribute]

    # Load first network to get structure
    n_first = pypsa.Network(input_network[0])

    if not maintenance_enabled:
        logger.info("Maintenance planning is DISABLED in config")
        logger.info(
            "Returning zero maintenance schedule (all generators fully available)"
        )

        # Get generators with planned maintenance requirements
        generators_with_maintenance = _get_generators_with_maintenance(n_first)

        # Create zero maintenance schedule (all generators fully available)
        if not generators_with_maintenance.empty:
            zero_schedule = pd.DataFrame(
                0.0, index=n_first.snapshots, columns=generators_with_maintenance.index
            )
        else:
            # No generators with maintenance requirements, create empty schedule
            zero_schedule = pd.DataFrame(index=n_first.snapshots)

        # Normalize timestamps to target year (year-agnostic)
        zero_schedule = normalize_timestamps_to_year_agnostic(
            zero_schedule, target_year=target_year
        )

        # Save to CSV
        logger.info(f"Saving zero maintenance plan to {output_path}")
        zero_schedule.to_csv(output_path)
        logger.info(
            "Zero maintenance plan saved successfully (maintenance planning disabled)"
        )
        logger.info("=" * 80)
        sys.exit(0)

    # Load the first network to check if it's a full year
    logger.info(f"Loading first network from {input_network[0]}")
    logger.info(
        f"Network loaded: {len(n_first.generators)} generators, {len(n_first.snapshots)} snapshots"
    )

    # Check if network spans a full year
    if not is_full_year_network(n_first):
        logger.warning(
            "Network snapshots do not span a full year. "
            "Skipping maintenance planning and returning zero maintenance schedule."
        )

        # Get generators with planned maintenance requirements
        generators_with_maintenance = _get_generators_with_maintenance(n_first)

        # Create zero maintenance schedule (all generators fully available)
        if not generators_with_maintenance.empty:
            zero_schedule = pd.DataFrame(
                0.0, index=n_first.snapshots, columns=generators_with_maintenance.index
            )
        else:
            # No generators with maintenance requirements, create empty schedule
            zero_schedule = pd.DataFrame(index=n_first.snapshots)

        # Normalize timestamps to target year (year-agnostic)
        zero_schedule = normalize_timestamps_to_year_agnostic(
            zero_schedule, target_year=target_year
        )

        # Save to CSV
        logger.info(f"Saving zero maintenance plan to {output_path}")
        zero_schedule.to_csv(output_path)
        logger.info(
            "Zero maintenance plan saved successfully (network too short for optimization)"
        )
        logger.info("=" * 80)
        sys.exit(0)

    # Check if maintenance plan should be read from CSV instead of optimized
    maint_plan_source = snakemake.config["solving"]["maintenance_planning"]["source"]

    if maint_plan_source == "plan-id":
        csv_path = resolve_input_path(
            snakemake.config["solving"]["maintenance_planning"]["csv_path"].replace(
                "{ty}", str(target_year)
            )
        )
        logger.info(f"Maintenance plan source is 'plan-id' — reading from: {csv_path}")
        wide_schedule = load_maintenance_plan_from_csv(
            csv_path, target_year=target_year, reference_network=n_first
        )
        logger.info(f"Saving maintenance plan to {output_path}")
        wide_schedule.to_csv(output_path)
        logger.info(
            f"Schedule shape: {wide_schedule.shape[0]} time periods x "
            f"{wide_schedule.shape[1]} generators"
        )
        logger.info("=" * 80)
        sys.exit(0)

    # Load ALL networks (all climate years) to compute average capacity margin
    logger.info(f"Loading {len(input_network)} networks for all climate years")
    networks = []
    for network_path in input_network:
        logger.info(f"Loading network from {network_path}")
        n_climate = pypsa.Network(network_path)
        networks.append(n_climate)

    logger.info(f"Loaded {len(networks)} networks successfully")

    # Apply config defaults to all networks for maintenance planning parameters
    # This ensures all networks have the required columns before optimization
    thermal_carriers = snakemake.config["eva"]["base_network"]["conventional_carriers"]
    min_hours_default = snakemake.config["solving"]["maintenance_planning"][
        "min_duration_hours"
    ]
    annual_windows_default = snakemake.config["solving"]["maintenance_planning"][
        "annual_windows"
    ]

    # TODO: remove this once the pre-builts are updated
    for net in networks:
        generators_with_maintenance = _get_generators_with_maintenance(net)

        if generators_with_maintenance.empty:
            continue

        # Apply defaults for planned outage window parameters if missing
        if "planned_outage_min_hours" not in net.generators.columns:
            net.generators["planned_outage_min_hours"] = min_hours_default
        else:
            # Fill NaN values with default for generators with maintenance
            net.generators.loc[
                generators_with_maintenance.index, "planned_outage_min_hours"
            ] = net.generators.loc[
                generators_with_maintenance.index, "planned_outage_min_hours"
            ].fillna(min_hours_default)

        if "planned_outage_annual_windows" not in net.generators.columns:
            net.generators["planned_outage_annual_windows"] = annual_windows_default
        else:
            # Fill NaN values with default for generators with maintenance
            net.generators.loc[
                generators_with_maintenance.index, "planned_outage_annual_windows"
            ] = net.generators.loc[
                generators_with_maintenance.index, "planned_outage_annual_windows"
            ].fillna(annual_windows_default)

    logger.info(
        f"Applied default maintenance planning parameters to all {len(networks)} networks"
    )

    # Use the first network as reference for optimization (same generators, snapshots structure across all climate years)
    n = networks[0]

    logger.info("Using simulated annealing optimization")

    # Simulated annealing parameters from config
    initial_acceptance = snakemake.config["solving"]["maintenance_planning"][
        "sa_initial_acceptance"
    ]
    alpha = snakemake.config["solving"]["maintenance_planning"]["sa_alpha"]
    max_iterations = snakemake.config["solving"]["maintenance_planning"][
        "sa_max_iterations"
    ]
    penalty_weight = snakemake.config["solving"]["maintenance_planning"][
        "penalty_weight"
    ]

    # IMPORTANT: With normalized objectives, penalty_weight should be moderate (1-1000)
    # If penalty_weight is too large, it completely dominates the objective
    if penalty_weight > 10000:
        logger.warning(
            f"penalty_weight={penalty_weight:.2e} is very large for normalized objectives. "
            f"This will cause the outage penalty to completely dominate the capacity margin term. "
            f"Recommended range: 1.0 to 1000.0. Auto-adjusting to 1000.0"
        )
        penalty_weight = 1000.0
        logger.info(f"Using penalty_weight={penalty_weight}")

    num_units_per_move = snakemake.config["solving"]["maintenance_planning"][
        "sa_units_per_move"
    ]
    n_parallel_runs = snakemake.config["solving"]["maintenance_planning"][
        "sa_parallel_runs"
    ]
    random_seed = snakemake.config["solving"]["maintenance_planning"]["sa_random_seed"]

    # Step 1: Generate maintenance schedule for NEW units (is_existing=False) based on inverse residual load
    logger.info("=" * 80)
    logger.info("STEP 1: Generating maintenance schedules for NEW units")
    logger.info("=" * 80)
    new_units_schedule = generate_maintenance_schedule_for_new_units(
        networks, snakemake.config
    )

    if not new_units_schedule.empty:
        logger.info(
            f"Generated schedules for {len(new_units_schedule.columns)} new generators"
        )
    else:
        logger.info("No new generators with maintenance requirements found")

    # Step 2: Run optimization for EXISTING units (is_existing=True)
    logger.info("=" * 80)
    logger.info("STEP 2: Running optimization for EXISTING units")
    logger.info("=" * 80)
    logger.info("Running per-bus optimization (buses optimized independently)")

    optimal_schedule, bus_results = optimize_maintenance_schedule_per_bus(
        networks,  # Pass all networks for robust optimization
        snakemake.config,  # Pass config
        initial_schedule=None,
        initial_acceptance=initial_acceptance,
        alpha=alpha,
        max_iterations=max_iterations,
        penalty_weight=penalty_weight,
        num_units_per_move=num_units_per_move,
        n_parallel_runs=n_parallel_runs,
        random_seed=random_seed,
        print_every=500,
    )

    logger.info("Simulated annealing optimization complete")
    # Compute total objective across all buses
    if bus_results:
        total_objective = sum(obj for obj, _ in bus_results.values())
        logger.info(f"Total objective across all buses: {total_objective:.2f}")
        logger.info(f"Optimized {len(bus_results)} buses independently")

        # Log per-bus results with improvement metrics
        logger.info("\n" + "=" * 80)
        logger.info("PER-BUS OPTIMIZATION SUMMARY")
        logger.info("=" * 80)

        total_initial = 0.0
        total_final = 0.0

        for bus, (final_obj, history) in bus_results.items():
            if history and len(history) > 0:
                initial_obj = history[0]
                improvement_pct = (
                    (initial_obj - final_obj) / initial_obj * 100
                    if initial_obj > 0
                    else 0.0
                )
                total_initial += initial_obj
                total_final += final_obj
                logger.info(
                    f"  {bus:20s}: Initial={initial_obj:12.2e}, Final={final_obj:12.2e}, "
                    f"Improvement={improvement_pct:6.2f}%"
                )
            else:
                logger.info(f"  {bus:20s}: Final={final_obj:12.2e} (no history)")
                total_final += final_obj

        # Overall improvement
        if total_initial > 0:
            overall_improvement = (total_initial - total_final) / total_initial * 100
            logger.info("=" * 80)
            logger.info(
                f"OVERALL: Initial={total_initial:.2e}, Final={total_final:.2e}, "
                f"Improvement={overall_improvement:.2f}%"
            )
        logger.info("=" * 80 + "\n")
    else:
        logger.info("No existing generators with maintenance requirements found")

    # Step 3: Combine schedules from existing units (optimized) and new units (derating-based)
    logger.info("=" * 80)
    logger.info("STEP 3: Combining maintenance schedules")
    logger.info("=" * 80)

    # Handle case where no maintenance schedule was generated for existing units
    if optimal_schedule.empty and new_units_schedule.empty:
        logger.warning(
            "No maintenance schedule was generated (no generators with planned_outage_rate > 0)"
        )
        logger.info(
            "Creating zero maintenance schedule (all generators fully available)"
        )

        # Create zero maintenance schedule (all generators fully available)
        zero_schedule = pd.DataFrame(index=n.snapshots)

        # Normalize timestamps to target year (year-agnostic)
        zero_schedule = normalize_timestamps_to_year_agnostic(
            zero_schedule, target_year=target_year
        )

        # Save to CSV
        logger.info(f"Saving zero maintenance plan to {output_path}")
        zero_schedule.to_csv(output_path)
        logger.info(
            "Zero maintenance plan saved successfully (no generators require maintenance)"
        )
        logger.info("=" * 80)
        sys.exit(0)

    # Process existing units schedule (unit-level from SA optimization)
    existing_units_wide_schedule = pd.DataFrame()
    if not optimal_schedule.empty:
        # Check if schedule is unit-level or fleet-level
        if "Unit" in optimal_schedule.columns:
            # Unit-level schedule (from SA) - expand to hourly then aggregate
            logger.info("Expanding existing units schedule to hourly resolution")
            hourly_schedule = expand_schedule_to_hourly(
                optimal_schedule, start_time=n.snapshots[0], end_time=n.snapshots[-1]
            )

            # Aggregate to fleet level (unavailability fraction)
            logger.info("Aggregating unit-level schedule to fleet-level unavailability")
            fleet_schedule = aggregate_unit_schedule_to_fleet(
                hourly_schedule, n, snakemake.config
            )
        else:
            # Fleet-level schedule - expand to hourly directly
            logger.info("Expanding fleet-level schedule to hourly resolution")
            hourly_schedule = expand_schedule_to_hourly(
                optimal_schedule, start_time=n.snapshots[0], end_time=n.snapshots[-1]
            )

            # Convert to unavailability fraction format
            logger.info("Converting to fleet-level unavailability format")
            fleet_schedule = hourly_schedule[
                ["Generator", "Time", "Maintenance"]
            ].copy()
            fleet_schedule.rename(
                columns={"Maintenance": "Unavailability_Fraction"}, inplace=True
            )

        # Pivot to wide format: Time (index) x Generators (columns)
        logger.info(
            "Converting existing units schedule to wide format (Time x Generators)"
        )
        existing_units_wide_schedule = fleet_schedule.pivot(
            index="Time", columns="Generator", values="Unavailability_Fraction"
        )

    # New units schedule is already in wide format (Time x Generators)
    # Combine both schedules (concatenate along columns axis)
    if not existing_units_wide_schedule.empty and not new_units_schedule.empty:
        logger.info("Combining existing and new units schedules")
        wide_schedule = pd.concat(
            [existing_units_wide_schedule, new_units_schedule], axis=1
        )
    elif not existing_units_wide_schedule.empty:
        logger.info("Using existing units schedule only (no new units)")
        wide_schedule = existing_units_wide_schedule
    else:
        logger.info("Using new units schedule only (no existing units)")
        wide_schedule = new_units_schedule

    # Normalize timestamps to target year (year-agnostic: same schedule applies to all climate years)
    wide_schedule = normalize_timestamps_to_year_agnostic(
        wide_schedule, target_year=target_year
    )

    # Save to CSV
    logger.info(f"Saving maintenance plan to {output_path}")
    wide_schedule.to_csv(output_path)

    logger.info("Maintenance plan saved successfully")
    logger.info(
        f"Schedule shape: {wide_schedule.shape[0]} time periods x {wide_schedule.shape[1]} generators"
    )
    logger.info(
        f"Generators with planned outages: {(wide_schedule.max(axis=0) > 0).sum()}"
    )
    logger.info("=" * 80)
