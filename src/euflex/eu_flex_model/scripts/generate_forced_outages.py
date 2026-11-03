# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System

"""
Monte Carlo scenario generation for stochastic modeling.

This module provides functions to generate Monte Carlo scenarios for various
uncertain parameters in the energy system model, such as forced outages.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

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


def outage_start_probability(outage_rate: float, outage_duration: int) -> float:
    """
    Per-hour probability that an available unit starts a forced outage.

    Outages are only sampled while a unit is available, so the long-run
    unavailability for a start probability p is p*d / (1 - p + p*d). Solving
    for unavailability == outage_rate gives r / (d*(1 - r) + r). This stays
    within [0, 1] for all r in [0, 1] and equals r when d == 1.
    """
    return outage_rate / (outage_duration * (1 - outage_rate) + outage_rate)


def generate_forced_outage_scenarios(
    n: pypsa.Network, seed: int | None = None, n_units_per_fleet: int = 100
) -> pd.DataFrame:
    """
    Generate forced outage scenarios based on average forced outage rates using fleet modeling.

    This function models each generator (e.g., "BE00 Gas") as a fleet of individual units.
    Outages are sampled independently for each unit, then aggregated to get fleet-level
    availability. This produces continuous availability values (0.0 to 1.0) representing
    the fraction of the fleet that is available, which is more realistic than binary on/off.

    Outage durations are carrier-specific (hardcoded):
    - Nuclear generators: 168 hours (7 days)
    - All other generators: 24 hours (1 day)

    Sampling for new outages is only done when the unit is not already in an outage period.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network containing generators with 'avg_forced_outage_rate' and 'n_units' attributes
    seed : int, optional
        Random seed for reproducibility. If None, results will not be reproducible.
    n_units_per_fleet : int, optional
        Default number of units per generator fleet if 'n_units' column is not available.
        Higher values produce smoother availability curves. Default: 100.

    Returns
    -------
    pd.DataFrame
        Fleet availability matrix with:
        - Index: snapshots (datetime index from network)
        - Columns: generator names (e.g., "BE00 Gas")
        - Values: 0.0 to 1.0 (fraction of fleet available)

    Notes
    -----
    - Only generators with non-zero avg_forced_outage_rate are included
    - Each generator is modeled as n_units independent units (from database or default)
    - Outages are sampled using Bernoulli distribution for each unit
    - Fleet availability is the mean across all units (continuous value)
    - Missing or NaN outage rates are treated as 0 (always fully available)
    - Uses modern numpy.random.Generator for reproducibility


    """
    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed)
    if seed is not None:
        logger.info(f"Using random seed {seed} for forced outage scenario generation")

    if "avg_forced_outage_rate" not in n.generators.columns:
        logger.warning(
            "No 'avg_forced_outage_rate' column found in generators. "
            "Returning all generators as available (no outages)."
        )
        return pd.DataFrame(1.0, index=n.snapshots, columns=n.generators.index)

    outage_rates = n.generators["avg_forced_outage_rate"].fillna(0)
    generators_with_outages = outage_rates[outage_rates > 0]

    if generators_with_outages.empty:
        logger.info(
            "No generators have non-zero forced outage rates. "
            "Returning all generators as available."
        )
        return pd.DataFrame(1.0, index=n.snapshots, columns=n.generators.index)

    # Check if n_units column exists in network data (from database)
    use_database_n_units = "n_units" in n.generators.columns
    if use_database_n_units:
        logger.info(
            f"Using fleet sizes from database (n_units column) for {len(generators_with_outages)} generators"
        )
    else:
        logger.info(
            f"Generating forced outage scenarios for {len(generators_with_outages)} generators "
            f"across {len(n.snapshots)} snapshots using default {n_units_per_fleet} units per fleet"
        )

    # Initialize fleet availability matrix (all generators fully available by default)
    fleet_availability = pd.DataFrame(
        1.0, index=n.snapshots, columns=n.generators.index
    )

    # Carrier-specific outage durations (hardcoded)
    # Nuclear: 7 days = 168 hours, Others: 1 day = 24 hours
    fo_duration_by_carrier = {
        "Nuclear": 168,  # 7 days
        "default": 24,  # 1 day
    }

    num_snapshots = len(n.snapshots)

    # For each generator fleet, model individual units
    for gen_name, outage_rate in generators_with_outages.items():
        # Get carrier for this generator
        carrier = n.generators.loc[gen_name, "carrier"]

        # Determine outage duration based on carrier
        # Check if carrier contains "Nuclear" (case-insensitive)
        if "nuclear" in str(carrier).lower():
            outage_duration = fo_duration_by_carrier["Nuclear"]
        else:
            outage_duration = fo_duration_by_carrier["default"]

        # Get fleet size for this generator (from database or default)
        if use_database_n_units:
            fleet_size = int(n.generators.loc[gen_name, "n_units"])
        else:
            fleet_size = n_units_per_fleet

        # Initialize unit availability matrix for this fleet
        # Shape: (num_snapshots, fleet_size)
        unit_availability = np.ones((num_snapshots, fleet_size))

        # Sample outages for each unit independently
        for unit_idx in range(fleet_size):
            in_outage_until = -1
            for t in range(num_snapshots):
                if t < in_outage_until:
                    unit_availability[t, unit_idx] = 0.0
                    continue
                # Sample new outage using Bernoulli distribution
                # rng.binomial(1, p) returns 1 with probability p (outage occurs)
                if rng.binomial(
                    1, outage_start_probability(outage_rate, outage_duration)
                ):
                    # Outage starts here, lasts for outage_duration
                    end_outage = min(t + outage_duration, num_snapshots)
                    unit_availability[t:end_outage, unit_idx] = 0.0
                    in_outage_until = end_outage

        # Aggregate unit availability to get fleet availability (mean across units)
        # This gives us a continuous value between 0.0 and 1.0
        fleet_availability.loc[:, gen_name] = unit_availability.mean(axis=1)

    # Calculate statistics on fleet availability
    affected_capacity = (fleet_availability < 1.0).sum().sum()
    total_gen_snapshots = len(n.snapshots) * len(generators_with_outages)
    affected_percentage = (
        (affected_capacity / total_gen_snapshots) * 100
        if total_gen_snapshots > 0
        else 0
    )

    # Calculate average availability reduction
    avg_availability = fleet_availability[generators_with_outages.index].mean().mean()

    logger.info(
        f"Generated forced outage scenarios: {affected_capacity:,} generator-snapshot combinations affected "
        f"({affected_percentage:.2f}% have reduced availability)"
    )
    logger.info(
        f"Average fleet availability: {avg_availability:.3f} "
        f"(1.0 = fully available, 0.0 = fully unavailable)"
    )

    return fleet_availability


def generate_link_forced_outages(
    n: pypsa.Network, seed: int | None = None, market_interconnection: str = "ntc"
) -> pd.DataFrame:
    """
    Generate forced outage scenarios for links (interconnectors) based on forced outage rates.

    This function samples outages for each pole of a link independently using a binomial
    distribution. The final link availability is the average availability across all poles.
    For example, if a link has 2 poles and one is out (availability=0) while the other is
    online (availability=1), the link's effective availability is 0.5 (50% capacity).

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network containing links with 'Forced Outage Rate [%]' and 'Number of poles [#]'
    seed : int, optional
        Random seed for reproducibility. If None, results will not be reproducible.
    market_interconnection : str, optional
        Market interconnection mode. Only "ntc" generates outages. Default: "ntc".

    Returns
    -------
    pd.DataFrame
        Link availability matrix with:
        - Index: snapshots (datetime index from network)
        - Columns: link names
        - Values: 0.0 to 1.0 (fraction of poles available)

    Notes
    -----
    - Only generates outages for market_interconnection == "ntc" or "physical"
    - For FBMC mode, returns all 1s (outages already included in flow-based domain)
    - Samples each pole independently and averages their availability
    - Missing or NaN outage rates are treated as 0 (always fully available)
    - Missing or NaN number of poles defaults to 1
    - Uses modern numpy.random.Generator for reproducibility
    - Outages last 168 hours (7 days)

    """
    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed)
    if seed is not None:
        logger.info(
            f"Using random seed {seed} for link forced outage scenario generation"
        )

    # Check if market interconnection is NTC or physical
    if market_interconnection not in ("ntc", "physical"):
        logger.info(
            f"Market interconnection mode is '{market_interconnection}', not 'ntc' or 'physical'. "
            "Skipping link forced outage generation (outages already included in flow-based domain)."
        )
        return pd.DataFrame(1.0, index=n.snapshots, columns=n.links.index)

    # Check if links have forced outage rate parameter
    if "Forced Outage Rate [%]" not in n.links.columns:
        logger.warning(
            "No 'Forced Outage Rate [%]' column found in links. "
            "Returning all links as available (no outages)."
        )
        return pd.DataFrame(1.0, index=n.snapshots, columns=n.links.index)

    # Convert percentage to probability (divide by 100)
    outage_rates = (n.links["Forced Outage Rate [%]"] / 100).fillna(0)
    links_with_outages = outage_rates[outage_rates > 0]

    if links_with_outages.empty:
        logger.info(
            "No links have non-zero forced outage rates. "
            "Returning all links as available."
        )
        return pd.DataFrame(1.0, index=n.snapshots, columns=n.links.index)

    # Get number of poles for each link (default to 1 if not specified)
    if "Number of poles [#]" in n.links.columns:
        n_poles = n.links["Number of poles [#]"].fillna(1).astype(int)
    else:
        logger.warning(
            "No 'Number of poles [#]' column found in links. "
            "Defaulting to 1 pole per link."
        )
        n_poles = pd.Series(1, index=n.links.index)

    logger.info(
        f"Generating forced outage scenarios for {len(links_with_outages)} links "
        f"across {len(n.snapshots)} snapshots (sampling {n_poles[links_with_outages.index].sum()} poles total)"
    )

    # Initialize link availability matrix (all links fully available by default)
    link_availability = pd.DataFrame(1.0, index=n.snapshots, columns=n.links.index)

    # Outage duration in hours (snapshots) - hardcoded to 7 days
    outage_duration = 168  # 7 days = 168 hours
    num_snapshots = len(n.snapshots)

    # For each link, sample outages for each pole independently
    for link_name, outage_rate in links_with_outages.items():
        num_poles = n_poles[link_name]

        # Initialize pole availability matrix for this link
        # Shape: (num_snapshots, num_poles)
        pole_availability = np.ones((num_snapshots, num_poles))

        # Sample outages for each pole independently
        for pole_idx in range(num_poles):
            in_outage_until = -1
            for t in range(num_snapshots):
                if t < in_outage_until:
                    pole_availability[t, pole_idx] = 0.0
                    continue
                # Sample new outage using Bernoulli distribution
                # rng.binomial(1, p) returns 1 with probability p (outage occurs)
                if rng.binomial(
                    1, outage_start_probability(outage_rate, outage_duration)
                ):
                    # Outage starts here, lasts for outage_duration
                    end_outage = min(t + outage_duration, num_snapshots)
                    pole_availability[t:end_outage, pole_idx] = 0.0
                    in_outage_until = end_outage

        # Link availability is the mean across all poles (fraction of poles available)
        link_availability.loc[:, link_name] = pole_availability.mean(axis=1)

    # Calculate statistics on link availability
    affected_capacity = (link_availability < 1.0).sum().sum()
    total_link_snapshots = len(n.snapshots) * len(links_with_outages)
    affected_percentage = (
        (affected_capacity / total_link_snapshots) * 100
        if total_link_snapshots > 0
        else 0
    )

    # Calculate average availability reduction
    avg_availability = link_availability[links_with_outages.index].mean().mean()

    logger.info(
        f"Generated link forced outage scenarios: {affected_capacity:,} link-snapshot combinations affected "
        f"({affected_percentage:.2f}% have reduced availability)"
    )
    logger.info(
        f"Average link availability: {avg_availability:.3f} "
        f"(1.0 = fully available, 0.0 = fully unavailable)"
    )

    return link_availability


def align_scenario_to_snapshots(
    scenario_df: pd.DataFrame,
    n: pypsa.Network,
    columns: pd.Index | list | None = None,
    fill_value: float = 1.0,
) -> pd.DataFrame:
    """
    Align a (possibly reference-climate-year) scenario to a network's snapshots.

    Forced-outage scenarios are generated once from a fixed reference network
    and reused across every climate year for the same target year, so the
    scenario's own datetime index generally belongs to a different calendar
    year than the network it is being applied to. This aligns by exact
    month-day-hour match instead of absolute date, so a snapshot with no
    matching month-day-hour in the scenario (e.g. Feb 29 when the reference
    year isn't a leap year) falls back to `fill_value` rather than carrying
    forward a neighbouring value.

    Parameters
    ----------
    scenario_df : pd.DataFrame
        Scenario values indexed by (reference-year) datetimes.
    n : pypsa.Network
        Network whose snapshots the scenario should be aligned to.
    columns : pd.Index | list | None
        Columns to reindex onto (e.g. the network's current generators/links).
        Defaults to the scenario's own columns.
    fill_value : float
        Value used for any snapshot/column not covered by the scenario.

    Returns
    -------
    pd.DataFrame
        Scenario reindexed onto `n.snapshots` and `columns`.
    """
    if columns is None:
        columns = scenario_df.columns

    if isinstance(scenario_df.index, pd.DatetimeIndex) and len(scenario_df.index) > 0:
        month_day_hour = scenario_df.index.strftime("%m-%d %H:%M:%S")
        aligned = scenario_df.set_axis(month_day_hour, axis="index").reindex(
            index=n.snapshots.strftime("%m-%d %H:%M:%S")
        )
        aligned.index = n.snapshots
    else:
        aligned = scenario_df.reindex(index=n.snapshots)

    return aligned.reindex(columns=columns).fillna(fill_value)


def apply_forced_outages(
    n: pypsa.Network,
    outage_scenarios: pd.DataFrame,
) -> tuple[pypsa.Network, pd.DataFrame]:
    """
    Apply pre-generated forced outage scenarios to network generators.

    This function takes pre-generated forced outage scenarios and applies them to the
    network by updating generator p_max_pu, p_min_pu, and availability values.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network with generators
    outage_scenarios : pd.DataFrame
        Binary (0/1) availability scenarios with:
        - Index: snapshots (datetime index matching network)
        - Columns: generator names
        - Values: 1 (available) or 0 (forced outage)

    Returns
    -------
    tuple[pypsa.Network, pd.DataFrame]
        - Modified network with updated p_max_pu, p_min_pu, and availability
        - Final availability matrix (base p_max_pu * outage scenarios)

    Notes
    -----
    - Modifies the network in-place by updating n.generators_t attributes
    - Uses get_switchable_as_dense to properly handle static and timeseries p_max_pu
    - For generators with timeseries p_max_pu (e.g., renewables), applies outages on top
    - For generators with static p_max_pu (e.g., conventional), converts to timeseries

    """
    logger.info("Applying forced outage scenarios to network generators")

    all_gens = n.generators.index

    # Align outage_scenarios to this network's generators and snapshots.
    # The DB path builds a full-year index and only includes generators with
    # avg_forced_outage_rate > 0; missing entries fill with 1.0 (fully available).
    outage_scenarios = align_scenario_to_snapshots(
        outage_scenarios, n, columns=all_gens, fill_value=1.0
    )

    # Get base p_max_pu and p_min_pu values (handles both static and timeseries)
    # IMPORTANT: If apply_maintenance_schedules has already been called, these will already
    # include the planned maintenance adjustments (p_max_pu and p_min_pu will be in generators_t)
    base_p_max_pu = as_dense(n, "Generator", "p_max_pu", all_gens)
    base_p_min_pu = as_dense(n, "Generator", "p_min_pu", all_gens)

    # Check if planned maintenance is active
    # Note: n.generators_t.availability stores availability (1.0 = fully available, <1.0 = under maintenance)
    # Note: outage_scenarios stores availability (1.0 = available, 0.0 = forced outage)
    # A Generator represents an aggregated fleet of units, so partial maintenance
    # (0.0 < availability < 1.0) still leaves some units active and able to suffer
    # a forced outage independently; only a fleet fully down for maintenance
    # (availability == 0.0) has no units left that could be forced out.
    if "availability" in n.generators_t and not n.generators_t["availability"].empty:
        planned_availability = n.generators_t["availability"]
        # Where the fleet is fully down for maintenance (availability == 0.0), set
        # forced outage scenarios to 1.0 (no additional forced outage to apply)
        forced_outage_scenarios = outage_scenarios.copy()
        maintenance_full = planned_availability <= 1e-9
        forced_outage_scenarios[maintenance_full] = (
            1.0  # Set to 1.0 = no forced outage when fully down for maintenance
        )
        logger.info(
            f"Prevented forced outages while fully down for planned maintenance: "
            f"{maintenance_full.sum().sum()} generator-snapshot combinations protected"
        )
    else:
        # No planned maintenance active, use forced outages as-is
        forced_outage_scenarios = outage_scenarios

    # Find the minimum availability allowed based on max_forced_outage_capacity
    if "max_forced_outage_capacity" in n.generators.columns:
        # Calculate maximum outage as fraction of p_nom
        max_outage_fraction = (
            as_dense(n, "Generator", "max_forced_outage_capacity", all_gens)
            / as_dense(n, "Generator", "p_nom", all_gens)
        ).clip(upper=1.0, lower=0.0)
        # Minimum availability = 1.0 - max_outage_fraction
        min_availability = 1.0 - max_outage_fraction
    else:
        # Default: allow full outage (minimum availability = 0)
        min_availability = pd.DataFrame(0.0, index=n.snapshots, columns=all_gens)

    # Apply forced outages on top of base p_max_pu (which may already include planned maintenance
    # and/or intraday ramp-alignment scaling). min_availability bounds how severe the forced
    # outage scenario itself can be (per max_forced_outage_capacity) — it must be applied to the
    # outage scenario before multiplying by base_p_max_pu, not to the resulting product. Clipping
    # the product would silently raise a legitimately tiny base_p_max_pu (e.g. from align_ramp_limits)
    # back up to min_availability whenever max_forced_outage_capacity is 0, which is not the same
    # thing as no forced outage being applied.
    forced_outage_scenarios = forced_outage_scenarios.clip(lower=min_availability)
    final_availability = base_p_max_pu * forced_outage_scenarios

    # Adjust p_min_pu: multiply by forced_outage_scenarios (sets to 0 during outages),
    # then clip to final_availability (handles derating cases)
    # This ensures: p_min_pu = 0 during outages, and p_min_pu <= p_max_pu always
    # base_p_min_pu already includes planned maintenance adjustments if apply_maintenance_schedules was called
    p_min_pu_timeseries_adjusted = (base_p_min_pu * forced_outage_scenarios).clip(
        upper=final_availability
    )

    # Update the network with final availability
    n.generators_t.p_max_pu = final_availability
    n.generators_t.p_min_pu = p_min_pu_timeseries_adjusted

    # Update availability time series to reflect combined planned and forced outages.
    # forced_outage_scenarios is already clipped to min_availability above, so no further
    # clipping is applied here — doing so would re-introduce the bug of overriding a
    # legitimately low planned availability back up to min_availability.
    if "availability" in n.generators_t and not n.generators_t["availability"].empty:
        # If there was a planned availability, update it to include forced outages
        # (but forced_outage_scenarios is already 1.0 where maintenance is active)
        n.generators_t.availability = (
            n.generators_t.availability * forced_outage_scenarios
        )
    else:
        # No planned maintenance, just set to forced outage scenarios
        n.generators_t.availability = forced_outage_scenarios

    # Logging statistics
    n_outages_applied = (final_availability < base_p_max_pu).sum().sum()
    total_gen_snapshots = len(n.snapshots) * len(all_gens)
    outage_impact_pct = (
        (n_outages_applied / total_gen_snapshots) * 100
        if total_gen_snapshots > 0
        else 0
    )

    logger.info(
        "Applied forced outages: %s generator-snapshot combinations affected (%s%% of total)",
        f"{n_outages_applied:,}",
        f"{outage_impact_pct:.2f}",
    )

    # Log capacity impact
    if "p_nom" in n.generators.columns:
        base_capacity = (
            (base_p_max_pu.multiply(n.generators.loc[all_gens, "p_nom"])).sum().sum()
        )
        available_capacity = (
            (final_availability.multiply(n.generators.loc[all_gens, "p_nom"]))
            .sum()
            .sum()
        )
        capacity_reduction_pct = (
            ((base_capacity - available_capacity) / base_capacity * 100)
            if base_capacity > 0
            else 0
        )
        logger.info(
            "Capacity impact: %s%% average reduction in available capacity",
            f"{capacity_reduction_pct:.2f}",
        )

    return n, final_availability


def apply_forced_outages_to_links(
    n: pypsa.Network,
    link_availability: pd.DataFrame,
) -> tuple[pypsa.Network, pd.DataFrame]:
    """
    Apply pre-generated forced outage scenarios to network links.

    This function takes pre-generated link outage scenarios and applies them to the
    network by updating link p_max_pu and (for bi-directional links) p_min_pu values.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network with links
    link_availability : pd.DataFrame
        Availability scenarios with:
        - Index: snapshots (datetime index matching network)
        - Columns: link names
        - Values: 0.0 to 1.0 (fraction of poles available)

    Returns
    -------
    tuple[pypsa.Network, pd.DataFrame]
        - Modified network with updated p_max_pu (and p_min_pu for bi-directional links)
        - Final availability matrix (base p_max_pu * link availability)

    Notes
    -----
    - Modifies the network in-place by updating n.links_t attributes
    - Uses get_switchable_as_dense to properly handle static and timeseries p_max_pu
    - For bi-directional links (FOR direction == "Bi-directional"), updates both p_max_pu and p_min_pu
    - For uni-directional links (FOR direction == "Uni-directional"), only updates p_max_pu
    - If FOR direction column is missing, assumes all links are bi-directional

    """
    logger.info("Applying forced outage scenarios to network links")

    all_links = n.links.index

    # Align link_availability to this network's links and snapshots (see
    # align_scenario_to_snapshots docstring for why this is month-day-hour based).
    link_availability = align_scenario_to_snapshots(
        link_availability, n, columns=all_links, fill_value=1.0
    )

    # Get base p_max_pu and p_min_pu values (handles both static and timeseries)
    base_p_max_pu = as_dense(n, "Link", "p_max_pu", all_links)
    base_p_min_pu = as_dense(n, "Link", "p_min_pu", all_links)

    # Multiply base p_max_pu by link availability to get final p_max_pu
    final_p_max_pu = base_p_max_pu * link_availability

    # Update the network with final p_max_pu
    n.links_t.p_max_pu = final_p_max_pu

    # Handle p_min_pu based on FOR direction
    if "FOR direction" in n.links.columns:
        # Get direction for each link
        for_direction = n.links["FOR direction"].fillna("Bi-directional")

        # Identify bi-directional links
        bidirectional_links = for_direction[for_direction == "Bi-directional"].index

        if len(bidirectional_links) > 0:
            # For bi-directional links, also multiply p_min_pu by availability
            final_p_min_pu = base_p_min_pu.copy()
            final_p_min_pu[bidirectional_links] = (
                base_p_min_pu[bidirectional_links]
                * link_availability[bidirectional_links]
            )
            n.links_t.p_min_pu = final_p_min_pu

            logger.info(
                f"Updated p_min_pu for {len(bidirectional_links)} bi-directional links"
            )

        # Count uni-directional links
        unidirectional_links = for_direction[for_direction == "Uni-directional"].index
        if len(unidirectional_links) > 0:
            logger.info(
                f"Kept original p_min_pu for {len(unidirectional_links)} uni-directional links"
            )
    else:
        # If no FOR direction column, assume all links are bi-directional
        logger.warning(
            "No 'FOR direction' column found. Assuming all links are bi-directional."
        )
        final_p_min_pu = base_p_min_pu * link_availability
        n.links_t.p_min_pu = final_p_min_pu

    # Logging statistics
    n_outages_applied = (final_p_max_pu < base_p_max_pu).sum().sum()
    total_link_snapshots = len(n.snapshots) * len(all_links)
    outage_impact_pct = (
        (n_outages_applied / total_link_snapshots) * 100
        if total_link_snapshots > 0
        else 0
    )

    logger.info(
        "Applied forced outages: %s link-snapshot combinations affected (%s%% of total)",
        f"{n_outages_applied:,}",
        f"{outage_impact_pct:.2f}",
    )

    # Log capacity impact
    if "p_nom" in n.links.columns:
        base_capacity = (
            (base_p_max_pu.multiply(n.links.loc[all_links, "p_nom"])).sum().sum()
        )
        available_capacity = (
            (final_p_max_pu.multiply(n.links.loc[all_links, "p_nom"])).sum().sum()
        )
        capacity_reduction_pct = (
            ((base_capacity - available_capacity) / base_capacity * 100)
            if base_capacity > 0
            else 0
        )
        logger.info(
            "Link capacity impact: %s%% average reduction in available capacity",
            f"{capacity_reduction_pct:.2f}",
        )

    return n, final_p_max_pu


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="generate_forced_outages",
    )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the base network
    network = pypsa.Network(
        snakemake.input.network  # ty: ignore[possibly-missing-attribute]
    )

    # Retrieve eva config dict
    config_eva = snakemake.config["eva"]
    forced_outages_config = config_eva["forced_outages"]

    seed_config = forced_outages_config["seed"]

    if seed_config is not None:
        # Parse seed list
        if isinstance(seed_config, str):
            # Parse comma-separated string
            seed_list = [int(s.strip()) for s in seed_config.split(",")]
        elif isinstance(seed_config, list):
            # Already a list
            seed_list = [int(s) for s in seed_config]
        else:
            # Single integer - treat as a list with one element
            seed_list = [int(seed_config)]

        # Extract forced outage number from wildcard (e.g., "fo1" -> 1, "fo2" -> 2)
        fo_str = snakemake.wildcards.fo  # ty: ignore[possibly-missing-attribute]
        fo_number = int(fo_str.replace("fo", ""))  # Extract the number

        # Get the seed for this forced outage sample (1-indexed)
        if fo_number <= len(seed_list):
            seed = seed_list[fo_number - 1]  # -1 for 0-indexed list
            logger.info(f"Using seed {seed} for {fo_str} (from seed list: {seed_list})")
        else:
            raise ValueError(
                f"Not enough seeds configured! {fo_str} requires at least {fo_number} seeds, "
                f"but only {len(seed_list)} provided: {seed_list}"
            )
    else:
        # No seed configured - results will not be reproducible
        seed = None
        logger.warning(
            "No seed configured in eva.forced_outages.seed - results will not be reproducible"
        )

    # Get number of units per fleet from config (default: 100)
    n_units_per_fleet = forced_outages_config["n_units_per_fleet"]
    logger.info(
        f"Modeling each generator fleet as {n_units_per_fleet} independent units"
    )

    # Generate forced outage scenarios for generators
    if forced_outages_config["enable"]:
        outage_scenarios = generate_forced_outage_scenarios(
            network, seed=seed, n_units_per_fleet=n_units_per_fleet
        )
    else:
        outage_scenarios = pd.DataFrame(
            1.0, index=network.snapshots, columns=network.generators.index
        )

    # Export generator outage scenarios to CSV
    outage_scenarios.to_csv(
        snakemake.output.outage_scenarios  # ty: ignore[possibly-missing-attribute]
    )
    logger.info(
        f"Generated forced outage scenarios and saved to {snakemake.output.outage_scenarios}"  # ty: ignore[possibly-missing-attribute]
    )

    # Generate forced outage scenarios for links (interconnectors)
    market_interconnection = config_eva.get("market_interconnection", "ntc")
    if forced_outages_config["enable"]:
        link_availability = generate_link_forced_outages(
            network, seed=seed, market_interconnection=market_interconnection
        )
    else:
        link_availability = pd.DataFrame(
            1.0, index=network.snapshots, columns=network.links.index
        )

    # Export link availability to CSV
    link_availability.to_csv(
        snakemake.output.link_availability  # ty: ignore[possibly-missing-attribute]
    )
    logger.info(
        f"Generated link availability and saved to {snakemake.output.link_availability}"  # ty: ignore[possibly-missing-attribute]
    )
