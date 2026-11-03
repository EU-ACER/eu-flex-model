import logging
import math
import pathlib

import cartopy.crs as ccrs
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib import colormaps
from matplotlib.patches import Patch

from euflex.eu_flex_model.scripts.generate_forced_outages import (
    align_scenario_to_snapshots,
)
from euflex.eu_flex_model.scripts.helpers.date_helpers import DateFormat
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
    set_scenario_config,
)

matplotlib.use("Agg")  # Use non-GUI backend to avoid X server issues
logger = logging.getLogger(__name__)


def validate_dsr_config_for_year(
    config: dict, target_year: int, available_years: dict | None = None
) -> dict:
    """
    Validate DSR configuration and data availability for target year.

    Parameters
    ----------
    config : dict
        Snakemake configuration dictionary
    target_year : int
        Target year to validate DSR data for
    available_years : dict, optional
        Dictionary with 'explicit' and 'implicit' keys containing available years.
        If None, validation will be based on config only.

    Returns
    -------
    dict
        Validation results with keys: enabled, explicit_available, implicit_available, warnings
    """
    result = {
        "enabled": False,
        "explicit_available": False,
        "implicit_available": False,
    }

    # Ensure warnings is always a list before appending
    if "warnings" not in result or not isinstance(result["warnings"], list):
        result["warnings"] = []

    # Check if DSR is enabled in config
    eva_config = config["eva"]
    dsr_config = eva_config["dsr"]

    result["enabled"] = dsr_config["enabled"]

    if not result["enabled"]:
        result["warnings"].append(
            f"DSR disabled in configuration (eva.dsr.enabled = {result['enabled']})"
        )
        return result

    # If available years are provided, use them for validation
    if available_years:
        explicit_dsr_years = available_years.get("explicit", [])
        implicit_dsr_years = available_years.get("implicit", [])
    else:
        # Try to get years from config, fallback to empty lists
        explicit_dsr_years = dsr_config["explicit_dsr"]["available_years"]
        implicit_dsr_years = dsr_config["implicit_dsr"]["available_years"]

    # Check explicit DSR
    explicit_enabled = dsr_config["explicit_dsr"]["enabled"]
    if explicit_enabled:
        result["explicit_available"] = (
            target_year in explicit_dsr_years if explicit_dsr_years else True
        )
        if explicit_dsr_years and not result["explicit_available"]:
            result["warnings"].append(
                f"Explicit DSR enabled but no data for year {target_year} (available: {explicit_dsr_years})"
            )

    # Check implicit DSR
    implicit_enabled = dsr_config["implicit_dsr"]["enabled"]
    if implicit_enabled:
        result["implicit_available"] = (
            target_year in implicit_dsr_years if implicit_dsr_years else True
        )
        if implicit_dsr_years and not result["implicit_available"]:
            result["warnings"].append(
                f"Implicit DSR enabled but no data for year {target_year} (available: {implicit_dsr_years})"
            )

    return result


def plot_dsr_validation_implicit(n: pypsa.Network, output_filename: str) -> None:
    """
    Create validation plots for implicit DSR (storage units with carrier == "DSR_implicit").

    Generates dsr_validation_implicit.png with the following subplots:
    - EV dispatch vs demand
    - HP dispatch vs demand
    - Battery SOC evolution
    - Capacity summary waterfall chart by BZ

    Parameters
    ----------
    n : pypsa.Network
        Network with DSR components and optimization results
    output_filename : str
        Path to save the implicit DSR validation plot
    """
    # Filter implicit DSR storage units
    if n.storage_units.empty:
        _create_no_data_plot(output_filename, "No Implicit DSR Storage Units Found")
        return

    implicit_dsr = n.storage_units[n.storage_units.carrier == "DSR_implicit"]

    if implicit_dsr.empty:
        _create_no_data_plot(output_filename, "No Implicit DSR Storage Units Found")
        return

    logger.info(f"Implicit DSR validation: {len(implicit_dsr)} storage units")

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Implicit DSR Validation (Storage Units)", fontsize=16, fontweight="bold"
    )

    # 1. EV Dispatch + Demand
    ax1 = axes[0, 0]
    _plot_implicit_dispatch_with_demand(n, implicit_dsr, ax1, tech_filter="EV")
    ax1.set_title("EV Dispatch vs Flexible Demand")
    _add_subplot_label(ax1, "a")

    # 2. HP Dispatch + Demand
    ax2 = axes[0, 1]
    _plot_implicit_dispatch_with_demand(n, implicit_dsr, ax2, tech_filter="HP")
    ax2.set_title("HP Dispatch vs Flexible Demand")
    _add_subplot_label(ax2, "b")

    # 3. Battery SOC
    ax3 = axes[1, 0]
    _plot_battery_soc(n, implicit_dsr, ax3)
    ax3.set_title("Battery State of Charge (SOC)")
    _add_subplot_label(ax3, "c")

    # 4. Capacity Summary Waterfall
    ax4 = axes[1, 1]
    _plot_capacity_waterfall(n, implicit_dsr, ax4)
    ax4.set_title("Implicit DSR Capacity Summary by BZ")
    _add_subplot_label(ax4, "d")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Implicit DSR validation plot saved: {output_filename}")


def plot_dsr_validation_explicit(n: pypsa.Network, output_filename: str) -> None:
    """
    Create validation plots for explicit DSR (generators with carrier == "DSR_explicit").

    Generates dsr_validation_explicit.png with the following subplots:
    - Load shedding dispatch time series
    - Load shifting dispatch time series
    - Max activation hours (stacked bar)
    - Capacity summary waterfall chart by BZ

    Parameters
    ----------
    n : pypsa.Network
        Network with DSR components and optimization results
    output_filename : str
        Path to save the explicit DSR validation plot
    """
    # Filter explicit DSR generators
    if n.generators.empty:
        _create_no_data_plot(output_filename, "No Explicit DSR Generators Found")
        return

    explicit_dsr = n.generators[n.generators.carrier == "DSR_explicit"]

    if explicit_dsr.empty:
        _create_no_data_plot(output_filename, "No Explicit DSR Generators Found")
        return

    logger.info(f"Explicit DSR validation: {len(explicit_dsr)} generators")

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Explicit DSR Validation (Generators)", fontsize=16, fontweight="bold")

    # 1. Load Shedding Dispatch
    ax1 = axes[0, 0]
    _plot_explicit_dispatch_by_type(n, explicit_dsr, ax1, dsr_type="load_shedding")
    ax1.set_title("Load Shedding Dispatch Time Series")
    _add_subplot_label(ax1, "a")

    # 2. Load Shifting Dispatch
    ax2 = axes[0, 1]
    _plot_explicit_dispatch_by_type(n, explicit_dsr, ax2, dsr_type="load_shifting")
    ax2.set_title("Load Shifting Dispatch Time Series")
    _add_subplot_label(ax2, "b")

    # 3. Max Activation Hours
    ax3 = axes[1, 0]
    _plot_max_activation_hours_stacked(n, explicit_dsr, ax3)
    ax3.set_title("Max Activation Hours (Used vs Remaining)")
    _add_subplot_label(ax3, "c")

    # 4. Capacity Summary Waterfall
    ax4 = axes[1, 1]
    _plot_capacity_waterfall(n, explicit_dsr, ax4)
    ax4.set_title("Explicit DSR Capacity Summary by BZ")
    _add_subplot_label(ax4, "d")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Explicit DSR validation plot saved: {output_filename}")


def _create_no_data_plot(output_filename: str, message: str) -> None:
    """Create a simple plot with a message when no data is available."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


def _add_subplot_label(ax, label: str) -> None:
    """Add a labeled box (a, b, c, d) to subplot."""
    ax.text(
        0.02,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="square,pad=0.3", facecolor="black", edgecolor="black"),
        zorder=1000,
    )


def _plot_implicit_dispatch_with_demand(
    n: pypsa.Network,
    implicit_dsr: pd.DataFrame,
    ax,
    tech_filter: str,
) -> None:
    """
    Plot flexible demand and adjusted demand (after implicit DSR dispatch).
    Fill the area where adjusted demand differs from flexible demand with
    colors showing demand increase or decrease.

    Parameters
    ----------
    n : pypsa.Network
        Network with DSR components
    implicit_dsr : pd.DataFrame
        DataFrame of implicit DSR storage units
    ax : matplotlib.axes.Axes
        Axes to plot on
    tech_filter : str
        Technology filter string (e.g., "EV", "HP")
    """

    # Filter storage units by technology
    tech_units = [idx for idx in implicit_dsr.index if tech_filter in idx]
    if not tech_units:
        ax.text(
            0.5,
            0.5,
            f"No {tech_filter} DSR Units Found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Check if required data is available
    if n.storage_units_t.p.empty or not hasattr(n.storage_units_t, "flexible_demand"):
        ax.text(
            0.5,
            0.5,
            "Missing demand or dispatch data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Aggregate flexible demand and dispatch for all units of this technology
    valid_units = [
        u
        for u in tech_units
        if u in n.storage_units_t.p.columns
        and u in n.storage_units_t.flexible_demand.columns
    ]
    if not valid_units:
        ax.text(
            0.5,
            0.5,
            f"No matching demand/dispatch data for {tech_filter}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    flexible_demand = n.storage_units_t.flexible_demand[valid_units].sum(axis=1)
    dispatch = n.storage_units_t.p[valid_units].sum(axis=1)

    # Compute adjusted demand = flexible_demand - dispatch
    adjusted_demand = flexible_demand - dispatch

    # Plot base and adjusted demand lines
    ax.plot(
        flexible_demand.index,
        flexible_demand,
        label="Flexible demand",
        color="tab:blue",
        linewidth=2,
    )
    ax.plot(
        adjusted_demand.index,
        adjusted_demand,
        label="Adjusted demand",
        color="tab:orange",
        linewidth=2,
    )

    # Fill area between them
    above_mask = adjusted_demand > flexible_demand  # demand increased (charging)
    below_mask = adjusted_demand < flexible_demand  # demand reduced (discharging)

    # Fill where adjusted demand is higher (charge)
    ax.fill_between(
        flexible_demand.index,
        flexible_demand,
        adjusted_demand,
        where=above_mask,
        interpolate=True,
        color="tab:red",
        alpha=0.2,
        label="Demand increase (charge)",
    )

    # Fill where adjusted demand is lower (discharge)
    ax.fill_between(
        flexible_demand.index,
        flexible_demand,
        adjusted_demand,
        where=below_mask,
        interpolate=True,
        color="tab:green",
        alpha=0.3,
        label="Demand reduction (discharge)",
    )

    # Format plot
    ax.set_ylabel("Power (MW)")
    ax.set_xlabel("Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(DateFormat.PLOT_DAY_ABBR_MONTH.value)
    )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(8)


def _plot_battery_soc(
    n: pypsa.Network,
    implicit_dsr: pd.DataFrame,
    ax,
) -> None:
    """
    Plot battery SOC evolution for implicit DSR batteries.

    Parameters
    ----------
    n : pypsa.Network
        Network with DSR components
    implicit_dsr : pd.DataFrame
        DataFrame of implicit DSR storage units
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    # Filter battery storage units
    battery_units = [idx for idx in implicit_dsr.index if "battery" in idx.lower()]

    if not battery_units:
        ax.text(
            0.5,
            0.5,
            "No Battery DSR Units Found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Get SOC data
    if n.storage_units_t.state_of_charge.empty:
        ax.text(
            0.5,
            0.5,
            "No SOC Data Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Plot SOC for each battery
    colors = colormaps["tab10"](np.linspace(0, 1, len(battery_units)))

    for idx, unit in enumerate(battery_units):
        if unit not in n.storage_units_t.state_of_charge.columns:
            continue

        soc = n.storage_units_t.state_of_charge[unit]

        if soc.abs().max() > 1e-6:
            ax.plot(
                soc.index,
                soc,
                label=unit,
                color=colors[idx],
                alpha=0.7,
                linewidth=2,
            )

    ax.set_ylabel("State of Charge (MWh)")
    ax.set_xlabel("Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(DateFormat.PLOT_DAY_ABBR_MONTH_YEAR.value)
    )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(8)


def _plot_explicit_dispatch_by_type(
    n: pypsa.Network,
    explicit_dsr: pd.DataFrame,
    ax,
    dsr_type: str,
) -> None:
    """
    Plot explicit DSR generator dispatch time series for a specific DSR type.

    Parameters
    ----------
    n : pypsa.Network
        Network with DSR components
    explicit_dsr : pd.DataFrame
        DataFrame of explicit DSR generators
    ax : matplotlib.axes.Axes
        Axes to plot on
    dsr_type : str
        DSR type filter ("load_shedding" or "load_shifting")
    """
    # Filter generators by dsr_type
    if "dsr_type" not in explicit_dsr.columns:
        ax.text(
            0.5,
            0.5,
            "No dsr_type column found in generators",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    type_gens = explicit_dsr[explicit_dsr.dsr_type == dsr_type]

    if type_gens.empty:
        ax.text(
            0.5,
            0.5,
            f"No {dsr_type} DSR Generators Found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    if n.generators_t.p.empty:
        ax.text(
            0.5,
            0.5,
            "No Dispatch Data Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Aggregate dispatch by BZ
    bz_dispatch = {}
    for gen in type_gens.index:
        if gen not in n.generators_t.p.columns:
            continue

        bus = type_gens.loc[gen, "bus"]
        bz = n.buses.loc[bus, "country"] if "country" in n.buses.columns else bus

        if bz not in bz_dispatch:
            bz_dispatch[bz] = n.generators_t.p[gen].copy()
        else:
            bz_dispatch[bz] += n.generators_t.p[gen]

    if not bz_dispatch:
        ax.text(
            0.5,
            0.5,
            f"No Active {dsr_type} Dispatch",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Plot aggregated dispatch per BZ
    colors = colormaps["tab20"](np.linspace(0, 1, len(bz_dispatch)))

    for idx, (bz, dispatch) in enumerate(bz_dispatch.items()):
        if dispatch.abs().max() > 1e-6:
            ax.plot(
                dispatch.index,
                dispatch,
                label=bz,
                color=colors[idx],
                alpha=0.7,
                linewidth=2,
            )

    ax.set_ylabel("Power (MW)")
    ax.set_xlabel("Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(DateFormat.PLOT_DAY_ABBR_MONTH_YEAR.value)
    )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(8)


def _plot_max_activation_hours_stacked(
    n: pypsa.Network,
    explicit_dsr: pd.DataFrame,
    ax,
) -> None:
    """
    Plot stacked bar chart showing used vs remaining activation hours.

    Parameters
    ----------
    n : pypsa.Network
        Network with DSR components
    explicit_dsr : pd.DataFrame
        DataFrame of explicit DSR generators
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    # Get generators with max_daily_hours constraint
    if "max_daily_hours" not in explicit_dsr.columns:
        ax.text(
            0.5,
            0.5,
            "No max_daily_hours constraint found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    constrained_gens = explicit_dsr[explicit_dsr["max_daily_hours"].notna()]

    if constrained_gens.empty:
        ax.text(
            0.5,
            0.5,
            "No DSR with max_daily_hours constraint found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    if n.generators_t.p.empty:
        ax.text(
            0.5,
            0.5,
            "No Dispatch Data Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Calculate used and remaining hours
    gen_labels = []
    hours_used = []
    hours_remaining = []
    max_hours = []

    for gen in constrained_gens.index:
        # Get the max_daily_hours limit
        limit = constrained_gens.loc[gen, "max_daily_hours"]

        # Calculate daily activation hours using commitment status
        # Check if status/commitment data exists
        if hasattr(n.generators_t, "status") and gen in n.generators_t.status.columns:
            # Use commitment status (binary variable indicating if generator is committed)
            commitment = n.generators_t.status[gen]
            daily_activation = commitment.groupby(commitment.index.date).apply(
                lambda x: x.sum()  # Sum the binary status values (0 or 1)
            )
        elif gen in n.generators_t.p.columns:
            # Fallback: use dispatch if status not available
            # For load shedding, activation means the generator is running (p > threshold)
            dispatch = n.generators_t.p[gen]
            daily_activation = dispatch.groupby(dispatch.index.date).apply(
                lambda x: (x > 0.01).sum()
            )
        else:
            continue

        # Get the mean daily activation
        mean_activation = daily_activation.mean() if len(daily_activation) > 0 else 0

        gen_labels.append(gen)
        hours_used.append(mean_activation)
        hours_remaining.append(max(0, limit - mean_activation))
        max_hours.append(limit)

    if not gen_labels:
        ax.text(
            0.5,
            0.5,
            "No Active DSR with max_daily_hours",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Create stacked bar chart
    x = np.arange(len(gen_labels))
    width = 0.6

    # Lower part: hours used
    ax.bar(
        x,
        hours_used,
        width,
        label="Hours Used",
        color="lightcoral",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Upper part: remaining hours
    ax.bar(
        x,
        hours_remaining,
        width,
        bottom=hours_used,
        label="Hours Remaining",
        color="lightblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Display max hours at the top of each bar
    for i, max_h in enumerate(max_hours):
        ax.text(
            x[i],
            max_h + max(max_hours) * 0.02,
            f"{max_h:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("DSR Technology")
    ax.set_ylabel("Hours")
    ax.set_xticks(x)
    ax.set_xticklabels(gen_labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max_hours) * 1.15)


def _plot_capacity_waterfall(
    n: pypsa.Network,
    dsr_components: pd.DataFrame,
    ax,
) -> None:
    """
    Plot waterfall chart of DSR capacity by bidding zone.

    Parameters
    ----------
    n : pypsa.Network
        Network with DSR components
    dsr_components : pd.DataFrame
        DataFrame of DSR components (generators or storage units)
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    # Aggregate capacity by bidding zone
    bz_capacity = {}

    for component in dsr_components.index:
        bus = dsr_components.loc[component, "bus"]
        bz = n.buses.loc[bus, "country"] if "country" in n.buses.columns else bus
        capacity = dsr_components.loc[component, "p_nom"]

        if bz not in bz_capacity:
            bz_capacity[bz] = 0
        bz_capacity[bz] += capacity

    if not bz_capacity:
        ax.text(
            0.5,
            0.5,
            "No Capacity Data Available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.axis("off")
        return

    # Sort BZs by capacity (descending)
    sorted_bzs = sorted(bz_capacity.items(), key=lambda x: x[1], reverse=True)

    # Create waterfall data
    bzs = [bz for bz, _ in sorted_bzs]
    capacities = [cap for _, cap in sorted_bzs]

    # Calculate cumulative positions for waterfall
    cumulative = [0]
    for cap in capacities:
        cumulative.append(cumulative[-1] + cap)

    # Add total
    bzs.append("Total")
    capacities.append(cumulative[-1])

    # Plot waterfall
    colors = ["steelblue"] * len(sorted_bzs) + ["darkgreen"]

    x = np.arange(len(bzs))

    # Plot individual BZ bars
    for i in range(len(sorted_bzs)):
        ax.bar(
            x[i],
            capacities[i],
            bottom=cumulative[i],
            color=colors[i],
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Add capacity value on bar
        ax.text(
            x[i],
            cumulative[i] + capacities[i] / 2,
            f"{capacities[i]:.0f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )

    # Plot total bar
    ax.bar(
        x[-1],
        capacities[-1],
        color=colors[-1],
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # Add total value on bar
    ax.text(
        x[-1],
        capacities[-1] / 2,
        f"{capacities[-1]:.0f}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )

    ax.set_ylabel("Capacity (MW)")
    ax.set_xlabel("Bidding Zone")
    ax.set_xticks(x)
    ax.set_xticklabels(bzs, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, cumulative[-1] * 1.1)


def get_bidding_zone_colors(n: pypsa.Network) -> dict:
    """
    Create color mapping for bidding zones based on their regional categorization.

    Categories:
    - Core: FBMC Core Capacity Calculation Region (Blue #2E86AB)
    - Virtual CORE: Virtual core bus (Dark Blue #1C5F7A)
    - Nordic: FBMC Nordic Region (Purple #A23B72)
    - Virtual NORDIC: Virtual nordic bus (Dark Purple #7A1C56)
    - Evolved FB: Evolved flow-based virtual buses (Teal #008080)
    - Other: Non-FBMC regions using NTC/AHC (Orange #F18F01)

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network with core_region and/or nordic_region columns in buses

    Returns
    -------
    dict
        Dictionary mapping bus names to hex colors
    """
    colors = {}

    for bus in n.buses.index:
        # Handle virtual cores explicitly by bus name
        if bus == "CORE":
            colors[bus] = "#1C5F7A"  # Dark blue for CORE virtual bus
        elif bus == "NORDIC":
            colors[bus] = "#7A1C56"  # Dark purple for NORDIC virtual bus
        else:
            # Check if this is an evolved FB bus (virtual buses that are not core/nordic)
            country = (
                n.buses.loc[bus, "country"] if "country" in n.buses.columns else None
            )
            is_core = (
                n.buses.loc[bus, "core_region"]
                if "core_region" in n.buses.columns
                else False
            )
            is_nordic = (
                n.buses.loc[bus, "nordic_region"]
                if "nordic_region" in n.buses.columns
                else False
            )

            # Check for evolved FB buses: virtual country, not core, not nordic
            if country == "virtual" and not is_core and not is_nordic:
                colors[bus] = "#008080"  # Teal for Evolved FB buses
            elif is_core:
                colors[bus] = "#2E86AB"  # Blue for Core region
            elif is_nordic:
                colors[bus] = "#A23B72"  # Purple for Nordic region
            else:
                colors[bus] = "#F18F01"  # Orange for Other regions

    return colors


def _bool_region(n: pypsa.Network, column: str) -> pd.Series:
    """
    Return a region membership column as a clean boolean Series.

    Region columns (e.g. ``core_region``) are initialised as ``False`` but
    virtual buses added afterwards leave ``NaN`` entries, which makes the column
    object dtype. ``~`` (bitwise invert) is unsupported on object arrays, so
    coerce to boolean here.

    Parameters
    ----------
    n : pypsa.Network
        Network whose buses are inspected.
    column : str
        Name of the region membership column.

    Returns
    -------
    pd.Series
        Boolean Series indexed by bus name, ``False`` where missing.
    """
    return (
        n.buses.get(column, pd.Series(False, index=n.buses.index))
        .fillna(False)
        .astype(bool)
    )


def add_bidding_zone_legend(ax, n: pypsa.Network) -> None:
    """
    Add legend showing bidding zone color coding.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the legend to
    n : pypsa.Network
        PyPSA network to determine which categories are present
    """
    legend_elements = []

    # Check for virtual cores
    has_core_virtual = "CORE" in n.buses.index
    has_nordic_virtual = "NORDIC" in n.buses.index

    # Check for physical core/nordic regions
    has_core_physical = n.buses.get(
        "core_region", pd.Series(False, index=n.buses.index)
    ).any()
    has_nordic_physical = n.buses.get(
        "nordic_region", pd.Series(False, index=n.buses.index)
    ).any()

    # Check for evolved FB buses (virtual buses that are not core/nordic)
    has_evolved_fb = False
    if "country" in n.buses.columns:
        virtual_buses = n.buses["country"] == "virtual"
        core_buses = _bool_region(n, "core_region")
        nordic_buses = _bool_region(n, "nordic_region")
        evolved_fb_buses = (
            virtual_buses
            & ~core_buses
            & ~nordic_buses
            & ~n.buses.index.isin(["CORE", "NORDIC"])
        )
        has_evolved_fb = evolved_fb_buses.any()

    # Check for other regions
    core_buses = _bool_region(n, "core_region")
    nordic_buses = _bool_region(n, "nordic_region")
    virtual_buses = n.buses.index.isin(["CORE", "NORDIC"])
    evolved_buses = pd.Series(False, index=n.buses.index)
    if "country" in n.buses.columns:
        evolved_buses = (
            (n.buses["country"] == "virtual")
            & ~core_buses
            & ~nordic_buses
            & ~virtual_buses
        )
    has_other = not (core_buses | nordic_buses | virtual_buses | evolved_buses).all()

    # Add legend entries based on what's present
    if has_core_physical:
        legend_elements.append(Patch(facecolor="#2E86AB", label="Core Region"))
    if has_core_virtual:
        legend_elements.append(Patch(facecolor="#1C5F7A", label="CORE Bus (Virtual)"))

    if has_nordic_physical:
        legend_elements.append(Patch(facecolor="#A23B72", label="Nordic Region"))
    if has_nordic_virtual:
        legend_elements.append(Patch(facecolor="#7A1C56", label="NORDIC Bus (Virtual)"))

    if has_evolved_fb:
        legend_elements.append(Patch(facecolor="#008080", label="Evolved FB (Virtual)"))

    if has_other:
        legend_elements.append(Patch(facecolor="#F18F01", label="Other Regions"))

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="lower left",
            bbox_to_anchor=(0, 0),
            fontsize=10,
            framealpha=0.9,
        )


def make_mock_plot(output_filename: str) -> None:
    """
    Generates and saves a mock plot with a specified filename.

    This function creates a mock plot containing a text displaying the provided
    output filename. The figure is saved as an image file with the name specified
    by the `output_filename` parameter.

    Parameters
    ----------
    output_filename : str
        The name of the output file where the plot will be saved.

    """
    plt.figure(figsize=(4, 3))
    plt.text(0.5, 0.5, f"Mock:\n {output_filename}", ha="center", va="center")
    plt.axis("off")
    plt.savefig(output_filename)
    plt.close()


def plot_precondition_with_storage_planning(
    n: pypsa.Network,
    storage_plans: pd.DataFrame,
    hydro: str,
    market_name: str,
    output_filename: str,
    step_hours: int,
) -> None:
    """
    Creates and saves a plot visualizing the state of charge (SOC) of a specified hydro storage
    unit based on pre-optimization data and a particular market results. Uses provided hydro storage
    data to generate the plot, and if the data is unavailable, generates a mock plot instead.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing storage unit data, including state of charge and
        state of charge set values for different hydro units.
    hydro : str
        The identifier for the specific hydro storage unit of interest within the `network` object.
    market_name : str
        The name of the market, used as the label in the plot to describe the type of market
        model results being represented.
    output_filename : str
        The file path where the generated plot (or mock plot) will be saved as an image file.

    Returns
    -------
    None
    """
    if not hydro:
        logger.info(
            f"Missing hydro identifier. Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)
        return

    try:
        soc_market = n.storage_units_t.state_of_charge[hydro]
        soc_set = storage_plans[hydro]
        soc_set = soc_set[soc_set.index.hour == 23].iloc[:: int(step_hours / 24)]
    except (KeyError, TypeError) as e:
        logger.info(
            f"Missing or malformed data for hydro '{hydro}': {e}. "
            f"Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)
        return

    # Make plot
    plt.figure(figsize=(12, 5))
    plt.plot(
        soc_market.index,
        soc_market,
        label=f"{market_name} Model State of Charge",
        linewidth=2,
    )
    plt.scatter(
        soc_set.dropna().index,
        soc_set.dropna(),
        label="State of Charge Set (pre-opt)",
        color="orange",
        zorder=3,
        s=10,
    )
    plt.title(f"State of Charge for {hydro}")
    plt.xlabel("Time")
    plt.ylabel("State of Charge [MWh]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to output file
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_interlinkage_day_ahead_intraday(
    n_da: pypsa.Network, n_id: pypsa.Network, output_filename: str
) -> None:
    """
    Visualize and compare the dispatch of committable generators in day-ahead and intraday networks,
    with improved handling of partial commitments in Linear Unit Commitment (Lin UC) formulation.

    This function processes two PyPSA networks: day-ahead (n_da) and intraday (n_id). It identifies
    committable generators and gathers their dispatch time series. The intraday bounds (p_min_pu, p_max_pu)
    are used directly as they already represent the commitment-adjusted operational constraints.

    Parameters
    ----------
    n_da : pypsa.Network
        The day-ahead network containing generator data and commitment status.
    n_id : pypsa.Network
        The intraday network containing generator data with commitment-adjusted bounds.
    output_filename : str
        The file path where the resulting plot will be saved.

    Returns
    -------
    None
        This function does not return any values.

    Raises
    ------
    None
        This function does not explicitly raise exceptions.

    Notes
    -----
    - If no committable generators are found in the day-ahead network, a mock plot is generated.
    - Intraday bounds (IntraDay_min/max) represent the commitment-adjusted operational constraints.
    - This visualization provides a cleaner representation of operational limits compared to binary commitment multiplication.
    """
    # Get list of committable generators from the day-ahead network
    committable_gens = n_da.generators.index[n_da.generators.committable]

    # Check if list is empty, if so, generate mock plot
    if committable_gens.empty:
        logger.info(
            f"Missing committable generators. Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)
    else:
        # Initialize empty results dictionary
        dispatch_results = {gen: pd.DataFrame() for gen in committable_gens}

        # Loop through networks and collect dispatch time series
        for label, network in [("DayAhead", n_da), ("IntraDay", n_id)]:
            for gen in committable_gens:
                if gen in network.generators_t.p:
                    # Get dispatch series and divide by p_nom
                    p_nom = network.generators.p_nom.get(gen, 0)
                    if p_nom > 0:
                        series = network.generators_t.p[gen].copy() / p_nom
                        series.name = label
                        dispatch_results[gen] = pd.concat(
                            [dispatch_results[gen], series], axis=1
                        )

        # Dynamically add available power limit data
        def add_power_limits_to_results(network, label_prefix=""):
            """Add available power limit time series to dispatch results."""
            # Define power limit attributes to check for
            limit_attrs = {
                "p_min_pu": f"{label_prefix}Commitment (p_min_pu)",
                "p_max_pu": f"{label_prefix}Commitment (p_max_pu)",
            }

            for attr, col_name in limit_attrs.items():
                try:
                    # Use get_switchable_as_dense for robust access to time-varying limits
                    limit_data = network.get_switchable_as_dense("Generator", attr)

                    # Add limits for each committable generator
                    for gen in committable_gens:
                        if gen in limit_data.columns:
                            dispatch_results[gen][col_name] = limit_data[gen]
                except (KeyError, AttributeError):
                    # Attribute doesn't exist in this network
                    continue

        # Add power limits only from day-ahead network (commitment limits)
        add_power_limits_to_results(n_id, "DayAhead_")

        # Plot results
        fig, axes = plt.subplots(
            len(committable_gens),
            1,
            figsize=(12, 3 * len(committable_gens)),
            sharex=True,
        )
        if len(committable_gens) == 1:
            axes = [axes]

        for ax, gen in zip(axes, committable_gens):
            df = dispatch_results[gen]

            if df.empty:
                continue

            # Define plotting styles for different data types
            plot_styles = {
                # Dispatch data - solid lines
                "DayAhead": {
                    "color": "#1f77b4",
                    "linestyle": "-",
                    "linewidth": 1.2,
                    "alpha": 1.0,
                },
                "IntraDay": {
                    "color": "#ff7f0e",
                    "linestyle": "-",
                    "linewidth": 1.2,
                    "alpha": 1.0,
                },
                # Day-ahead commitment limits - dashed lines
                "DayAhead_Commitment (p_min_pu)": {
                    "color": "#2ca02c",
                    "linestyle": "--",
                    "linewidth": 1.0,
                    "alpha": 0.8,
                },
                "DayAhead_Commitment (p_max_pu)": {
                    "color": "#d62728",
                    "linestyle": "--",
                    "linewidth": 1.0,
                    "alpha": 0.8,
                },
            }

            # Define preferred plotting order (dispatch first, then DA commitment limits)
            preferred_order = [
                "DayAhead",
                "IntraDay",
                "DayAhead_Commitment (p_min_pu)",
                "DayAhead_Commitment (p_max_pu)",
            ]

            # Plot available columns in preferred order
            for col in preferred_order:
                if col in df.columns and not df[col].isna().all():
                    style = plot_styles.get(
                        col,
                        {
                            "color": "gray",
                            "linestyle": "-",
                            "linewidth": 1.0,
                            "alpha": 0.7,
                        },
                    )

                    # Create meaningful label
                    if col.startswith("DayAhead_"):
                        label = col  # Keep as is: DayAhead_pmin, DayAhead_pmax
                    else:
                        label = col

                    ax.plot(df.index, df[col], label=label, **style)

            ax.set_ylabel("Dispatch (pu)")
            ax.set_title(f"Dispatch and Commitment Comparison: {gen}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim([0, 1.05])

        plt.xlabel("Time")
        plt.tight_layout()

        # Save to output file
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close()


def plot_network_flows(n: pypsa.Network, output_filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": ccrs.EqualEarth()})

    # Calculate raw average flow (keeping sign for direction)
    average_flow_raw = n.links_t.p0.mean() / n.links.p_nom * 100
    # Use absolute values for visualization properties
    average_flow_abs = average_flow_raw.abs()
    norm = plt.Normalize(vmin=average_flow_abs.min(), vmax=average_flow_abs.max())

    # Calculate dynamic line widths based on absolute flow (1-8 pixel range)
    max_flow = average_flow_abs.max()
    min_width, max_width = 1, 8
    if max_flow > 0:
        dynamic_widths = min_width + (average_flow_abs / max_flow) * (
            max_width - min_width
        )
    else:
        dynamic_widths = pd.Series(min_width, index=average_flow_abs.index)

    # Get bidding zone colors
    bus_colors = get_bidding_zone_colors(n)

    # Plot with raw flow values for correct arrow directions
    n.plot(
        ax=ax,
        # No basemap: PyPSA would otherwise fetch Natural Earth borders and
        # coastlines over the network on first use.
        color_geomap=False,
        bus_sizes=0.05,
        bus_colors=bus_colors,
        link_colors=average_flow_abs,  # Use absolute values for coloring
        link_norm=norm,
        link_cmap="viridis",
        link_widths=dynamic_widths * 0.2,
        flow=pd.concat(
            {"Link": average_flow_raw * 30 / (average_flow_abs + 1e-6)}
        ),  # Scale arrows by 30, add small value to prevent division by zero
    )
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=norm),
        ax=ax,
        label="Line Loading (%)",
        shrink=0.6,
    )

    # Add bidding zone legend
    add_bidding_zone_legend(ax, n)

    plt.title("Average Cross-border Flows", fontsize=14)
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_network_flows_at_one_timestamp(n: pypsa.Network, output_filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": ccrs.EqualEarth()})

    # Get the 12th timestamp
    timestamp = n.snapshots[18]

    # Compute raw flow (keeping sign for direction)
    flow_at_time_raw = n.links_t.p0.loc[timestamp] / n.links.p_nom * 100
    # Use absolute values for visualization properties
    flow_at_time_abs = flow_at_time_raw.abs()
    flow_norm = plt.Normalize(vmin=flow_at_time_abs.min(), vmax=flow_at_time_abs.max())
    prices_at_time = n.buses_t.marginal_price.loc[timestamp]
    # Skip prices for virtual buses
    prices_at_time = prices_at_time[~prices_at_time.index.str.contains("CORE|NORDIC")]

    # Calculate dynamic line widths based on absolute flow (1-8 pixel range)
    max_flow = flow_at_time_abs.max()
    min_width, max_width = 1, 8
    if max_flow > 0:
        dynamic_widths = min_width + (flow_at_time_abs / max_flow) * (
            max_width - min_width
        )
    else:
        dynamic_widths = pd.Series(min_width, index=flow_at_time_abs.index)

    # Get bidding zone colors
    bus_colors = get_bidding_zone_colors(n)

    # Plot the network
    n.plot(
        ax=ax,
        # No basemap: PyPSA would otherwise fetch Natural Earth borders and
        # coastlines over the network on first use.
        color_geomap=False,
        bus_sizes=0.05,
        bus_colors=bus_colors,
        link_colors=flow_at_time_abs,  # Use absolute values for coloring
        link_norm=flow_norm,
        link_cmap="viridis",
        link_widths=dynamic_widths * 0.2,
        flow=pd.concat(
            {"Link": flow_at_time_raw * 30 / (flow_at_time_abs + 1e-6)}
        ),  # Scale arrows by 30, add small value to prevent division by zero
    )

    # Add colorbar
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=flow_norm),
        ax=ax,
        label="Line Loading (%)",
        shrink=0.6,
    )

    # Add bidding zone legend
    add_bidding_zone_legend(ax, n)

    # Add clearing price labels
    for bus, coords in n.buses[["x", "y"]].iterrows():
        price = prices_at_time.get(bus, float("nan"))
        if pd.notna(price):
            ax.text(
                coords["x"],
                coords["y"] - 0.3,
                f"{price:.1f}",
                transform=ccrs.PlateCarree(),
                fontsize=8,
                ha="center",
                va="top",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="red",
                    alpha=0.8,
                    edgecolor="none",
                ),
                zorder=5,
            )
    plt.title(f"Network Flows and Prices at {timestamp}", fontsize=14)
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def get_subplot_grid(n_items: int, max_cols: int = 4):
    """
    Compute (rows, cols) for plotting `n_items` in a grid.
    Limits number of columns to `max_cols`.
    """
    cols = min(n_items, max_cols)
    rows = math.ceil(n_items / cols)
    return rows, cols


def plot_reserve_revenue_validation(n: pypsa.Network, output_filename: str) -> None:
    """
    Plot reserve revenue validation showing both energy and reserves revenue components by bidding zone.

    Creates a grid plot showing both energy revenue and reserves revenue time series, similar to
    NPV reserves implementation, for each bidding zone to validate reserve operations.
    Displays two revenue types that sum to find overall revenue:
    1. Energy revenue (from normal market operations)
    2. Reserves revenue (from providing reserve services)

    Parameters
    ----------
    n : pypsa.Network
        Network with reserves optimization results and reserve variables
    output_filename : str
        Path to save the validation plot
    """
    try:
        # Check if we have reserve variables available
        reserve_vars_gen = []
        reserve_vars_storage = []

        if hasattr(n, "generators_t") and n.generators_t:
            reserve_vars_gen = [
                col for col in n.generators_t.keys() if col.startswith("r_")
            ]

        if hasattr(n, "storage_units_t") and n.storage_units_t:
            reserve_vars_storage = [
                col for col in n.storage_units_t.keys() if col.startswith("r_")
            ]

        logger.info(
            f"Found {len(reserve_vars_gen)} reserve variables in generators_t: {reserve_vars_gen}"
        )
        logger.info(
            f"Found {len(reserve_vars_storage)} reserve variables in storage_units_t: {reserve_vars_storage}"
        )

        if not reserve_vars_gen and not reserve_vars_storage:
            logger.info(
                f"No reserve variables found in network. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Extract reserve types from variables (same as other reserve plots)
        reserve_types = []
        all_reserve_vars = reserve_vars_gen + reserve_vars_storage
        if all_reserve_vars:
            reserve_types = sorted(
                list(
                    set(
                        v.split("r_")[1].split("_")[0]
                        for v in all_reserve_vars
                        if "r_" in v
                    )
                )
            )

        # Select buses with reserve requirements first, then add others (same pattern as other reserve plots)
        buses_with_reserve_reqs = []
        for rtype in reserve_types:
            if f"reserve_{rtype}" in n.buses.columns:
                buses_with_reserve_reqs.extend(
                    n.buses.index[n.buses[f"reserve_{rtype}"].fillna(0) > 0]
                )

        # Remove duplicates and sort
        buses_with_reserve_reqs = sorted(list(set(buses_with_reserve_reqs)))

        # Add remaining buses from network to ensure we get up to 12 bidding zones total
        remaining_buses = [b for b in n.buses.index if b not in buses_with_reserve_reqs]
        selected_buses = (buses_with_reserve_reqs + remaining_buses)[:12]

        # Fallback: if we still don't have enough buses, use all network buses (up to 12)
        if len(selected_buses) < 2:
            logger.warning(
                f"Only {len(selected_buses)} buses selected. Using all network buses as fallback."
            )
            selected_buses = list(n.buses.index)[:12]

        logger.info(f"Detected reserve types: {reserve_types}")
        logger.info(f"Buses with reserve requirements: {buses_with_reserve_reqs}")
        logger.info(f"Total buses in network: {len(n.buses.index)}")
        logger.info(
            f"Selected buses for plotting: {len(selected_buses)} out of {len(n.buses.index)} total"
        )
        logger.info(f"Final selected buses: {selected_buses}")

        if not selected_buses:
            logger.info(
                f"No buses found for plotting. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Calculate energy and reserves revenue by bus (following NPV reserves implementation logic)
        bus_revenues = {}

        for bus in selected_buses:
            # Get components at this bus
            gen_at_bus = n.generators.index[n.generators.bus == bus]
            storage_at_bus = n.storage_units.index[n.storage_units.bus == bus]

            # Calculate energy revenue (normal dispatch * marginal price)
            energy_revenue = pd.Series(0, index=n.snapshots)

            # Get marginal prices for this bus
            prices = pd.Series(0, index=n.snapshots)
            if (
                hasattr(n, "buses_t")
                and hasattr(n.buses_t, "marginal_price")
                and bus in n.buses_t.marginal_price.columns
            ):
                prices = n.buses_t.marginal_price[bus]

            # Energy revenue from generators
            if len(gen_at_bus) > 0 and not n.generators_t.p.empty:
                gen_dispatch = n.generators_t.p[gen_at_bus].sum(axis=1)
                energy_revenue += gen_dispatch * prices

            # Energy revenue from storage units
            if len(storage_at_bus) > 0:
                if not n.storage_units_t.p_dispatch.empty:
                    storage_dispatch = n.storage_units_t.p_dispatch[storage_at_bus].sum(
                        axis=1
                    )
                    energy_revenue += storage_dispatch * prices
                if not n.storage_units_t.p_store.empty:
                    storage_store = n.storage_units_t.p_store[storage_at_bus].sum(
                        axis=1
                    )
                    energy_revenue -= storage_store * prices

            # Calculate reserves revenue (following compute_npv.py logic)
            reserves_revenue = pd.Series(0, index=n.snapshots)

            # Reserve price factor (simplified estimation similar to NPV calculation)
            reserve_price_factor = (
                0.15  # Reserve prices typically higher than energy prices
            )

            # Process generator reserves
            for var_name in reserve_vars_gen:
                if hasattr(n, "generators_t") and var_name in n.generators_t:
                    reserve_data = n.generators_t[var_name]
                    gen_reserves = reserve_data[
                        gen_at_bus.intersection(reserve_data.columns)
                    ]
                    if not gen_reserves.empty:
                        total_gen_reserves = gen_reserves.sum(axis=1)
                        reserves_revenue += (
                            total_gen_reserves * prices * reserve_price_factor
                        )

            # Process storage unit reserves
            for var_name in reserve_vars_storage:
                if hasattr(n, "storage_units_t") and var_name in n.storage_units_t:
                    reserve_data = n.storage_units_t[var_name]
                    storage_reserves = reserve_data[
                        storage_at_bus.intersection(reserve_data.columns)
                    ]
                    if not storage_reserves.empty:
                        total_storage_reserves = storage_reserves.sum(axis=1)
                        reserves_revenue += (
                            total_storage_reserves * prices * reserve_price_factor
                        )

            # Total revenue (same as NPV calculation: energy + reserves)
            total_revenue = energy_revenue + reserves_revenue

            bus_revenues[bus] = {
                "energy_revenue": energy_revenue,
                "reserves_revenue": reserves_revenue,
                "total_revenue": total_revenue,
            }

        # Create grid layout (same as other reserve plots)
        n_buses = len(selected_buses)
        rows, cols = get_subplot_grid(n_buses, max_cols=3)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, bus in enumerate(selected_buses):
            ax = axes[idx]
            revenues = bus_revenues[bus]

            # Plot both revenue types (following NPV reserves implementation)
            ax.plot(
                revenues["energy_revenue"].index,
                revenues["energy_revenue"].values,
                label="Energy Revenue",
                linewidth=2,
                color="blue",
                alpha=0.8,
            )

            ax.plot(
                revenues["reserves_revenue"].index,
                revenues["reserves_revenue"].values,
                label="Reserves Revenue",
                linewidth=2,
                color="green",
                alpha=0.8,
            )

            ax.plot(
                revenues["total_revenue"].index,
                revenues["total_revenue"].values,
                label="Total Revenue",
                linewidth=2,
                color="red",
                linestyle="--",
                alpha=0.9,
            )

            # Calculate and display statistics
            energy_mean = revenues["energy_revenue"].mean()
            reserves_mean = revenues["reserves_revenue"].mean()
            total_mean = revenues["total_revenue"].mean()
            energy_sum = revenues["energy_revenue"].sum()
            reserves_sum = revenues["reserves_revenue"].sum()
            total_sum = revenues["total_revenue"].sum()

            # Position statistics to avoid legend overlap
            ax.text(
                0.02,
                0.98,
                f"Energy: €{energy_mean:.1f}/h (€{energy_sum:.0f})\nReserves: €{reserves_mean:.1f}/h (€{reserves_sum:.0f})\nTotal: €{total_mean:.1f}/h (€{total_sum:.0f})",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                fontsize=7,
            )

            # Format subplot
            ax.set_title(f"BZ: {bus}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Revenue [€/h]", fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

            # Format x-axis for time series
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH_TIME.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(8)

        # Hide unused subplots
        for idx in range(len(selected_buses), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "NPV Reserves Revenue Validation: Energy + Reserves Revenue by Bidding Zone",
            fontsize=14,
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(
            f"NPV reserves revenue validation plot saved to {output_filename} with {len(selected_buses)} bidding zones"
        )

    except Exception as e:
        logger.error(
            f"Error in NPV reserves revenue validation: {e}. Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)


def plot_marginal_price_validation_da_vs_id(
    n_da: pypsa.Network, n_id: pypsa.Network, output_filename: str
) -> None:
    """
    Compare marginal prices between day-ahead and intraday networks by bus.

    This function validates marginal prices by plotting day-ahead vs intraday
    prices for each bus in a grid layout, showing potential price differences
    between market stages.

    Parameters
    ----------
    n_da : pypsa.Network
        Day-ahead network with marginal price results
    n_id : pypsa.Network
        Intraday network with marginal price results
    output_filename : str
        Path to save the validation plot
    """
    try:
        # Get marginal prices from both networks
        if n_da.buses_t.marginal_price.empty or n_id.buses_t.marginal_price.empty:
            logger.info(
                f"Missing marginal price data. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Get common buses between both networks
        common_buses = set(n_da.buses.index) & set(n_id.buses.index)

        if not common_buses:
            logger.info(
                f"No common buses found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Filter to buses that have marginal price data
        da_price_buses = set(n_da.buses_t.marginal_price.columns)
        id_price_buses = set(n_id.buses_t.marginal_price.columns)
        valid_buses = common_buses & da_price_buses & id_price_buses

        if not valid_buses:
            logger.info(
                f"No buses with valid marginal price data. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Sort buses for consistent plotting
        valid_buses = sorted(valid_buses)
        n_buses = len(valid_buses)

        # Calculate grid layout
        rows, cols = get_subplot_grid(n_buses, max_cols=4)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

        if n_buses == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else axes

        for idx, bus in enumerate(valid_buses):
            ax = axes[idx]

            # Get marginal price time series for this bus
            da_prices = n_da.buses_t.marginal_price[bus]
            id_prices = n_id.buses_t.marginal_price[bus]

            # Plot both time series
            ax.plot(
                da_prices.index,
                da_prices.values,
                label="Day-ahead",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.plot(
                id_prices.index,
                id_prices.values,
                label="Intraday",
                linewidth=1.5,
                alpha=0.8,
                linestyle="--",
            )

            # Calculate and display price difference statistics
            if len(da_prices) == len(id_prices):
                price_diff = abs(da_prices - id_prices)
                max_diff = price_diff.max()
                mean_diff = price_diff.mean()
                ax.text(
                    0.02,
                    0.98,
                    f"Max diff: {max_diff:.2f} €/MWh\nMean diff: {mean_diff:.2f} €/MWh",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )

            ax.set_title(f"{bus} Marginal Price Comparison")
            ax.set_ylabel("Price [€/MWh]")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)

        # Hide unused subplots
        if hasattr(axes, "__len__"):
            for idx in range(len(valid_buses), len(axes)):
                axes[idx].set_visible(False)

        plt.suptitle("Marginal Price Validation: Day-ahead vs Intraday", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

    except Exception as e:
        logger.error(
            f"Error in marginal price validation: {e}. Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)


def plot_dispatch_validation_integer_vs_linear(
    n_integer: pypsa.Network, n_linear: pypsa.Network, output_filename: str
) -> None:
    """
    Compare dispatch between integer unit commitment and linearized networks by technology.

    This function validates that the dispatch from both networks is equal for each technology
    by comparing the total dispatch across all generators of the same technology.

    Parameters
    ----------
    n_integer : pypsa.Network
        Network with integer unit commitment results
    n_linear : pypsa.Network
        Network with linearized unit commitment results
    output_filename : str
        Path to save the validation plot
    """
    # Get all generator technologies from both networks
    try:
        gen_techs_integer = n_integer.generators.carrier.unique()
        gen_techs_linear = n_linear.generators.carrier.unique()

        # Find common technologies
        common_techs = set(gen_techs_integer) & set(gen_techs_linear)

        if not common_techs:
            logger.info(
                f"No common technologies found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Compute total dispatch by technology for both networks
        dispatch_by_tech_integer = {}
        dispatch_by_tech_linear = {}

        for tech in common_techs:
            # Integer network dispatch by technology
            gens_integer = n_integer.generators[
                n_integer.generators.carrier == tech
            ].index
            if len(gens_integer) > 0 and tech in [
                n_integer.generators.loc[g, "carrier"] for g in gens_integer
            ]:
                dispatch_integer = n_integer.generators_t.p[gens_integer].sum(axis=1)
                dispatch_by_tech_integer[tech] = dispatch_integer

            # Linear network dispatch by technology
            gens_linear = n_linear.generators[n_linear.generators.carrier == tech].index
            if len(gens_linear) > 0 and tech in [
                n_linear.generators.loc[g, "carrier"] for g in gens_linear
            ]:
                dispatch_linear = n_linear.generators_t.p[gens_linear].sum(axis=1)
                dispatch_by_tech_linear[tech] = dispatch_linear

        # Create comparison plot
        n_techs = len(common_techs)
        if n_techs == 0:
            logger.info(
                f"No valid dispatch data found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        rows, cols = get_subplot_grid(n_techs, max_cols=3)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        if n_techs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else axes

        for idx, tech in enumerate(sorted(common_techs)):
            ax = axes[idx]

            if tech in dispatch_by_tech_integer and tech in dispatch_by_tech_linear:
                integer_data = dispatch_by_tech_integer[tech]
                linear_data = dispatch_by_tech_linear[tech]

                # Plot both time series
                ax.plot(
                    integer_data.index,
                    integer_data.values,
                    label="Integer UC",
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax.plot(
                    linear_data.index,
                    linear_data.values,
                    label="Linear UC",
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle="--",
                )

                # Calculate and display validation metrics
                if len(integer_data) == len(linear_data):
                    max_diff = abs(integer_data - linear_data).max()
                    mean_diff = abs(integer_data - linear_data).mean()
                    ax.text(
                        0.02,
                        0.98,
                        f"Max diff: {max_diff:.2f} MW\nMean diff: {mean_diff:.2f} MW",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    )

                ax.set_title(f"{tech} Technology Dispatch Comparison")
                ax.set_ylabel("Power [MW]")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Format x-axis
                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH.value)
                )
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {tech}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{tech} Technology (No Data)")

        # Hide unused subplots
        if hasattr(axes, "__len__"):
            for idx in range(len(common_techs), len(axes)):
                axes[idx].set_visible(False)

        plt.suptitle(
            "Dispatch Validation: Integer vs Linear Unit Commitment", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

    except Exception as e:
        logger.error(
            f"Error in dispatch validation: {e}. Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)


def plot_forecast_error(
    n_da: pypsa.Network,
    n_id: pypsa.Network,
    output_filename: str,
    component: str,
    time_series_attr: str,
    title_suffix: str,
    filter_carriers: list[str] | None = None,
) -> None:
    """
    Plots the forecast error for a specific network component over time by comparing
    day-ahead and intraday data. Generates a mock plot if no valid data is available
    for the requested component.

    Parameters
    ----------
    n_da : pypsa.Network
        Network object containing the day-ahead data.

    n_id : pypsa.Network
        Network object containing the intraday data.

    output_filename : str
        The full path and filename for saving the resulting plot.

    component : str
        The component of the network to plot (e.g., 'loads', 'generators').

    time_series_attr : str
        The attribute of the component representing the time series data to plot
        (e.g., 'p_nom', 'p_set').

    title_suffix : str
        The suffix to append to each subplot's title for additional context.

    filter_carriers : list[str], optional
        List of carrier names to filter components for plotting. If None, all
        components are included.

    Notes
    -----
    - The function will check for the consistency between 'n_da' and 'n_id' for the
      specified component and apply the carrier filter, if provided. An inconsistency or
      missing data results in generating a mock plot.
    - Subplots are dynamically arranged based on the number of items to display,
      with a shared y-axis and a time-based x-axis formatter.
    """

    # Get component dataframes
    comp_da = getattr(n_da, component)
    comp_id = getattr(n_id, component)

    # Filter by carriers if specified
    if filter_carriers:
        items = comp_da[comp_da.carrier.isin(filter_carriers)].index
        items_id = comp_id[comp_id.carrier.isin(filter_carriers)].index
        filter_desc = f"carriers={filter_carriers}"
    else:
        items = comp_da.index
        items_id = comp_id.index
        filter_desc = "all"

    if items.empty or items_id.empty or not items.equals(items_id):
        logger.info(
            f"Missing or wrong data for {component} ({filter_desc}). "
            f"DA items: {len(items)}, ID items: {len(items_id)}. "
            f"Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)
        return

    # Get time series DataFrames
    da_ts_df = getattr(getattr(n_da, f"{component}_t"), time_series_attr)
    id_ts_df = getattr(getattr(n_id, f"{component}_t"), time_series_attr)

    # Filter to only items that exist in both time series DataFrames
    items = items[items.isin(da_ts_df.columns) & items.isin(id_ts_df.columns)]

    if items.empty:
        logger.info(
            f"No items with time series data '{time_series_attr}' found for {component} ({filter_desc}). "
            f"Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)
        return

    # Compute subplot layout
    rows, cols = get_subplot_grid(len(items))
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axs = axs.flatten() if isinstance(axs, (list | np.ndarray)) else [axs]

    for idx, item in enumerate(items):
        da_ts = da_ts_df[item]
        id_ts = id_ts_df[item]
        axs[idx].plot(da_ts.index, da_ts.values, label="Day-Ahead")
        axs[idx].plot(id_ts.index, id_ts.values, label="Intraday")
        axs[idx].set_title(f"{item} {title_suffix}")
        axs[idx].set_xlabel("Time")
        axs[idx].legend()
        axs[idx].xaxis.set_major_formatter(
            mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH.value)
        )
        for label in axs[idx].get_xticklabels():
            label.set_rotation(45)

    # Hide unused axes if any
    for ax in axs[len(items) :]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_slack_results_interlinkage_soft_constraints(
    n_id: pypsa.Network, output_filename: str
) -> None:
    """
    Plot slack variables from intraday network with soft constraints in a 2x2 grid.

    This function creates a 2x2 grid plot showing up to 4 types of slack variables
    (upper, lower, ramp-up, ramp-down) from the intraday network when using soft
    constraints. Only series with at least one positive value are included to
    avoid cluttering the plot.

    Parameters
    ----------
    n_id : pypsa.Network
        Intraday network with soft constraints and slack variables
    output_filename : str
        Path to save the slack variables plot
    """
    # Define the 4 types of slack variables to look for
    slack_types = {
        "fix-p-upper-slack": "Upper Power Limit Slack",
        "fix-p-lower-slack": "Lower Power Limit Slack",
        "fix-p-ramp_limit_up-slack": "Ramp-Up Limit Slack",
        "fix-p-ramp_limit_down-slack": "Ramp-Down Limit Slack",
    }

    # Debug: Print what's available in the network

    # Extract slack variables from the model
    slack_data = {}

    for slack_var_name, display_name in slack_types.items():
        # slack_data[display_name] = n_id.generators_t.get(slack_var_name, pd.DataFrame())

        slack_df = n_id.generators_t.get(slack_var_name, pd.DataFrame())

        # Only include series that have at least one positive value
        series_with_positive_values = {}
        for col in slack_df.columns:
            max_val = slack_df[col].max()
            if max_val > 0:
                series_with_positive_values[col] = slack_df[col]

        if series_with_positive_values:
            slack_data[display_name] = series_with_positive_values

    # Check if we have any slack variables with positive values
    if not slack_data:
        extra_title = " - All slack variables are zero"
    else:
        extra_title = ""

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot each slack type
    plot_idx = 0
    for slack_type, series_dict in slack_data.items():
        if plot_idx >= 4:  # Only plot up to 4 types
            break

        ax = axes[plot_idx]

        # Plot each generator's slack series
        for gen_name, series in series_dict.items():
            ax.plot(
                series.index, series.values, label=gen_name, linewidth=1.5, alpha=0.8
            )

        ax.set_title(slack_type, fontsize=12, fontweight="bold")
        ax.set_ylabel("Slack Value [MW]", fontsize=10)
        ax.set_xlabel("Time", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        # Format x-axis
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH_TIME.value)
        )
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_fontsize(8)

        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, 4):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Slack Variables - Soft Constraints Interlinkage{extra_title}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close()


def plot_soc_validation_vs_dispatch(
    n: pypsa.Network,
    output_filename: str,
    storage_unit_name: str | None = None,
    validation_zones: list | None = None,
) -> None:
    """
    Plot SoC validation showing SoC, charging, discharging, and initial SoC components.

    This function creates enhanced validation plots to verify State of Charge (SoC) behavior
    using a 3xN panel layout with technology-based columns for clear visualization of storage
    operation. Each column represents a different storage technology (battery, H2, etc.).

    This addresses ticket #357 by validating SoC continuity and identifying
    discontinuities or misalignments in rolling horizon optimization.

    Parameters
    ----------
    n : pypsa.Network
        Network with solved storage units containing SoC and dispatch results
    output_filename : str
        Path to save the validation plot
    storage_unit_name : str, optional
        Specific storage unit to plot. If None, plots selected zones from validation_zones
    validation_zones : list, optional
        List of bidding zone names (e.g., ['BE00', 'DE00']) to plot for multi-zone validation.
        All storage technologies in these zones will be included. If None, uses all available
        storage units grouped by their technology types
    """
    try:
        # Determine which storage units to plot
        if storage_unit_name and storage_unit_name in n.storage_units.index:
            # Single unit plotting (backward compatibility)
            units_to_plot = [storage_unit_name]
        elif validation_zones:
            # Use specified validation zones - find all storage units in those zones
            units_to_plot = []
            for zone in validation_zones:
                # Find all storage units that belong to this zone
                zone_units = [
                    unit
                    for unit in n.storage_units.index
                    if unit.startswith(zone) or unit.split()[0] == zone
                ]
                units_to_plot.extend(zone_units)
        else:
            # Auto-select ALL storage units when nothing is specified
            available_units = []
            for candidate in n.storage_units.index:
                if candidate in n.storage_units_t.state_of_charge.columns:
                    if n.storage_units_t.state_of_charge[candidate].sum() != 0:
                        available_units.append(candidate)

            units_to_plot = available_units

        if not units_to_plot:
            logger.info(
                f"No storage units found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Group storage units by technology type across all zones
        tech_groups = {}
        zone_list = set()

        for unit in units_to_plot:
            # Extract bidding zone from unit name (e.g., "BE00 battery" -> "BE00")
            zone = unit.split()[0] if " " in unit else unit[:4]
            zone_list.add(zone)

            # Determine technology type from unit name
            if "battery" in unit.lower():
                tech = "battery"
            elif "H2" in unit or "hydrogen" in unit.lower():
                tech = "H2"
            elif "PHS" in unit or "pumped" in unit.lower():
                tech = "PHS"
            elif "hydro" in unit.lower():
                tech = "hydro"
            else:
                # Use the part after the zone as technology name
                tech = " ".join(unit.split()[1:]) if " " in unit else "storage"

            if tech not in tech_groups:
                tech_groups[tech] = []
            tech_groups[tech].append(unit)

        # Filter out empty technology groups
        tech_groups = {tech: units for tech, units in tech_groups.items() if units}

        if not tech_groups:
            logger.info(
                f"No storage units with recognizable technology found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Create dynamic layout: 3 rows, N columns (one per technology)
        num_techs = len(tech_groups)
        tech_names = list(tech_groups.keys())
        zone_str = ", ".join(sorted(zone_list))

        # Calculate optimal figure size: wider for more technologies, but keep reasonable proportions
        fig_width = max(8, min(5 * num_techs, 20))  # Min 8", max 20"
        fig, axes = plt.subplots(3, num_techs, figsize=(fig_width, 12), sharex=True)

        # Handle single technology case (axes is 1D instead of 2D)
        if num_techs == 1:
            axes = axes.reshape(3, 1)

        # Define distinct colors for different technologies
        tech_colors = {
            "battery": "blue",
            "H2": "green",
            "PHS": "purple",
            "hydro": "brown",
            "storage": "orange",
        }

        try:
            # Panel 1: State of Charge Evolution - by technology
            for tech_idx, (tech, units) in enumerate(tech_groups.items()):
                ax1 = axes[0, tech_idx]
                color = tech_colors.get(tech, f"C{tech_idx}")

                # Plot first unit of this technology (representative)
                for unit in units[:1]:  # Only plot first unit for clarity
                    if unit in n.storage_units_t.state_of_charge.columns:
                        actual_soc = n.storage_units_t.state_of_charge[unit]
                        initial_soc = (
                            n.storage_units.loc[unit, "state_of_charge_initial"]
                            if "state_of_charge_initial" in n.storage_units.columns
                            else 0.0
                        )

                        # Plot SoC time series
                        ax1.plot(
                            actual_soc.index,
                            actual_soc.values,
                            label=f"{tech} SoC",
                            linewidth=2.5,
                            color=color,
                            alpha=0.9,
                        )

                        # Plot initial SoC reference
                        ax1.axhline(
                            y=initial_soc,
                            color=color,
                            linestyle=":",
                            linewidth=1.5,
                            alpha=0.7,
                            label=f"{tech} Initial ({initial_soc:.0f} MWh)",
                        )

                # Add rolling horizon boundaries
                if hasattr(n, "snapshots") and len(n.snapshots) > 72:
                    for boundary_idx in range(72, len(n.snapshots), 72):
                        if boundary_idx < len(n.snapshots):
                            boundary_time = n.snapshots[boundary_idx]
                            ax1.axvline(
                                x=boundary_time,
                                color="orange",
                                linestyle="--",
                                alpha=0.6,
                                linewidth=1.5,
                                label="Rolling Horizon Boundary"
                                if boundary_idx == 72 and tech_idx == 0
                                else "",
                            )

                ax1.set_title(
                    f"{zone_str} - {tech} - State of Charge Evolution",
                    fontweight="bold",
                    fontsize=12,
                )
                ax1.set_ylabel("State of Charge [MWh]", fontweight="bold", fontsize=10)
                ax1.legend(loc="best", fontsize=8, framealpha=0.9, ncol=1)
                ax1.grid(True, alpha=0.3)

            # Panel 2: Charging and Discharging Power - by technology
            for tech_idx, (tech, units) in enumerate(tech_groups.items()):
                ax2 = axes[1, tech_idx]
                color = tech_colors.get(tech, f"C{tech_idx}")

                # Plot first unit of this technology (representative)
                for unit in units[:1]:  # Only plot first unit for clarity
                    if unit in n.storage_units_t.p_store.columns:
                        p_store = n.storage_units_t.p_store[unit]
                        p_dispatch = n.storage_units_t.p_dispatch[unit]

                        # Plot charging (positive) and discharging (negative for visual clarity)
                        ax2.plot(
                            p_store.index,
                            p_store.values,
                            label=f"{tech} Charging",
                            linewidth=2,
                            color=color,
                            alpha=0.8,
                        )
                        ax2.plot(
                            p_dispatch.index,
                            -p_dispatch.values,  # Negative for visual separation
                            label=f"{tech} Discharging",
                            linewidth=2,
                            color=color,
                            alpha=0.8,
                            linestyle="--",
                        )

                # Add zero reference line
                ax2.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.7)

                ax2.set_title(
                    f"{zone_str} - {tech} - Charging and Discharging Power",
                    fontweight="bold",
                    fontsize=12,
                )
                ax2.set_ylabel("Power [MW]", fontweight="bold", fontsize=10)
                ax2.legend(loc="best", fontsize=8, framealpha=0.9, ncol=1)
                ax2.grid(True, alpha=0.3)

            # Panel 3: Energy Balance Validation - by technology
            for tech_idx, (tech, units) in enumerate(tech_groups.items()):
                ax3 = axes[2, tech_idx]
                color = tech_colors.get(tech, f"C{tech_idx}")

                # Plot first unit of this technology (representative)
                for unit in units[:1]:  # Only plot first unit for clarity
                    if unit in n.storage_units_t.state_of_charge.columns:
                        actual_soc = n.storage_units_t.state_of_charge[unit]
                        p_store = n.storage_units_t.p_store[unit]
                        p_dispatch = n.storage_units_t.p_dispatch[unit]

                        # Get efficiency and initial values
                        eff_store = n.storage_units.loc[unit, "efficiency_store"]
                        eff_dispatch = n.storage_units.loc[unit, "efficiency_dispatch"]
                        initial_soc = (
                            n.storage_units.loc[unit, "state_of_charge_initial"]
                            if "state_of_charge_initial" in n.storage_units.columns
                            else 0.0
                        )

                        # Calculate expected SoC from cumulative dispatch
                        net_charging = p_store * eff_store - p_dispatch / eff_dispatch
                        time_diff = (
                            n.snapshots[1] - n.snapshots[0]
                        ).total_seconds() / 3600
                        expected_soc = initial_soc + (net_charging * time_diff).cumsum()

                        # Plot actual vs expected SoC
                        ax3.plot(
                            actual_soc.index,
                            actual_soc.values,
                            label=f"{tech} Actual SoC",
                            linewidth=2.5,
                            color=color,
                            alpha=0.9,
                        )
                        ax3.plot(
                            expected_soc.index,
                            expected_soc.values,
                            label=f"{tech} Expected SoC",
                            linewidth=2,
                            color=color,
                            alpha=0.7,
                            linestyle=":",
                        )

                ax3.set_title(
                    f"{zone_str} - {tech} - Energy Balance Validation",
                    fontweight="bold",
                    fontsize=12,
                )
                ax3.set_ylabel("State of Charge [MWh]", fontweight="bold", fontsize=10)
                ax3.set_xlabel("Time", fontweight="bold", fontsize=10)
                ax3.legend(loc="best", fontsize=8, framealpha=0.9, ncol=1)
                ax3.grid(True, alpha=0.3)

            # Format x-axis for all panels
            for row in range(3):
                for col in range(num_techs):
                    ax = axes[row, col]
                    ax.xaxis.set_major_formatter(
                        mdates.DateFormatter(DateFormat.PLOT_MONTH_DAY_TIME.value)
                    )
                    for label in ax.get_xticklabels():
                        label.set_rotation(45)
                        label.set_fontsize(9)

            # Overall title with technology breakdown
            tech_list = ", ".join(tech_names)
            plt.suptitle(
                f"Storage SoC Validation: {zone_str}\nTechnologies: {tech_list}",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )

        except Exception as e:
            logger.warning(f"Error plotting storage units {units_to_plot}: {e}")
            # Put error message on the first panel
            axes[0, 0].text(
                0.5,
                0.5,
                f"Error plotting units: {e}",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
                fontsize=12,
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for suptitle
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

    except Exception as e:
        logger.error(
            f"Error in SoC validation: {e}. Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)


def plot_soc_cumsum_validation_by_technology(
    n: pypsa.Network, output_filename: str, validation_zones: list | None = None
) -> None:
    """
    Create SOC vs dispatch cumsum plots by storage technology in a side-by-side layout.

    This function creates a single plot with columns for each storage technology (H2, Battery)
    showing the State of Charge compared to the cumulative dispatch sum. This addresses
    ticket #357 by validating SOC continuity and identifying discontinuities in rolling
    horizon optimization.

    Parameters
    ----------
    n : pypsa.Network
        Network with solved storage units containing SOC and dispatch results
    output_filename : str
        Path for saving the validation plot
    validation_zones : list, optional
        List of bidding zone names (e.g., ['BE00']) to validate. If None, uses all zones.
    """
    try:
        # Determine which zones to validate
        if validation_zones:
            zones_to_check = validation_zones
        else:
            # Extract unique zones from storage unit names
            zones_to_check = list(
                set(
                    [
                        unit.split()[0] if " " in unit else unit[:4]
                        for unit in n.storage_units.index
                    ]
                )
            )

        # Group storage units by technology across all zones
        tech_units = {}
        zone_list = set()

        for zone in zones_to_check:
            zone_units = [
                unit
                for unit in n.storage_units.index
                if unit.startswith(zone) or unit.split()[0] == zone
            ]
            zone_list.add(zone)

            for unit in zone_units:
                # Determine technology type
                if "H2" in unit or "hydrogen" in unit.lower():
                    tech = "H2"
                elif "battery" in unit.lower():
                    tech = "battery"
                elif "PHS" in unit or "pumped" in unit.lower():
                    tech = "PHS"
                elif "hydro" in unit.lower():
                    tech = "hydro"
                else:
                    # Use the part after the zone as technology name
                    tech = " ".join(unit.split()[1:]) if " " in unit else "storage"

                if tech not in tech_units:
                    tech_units[tech] = []
                tech_units[tech].append(unit)

        # Filter out empty technology groups and keep only those with valid SOC data
        valid_tech_units = {}
        for tech, units in tech_units.items():
            for unit in units:
                if (
                    unit in n.storage_units_t.state_of_charge.columns
                    and n.storage_units_t.state_of_charge[unit].sum() != 0
                ):
                    valid_tech_units[tech] = unit
                    break  # Take first valid unit per technology

        if not valid_tech_units:
            logger.info(
                f"No valid storage units found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Create side-by-side layout: 1 row, N columns (one per technology)
        num_techs = len(valid_tech_units)
        tech_names = list(valid_tech_units.keys())
        zone_str = ", ".join(sorted(zone_list))

        # Calculate optimal figure size
        fig_width = max(8, min(6 * num_techs, 20))  # Min 8", max 20"
        fig, axes = plt.subplots(1, num_techs, figsize=(fig_width, 6), sharex=True)

        # Handle single technology case
        if num_techs == 1:
            axes = [axes]

        # Define distinct colors for different technologies
        tech_colors = {
            "battery": "blue",
            "H2": "green",
            "PHS": "purple",
            "hydro": "brown",
            "storage": "orange",
        }

        try:
            for tech_idx, (tech, storage_unit) in enumerate(valid_tech_units.items()):
                ax = axes[tech_idx]
                color = tech_colors.get(tech, f"C{tech_idx}")

                if storage_unit not in n.storage_units_t.state_of_charge.columns:
                    continue

                # Get optimization results
                actual_soc = n.storage_units_t.state_of_charge[storage_unit]
                p_store = n.storage_units_t.p_store[storage_unit]
                p_dispatch = n.storage_units_t.p_dispatch[storage_unit]

                # Get efficiency and initial values
                eff_store = n.storage_units.loc[storage_unit, "efficiency_store"]
                eff_dispatch = n.storage_units.loc[storage_unit, "efficiency_dispatch"]
                initial_soc = (
                    n.storage_units.loc[storage_unit, "state_of_charge_initial"]
                    if "state_of_charge_initial" in n.storage_units.columns
                    else 0.0
                )

                # Calculate dispatch cumsum
                net_dispatch = p_store * eff_store - p_dispatch / eff_dispatch
                time_diff = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600
                dispatch_cumsum = initial_soc + (net_dispatch * time_diff).cumsum()

                # Plot dispatch cumsum and actual SOC
                ax.plot(
                    dispatch_cumsum.index,
                    dispatch_cumsum.values,
                    label=f"{tech} dispatch cumsum",
                    linewidth=2.5,
                    color=color,
                    alpha=0.9,
                )
                ax.plot(
                    actual_soc.index,
                    actual_soc.values,
                    label=f"{tech} variable SoC",
                    linewidth=2.5,
                    color=color,
                    alpha=0.7,
                    linestyle="--",
                )

                # Add rolling horizon boundaries if available
                if hasattr(n, "snapshots") and len(n.snapshots) > 72:
                    for boundary_idx in range(
                        72, len(n.snapshots), 48
                    ):  # Every 48h with 24h overlap
                        if boundary_idx < len(n.snapshots):
                            boundary_time = n.snapshots[boundary_idx]
                            ax.axvline(
                                x=boundary_time,
                                color="red",
                                linestyle=":",
                                alpha=0.6,
                                linewidth=1.5,
                                label="Rolling Horizon Boundary"
                                if boundary_idx == 72 and tech_idx == 0
                                else "",
                            )

                # Formatting
                ax.set_title(
                    f"{tech} - SoC vs Dispatch Cumsum", fontsize=12, fontweight="bold"
                )
                ax.set_xlabel("Time", fontsize=10, fontweight="bold")
                ax.set_ylabel("Energy [MWh]", fontsize=10, fontweight="bold")
                ax.legend(fontsize=9, loc="best", framealpha=0.9)
                ax.grid(True, alpha=0.3)

                # Format x-axis
                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter(DateFormat.PLOT_MONTH_DAY.value)
                )
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_fontsize(9)

            # Overall title with zone and technology information
            tech_list = ", ".join(tech_names)
            plt.suptitle(
                f"Storage SoC vs Dispatch Cumsum Validation: {zone_str}\nTechnologies: {tech_list}",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)  # Make room for suptitle
            plt.savefig(output_filename, bbox_inches="tight", dpi=300)
            plt.close()

            logger.info(f"Created SOC cumsum validation plot: {output_filename}")

        except Exception as e:
            logger.warning(f"Error plotting storage units: {e}")
            make_mock_plot(output_filename)

    except Exception as e:
        logger.error(
            f"Error in SOC cumsum validation: {e}. Mock plot generated for {output_filename}."
        )
        make_mock_plot(output_filename)


def plot_reserves_validation_day_ahead(
    n_da: pypsa.Network, output_filename: str
) -> None:
    """
    Plot system-wide day-ahead reserves validation.

    Creates a single plot showing:
    - Total system dispatch
    - Total system reserves provided (upward and downward)
    - Total system reserve requirements
    - Visual indicators when requirements are not met

    Parameters
    ----------
    n_da : pypsa.Network
        Day-ahead network with reserves results
    output_filename : str
        Path to save the validation plot
    """
    try:
        # Check if reserves variables are available
        gen_reserve_vars = []
        storage_reserve_vars = []

        if hasattr(n_da, "generators_t"):
            gen_reserve_vars = [col for col in n_da.generators_t.keys() if "r_" in col]
        if hasattr(n_da, "storage_units_t"):
            storage_reserve_vars = [
                col for col in n_da.storage_units_t.keys() if "r_" in col
            ]

        if not gen_reserve_vars and not storage_reserve_vars:
            logger.info(
                f"No reserves variables found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Extract reserve types
        all_reserve_vars = gen_reserve_vars + storage_reserve_vars
        reserve_types = sorted(
            list(
                set(
                    v.split("r_")[1].split("_")[0]
                    for v in all_reserve_vars
                    if "r_" in v
                )
            )
        )

        # Calculate system-wide dispatch
        total_dispatch = pd.Series(0, index=n_da.snapshots, dtype=float)

        # Add generator dispatch
        if hasattr(n_da, "generators_t") and "p" in n_da.generators_t:
            total_dispatch += n_da.generators_t.p.sum(axis=1)

        # Add storage dispatch (dispatch - store)
        if hasattr(n_da, "storage_units_t"):
            if "p_dispatch" in n_da.storage_units_t:
                total_dispatch += n_da.storage_units_t.p_dispatch.sum(axis=1)
            if "p_store" in n_da.storage_units_t:
                total_dispatch -= n_da.storage_units_t.p_store.sum(axis=1)

        # Calculate system-wide reserves provided
        up_reserves_provided = pd.Series(0, index=n_da.snapshots, dtype=float)
        down_reserves_provided = pd.Series(0, index=n_da.snapshots, dtype=float)

        # Add up all upward reserve types from generators
        if hasattr(n_da, "generators_t"):
            for rtype in reserve_types:
                var_name_up = f"r_{rtype}_up"
                var_name_down = f"r_{rtype}_down"

                if var_name_up in n_da.generators_t:
                    up_reserves_provided += n_da.generators_t[var_name_up].sum(axis=1)
                if var_name_down in n_da.generators_t:
                    down_reserves_provided += n_da.generators_t[var_name_down].sum(
                        axis=1
                    )

        # Add up all reserve types from storage units
        if hasattr(n_da, "storage_units_t"):
            for rtype in reserve_types:
                var_name_up = f"r_{rtype}_up"
                var_name_down = f"r_{rtype}_down"

                if var_name_up in n_da.storage_units_t:
                    up_reserves_provided += n_da.storage_units_t[var_name_up].sum(
                        axis=1
                    )
                if var_name_down in n_da.storage_units_t:
                    down_reserves_provided += n_da.storage_units_t[var_name_down].sum(
                        axis=1
                    )

        # Calculate system-wide reserve requirements
        up_requirement = 0
        down_requirement = 0

        for rtype in reserve_types:
            if f"reserve_{rtype}" in n_da.buses.columns:
                # Sum requirements across all buses
                req_sum = n_da.buses[f"reserve_{rtype}"].fillna(0).sum()
                up_requirement += req_sum
                down_requirement += req_sum  # Assuming symmetric requirements

        # Create single plot for system-wide validation
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot dispatch
        ax.plot(
            total_dispatch.index,
            total_dispatch.values,
            label="Total System Dispatch",
            linewidth=2.5,
            color="blue",
        )

        # Plot dispatch + upward reserves
        dispatch_plus_up = total_dispatch + up_reserves_provided
        ax.plot(
            dispatch_plus_up.index,
            dispatch_plus_up.values,
            label=f"Dispatch + Up Reserves (Provided: {up_reserves_provided.mean():.1f} MW avg)",
            linewidth=2,
            color="green",
            linestyle="-",
            alpha=0.8,
        )

        # Plot dispatch - downward reserves
        dispatch_minus_down = total_dispatch - down_reserves_provided
        ax.plot(
            dispatch_minus_down.index,
            dispatch_minus_down.values,
            label=f"Dispatch - Down Reserves (Provided: {down_reserves_provided.mean():.1f} MW avg)",
            linewidth=2,
            color="orange",
            linestyle="-",
            alpha=0.8,
        )

        # Add requirement lines if requirements exist
        if up_requirement > 0:
            # Plot required upward reserve level
            required_up_line = total_dispatch + up_requirement
            ax.plot(
                required_up_line.index,
                required_up_line.values,
                label=f"Required Up Reserve Level ({up_requirement:.1f} MW)",
                linewidth=1.5,
                color="darkgreen",
                linestyle="--",
                alpha=0.7,
            )

            # Check for violations
            violations_up = up_reserves_provided < up_requirement
            if violations_up.any():
                violation_times = up_reserves_provided.index[violations_up]
                for t in violation_times:
                    ax.axvspan(
                        t, t + pd.Timedelta(hours=1), alpha=0.15, color="red", zorder=0
                    )

        if down_requirement > 0:
            # Plot required downward reserve level
            required_down_line = total_dispatch - down_requirement
            ax.plot(
                required_down_line.index,
                required_down_line.values,
                label=f"Required Down Reserve Level ({down_requirement:.1f} MW)",
                linewidth=1.5,
                color="darkorange",
                linestyle="--",
                alpha=0.7,
            )

            # Check for violations
            violations_down = down_reserves_provided < down_requirement
            if violations_down.any():
                violation_times = down_reserves_provided.index[violations_down]
                for t in violation_times:
                    ax.axvspan(
                        t,
                        t + pd.Timedelta(hours=1),
                        alpha=0.1,
                        color="orange",
                        zorder=0,
                    )

        # Add compliance indicators
        total_violations = 0
        violation_text = []

        if up_requirement > 0:
            up_violations = (up_reserves_provided < up_requirement).sum()
            total_violations += up_violations
            if up_violations > 0:
                violation_text.append(f"↑ {up_violations} hours")

        if down_requirement > 0:
            down_violations = (down_reserves_provided < down_requirement).sum()
            total_violations += down_violations
            if down_violations > 0:
                violation_text.append(f"↓ {down_violations} hours")

        # Add status text
        if total_violations > 0:
            status_text = f"⚠ Violations: {', '.join(violation_text)}"
            status_color = "red"
        else:
            status_text = "✓ All reserve requirements met"
            status_color = "green"

        ax.text(
            0.02,
            0.98,
            status_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=status_color,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

        # Add summary statistics box
        stats_text = (
            f"Reserve Types: {', '.join(reserve_types)}\n"
            f"Up Req: {up_requirement:.1f} MW | Down Req: {down_requirement:.1f} MW\n"
            f"Up Provided (avg): {up_reserves_provided.mean():.1f} MW\n"
            f"Down Provided (avg): {down_reserves_provided.mean():.1f} MW"
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3),
        )

        # Format plot
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Power [MW]", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=10)

        # Format x-axis
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH_TIME.value)
        )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Set title with compliance summary
        if total_violations > 0:
            title = f"System-Wide Reserve Validation - ⚠ {total_violations} Total Violation Hours"
            title_color = "darkred"
        else:
            title = "System-Wide Reserve Validation - ✓ All Requirements Met"
            title_color = "darkgreen"

        ax.set_title(title, fontsize=14, fontweight="bold", color=title_color, pad=20)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"System-wide reserves validation plot saved to {output_filename}")
        if total_violations > 0:
            logger.warning(
                f"Found {total_violations} hours with reserve requirement violations"
            )

    except Exception as e:
        logger.error(f"Error creating reserves validation plot: {e}")
        make_mock_plot(output_filename)


def plot_reserves_interlinkage_p_limits(
    n_da: pypsa.Network, n_id: pypsa.Network, output_filename: str
) -> None:
    """
    Plot comparison of p_max_pu and p_min_pu values between day-ahead and intraday networks
    for generators providing reserves.

    This function identifies generators that have reserves in the day-ahead network
    (where r_FCR_up, r_FCR_down, r_FRR_up, r_FRR_down are present) and compares their
    p_max_pu and p_min_pu values between the day-ahead and intraday networks.

    Parameters
    ----------
    n_da : pypsa.Network
        Day-ahead network containing reserve data
    n_id : pypsa.Network
        Intraday network with potentially modified p_max_pu/p_min_pu values
    output_filename : str
        Path to save the validation plot

    Returns
    -------
    None
        This function does not return any values.

    Notes
    -----
    - If no generators with reserves are found, a mock plot is generated
    - The plot shows both p_max_pu and p_min_pu values for comparison
    - Day-ahead values are shown as solid lines, intraday values as dashed lines
    """
    try:
        # Get reserve types dynamically from bus columns
        bus_reserve_cols = [
            col for col in n_da.buses.columns if col.startswith("reserve_")
        ]
        reserve_type_names = [col.replace("reserve_", "") for col in bus_reserve_cols]

        # Build reserve variable names for both up and down directions
        reserve_types = []
        for rtype in reserve_type_names:
            reserve_types.extend([f"r_{rtype}_up", f"r_{rtype}_down"])

        # Find generators with reserves in day-ahead network
        reserve_gens = set()
        for reserve_type in reserve_types:
            if reserve_type in n_da.generators_t:
                reserve_gens.update(n_da.generators_t[reserve_type].columns)

        if not reserve_gens:
            logger.info(
                f"No generators with reserves found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Filter to only include generators that exist in both networks
        reserve_gens = [
            gen
            for gen in reserve_gens
            if gen in n_da.generators.index and gen in n_id.generators.index
        ]

        if not reserve_gens:
            logger.info(
                f"No common generators with reserves found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Calculate blocked reserves for each individual generator
        generator_blocked_reserves = {}

        for gen in reserve_gens:
            # Get p_max_pu and p_min_pu from both networks
            da_p_max = n_da.get_switchable_as_dense("Generator", "p_max_pu")[gen]
            da_p_min = n_da.get_switchable_as_dense("Generator", "p_min_pu")[gen]
            id_p_max = n_id.get_switchable_as_dense("Generator", "p_max_pu")[gen]
            id_p_min = n_id.get_switchable_as_dense("Generator", "p_min_pu")[gen]

            # Get generator p_nom for conversion to MW
            p_nom = n_da.generators.loc[gen, "p_nom"]

            # Calculate deviations (DA - ID) and convert to MW
            up_blocked = (
                da_p_max - id_p_max
            ).mean() * p_nom  # Positive means more blocked in ID
            down_blocked = (
                id_p_min - da_p_min
            ).mean() * p_nom  # Positive means more blocked in ID

            generator_blocked_reserves[gen] = {
                "up_blocked": up_blocked,
                "down_blocked": down_blocked,
            }

        # Limit number of generators to plot (select top generators by total reserves)
        max_gens_to_plot = 12  # Show at most 12 generators

        if len(reserve_gens) > max_gens_to_plot:
            # Calculate total reserves for each generator to select the most relevant ones
            gen_total_reserves = {}
            for gen in reserve_gens:
                total = 0
                for reserve_type in reserve_types:
                    if (
                        reserve_type in n_da.generators_t
                        and gen in n_da.generators_t[reserve_type].columns
                    ):
                        total += n_da.generators_t[reserve_type][gen].sum()
                gen_total_reserves[gen] = total

            # Sort generators by total reserves and take top max_gens_to_plot
            reserve_gens = sorted(
                reserve_gens, key=lambda g: gen_total_reserves[g], reverse=True
            )[:max_gens_to_plot]
            logger.info(
                f"Plotting top {max_gens_to_plot} generators by reserve provision out of {len(gen_total_reserves)} total"
            )

        # Create figure with subplots in grid layout
        n_gens = len(reserve_gens)
        n_cols = min(3, n_gens)  # Use up to 3 columns
        n_rows = (n_gens + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), sharex=True
        )

        # Flatten axes array for easier indexing
        if n_gens == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        # Add default value for i_gen, in case the for loop below is not executed
        i_gen = len(reserve_gens)

        # Plot for each generator
        for i_gen, gen in enumerate(reserve_gens):
            ax = axes[i_gen]

            # Get p_max_pu and p_min_pu from both networks
            # Day-ahead network
            da_p_max_pu = n_da.get_switchable_as_dense("Generator", "p_max_pu")[gen]
            da_p_min_pu = n_da.get_switchable_as_dense("Generator", "p_min_pu")[gen]

            # Intraday network
            id_p_max_pu = n_id.get_switchable_as_dense("Generator", "p_max_pu")[gen]
            id_p_min_pu = n_id.get_switchable_as_dense("Generator", "p_min_pu")[gen]

            # Get generator p_nom for conversion to MW
            p_nom = n_da.generators.loc[gen, "p_nom"]

            # Convert per-unit values to MW
            da_p_max = da_p_max_pu * p_nom
            da_p_min = da_p_min_pu * p_nom
            id_p_max = id_p_max_pu * p_nom
            id_p_min = id_p_min_pu * p_nom

            # Plot p_max in MW
            ax.plot(
                da_p_max.index,
                da_p_max,
                label="DA p_max",
                color="blue",
                linestyle="-",
                linewidth=2,
            )
            ax.plot(
                id_p_max.index,
                id_p_max,
                label="ID p_max",
                color="blue",
                linestyle="--",
                linewidth=2,
            )

            # Plot p_min in MW
            ax.plot(
                da_p_min.index,
                da_p_min,
                label="DA p_min",
                color="red",
                linestyle="-",
                linewidth=2,
            )
            ax.plot(
                id_p_min.index,
                id_p_min,
                label="ID p_min",
                color="red",
                linestyle="--",
                linewidth=2,
            )

            # Add reserve information to the plot
            reserve_info = []
            for reserve_type in reserve_types:
                if (
                    reserve_type in n_da.generators_t
                    and gen in n_da.generators_t[reserve_type].columns
                ):
                    avg_reserve = n_da.generators_t[reserve_type][gen].mean()
                    if avg_reserve > 0:
                        reserve_info.append(f"{reserve_type}: {avg_reserve:.1f} MW")

            # Get blocked reserves for this generator
            up_blocked = generator_blocked_reserves[gen]["up_blocked"]
            down_blocked = generator_blocked_reserves[gen]["down_blocked"]

            # Set labels and title
            ax.set_ylabel("MW")
            title_text = (
                f"Generator: {gen}\n"
                f"Average Reserves DA: {', '.join(reserve_info)}\n"
                f"Average Actual Reserves Blocked ID: Up {up_blocked:.1f} MW, Down {down_blocked:.1f} MW"
            )
            ax.set_title(title_text)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Format x-axis
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(8)

        # Hide unused subplots
        for j in range(i_gen + 1, len(axes)):
            axes[j].set_visible(False)

        # Add common x-label
        fig.text(0.5, 0.02, "Time", ha="center", fontsize=12)

        plt.suptitle(
            "Day-Ahead vs Intraday p_max_pu/p_min_pu Comparison for Generators with Reserves",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Reserves interlinkage p_limits plot saved to {output_filename}")

    except Exception as e:
        logger.error(f"Error creating reserves interlinkage p_limits plot: {e}")
        make_mock_plot(output_filename)


def plot_hydro_constraints_validation(n: pypsa.Network, output_filename: str) -> None:
    """
    Plot validation of hydro constraints (MIN_RES_TECH and MAX_RES_TECH).

    Shows state of charge limits vs actual SoC for storage units with:
    - state_of_charge_max constraints (from MAX_RES_TECH, pre-loaded in build_network_from_database)
    - state_of_charge_min constraints (from MIN_RES_TECH, pre-loaded in build_network_from_database)
    - p_max_pu constraints for pumping (MAX_PUMP)

    Note: Constraints are now loaded once during network building rather than at each validation.

    Parameters
    ----------
    n : pypsa.Network
        Network containing hydro storage units with pre-loaded constraints in n.storage_units_t
    output_filename : str
        Path to save the validation plot

    Returns
    -------
    None
    """
    try:
        # Find hydro storage units
        hydro_units = n.storage_units[
            n.storage_units.carrier.str.contains("hydro|PHS", case=False, na=False)
        ]

        if hydro_units.empty:
            logger.info(
                f"No hydro units found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Check for units with state_of_charge constraints (MIN_HIST/MAX_HIST)
        units_with_soc_constraints = []
        units_with_pump_limits = []

        # Check static constraints
        for unit in hydro_units.index:
            has_constraint = False

            # Check for state_of_charge_max (static)
            if "state_of_charge_max" in n.storage_units.columns:
                if pd.notna(n.storage_units.loc[unit, "state_of_charge_max"]):
                    has_constraint = True

            # Check for state_of_charge_min (static)
            if "state_of_charge_min" in n.storage_units.columns:
                if pd.notna(n.storage_units.loc[unit, "state_of_charge_min"]):
                    has_constraint = True

            if has_constraint:
                units_with_soc_constraints.append(unit)

            # Check for p_max_pu constraints (for pumping)
            if (
                "p_max_pu" in n.storage_units_t
                and unit in n.storage_units_t["p_max_pu"].columns
            ):
                # Check if p_max_pu varies (indicating pumping constraints)
                p_max_values = n.storage_units_t["p_max_pu"][unit]
                if p_max_values.std() > 0.01:  # Some variation indicates constraints
                    units_with_pump_limits.append(unit)

        # Check time-varying constraints
        if "state_of_charge_max" in n.storage_units_t:
            for unit in hydro_units.index:
                if unit in n.storage_units_t["state_of_charge_max"].columns:
                    if unit not in units_with_soc_constraints:
                        units_with_soc_constraints.append(unit)

        if "state_of_charge_min" in n.storage_units_t:
            for unit in hydro_units.index:
                if unit in n.storage_units_t["state_of_charge_min"].columns:
                    if unit not in units_with_soc_constraints:
                        units_with_soc_constraints.append(unit)

        # Create figure with subplots
        n_plots = 0
        if units_with_soc_constraints:
            n_plots += 1
        if units_with_pump_limits:
            n_plots += 1

        if n_plots == 0:
            logger.info(
                f"No hydro units with constraints found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Plot 1: MIN_HIST/MAX_HIST (state_of_charge_min/max) constraints
        if units_with_soc_constraints:
            ax = axes[plot_idx]
            plot_idx += 1

            # Select up to 3 units to plot for clarity
            units_to_plot = units_with_soc_constraints[:3]

            colors = colormaps["Set2"](np.linspace(0, 1, len(units_to_plot)))

            for i, unit in enumerate(units_to_plot):
                color = colors[i]

                # Get actual state of charge
                if (
                    "state_of_charge" in n.storage_units_t
                    and unit in n.storage_units_t["state_of_charge"].columns
                ):
                    soc = n.storage_units_t["state_of_charge"][unit]
                else:
                    # Calculate SoC from dispatch if not available
                    continue

                # Get max constraints
                if (
                    "state_of_charge_max" in n.storage_units_t
                    and unit in n.storage_units_t["state_of_charge_max"].columns
                ):
                    soc_max = n.storage_units_t["state_of_charge_max"][unit]
                elif "state_of_charge_max" in n.storage_units.columns:
                    soc_max = pd.Series(
                        n.storage_units.loc[unit, "state_of_charge_max"],
                        index=soc.index,
                    )
                else:
                    continue

                # Get min constraints if available
                soc_min = None
                if (
                    "state_of_charge_min" in n.storage_units_t
                    and unit in n.storage_units_t["state_of_charge_min"].columns
                ):
                    soc_min = n.storage_units_t["state_of_charge_min"][unit]
                elif "state_of_charge_min" in n.storage_units.columns:
                    if pd.notna(n.storage_units.loc[unit, "state_of_charge_min"]):
                        soc_min = pd.Series(
                            n.storage_units.loc[unit, "state_of_charge_min"],
                            index=soc.index,
                        )

                # Plot actual SoC
                ax.plot(
                    soc.index,
                    soc,
                    label=f"{unit} - Actual SoC",
                    color=color,
                    linewidth=2,
                )

                # Plot max constraint
                ax.plot(
                    soc.index,
                    soc_max,
                    label=f"{unit} - Max Limit",
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )

                # Plot min constraint if available
                if soc_min is not None:
                    ax.plot(
                        soc.index,
                        soc_min,
                        label=f"{unit} - Min Limit",
                        color=color,
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                    )

                # Fill area between constraints
                ax.fill_between(
                    soc.index,
                    soc_min if soc_min is not None else 0,
                    soc_max,
                    color=color,
                    alpha=0.1,
                )

            ax.set_ylabel("State of Charge [MWh]")
            ax.set_title(
                "Hydro Storage Constraints Validation (MIN_HIST/MAX_HIST - State of Charge Limits)"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(8)

        # Plot 2: MAX_PUMP (pumping power constraints)
        if units_with_pump_limits:
            ax = axes[plot_idx]

            # Select up to 3 units to plot for clarity
            units_to_plot = units_with_pump_limits[:3]

            colors = colormaps["Set3"](np.linspace(0, 1, len(units_to_plot)))

            for i, unit in enumerate(units_to_plot):
                color = colors[i]

                # Get p_max_pu constraints
                if (
                    "p_max_pu" in n.storage_units_t
                    and unit in n.storage_units_t["p_max_pu"].columns
                ):
                    p_max_pu = n.storage_units_t["p_max_pu"][unit]
                else:
                    continue

                # Get actual dispatch (negative is pumping)
                if "p" in n.storage_units_t and unit in n.storage_units_t["p"].columns:
                    dispatch = n.storage_units_t["p"][unit]
                else:
                    continue

                # Get p_nom for scaling
                p_nom = n.storage_units.loc[unit, "p_nom"]

                # Convert to MW
                p_max_mw = p_max_pu * p_nom

                # Plot pumping limit (negative of p_max for pumping)
                ax.plot(
                    p_max_pu.index,
                    -p_max_mw,
                    label=f"{unit} - Max Pump Power",
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                )

                # Plot actual pumping (negative values)
                pumping = dispatch.where(dispatch < 0, 0)
                ax.plot(
                    pumping.index,
                    pumping,
                    label=f"{unit} - Actual Pumping",
                    color=color,
                    linewidth=2,
                    alpha=0.7,
                )

                # Fill area to show constraint
                ax.fill_between(p_max_pu.index, 0, -p_max_mw, color=color, alpha=0.1)

            ax.set_ylabel("Pumping Power [MW]")
            ax.set_title("Hydro Pumping Constraints Validation (MAX_PUMP)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower right", fontsize=9)
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(8)

            # Invert y-axis so pumping (negative) appears as positive
            ax.invert_yaxis()

        plt.xlabel("Time")
        plt.suptitle("Hydro Constraints Validation", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Hydro constraints validation plot saved to {output_filename}")

    except Exception as e:
        logger.error(f"Error creating hydro constraints validation plot: {e}")
        make_mock_plot(output_filename)


def plot_must_run_dispatch_validation(n: pypsa.Network, output_filename: str) -> None:
    """
    Plot validation of must-run generator dispatch requirements.

    Creates a grid plot showing must-run generator dispatch vs their minimum
    required dispatch (p_min_pu * p_nom). Each subplot shows one generator
    with actual dispatch and minimum dispatch requirement over time.

    Parameters
    ----------
    n : pypsa.Network
        Network containing generators with is_must_run=True
    output_filename : str
        Path to save the validation plot

    Returns
    -------
    None
    """
    try:
        # Check if is_must_run column exists and identify must-run generators
        if "is_must_run" not in n.generators.columns:
            logger.info(
                f"No is_must_run column found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Get boolean mask and ensure it's boolean type to avoid indexing issues
        is_must_run_mask = n.generators["is_must_run"].astype(bool)

        # Exclude load shedding and spillage generators
        if "carrier" in n.generators.columns:
            exclude_carriers = ["load shedding", "load spillage"]
            is_excluded = n.generators["carrier"].isin(exclude_carriers)
            is_must_run_mask = is_must_run_mask & ~is_excluded

        must_run_gens = n.generators.index[is_must_run_mask]

        if len(must_run_gens) == 0:
            logger.info(
                f"No must-run generators found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Limit to first 25 generators for 5x5 grid
        must_run_gens = must_run_gens[:25]
        n_gens = len(must_run_gens)

        logger.info(f"Plotting validation for {n_gens} must-run generators")

        # Get actual dispatch data
        if "p" not in n.generators_t or n.generators_t.p.empty:
            logger.info(
                f"No dispatch data found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Check if any must-run generators have dispatch data
        common_gens = must_run_gens.intersection(n.generators_t.p.columns)
        if len(common_gens) == 0:
            logger.info(
                f"No must-run generators found in dispatch data. "
                f"Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Use only generators that have dispatch data
        must_run_gens = common_gens[:25]  # Limit to first 25
        n_gens = len(must_run_gens)

        actual_dispatch = n.generators_t.p[must_run_gens]

        # Get must-run capacity (minimum required dispatch)
        if "must-run capacity" not in n.generators.columns:
            logger.info(
                f"No 'must-run capacity' column found. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        must_run_capacity = n.generators.loc[must_run_gens, "must-run capacity"]

        # Check if must-run capacity can be timeseries or if it's static
        # If it's in generators_t, use that, otherwise broadcast the static value
        if "must-run capacity" in n.generators_t:
            min_required_dispatch = n.generators_t["must-run capacity"][must_run_gens]
        else:
            # Create a dataframe with the static must-run capacity repeated for all timesteps
            min_required_dispatch = pd.DataFrame(
                index=actual_dispatch.index,
                columns=must_run_gens,
                data={gen: must_run_capacity[gen] for gen in must_run_gens},
            )

        # Create subplot grid (max 5x5)
        rows, cols = get_subplot_grid(n_gens, max_cols=5)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))

        # Ensure axes is always 2D array
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for i, gen in enumerate(must_run_gens):
            row = i // cols
            col = i % cols
            ax = axes[row][col]

            # Get time series for this generator
            actual = actual_dispatch[gen]
            min_required = min_required_dispatch[gen]

            # Plot actual dispatch
            ax.plot(
                actual.index,
                actual,
                label="Actual Dispatch",
                color="blue",
                linewidth=2,
                alpha=0.8,
            )

            # Plot minimum required dispatch
            ax.plot(
                min_required.index,
                min_required,
                label="Min Required",
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

            # Check for violations (actual < minimum)
            violations = actual < min_required
            if violations.any():
                violation_times = actual.index[violations]
                ax.scatter(
                    violation_times,
                    actual[violations],
                    color="red",
                    s=20,
                    alpha=0.7,
                    label="Violations",
                    zorder=5,
                )

            # Formatting
            ax.set_title(f"{gen}", fontsize=10, pad=5)
            ax.set_ylabel("Power [MW]", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=7)

            # Format x-axis
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(7)

            # Add legend only to first subplot
            if i == 0:
                ax.legend(fontsize=8, loc="best")

        # Hide unused subplots
        for i in range(n_gens, rows * cols):
            row = i // cols
            col = i % cols
            axes[row][col].set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.94, bottom=0.08)
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Must-run dispatch validation plot saved to {output_filename}")

    except Exception as e:
        logger.error(f"Error creating must-run dispatch validation plot: {e}")
        make_mock_plot(output_filename)


def plot_generator_dispatch_and_availability(
    n: pypsa.Network,
    outage_scenarios: pd.DataFrame,
    output_filename: str,
    max_generators: int = 12,
) -> None:
    """
    Plot generator dispatch overlaid with availability profile from forced outages.

    Shows how generator dispatch responds to availability constraints from forced outages.
    Each subplot shows one generator's dispatch (p) and its availability factor (p_max_pu).

    Parameters
    ----------
    n : pypsa.Network
        Solved network with generator dispatch results
    outage_scenarios : pd.DataFrame
        Availability factors (0-1) for each generator over time, from forced outage scenarios
    output_filename : str
        Path to save the plot
    max_generators : int, optional
        Maximum number of generators to plot (default: 12)
    """
    try:
        # Get committable generators with dispatch
        if not hasattr(n.dynamic("Generator"), "p") or n.dynamic("Generator").p.empty:
            logger.info(
                f"No generator dispatch data. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Find generators with both dispatch and outage data
        dispatch = n.dynamic("Generator").p
        common_gens = list(set(dispatch.columns) & set(outage_scenarios.columns))

        if not common_gens:
            logger.info(
                f"No generators with both dispatch and outage data. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Select generators with most interesting data (those that had outages)
        # Calculate variance in availability to find generators with outages
        availability_variance = outage_scenarios[common_gens].var()
        selected_gens = availability_variance.nlargest(max_generators).index.tolist()

        if not selected_gens:
            selected_gens = common_gens[:max_generators]

        n_gens = len(selected_gens)
        rows = (n_gens + 2) // 3
        cols = min(3, n_gens)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)

        # Flatten axes for easier indexing
        if n_gens == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_gens > 1 else axes

        for i, gen in enumerate(selected_gens):
            ax = axes[i]

            # Get generator capacity
            p_nom = n.generators.loc[gen, "p_nom"]

            # Get availability data
            availability = n.generators_t.availability[gen]
            availability_mw = availability * p_nom

            # Plot availability as filled area
            ax.fill_between(
                availability.index,
                0,
                availability_mw,
                label="Available Capacity",
                color="lightgreen",
                alpha=0.3,
            )
            ax.plot(
                availability.index,
                availability_mw,
                label="Available Capacity",
                color="green",
                linewidth=1.5,
                alpha=0.7,
                linestyle="--",
            )

            # Plot dispatch on top
            dispatch_mw = dispatch[gen]
            ax.plot(
                dispatch_mw.index,
                dispatch_mw,
                label="Dispatch",
                color="blue",
                linewidth=2,
                alpha=0.9,
            )

            # Highlight outage periods (availability < 1)
            outages = availability < 0.99
            if outages.any():
                outage_periods = availability.index[outages]
                ax.axvspan(
                    outage_periods[0],
                    outage_periods[-1],
                    alpha=0.15,
                    color="red",
                    label="Outage Period",
                )

            # Formatting
            ax.set_title(f"{gen}", fontsize=10, fontweight="bold")
            ax.set_ylabel("MW", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH_TIME.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(8)

            # Add legend only to first subplot
            if i == 0:
                ax.legend(fontsize=8, loc="upper left")

        # Hide unused subplots
        for i in range(n_gens, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(
            "Generator Dispatch and Availability Profile",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(
            f"Generator dispatch and availability plot saved to {output_filename}"
        )

    except Exception as e:
        logger.error(f"Error creating generator dispatch and availability plot: {e}")
        make_mock_plot(output_filename)


def plot_outage_scenarios_heatmap(
    outage_scenarios: pd.DataFrame,
    output_filename: str,
    max_generators: int = 30,
) -> None:
    """
    Plot forced outage scenarios as a heatmap showing availability over time.

    Creates a heatmap where each row is a generator and each column is a time step.
    Color indicates availability (1 = available, 0 = forced outage).

    Parameters
    ----------
    outage_scenarios : pd.DataFrame
        Availability factors (0-1) for each generator over time
    output_filename : str
        Path to save the plot
    max_generators : int, optional
        Maximum number of generators to show (default: 30)
    """
    try:
        if outage_scenarios.empty:
            logger.info(
                f"No outage scenario data. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Find generators that experienced outages
        availability_variance = outage_scenarios.var()
        affected_gens = availability_variance[availability_variance > 0].index.tolist()

        if not affected_gens:
            logger.info(
                f"No generators with outages. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Select top affected generators
        selected_gens = affected_gens[:max_generators]
        data = outage_scenarios[selected_gens].T

        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, min(12, len(selected_gens) * 0.4)))

        # Plot heatmap
        im = ax.imshow(
            data.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        # Set ticks and labels
        ax.set_yticks(np.arange(len(selected_gens)))
        ax.set_yticklabels(selected_gens, fontsize=8)

        # X-axis: show every Nth timestamp
        n_timestamps = len(data.columns)
        step = max(1, n_timestamps // 20)  # Show ~20 labels
        x_ticks = np.arange(0, n_timestamps, step)
        x_labels = [
            data.columns[i].strftime(DateFormat.PLOT_DAY_MONTH_TIME.value)
            for i in x_ticks
        ]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

        # Labels
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Generator", fontsize=12)
        ax.set_title(
            "Forced Outage Scenarios - Generator Availability",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Availability Factor", rotation=270, labelpad=20, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Outage scenarios heatmap saved to {output_filename}")

    except Exception as e:
        logger.error(f"Error creating outage scenarios heatmap: {e}")
        make_mock_plot(output_filename)


def plot_generator_availability_profile(
    outage_scenarios: pd.DataFrame,
    output_filename: str,
    max_generators: int = 12,
) -> None:
    """
    Plot availability profiles over time for generators affected by forced outages.

    Shows individual line plots of availability factors for each generator, making it
    easy to see when and which generators experience outages.

    Parameters
    ----------
    outage_scenarios : pd.DataFrame
        Availability factors (0-1) for each generator over time
    output_filename : str
        Path to save the plot
    max_generators : int, optional
        Maximum number of generators to plot (default: 12)
    """
    try:
        if outage_scenarios.empty:
            logger.info(
                f"No outage scenario data. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Find generators that experienced outages (variance > 0)
        availability_variance = outage_scenarios.var()
        affected_gens = availability_variance[availability_variance > 0].index.tolist()

        if not affected_gens:
            logger.info(
                f"No generators with outages. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Select generators
        selected_gens = affected_gens[:max_generators]
        n_gens = len(selected_gens)

        rows = (n_gens + 2) // 3
        cols = min(3, n_gens)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=True)

        # Flatten axes for easier indexing
        if n_gens == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_gens > 1 else axes

        for i, gen in enumerate(selected_gens):
            ax = axes[i]

            availability = outage_scenarios[gen]

            # Plot availability
            ax.plot(
                availability.index,
                availability,
                label="Availability",
                color="green",
                linewidth=1.5,
            )

            # Fill outage periods
            ax.fill_between(
                availability.index,
                availability,
                1,
                where=(availability < 0.99),
                color="red",
                alpha=0.3,
                label="Outage",
            )

            # Add horizontal line at 1.0
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

            # Formatting
            ax.set_title(f"{gen}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Availability Factor", fontsize=9)
            ax.set_ylim(-0.05, 1.1)
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH_TIME.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(8)

            # Add legend only to first subplot
            if i == 0:
                ax.legend(fontsize=8, loc="lower left")

        # Hide unused subplots
        for i in range(n_gens, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(
            "Generator Availability Profiles - Forced Outages",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Generator availability profile plot saved to {output_filename}")

    except Exception as e:
        logger.error(f"Error creating generator availability profile plot: {e}")
        make_mock_plot(output_filename)


def plot_link_availability_profile(
    link_availability: pd.DataFrame,
    output_filename: str,
    max_links: int = 12,
) -> None:
    """
    Plot individual availability profiles for links over time.

    Creates a multi-panel plot showing availability timeseries for selected links.

    Parameters
    ----------
    link_availability : pd.DataFrame
        Availability factors (0-1) for each link over time
    output_filename : str
        Path to save the plot
    max_links : int, optional
        Maximum number of links to show (default: 12)
    """
    try:
        if link_availability.empty:
            logger.info(
                f"No link availability data. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Find links with reduced availability
        availability_variance = link_availability.var()
        affected_links = availability_variance[availability_variance > 0].index.tolist()

        if not affected_links:
            logger.info(
                f"No links with reduced availability. Mock plot generated for {output_filename}."
            )
            make_mock_plot(output_filename)
            return

        # Select top affected links
        selected_links = affected_links[:max_links]
        n_links = len(selected_links)

        # Calculate grid dimensions
        n_cols = min(3, n_links)
        n_rows = (n_links + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).flatten()

        for i, link in enumerate(selected_links):
            ax = axes[i]
            availability = link_availability[link]

            # Plot availability
            ax.plot(
                availability.index,
                availability,
                linewidth=1.5,
                color="green",
                label="Available",
            )

            # Highlight reduced availability periods (< 100%)
            reduced = availability < 0.99
            if reduced.any():
                ax.fill_between(
                    availability.index,
                    availability,
                    1,
                    where=(availability < 0.99),
                    color="red",
                    alpha=0.3,
                    label="Reduced Availability",
                )

            # Add horizontal line at 1.0
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

            # Formatting
            ax.set_title(f"{link}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Availability Factor", fontsize=9)
            ax.set_ylim(-0.05, 1.1)
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter(DateFormat.PLOT_DAY_MONTH_TIME.value)
            )
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_fontsize(8)

            # Add legend only to first subplot
            if i == 0:
                ax.legend(fontsize=8, loc="lower left")

        # Hide unused subplots
        for i in range(n_links, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(
            "Link Availability Profiles - Interconnector Forced Outages",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Link availability profile plot saved to {output_filename}")

    except Exception as e:
        logger.error(f"Error creating link availability profile plot: {e}")
        make_mock_plot(output_filename)


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="plot_process_validation",
        configfiles=[
            pathlib.Path("config", "test", "config.eva-regional-test.yaml"),
        ],
        it=1,
        run="eva-low-span",
        cy=2013,
        ty=2025,
    )

    configure_logging(snakemake)

    # Suppress verbose matplotlib font manager debug logs unless logging level is DEBUG
    if logging.getLogger().level != logging.DEBUG:
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    set_scenario_config(snakemake)

    solve_opts = snakemake.params.solving["options"]
    horizon = solve_opts.get("horizon", 72)
    overlap = solve_opts.get("overlap", 24)

    # Load the solved network
    # Storage plan is the same for both day-ahead and intraday (based on forecast)
    storage_plan = pd.read_csv(
        snakemake.input.storage_plans,  # ty: ignore[possibly-missing-attribute]
        index_col=0,
        parse_dates=True,
    )
    storage_plan_intraday = storage_plan
    storage_plan_day_ahead = storage_plan
    # Perfect-foresight (PF) mode uses a single market timeframe: the rule in
    # solve_eva_pf.smk supplies market_network/market_network_integer instead of
    # day_ahead_network/intraday_network. Detect the mode by checking the input.
    if hasattr(snakemake.input, "market_network"):
        network_day_ahead = pypsa.Network(snakemake.input.market_network)
        network_intraday = network_day_ahead
    else:
        network_day_ahead = pypsa.Network(
            snakemake.input.day_ahead_network  # ty: ignore[possibly-missing-attribute]
        )
        network_intraday = pypsa.Network(
            snakemake.input.intraday_network  # ty: ignore[possibly-missing-attribute]
        )

    # Load outage scenarios if available. This is generated once per (ty, fo) from
    # a fixed reference climate year, so align it onto this run's actual snapshots
    # (month-day-hour matching) before use. When forced outages are disabled,
    # mirror run_intra_day.py/run_single_timeframe.py's fallback (fully available)
    # rather than showing outages that were never actually applied to the solve.
    outage_scenarios = None
    if hasattr(snakemake.input, "outage_scenarios"):
        if snakemake.config["eva"]["forced_outages"]["enable"]:
            outage_scenarios = pd.read_csv(  # ty: ignore[no-matching-overload]
                snakemake.input.outage_scenarios, index_col=0, parse_dates=True
            )
            outage_scenarios = align_scenario_to_snapshots(
                outage_scenarios, network_intraday
            )
        else:
            outage_scenarios = pd.DataFrame(
                1.0,
                index=network_intraday.snapshots,
                columns=network_intraday.generators.index,
            )
        logger.info(f"Loaded outage scenarios with shape {outage_scenarios.shape}")

    # Load link availability if available. This is generated once per (ty, fo)
    # from a fixed reference climate year, so align it onto this run's actual
    # snapshots (month-day-hour matching) before use.
    link_availability = None
    if hasattr(snakemake.input, "link_availability"):
        link_availability = pd.read_csv(  # ty: ignore[no-matching-overload]
            snakemake.input.link_availability, index_col=0, parse_dates=True
        )
        link_availability = align_scenario_to_snapshots(
            link_availability, network_intraday
        )
        logger.info(f"Loaded link availability with shape {link_availability.shape}")

    # Load integer network if available (when linearized_unit_commitment=False).
    # PF mode uses market_network_integer; non-PF uses day_ahead_network_integer.
    network_day_ahead_integer = None
    if hasattr(snakemake.input, "market_network_integer"):
        network_day_ahead_integer = pypsa.Network(
            snakemake.input.market_network_integer
        )
    elif hasattr(snakemake.input, "day_ahead_network_integer"):
        network_day_ahead_integer = pypsa.Network(
            snakemake.input.day_ahead_network_integer
        )

    # ---------------------------------------
    # Pre-opt results

    # ---------------------------------------
    # Pre-opt set points in day-ahead

    hydro_units = network_day_ahead.storage_units[
        network_day_ahead.storage_units.carrier.str.contains(
            "Hydro - Reservoir|Hydro - Pump Storage Closed Loop \\(Pumping\\)",
            case=False,
            regex=True,
        )
    ]
    if hydro_units.empty:
        hydro_to_plot = ""
    else:
        hydro_to_plot = hydro_units.iloc[0].name

    if (
        hasattr(snakemake.output, "precondition_day_ahead")
        and snakemake.output.precondition_day_ahead
    ):
        plot_precondition_with_storage_planning(
            n=network_day_ahead,
            storage_plans=storage_plan_day_ahead,
            hydro=hydro_to_plot,
            market_name="Day-ahead",
            output_filename=snakemake.output.precondition_day_ahead,  # ty: ignore[invalid-argument-type]
            step_hours=horizon - overlap,
        )

    # ---------------------------------------
    # Pre-opt set points in intra-day
    if (
        hasattr(snakemake.output, "precondition_intraday")
        and snakemake.output.precondition_intraday
    ):
        plot_precondition_with_storage_planning(
            n=network_intraday,
            storage_plans=storage_plan_intraday,
            hydro=hydro_to_plot,
            market_name="Intraday",
            output_filename=snakemake.output.precondition_intraday,  # ty: ignore[invalid-argument-type]
            step_hours=horizon - overlap,
        )

    # ---------------------------------------
    # Pre-opt set points in market (PF mode)
    if (
        hasattr(snakemake.output, "precondition_market")
        and snakemake.output.precondition_market
    ):
        plot_precondition_with_storage_planning(
            n=network_day_ahead,
            storage_plans=storage_plan_day_ahead,
            hydro=hydro_to_plot,
            market_name="Market",
            output_filename=snakemake.output.precondition_market,  # ty: ignore[invalid-argument-type]
            step_hours=horizon - overlap,
        )

    # ---------------------------------------
    # Day-ahead -> intraday interlinkage
    if (
        hasattr(snakemake.output, "interlinkage_day_ahead_intraday")
        and snakemake.output.interlinkage_day_ahead_intraday
    ):
        plot_interlinkage_day_ahead_intraday(
            n_da=network_day_ahead,
            n_id=network_intraday,
            output_filename=snakemake.output.interlinkage_day_ahead_intraday,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Network flows
    if (
        hasattr(snakemake.output, "network_flows_day_ahead")
        and snakemake.output.network_flows_day_ahead
    ):
        plot_network_flows(
            network_day_ahead,
            snakemake.output.network_flows_day_ahead,  # ty: ignore[invalid-argument-type]
        )
    if (
        hasattr(snakemake.output, "network_flows_intraday")
        and snakemake.output.network_flows_intraday
    ):
        plot_network_flows(
            network_intraday,
            snakemake.output.network_flows_intraday,  # ty: ignore[invalid-argument-type]
        )
    if (
        hasattr(snakemake.output, "network_flows_day_ahead_hour_12")
        and snakemake.output.network_flows_day_ahead_hour_12
    ):
        plot_network_flows_at_one_timestamp(
            network_day_ahead,
            snakemake.output.network_flows_day_ahead_hour_12,  # ty: ignore[invalid-argument-type]
        )
    if (
        hasattr(snakemake.output, "network_flows_intraday_hour_12")
        and snakemake.output.network_flows_intraday_hour_12
    ):
        plot_network_flows_at_one_timestamp(
            network_intraday,
            snakemake.output.network_flows_intraday_hour_12,  # ty: ignore[invalid-argument-type]
        )

    # Network flows - market (PF mode)
    if (
        hasattr(snakemake.output, "network_flows_market")
        and snakemake.output.network_flows_market
    ):
        plot_network_flows(
            network_day_ahead,
            snakemake.output.network_flows_market,  # ty: ignore[invalid-argument-type]
        )
    if (
        hasattr(snakemake.output, "network_flows_market_hour_12")
        and snakemake.output.network_flows_market_hour_12
    ):
        plot_network_flows_at_one_timestamp(
            network_day_ahead,
            snakemake.output.network_flows_market_hour_12,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Forecast error load
    if (
        hasattr(snakemake.output, "forecast_error_load")
        and snakemake.output.forecast_error_load
    ):
        plot_forecast_error(
            n_da=network_day_ahead,
            n_id=network_intraday,
            output_filename=snakemake.output.forecast_error_load,  # ty: ignore[invalid-argument-type]
            component="loads",
            time_series_attr="p_set",
            title_suffix="Load",
        )

    # ---------------------------------------
    # Forecast error solar
    if (
        hasattr(snakemake.output, "forecast_error_solar")
        and snakemake.output.forecast_error_solar
    ):
        plot_forecast_error(
            n_da=network_day_ahead,
            n_id=network_intraday,
            output_filename=str(
                snakemake.output.forecast_error_solar
            ),  # ty: ignore[invalid-argument-type]
            component="generators",
            time_series_attr="p_max_pu",
            title_suffix="p_max",
            filter_carriers=["Solar (PV)"],
        )

    # ---------------------------------------
    # Forecast error wind
    if (
        hasattr(snakemake.output, "forecast_error_wind")
        and snakemake.output.forecast_error_wind
    ):
        plot_forecast_error(
            n_da=network_day_ahead,
            n_id=network_intraday,
            output_filename=snakemake.output.forecast_error_wind,  # ty: ignore[invalid-argument-type]
            component="generators",
            time_series_attr="p_max_pu",
            title_suffix="p_max",
            filter_carriers=["Wind Onshore", "Wind Offshore"],
        )

    # ---------------------------------------
    # Forecast error run-of-river (ror)
    if (
        hasattr(snakemake.output, "forecast_error_ror")
        and snakemake.output.forecast_error_ror
    ):
        plot_forecast_error(
            n_da=network_day_ahead,
            n_id=network_intraday,
            output_filename=snakemake.output.forecast_error_ror,  # ty: ignore[invalid-argument-type]
            component="generators",
            time_series_attr="p_max_pu",
            title_suffix="p_max",
            filter_carriers=["Hydro - Run of River (Turbine)"],
        )

    # ---------------------------------------
    # Forecast error hydro inflows
    if (
        hasattr(snakemake.output, "forecast_error_hydro_inflows")
        and snakemake.output.forecast_error_hydro_inflows
    ):
        plot_forecast_error(
            n_da=network_day_ahead,
            n_id=network_intraday,
            output_filename=snakemake.output.forecast_error_hydro_inflows,  # ty: ignore[invalid-argument-type]
            component="storage_units",
            time_series_attr="inflow",
            title_suffix="inflow",
            filter_carriers=["Hydro - Reservoir (Generator)"],
        )

    # ---------------------------------------
    # Dispatch validation (integer vs linear) - only when integer network is available
    if (
        network_day_ahead_integer is not None
        and hasattr(snakemake.output, "dispatch_validation_integer_vs_linear")
        and snakemake.output.dispatch_validation_integer_vs_linear
    ):
        plot_dispatch_validation_integer_vs_linear(
            n_integer=network_day_ahead_integer,
            n_linear=network_day_ahead,
            output_filename=snakemake.output.dispatch_validation_integer_vs_linear,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Marginal price validation (day-ahead vs intraday)
    if (
        hasattr(snakemake.output, "marginal_price_validation_da_vs_id")
        and snakemake.output.marginal_price_validation_da_vs_id
    ):
        plot_marginal_price_validation_da_vs_id(
            n_da=network_day_ahead,
            n_id=network_intraday,
            output_filename=snakemake.output.marginal_price_validation_da_vs_id,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Slack variables interlinkage (soft constraints) - only when soft constraints are enabled
    if (
        hasattr(snakemake.output, "slack_results_interlinkage_soft_constraints")
        and snakemake.output.slack_results_interlinkage_soft_constraints
    ):
        plot_slack_results_interlinkage_soft_constraints(
            n_id=network_intraday,
            output_filename=snakemake.output.slack_results_interlinkage_soft_constraints,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Reserves validation (day-ahead) - only when reserves data is available
    if (
        hasattr(snakemake.output, "reserves_validation_day_ahead")
        and snakemake.output.reserves_validation_day_ahead
    ):
        plot_reserves_validation_day_ahead(
            n_da=network_day_ahead,
            output_filename=snakemake.output.reserves_validation_day_ahead,  # ty: ignore[invalid-argument-type]
        )

    # Reserves validation - market (PF mode)
    if (
        hasattr(snakemake.output, "reserves_validation_market")
        and snakemake.output.reserves_validation_market
    ):
        plot_reserves_validation_day_ahead(
            n_da=network_day_ahead,
            output_filename=snakemake.output.reserves_validation_market,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Reserves interlinkage p_limits validation (day-ahead vs intraday) - only when reserves data is available
    if (
        hasattr(snakemake.output, "reserves_interlinkage_p_limits")
        and snakemake.output.reserves_interlinkage_p_limits
    ):
        plot_reserves_interlinkage_p_limits(
            n_da=network_day_ahead,
            n_id=network_intraday,
            output_filename=snakemake.output.reserves_interlinkage_p_limits,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # SoC validation for day-ahead network
    if (
        hasattr(snakemake.output, "soc_validation_day_ahead")
        and snakemake.output.soc_validation_day_ahead
    ):
        # Get validation zones from configuration
        validation_zones = snakemake.config["eva"]["soc_validation_zones"]
        plot_soc_validation_vs_dispatch(
            n=network_day_ahead,
            output_filename=snakemake.output.soc_validation_day_ahead,  # ty: ignore[invalid-argument-type]
            validation_zones=validation_zones,
        )

    # ---------------------------------------
    # SoC validation for intraday network
    if (
        hasattr(snakemake.output, "soc_validation_intraday")
        and snakemake.output.soc_validation_intraday
    ):
        # Get validation zones from configuration
        validation_zones = snakemake.config["eva"]["soc_validation_zones"]
        plot_soc_validation_vs_dispatch(
            n=network_intraday,
            output_filename=snakemake.output.soc_validation_intraday,  # ty: ignore[invalid-argument-type]
            validation_zones=validation_zones,
        )

    # ---------------------------------------
    # SOC cumsum validation by technology for day-ahead network
    if (
        hasattr(snakemake.output, "soc_cumsum_validation_day_ahead")
        and snakemake.output.soc_cumsum_validation_day_ahead
    ):
        # Get validation zones from configuration
        validation_zones = snakemake.config["eva"]["soc_validation_zones"]
        plot_soc_cumsum_validation_by_technology(
            n=network_day_ahead,
            output_filename=snakemake.output.soc_cumsum_validation_day_ahead,  # ty: ignore[invalid-argument-type]
            validation_zones=validation_zones,
        )

    # ---------------------------------------
    # SOC cumsum validation by technology for intraday network
    if (
        hasattr(snakemake.output, "soc_cumsum_validation_intraday")
        and snakemake.output.soc_cumsum_validation_intraday
    ):
        # Get validation zones from configuration
        validation_zones = snakemake.config["eva"]["soc_validation_zones"]
        plot_soc_cumsum_validation_by_technology(
            n=network_intraday,
            output_filename=snakemake.output.soc_cumsum_validation_intraday,  # ty: ignore[invalid-argument-type]
            validation_zones=validation_zones,
        )

    # SoC validation - market (PF mode)
    if (
        hasattr(snakemake.output, "soc_validation_market")
        and snakemake.output.soc_validation_market
    ):
        validation_zones = snakemake.config["eva"]["soc_validation_zones"]
        plot_soc_validation_vs_dispatch(
            n=network_day_ahead,
            output_filename=snakemake.output.soc_validation_market,  # ty: ignore[invalid-argument-type]
            validation_zones=validation_zones,
        )

    # SoC cumsum validation - market (PF mode)
    if (
        hasattr(snakemake.output, "soc_cumsum_validation_market")
        and snakemake.output.soc_cumsum_validation_market
    ):
        validation_zones = snakemake.config["eva"]["soc_validation_zones"]
        plot_soc_cumsum_validation_by_technology(
            n=network_day_ahead,
            output_filename=snakemake.output.soc_cumsum_validation_market,  # ty: ignore[invalid-argument-type]
            validation_zones=validation_zones,
        )

    # ---------------------------------------
    # Reserve revenue validation (day-ahead) - only when reserves data is available
    if (
        hasattr(snakemake.output, "reserve_revenue_validation_day_ahead")
        and snakemake.output.reserve_revenue_validation_day_ahead
    ):
        plot_reserve_revenue_validation(
            n=network_day_ahead,
            output_filename=snakemake.output.reserve_revenue_validation_day_ahead,  # ty: ignore[invalid-argument-type]
        )

    # Reserve revenue validation - market (PF mode)
    if (
        hasattr(snakemake.output, "reserve_revenue_validation_market")
        and snakemake.output.reserve_revenue_validation_market
    ):
        plot_reserve_revenue_validation(
            n=network_day_ahead,
            output_filename=snakemake.output.reserve_revenue_validation_market,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # DSR validation plots
    for i, j in [("day_ahead", network_day_ahead), ("intraday", network_intraday)]:
        for k in ["explicit", "implicit"]:
            plot_name = f"dsr_validation_{i}_{k}"
            if hasattr(snakemake.output, plot_name) and getattr(
                snakemake.output, plot_name
            ):
                if k == "explicit":
                    plot_dsr_validation_explicit(
                        n=j,
                        output_filename=getattr(snakemake.output, plot_name),
                    )
                else:
                    plot_dsr_validation_implicit(
                        n=j,
                        output_filename=getattr(snakemake.output, plot_name),
                    )

    # DSR validation - market (PF mode)
    if (
        hasattr(snakemake.output, "dsr_validation_market_explicit")
        and snakemake.output.dsr_validation_market_explicit
    ):
        plot_dsr_validation_explicit(
            n=network_day_ahead,
            output_filename=snakemake.output.dsr_validation_market_explicit,  # ty: ignore[invalid-argument-type]
        )
    if (
        hasattr(snakemake.output, "dsr_validation_market_implicit")
        and snakemake.output.dsr_validation_market_implicit
    ):
        plot_dsr_validation_implicit(
            n=network_day_ahead,
            output_filename=snakemake.output.dsr_validation_market_implicit,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Hydro constraints validation (day-ahead)
    if (
        hasattr(snakemake.output, "hydro_constraints_validation_day_ahead")
        and snakemake.output.hydro_constraints_validation_day_ahead
    ):
        plot_hydro_constraints_validation(
            n=network_day_ahead,
            output_filename=snakemake.output.hydro_constraints_validation_day_ahead,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Hydro constraints validation (intraday)
    if (
        hasattr(snakemake.output, "hydro_constraints_validation_intraday")
        and snakemake.output.hydro_constraints_validation_intraday
    ):
        plot_hydro_constraints_validation(
            n=network_intraday,
            output_filename=snakemake.output.hydro_constraints_validation_intraday,  # ty: ignore[invalid-argument-type]
        )

    # Hydro constraints validation - market (PF mode)
    if (
        hasattr(snakemake.output, "hydro_constraints_validation_market")
        and snakemake.output.hydro_constraints_validation_market
    ):
        plot_hydro_constraints_validation(
            n=network_day_ahead,
            output_filename=snakemake.output.hydro_constraints_validation_market,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Must-run dispatch validation (day-ahead)
    if (
        hasattr(snakemake.output, "must_run_dispatch_validation_day_ahead")
        and snakemake.output.must_run_dispatch_validation_day_ahead
    ):
        plot_must_run_dispatch_validation(
            n=network_day_ahead,
            output_filename=snakemake.output.must_run_dispatch_validation_day_ahead,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Must-run dispatch validation (intraday)
    if (
        hasattr(snakemake.output, "must_run_dispatch_validation_intraday")
        and snakemake.output.must_run_dispatch_validation_intraday
    ):
        plot_must_run_dispatch_validation(
            n=network_intraday,
            output_filename=snakemake.output.must_run_dispatch_validation_intraday,  # ty: ignore[invalid-argument-type]
        )

    # Must-run dispatch validation - market (PF mode)
    if (
        hasattr(snakemake.output, "must_run_dispatch_validation_market")
        and snakemake.output.must_run_dispatch_validation_market
    ):
        plot_must_run_dispatch_validation(
            n=network_day_ahead,
            output_filename=snakemake.output.must_run_dispatch_validation_market,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Generator dispatch and availability (intraday with forced outages)
    if (
        hasattr(snakemake.output, "generator_dispatch_and_availability")
        and snakemake.output.generator_dispatch_and_availability
        and outage_scenarios is not None
    ):
        plot_generator_dispatch_and_availability(
            n=network_intraday,
            outage_scenarios=outage_scenarios,
            output_filename=snakemake.output.generator_dispatch_and_availability,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Outage scenarios heatmap
    if (
        hasattr(snakemake.output, "outage_scenarios_heatmap")
        and snakemake.output.outage_scenarios_heatmap
        and outage_scenarios is not None
    ):
        plot_outage_scenarios_heatmap(
            outage_scenarios=outage_scenarios,
            output_filename=snakemake.output.outage_scenarios_heatmap,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Generator availability profile
    if (
        hasattr(snakemake.output, "generator_availability_profile")
        and snakemake.output.generator_availability_profile
        and outage_scenarios is not None
    ):
        plot_generator_availability_profile(
            outage_scenarios=outage_scenarios,
            output_filename=snakemake.output.generator_availability_profile,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Link availability profile
    if (
        hasattr(snakemake.output, "link_availability_profile")
        and snakemake.output.link_availability_profile
        and link_availability is not None
    ):
        plot_link_availability_profile(
            link_availability=link_availability,
            output_filename=snakemake.output.link_availability_profile,  # ty: ignore[invalid-argument-type]
        )

    # ---------------------------------------
    # Touch the output flag to indicate completion
    pathlib.Path(
        snakemake.output.flag  # ty: ignore[possibly-missing-attribute]
    ).touch()
