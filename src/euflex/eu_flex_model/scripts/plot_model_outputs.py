#!/usr/bin/env python3
"""
Plot consolidated EVA model outputs, including flexibility indicators and market metrics.

This script creates comprehensive visualisations for:
1. Base case scenarios
2. Dummy storage optimisation scenarios with varying mh (max hours) and ct (curtailment target)
3. Comparisons between base and optimised scenarios
4. Catalogue plots for all available model outputs
"""

import logging
import pathlib
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
)
from euflex.eu_flex_model.scripts.plot_model_output_catalogue import (
    generate_catalogue_plots,
)

matplotlib.use("Agg")  # Use non-GUI backend to avoid X server issues

logger = logging.getLogger(__name__)


def setup_plotting_style() -> None:
    """Set up a clean plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    sns.set_palette("husl")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9


def get_available_dimensions(dt: xr.DataTree) -> dict:
    """
    Detect available dimensions in the datatree.

    Parameters
    ----------
    dt : xarray.DataTree
        Input datatree with flexibility indicators

    Returns
    -------
    dict
        Dictionary with available dimension values
    """
    dims = {}

    # Check if this is a full datatree with "base" section or a single scenario
    if "base" in dt:
        # Full datatree - get dimensions from base
        data = dt["base"].ds
    else:
        # Single scenario dataset
        data = dt.ds

    # Base dimensions (always present)
    dims["TY"] = data.coords["TY"].values
    dims["CY"] = data.coords["CY"].values
    dims["BZ"] = data.coords["BZ"].values
    dims["carrier"] = data.coords["carrier"].values

    # Dummy storage optimization dimensions (if present)
    if "dummy_storage_optimization" in dt:
        dummy_ds = dt["dummy_storage_optimization"].ds
        if "mh" in dummy_ds.coords:
            dims["mh"] = dummy_ds.coords["mh"].values
        else:
            dims["mh"] = []
        if "ct" in dummy_ds.coords:
            dims["ct"] = dummy_ds.coords["ct"].values
        else:
            dims["ct"] = []
    else:
        # No dummy storage optimization data
        dims["mh"] = []
        dims["ct"] = []

    return dims


def create_color_map(dims: dict, data: xr.Dataset | None = None):
    """
    Create consistent color mappings for different dimensions.

    Parameters
    ----------
    dims : dict
        Dictionary with dimension values
    data : xarray.Dataset, optional
        Dataset that may contain _color coordinate

    Returns
    -------
    dict
        Dictionary with color mappings
    """
    colors = {}

    # Colors for different mh values (storage durations)
    if "mh" in dims:
        n_mh = len(dims["mh"])
        colors["mh"] = dict(
            zip(dims["mh"], plt.colormaps["Set1"](np.linspace(0, 0.8, n_mh)))
        )

    # Colors for different ct values (curtailment targets)
    if "ct" in dims:
        n_ct = len(dims["ct"])
        colors["ct"] = dict(
            zip(dims["ct"], plt.colormaps["Set2"](np.linspace(0.2, 0.8, n_ct)))
        )

    # Colors for carriers - check if _color field exists in dataset
    carrier_colors = {}

    # Try to get colors from dataset if available
    if data is not None and "_color" in data.data_vars:
        color_data = data["_color"]
        for i, carrier in enumerate(data["carrier"].values):
            if color_data.ndim == 1:  # 1D array indexed by carrier
                color_val = color_data.isel(carrier=i).item()
            else:  # Multi-dimensional, select first occurrence
                color_slice = color_data.isel(carrier=i)
                if color_slice.ndim == 0:
                    color_val = color_slice.item()
                else:
                    # If still multi-dimensional, take first element
                    color_val = color_slice.values.flat[0]

            # Ensure it's a string and valid color
            if isinstance(color_val, str) and color_val.startswith("#"):
                carrier_colors[str(carrier)] = color_val

    # Default colors for carriers not in dataset or as fallback
    default_carrier_colors = {
        "solar": "#f9d71c",
        "solar-hsat": "#ffc107",
        "onwind": "#74c476",
        "offwind-ac": "#41ab5d",
        "offwind-dc": "#238b45",
        "hydro": "#2196f3",
        "ror": "#64b5f6",
        "battery": "#9c27b0",
        "H2": "#e91e63",
        "CCGT": "#ff9800",
        "OCGT": "#ff5722",
        "nuclear": "#795548",
        "biomass": "#8bc34a",
        "coal": "#424242",
        "lignite": "#616161",
        "oil": "#37474f",
        "gas": "#607d8b",
        "dummy": "#f44336",
    }

    # Merge with defaults for any missing colors
    for carrier, color in default_carrier_colors.items():
        if carrier not in carrier_colors:
            carrier_colors[carrier] = color

    colors["carrier"] = carrier_colors

    # Colors for target years
    if "TY" in dims:
        n_ty = len(dims["TY"])
        colors["TY"] = dict(
            zip(dims["TY"], plt.colormaps["viridis"](np.linspace(0.3, 0.9, n_ty)))
        )

    return colors


def rename_long_labels(labels: list):
    """
    Rename long carrier labels.
    """
    mapping = {
        "Hydro - Pump Storage Closed Loop (Pumping)": "Pump Storage (closed)",
        "Hydro - Pump Storage Open Loop (Pumping)": "Pump Storage (open)",
        "Hydro - Reservoir": "Reservoir",
        "Hydro - Run of River (Turbine)": "Run-of-River",
        "Not defined or splitting not known RES": "Not defined",
    }
    return [mapping.get(label, label) for label in labels]


def create_flexibility_plots(
    dt: xr.DataTree,
    output_path: Path,
    is_dummy_storage: bool = False,
    mh: float | None = None,
    ct: float | None = None,
    ct_str: str | None = None,
) -> Path:
    """
    Create comprehensive flexibility assessment plots.

    Parameters
    ----------
    dt : xarray.DataTree
        Input datatree with flexibility indicators
    output_path : Path
        Path to save output plot
    is_dummy_storage : bool
        Flag indicating if the scenario is a dummy storage scenario
    mh : float
        Storage duration in hours
    ct : float
        Curtailment target value (e.g., 0.1 for 10%)
    ct_str : str
        Curtailment target string identifier (e.g., "ct1", "ct2")

    Returns
    -------
    Path
        Path to saved figure
    """
    dims = get_available_dimensions(dt)

    # Handle both base case and dummy storage scenario data
    if not is_dummy_storage:
        data = dt["base"].ds
        title_prefix = "Base Case"
    else:
        if mh is None or ct is None:
            raise ValueError("For dummy storage scenarios, mh and ct must be provided.")
        # This is a dummy storage scenario
        # Use ct_str for data selection if provided, otherwise convert ct to string
        ct_selector = ct_str if ct_str is not None else str(ct)
        data = dt["dummy_storage_optimization"].ds.sel(mh=str(mh), ct=ct_selector)
        title_prefix = (
            f"Dummy Storage Scenario ({mh}h, {ct * 100:.1f}% curtailment target)"
        )

    colors = create_color_map(dims, data)
    n_tys = len(dims["TY"])

    # Create figure with subplots using nested GridSpec for custom spacing
    n_rows = 4
    fig_height = 18
    fig = plt.figure(figsize=(6 + n_tys * 4, fig_height))

    # Create main grid with two columns: first column and the rest
    # Use width_ratios to control relative widths, adding extra space after first column
    gs_main = gridspec.GridSpec(1, 2, figure=fig, wspace=0.5, width_ratios=[1, n_tys])

    # Create subgrid for the first column (n_rows rows, 1 column)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        n_rows, 1, subplot_spec=gs_main[0], hspace=0.3
    )

    # Create subgrid for the remaining columns (n_rows rows, n_tys columns)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_tys, subplot_spec=gs_main[1], hspace=0.3, wspace=0.4
    )

    ax = fig.add_subplot(gs_left[0, 0])
    plot_capacity_evolution(data, ax, dims, colors)

    # Create production mix subplots for each target year
    for i, ty in enumerate(dims["TY"]):
        ax = fig.add_subplot(gs_right[0, i])
        plot_production_mix(data, ax, ty, colors)

    # Create consumption mix subplots for each target year (new row)
    for i, ty in enumerate(dims["TY"]):
        ax = fig.add_subplot(gs_right[1, i])
        plot_consumption_mix(data, ax, ty, colors)

    ax = fig.add_subplot(gs_left[2, 0])
    plot_load_duration_curves(data, ax, dims, colors)

    # Create curtailment heatmap subplots for each target year
    for i, ty in enumerate(dims["TY"]):
        ax = fig.add_subplot(gs_right[2, i])
        plot_curtailment_heatmap(data, ax, ty)

    ax = fig.add_subplot(gs_left[1, 0])
    plot_storage_capacities(data, ax, colors)

    # Create flexibility metrics subplots for each target year
    for i, ty in enumerate(dims["TY"]):
        ax = fig.add_subplot(gs_right[3, i])
        plot_flexibility_metrics(data, ax, ty)

    # Add curtailment comparison plot for dummy storage scenarios only
    if is_dummy_storage:
        ax = fig.add_subplot(gs_left[3, 0])
        plot_curtailment_comparison(data, ax, dims, colors)

    # Add main title
    fig.suptitle(
        f"{title_prefix} Flexibility Assessment", fontsize=16, fontweight="bold", y=0.98
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"Base case plots saved to {output_path}")
    return output_path


def create_comparison_plots(dt: xr.DataTree, output_path: Path):
    """
    Create comparison plots between base case and dummy storage optimization.

    Parameters
    ----------
    dt : xarray.DataTree
        Input datatree with flexibility indicators
    output_path : Path
        Path to save output plot

    Returns
    -------
    Path
        Path to saved figure
    """
    dims = get_available_dimensions(dt)
    base_data = dt["base"].ds
    colors = create_color_map(dims, base_data)
    dummy_data = dt["dummy_storage_optimization"].ds

    # Check if dummy storage optimization data exists
    if not dummy_data.data_vars:
        logger.info(
            "No dummy storage optimization data available (dummy storage is disabled)"
        )
        # Create a simple plot indicating dummy storage is disabled
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Dummy Storage Optimization Disabled\n\nNo comparison data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            style="italic",
            color="gray",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.suptitle("Dummy Storage Comparison", fontsize=18, fontweight="bold")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        return output_path

    # Create figure
    fig = plt.figure(figsize=(12, 6))

    # Storage Capacity vs Curtailment Overview
    ax = fig.add_subplot(111)
    plot_storage_capacity_vs_curtailment(base_data, dummy_data, ax, dims, colors)

    # Add main title
    fig.suptitle(
        "Impact of Dummy Storages",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save figure
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"Comparison plots saved to {output_path}")
    return output_path


# Individual plotting functions for base case
def plot_capacity_evolution(data, ax, dims, colors):
    """Plot capacity evolution across target years."""
    capacities = data["production_capacities"].mean(dim=["CY", "BZ"])
    capacities = capacities / 1e3  # Convert to GWh
    carriers = capacities["carrier"].values
    ty_values = capacities["TY"].values

    x = np.arange(len(ty_values))
    width = 0.8 / len(carriers)

    for i, carrier in enumerate(carriers):
        values = capacities.sel(carrier=carrier).values
        offset = (i - len(carriers) / 2 + 0.5) * width
        color = colors["carrier"].get(str(carrier), "#888888")
        ax.bar(x + offset, values, width, label=str(carrier), color=color, alpha=0.8)

    ax.set_xlabel("Target Year")
    ax.set_ylabel("Capacity [GW]")
    ax.set_title("Production Capacity Evolution")
    ax.set_xticks(x)
    ax.set_xticklabels(ty_values)
    handles, labels = ax.get_legend_handles_labels()
    labels = rename_long_labels(labels)
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 1.2),
        loc="upper left",
        ncol=1,
        fontsize=8,
        frameon=True,
        fancybox=True,
    )
    ax.grid(True, alpha=0.3)


def plot_production_mix(
    data: xr.Dataset, ax: plt.Axes, ty: int, colors: dict[str, dict[str, str]]
) -> None:
    """
    Plot the electricity production mix for a specific target year as a pie chart.

    This function aggregates electricity supply data across hours and climate years,
    then sums over bidding zones to create a pie chart of production by carrier.

    Parameters
    ----------
    data : xr.Dataset
        Input dataset containing electricity supply information.
        Expected to have dimensions: hour, CY (climate years), BZ (bidding zones)
        and a data variable "supply" with carrier information.

    ax : plt.Axes
        Matplotlib Axes object on which the pie chart will be plotted.

    ty : int
        Target year for which the production mix is to be visualized.

    colors : Dict[str, Dict[str, str]]
        Dictionary mapping carrier types to their corresponding color codes.
        Expected structure: {"carrier": {"carrier_name": "color_code"}}

    Notes
    -----
    - Carriers with less than 5% contribution are not labeled.
    - Percentage labels are shown for contributions > 3%.
    - Total production is displayed in the chart title (in TWh).
    """

    # Average over climate years, sum over bidding zones
    prod_data = data["supply"].sum("hour").mean(dim="CY").sum(dim="BZ")

    carriers = prod_data["carrier"].values
    values = prod_data.sel(TY=ty).values
    carrier_colors = [colors["carrier"].get(str(c), "#888888") for c in carriers]

    # Create labels with carrier names for values > 5%
    labels = []
    for carrier, value in zip(carriers, values):
        pct = value / values.sum() * 100
        if pct > 5:
            # Add line breaks to long labels
            label = str(carrier)
            if len(label) > 10:
                label = label.replace(" ", "\n", 1)  # Add one line break
            labels.append(label)
        else:
            labels.append("")

    # Create pie chart with labels
    ax.pie(
        values,
        labels=labels,
        colors=carrier_colors,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 3 else "",
        startangle=90,
        textprops={"fontsize": 9},
    )
    total_production = values.sum()

    # Set title
    ax.set_title(f"TY {ty} Electricity Mix\n(Total: {total_production / 1e6:.2f} TWh)")


def plot_consumption_mix(
    data: xr.Dataset, ax: plt.Axes, ty: int, colors: dict[str, dict[str, str]]
) -> None:
    """
    Plot the energy consumption mix for a specific target year as a pie chart.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing energy withdrawal data.
        Expected to have dimensions including 'hour', 'CY' (climate years),
        and 'BZ' (bidding zones), with a 'carrier' coordinate.

    ax : matplotlib.axes.Axes
        The matplotlib Axes object on which to draw the pie chart.

    ty : int
        Target year for which the consumption mix is to be plotted.

    colors : dict
        A nested dictionary of color mappings for energy carriers.
        Expected structure: {"carrier": {"carrier_name": "color_hex"}}

    Notes
    -----
    The function performs the following key steps:
    - Aggregates withdrawal data by summing over hours and averaging over climate years
    - Sums across bidding zones
    - Creates a pie chart with carrier-specific colors
    - Labels carriers with >1% contribution
    - Displays percentage for carriers with >3% contribution
    """

    # Average over climate years, sum over bidding zones
    cons_data = data["withdrawal"].sum("hour").mean(dim="CY").sum(dim="BZ")

    carriers = cons_data["carrier"].values
    values = cons_data.sel(TY=ty).values
    carrier_colors = [colors["carrier"].get(str(c), "#888888") for c in carriers]

    # Create labels with carrier names for values > 1%
    labels = []
    for carrier, value in zip(carriers, values):
        pct = value / values.sum() * 100
        if pct > 1:
            # Add line breaks to long labels
            label = str(carrier)
            if len(label) > 10:
                label = label.replace(" ", "\n", 1)  # Add one line break
            labels.append(label)
        else:
            labels.append("")

    # Create pie chart with labels
    ax.pie(
        values,
        labels=labels,
        colors=carrier_colors,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 3 else "",
        startangle=90,
        textprops={"fontsize": 9},
    )
    total_consumption = values.sum()

    # Set title
    ax.set_title(f"TY {ty} Consumption Mix\n(Total: {total_consumption / 1e6:.2f} TWh)")


def plot_storage_capacities(
    data: xr.Dataset, ax: plt.Axes, colors: dict[str, dict[str, str]]
) -> None:
    """
    Plot storage capacity evolution across target years.

    This function creates a bar plot showing the mean storage capacities
    for different carriers across target years.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing storage capacities with dimensions
        including 'CY', 'BZ', 'carrier', and 'TY'.
    ax : matplotlib.axes.Axes
        The matplotlib axes on which to plot the storage capacities.
    colors : dict
        A nested dictionary containing color mappings for carriers.
        Expected structure: {'carrier': {'carrier_name': 'color_code'}}

    Returns
    -------
    None
        The function modifies the provided matplotlib axes in-place.

    Notes
    -----
    - Capacities are converted from base units to GW (gigawatts)
    - If no storage carriers exist after filtering, a text message is displayed
    - The plot includes a legend, grid, and axis labels
    """

    storage = data["storage_capacities"].mean(dim=["CY", "BZ"]).dropna(dim="carrier")
    storage = storage / 1e3  # Convert to GW
    carriers = storage["carrier"].values
    ty_values = storage["TY"].values

    # Handle case where no storage carriers exist after dropna
    if len(carriers) == 0:
        ax.text(
            0.5,
            0.5,
            "No storage capacities available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Target Year")
        ax.set_ylabel("Capacity [GW]")
        return

    x = np.arange(len(ty_values))
    width = 0.8 / len(carriers)

    for i, carrier in enumerate(carriers):
        values = storage.sel(carrier=carrier).values
        offset = (i - len(carriers) / 2 + 0.5) * width
        color = colors["carrier"].get(str(carrier), "#888888")
        ax.bar(x + offset, values, width, label=str(carrier), color=color, alpha=0.8)

    ax.set_xlabel("Target Year")
    ax.set_ylabel("Capacity [GW]")
    ax.set_title("Storage Capacity Evolution")
    ax.set_xticks(x)
    ax.set_xticklabels(ty_values)
    handles, labels = ax.get_legend_handles_labels()
    labels = rename_long_labels(labels)
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.05, 0),
        loc="lower left",
        ncol=1,
        fontsize=8,
        frameon=True,
        fancybox=True,
    )
    ax.grid(True, alpha=0.3)


def plot_load_duration_curves(data, ax, dims, colors):
    """Plot residual load duration curves with clear grouping by target year."""
    residual_load = data["residual_load"]

    # Use distinct line styles for different climate years
    line_styles = ["-", "--", "-.", ":"]

    # Plot for each TY with different CY variations
    for i, ty in enumerate(dims["TY"]):
        ty_color = colors["TY"][ty]

        for j, cy in enumerate(dims["CY"][:3]):  # Limit to first 3 CY for clarity
            rl = residual_load.sel(TY=ty, CY=cy).isel(BZ=0)
            sorted_values = np.sort(rl.values)[::-1]
            hours = np.arange(len(sorted_values))

            # Use consistent color for same TY, different line style for CY
            line_style = line_styles[j % len(line_styles)]
            label = f"TY {ty}, CY {cy}"

            ax.plot(
                hours,
                sorted_values / 1000,
                color=ty_color,
                linestyle=line_style,
                label=label,
                alpha=0.8,
                linewidth=1.5,
            )

    ax.set_xlabel("Hours")
    ax.set_ylabel("Residual Load [GW]")
    ax.set_title("Residual Load Duration Curves")

    # Create custom legend with grouping
    handles, labels = ax.get_legend_handles_labels()

    # Sort by TY for better grouping in legend
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    labels, handles = zip(*sorted_pairs)

    ax.legend(handles, labels, fontsize=7, ncol=len(dims["TY"]))
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)


def plot_curtailment_heatmap(data, ax, ty):
    """Plot temporal curtailment patterns for a specific target year (averaged over climate years)."""
    # Sum curtailment across all renewable carriers
    renewable_carriers = RENEWABLE_CARRIERS

    # Filter for available renewable carriers
    available_carriers = [
        c for c in renewable_carriers if c in data["curtailment"].carrier.values
    ]

    if not available_carriers:
        # No renewable curtailment data available
        ax.text(
            0.5,
            0.5,
            "No renewable\ncurtailment data\navailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f"TY {ty} Curtailment Pattern")
        return

    curt = (
        data["curtailment"].sel(carrier=available_carriers).sum(dim=["carrier", "BZ"])
    )
    # Average across climate years
    curt_mean = curt.mean("CY")

    # Get data for this TY
    curt_ty = curt_mean.sel(TY=ty)

    # Reshape to days x hours
    n_hours = len(curt_ty["hour"])
    n_days = n_hours // 24

    if n_days * 24 != n_hours:
        # Not exact days, show as is
        im = ax.imshow(curt_ty.values.reshape(1, -1), aspect="auto", cmap="YlOrRd")
        ax.set_ylabel("Value", fontsize=9)
        ax.set_xlabel("Snapshot", fontsize=9)
    else:
        # Show first 7 days for clarity
        days_to_show = min(7, n_days)
        curt_matrix = curt_ty.values[: days_to_show * 24].reshape(days_to_show, 24)
        im = ax.imshow(curt_matrix.T, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_ylabel("Hour", fontsize=9)
        ax.set_xlabel("Day", fontsize=9)
        ax.set_yticks(range(0, 24, 6))
        ax.set_yticklabels(range(0, 24, 6), fontsize=8)
        ax.set_xticks(range(days_to_show))
        ax.set_xticklabels(range(1, days_to_show + 1), fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MW", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(f"TY {ty} Curtailment Pattern")


def plot_flexibility_metrics(data, ax, ty):
    """Plot summary flexibility metrics for a specific target year."""
    # Select data for the specific target year and average over climate years
    data_ty = data.sel(TY=ty).mean(dim="CY").sum("BZ")

    # Total curtailment
    renewable_carriers = RENEWABLE_CARRIERS
    # Filter for available renewable carriers
    available_carriers = [
        c for c in renewable_carriers if c in data_ty["curtailment"].carrier.values
    ]
    if available_carriers:
        total_curt = (
            data_ty["curtailment"].sel(carrier=available_carriers).sum().item() / 1e6
        )  # TWh
    else:
        total_curt = 0.0

    # Total ENS
    total_ens = data_ty["energy_not_served"].sum().item() / 1e3  # GWh

    # Peak residual load
    peak_load = data_ty["residual_load"].max().item() / 1e3  # GW

    seasonal_flexibility_needs = data_ty["seasonal_flexibility_need"].item() / 1e3

    # Create bar chart
    metrics = {
        "Curtailment\n(TWh)": total_curt,
        "ENS\n(GWh)": total_ens,
        "Peak Load\n(GW)": peak_load,
        "Seasonal\nFlexibility Needs\n(GW)": seasonal_flexibility_needs,
    }

    x = np.arange(len(metrics))
    colors_list = ["orange", "red", "blue", "purple"]

    ax.bar(x, list(metrics.values()), color=colors_list, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.set_title(f"TY {ty} Metrics")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, value in enumerate(metrics.values()):
        ax.text(
            i,
            value * 1.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8
            ),
        )


# Individual plotting functions for comparison
def plot_storage_capacity_vs_curtailment(base_data, dummy_data, ax, dims, colors):
    """Plot dummy storage capacity vs total curtailment for different energy-power ratios."""

    # Check if Dummy Storage exists in the data
    if "Dummy Storage" not in dummy_data["production_capacities"].carrier.values:
        # No dummy storage data - create empty plot with message
        ax.text(
            0.5,
            0.5,
            "Dummy Storage Optimization Disabled",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            color="gray",
        )
        ax.set_xlabel("Total Curtailment [%]")
        ax.set_ylabel("Installed Dummy Storage Capacity [GW]")
        ax.set_title("Dummy Storage Capacity vs. Curtailment Target")
        ax.grid(True, alpha=0.3)
        return

    # Plot data points for each mh (energy-power ratio)
    for mh in dims["mh"]:
        storage_caps = []
        curtailments = []

        for ct in dims["ct"]:
            # Get dummy storage capacity (aggregate all carriers, all dimensions)
            storage = dummy_data["production_capacities"].sel(mh=mh, ct=ct)
            dummy_storage = (
                storage.sel(carrier="Dummy Storage").sum().item() / 1e3
            )  # Convert to GW

            storage_caps.append(dummy_storage)
            # Convert ct string to numeric value (ct1 -> index 1, get value from config)
            ct_index = int(ct[2:])  # Extract number after 'ct'
            curtailment_targets = snakemake.config["eva"]["dummy_storage_optimization"][
                "curtailment_targets"
            ]
            ct_value = curtailment_targets[ct_index - 1]
            curtailments.append(ct_value * 100)

        # Plot points for this mh value
        color = colors["mh"][mh]
        ax.scatter(
            curtailments,
            storage_caps,
            s=100,
            color=color,
            label=mh,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        # Connect points with lines if multiple ct values
        if len(dims["ct"]) > 1:
            ax.plot(
                curtailments,
                storage_caps,
                color=color,
                alpha=0.3,
                linestyle="--",
                linewidth=1,
            )

    ax.legend(
        title="Energy/Power Ratio [h]",
        loc="upper left",
        fontsize=9,
        bbox_to_anchor=(1.0, 1.01),
        frameon=True,
        fancybox=True,
    )
    ax.set_xlabel("Curtailment Target [%]")
    ax.set_ylabel("Installed Dummy Storage Capacity [GW]")
    ax.set_title("Storage Capacity vs Curtailment")
    ax.grid(True, alpha=0.3)

    # Set reasonable axis limits
    # ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0)


def plot_network_needs_overview(data, ax, dims, colors, ct):
    """Plot network needs overview - aggregated over all bidding zones."""
    if "network_needs" not in data:
        ax.text(
            0.5,
            0.5,
            "Network Needs\nNot Available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            color="gray",
        )
        ax.set_title("Network Needs Overview")
        return

    network_needs = data["network_needs"]

    # Sum network needs across all bidding zones for overview
    # Handle both 'bus' and 'Bus' dimension names
    bus_dim = None
    for dim in network_needs.dims:
        if dim.lower() in ["bus", "buses"] or dim in ["Bus", "Buses"]:
            bus_dim = dim
            break

    if bus_dim:
        total_network_needs = (
            network_needs.sum(dim=bus_dim) / 1e3
        ) * 100  # Convert to % and GW
    else:
        total_network_needs = (network_needs / 1e3) * 100

    # Get first available TY if it exists in the data, otherwise use from dims
    if "TY" in total_network_needs.dims and "CY" in total_network_needs.dims:
        first_ty = (
            total_network_needs.TY.values[0]
            if len(total_network_needs.TY) > 0
            else dims["TY"][0]
        )
        first_cy = (
            total_network_needs.CY.values[0]
            if len(total_network_needs.CY) > 0
            else dims["CY"][0]
        )
        example_data = (
            total_network_needs.isel(TY=0, CY=0)
            if len(total_network_needs.dims) > 1
            else total_network_needs
        )
        title_suffix = f"(TY {first_ty}, CY {first_cy})"
    else:
        example_data = total_network_needs
        first_ty = dims["TY"][0]
        title_suffix = f"(TY {first_ty})"

    # Use snapshot dimension for time axis
    if "snapshot" in example_data.dims:
        hours = np.arange(len(example_data.snapshot))
    else:
        hours = np.arange(len(example_data))

    # Plot network needs timeseries
    ax.plot(
        hours,
        example_data,
        label="Network Needs",
        color="red",
        alpha=0.7,
        linewidth=1.5,
    )

    # Add horizontal line for curtailment target (as percentage)
    curtailment_pct = ct * 100
    ax.axhline(
        y=curtailment_pct,
        color="orange",
        linestyle="--",
        alpha=0.8,
        label=f"Curtailment Target ({curtailment_pct:.1f}%)",
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Percentage [%]")
    ax.set_title(f"Network Needs Overview {title_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_curtailment_comparison(data, ax, dims, colors):
    """
    Plot comparison of original curtailment, adjusted curtailment, and network needs for a single example bus, TY, and CY.

    This function shows per-snapshot curtailment data including:
    1. Original curtailment (before network needs adaptation)
    2. Adjusted curtailment (after network needs adaptation)
    3. Network needs requirements

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing curtailment and network needs data
    ax : matplotlib.axes.Axes
        Axes object to plot on
    dims : dict
        Dictionary containing available dimensions
    colors : dict
        Color mapping dictionary
    """
    ax.set_title(
        "Curtailment Comparison with Network Needs (Example Bus)",
        fontweight="bold",
        pad=20,
    )

    # Check if we have network needs data (only available in dummy storage scenarios)
    if "network_needs" not in data:
        ax.text(
            0.5,
            0.5,
            "Network needs data not available\n(Base scenario only)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    # Select the first TY and CY if present
    sel_dict = {}
    for dim in ["TY", "ty"]:
        if dim in data["network_needs"].dims:
            sel_dict[dim] = data["network_needs"].coords[dim].values[0]
    for dim in ["CY", "cy"]:
        if dim in data["network_needs"].dims:
            sel_dict[dim] = data["network_needs"].coords[dim].values[0]

    # Select the first bus (or BZ) as the example
    bus_dim = None
    for dim in ["bus", "BZ", "Bus"]:
        if dim in data["network_needs"].dims:
            bus_dim = dim
            break
    if bus_dim is None:
        raise ValueError("No bus/BZ dimension found in network_needs data.")
    sel_dict[bus_dim] = data["network_needs"].coords[bus_dim].values[0]
    example_bus = sel_dict[bus_dim]

    # Extract 1D timeseries for the example bus, TY, and CY
    network_needs = data["network_needs"].sel(sel_dict)
    if "original_curtailment_timeseries" in data:
        original_curtailment = data["original_curtailment_timeseries"].sel(sel_dict)
    else:
        original_curtailment = None
    if "adjusted_curtailment_timeseries" in data:
        adjusted_curtailment = data["adjusted_curtailment_timeseries"].sel(sel_dict)
    else:
        adjusted_curtailment = None

    # Create time axis
    if hasattr(network_needs, "hour"):
        time_axis = network_needs.hour.values
        x_label = "Hour of Year"
    elif hasattr(network_needs, "snapshot"):
        time_axis = np.arange(len(network_needs.snapshot))
        x_label = "Snapshot"
    else:
        time_axis = np.arange(network_needs.shape[0])
        x_label = "Time Step"

    # Plot network needs
    ax.plot(
        time_axis,
        network_needs.values.flatten(),
        label=f"Network Needs ({example_bus})",
        color="red",
        alpha=0.8,
        linewidth=1.5,
    )
    # Plot original curtailment
    if original_curtailment is not None:
        ax.plot(
            time_axis,
            original_curtailment.values.flatten(),
            label=f"Original Curtailment ({example_bus})",
            color="blue",
            alpha=0.7,
            linewidth=1.2,
        )
    # Plot adjusted curtailment
    if adjusted_curtailment is not None:
        ax.plot(
            time_axis,
            adjusted_curtailment.values.flatten(),
            label=f"Adjusted Curtailment ({example_bus})",
            color="green",
            alpha=0.8,
            linewidth=1.5,
            linestyle="--",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Curtailment [MW]")
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=3)
    ax.grid(True, alpha=0.3)
    if len(time_axis) > 100:
        n_ticks = min(10, len(time_axis) // 10)
        if n_ticks > 0:
            tick_indices = np.linspace(0, len(time_axis) - 1, n_ticks, dtype=int)
            ax.set_xticks(time_axis[tick_indices])
            ax.tick_params(axis="x", rotation=45)


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="plot_model_outputs",
        configfiles=[
            pathlib.Path("config", "test", "config.eva-quick-test.yaml"),
        ],
    )

    configure_logging(snakemake)

    # Suppress verbose matplotlib font manager debug logs unless logging level is DEBUG
    if logging.getLogger().level != logging.DEBUG:
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Main plotting function
    setup_plotting_style()

    RENEWABLE_CARRIERS = snakemake.params.renewable_carriers

    # Load the flexibility indicators data
    dt = xr.open_datatree(
        snakemake.input.model_outputs,  # ty: ignore[possibly-missing-attribute]
        decode_timedelta=True,
    )

    # Generate three categories of plots
    logger.info("Creating base case plots...")
    create_flexibility_plots(
        dt,
        Path(snakemake.output.base_case),  # ty: ignore[possibly-missing-attribute]
    )

    dummy_data = dt["dummy_storage_optimization"].ds
    dims = get_available_dimensions(dt)

    # Only create dummy storage plots if there are dummy storage outputs
    if snakemake.output.dummy_storage:  # ty: ignore[possibly-missing-attribute]
        for (
            output_file
        ) in snakemake.output.dummy_storage:  # ty: ignore[possibly-missing-attribute]
            output_path = Path(output_file)
            # Extract mh and ct from filename
            filename = output_path.stem
            parts = filename.split("_")
            mh = int(parts[-2])
            ct_str = parts[-1]  # Now in format "ct1", "ct2", etc.

            # Extract the index from ctX format
            ct_index = int(ct_str[2:])  # Extract number after 'ct'

            # Get the actual curtailment value from config
            curtailment_targets = snakemake.config["eva"]["dummy_storage_optimization"][
                "curtailment_targets"
            ]
            ct = curtailment_targets[
                ct_index - 1
            ]  # ct1 -> index 0, ct2 -> index 1, etc.

            logger.info(
                f"Creating plot for dummy storage scenario mh={mh}h, ct={ct * 100:.1f}% (from {ct_str})"
            )

            create_flexibility_plots(
                dt, output_path, is_dummy_storage=True, mh=mh, ct=ct, ct_str=ct_str
            )

    logger.info("Creating comparison plots...")
    create_comparison_plots(
        dt,
        Path(snakemake.output.comparison),  # ty: ignore[possibly-missing-attribute]
    )

    logger.info("Generating catalogue plots...")
    catalogue_dir = Path(
        snakemake.output.catalogue_dir  # ty: ignore[possibly-missing-attribute]
    )
    dataset_path = Path(
        snakemake.input.model_outputs  # ty: ignore[possibly-missing-attribute]
    )

    catalogue_groups: list[str] = ["base"]
    catalogue_datasets = {"base": dt["base"].ds}

    if "dummy_storage_optimization" in dt:
        catalogue_groups.append("dummy_storage_optimization")
        catalogue_datasets["dummy_storage_optimization"] = dt[
            "dummy_storage_optimization"
        ].ds

    generate_catalogue_plots(
        dataset_path,
        catalogue_dir,
        catalogue_groups,
        datasets=catalogue_datasets,
    )
