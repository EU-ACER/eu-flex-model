"""Plot dispatcher for EVA model output catalogues."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import cycle
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

matplotlib.use("Agg")  # Use non-GUI backend to avoid X server issues
plt.style.use("seaborn-v0_8-whitegrid")

Plotter = Callable[[xr.DataArray, str, dict[str, Any]], list[Path]]


def get_label(da: xr.DataArray) -> str:
    """Return a preferred label for a data array."""

    if "label" in da.attrs and da.attrs["label"]:
        return str(da.attrs["label"])
    if "description" in da.attrs and da.attrs["description"]:
        return str(da.attrs["description"])
    if da.name is not None:
        return str(da.name)
    return "value"


def format_with_units(label: str, units: str | None) -> str:
    """Combine a label and its units for axis text."""

    if not units:
        return label
    stripped = units.strip()
    if not stripped:
        return label
    if any(token in label for token in ("[", "(", "{")):
        return label
    return f"{label} [{stripped}]"


def scale_figsize(
    n_rows: int,
    n_cols: int,
    *,
    base_width: float = 4.5,
    base_height: float = 3.0,
    max_width: float = 22.0,
    max_height: float = 18.0,
) -> tuple[float, float]:
    """Scale figure size for grids with sensible bounds."""

    width = base_width * max(1, n_cols)
    height = base_height * max(1, n_rows)
    return min(width, max_width), min(height, max_height)


def reduce_extra_dimensions(da: xr.DataArray, keep: set[str]) -> xr.DataArray:
    """
    Aggregate over dimensions that are not part of the keep set.

    Uses appropriate aggregation based on variable type:
    - Intensive variables (prices, volatilities, ratios): mean
    - Extensive variables (energy, capacity, power): sum
    """
    drop_dims = [dim for dim in da.dims if dim not in keep]
    if not drop_dims:
        return da

    # Determine if this is an intensive variable (should use mean) or extensive (should use sum)
    var_name = da.name if hasattr(da, "name") else ""
    description = da.attrs.get("description", "").lower()
    units = da.attrs.get("units", "").lower()

    # Check if this is an intensive variable
    is_intensive = (
        "price" in str(var_name).lower()
        or "price" in description
        or "€/" in units
        or "eur/" in units
        or "volatility" in str(var_name).lower()
        or "volatility" in description
        or "average" in description  # already averaged quantities
    )

    if is_intensive:
        # Use mean for intensive variables (prices, ratios, etc.)
        return da.mean(drop_dims, skipna=True)
    else:
        # Use sum for extensive variables (energy, capacity, etc.)
        return da.sum(drop_dims, skipna=True)


def iter_plot_axes(grid: Any) -> list[Any]:
    """Flatten axes from a matplotlib grid or seaborn FacetGrid."""

    axes = getattr(grid, "axs", None)
    if axes is None:
        axes = getattr(grid, "axes", None)
    if axes is None:
        return []
    return [ax for ax in np.ravel(axes) if ax is not None]


def apply_suptitle(fig: plt.Figure, title: str) -> None:
    """Apply a super-title while leaving room for subplot titles."""

    fig.subplots_adjust(top=0.85)
    fig.suptitle(title, y=0.97, fontsize=12)


def enforce_integer_ticks(ax: plt.Axes, axis: str = "x") -> None:
    """Force integer ticks on the specified axis."""

    if axis == "x":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    elif axis == "y":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    elif axis == "both":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))


def enable_grid(ax: plt.Axes, x: bool = True, y: bool = True) -> None:
    """Enable a dashed grid with light styling."""

    axis: str
    if x and y:
        axis = "both"
    elif x:
        axis = "x"
    elif y:
        axis = "y"
    else:
        return
    ax.grid(True, axis=axis, linestyle="--", linewidth=0.4, color="k", alpha=1.0)
    for spine in ax.spines.values():
        spine.set_color("k")
        spine.set_linewidth(0.4)


def coerce_coord_to_int(data: xr.DataArray, coord_name: str) -> xr.DataArray:
    """Return a copy with a coordinate coerced to integers when possible."""

    coord = data.coords.get(coord_name)
    if coord is None:
        return data
    values: list[Any] = []
    for value in coord.values.tolist():
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            return data
    return data.assign_coords({coord_name: values})


def aggregate_climate_year(
    da: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Average over climate years and optionally return the standard deviation."""

    if "CY" not in da.dims:
        return da, None
    mean = da.mean("CY")
    std = da.std("CY")
    return mean, std


def resolve_carrier_colors(carriers: list[str], palette: dict[str, str]) -> list[str]:
    """Map carriers to colours using a palette and a fallback cycle."""

    fallback_cycle = cycle(mcolors.TABLEAU_COLORS.values())
    colours: list[str] = []
    for carrier in carriers:
        colour = palette.get(carrier)
        if not colour:
            colour = next(fallback_cycle)
        colours.append(colour)
    return colours


def prepare_dataset(ds: xr.Dataset) -> tuple[xr.Dataset, dict[str, str]]:
    """Attach friendly carrier labels and build a colour palette."""

    if "carrier" not in ds.coords:
        return ds, {}

    carriers = ds.coords["carrier"].to_series()
    colors = ds.get("_color")
    nice_names = ds.get("_nice_name")

    if colors is None or nice_names is None:
        return ds, {}

    raw_colors = colors.to_series().fillna("")
    raw_names = nice_names.to_series().fillna("")

    fallback_cycle = cycle(mcolors.TABLEAU_COLORS.values())
    palette: dict[str, str] = {}
    friendly_labels: list[str] = []

    for carrier in carriers:
        nice = raw_names.get(carrier, "").strip()
        label = nice or carrier
        friendly_labels.append(label)

        raw_color = raw_colors.get(carrier, "").strip()
        colour = raw_color or mcolors.to_hex(next(fallback_cycle))
        palette[label] = colour

    ds = ds.assign_coords(carrier=("carrier", friendly_labels))
    return ds, palette


def ensure_output_dir(plot_root: Path, subdir: str) -> Path:
    """Ensure the catalogue output directory exists."""

    out_dir = plot_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _bz_label(bz: Any) -> str:
    if bz is None:
        return "All BZ"
    return f"BZ {bz}"


def plot_hourly_with_carriers(
    da: xr.DataArray, group: str, opts: dict[str, Any]
) -> list[Path]:
    group_label = opts.get("group_label", group)
    value_label = get_label(da)
    value_label_with_units = format_with_units(value_label, da.attrs.get("units"))

    # Shorten long curtailment labels
    if "Curtailment of renewable generation [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Curtailment of renewable generation [MWh]",
            "Curtailment of renewable\ngeneration [MWh]",
        )
    # Shorten long short-term shortfall curtailment labels
    elif "Maximum short-term shortfall curtailment [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Maximum short-term shortfall curtailment [MWh]",
            "Maximum short-term\nshortfall curtailment [MWh]",
        )
    elif "Total short-term shortfall curtailment [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Total short-term shortfall curtailment [MWh]",
            "Total short-term\nshortfall curtailment [MWh]",
        )
    # Shorten long short-term shortfall energy not served labels
    elif "Total short-term shortfall energy not served [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Total short-term shortfall energy not served [MWh]",
            "Total short-term shortfall\nenergy not served [MWh]",
        )
    elif "Short-term shortfall energy not served [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Short-term shortfall energy not served [MWh]",
            "Short-term shortfall\nenergy not served [MWh]",
        )
    # Shorten long net power transfer labels
    elif "Net power transfer between bidding zones" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Net power transfer between bidding zones", "Net power transfer"
        )
    # Shorten long ramping shortfall indicator labels
    elif (
        "Ramping shortfall indicator (difference in curtailment + ENS with vs without ramping constraints)"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Ramping shortfall indicator (difference in curtailment + ENS with vs without ramping constraints)",
            "Ramping shortfall indicator",
        )
    # Shorten long residual load labels
    elif (
        "Residual load (demand - potential renewable supply - must-run supply)"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Residual load (demand - potential renewable supply - must-run supply)",
            "Residual load",
        )

    # Remove (day_ahead) and (intraday) from reserve capacity labels
    if "(day_ahead)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (day_ahead)", "")
    if "(intraday)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (intraday)", "")

    palette: dict[str, str] = opts.get("palette", {})
    plot_root: Path = opts["plot_root"]
    mean, std = aggregate_climate_year(da)
    mean = coerce_coord_to_int(mean, "TY")
    if std is not None:
        std = coerce_coord_to_int(std, "TY")

    bz_coord = mean.coords.get("BZ")
    bz_list = bz_coord.values.tolist() if bz_coord is not None else [None]
    ty_coord = mean.coords.get("TY")
    ty_list = ty_coord.values.tolist() if ty_coord is not None else [None]
    carrier_coord = mean.coords.get("carrier")
    carriers = (
        [str(value) for value in carrier_coord.values.tolist()]
        if carrier_coord is not None
        else []
    )
    hour_coord = mean.coords.get("hour")
    if hour_coord is not None:
        hours = hour_coord.values
    else:
        hours = np.arange(mean.sizes.get("hour", 0))

    color_map = {
        carrier: colour
        for carrier, colour in zip(carriers, resolve_carrier_colors(carriers, palette))
    }

    n_rows = max(1, len(bz_list))
    n_cols = max(1, len(ty_list))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=scale_figsize(
            n_rows,
            n_cols,
            base_width=4.8,
            base_height=2.8,
            max_width=22.0,
            max_height=18.0,
        ),
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for row, bz in enumerate(bz_list):
        for col, ty in enumerate(ty_list):
            ax = axes[row, col]
            subset = mean
            subset_std = std
            if ty is not None:
                subset = subset.sel(TY=ty)
                if subset_std is not None:
                    subset_std = subset_std.sel(TY=ty)
            if bz is not None:
                subset = subset.sel(BZ=bz)
                if subset_std is not None:
                    subset_std = subset_std.sel(BZ=bz)
            subset = subset.squeeze()
            if subset_std is not None:
                subset_std = subset_std.squeeze()

            for carrier in carriers:
                series = reduce_extra_dimensions(subset.sel(carrier=carrier), {"hour"})
                color = color_map.get(carrier)
                (line,) = ax.plot(
                    hours,
                    series.values,
                    label=carrier,
                    color=color,
                    linewidth=1.2,
                )
                if carrier not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(carrier)
                if subset_std is not None:
                    spread = reduce_extra_dimensions(
                        subset_std.sel(carrier=carrier), {"hour"}
                    )
                    if np.all(np.isfinite(spread.values)):
                        ax.fill_between(
                            hours,
                            series.values - spread.values,
                            series.values + spread.values,
                            color=color,
                            alpha=0.18,
                        )

            if row == n_rows - 1:
                ax.set_xlabel("Hour", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{_bz_label(bz)}\n{value_label_with_units}", fontsize=8)
            title = "Target year"
            if ty is not None:
                title = f"Target year = {ty}"
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            enable_grid(ax, x=True, y=True)

    if legend_handles:
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            ncol=1,
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=8,
            framealpha=0.9,
        )
        for line in legend.legend_handles:
            colour = color_map.get(
                line.get_label()  # ty: ignore[possibly-missing-attribute]
            )
            if colour:
                line.set_color(colour)  # ty: ignore[unresolved-attribute]

    fig.tight_layout(rect=(0, 0, 1, 0.89))
    apply_suptitle(fig, f"{group_label}: {value_label} (hourly)")

    out_dir = ensure_output_dir(plot_root, group)
    out_path = out_dir / f"{da.name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def plot_hourly_no_carrier(
    da: xr.DataArray, group: str, opts: dict[str, Any]
) -> list[Path]:
    group_label = opts.get("group_label", group)
    value_label = get_label(da)
    value_label_with_units = format_with_units(value_label, da.attrs.get("units"))

    # Shorten long curtailment labels
    if "Curtailment of renewable generation [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Curtailment of renewable generation [MWh]",
            "Curtailment of renewable\ngeneration [MWh]",
        )
    # Shorten long short-term shortfall curtailment labels
    elif "Maximum short-term shortfall curtailment [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Maximum short-term shortfall curtailment [MWh]",
            "Maximum short-term\nshortfall curtailment [MWh]",
        )
    elif "Total short-term shortfall curtailment [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Total short-term shortfall curtailment [MWh]",
            "Total short-term\nshortfall curtailment [MWh]",
        )
    # Shorten long short-term shortfall energy not served labels
    elif "Total short-term shortfall energy not served [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Total short-term shortfall energy not served [MWh]",
            "Total short-term shortfall\nenergy not served [MWh]",
        )
    elif "Short-term shortfall energy not served [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Short-term shortfall energy not served [MWh]",
            "Short-term shortfall\nenergy not served [MWh]",
        )
    # Shorten long net power transfer labels
    elif "Net power transfer between bidding zones" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Net power transfer between bidding zones", "Net power transfer"
        )
    # Shorten long ramping shortfall indicator labels
    elif (
        "Ramping shortfall indicator (difference in curtailment + ENS with vs without ramping constraints)"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Ramping shortfall indicator (difference in curtailment + ENS with vs without ramping constraints)",
            "Ramping shortfall indicator",
        )
    # Shorten long residual load labels
    elif (
        "Residual load (demand - potential renewable supply - must-run supply)"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Residual load (demand - potential renewable supply - must-run supply)",
            "Residual load",
        )

    # Remove (day_ahead) and (intraday) from reserve capacity labels
    if "(day_ahead)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (day_ahead)", "")
    if "(intraday)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (intraday)", "")

    plot_root: Path = opts["plot_root"]
    mean, std = aggregate_climate_year(da)
    mean = coerce_coord_to_int(mean, "TY")
    if std is not None:
        std = coerce_coord_to_int(std, "TY")

    bz_coord = mean.coords.get("BZ")
    bz_list = bz_coord.values.tolist() if bz_coord is not None else [None]
    ty_coord = mean.coords.get("TY")
    ty_list = ty_coord.values.tolist() if ty_coord is not None else [None]
    hour_coord = mean.coords.get("hour")
    if hour_coord is not None:
        hours = hour_coord.values
    else:
        hours = np.arange(mean.sizes.get("hour", 0))

    n_rows = max(1, len(bz_list))
    n_cols = max(1, len(ty_list))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=scale_figsize(
            n_rows,
            n_cols,
            base_width=4.5,
            base_height=2.8,
            max_width=20.0,
            max_height=16.0,
        ),
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    for row, bz in enumerate(bz_list):
        for col, ty in enumerate(ty_list):
            ax = axes[row, col]
            subset = mean
            subset_std = std
            if ty is not None:
                subset = subset.sel(TY=ty)
                if subset_std is not None:
                    subset_std = subset_std.sel(TY=ty)
            if bz is not None:
                subset = subset.sel(BZ=bz)
                if subset_std is not None:
                    subset_std = subset_std.sel(BZ=bz)
            subset = reduce_extra_dimensions(subset.squeeze(), {"hour"})
            values = subset.values
            ax.plot(hours, values, color="#0f4c81", linewidth=1.3)
            if subset_std is not None:
                spread = reduce_extra_dimensions(subset_std.squeeze(), {"hour"})
                spread_vals = spread.values
                if np.all(np.isfinite(spread_vals)):
                    ax.fill_between(
                        hours,
                        values - spread_vals,
                        values + spread_vals,
                        color="#0f4c81",
                        alpha=0.18,
                    )
            if row == n_rows - 1:
                ax.set_xlabel("Hour", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{_bz_label(bz)}\n{value_label_with_units}", fontsize=8)
            title = "Target year"
            if ty is not None:
                title = f"Target year = {ty}"
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            enable_grid(ax, x=True, y=True)

    fig.tight_layout(rect=(0, 0, 1, 0.89))
    apply_suptitle(fig, f"{group_label}: {value_label} (hourly)")

    out_dir = ensure_output_dir(plot_root, group)
    out_path = out_dir / f"{da.name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def plot_period_needs(da: xr.DataArray, group: str, opts: dict[str, Any]) -> list[Path]:
    group_label = opts.get("group_label", group)
    value_label = get_label(da)
    value_label_with_units = format_with_units(value_label, da.attrs.get("units"))

    # Shorten long flexibility need labels - try broader matching first
    if "flexibility need" in value_label_with_units and "(" in value_label_with_units:
        if "Daily" in value_label_with_units:
            value_label_with_units = "Daily flexibility need"
        elif "Weekly" in value_label_with_units:
            value_label_with_units = "Weekly flexibility need"
        elif "Seasonal" in value_label_with_units:
            value_label_with_units = "Seasonal flexibility need"

    # Shorten long total generation labels
    if (
        "Total generation by technology per day (day_ahead) [MWh]"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Total generation by technology per day (day_ahead) [MWh]",
            "Total generation by technology per day [MWh]",
        )
    elif (
        "Total generation by technology per day (intraday) [MWh]"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Total generation by technology per day (intraday) [MWh]",
            "Total generation by technology per day [MWh]",
        )

    # Remove (day_ahead) and (intraday) from reserve capacity labels
    if "(day_ahead)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (day_ahead)", "")
    if "(intraday)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (intraday)", "")

    plot_root: Path = opts["plot_root"]
    period = "day" if "day" in da.dims else "week"
    mean, std = aggregate_climate_year(da)
    mean = coerce_coord_to_int(mean, "TY")
    if std is not None:
        std = coerce_coord_to_int(std, "TY")

    bz_coord = mean.coords.get("BZ")
    bz_list = bz_coord.values.tolist() if bz_coord is not None else [None]
    ty_coord = mean.coords.get("TY")
    ty_list = ty_coord.values.tolist() if ty_coord is not None else [None]
    period_coord = mean.coords.get(period)
    if period_coord is not None:
        period_values = period_coord.values
    else:
        period_values = np.arange(mean.sizes.get(period, 0))

    n_rows = max(1, len(bz_list))
    n_cols = max(1, len(ty_list))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=scale_figsize(
            n_rows,
            n_cols,
            base_width=4.0,
            base_height=2.8,
            max_width=18.0,
            max_height=16.0,
        ),
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    for row, bz in enumerate(bz_list):
        for col, ty in enumerate(ty_list):
            ax = axes[row, col]
            subset = mean
            subset_std = std
            if ty is not None:
                subset = subset.sel(TY=ty)
                if subset_std is not None:
                    subset_std = subset_std.sel(TY=ty)
            if bz is not None:
                subset = subset.sel(BZ=bz)
                if subset_std is not None:
                    subset_std = subset_std.sel(BZ=bz)
            subset = reduce_extra_dimensions(subset.squeeze(), {period})
            values = subset.values
            ax.plot(period_values, values, color="#0f4c81", linewidth=1.3)
            if subset_std is not None:
                spread = reduce_extra_dimensions(subset_std.squeeze(), {period})
                spread_vals = spread.values
                if np.all(np.isfinite(spread_vals)):
                    ax.fill_between(
                        period_values,
                        values - spread_vals,
                        values + spread_vals,
                        color="#0f4c81",
                        alpha=0.18,
                    )
            if row == n_rows - 1:
                ax.set_xlabel(period.capitalize(), fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{_bz_label(bz)}\n{value_label_with_units}", fontsize=8)
            title = "Target year"
            if ty is not None:
                title = f"Target year = {ty}"
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
            enforce_integer_ticks(ax, axis="x")
            enable_grid(ax, x=True, y=True)

    fig.tight_layout(rect=(0, 0, 1, 0.89))
    periodly = period + "ly" if period != "day" else "daily"
    apply_suptitle(fig, f"{group_label}: {value_label} ({periodly})")

    out_dir = ensure_output_dir(plot_root, group)
    out_path = out_dir / f"{da.name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def plot_carrier_bars(da: xr.DataArray, group: str, opts: dict[str, Any]) -> list[Path]:
    group_label = opts.get("group_label", group)
    value_label = get_label(da)
    value_label_with_units = format_with_units(value_label, da.attrs.get("units"))

    # Remove (day_ahead) and (intraday) from reserve capacity labels
    if "(day_ahead)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (day_ahead)", "")
    if "(intraday)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (intraday)", "")

    palette: dict[str, str] = opts.get("palette", {})
    plot_root: Path = opts["plot_root"]
    out_dir = ensure_output_dir(plot_root, group)

    mean, std = aggregate_climate_year(da)
    mean = coerce_coord_to_int(mean, "TY")
    if std is not None:
        std = coerce_coord_to_int(std, "TY")

    bz_coord = mean.coords.get("BZ")
    bz_list = bz_coord.values.tolist() if bz_coord is not None else [None]
    ty_coord = mean.coords.get("TY")
    ty_list = ty_coord.values.tolist() if ty_coord is not None else [None]
    carrier_coord = mean.coords.get("carrier")
    carriers = (
        [str(value) for value in carrier_coord.values.tolist()]
        if carrier_coord is not None
        else []
    )

    n_rows = max(1, len(bz_list))
    n_cols = max(1, len(ty_list))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=scale_figsize(
            n_rows,
            n_cols,
            base_width=4.0,
            base_height=3.2,
            max_width=20.0,
            max_height=18.0,
        ),
        sharex="col",
        sharey="row",
        squeeze=False,
    )

    carrier_colors = resolve_carrier_colors(carriers, palette)

    for row, bz in enumerate(bz_list):
        for col, ty in enumerate(ty_list):
            ax = axes[row, col]
            subset = mean
            subset_std = std
            if ty is not None:
                subset = subset.sel(TY=ty)
                if subset_std is not None:
                    subset_std = subset_std.sel(TY=ty)
            if bz is not None:
                subset = subset.sel(BZ=bz)
                if subset_std is not None:
                    subset_std = subset_std.sel(BZ=bz)
            subset = reduce_extra_dimensions(subset.squeeze(), {"carrier"})
            values = subset.values
            if subset_std is not None:
                subset_std = reduce_extra_dimensions(subset_std.squeeze(), {"carrier"})
                errors = subset_std.values
            else:
                errors = None
            y = np.arange(len(carriers))
            ax.barh(
                y,
                values,
                color=carrier_colors,
                xerr=errors,
                capsize=3,
                error_kw={"elinewidth": 0.4},
            )
            ax.set_yticks(y)
            ax.set_yticklabels(carriers, fontsize=6)
            if col == 0:
                ax.set_ylabel(_bz_label(bz), fontsize=7)
            if row == n_rows - 1:
                ax.set_xlabel(value_label_with_units, fontsize=6)
            title = "Target year"
            if ty is not None:
                title = f"Target year = {ty}"
            ax.set_title(title, fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
            ax.tick_params(axis="x", which="major", labelsize=7)
            enable_grid(ax, x=True, y=False)

    fig.tight_layout(rect=(0, 0, 1, 0.89))
    apply_suptitle(fig, f"{group_label}: {value_label}")

    out_path = out_dir / f"{da.name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


def plot_scalar_surface(
    da: xr.DataArray, group: str, opts: dict[str, Any]
) -> list[Path]:
    group_label = opts.get("group_label", group)
    value_label = get_label(da)
    value_label_with_units = format_with_units(value_label, da.attrs.get("units"))

    # Shorten long crossborder labels
    if "technology contributions to crossborder flows" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "technology contributions to crossborder flows", "Technology contributions"
        )
    elif "short-term contributions to crossborder flows" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "short-term contributions to crossborder flows", "Short-term contributions"
        )
    # Shorten long ramping requirement labels
    elif (
        "Maximum ramping requirement (largest hour-to-hour change in residual load)"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Maximum ramping requirement (largest hour-to-hour change in residual load)",
            "Maximum ramping requirement",
        )
    # Shorten long net power transfer labels
    elif "Net power transfer between bidding zones" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Net power transfer between bidding zones", "Net power transfer"
        )
    # Shorten long seasonal flexibility need labels
    elif (
        "Seasonal flexibility need (sum of absolute deviations from seasonal mean residual load)"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Seasonal flexibility need (sum of absolute deviations from seasonal mean residual load)",
            "Seasonal flexibility need",
        )
    # Shorten long energy not served labels
    elif "Maximum short-term shortfall energy not served" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Maximum short-term shortfall energy not served", "Energy not served"
        )
    # Shorten long short-term shortfall curtailment labels
    elif "Maximum short-term shortfall curtailment [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Maximum short-term shortfall curtailment [MWh]",
            "Maximum short-term\nshortfall curtailment [MWh]",
        )
    elif "Total short-term shortfall curtailment [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Total short-term shortfall curtailment [MWh]",
            "Total short-term\nshortfall curtailment [MWh]",
        )
    # Shorten long total short-term shortfall energy not served labels
    elif "Total short-term shortfall energy not served [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Total short-term shortfall energy not served [MWh]",
            "Total short-term shortfall\nenergy not served [MWh]",
        )
    elif "Short-term shortfall energy not served [MWh]" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(
            "Short-term shortfall energy not served [MWh]",
            "Short-term shortfall\nenergy not served [MWh]",
        )
    # Shorten long total generation labels
    elif (
        "Total generation by technology per day (day_ahead) [MWh]"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Total generation by technology per day (day_ahead) [MWh]",
            "Total generation by technology per day [MWh]",
        )
    elif (
        "Total generation by technology per day (intraday) [MWh]"
        in value_label_with_units
    ):
        value_label_with_units = value_label_with_units.replace(
            "Total generation by technology per day (intraday) [MWh]",
            "Total generation by technology per day [MWh]",
        )

    # Remove (day_ahead) and (intraday) from reserve capacity labels
    if "(day_ahead)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (day_ahead)", "")
    if "(intraday)" in value_label_with_units:
        value_label_with_units = value_label_with_units.replace(" (intraday)", "")

    plot_root: Path = opts["plot_root"]
    out_dir = ensure_output_dir(plot_root, group)

    mean, std = aggregate_climate_year(da)
    mean = coerce_coord_to_int(mean, "TY")
    if std is not None:
        std = coerce_coord_to_int(std, "TY")

    bz_coord = mean.coords.get("BZ")
    bz_list = bz_coord.values.tolist() if bz_coord is not None else [None]
    ty_coord = mean.coords.get("TY")
    ty_list = ty_coord.values.tolist() if ty_coord is not None else []

    n_rows = max(1, len(bz_list))
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=scale_figsize(
            n_rows,
            1,
            base_width=5.5,
            base_height=3.0,
            max_width=10.0,
            max_height=18.0,
        ),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row, bz in enumerate(bz_list):
        ax = axes[row, 0]
        subset = mean
        subset_std = std
        if bz is not None:
            subset = subset.sel(BZ=bz)
            if subset_std is not None:
                subset_std = subset_std.sel(BZ=bz)
        values = reduce_extra_dimensions(subset.squeeze(), {"TY"})
        ax.plot(ty_list, values.values, marker="o", color="#0f4c81", linewidth=1.4)
        if subset_std is not None:
            spread = reduce_extra_dimensions(subset_std.squeeze(), {"TY"})
            spread_vals = spread.values
            if np.all(np.isfinite(spread_vals)):
                ax.fill_between(
                    ty_list,
                    values.values - spread_vals,
                    values.values + spread_vals,
                    color="#0f4c81",
                    alpha=0.18,
                )
        ax.set_ylabel(value_label_with_units, fontsize=8)
        ax.set_title(_bz_label(bz), fontsize=9)
        # Set reasonable x-axis limits based on actual target years
        if ty_list:
            if len(ty_list) == 1:
                # Single year: use ±1 year margin
                margin = 1
            else:
                margin = (max(ty_list) - min(ty_list)) * 0.05
            ax.set_xlim(min(ty_list) - margin, max(ty_list) + margin)
        ax.tick_params(axis="both", which="major", labelsize=7)
        enforce_integer_ticks(ax)
        enable_grid(ax, x=True, y=True)

    axes[-1, 0].set_xlabel("Target year", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.89))
    apply_suptitle(fig, f"{group_label}: {value_label}")

    out_path = out_dir / f"{da.name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


DISPATCH_TABLE: list[tuple[Callable[[xr.DataArray], bool], Plotter]] = [
    (lambda da: {"hour", "carrier"}.issubset(da.dims), plot_hourly_with_carriers),
    (lambda da: "hour" in da.dims and "carrier" not in da.dims, plot_hourly_no_carrier),
    (lambda da: any(dim in da.dims for dim in ("day", "week")), plot_period_needs),
    (
        lambda da: "carrier" in da.dims
        and "hour" not in da.dims
        and not any(dim in da.dims for dim in ("day", "week")),
        plot_carrier_bars,
    ),
    (
        lambda da: set(da.dims).issubset({"TY", "CY", "BZ"}) and "TY" in da.dims,
        plot_scalar_surface,
    ),
]


def dispatch_plot(da: xr.DataArray, group: str, opts: dict[str, Any]) -> list[Path]:
    for predicate, plotter in DISPATCH_TABLE:
        if predicate(da):
            return plotter(da, group, opts)
    return []


def catalogue_group(
    dataset_group: str,
    output_subdir: str,
    *,
    dataset_path: Path | None,
    plot_root: Path,
    selectors: Mapping[str, Any] | None = None,
    label: str | None = None,
    dataset: xr.Dataset | None = None,
) -> dict[str, list[Path]]:
    """Generate catalogue plots for a dataset group and optional slice."""

    group_label = label or output_subdir
    outputs: dict[str, list[Path]] = {}

    ds: xr.Dataset | None = dataset
    close_dataset = False
    if ds is None:
        if dataset_path is None:
            raise ValueError("dataset_path is required when no dataset is provided")
        ds = xr.open_dataset(dataset_path, group=dataset_group, decode_timedelta=True)
        close_dataset = True

    try:
        subset = ds.sel(selectors, drop=True) if selectors else ds
        subset, palette = prepare_dataset(subset)

        for name, data_array in subset.data_vars.items():
            if name in {"_color", "_nice_name"}:
                continue
            saved = dispatch_plot(
                data_array,
                output_subdir,
                {
                    "palette": palette,
                    "group_label": group_label,
                    "plot_root": plot_root,
                },
            )
            if saved:
                outputs[str(name)] = saved
    finally:
        if close_dataset and ds is not None:
            ds.close()

    return outputs


def _iter_dummy_storage_selectors(
    dataset_path: Path | None,
    dataset: xr.Dataset | None = None,
) -> list[dict[str, Any]]:
    """Return selector dictionaries for available dummy storage scenarios."""

    ds: xr.Dataset | None = dataset
    close_dataset = False
    if ds is None:
        if dataset_path is None:
            return []
        try:
            ds = xr.open_dataset(
                dataset_path, group="dummy_storage_optimization", decode_timedelta=True
            )
        except ValueError:
            return []
        close_dataset = True

    mh_coord = ds.coords.get("mh")
    ct_coord = ds.coords.get("ct")
    if mh_coord is None or ct_coord is None:
        if close_dataset and ds is not None:
            ds.close()
        return []

    mh_values = mh_coord.values.tolist()
    ct_values = ct_coord.values.tolist()

    if close_dataset and ds is not None:
        ds.close()

    selectors: list[dict[str, Any]] = []
    for mh_val in mh_values:
        for ct_val in ct_values:
            selectors.append({"mh": mh_val, "ct": ct_val})
    return selectors


def generate_catalogue_plots(
    dataset_path: Path | None,
    plot_root: Path,
    groups: Iterable[str] | None = None,
    datasets: Mapping[str, xr.Dataset] | None = None,
) -> dict[str, dict[str, list[Path]]]:
    """Generate catalogue plots for the selected dataset groups."""

    if dataset_path is None and datasets is None:
        raise ValueError(
            "Provide dataset_path or a datasets mapping for catalogue generation"
        )

    plot_root.mkdir(parents=True, exist_ok=True)

    if groups is None:
        selected_groups = ("base",)
    elif isinstance(groups, str):
        selected_groups = (groups,)
    else:
        selected_groups = tuple(groups)

    catalogue: dict[str, dict[str, list[Path]]] = {}

    for group in selected_groups:
        dataset = datasets.get(group) if datasets is not None else None
        if group == "dummy_storage_optimization":
            scenario_catalogue: dict[str, dict[str, list[Path]]] = {}
            for selectors in _iter_dummy_storage_selectors(dataset_path, dataset):
                if not selectors:
                    continue
                mh_str = str(selectors["mh"])
                ct_str = str(selectors["ct"])
                subdir = f"{group}/mh-{mh_str}_ct-{ct_str}"
                label = f"Dummy storage optimisation (mh={mh_str}, ct={ct_str})"
                outputs = catalogue_group(
                    group,
                    subdir,
                    dataset_path=dataset_path,
                    plot_root=plot_root,
                    selectors=selectors,
                    label=label,
                    dataset=dataset,
                )
                scenario_catalogue[f"mh={mh_str}_ct={ct_str}"] = outputs
            if scenario_catalogue:
                catalogue[group] = scenario_catalogue  # ty: ignore[invalid-assignment]
            continue

        try:
            outputs = catalogue_group(
                group,
                group,
                dataset_path=dataset_path,
                plot_root=plot_root,
                dataset=dataset,
            )
        except ValueError:
            continue
        catalogue[group] = outputs

    return catalogue
