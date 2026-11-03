import logging
from datetime import timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.helpers.date_helpers import drop_feb_29
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)

COMPRESSION = {"zlib": True, "complevel": 4}


def handle_transition_period_end(
    timestamp: pd.Timestamp,
    transition_end_date: pd.Timestamp | None,
    pending_events: int,
) -> tuple[int, bool, pd.Timestamp | None]:
    """
    Handle the end of a transition period for price cap events.

    Checks if the current timestamp is past the end of a transition period.
    If so, it returns the pending events and signals the end of the transition.

    Parameters
    ----------
    timestamp : pd.Timestamp
        The current timestamp being evaluated.
    transition_end_date : pd.Timestamp or None
        The timestamp marking the end of the current transition period.
        If None, it indicates that there is no active transition period.
    pending_events : int
        The number of events to be triggered once the transition period concludes.

    Returns
    -------
    tuple[int, bool, pd.Timestamp or None]
        A tuple containing:
        - int: The number of events to add (equal to `pending_events` if the
          transition period has ended, otherwise 0).
        - bool: A flag indicating if a transition period is still active
          (True if `transition_end_date` is not None, False otherwise).
        - pd.Timestamp or None: The updated `transition_end_date`. It is reset
          to None if the period has ended.
    """
    is_transition_active = transition_end_date is not None

    if (
        is_transition_active
        and transition_end_date is not None
        and timestamp > transition_end_date
    ):
        return pending_events, False, None

    return 0, is_transition_active, transition_end_date


def detect_and_handle_spikes(
    timestamp: pd.Timestamp,
    timestamp_prices: pd.Series,
    spike_threshold: float,
    active_spikes: list,
    in_transition_period: bool,
) -> tuple[list, bool, pd.Timestamp | None]:
    """
    Detect price spikes and trigger events if conditions are met.

    This function checks if any prices at the given timestamp exceed the spike
    threshold. If so, and if not in a transition period, it records the spike.
    It maintains a rolling list of spikes from the last 30 days. If spikes
    occur on at least two unique days within this window, it triggers a price
    cap event and returns the start of a new transition period.

    Parameters
    ----------
    timestamp : pd.Timestamp
        The current timestamp being evaluated.
    timestamp_prices : pd.Series
        A Series of prices for the current timestamp, indexed by bus.
    spike_threshold : float
        The price level above which a price is considered a spike.
    active_spikes : list
        A list of previously detected spikes, where each spike is a tuple of
        (date, bus, price).
    in_transition_period : bool
        A flag indicating if the system is currently in a transition period.

    Returns
    -------
    tuple[list, bool, pd.Timestamp or None]
        A tuple containing:
        - list: The updated list of active spikes.
        - bool: A flag indicating if a new price cap event was triggered.
        - pd.Timestamp or None: The end date for the new transition period if
          an event was triggered, otherwise None.
    """
    current_date = timestamp.date()
    spike_mask = timestamp_prices >= spike_threshold
    spike_buses = timestamp_prices[spike_mask].index.tolist()

    if not in_transition_period and spike_buses:
        for bus in spike_buses:
            active_spikes.append((current_date, bus, timestamp_prices[bus]))

        cutoff_date = current_date - timedelta(days=30)
        active_spikes = [
            (date, bus, price)
            for date, bus, price in active_spikes
            if date >= cutoff_date
        ]

        unique_spike_dates = {date for date, _, _ in active_spikes}
        if len(unique_spike_dates) >= 2:
            logging.info(
                f"Price cap event triggered at {timestamp}. "
                f"Will increment count after transition period."
            )
            return [], True, timestamp + timedelta(days=28)

    return active_spikes, False, None


def update_price_cap(
    n: pypsa.Network,
    starting_price_or: float,
    starting_price_from_years: float,
    increment: float,
    maximum_price_cap: float,
) -> tuple[float, int, pd.Series]:
    """
    Update the marginal price of the buses based on HMMCP (Harmonised Maximum Market Clearing Price) methodology.

    This function implements ACER's HMMCP methodology as defined in Annex 1:
    - Price cap increases by increment when clearing price exceeds 70% of ORIGINAL harmonised maximum
      clearing price in at least 2 market time units in at least 2 different days within 30 rolling days
    - Events are detected across ALL bidding zones (not per individual zone)
    - After an event, a 28-day transition period applies where price is capped at pre-event value
    - Events during transition periods are ignored

    Parameters
    ----------
    n : pypsa.Network
        The network containing marginal prices in buses_t['marginal_price']
    starting_price_or : float
        Initial harmonised maximum clearing price for SDAC (EUR/MWh)
    starting_price_from_years : float
        Reference year for the starting price (not used in current implementation)
    increment : float
        Amount to increase price cap when event occurs (EUR/MWh), typically 500
    maximum_price_cap: float
        Maximum allowable price cap (EUR/MWh)

    This function modifies the network in place.
    """
    marginal_prices = n.buses_t["marginal_price"]
    event_count = 0
    transition_end_date = None
    active_spikes = []
    spike_threshold = starting_price_or * 0.7
    sorted_timestamps = marginal_prices.index.sort_values()
    price_cap_timeseries = pd.Series(index=sorted_timestamps, dtype=float)

    for timestamp in sorted_timestamps:
        events_to_add, in_transition_period, transition_end_date = (
            handle_transition_period_end(timestamp, transition_end_date, 1)
        )
        event_count += events_to_add

        current_price_cap = min(
            starting_price_from_years + (event_count * increment), maximum_price_cap
        )
        price_cap_timeseries.loc[timestamp] = current_price_cap

        timestamp_prices = marginal_prices.loc[timestamp].copy()

        active_spikes, event_triggered, new_transition_end = detect_and_handle_spikes(
            timestamp,
            timestamp_prices,
            spike_threshold,
            active_spikes,
            in_transition_period,
        )
        if event_triggered:
            transition_end_date = new_transition_end

        update_mask = abs(timestamp_prices - starting_price_or) <= 1e-2
        timestamp_prices[update_mask] = current_price_cap
        marginal_prices.loc[timestamp] = timestamp_prices

    n.buses_t["marginal_price"] = marginal_prices
    final_price_cap = min(
        starting_price_from_years + (event_count * increment), maximum_price_cap
    )
    logging.info(
        f"Price cap update completed. Processed {len(marginal_prices.columns)} buses. "
        f"Final price cap: {final_price_cap} EUR/MWh, Events triggered: {event_count}"
    )

    return final_price_cap, event_count, price_cap_timeseries


def reconstruct_price_cap_series(
    centroid_ts: pd.Series,
    week_clusters: pd.DataFrame,
    cy: str,
    default_cap_value: float,
) -> pd.Series:
    """
    Reconstruct a full-year price cap series from a centroid-only series.
    """
    # 1. Construct the full year hourly index
    full_index = pd.date_range(
        start=f"{cy}-01-01 00:00:00", end=f"{cy}-12-31 23:00:00", freq="h"
    )
    # Drop Feb 29 so this stays 8760h, matching n.snapshots (which is always
    # leap-day-free) regardless of whether `cy` is a leap year. Without this,
    # a leap `cy` produces an 8784-length series that later fails to align
    # against an 8760-length reserve-price mask in compute_npv.py.
    full_index = drop_feb_29(full_index)

    # Preserve timezone if centroid_ts index is timezone-aware
    if centroid_ts.index.tz is not None:
        full_index = full_index.tz_localize(centroid_ts.index.tz)

    # Initialize the full-year series with NaN to allow ffill/bfill to propagate values to unmapped gaps
    reconstructed = pd.Series(float("nan"), index=full_index)
    reconstructed.index.name = "snapshot"

    # 2. Filter week clusters for the target climate year
    wc_cy = week_clusters[week_clusters["cy"].astype(str) == str(cy)]

    # 3. Build a mapping of cluster_id -> centroid values
    centroid_values = {}
    for cluster_id in week_clusters["cluster_id"].unique():
        centroid_rows = week_clusters[
            (week_clusters["cluster_id"] == cluster_id)
            & (week_clusters["centroid_flag"])
        ]
        if centroid_rows.empty:
            logging.warning(f"No centroid found for cluster {cluster_id}")
            continue

        c_row = centroid_rows.iloc[0]
        c_start = pd.Timestamp(c_row["chunk_start"])
        if c_start.tz is not None:
            c_start = c_start.tz_localize(None)
        c_end = pd.Timestamp(c_row["chunk_end"])
        if c_end.tz is not None:
            c_end = c_end.tz_localize(None)

        # Extract centroid values by filtering index within chunk_start/chunk_end
        if centroid_ts.index.tz is not None:
            centroid_index_naive = centroid_ts.index.tz_localize(None)
        else:
            centroid_index_naive = centroid_ts.index

        c_mask = (centroid_index_naive >= c_start) & (centroid_index_naive <= c_end)
        c_vals = centroid_ts[c_mask].values

        centroid_values[cluster_id] = c_vals

    # 4. Map each chunk in the target climate year
    for _, row in wc_cy.iterrows():
        chunk_start = pd.Timestamp(row["chunk_start"])
        if chunk_start.tz is not None:
            chunk_start = chunk_start.tz_localize(None)
        chunk_end = pd.Timestamp(row["chunk_end"])
        if chunk_end.tz is not None:
            chunk_end = chunk_end.tz_localize(None)

        cluster_id = int(row["cluster_id"])

        if cluster_id not in centroid_values:
            continue

        c_vals = centroid_values[cluster_id]
        if len(c_vals) == 0:
            logging.warning(f"Centroid values empty for cluster {cluster_id}")
            continue

        # Get target snapshots in the reconstructed series
        if reconstructed.index.tz is not None:
            reconstructed_index_naive = reconstructed.index.tz_localize(None)
        else:
            reconstructed_index_naive = reconstructed.index

        chunk_mask = (reconstructed_index_naive >= chunk_start) & (
            reconstructed_index_naive <= chunk_end
        )
        chunk_snaps = reconstructed.index[chunk_mask]

        if len(chunk_snaps) == 0:
            continue

        if len(chunk_snaps) == len(c_vals):
            reconstructed.loc[chunk_snaps] = c_vals
        else:
            if len(chunk_snaps) < len(c_vals):
                reconstructed.loc[chunk_snaps] = c_vals[: len(chunk_snaps)]
            else:
                padded_vals = list(c_vals)
                while len(padded_vals) < len(chunk_snaps):
                    padded_vals.append(c_vals[-1])
                reconstructed.loc[chunk_snaps] = padded_vals

    # Forward-fill and backward-fill any unmapped gaps (e.g. trailing snapshots)
    reconstructed = reconstructed.ffill().bfill()
    # Finally, fill any remaining NaNs (if the entire series was NaN) with default_cap_value
    reconstructed = reconstructed.fillna(default_cap_value)
    return reconstructed


def calculate_starting_price_ty(
    method: str,
    current_ty: int,
    previous_ty: int,
    increment: float,
    events: int,
    final_price_previous: float,
    initial_price_or: float,
    expected_final_price_previous: float | None = None,
    expected_events_previous: float | None = None,
) -> float:
    """
    Calculate the starting price cap for a target year based on the chosen method.

    Parameters
    ----------
    method : str
        Method to use: "interpolation", "last-year", "none", or "expected"
    current_ty : int
        Current target year being processed
    previous_ty : int
        Previous target year that was simulated
    increment : float
        Price cap increment amount (EUR/MWh)
    events : int
        Number of events that occurred in previous_ty (for current climate year)
    final_price_previous : float
        Final price cap from previous_ty (for current climate year)
    initial_price_or : float
        Initial/original price cap configured for current_ty
    expected_final_price_previous : float | None
        Expected final price cap from previous_ty across all climate years
        (only used for "expected" method)
    expected_events_previous : float | None
        Expected (mean) number of events from previous_ty across all climate years
        (only used for "expected" method)

    Returns
    -------
    float
        Starting price cap for current_ty
    """
    if method == "interpolation":
        interpolated_price = (
            current_ty - previous_ty - 1
        ) * increment * events + final_price_previous
        initial_price = max(interpolated_price, initial_price_or)
    elif method == "last-year":
        initial_price = max(final_price_previous, initial_price_or)
    elif method == "none":
        initial_price = initial_price_or
    elif method == "expected":
        if expected_final_price_previous is None:
            raise ValueError(
                "expected method requires expected_final_price_previous parameter"
            )
        if expected_events_previous is None:
            raise ValueError(
                "expected method requires expected_events_previous parameter"
            )
        # Use the expected final price cap and expected events from all CYs to interpolate
        # This accounts for the average behavior across all climate scenarios
        interpolated_price = (
            current_ty - previous_ty - 1
        ) * increment * expected_events_previous + expected_final_price_previous
        initial_price = max(interpolated_price, initial_price_or)
    else:
        raise ValueError(f"Unknown price cap method: {method}")

    return initial_price


def generate_price_cap_comparison_plots(
    price_cap_data: dict, market_type: str, output_paths: list, original_caps: dict
):
    """
    Generate comparison plots showing price cap evolution across target years.

    Parameters
    ----------
    price_cap_data : dict
        Dictionary structure: {cy: {ty: pd.Series}}
    market_type : str
        "day-ahead" or "intraday"
    output_paths : list
        List of output file paths for the plots
    original_caps : dict
        Dictionary of original price cap values: {ty: float}
    """
    _, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Get all unique target years and sort them
    all_target_years = set()
    for cy_data in price_cap_data.values():
        all_target_years.update(cy_data.keys())
    sorted_target_years = sorted(all_target_years)

    # Plot each climate year as a separate line
    for cy_val in sorted(price_cap_data.keys()):
        cy_data = price_cap_data[cy_val]

        # Concatenate price cap timeseries across target years
        price_cap_list = []
        for ty_val in sorted_target_years:
            if ty_val in cy_data:
                price_cap_ts = cy_data[ty_val].copy()
                # Adjust time index to represent target year
                price_cap_ts.index = price_cap_ts.index.map(
                    lambda x, target_year=ty_val: x.replace(year=target_year)
                )
                price_cap_list.append(price_cap_ts)

        if price_cap_list:
            # Concatenate and sort by time
            combined_price_caps = pd.concat(price_cap_list, axis=0).sort_index()

            # Plot this climate year's price cap evolution
            ax.plot(
                combined_price_caps.index,
                combined_price_caps.values,
                label=f"Climate Year {cy_val}",
                linewidth=1.5,
                alpha=0.8,
            )

    # Add original configured price caps as reference lines
    for ty_val in sorted_target_years:
        if ty_val in original_caps:
            # Create a horizontal line for the year
            year_start = pd.Timestamp(year=ty_val, month=1, day=1)
            year_end = pd.Timestamp(year=ty_val, month=12, day=31)

            ax.plot(
                [mdates.date2num(year_start), mdates.date2num(year_end)],
                [original_caps[ty_val], original_caps[ty_val]],
                "--",
                color="red",
                alpha=0.7,
                linewidth=1,
                label=f"Original Cap {ty_val}"
                if ty_val == sorted_target_years[0]
                else "",
            )

    ax.set_ylabel("Price Cap (EUR/MWh)")
    ax.set_xlabel("Time")
    ax.set_title(f"{market_type.title()} Price Cap Evolution Across Target Years")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    if output_paths:
        plot_path = output_paths[0]  # Single plot per market type
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved {market_type} price cap evolution plot to {plot_path}")

    plt.close()


def generate_marginal_price_comparison_plots(
    price_data: dict, market_type: str, output_paths: list
):
    """
    Generate comparison plots showing marginal prices before and after price cap updates.

    Parameters
    ----------
    price_data : dict
        Dictionary structure: {cy: {"before": {ty: DataFrame}, "after": {ty: DataFrame}}}
    market_type : str
        "day-ahead" or "intraday"
    output_paths : list
        List of output file paths for the plots
    """
    for cy_val in price_data.keys():
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Get sorted target years
        sorted_years = sorted(price_data[cy_val]["before"].keys())

        # Adjust time indices based on target year and concatenate
        before_prices_list = []
        after_prices_list = []

        for ty_val in sorted_years:
            # Get the data for this target year
            before_df = price_data[cy_val]["before"][ty_val].copy()
            after_df = price_data[cy_val]["after"][ty_val].copy()

            # Adjust the time index to represent the target year instead of climate year
            # Replace the year component of the datetime index with the target year
            before_df.index = before_df.index.map(
                lambda x, target_year=ty_val: x.replace(year=target_year)
            )
            after_df.index = after_df.index.map(
                lambda x, target_year=ty_val: x.replace(year=target_year)
            )

            before_prices_list.append(before_df)
            after_prices_list.append(after_df)

        # Concatenate all target years for continuous timeline
        before_prices_all = pd.concat(before_prices_list, axis=0).sort_index()
        after_prices_all = pd.concat(after_prices_list, axis=0).sort_index()

        # Plot marginal prices (mean across all buses)
        ax1.plot(
            before_prices_all.index,
            before_prices_all.mean(axis=1),
            label="Before Update",
            alpha=0.7,
            linewidth=1,
        )
        ax1.plot(
            after_prices_all.index,
            after_prices_all.mean(axis=1),
            label="After Update",
            alpha=0.7,
            linewidth=1,
        )
        ax1.set_ylabel("Marginal Price (EUR/MWh)")
        ax1.set_title(
            f"{market_type.title()} Marginal Price Comparison - Climate Year {cy_val}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot price difference (After - Before)
        price_diff = after_prices_all.mean(axis=1) - before_prices_all.mean(axis=1)
        ax2.plot(price_diff.index, price_diff, color="red", alpha=0.7, linewidth=1)
        ax2.set_ylabel("Price Difference (EUR/MWh)")
        ax2.set_xlabel("Time")
        ax2.set_title("Price Difference (After - Before)")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()

        # Find and save to the correct output path
        plot_path = None
        for output_path in output_paths:
            if f"_{cy_val}_" in output_path:
                plot_path = output_path
                break

        if plot_path:
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logging.info(
                f"Saved {market_type} comparison plot for climate year {cy_val} to {plot_path}"
            )

        plt.close()


def _process_centroid_group(
    centroid_networks: list,
    starting_price_or: float,
    starting_price_from_years: float,
    increment: float,
    maximum_price_cap: float,
    enable_price_cap: bool,
) -> tuple[list, float, int, "pd.Series"]:
    """
    Apply HMMCP price cap logic to a group of centroid networks for one (ty, cy).

    Concatenates marginal prices from all centroids in chronological order, runs
    the HMMCP detection on the combined timeline, then slices updated prices back
    to each centroid network (in-place).

    Parameters
    ----------
    centroid_networks:
        List of (pypsa.Network, chunk_start) tuples, sorted by chunk_start.
    starting_price_or, starting_price_from_years, increment, maximum_price_cap:
        As per update_price_cap().
    enable_price_cap:
        Whether HMMCP logic is active.

    Returns
    -------
    (updated_networks, final_price_cap, event_count, combined_price_cap_ts)
    """
    # Build a combined network proxy holding concatenated prices
    import pypsa as _pypsa

    sorted_nets = sorted(centroid_networks, key=lambda x: x[1])

    if not enable_price_cap:
        combined_index = pd.concat(
            [n.buses_t["marginal_price"] for n, _ in sorted_nets]
        ).index
        price_cap_value = starting_price_from_years
        price_cap_ts = pd.Series(data=price_cap_value, index=combined_index)
        return [n for n, _ in sorted_nets], price_cap_value, 0, price_cap_ts

    combined_prices = pd.concat(
        [n.buses_t["marginal_price"] for n, _ in sorted_nets]
    ).sort_index()

    proxy = _pypsa.Network()
    proxy.meta = sorted_nets[0][0].meta.copy()
    proxy.buses_t["marginal_price"] = combined_prices

    final_price_cap, event_count, price_cap_ts = update_price_cap(
        proxy,
        starting_price_or=starting_price_or,
        starting_price_from_years=starting_price_from_years,
        increment=increment,
        maximum_price_cap=maximum_price_cap,
    )

    updated_combined = proxy.buses_t["marginal_price"]

    # Slice updated prices back to each centroid network
    for n, _ in sorted_nets:
        idx = n.buses_t["marginal_price"].index
        n.buses_t["marginal_price"] = updated_combined.reindex(idx)

    return [n for n, _ in sorted_nets], final_price_cap, event_count, price_cap_ts


if __name__ == "__main__":
    from snakemake.script import snakemake

    configure_logging(snakemake)
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    # Suppress verbose matplotlib font manager debug logs unless logging level is DEBUG
    if logging.getLogger().level != logging.DEBUG:
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Initialization of price cap
    config_price_cap = snakemake.config["eva"]["price_cap"]

    # Track events and final price caps per climate year
    event_count_da_per_cy = {}
    event_count_id_per_cy = {}
    final_price_cap_da_per_cy = {}
    final_price_cap_id_per_cy = {}

    # For expected method, track final caps and events per target year across all climate years
    final_caps_da_per_ty = {}  # {ty: {cy: final_cap}}
    final_caps_id_per_ty = {}  # {ty: {cy: final_cap}}
    events_da_per_ty = {}  # {ty: {cy: event_count}}
    events_id_per_ty = {}  # {ty: {cy: event_count}}

    # Store data for plotting by climate year
    da_price_data = {}
    id_price_data = {}

    # Store price cap data for plotting
    da_price_cap_data = {}
    id_price_cap_data = {}

    # Store original caps for comparison
    da_original_caps = {}
    id_original_caps = {}

    # Check if perfect foresight mode (only market networks)
    perfect_foresight = snakemake.config["eva"]["perfect_foresight"]["enable"]

    if perfect_foresight:
        # Perfect foresight mode: use market networks (treated like ID)
        # Use ID config values as requested
        network_input_before = snakemake.input.market_networks_before_update  # type: ignore[attr-defined]
        network_output = snakemake.output.market_networks  # type: ignore[attr-defined]
        price_cap_output = snakemake.output.market_price_cap_timeseries  # type: ignore[attr-defined]
        price_cap_config_key = "intraday_values"  # Use ID values
    else:
        # Standard EVA mode: use day-ahead networks for STEP 1
        network_input_before = snakemake.input.day_ahead_networks_before_update  # type: ignore[attr-defined]
        network_output = snakemake.output.day_ahead_networks  # type: ignore[attr-defined]
        price_cap_output = snakemake.output.day_ahead_price_cap_timeseries  # type: ignore[attr-defined]
        price_cap_config_key = "day_ahead_values"

    # Detect clustered mode: any input network has cluster_id in meta
    _first_net_probe = pypsa.Network(network_input_before[0])
    _is_clustered_iteration = _first_net_probe.meta.get("cluster_id") is not None

    if _is_clustered_iteration:
        # Clustered mode: group centroid networks by (ty, cy), process as combined timeline.
        # network_output/price_cap_output have one entry per (ty, cy) -- build a lookup by
        # parsing wildcards from the output paths rather than using the centroid flat index.
        def _parse_ty_cy_from_path(path: str) -> tuple:
            """Extract (ty, cy) from a path like ..._{ty}_{cy}_forecast_{it}.csv/nc."""
            import re

            m = re.search(r"_(\d{4})_(\d{4})_", str(path))
            if m:
                return int(m.group(1)), m.group(2)
            return None, None

        _net_out_by_ty_cy = {}
        for p in network_output:
            ty_p, cy_p = _parse_ty_cy_from_path(str(p))
            if ty_p is not None:
                _net_out_by_ty_cy[(ty_p, cy_p)] = str(p)

        _cap_out_by_ty_cy = {}
        for p in price_cap_output:
            ty_p, cy_p = _parse_ty_cy_from_path(str(p))
            if ty_p is not None:
                _cap_out_by_ty_cy[(ty_p, cy_p)] = str(p)

        _centroid_groups: dict = {}  # (ty, cy) -> list of (network, chunk_start, net_path)
        for i, net_path in enumerate(network_input_before):
            n = pypsa.Network(net_path) if i > 0 else _first_net_probe
            wcs = n.meta["wildcards"]
            cy_val = wcs["cy"]
            ty_val = int(wcs["ty"])
            chunk_start = (
                pd.Timestamp(n.snapshots[0])
                if len(n.snapshots) > 0
                else pd.Timestamp("2000-01-01")
            )
            key_grp = (ty_val, cy_val)
            if key_grp not in _centroid_groups:
                _centroid_groups[key_grp] = []
            _centroid_groups[key_grp].append((n, chunk_start, net_path))

        # Pass 1 accumulator: local price cap series per (ty, cy) for cross-CY reconstruction.
        _da_local_caps: dict = {}  # (ty, cy) -> price_cap_da_ts

        for (ty_val, cy_val), group in _centroid_groups.items():
            year_index = snakemake.config["scenario"]["ty"].index(ty_val)
            cy = cy_val
            ty = ty_val

            if cy not in da_price_data:
                da_price_data[cy] = {"before": {}, "after": {}}
                da_price_cap_data[cy] = {}
            if cy not in event_count_da_per_cy:
                event_count_da_per_cy[cy] = 0
                final_price_cap_da_per_cy[cy] = config_price_cap["day_ahead_values"][0]
            if ty not in final_caps_da_per_ty:
                final_caps_da_per_ty[ty] = {}
            if ty not in events_da_per_ty:
                events_da_per_ty[ty] = {}

            nets_and_starts = [(n, cs) for n, cs, _ in group]
            da_price_data[cy]["before"][ty] = pd.concat(
                [
                    n.buses_t["marginal_price"]
                    for n, _ in sorted(nets_and_starts, key=lambda x: x[1])
                ]
            ).sort_index()

            if config_price_cap.get("enable", False):
                if year_index == 0:
                    starting_price_da = config_price_cap.get(
                        price_cap_config_key, [4500]
                    )[year_index]
                else:
                    expected_final_price_prev = None
                    expected_events_prev = None
                    if config_price_cap.get("multi_year_method") == "expected":
                        previous_ty = snakemake.config["scenario"]["ty"][year_index - 1]
                        if (
                            previous_ty in final_caps_da_per_ty
                            and final_caps_da_per_ty[previous_ty]
                        ):
                            expected_final_price_prev = sum(
                                final_caps_da_per_ty[previous_ty].values()
                            ) / len(final_caps_da_per_ty[previous_ty])
                            expected_events_prev = sum(
                                events_da_per_ty[previous_ty].values()
                            ) / len(events_da_per_ty[previous_ty])
                    starting_price_da = calculate_starting_price_ty(
                        method=config_price_cap.get(
                            "multi_year_method", "interpolation"
                        ),
                        current_ty=ty,
                        previous_ty=snakemake.config["scenario"]["ty"][year_index - 1],
                        increment=config_price_cap.get("increment", 500),
                        events=event_count_da_per_cy[cy],
                        final_price_previous=final_price_cap_da_per_cy[cy],
                        initial_price_or=config_price_cap.get(
                            price_cap_config_key, [4500]
                        )[year_index],
                        expected_final_price_previous=expected_final_price_prev,
                        expected_events_previous=expected_events_prev,
                    )
            else:
                starting_price_da = config_price_cap.get(price_cap_config_key, [4500])[
                    year_index
                ]

            updated_nets, final_price_cap_day_ahead, event_count_da, price_cap_da_ts = (
                _process_centroid_group(
                    nets_and_starts,
                    starting_price_or=config_price_cap.get(
                        price_cap_config_key, [4500]
                    )[year_index],
                    starting_price_from_years=starting_price_da,
                    increment=config_price_cap.get("increment", 500),
                    maximum_price_cap=config_price_cap.get("maximum", 20000),
                    enable_price_cap=config_price_cap.get("enable", False),
                )
            )

            event_count_da_per_cy[cy] = event_count_da
            final_price_cap_da_per_cy[cy] = final_price_cap_day_ahead
            final_caps_da_per_ty[ty][cy] = final_price_cap_day_ahead
            events_da_per_ty[ty][cy] = event_count_da

            da_price_cap_data[cy][ty] = price_cap_da_ts.copy()
            da_original_caps[ty] = config_price_cap.get(price_cap_config_key, [4500])[
                year_index
            ]
            da_price_data[cy]["after"][ty] = pd.concat(
                [
                    n.buses_t["marginal_price"]
                    for n, _ in sorted(nets_and_starts, key=lambda x: x[1])
                ]
            ).sort_index()

            # Store local cap series for cross-CY global reconstruction (Pass 2).
            _da_local_caps[(ty_val, cy_val)] = price_cap_da_ts

            # Lookup the single merged output paths for this (ty, cy).
            out_net_path = _net_out_by_ty_cy.get((ty, cy))

            sorted_nets = [n for n, _ in sorted(nets_and_starts, key=lambda x: x[1])]

            # Export placeholder merged network (first centroid) for downstream rules.
            if out_net_path:
                sorted_nets[0].export_to_netcdf(out_net_path, compression=COMPRESSION)

            # FIX 2a: Save ALL updated centroid networks back to their input paths so
            # compute_npv reads capped prices rather than the pre-cap files on disk.
            for n_upd, _, net_path in group:
                n_upd.export_to_netcdf(net_path, compression=COMPRESSION)
                logging.info(f"Saved price-capped centroid DA network to {net_path}")

            # Write chunk revenues: one output CSV per (ty, cy)
            if snakemake.config["eva"]["advanced_clustering"]["enable"]:  # type: ignore[attr-defined]
                from euflex.eu_flex_model.scripts.extract_chunk_revenues import (
                    extract_chunk_revenues,
                )

                horizon_days = snakemake.config["eva"]["advanced_clustering"][
                    "horizon_days"
                ]  # type: ignore[attr-defined]
                chunk_revenues_output = snakemake.output.chunk_revenues  # type: ignore[attr-defined]
                # chunk_revenues has one entry per (ty, cy) -- find index by sorted (ty, cy) order
                sorted_keys = sorted(_centroid_groups.keys())
                cy_idx = sorted_keys.index((ty_val, cy_val))
                chunk_df = extract_chunk_revenues(
                    sorted_nets[0], ty=str(ty), cy=cy, horizon_days=horizon_days
                )
                chunk_df.to_csv(chunk_revenues_output[cy_idx])
                logging.info(f"Wrote chunk revenues for ty={ty}, cy={cy}")

        # Pass 2: Build per-ty global cap series (spans all CYs) and write price cap CSVs.
        # Using a global series fixes the cross-CY lookup gap in reconstruct_price_cap_series:
        # clusters whose medoid fell in a different CY from the target now resolve correctly.
        _da_global_caps: dict = {}  # ty -> concatenated price_cap_ts across all CYs
        for (ty_val, cy_val), cap_ts in _da_local_caps.items():
            if ty_val not in _da_global_caps:
                _da_global_caps[ty_val] = []
            _da_global_caps[ty_val].append(cap_ts)
        _da_global_caps = {
            ty_val: pd.concat(series_list).sort_index()
            for ty_val, series_list in _da_global_caps.items()
        }

        for (ty_val, cy_val), group in _centroid_groups.items():
            year_index = snakemake.config["scenario"]["ty"].index(ty_val)
            out_cap_path = _cap_out_by_ty_cy.get((ty_val, cy_val))
            if not out_cap_path:
                continue
            week_clusters_file = None
            if (
                hasattr(snakemake.input, "week_clusters")
                and snakemake.input.week_clusters
            ):
                for path in snakemake.input.week_clusters:
                    if f"week_clusters_{ty_val}_" in str(path):
                        week_clusters_file = path
                        break
            if week_clusters_file:
                logging.info(
                    f"Reconstructing full-year DA price cap for ty={ty_val}, cy={cy_val} using {week_clusters_file}"
                )
                wc_df = pd.read_csv(week_clusters_file)
                # FIX 2b: pass global series so cross-CY centroid lookups resolve.
                global_ts = _da_global_caps[ty_val]
                reconstructed_cap_da = reconstruct_price_cap_series(
                    centroid_ts=global_ts,
                    week_clusters=wc_df,
                    cy=cy_val,
                    default_cap_value=config_price_cap.get(
                        price_cap_config_key, [4500]
                    )[year_index],
                )
                reconstructed_cap_da.to_frame("price_cap").to_csv(out_cap_path)
            else:
                logging.warning(
                    f"Could not find week_clusters file for ty={ty_val} in clustered mode. Writing centroid-only DA price cap."
                )
                _da_local_caps[(ty_val, cy_val)].to_frame("price_cap").to_csv(
                    out_cap_path
                )

    else:
        # STEP 1 (full-run): Process primary networks one at a time
        for i, net_path in enumerate(network_input_before):
            da = pypsa.Network(net_path)
            wildcards = da.meta["wildcards"]
            cy = wildcards["cy"]
            ty = int(wildcards["ty"])
            year_index = snakemake.config["scenario"]["ty"].index(ty)

            # Initialize structures
            if cy not in da_price_data:
                da_price_data[cy] = {"before": {}, "after": {}}
                da_price_cap_data[cy] = {}
            if cy not in event_count_da_per_cy:
                event_count_da_per_cy[cy] = 0
                final_price_cap_da_per_cy[cy] = config_price_cap["day_ahead_values"][0]
            if ty not in final_caps_da_per_ty:
                final_caps_da_per_ty[ty] = {}
            if ty not in events_da_per_ty:
                events_da_per_ty[ty] = {}

            # Store before prices
            da_price_data[cy]["before"][ty] = da.buses_t["marginal_price"].copy()

            # Update price cap
            if config_price_cap.get("enable", False):
                if year_index == 0:
                    starting_price_da = config_price_cap.get(
                        price_cap_config_key, [4500]
                    )[year_index]
                else:
                    # Calculate expected final price and events from previous target year if using expected
                    expected_final_price_prev = None
                    expected_events_prev = None
                    if config_price_cap.get("multi_year_method") == "expected":
                        previous_ty = snakemake.config["scenario"]["ty"][year_index - 1]
                        if (
                            previous_ty in final_caps_da_per_ty
                            and final_caps_da_per_ty[previous_ty]
                        ):
                            # Calculate mean across all climate years for previous target year
                            expected_final_price_prev = sum(
                                final_caps_da_per_ty[previous_ty].values()
                            ) / len(final_caps_da_per_ty[previous_ty])

                            # Calculate mean events across all climate years for previous target year
                            expected_events_prev = sum(
                                events_da_per_ty[previous_ty].values()
                            ) / len(events_da_per_ty[previous_ty])

                            logging.info(
                                f"Day-ahead TY={ty}, CY={cy}: Using expected final price cap "
                                f"from TY={previous_ty}: {expected_final_price_prev:.2f} EUR/MWh "
                                f"and expected events: {expected_events_prev:.2f} "
                                f"(mean of {len(final_caps_da_per_ty[previous_ty])} climate years)"
                            )

                    starting_price_da = calculate_starting_price_ty(
                        method=config_price_cap.get(
                            "multi_year_method", "interpolation"
                        ),
                        current_ty=ty,
                        previous_ty=snakemake.config["scenario"]["ty"][year_index - 1],
                        increment=config_price_cap.get("increment", 500),
                        events=event_count_da_per_cy[cy],
                        final_price_previous=final_price_cap_da_per_cy[cy],
                        initial_price_or=config_price_cap.get(
                            price_cap_config_key, [4500]
                        )[year_index],
                        expected_final_price_previous=expected_final_price_prev,
                        expected_events_previous=expected_events_prev,
                    )

                final_price_cap_day_ahead, event_count_da, price_cap_da_ts = (
                    update_price_cap(
                        da,
                        starting_price_or=config_price_cap.get(
                            price_cap_config_key, [4500]
                        )[year_index],
                        starting_price_from_years=starting_price_da,
                        increment=config_price_cap.get("increment", 500),
                        maximum_price_cap=config_price_cap.get("maximum", 20000),
                    )
                )
                # Update per-climate-year tracking
                event_count_da_per_cy[cy] = event_count_da
                final_price_cap_da_per_cy[cy] = final_price_cap_day_ahead
                final_caps_da_per_ty[ty][cy] = final_price_cap_day_ahead
                events_da_per_ty[ty][cy] = event_count_da
            else:
                da_price_cap_value = config_price_cap.get(price_cap_config_key, [4500])[
                    year_index
                ]
                price_cap_da_ts = pd.Series(
                    data=da_price_cap_value, index=da.buses_t["marginal_price"].index
                )

            # Store results
            da_price_cap_data[cy][ty] = price_cap_da_ts.copy()
            da_original_caps[ty] = config_price_cap.get(price_cap_config_key, [4500])[
                year_index
            ]
            da_price_data[cy]["after"][ty] = da.buses_t["marginal_price"].copy()

            # Save outputs
            price_cap_da_ts.to_frame("price_cap").to_csv(price_cap_output[i])
            da.export_to_netcdf(network_output[i], compression=COMPRESSION)

            # Extract chunk revenues while DA network is already in memory with capped prices.
            if snakemake.config["eva"]["advanced_clustering"]["enable"]:  # type: ignore[attr-defined]
                from euflex.eu_flex_model.scripts.extract_chunk_revenues import (
                    extract_chunk_revenues,
                )

                horizon_days = snakemake.config["eva"]["advanced_clustering"][
                    "horizon_days"
                ]  # type: ignore[attr-defined]
                chunk_df = extract_chunk_revenues(
                    da, ty=str(ty), cy=cy, horizon_days=horizon_days
                )
                chunk_revenues_output = snakemake.output.chunk_revenues  # type: ignore[attr-defined]
                chunk_df.to_csv(chunk_revenues_output[i])
                logging.info(f"Wrote chunk revenues for ty={ty}, cy={cy}")

    # STEP 2: Process intraday networks (multiple per ty/cy - one per forced outage scenario)
    # Skip this step in perfect foresight mode
    if not perfect_foresight:
        # Build lookup dicts for intraday outputs, keyed by (ty, cy, fo) parsed from path.
        # In clustered mode there are k centroid ID networks per (ty, cy, fo) but only
        # one output entry per (ty, cy, fo); indexing by flat position would go out of range.
        def _parse_ty_cy_fo_from_path(path: str) -> tuple:
            import re

            # Pattern: ..._{ty}_{cy}_actual_{fo}_...
            m = re.search(r"_(\d{4})_(\d{4})_actual_([^_/]+)_", str(path))
            if m:
                return int(m.group(1)), m.group(2), m.group(3)
            return None, None, None

        _id_net_out_by_key = {}
        for p in snakemake.output.intraday_networks:  # type: ignore[attr-defined]
            ty_p, cy_p, fo_p = _parse_ty_cy_fo_from_path(str(p))
            if ty_p is not None:
                _id_net_out_by_key[(ty_p, cy_p, fo_p)] = str(p)

        _id_cap_out_by_key = {}
        for p in snakemake.output.intraday_price_cap_timeseries:  # type: ignore[attr-defined]
            ty_p, cy_p, fo_p = _parse_ty_cy_fo_from_path(str(p))
            if ty_p is not None:
                _id_cap_out_by_key[(ty_p, cy_p, fo_p)] = str(p)

        # Track which (ty, cy, fo) groups have already been written (clustered mode)
        _id_groups_written: set = set()
        id_price_caps_by_key = {}

        for i, id_path in enumerate(snakemake.input.intraday_networks_before_update):  # type: ignore[attr-defined]
            id_net = pypsa.Network(id_path)
            ty_path, cy_path, fo_path = _parse_ty_cy_fo_from_path(id_path)
            if ty_path is not None:
                ty = ty_path
                cy = cy_path
                fo = fo_path
            else:
                wildcards = id_net.meta["wildcards"]
                cy = wildcards["cy"]
                ty = int(wildcards["ty"])
                fo = wildcards.get("fo", "")
            year_index = snakemake.config["scenario"]["ty"].index(ty)
            _id_out_key = (ty, cy, fo)

            # Initialize structures
            if cy not in id_price_data:
                id_price_data[cy] = {"before": {}, "after": {}}
                id_price_cap_data[cy] = {}
            if cy not in event_count_id_per_cy:
                event_count_id_per_cy[cy] = 0
                final_price_cap_id_per_cy[cy] = config_price_cap["intraday_values"][0]
            if ty not in final_caps_id_per_ty:
                final_caps_id_per_ty[ty] = {}
            if ty not in events_id_per_ty:
                events_id_per_ty[ty] = {}

            # Store before prices
            id_price_data[cy]["before"][ty] = id_net.buses_t["marginal_price"].copy()

            # Update price cap
            if config_price_cap.get("enable", False):
                if year_index == 0:
                    starting_price_id = config_price_cap.get("intraday_values", [4500])[
                        year_index
                    ]
                else:
                    # Calculate expected final price and events from previous target year if using expected
                    expected_final_price_prev = None
                    expected_events_prev = None
                    if config_price_cap.get("multi_year_method") == "expected":
                        previous_ty = snakemake.config["scenario"]["ty"][year_index - 1]
                        if (
                            previous_ty in final_caps_id_per_ty
                            and final_caps_id_per_ty[previous_ty]
                        ):
                            # Calculate mean across all climate years for previous target year
                            expected_final_price_prev = sum(
                                final_caps_id_per_ty[previous_ty].values()
                            ) / len(final_caps_id_per_ty[previous_ty])

                            # Calculate mean events across all climate years for previous target year
                            expected_events_prev = sum(
                                events_id_per_ty[previous_ty].values()
                            ) / len(events_id_per_ty[previous_ty])

                            logging.info(
                                f"Intraday TY={ty}, CY={cy}: Using expected final price cap "
                                f"from TY={previous_ty}: {expected_final_price_prev:.2f} EUR/MWh "
                                f"and expected events: {expected_events_prev:.2f} "
                                f"(mean of {len(final_caps_id_per_ty[previous_ty])} climate years)"
                            )

                    starting_price_id = calculate_starting_price_ty(
                        method=config_price_cap.get(
                            "multi_year_method", "interpolation"
                        ),
                        current_ty=ty,
                        previous_ty=snakemake.config["scenario"]["ty"][year_index - 1],
                        increment=config_price_cap.get("increment", 500),
                        events=event_count_id_per_cy[cy],
                        final_price_previous=final_price_cap_id_per_cy[cy],
                        initial_price_or=config_price_cap.get(
                            "intraday_values", [4500]
                        )[year_index],
                        expected_final_price_previous=expected_final_price_prev,
                        expected_events_previous=expected_events_prev,
                    )

                final_price_cap_intraday, event_count_id, price_cap_id_ts = (
                    update_price_cap(
                        id_net,
                        starting_price_or=config_price_cap.get(
                            "intraday_values", [4500]
                        )[year_index],
                        starting_price_from_years=starting_price_id,
                        increment=config_price_cap.get("increment", 500),
                        maximum_price_cap=config_price_cap.get("maximum", 20000),
                    )
                )
                # Update per-climate-year tracking
                event_count_id_per_cy[cy] = event_count_id
                final_price_cap_id_per_cy[cy] = final_price_cap_intraday
                final_caps_id_per_ty[ty][cy] = final_price_cap_intraday
                events_id_per_ty[ty][cy] = event_count_id
            else:
                id_price_cap_value = config_price_cap.get("intraday_values", [4500])[
                    year_index
                ]
                price_cap_id_ts = pd.Series(
                    data=id_price_cap_value,
                    index=id_net.buses_t["marginal_price"].index,
                )

            # Store results
            id_price_cap_data[cy][ty] = price_cap_id_ts.copy()
            id_original_caps[ty] = config_price_cap.get("intraday_values", [4500])[
                year_index
            ]
            id_price_data[cy]["after"][ty] = id_net.buses_t["marginal_price"].copy()

            if _is_clustered_iteration:
                if _id_out_key not in id_price_caps_by_key:
                    id_price_caps_by_key[_id_out_key] = []
                id_price_caps_by_key[_id_out_key].append(price_cap_id_ts)

                # FIX 2a: Save updated centroid ID network back to its input path so
                # compute_npv reads the capped prices rather than the pre-cap file.
                id_net.export_to_netcdf(id_path, compression=COMPRESSION)
                logging.info(f"Saved price-capped centroid ID network to {id_path}")

            # Save outputs -- use lookup to handle clustered mode where there are k centroid
            # networks per (ty, cy, fo) but only one output path per (ty, cy, fo).
            out_id_cap = _id_cap_out_by_key.get(_id_out_key)
            out_id_net = _id_net_out_by_key.get(_id_out_key)
            if _id_out_key not in _id_groups_written:
                if out_id_cap and not _is_clustered_iteration:
                    price_cap_id_ts.to_frame("price_cap").to_csv(out_id_cap)
                if out_id_net:
                    id_net.export_to_netcdf(out_id_net, compression=COMPRESSION)
                _id_groups_written.add(_id_out_key)

        # Write reconstructed intraday price caps in clustered mode.
        # FIX 2b: Build per-ty global series across all (ty, cy, fo) entries so that
        # cross-CY cluster lookups in reconstruct_price_cap_series resolve correctly.
        if _is_clustered_iteration:
            _id_global_caps: dict = {}  # ty -> concatenated cap series across all CYs/FOs
            for (ty_val, cy_val, fo_val), ts_list in id_price_caps_by_key.items():
                if ty_val not in _id_global_caps:
                    _id_global_caps[ty_val] = []
                _id_global_caps[ty_val].extend(ts_list)
            _id_global_caps = {
                ty_val: pd.concat(series_list).sort_index()
                for ty_val, series_list in _id_global_caps.items()
            }

            for (ty_val, cy_val, fo_val), ts_list in id_price_caps_by_key.items():
                out_id_cap = _id_cap_out_by_key.get((ty_val, cy_val, fo_val))
                if out_id_cap:
                    week_clusters_file = None
                    if (
                        hasattr(snakemake.input, "week_clusters")
                        and snakemake.input.week_clusters
                    ):
                        for path in snakemake.input.week_clusters:
                            if f"week_clusters_{ty_val}_" in str(path):
                                week_clusters_file = path
                                break
                    if week_clusters_file:
                        logging.info(
                            f"Reconstructing full-year ID price cap for ty={ty_val}, cy={cy_val}, fo={fo_val} "
                            f"using {week_clusters_file}"
                        )
                        wc_df = pd.read_csv(week_clusters_file)
                        year_idx = snakemake.config["scenario"]["ty"].index(ty_val)
                        global_ts = _id_global_caps[ty_val]
                        reconstructed_cap_id = reconstruct_price_cap_series(
                            centroid_ts=global_ts,
                            week_clusters=wc_df,
                            cy=cy_val,
                            default_cap_value=config_price_cap.get(
                                "intraday_values", [4500]
                            )[year_idx],
                        )
                        reconstructed_cap_id.to_frame("price_cap").to_csv(out_id_cap)
                    else:
                        logging.warning(
                            f"Could not find week_clusters file for ty={ty_val} in clustered mode. Writing centroid-only ID price cap."
                        )
                        pd.concat(ts_list).sort_index().to_frame("price_cap").to_csv(
                            out_id_cap
                        )

    # Generate plots
    if perfect_foresight:
        # Perfect foresight mode: only one set of plots (market)
        generate_marginal_price_comparison_plots(
            da_price_data,
            "market",
            snakemake.output.marginal_price_comparison_market,  # type: ignore[attr-defined]
        )
        generate_price_cap_comparison_plots(
            da_price_cap_data,
            "market",
            [snakemake.output.price_cap_evolution_market],  # type: ignore[attr-defined]
            da_original_caps,
        )
    else:
        # Standard EVA mode: separate DA and ID plots
        generate_marginal_price_comparison_plots(
            da_price_data,
            "day-ahead",
            snakemake.output.marginal_price_comparison_day_ahead,  # type: ignore[attr-defined]
        )
        generate_marginal_price_comparison_plots(
            id_price_data,
            "intraday",
            snakemake.output.marginal_price_comparison_intraday,  # type: ignore[attr-defined]
        )
        generate_price_cap_comparison_plots(
            da_price_cap_data,
            "day-ahead",
            [snakemake.output.price_cap_evolution_day_ahead],  # type: ignore[attr-defined]
            da_original_caps,
        )
        generate_price_cap_comparison_plots(
            id_price_cap_data,
            "intraday",
            [snakemake.output.price_cap_evolution_intraday],  # type: ignore[attr-defined]
            id_original_caps,
        )
