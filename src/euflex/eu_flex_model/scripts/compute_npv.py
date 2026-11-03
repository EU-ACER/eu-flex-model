# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System

import logging
import math
import re

import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
)

# Exogenous revenue column names on n.generators / n.storage_units (€/kW/year).
# These columns are attached by add_exogenous_revenues_from_db() in build_network_from_database.py
# when costs.exogenous_revenues.enable is true, or can be added manually via
# prepare-sensitivity (editing generators.csv / storage_units.csv).
EXOGENOUS_REVENUE_COLUMNS = [
    "CHP_revenue",
    "balancing_revenue",
    "CM_revenue",
    "congestion_revenue",
]


def load_exogenous_revenues(
    asset_index: pd.MultiIndex,
    network: pypsa.Network,
) -> pd.Series:
    """
    Calculate exogenous revenues from generator and storage unit attributes.

    Reads per-asset revenue columns (e.g. CHP_revenue, CM_revenue) that are
    already stored on ``n.generators`` and ``n.storage_units`` in €/kW/year.
    Multiple revenue categories are summed per asset and converted to total
    €/year using installed capacity:

        total_revenue = sum(revenue_columns) * p_nom_MW * 1000

    The revenue columns are populated during network construction
    (see ``add_exogenous_revenues_from_db`` in ``build_network_from_database.py``) or can
    be added manually via the ``prepare-sensitivity`` workflow.

    Parameters
    ----------
    asset_index : pd.MultiIndex
        MultiIndex of (component, name) for all assets.
    network : pypsa.Network
        The PyPSA network containing generator and storage unit attributes
        including any exogenous revenue columns.

    Returns
    -------
    pd.Series
        Exogenous revenue in €/year per asset, indexed by asset_index.
    """
    exog_revenue = pd.Series(0.0, index=asset_index, name="exogenous_revenue")

    if asset_index.empty:
        return exog_revenue

    # Scale annual revenue to the fraction of the year covered by this network.
    # snapshot_weightings["objective"] gives the hours represented by each snapshot,
    # so summing them yields the total modelled hours (e.g. 168 h for one week).
    HOURS_PER_YEAR = 8760.0
    network_hours = network.snapshot_weightings["objective"].sum()
    year_fraction = network_hours / HOURS_PER_YEAR
    logging.info(
        f"Exogenous revenue year fraction: {network_hours:.1f} h / {HOURS_PER_YEAR} h = {year_fraction:.4f}"
    )

    generators_df = network.generators
    generator_assets = [(c, name) for c, name in asset_index if c == "Generator"]

    for _, asset_name in generator_assets:
        if asset_name not in generators_df.index:
            continue

        p_nom = generators_df.loc[asset_name, "p_nom"]

        # Sum all exogenous revenue columns present on generators
        total_per_kw = 0.0
        for col in EXOGENOUS_REVENUE_COLUMNS:
            if col in generators_df.columns:
                val = generators_df.loc[asset_name, col]
                if pd.notna(val) and val != 0:
                    total_per_kw += val

        if total_per_kw != 0:
            # Convert €/kW/year → €/year for this network window
            total_revenue = total_per_kw * p_nom * 1000 * year_fraction
            exog_revenue.loc[("Generator", asset_name)] = total_revenue
            logging.info(
                f"Generator {asset_name} @ {generators_df.loc[asset_name, 'bus']}: "
                f"exogenous revenue = {total_per_kw} €/kW/y "
                f"× {p_nom:.1f} MW × {year_fraction:.4f} = {total_revenue:.2f} €"
            )

    storage_units_df = network.storage_units
    storage_assets = [(c, name) for c, name in asset_index if c == "StorageUnit"]

    for _, asset_name in storage_assets:
        if asset_name not in storage_units_df.index:
            continue

        p_nom = storage_units_df.loc[asset_name, "p_nom"]

        # Sum all exogenous revenue columns present on storage units
        total_per_kw = 0.0
        for col in EXOGENOUS_REVENUE_COLUMNS:
            if col in storage_units_df.columns:
                val = storage_units_df.loc[asset_name, col]
                if pd.notna(val) and val != 0:
                    total_per_kw += val

        if total_per_kw != 0:
            # Convert €/kW/year → €/year for this network window
            total_revenue = total_per_kw * p_nom * 1000 * year_fraction
            exog_revenue.loc[("StorageUnit", asset_name)] = total_revenue
            logging.info(
                f"StorageUnit {asset_name} @ {storage_units_df.loc[asset_name, 'bus']}: "
                f"exogenous revenue = {total_per_kw} €/kW/y "
                f"× {p_nom:.1f} MW × {year_fraction:.4f} = {total_revenue:.2f} €"
            )

    gen_non_zero = sum(
        1 for _, n in generator_assets if exog_revenue.get(("Generator", n), 0) != 0
    )
    su_non_zero = sum(
        1 for _, n in storage_assets if exog_revenue.get(("StorageUnit", n), 0) != 0
    )
    logging.info(
        f"Exogenous revenues: "
        f"{gen_non_zero}/{len(generator_assets)} generators, "
        f"{su_non_zero}/{len(storage_assets)} storage units, "
        f"total = {exog_revenue.sum():.2f} €"
    )
    return exog_revenue


def get_installed_capacity_per_asset(
    n: pypsa.Network, index: pd.MultiIndex
) -> pd.Series:
    mw_factor = 1  # scale nom pu values to MW

    capacities = {}
    for c in index.unique("component"):
        df = n.static(c)
        subindex = index.get_loc_level(c)[1]
        if c in ["Generator", "StorageUnit"] and "p_nom" in df.columns:
            capacities[c] = df.p_nom[subindex] * mw_factor
        elif c == "Store" and "e_nom" in df.columns:
            capacities[c] = df.e_nom[subindex] * mw_factor
        else:
            capacities[c] = pd.Series(0, index=subindex)
    return pd.Series(pd.concat(capacities))


def apply_relative_dispatch(id: pypsa.Network, da: pypsa.Network) -> None:
    """
    Applies the relative dispatch from the day-ahead network to the intraday network.
    This function modifies the intraday network in place.
    """
    for c in comps:
        id.dynamic(c)["p"] = id.dynamic(c)["p"].sub(da.dynamic(c)["p"], fill_value=0)


def _parse_reserve_key(attr_name: str) -> str | None:
    """Parse reserve type and direction from attribute name."""
    parts = attr_name.split("_")
    if len(parts) >= 3 and parts[0] == "r":
        # Check if the attribute ends with _price or has _up/_down
        if (
            attr_name.endswith("_price")
            and len(parts) >= 4
            or "_up" in attr_name
            or "_down" in attr_name
        ):
            # e.g., r_FCR_up_price -> FCR_up or r_FCR_up -> FCR_up
            return f"{parts[1]}_{parts[2]}"
    return None


def _update_reserve_prices_with_price_cap(
    reserve_prices: dict, price_cap_csv_path: str, original_price_cap: float
) -> dict:
    """Update reserve prices using price cap data from CSV."""
    # Load price cap timeseries from CSV
    price_cap_df = pd.read_csv(price_cap_csv_path, index_col=0, parse_dates=True)
    price_cap_series = price_cap_df["price_cap"]

    # Update each reserve price DataFrame - for each bidding zone (column)
    for key, price_df in reserve_prices.items():
        for col in price_df.columns:  # Each column is a bidding zone
            mask = abs(price_df[col] - original_price_cap) <= 1e-2
            price_df.loc[mask, col] = price_cap_series.loc[mask]

    return reserve_prices


def _load_reserve_prices(n: pypsa.Network) -> dict[str, pd.DataFrame]:
    """Load reserve prices from network sources."""
    reserve_prices = {}

    # Define sources to check
    sources = [
        (n.buses_t, "network.buses_t"),
        (getattr(n, "_reserve_prices", {}), "network._reserve_prices"),
    ]

    for source, source_name in sources:
        if not source:
            continue

        price_attrs = [
            attr
            for attr in source.keys()
            if attr.startswith("r_") and attr.endswith("_price")
        ]

        for ts_name in price_attrs:
            key = _parse_reserve_key(ts_name)
            if key:
                try:
                    reserve_prices[key] = source[ts_name]
                except Exception as e:
                    logging.warning(
                        f"Could not load reserve price from {source_name} {ts_name}: {e}"
                    )

        if reserve_prices:
            logging.info(
                f"Loaded {len(reserve_prices)} reserve price series from {source_name}"
            )
            break

    return reserve_prices


def _align_time_series(quantities: pd.Series, prices: pd.Series) -> float:
    """
    Align time series and calculate revenue contribution.

    Aligns two time series (quantities and prices) by their indices and computes
    the sum of element-wise products. Handles timezone-aware indices and provides
    fallback alignment strategies if direct index matching fails.

    Parameters
    ----------
    quantities : pd.Series
        Time series of quantities with a DatetimeIndex or comparable index.
        Values should be numeric (int or float).
    prices : pd.Series
        Time series of prices with a DatetimeIndex or comparable index.
        Values should be numeric (int or float). Index should ideally align
        with the quantities index.

    Returns
    -------
    float
        The sum of element-wise products (quantities * prices) for all aligned
        time points. Returns 0.0 if no common indices are found or if an error
        occurs during alignment.
    """
    try:
        # Remove timezone info for alignment
        quant_index = (
            quantities.index.tz_localize(None)  # ty: ignore[possibly-missing-attribute]
            if hasattr(quantities.index, "tz")
            else quantities.index
        )
        price_index = (
            prices.index.tz_localize(None)  # ty: ignore[possibly-missing-attribute]
            if hasattr(prices.index, "tz")
            else prices.index
        )

        common_index = quant_index.intersection(price_index)

        if not common_index.empty:
            return (quantities.loc[common_index] * prices.loc[common_index]).sum()

        # Fallback: string-based alignment
        quant_str = quantities.index.astype(str)
        price_str = prices.index.astype(str)
        common_str = quant_str.intersection(price_str)

        if not common_str.empty:
            return (
                quantities[quant_str.isin(common_str)].values
                * prices[price_str.isin(common_str)].values
            ).sum()

        # Last resort: assume same order if same length
        if len(quantities) == len(prices):
            logging.warning("Assuming same order for time series alignment")
            return (quantities.values * prices.values).sum()

    except Exception as e:
        logging.error(f"Error aligning time series: {e}")

    return 0.0


def calculate_reserves_revenue(
    n: pypsa.Network, asset_index: pd.MultiIndex
) -> pd.Series:
    """
    Calculate reserves revenue for assets following equation (10.3):
    Revenue_{n,s}^{rsv} = Σ_t Σ_i Σ_j λ_{n,t}^{i,j} * r_{n,s,t}^{i,j}

    Note: This calculates revenue for ALL assets, not just extendable ones.
    For non-extendable assets, the NPV is informational only.
    """
    reserves_revenue = pd.Series(0.0, index=asset_index, name="reserves_revenue")

    if asset_index.empty:
        logging.warning(
            "Asset index is empty. No assets to calculate reserves revenue for."
        )
        return reserves_revenue

    try:
        reserve_prices = _load_reserve_prices(n)
        if not reserve_prices:
            logging.warning(
                "No reserve price data available. Reserves revenue set to zero."
            )
            return reserves_revenue

        # Update reserve prices with price cap data
        wildcards = n.meta["wildcards"]
        price_cap_csv_path = getattr(n, "_price_cap_csv_path", None)
        config_price_cap = getattr(n, "_config_price_cap", None)

        if price_cap_csv_path and config_price_cap:
            ty = int(wildcards["ty"])
            year_index = config_price_cap.get("scenario", {}).get("ty", []).index(ty)
            original_price_cap = (
                config_price_cap.get("eva", {})
                .get("price_cap", {})
                .get("day_ahead_values", [4500])[year_index]
            )
            reserve_prices = _update_reserve_prices_with_price_cap(
                reserve_prices, price_cap_csv_path, original_price_cap
            )

        # Calculate revenue for each asset
        for component, asset_name in asset_index:
            component_t = (
                getattr(n, f"{component.lower()}s_t", None)
                if component == "Generator"
                else getattr(n, "storage_units_t", None)
            )
            if not component_t:
                continue

            # Get reserve attributes for this component
            reserve_attrs = [
                attr for attr in component_t.keys() if _parse_reserve_key(attr)
            ]

            asset_revenue = 0.0
            for reserve_attr in reserve_attrs:
                key = _parse_reserve_key(reserve_attr)
                if (
                    not key
                    or key not in reserve_prices
                    or asset_name not in component_t[reserve_attr].columns
                ):
                    continue

                quantities = component_t[reserve_attr][asset_name]
                if (quantities == 0).all():
                    continue

                # Get asset bus and corresponding prices
                component_attr = (
                    "storage_units" if component == "StorageUnit" else "generators"
                )
                asset_bus = getattr(n, component_attr).bus[asset_name]

                prices_df = reserve_prices[key]
                if isinstance(prices_df, pd.DataFrame):
                    asset_prices = prices_df.get(
                        asset_bus,
                        prices_df.iloc[:, 0]
                        if not prices_df.empty
                        else pd.Series(0, index=prices_df.index),
                    )
                else:
                    asset_prices = prices_df

                # Calculate revenue contribution
                revenue_contribution = _align_time_series(quantities, asset_prices)
                asset_revenue += revenue_contribution

                if revenue_contribution > 0:
                    logging.info(
                        f"Asset {asset_name}: reserve {key} revenue = {revenue_contribution:.2f}"
                    )

            reserves_revenue.loc[(component, asset_name)] = asset_revenue

        # Summary logging
        total_revenue = reserves_revenue.sum()
        non_zero_assets = (reserves_revenue != 0).sum()
        logging.info(
            f"Reserves revenue: {non_zero_assets}/{len(asset_index)} assets, total = {total_revenue:.2f}"
        )

    except Exception as e:
        logging.error(f"Error calculating reserves revenue: {e}")

    return reserves_revenue


def calculate_suc_sdc(
    n: pypsa.Network,
    asset_index: pd.MultiIndex,
    include_carriers: list[str] | None = None,
    exclude_carriers: list[str] | None = None,
) -> pd.Series:
    """
    Calculate start-up and shut-down costs for committable generators.

    PyPSA 0.33.2 does not include start-up costs (SUC) or shut-down costs (SDC)
    in n.statistics.opex(). This function computes them manually from the
    post-optimization binary timeseries:

        SUC_asset = sum_t( start_up_t * start_up_cost )
        SDC_asset = sum_t( shut_down_t * shut_down_cost )

    Only applies to Generator components with committable=True. StorageUnits
    do not have start_up_cost/shut_down_cost attributes in PyPSA 0.33.2.

    Use ``include_carriers`` or ``exclude_carriers`` (mutually exclusive) to
    restrict which generators are considered:

    - DA networks: pass ``exclude_carriers=id_free_commitment_carriers`` so that
      free-committable generators are excluded (their costs come from the ID network).
    - ID networks: pass ``include_carriers=id_free_commitment_carriers`` so that
      only free-committable generators contribute (fixed generators use DA costs).

    Parameters
    ----------
    n : pypsa.Network
        A solved PyPSA network containing generators_t['start_up'] and
        generators_t['shut_down'] timeseries.
    asset_index : pd.MultiIndex
        MultiIndex of (component, name) for all assets.
    include_carriers : list[str] | None
        If set, only generators whose carrier is in this list are included.
        Mutually exclusive with ``exclude_carriers``.
    exclude_carriers : list[str] | None
        If set, generators whose carrier is in this list are skipped.
        Mutually exclusive with ``include_carriers``.

    Returns
    -------
    pd.Series
        Combined SUC + SDC cost in € per asset, indexed by asset_index.
        Zero for non-committable generators, StorageUnits, and any asset
        where start_up/shut_down timeseries are absent.
    """
    if include_carriers is not None and exclude_carriers is not None:
        raise ValueError(
            "include_carriers and exclude_carriers are mutually exclusive — set only one."
        )

    suc_sdc = pd.Series(0.0, index=asset_index, name="suc_sdc")

    if asset_index.empty:
        return suc_sdc

    gens_df = n.generators
    start_up_ts = n.generators_t.get("start_up")
    shut_down_ts = n.generators_t.get("shut_down")

    if start_up_ts is None and shut_down_ts is None:
        logging.debug(
            "No start_up/shut_down timeseries found — network may not be solved yet. "
            "SUC and SDC set to zero."
        )
        return suc_sdc

    generator_assets = [name for comp, name in asset_index if comp == "Generator"]

    for asset_name in generator_assets:
        if asset_name not in gens_df.index:
            continue
        if not gens_df.loc[asset_name, "committable"]:
            continue

        carrier = gens_df.loc[asset_name, "carrier"]
        if include_carriers is not None and carrier not in include_carriers:
            continue
        if exclude_carriers is not None and carrier in exclude_carriers:
            continue

        suc_cost = gens_df.loc[asset_name, "start_up_cost"]
        sdc_cost = gens_df.loc[asset_name, "shut_down_cost"]

        asset_suc = 0.0
        asset_sdc = 0.0

        if start_up_ts is not None and asset_name in start_up_ts.columns:
            asset_suc = (start_up_ts[asset_name] * suc_cost).sum()

        if shut_down_ts is not None and asset_name in shut_down_ts.columns:
            asset_sdc = (shut_down_ts[asset_name] * sdc_cost).sum()

        total = asset_suc + asset_sdc
        if abs(total) > 1e-9:
            suc_sdc.loc[("Generator", asset_name)] = total
            logging.info(
                f"Generator {asset_name}: SUC={asset_suc:.2f}, SDC={asset_sdc:.2f}, "
                f"total={total:.2f} €"
            )

    non_zero = int((suc_sdc.abs() > 1e-9).sum())
    logging.info(
        f"SUC+SDC: {non_zero}/{len(generator_assets)} generators have non-zero costs, "
        f"total={suc_sdc.sum():.2f} €"
    )
    return suc_sdc


def _build_cluster_distribution(
    week_clusters_paths: list[str],
) -> tuple[dict, dict]:
    """
    Build cluster-to-cy distribution tables from week_clusters CSVs.

    Each path corresponds to one target year (ty). Counts are kept per-ty so
    that multiple target-year files do not accumulate into each other.

    Returns
    -------
    n_per_cluster_cy : dict
        {ty -> {cluster_id -> {cy -> count}}} -- weeks per (ty, cluster, dest_cy).
    n_chunks_per_cy : dict
        {ty -> {cy -> total_weeks}} -- total weeks in each (ty, cy).
    """
    n_per_cluster_cy: dict = {}
    n_chunks_per_cy: dict = {}

    for path in week_clusters_paths:
        # Extract ty from filename: week_clusters_{ty}_{it}.csv
        m = re.search(r"week_clusters_(\d+)_\d+\.csv", str(path))
        ty_str = m.group(1) if m else "unknown"

        df = pd.read_csv(path)
        n_per_cluster_cy[ty_str] = {}
        n_chunks_per_cy[ty_str] = {}

        for cy_val, grp in df.groupby("cy"):
            cy_str = str(cy_val)
            n_chunks_per_cy[ty_str][cy_str] = len(grp)
            for _, row in grp.iterrows():
                cid = int(row["cluster_id"])
                if cid not in n_per_cluster_cy[ty_str]:
                    n_per_cluster_cy[ty_str][cid] = {}
                n_per_cluster_cy[ty_str][cid][cy_str] = (
                    n_per_cluster_cy[ty_str][cid].get(cy_str, 0) + 1
                )

    return n_per_cluster_cy, n_chunks_per_cy


def npv_per_mw_at_target_years(
    annual_profit: pd.DataFrame,
    capacities: pd.DataFrame,
    discount_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the NPV per MW of each asset at each target year.

    Annual profit is first normalised by the capacity of the same target year,
    giving a profit per MW. This is then interpolated linearly between target
    years, held constant after the last target year, discounted, and summed
    from each target year onwards. The result is the value of one MW that is
    present from that target year until decommissioning.

    Target years where an asset has no capacity carry no information about its
    profit per MW; those are filled from the neighbouring target years. Assets
    without capacity in any target year get NaN.

    Parameters
    ----------
    annual_profit : pd.DataFrame
        Total annual profit (€/year), indexed by asset, with target years as
        integer columns.
    capacities : pd.DataFrame
        Installed capacity (MW), same layout as ``annual_profit``.
    discount_factors : pd.DataFrame
        Discount factor per asset and year, with all years from the first
        target year until the last decommissioning year as integer columns.
        Zero for years in which an asset is decommissioned.

    Returns
    -------
    pd.DataFrame
        NPV per MW (€/MW), indexed by asset, with target years as columns.
    """
    ty_years = annual_profit.columns
    all_years = discount_factors.columns

    capacities = capacities.reindex(index=annual_profit.index, columns=ty_years)
    profit_per_mw = annual_profit.div(capacities.where(capacities > 0))

    # Interpolate profit per MW between target years; beyond the first and last
    # available target year, the nearest value is held constant.
    interpolated_profit_per_mw = (
        profit_per_mw.T.reindex(all_years)
        .interpolate(method="linear", axis=0, limit_direction="both")
        .T
    )

    discounted_profit_per_mw = interpolated_profit_per_mw.mul(discount_factors)
    npv_from_year_onward = (
        discounted_profit_per_mw.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    )
    return npv_from_year_onward[ty_years]


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="compute_npv",
        it="0",
    )

    configure_logging(snakemake)
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    comps = ["Generator", "StorageUnit"]

    revenues = {}
    expenses = {}

    # Detailed tracking for NPV computation log (logging only - does not affect calculations)
    energy_revenues_log = {}
    reserves_revenues_log = {}  # Only day-ahead, intraday is always 0
    exogenous_revenues_log = {}  # Only day-ahead, intraday is always 0
    opex_log = {}
    opex_suc_sdc_log = {}  # SUC/SDC logged separately for transparency
    capex_annualized_log = {}

    all_assets = []

    # Check if perfect foresight mode (only market networks)
    perfect_foresight = snakemake.config["eva"]["perfect_foresight"]["enable"]

    if perfect_foresight:
        # Perfect foresight mode: use market networks
        network_paths = (
            snakemake.input.market_networks
        )  # ty: ignore[possibly-missing-attribute]
    else:
        # Standard EVA mode: use day-ahead networks
        network_paths = (
            snakemake.input.day_ahead_networks
        )  # ty: ignore[possibly-missing-attribute]

    # Snakemake unwraps single-element named inputs to a plain string; ensure list.
    if isinstance(network_paths, str):
        network_paths = [network_paths]

    asset_carrier_bus: dict[tuple[str, str], tuple[str, str]] = {}
    for net_path in network_paths:
        n = pypsa.Network(net_path)
        for c in comps:
            df = n.static(c)
            if c in ["Generator", "StorageUnit"]:
                # Include ALL assets for NPV calculation (not just extendable/decommissionable)
                # NPV for non-extendable assets is informational only
                all_assets.extend([(c, name) for name in df.index])
                for name in df.index:
                    if (c, name) not in asset_carrier_bus:
                        asset_carrier_bus[(c, name)] = (
                            df.loc[name, "carrier"],
                            df.loc[name, "bus"],
                        )

    # Create union of all unique assets
    asset_index = pd.MultiIndex.from_tuples(
        list(dict.fromkeys(all_assets)), names=["component", "name"]
    ).sort_values()

    # Detect clustered iteration and build cross-cy distribution tables.
    # week_clusters is present in snakemake.input only when clustering is enabled.
    _clustering_enabled = snakemake.config["eva"]["advanced_clustering"]["enable"]
    _first_net_probe = pypsa.Network(next(iter(network_paths)))
    _is_clustered_iteration = (
        _clustering_enabled and _first_net_probe.meta.get("cluster_id") is not None
    )

    if _is_clustered_iteration:
        _wc_paths = (
            snakemake.input.week_clusters
        )  # ty: ignore[possibly-missing-attribute]
        if isinstance(_wc_paths, str):
            _wc_paths = [_wc_paths]
        _n_per_cluster_cy, _n_chunks_per_cy = _build_cluster_distribution(
            list(_wc_paths)
        )
        # Collect all dest CYs across all TYs (order-stable, deduplicated)
        _all_dest_cys = list(
            dict.fromkeys(
                cy for ty_counts in _n_chunks_per_cy.values() for cy in ty_counts
            )
        )
        logging.info(
            f"Clustered iteration: distributing centroid revenues across CYs "
            f"{_all_dest_cys}. Cluster->CY counts: {_n_per_cluster_cy}"
        )
    else:
        _n_per_cluster_cy = {}
        _n_chunks_per_cy = {}
        _all_dest_cys = []

    # Initialize collections for lifetimes, capacities, and hurdle rates
    lifetimes_collected = {}  # (component, name) → lifetime value
    decommissioning_years_collected = {}  # (component, name) → decommissioning_year
    capacities_by_ty = {}  # ty → dict[(component, name) → capacity]
    hurdle_rates_by_ty = {}  # ty → dict[(component, name) → hurdle_rate]

    # Needed parameters
    discount_rate = snakemake.config["costs"]["discount_rate"]
    use_hurdle_rates = snakemake.config["costs"]["use_hurdle_rates"]
    id_free_commitment_carriers = snakemake.config["eva"]["base_network"][
        "id_free_commitment_carriers"
    ]

    if asset_index.empty:
        logging.info(
            "No assets found. Creating empty NPV and NPV computation log files."
        )
        # Create empty DataFrame with proper structure to match expected format
        # Get target years from config for column structure
        ty_years = snakemake.config["scenario"]["ty"]
        cy_years = snakemake.config["scenario"]["cy"]
        empty_index = pd.MultiIndex.from_tuples([], names=["component", "name"])
        empty_npv = pd.DataFrame(index=empty_index, columns=ty_years, dtype=float)
        empty_npv.to_csv(snakemake.output.npv)  # ty: ignore[possibly-missing-attribute]

        # Create empty NPV computation log with proper column structure
        # Generate column names for all metrics
        log_columns = ["component", "name"]

        # Add energy_revenue, opex and opex_suc_sdc columns (have ft, ty, cy structure)
        for metric in ["energy_revenue", "opex", "opex_suc_sdc"]:
            for ft in ["da", "id"]:
                for ty in ty_years:
                    for cy in cy_years:
                        log_columns.append(f"{metric}_{ft}_{ty}_{cy}")

        # Add reserves_revenue, exogenous_revenue and capex_annualized columns (only ty, cy structure)
        for metric in ["reserves_revenue", "exogenous_revenue", "capex_annualized"]:
            for ty in ty_years:
                for cy in cy_years:
                    log_columns.append(f"{metric}_{ty}_{cy}")

        # Create empty DataFrame with these columns
        empty_npv_computation_log = pd.DataFrame(columns=pd.Index(log_columns))
        empty_npv_computation_log.to_csv(
            snakemake.output.npv_computation_log,  # ty: ignore[possibly-missing-attribute]
            index=False,
        )

        logging.info(
            f"Created empty NPV computation log with {len(log_columns)} columns"
        )
        exit()

    # In perfect foresight mode, use market networks; in standard mode, use day-ahead networks
    if perfect_foresight:
        network_input = (
            snakemake.input.market_networks  # ty: ignore[possibly-missing-attribute]
        )
        price_cap_input = (
            snakemake.input.market_price_cap_timeseries  # ty: ignore[possibly-missing-attribute]
        )
    else:
        network_input = (
            snakemake.input.day_ahead_networks
        )  # ty: ignore[possibly-missing-attribute]
        price_cap_input = (
            snakemake.input.day_ahead_price_cap_timeseries
        )  # ty: ignore[possibly-missing-attribute]

    # Snakemake unwraps single-element named inputs to a plain string; ensure list.
    if isinstance(network_input, str):
        network_input = [network_input]

    network_input = [p for p in network_input if str(p).endswith(".nc")]
    primary_networks_list = [(net_path, i) for i, net_path in enumerate(network_input)]

    # Define kwargs for statistics calculations
    # groupby=False ensures per-unit values instead of grouping by carrier
    stats_kwargs = dict(groupby=False, comps=comps)

    # Process each primary network (DA in standard mode, market in perfect foresight mode)
    primary_processed = {}
    for net_path, i in primary_networks_list:
        primary = pypsa.Network(net_path)
        primary._file_path = net_path  # ty: ignore[unresolved-attribute]
        wildcards = primary.meta["wildcards"]

        ty, cy = wildcards["ty"], wildcards["cy"]
        cluster_id = primary.meta.get("cluster_id")

        # In perfect foresight mode, we treat market networks like ID (use "id" label)
        # In standard mode, use "da" label for day-ahead networks
        market_type = "id" if perfect_foresight else "da"
        # Accumulate revenues under (ty, cy) scenario key; centroid networks share the key
        key = (market_type, ty, cy)

        # Match price cap by (ty, cy) from network wildcards, not by position --
        # in clustered mode there are k centroid networks per (ty, cy) but only one price cap file.
        matching_caps = [p for p in price_cap_input if f"_{ty}_{cy}_" in str(p)]
        if matching_caps:
            primary._price_cap_csv_path = matching_caps[
                0
            ]  # ty: ignore[unresolved-attribute]
        else:
            logging.warning(
                f"No price cap CSV found for ty={ty}, cy={cy}. "
                f"Available: {price_cap_input}"
            )
            primary._price_cap_csv_path = price_cap_input[
                0
            ]  # ty: ignore[unresolved-attribute]
        primary._config_price_cap = snakemake.config  # ty: ignore[unresolved-attribute]

        # Collect lifetimes and capacities for ALL assets in this network
        for c in comps:
            # Include ALL assets for NPV calculation (not just extendable/decommissionable)
            df = primary.static(c)
            assets_in_network = df.index

            for asset_name in assets_in_network:
                asset_key = (c, asset_name)

                # Collect lifetime and decommissioning_year (first occurrence only)
                if asset_key not in lifetimes_collected:
                    lifetimes_collected[asset_key] = df.loc[asset_name, "lifetime"]
                    decom_year = df.loc[asset_name, "decommissioning_year"]
                    decommissioning_years_collected[asset_key] = int(decom_year)

                # Collect capacity and hurdle_rate for this target year (first cy only)
                if ty not in capacities_by_ty:
                    capacities_by_ty[ty] = {}
                    hurdle_rates_by_ty[ty] = {}
                if asset_key not in capacities_by_ty[ty]:
                    if c in ["Generator", "StorageUnit"] and "p_nom" in df.columns:
                        capacities_by_ty[ty][asset_key] = df.loc[asset_name, "p_nom"]
                        # Collect hurdle_rate for this TY
                        if "hurdle_rate" in df.columns:
                            hurdle_rates_by_ty[ty][asset_key] = df.loc[
                                asset_name, "hurdle_rate"
                            ]
                        else:
                            hurdle_rates_by_ty[ty][asset_key] = (
                                discount_rate  # Fallback
                            )

        s_primary = primary.statistics

        # Using reindex to avoid problems when there is no revenue for an extendable component
        raw_revenue = s_primary.revenue(
            **stats_kwargs  # ty: ignore[invalid-argument-type]
        )
        energy_revenue = raw_revenue.reindex(asset_index).fillna(0)

        # Calculate reserves revenue, scaled by snapshot weights for centroid networks
        logging.info(
            f"Calculating reserves revenue for {len(asset_index)} extendable assets..."
        )
        reserves_revenue = calculate_reserves_revenue(primary, asset_index)
        if cluster_id is not None:
            # Centroid networks represent multiple weeks; snapshot_weightings.generators
            # holds the scaled_weight multiplier set during slicing
            scaled_weights = primary.snapshot_weightings["generators"]
            weight_scalar = scaled_weights.sum() / max(len(scaled_weights), 1)
            reserves_revenue = reserves_revenue * weight_scalar

        # Calculate exogenous revenues from network attributes (only if explicitly enabled)
        exog_enabled = snakemake.config["costs"]["exogenous_revenues"]["enable"]
        if exog_enabled:
            exogenous_revenue = load_exogenous_revenues(asset_index, primary)
        else:
            exogenous_revenue = pd.Series(
                0.0, index=asset_index, name="exogenous_revenue"
            )
            logging.info("Exogenous revenues disabled — skipping.")
        # Log summary for debugging
        energy_total = energy_revenue.sum()
        reserves_total = reserves_revenue.sum()
        exogenous_total = exogenous_revenue.sum()
        logging.info(
            f"Revenue summary: Energy={energy_total:.2f}, "
            f"Reserves={reserves_total:.2f}, "
            f"Exogenous={exogenous_total:.2f}"
        )

        # Total revenue includes energy, reserves, and exogenous
        revenue = energy_revenue + reserves_revenue + exogenous_revenue

        # Calculate OPEX and CAPEX separately
        opex_marginal = (
            s_primary.opex(**stats_kwargs)  # ty: ignore[invalid-argument-type]
            .reindex(asset_index)
            .fillna(0)
        )
        # PyPSA 0.33.2 does not include start-up/shut-down costs in opex().
        # Add them manually.
        # Free-committable generators are excluded here: their realized suc/sdc
        # come from the ID network.
        suc_sdc = calculate_suc_sdc(
            primary, asset_index, exclude_carriers=id_free_commitment_carriers
        )
        opex_primary = opex_marginal + suc_sdc

        capex_primary = (
            s_primary.installed_capex(
                **stats_kwargs  # ty: ignore[invalid-argument-type]
            )
            .reindex(asset_index)
            .fillna(0)
        )

        # capital_cost is already 0 CAPEX-annuity for every existing asset
        # (apply_existing_asset_attributes forces CAPEX=0 upstream), so for
        # decommissionable assets capex_primary is already pure annuitized
        # FOM, not sunk investment cost -- it must count toward the
        # decommission NPV like it already does for non-decommissionable
        # existing assets, so it is not zeroed out here.
        capex_for_npv = capex_primary

        expense = opex_primary + capex_for_npv

        if _is_clustered_iteration and cluster_id is not None:
            # Cross-CY redistribution:
            # 1. Unscale PyPSA statistics (scaled by snapshot_weightings) to raw weekly values.
            # 2. Distribute raw weekly values to every destination CY weighted by N_c_destcy.
            # 3. Scale CAPEX/exogenous by N_c_destcy / n_chunks_destcy (fractional year share).
            #
            # Read scaled_weight from snapshot_weightings directly -- cluster_weight_scaled
            # is not preserved in meta after slice_network_to_centroid's __main__ block
            # overwrites m.meta with snakemake.config.
            sw_series = primary.snapshot_weightings["generators"]
            scaled_weight = float(sw_series.sum() / max(len(sw_series), 1))
            if math.isclose(scaled_weight, 0.0, abs_tol=1e-9):
                scaled_weight = 1.0

            # Raw (unscaled) weekly values
            raw_energy_revenue = energy_revenue / scaled_weight
            raw_opex_marginal = opex_marginal / scaled_weight
            raw_suc_sdc = (
                suc_sdc  # already unscaled (direct sum of binary events * cost)
            )
            raw_reserves_revenue = reserves_revenue / scaled_weight
            raw_exogenous_revenue = (
                exogenous_revenue  # annual value; handled via fraction
            )
            raw_capex_for_npv = capex_for_npv  # annual value; handled via fraction
            raw_capex_primary = capex_primary  # for logging

            # Per-TY lookup: week_clusters are keyed by ty to avoid cross-TY accumulation
            ty_str = str(ty)
            ty_cluster_counts = _n_per_cluster_cy.get(ty_str, {})
            ty_chunk_counts = _n_chunks_per_cy.get(ty_str, {})
            dest_counts = ty_cluster_counts.get(cluster_id, {})

            for dest_cy in _all_dest_cys:
                n_c_cy = dest_counts.get(dest_cy, 0)
                if n_c_cy == 0:
                    continue
                n_chunks_cy = ty_chunk_counts.get(dest_cy, 1)
                cy_fraction = n_c_cy / n_chunks_cy

                dest_key = ("da", ty, dest_cy)
                dest_year_key = (ty, dest_cy)

                # Energy revenue + reserves + exogenous: all scale by week count / fraction
                opex_scaled = raw_opex_marginal * n_c_cy
                suc_sdc_scaled = raw_suc_sdc * n_c_cy
                capex_scaled = raw_capex_for_npv * cy_fraction
                exog_scaled = raw_exogenous_revenue * cy_fraction
                reserves_scaled = raw_reserves_revenue * n_c_cy

                revenues[dest_key] = revenues.get(dest_key, 0) + (
                    raw_energy_revenue * n_c_cy + reserves_scaled + exog_scaled
                )
                expenses[dest_key] = expenses.get(dest_key, 0) + (
                    opex_scaled + suc_sdc_scaled + capex_scaled
                )

                # Logging accumulation
                energy_revenues_log[dest_key] = (
                    energy_revenues_log.get(dest_key, 0) + raw_energy_revenue * n_c_cy
                )
                opex_log[dest_key] = opex_log.get(dest_key, 0) + opex_scaled
                opex_suc_sdc_log[dest_key] = (
                    opex_suc_sdc_log.get(dest_key, 0) + suc_sdc_scaled
                )
                if dest_year_key not in reserves_revenues_log:
                    reserves_revenues_log[dest_year_key] = reserves_scaled
                else:
                    reserves_revenues_log[dest_year_key] = (
                        reserves_revenues_log[dest_year_key] + reserves_scaled
                    )
                if dest_year_key not in exogenous_revenues_log:
                    exogenous_revenues_log[dest_year_key] = exog_scaled
                else:
                    exogenous_revenues_log[dest_year_key] = (
                        exogenous_revenues_log[dest_year_key] + exog_scaled
                    )
                # CAPEX logged once per (ty, dest_cy) -- accumulate fractional portions
                if dest_year_key not in capex_annualized_log:
                    capex_annualized_log[dest_year_key] = (
                        raw_capex_primary * cy_fraction
                    )
                else:
                    capex_annualized_log[dest_year_key] = (
                        capex_annualized_log[dest_year_key]
                        + raw_capex_primary * cy_fraction
                    )
        else:
            # Full-run (non-clustered): accumulate directly under this network's (ty, cy).
            revenues[key] = revenues.get(key, 0) + revenue
            expenses[key] = expenses.get(key, 0) + expense

            energy_revenues_log[key] = energy_revenues_log.get(key, 0) + energy_revenue
            opex_log[key] = opex_log.get(key, 0) + opex_marginal
            opex_suc_sdc_log[key] = opex_suc_sdc_log.get(key, 0) + suc_sdc

            year_key = (wildcards["ty"], wildcards["cy"])
            if year_key not in reserves_revenues_log:
                reserves_revenues_log[year_key] = reserves_revenue
            else:
                reserves_revenues_log[year_key] = (
                    reserves_revenues_log[year_key] + reserves_revenue
                )
            if year_key not in exogenous_revenues_log:
                exogenous_revenues_log[year_key] = exogenous_revenue
            else:
                exogenous_revenues_log[year_key] = (
                    exogenous_revenues_log[year_key] + exogenous_revenue
                )
            if year_key not in capex_annualized_log:
                capex_annualized_log[year_key] = capex_primary

        # Store processed network for matching with ID networks
        # Use (ty, cy, cluster_id) key so centroid ID networks can be matched precisely
        lookup_key = (ty, cy, cluster_id) if cluster_id is not None else (ty, cy)
        primary_processed[lookup_key] = primary

    # Process intraday networks only in standard EVA mode (not in perfect foresight mode)
    if not perfect_foresight:
        for id_path in (
            snakemake.input.intraday_networks  # ty: ignore[possibly-missing-attribute]
        ):
            id = pypsa.Network(id_path)
            # Parse wildcards from path to be robust against missing metadata
            m = re.search(r"_(\d{4})_(\d{4})_actual_([^_/]+)_", str(id_path))
            if m:
                ty, cy, fo = m.group(1), m.group(2), m.group(3)
                id_wildcards = {"ty": ty, "cy": cy, "fo": fo}
            else:
                id_wildcards = id.meta.get("wildcards", {})
                ty, cy = id_wildcards.get("ty", ""), id_wildcards.get("cy", "")
            id_cluster_id = id.meta.get("cluster_id")

            # Prefer exact (ty, cy, id_cluster_id) match for centroid ID networks
            # Match types: in primary_processed keys, ty might be string or int.
            # Handle lookup by checking both string and int representations of ty.
            ty_lookup = int(ty) if isinstance(ty, str) and ty.isdigit() else ty
            lookup_key_typed = (
                (ty_lookup, cy, id_cluster_id)
                if id_cluster_id is not None
                else (ty_lookup, cy)
            )
            if lookup_key_typed not in primary_processed:
                # Fallback to string representation lookup
                ty_str = str(ty)
                lookup_key_str = (
                    (ty_str, cy, id_cluster_id)
                    if id_cluster_id is not None
                    else (ty_str, cy)
                )
                if lookup_key_str in primary_processed:
                    lookup_key_typed = lookup_key_str

            if lookup_key_typed not in primary_processed:
                # Fallback to (ty, cy) without cluster_id
                ty_lookup = int(ty) if isinstance(ty, str) and ty.isdigit() else ty
                if (ty_lookup, cy) in primary_processed:
                    lookup_key_typed = (ty_lookup, cy)
                elif (str(ty), cy) in primary_processed:
                    lookup_key_typed = (str(ty), cy)
                else:
                    raise ValueError(
                        f"No matching day-ahead network found for intraday network "
                        f"with ty={ty}, cy={cy}, cluster_id={id_cluster_id}. "
                        f"Available: {list(primary_processed.keys())}"
                    )

            da = primary_processed[lookup_key_typed]
            key = ("id", ty, cy, id_wildcards.get("fo", ""))
            apply_relative_dispatch(id, da)
            s_id = id.statistics

            # Using reindex to avoid problems when there is no revenue for a component
            revenue = (
                s_id.revenue(
                    **stats_kwargs  # ty: ignore[invalid-argument-type]
                )
                .reindex(asset_index)
                .fillna(0)
            )
            expense_marginal = (
                s_id.opex(
                    **stats_kwargs  # ty: ignore[invalid-argument-type]
                )
                .reindex(asset_index)
                .fillna(0)
            )
            # PyPSA 0.33.2 does not include start-up/shut-down costs in opex().
            # Add them manually.
            # Only free-committable generators contribute ID suc/sdc: their
            # commitment is re-optimized in ID. Fixed generators already have
            # their costs from the DA network.
            suc_sdc_id = calculate_suc_sdc(
                id, asset_index, include_carriers=id_free_commitment_carriers
            )
            expense = expense_marginal + suc_sdc_id

            if _is_clustered_iteration and id_cluster_id is not None:
                # Cross-CY redistribution for intraday centroid networks.
                # ID networks have no CAPEX/exogenous -- only energy revenue + marginal opex + SUC/SDC.
                sw_series_id = id.snapshot_weightings["generators"]
                scaled_weight_id = float(sw_series_id.sum() / max(len(sw_series_id), 1))
                if math.isclose(scaled_weight_id, 0.0, abs_tol=1e-9):
                    scaled_weight_id = 1.0

                raw_id_revenue = revenue / scaled_weight_id
                raw_id_opex = expense_marginal / scaled_weight_id
                raw_id_suc_sdc = suc_sdc_id  # already unscaled

                fo = id_wildcards.get("fo", "")
                ty_str_id = str(ty)
                ty_cluster_counts_id = _n_per_cluster_cy.get(ty_str_id, {})
                ty_chunk_counts_id = _n_chunks_per_cy.get(ty_str_id, {})
                dest_counts = ty_cluster_counts_id.get(id_cluster_id, {})
                for dest_cy in _all_dest_cys:
                    n_c_cy = dest_counts.get(dest_cy, 0)
                    if n_c_cy == 0:
                        continue
                    dest_key = ("id", ty, dest_cy, fo)
                    revenues[dest_key] = (
                        revenues.get(dest_key, 0) + raw_id_revenue * n_c_cy
                    )
                    expenses[dest_key] = expenses.get(dest_key, 0) + (
                        raw_id_opex * n_c_cy + raw_id_suc_sdc * n_c_cy
                    )
                    energy_revenues_log[dest_key] = (
                        energy_revenues_log.get(dest_key, 0) + raw_id_revenue * n_c_cy
                    )
                    opex_log[dest_key] = (
                        opex_log.get(dest_key, 0) + raw_id_opex * n_c_cy
                    )
                    opex_suc_sdc_log[dest_key] = (
                        opex_suc_sdc_log.get(dest_key, 0) + raw_id_suc_sdc * n_c_cy
                    )
            else:
                revenues[key] = revenues.get(key, 0) + revenue
                expenses[key] = expenses.get(key, 0) + expense

                # For intraday: only energy revenue and opex (reserves and capex already logged from DA)
                energy_revenues_log[key] = revenue
                opex_log[key] = expense_marginal
                opex_suc_sdc_log[key] = suc_sdc_id

    # Separate DA and ID revenues/expenses
    # In perfect foresight mode, everything is labeled as "id" (market networks treated like ID)
    da_revenues = {k: v for k, v in revenues.items() if k[0] == "da"}
    da_expenses = {k: v for k, v in expenses.items() if k[0] == "da"}
    id_revenues = {k: v for k, v in revenues.items() if k[0] == "id"}
    id_expenses = {k: v for k, v in expenses.items() if k[0] == "id"}

    if perfect_foresight:
        # Perfect foresight mode: only ID (market) revenues/expenses exist
        # No DA networks, so create empty DA dataframes
        da_revenues_df = pd.DataFrame()
        da_expenses_df = pd.DataFrame()

        # ID revenues have (ft, ty, cy) structure in perfect foresight (no separate fo averaging needed in file structure)
        # However, the data may still have fo dimension if forced outages are enabled
        if id_revenues:
            # Check if we have fo dimension by looking at key structure
            sample_key = next(iter(id_revenues))
            if len(sample_key) == 4:  # (ft, ty, cy, fo)
                id_revenues_df = pd.concat(
                    id_revenues, axis=1, names=["ft", "ty", "cy", "fo"]
                )
                id_expenses_df = pd.concat(
                    id_expenses, axis=1, names=["ft", "ty", "cy", "fo"]
                )
                # Average over forced outage samples
                id_revenues_avg = (
                    id_revenues_df.T.groupby(level=["ft", "ty", "cy"]).mean().T
                )
                id_expenses_avg = (
                    id_expenses_df.T.groupby(level=["ft", "ty", "cy"]).mean().T
                )
            else:  # (ft, ty, cy) - no fo dimension
                id_revenues_avg = pd.concat(
                    id_revenues, axis=1, names=["ft", "ty", "cy"]
                )
                id_expenses_avg = pd.concat(
                    id_expenses, axis=1, names=["ft", "ty", "cy"]
                )
        else:
            id_revenues_avg = pd.DataFrame()
            id_expenses_avg = pd.DataFrame()

        # In perfect foresight, revenues/expenses are only from ID (market networks)
        revenues_df = id_revenues_avg
        expenses_df = id_expenses_avg
    else:
        # Standard EVA mode: combine DA and ID results
        # Combine DA results (no forced outage dimension)
        da_revenues_df = pd.concat(da_revenues, axis=1, names=["ft", "ty", "cy"])
        da_expenses_df = pd.concat(da_expenses, axis=1, names=["ft", "ty", "cy"])

        # Combine ID results (has forced outage dimension) and average over fo samples
        if id_revenues:
            id_revenues_df = pd.concat(
                id_revenues, axis=1, names=["ft", "ty", "cy", "fo"]
            )
            id_expenses_df = pd.concat(
                id_expenses, axis=1, names=["ft", "ty", "cy", "fo"]
            )

            # Average over forced outage samples to get (ft, ty, cy) structure
            id_revenues_avg = (
                id_revenues_df.T.groupby(level=["ft", "ty", "cy"]).mean().T
            )
            id_expenses_avg = (
                id_expenses_df.T.groupby(level=["ft", "ty", "cy"]).mean().T
            )
        else:
            # No intraday networks (shouldn't happen but handle gracefully)
            id_revenues_avg = pd.DataFrame()
            id_expenses_avg = pd.DataFrame()

        # Combine DA and ID (averaged) results
        revenues_df = pd.concat([da_revenues_df, id_revenues_avg], axis=1)
        expenses_df = pd.concat([da_expenses_df, id_expenses_avg], axis=1)

    # Calculate Net Profit for each asset, ty, cy
    net_profit = revenues_df.sub(expenses_df, fill_value=0).fillna(0)

    # Step 1: Calculate annual profit per target year (average or weighted over CYs)
    profit_by_ty_cy = net_profit.T.groupby(level=["ty", "cy"]).sum()

    npv_cy_config = snakemake.config["eva"]["npv"]["weighted_calculation"]
    if npv_cy_config["enabled"]:
        raw_weights = npv_cy_config["cy_weights"] or {}
        # Normalise keys to strings to match wildcard-sourced CY values
        all_cy_weights: dict[str, float] = {
            str(k): float(v) for k, v in raw_weights.items()
        }
        # Filter to only CYs that are actually present in this simulation run.
        present_cys = set(profit_by_ty_cy.index.get_level_values("cy").unique())
        cy_weights: dict[str, float] = {
            cy: w for cy, w in all_cy_weights.items() if cy in present_cys
        }
        total_weight = sum(cy_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"eva.npv.weighted_calculation.cy_weights must sum to 1.0 for the "
                f"simulated CYs {sorted(present_cys)}, got {total_weight:.6f} "
                f"(weights: {cy_weights})"
            )
        cy_level = profit_by_ty_cy.index.get_level_values("cy")
        weight_values = cy_level.map(lambda cy: cy_weights.get(str(cy), 0.0))
        annual_profit = (
            profit_by_ty_cy.mul(weight_values.values, axis=0)
            .groupby(level="ty")
            .sum()
            .T
        )
        logging.info(f"Weighted NPV CY averaging enabled. Weights: {cy_weights}")
    else:
        annual_profit = profit_by_ty_cy.groupby(level="ty").mean().T

    # Step 2: Interpolate profits across all years for each asset
    annual_profit.columns = annual_profit.columns.astype(int)
    ty_years = annual_profit.columns

    # --- Build lifetimes, decommissioning years, and capacities from collected data ---
    # Convert collected lifetimes to Series
    lifetimes = pd.Series(lifetimes_collected).reindex(annual_profit.index)

    # Replace NaN lifetimes with 0 (assets that don't exist in any TY/CY)
    lifetimes = lifetimes.fillna(0.0)

    # Convert collected decommissioning_years to Series
    decommissioning_years = pd.Series(decommissioning_years_collected).reindex(
        annual_profit.index
    )

    # Convert collected capacities to DataFrame (columns = ty_years)
    capacity_df = pd.DataFrame(
        {int(ty): pd.Series(cap_dict) for ty, cap_dict in capacities_by_ty.items()}
    ).reindex(index=annual_profit.index)

    # --- Build year range for interpolation and discounting ---
    min_ty = ty_years.min()
    # Use maximum decommissioning year to determine year range
    max_year = int(decommissioning_years.max())
    all_years = pd.Index(range(min_ty, max_year + 1), name="year")

    # --- NPV Calculation ---
    present_year = min_ty  # Assume base year

    # Get discount rates (either hurdle rates or fallback discount_rate)
    if use_hurdle_rates:
        logging.info("Using technology-specific hurdle rates for NPV calculation")
        # Build DataFrame with TY-specific hurdle rates: rows=assets, columns=TY years
        # This allows different hurdle rates per asset per target year
        discount_rates_by_ty_df = pd.DataFrame(
            {
                int(ty): pd.Series(rates_dict)
                for ty, rates_dict in hurdle_rates_by_ty.items()
            }
        ).reindex(index=annual_profit.index)

        # Fill missing values with global discount_rate
        discount_rates_by_ty_df = discount_rates_by_ty_df.fillna(discount_rate)

        if discount_rates_by_ty_df.empty:
            # Fallback if no hurdle rates were collected
            logging.warning(
                "No hurdle rates found in networks, using global discount rate"
            )
            # Create single-column DataFrame for consistency
            discount_rates_by_ty_df = pd.DataFrame(
                {ty_years[0]: discount_rate}, index=annual_profit.index
            )
    else:
        logging.info(
            f"Using global discount rate ({discount_rate}) for all assets (use_hurdle_rates=False)"
        )
        # Create single-column DataFrame for consistency
        discount_rates_by_ty_df = pd.DataFrame(
            {ty_years[0]: discount_rate}, index=annual_profit.index
        )

    # Calculate asset-specific discount factors for all years
    lifetimes = lifetimes.reindex(annual_profit.index)
    discount_factors = pd.DataFrame(
        index=annual_profit.index,
        columns=all_years,
        dtype=float,
    )

    # Calculate discount factor for each asset with TY-specific hurdle rates
    # For each year, use the appropriate TY's hurdle rate
    ty_years_sorted = sorted(discount_rates_by_ty_df.columns)

    for asset in discount_rates_by_ty_df.index:
        for year in all_years:
            # Determine which TY's rate to use for this year
            # Use the rate from the TY that year falls within
            ty_for_year = ty_years_sorted[0]  # Default to first TY
            for ty in ty_years_sorted:
                if year >= ty:
                    ty_for_year = ty
                else:
                    break

            rate = discount_rates_by_ty_df.loc[asset, ty_for_year]
            discount_factors.loc[asset, year] = 1 / (1 + rate) ** (year - present_year)

    # Zero discounting beyond asset decommissioning year
    # For each asset, zero out all years >= decommissioning_year
    for asset in decommissioning_years.index:
        decom_year = decommissioning_years[asset]
        discount_factors.loc[asset, all_years >= decom_year] = 0

    # Zero out for assets with lifetime <= 0 (assets not in any TY/CY)
    discount_factors.loc[lifetimes <= 0] = 0

    # NPV per unit capacity at each target year
    npv_at_target_years = npv_per_mw_at_target_years(
        annual_profit, capacity_df, discount_factors
    )

    # Create detailed NPV computation log (does not affect NPV calculation)
    logging.info(
        "Creating NPV computation log with detailed revenue/expense breakdown..."
    )

    # Separate DA and ID logs similar to main calculation
    da_energy_log = {k: v for k, v in energy_revenues_log.items() if k[0] == "da"}
    id_energy_log = {k: v for k, v in energy_revenues_log.items() if k[0] == "id"}
    da_opex_log = {k: v for k, v in opex_log.items() if k[0] == "da"}
    id_opex_log = {k: v for k, v in opex_log.items() if k[0] == "id"}
    da_opex_suc_sdc_log = {k: v for k, v in opex_suc_sdc_log.items() if k[0] == "da"}
    id_opex_suc_sdc_log = {k: v for k, v in opex_suc_sdc_log.items() if k[0] == "id"}

    # Convert detailed tracking dictionaries to DataFrames
    # Handle DA (3-level) and ID (4-level) separately, then average ID over fo
    da_energy_df = (
        pd.concat(da_energy_log, axis=1, names=["ft", "ty", "cy"])
        if da_energy_log
        else pd.DataFrame()
    )
    da_opex_df = (
        pd.concat(da_opex_log, axis=1, names=["ft", "ty", "cy"])
        if da_opex_log
        else pd.DataFrame()
    )

    if id_energy_log:
        # Check key structure to determine if we have fo dimension
        sample_key = next(iter(id_energy_log))
        has_fo_dimension = len(sample_key) == 4  # (ft, ty, cy, fo)

        if has_fo_dimension:
            # Standard mode: keys are (ft, ty, cy, fo)
            id_energy_df = pd.concat(
                id_energy_log, axis=1, names=["ft", "ty", "cy", "fo"]
            )
            id_opex_df = pd.concat(id_opex_log, axis=1, names=["ft", "ty", "cy", "fo"])
            id_opex_suc_sdc_df = pd.concat(
                id_opex_suc_sdc_log, axis=1, names=["ft", "ty", "cy", "fo"]
            )
            # Average over forced outage samples
            id_energy_avg = id_energy_df.T.groupby(level=["ft", "ty", "cy"]).mean().T
            id_opex_avg = id_opex_df.T.groupby(level=["ft", "ty", "cy"]).mean().T
            id_opex_suc_sdc_avg = (
                id_opex_suc_sdc_df.T.groupby(level=["ft", "ty", "cy"]).mean().T
            )
        else:
            # Perfect foresight mode: keys are (ft, ty, cy)
            id_energy_avg = pd.concat(id_energy_log, axis=1, names=["ft", "ty", "cy"])
            id_opex_avg = pd.concat(id_opex_log, axis=1, names=["ft", "ty", "cy"])
            id_opex_suc_sdc_avg = (
                pd.concat(id_opex_suc_sdc_log, axis=1, names=["ft", "ty", "cy"])
                if id_opex_suc_sdc_log
                else pd.DataFrame()
            )
    else:
        id_energy_avg = pd.DataFrame()
        id_opex_avg = pd.DataFrame()
        id_opex_suc_sdc_avg = pd.DataFrame()

    # Combine DA and ID (averaged)
    energy_revenues_df = pd.concat([da_energy_df, id_energy_avg], axis=1)
    opex_df = pd.concat([da_opex_df, id_opex_avg], axis=1)
    da_opex_suc_sdc_df = (
        pd.concat(da_opex_suc_sdc_log, axis=1, names=["ft", "ty", "cy"])
        if da_opex_suc_sdc_log
        else pd.DataFrame()
    )
    opex_suc_sdc_df = pd.concat([da_opex_suc_sdc_df, id_opex_suc_sdc_avg], axis=1)

    reserves_revenues_df = (
        pd.concat(reserves_revenues_log, axis=1, names=["ty", "cy"])
        if reserves_revenues_log
        else pd.DataFrame()
    )
    exogenous_revenues_df = (
        pd.concat(exogenous_revenues_log, axis=1, names=["ty", "cy"])
        if exogenous_revenues_log
        else pd.DataFrame()
    )
    capex_annualized_df = (
        pd.concat(capex_annualized_log, axis=1, names=["ty", "cy"])
        if capex_annualized_log
        else pd.DataFrame()
    )

    # Create multi-level columns for the log
    log_dfs = {
        "energy_revenue": energy_revenues_df,
        "reserves_revenue": reserves_revenues_df,
        "exogenous_revenue": exogenous_revenues_df,
        "opex": opex_df,
        "opex_suc_sdc": opex_suc_sdc_df,
        "capex_annualized": capex_annualized_df,
    }

    # Combine all components into a single DataFrame with multi-level columns
    # Handle different column structures
    combined_dfs = []

    # Add energy, opex and opex_suc_sdc (have ft, ty, cy structure)
    for metric in ["energy_revenue", "opex", "opex_suc_sdc"]:
        df = log_dfs[metric]
        df.columns = pd.MultiIndex.from_tuples(
            [(metric, ft, ty, cy) for ft, ty, cy in df.columns],
            names=["metric", "ft", "ty", "cy"],
        )
        combined_dfs.append(df)

    # Add reserves_revenue, exogenous_revenue, and capex_annualized (only ty, cy structure)
    for metric in ["reserves_revenue", "exogenous_revenue", "capex_annualized"]:
        df = log_dfs[metric]
        df.columns = pd.MultiIndex.from_tuples(
            [(metric, ty, cy) for ty, cy in df.columns], names=["metric", "ty", "cy"]
        )
        combined_dfs.append(df)

    # Combine all DataFrames
    npv_computation_log = pd.concat(combined_dfs, axis=1)

    # Sort columns
    npv_computation_log = npv_computation_log.sort_index(axis=1)

    # Flatten column names for CSV output
    flattened_columns = []
    for col in npv_computation_log.columns:
        if len(col) == 4:  # metric, ft, ty, cy
            flattened_columns.append(f"{col[0]}_{col[1]}_{col[2]}_{col[3]}")
        elif len(col) == 3:  # metric, ty, cy (for capex)
            flattened_columns.append(f"{col[0]}_{col[1]}_{col[2]}")

    npv_computation_log.columns = flattened_columns

    # Reset index to have component and name as regular columns
    npv_computation_log = npv_computation_log.reset_index()

    # Add carrier and bus columns so results can be grouped by carrier
    npv_out = npv_at_target_years.reset_index()
    npv_out.insert(
        2,
        "carrier",
        npv_out.apply(
            lambda r: asset_carrier_bus.get((r["component"], r["name"]), ("", ""))[0],
            axis=1,
        ),
    )
    npv_out.insert(
        3,
        "bus",
        npv_out.apply(
            lambda r: asset_carrier_bus.get((r["component"], r["name"]), ("", ""))[1],
            axis=1,
        ),
    )
    npv_out.to_csv(
        snakemake.output.npv,  # ty: ignore[possibly-missing-attribute]
        index=False,
    )
    npv_computation_log.to_csv(
        snakemake.output.npv_computation_log,  # ty: ignore[possibly-missing-attribute]
        index=False,
    )

    logging.info(
        f"NPV computation log saved with {len(npv_computation_log)} assets and {len(npv_computation_log.columns) - 2} metric columns"
    )
