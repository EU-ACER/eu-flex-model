# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
#
# SPDX-License-Identifier: MIT


import logging
from collections.abc import Iterable

import linopy
import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

REGION_COLS = ["geometry", "name", "x", "y", "country"]


def _patch_pypsa_meshed_threshold():
    """Patch linopy.Model.__init__ wrapper to filter out meshed_threshold."""

    original_model_init = linopy.Model.__init__

    def patched_model_init(self, *args, **kwargs):
        # Remove meshed_threshold if present - it's not a valid linopy.Model parameter
        kwargs.pop("meshed_threshold", None)
        return original_model_init(self, *args, **kwargs)

    linopy.Model.__init__ = patched_model_init  # ty: ignore[invalid-assignment]


# Apply the patch immediately when helpers module is imported
_patch_pypsa_meshed_threshold()


def rename_techs(label: str) -> str:
    """
    Rename technology labels for better readability.

    Removes some prefixes and renames if certain conditions defined in function body are met.

    Parameters
    ----------
    label: str
        Technology label to be renamed

    Returns
    -------
    str
        Renamed label
    """
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
        # "CC": "CC"
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "offwind-float": "offshore wind (Float)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "NH3": "ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new
    return label


def add_missing_carriers(n: pypsa.Network, carriers: Iterable) -> None:
    """
    Add missing carriers to the network without raising errors.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to add carriers to.
    carriers : Iterable
        Iterable of carrier names to ensure exist in the network.

    Returns
    -------
    None
        Modifies the network in-place.
    """
    # Order-preserving difference: n.carriers row order follows from this, and
    # a hash-ordered set would make it vary between runs.
    known = set(n.carriers.index)
    missing_carriers = [c for c in dict.fromkeys(carriers) if c not in known]
    if len(missing_carriers) > 0:
        n.add("Carrier", missing_carriers)


def sanitize_carriers(n: pypsa.Network, config: dict) -> None:
    """
    Sanitize the carrier information in a PyPSA Network object.

    The function ensures that all unique carrier names are present in the network's
    carriers attribute, and adds nice names and colors for each carrier according
    to the provided configuration dictionary.

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA Network object that represents an electrical power system.
    config : dict
        A dictionary containing configuration information, specifically the
        "plotting" key with "nice_names" and "tech_colors" keys for carriers.

    Returns
    -------
    None
        The function modifies the 'n' PyPSA Network object in-place, updating the
        carriers attribute with nice names and colors.

    Warnings
    --------
    Raises a warning if any carrier's "tech_colors" are not defined in the config dictionary.
    """
    for c in n.iterate_components():
        if "carrier" in c.df:
            add_missing_carriers(n, c.df.carrier)

    invalid = n.carriers.index.to_series().astype(str).str.strip() == ""
    if invalid.any():
        to_drop = n.carriers.index[invalid]
        logger.warning(f"Dropping blank/invalid carriers: {to_drop.tolist()}")
        n.carriers.drop(index=to_drop, inplace=True)

    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series())
    )
    n.carriers["nice_name"] = n.carriers.nice_name.where(
        n.carriers.nice_name != "", nice_names
    )

    tech_colors = config["plotting"]["tech_colors"]
    colors = pd.Series(tech_colors).reindex(carrier_i)
    # try to fill missing colors with tech_colors after renaming
    missing_colors_i = colors[colors.isna()].index
    colors[missing_colors_i] = missing_colors_i.map(rename_techs).map(tech_colors)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = n.carriers.color.where(n.carriers.color != "", colors)


# Custom (non-default PyPSA) attributes that are actually used somewhere in the workflow.
# Columns present in the network but absent from all three lists will be dropped by
# reduce_network_size().  Add a column here when you add a new custom attribute to the
# workflow; remove it when you retire one.
CUSTOM_ATTRS: dict[str, dict[str, list[str]]] = {
    # ── n.generators (static) ───────────────────────────────────────────────
    "generators": {
        "static": [
            # must-run
            "is_must_run",
            "must-run capacity",
            "weighted average must-run ratio",
            # unit commitment
            "committable",
            "ramp_limit_up",
            "ramp_limit_down",
            "ramp_limit_start_up",
            "ramp_limit_shut_down",
            "start_up_cost",
            "shut_down_cost",
            "stand_by_cost",
            # outage modelling
            "avg_forced_outage_rate",
            "n_units",
            "max_forced_outage_capacity",
            "planned_outage_rate",
            "planned_outage_min_hours",
            "planned_outage_annual_windows",
            "is_existing",
            # reserve eligibility (read via reserve_col = f"reserve_{type}" loop)
            "reserve_FCR",
            "reserve_FRR",
            # emissions
            "gross_co2_emissions",
            # identification
            "unit_id",
            "dsr_type",
            # DSR-specific
            "max_daily_hours",
            "shifting_time_window",
            # EVA lifecycle / cost
            "lifetime",
            "build_year",
            "decommissioning_year",
            "lifetime_extendable",
            "lifetime_extension_capex",
            "lifetime_extension_years",
            "Lifetime_Extension_FOM",
            "hurdle_rate",
            "hurdle_rate_demothballing",
            "hurdle_rate_lifetime_extension",
            "Mothballing_FOM",
            "Mothballing_Cost",
            "Demothballing_Cost",
            # capacity management
            "p_nom_decommissionable",
            "FCR_up_amount",
            "FCR_down_amount",
            "FRR_up_amount",
            "FRR_down_amount",
            "max_fcr_provision",
            "max_frr_provision_up",
            "max_frr_provision_down",
            "CHP_revenue",
            "CM_revenue",
            "balancing_revenue",
            "congestion_revenue",
        ],
        "timeseries": [
            "availability",
            "maintenance_plan",
            "must_run_capacity",
            "p_max_pu_original",
            "ramp_limit_up",
            "ramp_limit_down",
            # reserve results written by market_functions
            "r_FCR_up",
            "r_FCR_down",
            "r_FRR_up",
            "r_FRR_down",
        ],
    },
    # ── n.storage_units (static) ────────────────────────────────────────────
    "storage_units": {
        "static": [
            # reserve eligibility
            "reserve_FCR",
            "reserve_FRR",
            # identification
            "unit_id",
            # capacity management
            "p_nom_decommissionable",
            # EVA lifecycle / cost
            "lifetime",
            "build_year",
            "decommissioning_year",
            "lifetime_extendable",
            "lifetime_extension_capex",
            "lifetime_extension_years",
            "Lifetime_Extension_FOM",
            "hurdle_rate",
            "hurdle_rate_lifetime_extension",
            "FCR_up_amount",
            "FCR_down_amount",
            "FRR_up_amount",
            "FRR_down_amount",
            "CHP_revenue",
            "CM_revenue",
            "balancing_revenue",
            "congestion_revenue",
        ],
        "timeseries": [
            "min_pump_mw",
            "storage_plan",
            "state_of_charge_max",
            "state_of_charge_min",
            "total_reserve_up",
            "total_reserve_down",
            # reserve results
            "r_FRR_up",
            "r_FRR_down",
            # DSR flexible-demand tracking
            "flexible_demand",
        ],
    },
    # ── n.buses (static) ────────────────────────────────────────────────────
    "buses": {
        "static": [
            "country",
            "core_region",
            "nordic_region",
            "co2_price",
            # reserve blocking
            "reserve_FCR_id_blocked",
            "reserve_FRR_id_blocked",
            "reserve_FRR_reserve_blocking_method",
            # dynamic reserve columns added by market_functions (reserve_FCR, reserve_FRR, …)
            # kept via startswith("reserve_") check in reduce_network_size
        ],
        "timeseries": [
            "marginal_price",
            "network_needs",
            "curtailment_targets",
            "original_curtailment_timeseries",
            "adjusted_curtailment_timeseries",
            # intra-day FRR blocking
            "frr_blocked_up",
            "frr_blocked_down",
        ],
    },
}


def reduce_network_size(n: pypsa.Network) -> pypsa.Network:
    """
    Drop custom (non-default PyPSA) attributes that are never used in the workflow.

    Only static columns and timeseries DataFrames listed in ``CUSTOM_ATTRS`` are
    kept.  All default PyPSA columns (those present in a freshly-created component
    DataFrame) are always preserved.

    Parameters
    ----------
    n : pypsa.Network
        Network as returned by ``build_network_from_database``, possibly with many
        extra columns that were loaded from the database but are never read by any
        downstream script.

    Returns
    -------
    pypsa.Network
        The same network object, modified in-place, with unused columns removed.
    """
    component_map = {
        "generators": ("Generator", n.generators, n.generators_t),
        "storage_units": ("StorageUnit", n.storage_units, n.storage_units_t),
        "buses": ("Bus", n.buses, n.buses_t),
    }

    for key, (comp_name, static_df, ts_obj) in component_map.items():
        # ── default columns for this component ──────────────────────────────
        default_cols = set(pypsa.Network().df(comp_name).columns)

        kept_custom = set(CUSTOM_ATTRS[key]["static"])
        # buses: also keep any column that starts with "reserve_" (dynamic names like
        # reserve_FCR, reserve_FRR added by market_functions via reserve_type loop)
        if key == "buses":
            extra = {c for c in static_df.columns if c.startswith("reserve_")}
            kept_custom |= extra

        allowed_static = default_cols | kept_custom
        drop_static = [c for c in static_df.columns if c not in allowed_static]
        if drop_static:
            static_df.drop(columns=drop_static, inplace=True)
            logger.debug(
                f"reduce_network_size: dropped {len(drop_static)} static columns "
                f"from {key}: {drop_static}"
            )

        # ── timeseries ───────────────────────────────────────────────────────
        default_ts = set(pypsa.Network().pnl(comp_name).keys())
        allowed_ts = default_ts | set(CUSTOM_ATTRS[key]["timeseries"])
        drop_ts = [k for k in list(ts_obj) if k not in allowed_ts]
        for k in drop_ts:
            del ts_obj[k]
        if drop_ts:
            logger.debug(
                f"reduce_network_size: dropped {len(drop_ts)} timeseries keys "
                f"from {key}_t: {drop_ts}"
            )

    return n


def must_run_requirement_mw(
    n: pypsa.Network,
    sns: pd.Index,
    gens: pd.Index,
    *,
    derate_availability: bool = True,
) -> pd.DataFrame:
    """
    Compute the availability-derated must-run requirement in MW.

    Mirrors the requirement enforced by ``add_must_run_constraints``: the
    ``must_run_capacity`` timeseries where present (ratio already
    included), otherwise ``must-run capacity`` times the weighted average
    must-run ratio, both derated by the rolling-minimum availability used
    there to avoid conflicts with ramp constraints.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    sns : pd.Index
        The snapshots to be processed.
    gens : pd.Index
        Generators to evaluate. Generators without a must-run requirement
        return 0.0.
    derate_availability : bool, default True
        Derate the requirement by the rolling-minimum availability, as
        ``add_must_run_constraints`` does. Pass ``False`` for the requirement
        at full availability, which is the worst case the reserve-headroom
        patches have to leave room for.

    Returns
    -------
    pd.DataFrame
        Must-run requirement in MW, indexed by ``sns`` with ``gens`` as columns.
    """
    if (
        derate_availability
        and "availability" in n.generators_t
        and not n.generators_t.availability.empty
    ):
        availability = n.generators_t.availability.reindex(
            columns=gens, fill_value=1.0
        ).reindex(sns, fill_value=1.0)
    else:
        availability = pd.DataFrame(1.0, index=sns, columns=gens)
    min_availability = availability.rolling(window=6, min_periods=1).min()

    if "weighted average must-run ratio" in n.generators.columns:
        weighted_ratios = n.generators.loc[
            gens, "weighted average must-run ratio"
        ].fillna(1.0)
    else:
        weighted_ratios = pd.Series(1.0, index=gens)

    must_run_base = pd.DataFrame(0.0, index=sns, columns=gens)

    has_timeseries = pd.Series(False, index=gens)
    if "must_run_capacity" in n.generators_t:
        timeseries_cols = n.generators_t.must_run_capacity.columns
        for gen in gens:
            if (
                gen in timeseries_cols
                and not n.generators_t.must_run_capacity[gen].isna().all()
            ):
                has_timeseries[gen] = True

    timeseries_gens = gens[has_timeseries]
    if len(timeseries_gens) > 0:
        must_run_base[timeseries_gens] = n.generators_t.must_run_capacity[
            timeseries_gens
        ].reindex(sns, fill_value=0.0)

    static_gens = gens[~has_timeseries]
    if len(static_gens) > 0 and "must-run capacity" in n.generators.columns:
        static_values = (
            n.generators.loc[static_gens, "must-run capacity"].fillna(0.0)
            * weighted_ratios.loc[static_gens]
        )
        for gen in static_gens:
            must_run_base[gen] = static_values[gen]

    return must_run_base * min_availability


def as_dense(
    n: pypsa.Network, component: str, attr: str, inds: pd.Index
) -> pd.DataFrame:
    """
    Return ``n.get_switchable_as_dense`` with columns in the order of ``inds``.

    PyPSA returns columns in component order and silently drops names that
    are not in the network (PyPSA/PyPSA#1977).

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    component : str
        Component name, e.g. ``"Generator"``.
    attr : str
        Attribute name, e.g. ``"p_max_pu"``.
    inds : pd.Index
        Component names to select.

    Returns
    -------
    pd.DataFrame
        Snapshots x ``inds`` values.

    Raises
    ------
    KeyError
        If a name in ``inds`` is not in the network.
    """
    return n.get_switchable_as_dense(component, attr, inds=inds)[inds]
