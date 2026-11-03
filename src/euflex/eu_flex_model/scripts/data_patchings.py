# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
"""
Data Patchings - Temporary data fixes and workarounds for network data quality issues.

This module contains temporary patching functions that work around issues
in the network data. These functions should be removed once the underlying
data issues are fixed.

All functions in this module are marked with TODO comments indicating when
they can be removed.
"""

import logging

import numpy as np
import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.helpers.pypsa_helpers import (
    as_dense,
    must_run_requirement_mw,
)

logger = logging.getLogger(__name__)


def patch_reserve_providers(n: pypsa.Network) -> pypsa.Network:
    """
    Convert reserve provider columns from float to boolean/integer.

    PyPSA may store reserve_FCR and reserve_FRR columns as floats (1.0, 0.0),
    but pandas index boolean selection requires actual boolean or integer types.
    This function converts these columns to bool type for all component tables.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to patch

    Returns
    -------
    pypsa.Network
        The patched network
    """
    reserve_cols = ["reserve_FCR", "reserve_FRR"]

    # Patch generators
    for col in reserve_cols:
        if col in n.generators.columns:
            n.generators[col] = n.generators[col].astype(bool)
            logger.debug(f"Converted generators.{col} to bool")

    # Patch storage units
    for col in reserve_cols:
        if col in n.storage_units.columns:
            n.storage_units[col] = n.storage_units[col].astype(bool)
            logger.debug(f"Converted storage_units.{col} to bool")

    return n


def patch_p_min_pu_inconsistencies(n: pypsa.Network) -> pypsa.Network:
    """
    Make sure p_min_pu < p_max_pu in all time steps. Note that this is a temporary fix, this should really be addressed in DB or network creation.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to patch

    Returns
    -------
    pypsa.Network
        The patched network

    """
    # Fix p_min_pu vs p_max_pu inconsistencies before export
    # Ensure p_min_pu <= p_max_pu for all generators at all snapshots
    logger.info("Ensuring p_min_pu <= p_max_pu consistency for generators...")
    all_gens = n.generators.index
    p_max_pu_ts = as_dense(n, "Generator", "p_max_pu", all_gens)
    p_min_pu_ts = as_dense(n, "Generator", "p_min_pu", all_gens)

    violations = p_min_pu_ts > p_max_pu_ts
    n_violations = violations.sum().sum()

    if n_violations > 0:
        logger.warning(
            f"Found {n_violations} generator-snapshot combinations where p_min_pu > p_max_pu. "
            f"Clipping p_min_pu to p_max_pu to ensure feasibility."
        )
        # Clip p_min_pu to p_max_pu element-wise
        p_min_pu_ts_fixed = np.minimum(p_min_pu_ts, p_max_pu_ts)
        n.generators_t.p_min_pu = p_min_pu_ts_fixed
        n.generators["p_min_pu"] = p_min_pu_ts_fixed.min()

    return n


# TODO: the below patching function should be removed, the correction should be done in the DB.
#  Validate dataframes should also catch this problem before running
def patch_p_set_p_min_max_inconsistencies(n: pypsa.Network) -> pypsa.Network:
    """
    Fix inconsistencies in p_set timeseries where values are outside p_min_pu and p_max_pu limits.

    This function checks the p_set timeseries for generators against their
    corresponding p_min_pu and p_max_pu limits. If any p_set values are found to be
    below p_min_pu * p_nom or above p_max_pu * p_nom, they are clipped to ensure feasibility.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to fix

    Returns
    -------
    pypsa.Network
        The fixed network
    """
    if "p_set" not in n.generators_t or n.generators_t.p_set.empty:
        return n

    p_set_df = n.generators_t.p_set
    gens = p_set_df.columns

    p_nom = n.generators.loc[gens, "p_nom"]
    p_min_pu = as_dense(n, "Generator", "p_min_pu", gens)
    p_max_pu = as_dense(n, "Generator", "p_max_pu", gens)

    lower = p_min_pu.multiply(p_nom)
    upper = p_max_pu.multiply(p_nom)

    p_set_clipped = p_set_df.clip(lower=lower, upper=upper)

    changed_mask = p_set_clipped != p_set_df
    if changed_mask.any().any():
        n_clipped = int(changed_mask.sum().sum())
        affected_gens = changed_mask.columns[changed_mask.any()].tolist()
        logger.warning(
            f"patch_p_set_p_min_max_inconsistencies: clipped {n_clipped} p_set values "
            f"for {len(affected_gens)} generator(s) to be within [p_min_pu * p_nom, "
            f"p_max_pu * p_nom] limits. Affected generators: {affected_gens}"
        )
        n.generators_t.p_set = p_set_clipped

    return n


def patch_expansion_limit_inconsistencies(n: pypsa.Network) -> int:
    """
    Fix generators and storage units where p_nom_max < p_nom_min.

    This function identifies extendable components where the maximum expansion
    limit is smaller than the minimum, which leads to infeasibility. It fixes
    this by setting p_nom_max = p_nom_min for affected components.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to fix

    Returns
    -------
    int
        Number of components fixed
    """
    fixed_count = 0

    # Fix generators
    extendable_gens = n.generators.index[n.generators.p_nom_extendable]
    if not extendable_gens.empty:
        # Find generators where p_nom_max < p_nom_min
        inconsistent_mask = (
            n.generators.loc[extendable_gens, "p_nom_max"]
            < n.generators.loc[extendable_gens, "p_nom_min"]
        )
        inconsistent_gens = extendable_gens[inconsistent_mask]

        if len(inconsistent_gens) > 0:
            logger.warning(  # TODO: Use a data
                f"Found {len(inconsistent_gens)} generators with p_nom_max < p_nom_min. "
                f"Setting p_nom_max = p_nom_min for these generators."
            )
            n.generators.loc[inconsistent_gens, "p_nom_max"] = n.generators.loc[
                inconsistent_gens, "p_nom_min"
            ]
            fixed_count += len(inconsistent_gens)

            # Log first few examples
            for gen in inconsistent_gens[:5]:
                logger.debug(
                    f"  Fixed {gen}: p_nom_max set to {n.generators.loc[gen, 'p_nom_max']:.2f} MW"
                )
            if len(inconsistent_gens) > 5:
                logger.debug(f"  ... and {len(inconsistent_gens) - 5} more")

    # Fix storage units
    extendable_storage = n.storage_units.index[n.storage_units.p_nom_extendable]
    if not extendable_storage.empty:
        # Find storage units where p_nom_max < p_nom_min
        inconsistent_mask = (
            n.storage_units.loc[extendable_storage, "p_nom_max"]
            < n.storage_units.loc[extendable_storage, "p_nom_min"]
        )
        inconsistent_storage = extendable_storage[inconsistent_mask]

        if len(inconsistent_storage) > 0:
            logger.warning(
                f"Found {len(inconsistent_storage)} storage units with p_nom_max < p_nom_min. "
                f"Setting p_nom_max = p_nom_min for these storage units."
            )
            n.storage_units.loc[inconsistent_storage, "p_nom_max"] = (
                n.storage_units.loc[inconsistent_storage, "p_nom_min"]
            )
            fixed_count += len(inconsistent_storage)

            # Log first few examples
            for storage in inconsistent_storage[:5]:
                logger.debug(
                    f"  Fixed {storage}: p_nom_max set to {n.storage_units.loc[storage, 'p_nom_max']:.2f} MW"
                )
            if len(inconsistent_storage) > 5:
                logger.debug(f"  ... and {len(inconsistent_storage) - 5} more")

    return fixed_count


def patch_reserve_amounts_exceed_pu_bounds(n: pypsa.Network) -> pypsa.Network:
    """
    Scale down reserve amounts where combined up+down derating would violate
    p_max_pu / p_min_pu bounds.

    For each generator/storage unit, computes the total per-unit up-direction
    reserve (sum of all *_up_amount columns) / p_nom and down-direction reserve
    (sum of all *_down_amount columns) / p_nom. When their sum exceeds the
    minimum available headroom (min over snapshots of p_max_pu - p_min_pu),
    both directions are scaled down proportionally so the combined derating
    fits within the headroom. A warning is logged for affected units.

    The condition checked is:
        up_pu + down_pu <= min(p_max_pu - p_min_pu)
    which guarantees that after derating new_max >= new_min at every snapshot.

    This function should be called at network build time so that
    apply_derated_reserves in market_functions.py does not need to
    defensively clip the resulting p_max_pu / p_min_pu values.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to patch.

    Returns
    -------
    pypsa.Network
        The patched network.
    """
    for component_name, c in [
        ("generators", "Generator"),
        ("storage_units", "StorageUnit"),
    ]:
        static = n.static(c)
        if static.empty:
            continue

        up_present = [col for col in static.columns if col.endswith("_up_amount")]
        down_present = [col for col in static.columns if col.endswith("_down_amount")]
        if not up_present and not down_present:
            continue

        p_nom = static["p_nom"]
        min_headroom_pu = (
            n.get_switchable_as_dense(c, "p_max_pu")
            - n.get_switchable_as_dense(c, "p_min_pu")
        ).min(axis=0)

        zero = pd.Series(0.0, index=static.index)
        total_up_pu = (
            sum((static[col].fillna(0.0) for col in up_present), zero)
            .div(p_nom)
            .fillna(0.0)
        )
        total_down_pu = (
            sum((static[col].fillna(0.0) for col in down_present), zero)
            .div(p_nom)
            .fillna(0.0)
        )
        combined_pu = total_up_pu + total_down_pu

        violations = combined_pu > min_headroom_pu
        if not violations.any():
            continue

        violating_units = violations[violations].index.tolist()
        logger.warning(
            f"patch_reserve_amounts_exceed_pu_bounds: {len(violating_units)} {component_name} "
            f"have combined reserve amounts (up + down) exceeding available headroom "
            f"(min p_max_pu - p_min_pu). Scaling down to feasible values. "
            f"Affected units: {violating_units}"
        )

        scale = pd.Series(1.0, index=static.index)
        scale[violations] = min_headroom_pu[violations] / combined_pu[violations]
        for col in up_present + down_present:
            n.static(c)[col] = static[col].fillna(0.0).mul(scale)

    return n


def patch_reserve_amounts_exceed_must_run_headroom(n: pypsa.Network) -> pypsa.Network:
    """
    Rebalance upward reserve amounts that exceed a generator's headroom above
    its must-run requirement.

    In derated mode every ``*_up_amount`` is subtracted from ``p_max_pu``, so
    an up amount larger than ``p_nom * p_max_pu`` minus the must-run
    requirement makes the must-run constraint infeasible. Prebuilt networks
    assigned amounts proportionally to ``p_nom`` without knowledge of
    must-run, so this patch re-caps each violating generator at its headroom
    and redistributes the excess to the other generators at the same bus that
    provide the same reserve type, proportionally to their remaining
    headroom. Bus totals per reserve type are preserved unless the whole
    group runs out of headroom, in which case a warning reports the
    shortfall.

    Amount columns are processed in sorted order and amounts from previously
    processed columns count against the headroom, mirroring
    ``assign_derated_reserve_amounts_to_units``.

    TODO: remove once all prebuilt networks are rebuilt with the
    headroom-aware assign_derated_reserve_amounts_to_units.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to patch.

    Returns
    -------
    pypsa.Network
        The patched network.
    """
    gens = n.generators
    up_cols = sorted(col for col in gens.columns if col.endswith("_up_amount"))
    if gens.empty or not up_cols:
        return n

    tol = 1e-6
    capacity = gens["p_nom"] * gens["p_max_pu"]
    must_run = must_run_requirement_mw(
        n, n.snapshots, gens.index, derate_availability=False
    ).max()
    # Redistribution must also respect the pu-bounds headroom that
    # patch_reserve_amounts_exceed_pu_bounds enforces afterwards (combined
    # up+down within min over snapshots of p_max_pu - p_min_pu), otherwise
    # redistributed amounts would immediately be scaled back down there.
    down_cols = [col for col in gens.columns if col.endswith("_down_amount")]
    total_down = gens[down_cols].fillna(0.0).sum(axis=1) if down_cols else 0.0
    pu_bounds_headroom = (
        n.get_switchable_as_dense("Generator", "p_max_pu")
        - n.get_switchable_as_dense("Generator", "p_min_pu")
    ).min(axis=0) * gens["p_nom"] - total_down
    assigned = pd.Series(0.0, index=gens.index)

    for col in up_cols:
        amounts = gens[col].fillna(0.0)
        headroom = (
            (capacity - must_run).clip(upper=pu_bounds_headroom) - assigned
        ).clip(lower=0.0)

        for bus, group in amounts[amounts > 0].groupby(gens["bus"]):
            group_headroom = headroom[group.index]
            if not (group > group_headroom + tol).any():
                assigned[group.index] += group
                continue

            total = group.sum()
            total_headroom = group_headroom.sum()
            if total > total_headroom + tol:
                logger.warning(
                    f"patch_reserve_amounts_exceed_must_run_headroom: {col} total "
                    f"of {total:.1f} MW at bus {bus!r} exceeds the generators' "
                    f"headroom above must-run of {total_headroom:.1f} MW. "
                    f"Assigning only the headroom "
                    f"({total - total_headroom:.1f} MW shortfall)."
                )
                rebalanced = group_headroom
            else:
                logger.info(
                    f"patch_reserve_amounts_exceed_must_run_headroom: rebalancing "
                    f"{col} at bus {bus!r} — "
                    f"{group.index[group > group_headroom + tol].tolist()} exceed "
                    f"their headroom above must-run; redistributing within the bus."
                )
                rebalanced = group_headroom / total_headroom * total

            n.generators.loc[group.index, col] = rebalanced
            assigned[group.index] += rebalanced

    return n
