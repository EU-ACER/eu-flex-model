# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
import pathlib
import sys

from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_parent_run,
)
from pathlib import Path
from euflex.eu_flex_model.scripts.run_discovery import (
    find_iteration_networks,
    find_restart_networks,
)
import os
import pandas as pd
from euflex.eu_flex_model.scripts.paths import (
    get_path_config,
    find_project_root,
    resolve_input_path,
)


# Generate forced outage sample identifiers based on config
FORCED_OUTAGES_ENABLED = config["eva"]["forced_outages"]["enable"]
if FORCED_OUTAGES_ENABLED:
    n_samples = config["eva"]["forced_outages"]["samples"]
    FORCED_OUTAGE_SAMPLES = [f"fo{i+1}" for i in range(n_samples)]
else:
    FORCED_OUTAGE_SAMPLES = ["fo1"]  # Default single sample when disabled

# Add forced outage samples to scenario configuration
config["scenario"]["fo"] = FORCED_OUTAGE_SAMPLES

# SCENARIO with fo wildcard (for intraday/actual networks)
SCENARIO = dict(**config["scenario"], run="{run}")

# SCENARIO without fo wildcard (for day-ahead/forecast networks)
SCENARIO_NO_FO = {k: v for k, v in config["scenario"].items() if k != "fo"}
SCENARIO_NO_FO["run"] = "{run}"

path_config = get_path_config()


def sensitivity_network(wcs) -> str:
    """Folder of network CSVs that sensitivity mode builds the base network from.

    ``eva.build_network.sensitivity_path`` names it, with ``{run}``, ``{ty}``
    and ``{cy}`` placeholders; a relative path is looked up as other inputs
    are (see ``resolve_input_path``). Left empty, it is the folder that the
    ``prepare_sensitivity`` export of ``run.parent`` writes to.
    """
    run = config["run"]["parent"]
    template = config["eva"]["build_network"]["sensitivity_path"]
    # sensitivity_path is only ever read, so it may point anywhere, like
    # workflow_prebuilt.path; anything that writes or deletes must use the
    # confined default folder below
    if template:
        path = resolve_input_path(template.format(run=run, ty=wcs.ty, cy=wcs.cy))
    else:
        base = Path(path_config.sensitivity_dir).resolve()
        path = Path(base, f"base_{run}", f"ty_{wcs.ty}_cy_{wcs.cy}")
        # Security: run.parent must not lead out of the sensitivity directory
        if not path.resolve().is_relative_to(base):
            raise ValueError(
                f"Invalid run.parent: resolved path {path.resolve()} escapes "
                f"base directory {base}"
            )
    return path.resolve().as_posix()


def restart_base_network(wcs, ft: str) -> str:
    """
    Return the parent run network that seeds iteration 0 in restart mode.

    The CLI already resolved and validated the parent run, so anything failing
    here means the parent run changed on disk after validation.

    Parameters
    ----------
    wcs
        Snakemake wildcards, providing ty and cy.
    ft : str
        Forecast type to fetch. Perfect-foresight runs only use "actual".

    Returns
    -------
    str
        Path to the parent run's network for this scenario and forecast type.
    """
    restart_cfg = config["eva"]["restart_from"]
    parent_run = restart_cfg["parent_run"]
    parent_run_path = Path(restart_cfg["parent_run_path"])

    if not parent_run_path.exists():
        raise ValueError(
            f"Restart mode: parent run path does not exist: {parent_run_path}"
        )

    if config["eva"]["perfect_foresight"]["enable"]:
        required_fts = ("actual",)
    else:
        required_fts = ("actual", "forecast")

    try:
        restart_networks = find_restart_networks(parent_run_path, required_fts)
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(
            f"Restart mode: failed to find restart networks in parent run "
            f"{parent_run}: {e}"
        )

    key = f"{wcs.ty}_{wcs.cy}"
    if key not in restart_networks:
        raise ValueError(
            f"Restart mode: no restart network for TY={wcs.ty}, CY={wcs.cy} in parent "
            f"run {parent_run}. Available: {sorted(restart_networks)}"
        )

    iteration, variants = restart_networks[key]
    if ft not in variants:
        raise ValueError(
            f"Restart mode: no '{ft}' network at iteration {iteration} for "
            f"TY={wcs.ty}, CY={wcs.cy} in parent run {parent_run}"
        )

    return str(variants[ft])


def storage_plan(wcs) -> str:
    """
    Determines the storage plan input file based on frequency.

    Runs plan_storage for iteration 0 and every N iterations thereafter,
    where N is defined in the config file (eva.storage_planning.storage_plan_frequency).
    Otherwise, uses the storage plan from the last iteration where
    plan_storage was run.

    Note: Storage plan is always based on the network used by the active mode
    and shared between all market timeframes.
    """
    current_it = int(wcs.it)
    N = config["eva"]["storage_planning"]["storage_plan_frequency"]

    target_it = (
        current_it
        if current_it == 0 or current_it % N == 0
        else current_it - (current_it % N)
    )

    return resources(f"pre_optimization/storage_plan_{wcs.ty}_{wcs.cy}_{target_it}.csv")


def final_iteration_storage_plan(wcs) -> str:
    """
    Determines the storage plan input file for the final (converged) EVA iteration.

    Same rounding convention as storage_plan() above, but anchored on
    _EVA_TRIES (the last iteration the EVA loop actually ran) instead of a
    per-iteration {it} wildcard. Used by rules that run once after the EVA
    loop finishes - e.g. calculate_model_outputs - and therefore have no {it}
    wildcard of their own to plug into storage_plan().
    """
    N = config["eva"]["storage_planning"]["storage_plan_frequency"]
    # 0 % N == 0 for any N, so this already covers _EVA_TRIES == 0 without
    # a separate branch.
    target_it = _EVA_TRIES - (_EVA_TRIES % N)
    return resources(f"pre_optimization/storage_plan_{wcs.ty}_{wcs.cy}_{target_it}.csv")


def maintenance_plan(wcs):
    """
    Determines the maintenance plan input file for precondition_network.

    Runs plan_maintenance for iteration 0 and every N iterations thereafter,
    where N is defined in the config file (eva.maintenance_plan_frequency).
    Otherwise, uses the maintenance plan from the last iteration where
    plan_maintenance was run.
    """
    current_it = int(wcs.it)
    N = config["eva"]["maintenance_plan_frequency"]

    target_it = (
        current_it
        if current_it == 0 or current_it % N == 0
        else current_it - (current_it % N)
    )

    return resources(
        f"pre_optimization/maintenance_plan_{wcs.ty}_actual_{target_it}.csv"
    )


def unit_states_csv_input(wcs):
    """Return unit_states CSV for it>0, empty list for it=0 (no updates needed)."""
    if int(wcs.it) == 0:
        return []
    return resources(f"capacities/unit_states_{int(wcs.it)-1}.csv")


def previous_unit_states(wcs):
    """
    Returns the previous iteration's unit_states file for state persistence.

    For iteration 0, returns an empty list (no previous state).
    For iteration > 0, returns the unit_states file from iteration - 1.
    """
    current_it = int(wcs.it)

    if current_it == 0:
        return []

    return resources(f"capacities/unit_states_{current_it-1}.csv")


_EVA_TRIES = 0


def eva_input(wildcards):
    if not isinstance(config["run"]["name"], str):
        raise TypeError(
            "The EVA workflow only support one run name at a time. "
            "The config run.name must be a string"
        )

    check = checkpoints.update_capacity
    global _EVA_TRIES

    fn = check.get(it=_EVA_TRIES, run=config["run"]["name"]).output.convergence_check
    if not os.path.exists(fn):
        _EVA_TRIES += 1
    else:
        with open(fn) as f:
            status = f.read().strip()
        if status in ("Running", "ClusteredConverged"):
            _EVA_TRIES += 1
    return check.get(it=_EVA_TRIES, run=config["run"]["name"]).output.convergence_check


def process_validation_plots(wildcards):
    N = config["eva"]["validation_plot_frequency"]
    current_it = _EVA_TRIES

    # Validation plots require day_ahead_network_integer which is only produced
    # by run_day_ahead (full-year solve). Skip at clustered iterations to avoid
    # pulling run_day_ahead back into the DAG unnecessarily.
    if N == 0 or not is_full_run(current_it):
        return []
    elif current_it % N == 1:
        return expand(
            resources("plots/process_validation_{ty}_{cy}_{fo}_{it}.touch"),
            **SCENARIO,
            it=_EVA_TRIES,
        )
    else:
        return []


def integration_test_outputs(wildcards):
    """Return integration test outputs if enabled in config."""
    if config.get("eva", {}).get("integration_tests", {}).get("enable", False):
        return expand(
            resources("integration_tests/test_report_{ty}_{cy}_{fo}_{it}.txt"),
            **SCENARIO,
            it=_EVA_TRIES,
        )
    else:
        return []


def iteration_cleanup_flags(wildcards):
    """Return cleanup touch files for all completed iterations up to _EVA_TRIES."""
    return expand(
        resources("cleanup/cleanup_iteration_{it}.touch"),
        it=range(_EVA_TRIES + 1),
        run=config["run"]["name"],
    ) + expand(
        resources("cleanup/cleanup_before_cap_update_{it}.touch"),
        it=range(_EVA_TRIES + 1),
        run=config["run"]["name"],
    )


def is_full_run(it):
    """Return True if this iteration runs the full set of weeks (not clustered).

    The mode is computed deterministically from the config (same logic as
    determine_clustering_mode) rather than reading clustering_mode_{it}.txt,
    because that file is produced AFTER compute_npv(it) — the very rule that
    needs this answer — creating a circular dependency.

    Iterations that are always full runs:
    - it == 0 (first iteration)
    - it % freq == 0 (every freq-th iteration)
    - it == max_it - 1 (last iteration must be a full run)
    - the iteration immediately following a ClusteredConverged status (validation run)
    """
    if not config["eva"]["advanced_clustering"]["enable"]:
        return True
    it = int(it)
    freq = config["eva"]["advanced_clustering"].get("full_run_frequency", 1)
    max_it = config["eva"]["max_it"]
    if it == 0 or it % freq == 0 or it >= max_it - 1:
        return True
    # Force full run if the previous iteration wrote ClusteredConverged.
    #
    # Built through resources() rather than by hand: the path carries no run
    # name (RDIR is empty, and the run directory is chosen by whoever invokes
    # snakemake), so an interpolated one never existed and this check silently
    # never fired.
    if it > 0:
        prev_conv = pathlib.Path(resources(f"convergence/convergence_{it-1}.txt"))
        if prev_conv.exists() and prev_conv.read_text().strip() == "ClusteredConverged":
            return True
    return False


def day_ahead_network_inputs(wcs):
    """Return day-ahead network inputs for compute_npv.

    Full-run iterations: standard full-year day_ahead_network (post update_prices).
    Clustered iterations: k centroid DA networks directly (no reconstruction step).
    """
    ty_list = [wcs.ty] if hasattr(wcs, "ty") else config["scenario"]["ty"]

    if config["eva"]["advanced_clustering"]["enable"] and not is_full_run(int(wcs.it)):
        prev_it = int(wcs.it) - 1
        paths = []
        for ty in ty_list:
            cp = checkpoints.cluster_weeks.get(ty=ty, it=prev_it)
            df = pd.read_csv(cp.output.week_clusters)
            centroids = df[df["centroid_flag"]]
            for row in centroids.itertuples():
                paths.append(
                    resources(
                        f"networks/centroid_da_network_before_cap_update_{ty}_{row.cy}_forecast_{row.cluster_id}_{wcs.it}.nc"
                    )
                )
        return paths

    return expand(
        resources("networks/day_ahead_network_{ty}_{cy}_forecast_{it}.nc"),
        ty=ty_list,
        cy=config["scenario"]["cy"],
        it=wcs.it,
    )


def reconstruct_da_status_inputs(wcs):
    """Return centroid DA status CSV paths for reconstruct_da_status.

    Includes ALL centroid status files for the target year (across all climate years).
    This ensures that if any representative centroid is infeasible, it is caught
    for all climate years reconstructed from it, and prevents empty list issues
    when all centroids land in a different climate year.
    """
    if not config["eva"]["advanced_clustering"]["enable"]:
        return []
    if is_full_run(int(wcs.it)):
        return []
    ty = wcs.ty
    prev_it = int(wcs.it) - 1
    checkpoints.cluster_weeks.get(ty=ty, it=prev_it)
    csv_path = checkpoints.cluster_weeks.get(ty=ty, it=prev_it).output.week_clusters
    df = pd.read_csv(csv_path)
    centroids = df[df["centroid_flag"]]
    return [
        resources(
            f"market_models/centroid_da_model_status_{ty}_{row.cy}_forecast_{row.cluster_id}_{wcs.it}.csv"
        )
        for row in centroids.itertuples()
    ]


def check_infeasibilities_day_ahead_status_input(wcs):
    """Route day_ahead_status input: centroid statuses (clustered) or normal (full run)."""
    if not config["eva"]["advanced_clustering"]["enable"] or is_full_run(int(wcs.it)):
        return resources(
            "market_models/day_ahead_network_model_status_{ty}_{cy}_forecast_{it}.csv"
        ).format(**wcs)
    # Clustered: return all centroid DA status files for this (ty, cy)
    prev_it = int(wcs.it) - 1
    ty = wcs.ty
    cp = checkpoints.cluster_weeks.get(ty=ty, it=prev_it)
    df = pd.read_csv(cp.output.week_clusters)
    centroids = df[(df["centroid_flag"]) & (df["cy"] == wcs.cy)]
    return [
        resources(
            f"market_models/centroid_da_model_status_{ty}_{row.cy}_forecast_{row.cluster_id}_{wcs.it}.csv"
        )
        for row in centroids.itertuples()
    ]


def intraday_networks_for_npv(wcs):
    """Return intraday network inputs for compute_npv.

    Full-run iterations: standard full-year intraday_network (post update_prices).
    Clustered iterations: k centroid ID networks directly (no reconstruction step).
    """
    if config["eva"]["advanced_clustering"]["enable"] and not is_full_run(int(wcs.it)):
        prev_it = int(wcs.it) - 1
        fo_list = config["scenario"]["fo"]
        paths = []
        for ty in config["scenario"]["ty"]:
            cp = checkpoints.cluster_weeks.get(ty=ty, it=prev_it)
            df = pd.read_csv(cp.output.week_clusters)
            centroids = df[df["centroid_flag"]]
            for row in centroids.itertuples():
                for fo in fo_list:
                    paths.append(
                        resources(
                            f"networks/centroid_intraday_network_before_cap_update_{ty}_{row.cy}_actual_{fo}_{row.cluster_id}_{wcs.it}.nc"
                        )
                    )
        return paths

    return expand(
        resources("networks/intraday_network_{ty}_{cy}_actual_{fo}_{it}.nc"),
        **SCENARIO,
        it=wcs.it,
    )


def intraday_da_baseline_input(wcs):
    """Return the DA network used as baseline for run_intraday (full-run iterations only)."""
    return resources(
        "networks/day_ahead_network_before_cap_update_{ty}_{cy}_forecast_{it}.nc"
    ).format(**wcs)


def reconstruct_id_status_inputs(wcs):
    """Return centroid ID status CSV paths for reconstruct_id_status."""
    if not config["eva"]["advanced_clustering"]["enable"]:
        return []
    if is_full_run(int(wcs.it)):
        return []
    ty = wcs.ty
    prev_it = int(wcs.it) - 1
    checkpoints.cluster_weeks.get(ty=ty, it=prev_it)
    csv_path = checkpoints.cluster_weeks.get(ty=ty, it=prev_it).output.week_clusters
    df = pd.read_csv(csv_path)
    centroids = df[df["centroid_flag"]]
    return [
        resources(
            f"market_models/centroid_id_model_status_{ty}_{row.cy}_actual_{wcs.fo}_{row.cluster_id}_{wcs.it}.csv"
        )
        for row in centroids.itertuples()
    ]


def check_infeasibilities_intraday_status_input(wcs):
    """Route intraday_status input: centroid statuses (clustered) or normal (full run)."""
    if not config["eva"]["advanced_clustering"]["enable"] or is_full_run(int(wcs.it)):
        return resources(
            "market_models/intraday_model_status_{ty}_{cy}_actual_{fo}_{it}.csv"
        ).format(**wcs)
    # Clustered: return all centroid ID status files for this (ty, cy, fo)
    prev_it = int(wcs.it) - 1
    ty = wcs.ty
    cp = checkpoints.cluster_weeks.get(ty=ty, it=prev_it)
    df = pd.read_csv(cp.output.week_clusters)
    centroids = df[(df["centroid_flag"]) & (df["cy"] == wcs.cy)]
    return [
        resources(
            f"market_models/centroid_id_model_status_{ty}_{row.cy}_actual_{wcs.fo}_{row.cluster_id}_{wcs.it}.csv"
        )
        for row in centroids.itertuples()
    ]


def intraday_networks_before_update_input(wcs):
    """Route update_prices ID input: centroid ID networks (clustered) or normal (full run)."""
    ty_list = [wcs.ty] if hasattr(wcs, "ty") else config["scenario"]["ty"]
    fo_list = [wcs.fo] if hasattr(wcs, "fo") else config["scenario"]["fo"]
    if not config["eva"]["advanced_clustering"]["enable"] or is_full_run(int(wcs.it)):
        return expand(
            resources(
                "networks/intraday_network_before_cap_update_{ty}_{cy}_actual_{fo}_{it}.nc"
            ),
            ty=ty_list,
            cy=config["scenario"]["cy"],
            fo=fo_list,
            it=wcs.it,
        )
    # Clustered: return centroid ID networks directly (no reconstruction)
    prev_it = int(wcs.it) - 1
    paths = []
    for ty in ty_list:
        cp = checkpoints.cluster_weeks.get(ty=ty, it=prev_it)
        df = pd.read_csv(cp.output.week_clusters)
        centroids = df[df["centroid_flag"]]
        for row in centroids.itertuples():
            for fo in fo_list:
                paths.append(
                    resources(
                        f"networks/centroid_intraday_network_before_cap_update_{ty}_{row.cy}_actual_{fo}_{row.cluster_id}_{wcs.it}.nc"
                    )
                )
    return paths


def day_ahead_networks_before_update_input(wcs):
    """Route update_prices DA input: centroid DA networks (clustered) or normal (full run)."""
    ty_list = [wcs.ty] if hasattr(wcs, "ty") else config["scenario"]["ty"]
    if not config["eva"]["advanced_clustering"]["enable"] or is_full_run(int(wcs.it)):
        return expand(
            resources(
                "networks/day_ahead_network_before_cap_update_{ty}_{cy}_forecast_{it}.nc"
            ),
            ty=ty_list,
            cy=config["scenario"]["cy"],
            it=wcs.it,
        )
    # Clustered: return centroid DA networks directly (no reconstruction)
    prev_it = int(wcs.it) - 1
    paths = []
    for ty in ty_list:
        cp = checkpoints.cluster_weeks.get(ty=ty, it=prev_it)
        df = pd.read_csv(cp.output.week_clusters)
        centroids = df[df["centroid_flag"]]
        for row in centroids.itertuples():
            paths.append(
                resources(
                    f"networks/centroid_da_network_before_cap_update_{ty}_{row.cy}_forecast_{row.cluster_id}_{wcs.it}.nc"
                )
            )
    return paths


def update_prices_week_clusters_input(wcs):
    """Route update_prices week_clusters input: centroid mappings (clustered) or empty (full run)."""
    if not config["eva"]["advanced_clustering"]["enable"] or is_full_run(int(wcs.it)):
        return []
    prev_it = int(wcs.it) - 1
    ty_list = config["scenario"]["ty"]
    return [resources(f"clustering/week_clusters_{ty}_{prev_it}.csv") for ty in ty_list]


rule plan_maintenance:
    params:
        maintenance_preopt_res=config_provider("maintenance_preopt_res"),
        preopt_sa_tolerance=config_provider("preopt_sa_tolerance"),
    input:
        network=expand(
            resources("networks/base_network_{{ty}}_{cy}_actual_{{it}}.nc"),
            cy=config["scenario"]["cy"],
            run=config["run"]["name"],
        ),
    output:
        maintenance_plans=resources(
            "pre_optimization/maintenance_plan_{ty}_actual_{it}.csv"
        ),
    log:
        logs("plan_maintenance_{ty}_actual_{it}.log"),
    benchmark:
        "benchmarks/planning/plan_maintenance_{ty}_actual_{it}.tsv"
    script:
        "../scripts/plan_maintenance.py"


def generate_forced_outages_input(wildcards):
    """Reference network for forced-outage generation.

    Fleet composition (generator/link index, avg_forced_outage_rate, FOR-rate,
    poles) is invariant across climate years and across it>=1 for a fixed ty.
    Reuse the first configured climate year's it==0 network as the one-time
    reference fleet instead of depending on the caller's {cy}/{it}.
    """
    reference_cy = config["scenario"]["cy"][0]
    return resources(f"networks/base_network_{wildcards.ty}_{reference_cy}_actual_0.nc")


rule generate_forced_outages:
    input:
        network=generate_forced_outages_input,
    output:
        outage_scenarios=resources("outages/outage_scenario_{ty}_{fo}.csv"),
        link_availability=resources("outages/link_availability_{ty}_{fo}.csv"),
    benchmark:
        "benchmarks/planning/generate_forced_outages_{ty}_{fo}.tsv"
    log:
        logs("generate_forced_outages_{ty}_{fo}.log"),
    script:
        "../scripts/generate_forced_outages.py"
