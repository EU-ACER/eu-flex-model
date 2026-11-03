# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System


import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.exceptions import (
    InfeasibilityError,
    maybe_print_solver_infeasibilities,
)
from euflex.eu_flex_model.scripts.extensions import (
    prepare_solver_instrumentation,
    run_optimization,
)
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
from euflex.eu_flex_model.scripts.helpers.solver_helpers import (
    with_gurobi_exports,
)
from euflex.eu_flex_model.scripts.market_functions import (
    add_explicit_dsr_constraints,
    set_all_generators_not_committable,
    turn_off_extendable,
    update_technical_constraints,
    write_model_to_lp_file,
)
from euflex.eu_flex_model.scripts.paths import get_path_config

# Patch is applied automatically when helpers package is imported via helpers/__init__.py

logging.getLogger("pypsa").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

"""
Pre-optimization of storage dispatch based on a simplified network model.

This script loads a base PyPSA network, applies scenario parameters and solving options,
solves the network, extracts dispatch and state of charge for hydro storage units,
and saves the results to CSV and NetCDF.

Used in the `plan_storage` rule within the Snakemake workflow.

Parameters
----------
snakemake.input.network : str
    Path to the preprocessed base network NetCDF file to use for pre-optimization.

snakemake.output.storage_plans : str
    Output path for the CSV file containing state of charge time series for hydro storage units.

snakemake.output.dispatch : str
    Output path for the CSV file containing dispatch time series for hydro storage units.

snakemake.output.network : str
    Output path for the resulting optimized PyPSA network in NetCDF format.

snakemake.params.solving : dict
    Solving configuration dictionary including optimization options.

snakemake.config : dict
    Full configuration loaded by Snakemake from YAML files.

Notes
-----
- The script filters to hydro carriers only for SOC and discharge.
- This module relies on PyPSA and custom utility functions defined in the project.
"""


def extra_functionality(n: pypsa.Network, sns: pd.Index) -> None:
    """
    Define additional functionality to be executed during the optimization process.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    sns : pd.Index
        The snapshots to be processed.
    """
    add_storage = n.meta["eva"]["storage_planning"]["storage_constraints_enabled"]
    if add_storage:
        logger.info(
            "Applying technical constraints in storage planning optimization..."
        )
        update_technical_constraints(n, sns)
    logger.info("Applying explicit DSR constraints in storage planning optimization...")
    add_explicit_dsr_constraints(n, sns)


def run_storage_planning(
    network: pypsa.Network,
    config: dict,
    solving: dict,
    log: str,
    preopt_targeted_carriers: list[str],
    use_cyclic_soc: bool,
    solver_export_stem: str = "storage_planning",
):
    """
    Runs pre-optimization for hydro storage dispatch based on simplified solving.

    Parameters
    ----------
    network : pypsa.Network
        Base network to be solved.

    config : dict
        Global configuration dictionary from Snakemake.

    solving : dict
        Dictionary with solving options and foresight information.
    log : str
        Path to the log file where the optimization process outputs are written.
        Primarily used for debugging and record-keeping.
    preopt_targeted_carriers : list[str]
        Storage unit carriers treated as hydro storage for SOC/dispatch extraction,
        from `eva.storage_planning.preopt_targeted_carriers`.
    use_cyclic_soc : bool
        Whether cyclic state of charge (SOC start == SOC end) is enforced for the
        targeted carriers during this pre-optimization, from
        `eva.storage_planning.use_cyclic_soc`.

    Returns
    -------
    network : pypsa.Network
        Solved network object.

    soc_df : pandas.DataFrame
        Time series of state of charge (SOC) for hydro storage units.

    dispatch_df : pandas.DataFrame
        Time series of dispatch power for hydro storage units.
    """

    logger.info("Preparing PyPSA network.")
    network, _ = turn_off_extendable(network)
    network = set_all_generators_not_committable(network, n_da=network)

    targeted_storage_units = network.storage_units.carrier.isin(
        preopt_targeted_carriers
    )
    network.storage_units.loc[targeted_storage_units, "cyclic_state_of_charge"] = (
        use_cyclic_soc
    )

    # Fix p_min_pu vs p_max_pu inconsistencies
    # Ensure p_min_pu <= p_max_pu for all generators at all snapshots
    logger.info("Ensuring p_min_pu <= p_max_pu consistency for generators...")
    all_gens = network.generators.index
    p_max_pu_ts = as_dense(network, "Generator", "p_max_pu", all_gens)
    p_min_pu_ts = as_dense(network, "Generator", "p_min_pu", all_gens)

    # Find violations where p_min_pu > p_max_pu
    violations = p_min_pu_ts > p_max_pu_ts
    n_violations = violations.sum().sum()

    if n_violations > 0:
        logger.warning(
            f"Found {n_violations} generator-snapshot combinations where p_min_pu > p_max_pu. "
            f"Clipping p_min_pu to p_max_pu to ensure feasibility."
        )
        p_min_pu_ts_fixed = np.minimum(p_min_pu_ts, p_max_pu_ts)
        network.generators_t.p_min_pu = p_min_pu_ts_fixed

        # Also update static values to match (take minimum across all snapshots)
        network.generators["p_min_pu"] = p_min_pu_ts_fixed.min()

    logger.info("Solving network for storage planning...")
    if config is None:
        config = {}
    if solving is None:
        solving = {}

    solver = solving["solver"]
    solver_name = solver["name"]
    option_key = solver["options"]
    solver_options = solving["solver_options"][option_key]

    if solver_name == "gurobi":
        solver_options = with_gurobi_exports(
            solver_name=solver_name,
            solver_options=solver_options,
            export_dir="resources/solver_exports",
            stem=solver_export_stem,
            export_log=config["eva"]["export_gurobi_logs"],
            export_model=config["eva"]["export_gurobi_mps"],
        )
    cf_solving = solving["options"]
    # Extract from config
    kwargs = {
        "solver_name": solver_name,
        "solver_options": solver_options,
        "linearized_unit_commitment": cf_solving["linearized_unit_commitment"],
        "model_kwargs": {"meshed_threshold": 10000},
    }

    model_kwargs = kwargs.pop("model_kwargs", {})
    linearized_unit_commitment = kwargs.pop("linearized_unit_commitment", False)

    network.optimize.create_model(
        linearized_unit_commitment=linearized_unit_commitment,
        **model_kwargs,
    )

    log_fn = log
    if solver_name == "gurobi" and config["eva"]["export_gurobi_logs"]:
        log_fn = None

    # Optional: Add config to network for access in extra_functionality
    setattr(network, "config", config)

    status, condition = run_optimization(
        optimize_callable=network.optimize.solve_model,
        optimize_kwargs=dict(
            extra_functionality=extra_functionality,
            log_fn=log_fn,
            **kwargs,
        ),
        network=network,
        module="plan_storage",
    )
    if status != "ok":
        maybe_print_solver_infeasibilities(
            getattr(network, "model", None), solver_name, logger
        )
        filename = Path(network.meta["infeasibility_lp_file"])
        logger.info("Writing .lp file for the failing model...")
        write_model_to_lp_file(network, filename.as_posix(), True, False)
        logger.info(f".lp file ready in {filename}")
        logger.error(
            "INFEASIBILITY DETECTED during storage planning (status=%s, condition=%s)",
            status,
            condition,
        )
        raise InfeasibilityError(
            message=(
                "Storage planning optimisation failed: "
                f"status={status}, condition={condition}."
            ),
            source="plan_storage.run_storage_planning",
            details={
                "status": status,
                "condition": str(condition),
                "solver_name": solver_name,
            },
        )

    hydro_storage_units = network.storage_units[
        network.storage_units.carrier.isin(preopt_targeted_carriers)
    ]

    soc_df = network.storage_units_t.state_of_charge[hydro_storage_units.index]
    dispatch_df = network.storage_units_t.p[hydro_storage_units.index]

    return network, soc_df, dispatch_df


def get_static_storage_planning(
    network: pypsa.Network,
    static_soc_target: float,
    preopt_targeted_carriers: list[str],
):
    """
    Create hydro storage dispatch based on static SOC target.

    Parameters
    ----------
    network : pypsa.Network
        Base network to be solved.
    static_soc_target : float
        Static SOC target (p.u.) applied to all hydro storage units.
    preopt_targeted_carriers : list[str]
        Storage unit carriers treated as hydro storage, from
        `eva.storage_planning.preopt_targeted_carriers`.

    Returns
    -------
    soc_df : pandas.DataFrame
        Time series of state of charge (SOC) for hydro storage units.
    """
    hydro_storage_units = network.storage_units[
        network.storage_units.carrier.isin(preopt_targeted_carriers)
    ]

    max_hours = hydro_storage_units.max_hours.fillna(8760)
    soc_target = hydro_storage_units.p_nom * max_hours * static_soc_target
    soc_df = pd.DataFrame(
        [soc_target.values] * len(network.snapshots),
        index=network.snapshots,
        columns=hydro_storage_units.index,
    )

    logger.info(
        f"Storage planning disabled, set SOC value static to {static_soc_target} p.u."
    )
    return soc_df


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="plan_storage",
    )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    solve_opts = snakemake.params.solving

    n = pypsa.Network(snakemake.input.network)  # ty: ignore[possibly-missing-attribute]
    storage_planning_cfg = snakemake.config["eva"]["storage_planning"]

    run_name = snakemake.config["run"]["name"]
    path_config = get_path_config()
    run_dir = Path(path_config.working_dir, run_name)

    # Always set infeasibility LP path so it is available if the solve fails
    wc = snakemake.wildcards
    target_year = wc.ty  # ty: ignore[possibly-missing-attribute]
    climate_year = wc.cy  # ty: ignore[possibly-missing-attribute]
    iteration = wc.it  # ty: ignore[possibly-missing-attribute]
    infeasibility_lp_dir = run_dir / "resources" / "market_models"
    infeasibility_lp_dir.mkdir(parents=True, exist_ok=True)
    n.meta["infeasibility_lp_file"] = str(
        infeasibility_lp_dir
        / f"{snakemake.rule}_{target_year}_{climate_year}_forecast_{iteration}_infeasible.lp"
    )

    prepare_solver_instrumentation(n, snakemake, run_dir / "benchmarks")

    # Sync live config into n.meta so extra_functionality respects CLI overrides
    # (e.g. --no-storage-constraints) that are applied after the network was built.
    n.meta.setdefault("eva", {}).setdefault("storage_planning", {})
    n.meta["eva"]["storage_planning"]["storage_constraints_enabled"] = (
        storage_planning_cfg["storage_constraints_enabled"]
    )

    preopt_targeted_carriers = storage_planning_cfg["preopt_targeted_carriers"]

    if storage_planning_cfg["enable"]:
        solver_export_stem = (
            f"{snakemake.rule}_{target_year}_{climate_year}_forecast_{iteration}"
        )
        n, soc, dispatch = run_storage_planning(
            n,
            snakemake.config,
            solve_opts,
            log=snakemake.log[0],
            preopt_targeted_carriers=preopt_targeted_carriers,
            use_cyclic_soc=storage_planning_cfg["use_cyclic_soc"],
            solver_export_stem=solver_export_stem,
        )
    else:
        soc = get_static_storage_planning(
            n, storage_planning_cfg["static_soc_target"], preopt_targeted_carriers
        )

    # Load base network to get the full snapshot range (resampled network may be truncated)
    base_n = pypsa.Network(
        snakemake.input.base_network  # ty: ignore[possibly-missing-attribute]
    )

    # Build hourly index covering the full base network period
    # The resampled network has coarse resolution (e.g., 8h) and may be truncated,
    # but we need to cover the full base network period for the market models
    start = base_n.snapshots.min().floor("h")
    end = base_n.snapshots.max().ceil("h")
    hourly_index = pd.date_range(start=start, end=end, freq="h")

    # define output paths
    math_model_output_path = (
        snakemake.output.mathematical_model  # ty: ignore[possibly-missing-attribute]
    )
    storage_plans_output_path = (
        snakemake.output.storage_plans  # ty: ignore[possibly-missing-attribute]
    )

    # Reindex to hourly resolution and interpolate between coarse pre-optimization targets
    # Forward fill for any trailing hours beyond the pre-optimization period
    soc = soc.reindex(hourly_index).interpolate(method="time").ffill()

    # Save storage plan
    soc.to_csv(storage_plans_output_path)

    # Export LP file if requested
    export_preopt = snakemake.config["eva"]["export_lp_models"]
    if export_preopt and hasattr(n, "model"):
        logger.info("Exporting pre-optimization model to LP file...")
        try:
            write_model_to_lp_file(
                n,
                math_model_output_path,
                generate_lp=True,
            )
            logger.info(f"Pre-optimization model exported to {math_model_output_path}")
        except Exception as e:
            logger.warning(f"Could not export pre-optimization model: {e}")
            # Create empty file so Snakemake doesn't fail
            with open(
                math_model_output_path,
                "w",
            ) as f:
                f.write(f"! Error exporting LP file: {e}\n")
    else:
        # Create empty file so Snakemake doesn't fail when export is not requested
        with open(
            math_model_output_path,
            "w",
        ) as f:
            f.write("! LP export not requested (set eva.export_lp_models: true)\n")

    logger.info("Results saved successfully")
