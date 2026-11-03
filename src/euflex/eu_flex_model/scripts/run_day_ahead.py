# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
import pathlib
from pathlib import Path

import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.extensions import (
    prepare_solver_instrumentation,
)
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
    set_scenario_config,
)
from euflex.eu_flex_model.scripts.market_functions import (
    run_market_model,
    sanitize_dual_names,
    write_model_to_lp_file,
)
from euflex.eu_flex_model.scripts.paths import get_path_config
from euflex.eu_flex_model.scripts.plan_maintenance import (
    apply_maintenance_schedules,
)

if __name__ == "__main__":
    # Configure PyPSA logging level
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="run_day_ahead",
    )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Retrieves the input network from snakemake
    network = pypsa.Network(
        snakemake.input.network  # ty: ignore[possibly-missing-attribute]
    )
    run_name = snakemake.config["run"]["name"]
    path_config = get_path_config()
    run_dir = Path(path_config.working_dir, run_name)

    # Always set infeasibility LP stem so it is available if the solve fails.
    # market_functions appends the rolling-horizon index and ".lp" to this stem.
    wc = snakemake.wildcards
    target_year = wc.ty  # ty: ignore[possibly-missing-attribute]
    climate_year = wc.cy  # ty: ignore[possibly-missing-attribute]
    iteration = wc.it  # ty: ignore[possibly-missing-attribute]
    infeasibility_lp_dir = run_dir / "resources" / "market_models"
    infeasibility_lp_dir.mkdir(parents=True, exist_ok=True)
    network.meta["infeasibility_lp_file"] = str(
        infeasibility_lp_dir
        / f"{snakemake.rule}_{target_year}_{climate_year}_forecast_{iteration}_infeasible"
    )

    prepare_solver_instrumentation(network, snakemake, run_dir / "benchmarks")

    # Load maintenance plan
    maintenance_plans_csv = (
        snakemake.input.maintenance_plans  # ty: ignore[possibly-missing-attribute]
    )

    maintenance_plan = pd.read_csv(
        maintenance_plans_csv,
        index_col=0,
        parse_dates=True,
    )

    network.generators_t["maintenance_plan"] = maintenance_plan.reindex(
        network.snapshots
    )

    network = apply_maintenance_schedules(network, maintenance_plan)

    # Run day-ahead market using rolling horizon
    res = run_market_model(
        n=network,
        storage_plans_csv=pathlib.Path(
            snakemake.input.storage_plans  # ty: ignore[possibly-missing-attribute]
        ),
        market_interconnection=snakemake.params.market_interconnection,
        climate_year=snakemake.wildcards.cy,  # ty: ignore[possibly-missing-attribute]
        target_year=snakemake.wildcards.ty,  # ty: ignore[possibly-missing-attribute]
        config=snakemake.config,
        log=snakemake.log[0],
        is_day_ahead=True,
        status_log_csv=snakemake.output.day_ahead_rh_status_csv,  # ty: ignore[possibly-missing-attribute]
    )
    network_integer = pypsa.Network()
    if isinstance(res, pypsa.Network):
        # Export an empty integer network if model is linearized
        network = res
    elif isinstance(res, tuple):
        network, network_integer = res

    # Export the integer network of first step of day-ahead
    network_integer.export_to_netcdf(
        snakemake.output.day_ahead_network_integer,  # ty: ignore[possibly-missing-attribute]
        compression=snakemake.config["compression"],
    )

    # Write final rolling horizon model equations to folder
    write_lp = (
        snakemake.config["eva"]["export_lp_models"]
        and snakemake.wildcards.it == "0"  # ty: ignore[possibly-missing-attribute]
    )
    write_model_to_lp_file(
        network,
        snakemake.output.mathematical_model,  # ty: ignore[possibly-missing-attribute]
        write_lp,
    )

    # Sanitize dual variable names to expected format for downstream calculations
    network = sanitize_dual_names(network)

    # Update wildcards in network meta before export
    network.meta["wildcards"] = dict(snakemake.wildcards)

    # Export the resulting network of day-ahead
    network.export_to_netcdf(
        snakemake.output.day_ahead_network,  # ty: ignore[possibly-missing-attribute]
        compression=snakemake.config["compression"],
    )
