# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System


import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.extensions import (
    prepare_solver_instrumentation,
)
from euflex.eu_flex_model.scripts.generate_forced_outages import (
    align_scenario_to_snapshots,
    apply_forced_outages,
    apply_forced_outages_to_links,
)
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
    set_scenario_config,
)
from euflex.eu_flex_model.scripts.market_functions import (
    prepare_intraday_network,
    run_market_model,
    write_model_to_lp_file,
)
from euflex.eu_flex_model.scripts.paths import get_path_config
from euflex.eu_flex_model.scripts.plan_maintenance import (
    apply_maintenance_schedules,
)

if __name__ == "__main__":
    snakemake = get_snakemake(globals_scope=globals(), rule="run_intraday")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Retrieves the input network from snakemake
    network = pypsa.Network(
        snakemake.input.network  # ty: ignore[possibly-missing-attribute]
    )
    network_day_ahead = pypsa.Network(
        snakemake.input.day_ahead_network  # ty: ignore[possibly-missing-attribute]
    )

    tol_ramp = snakemake.config["eva"]["patching_tol_ramp"]
    id_free_commitment_carriers = snakemake.config["eva"]["base_network"][
        "id_free_commitment_carriers"
    ]
    network = prepare_intraday_network(
        network,
        network_day_ahead,
        tol_ramp=tol_ramp,
        id_free_commitment_carriers=id_free_commitment_carriers,
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

    maintenance_plans_csv = (
        snakemake.input.maintenance_plans  # ty: ignore[possibly-missing-attribute]
    )
    maintenance_plan = pd.read_csv(maintenance_plans_csv, index_col=0, parse_dates=True)
    network.generators_t["maintenance_plan"] = maintenance_plan.reindex(
        network.snapshots
    )

    network = apply_maintenance_schedules(network, maintenance_plan)

    # Check if ramping constraints should be disabled (for ramping shortfall calculation)
    disable_ramping = getattr(snakemake.params, "disable_ramping", False)

    if disable_ramping:
        logging.info(
            "Disabling ramping constraints for ramping shortfall calculation..."
        )
        # Set flag for extra_functionality to skip reserve ramping constraints
        network.meta["disable_ramping_for_shortfall"] = True

        # Disable PyPSA built-in ramping constraints by setting NAn ramp limits
        # Set both static and time-varying ramp limits to ensure all ramping is disabled
        network.generators["ramp_limit_up"] = np.nan
        network.generators["ramp_limit_down"] = np.nan

        # Override time-varying ramp limits created by align_ramp_limits()
        # PyPSA prioritizes time-varying limits over static ones
        if "ramp_limit_up" in network.generators_t:
            network.generators_t["ramp_limit_up"][:] = np.nan
        if "ramp_limit_down" in network.generators_t:
            network.generators_t["ramp_limit_down"][:] = np.nan

        logging.info(
            "Set all ramp limits (static and time-varying) to infinity to disable ramping constraints"
        )

    # Load forced outage scenarios if enabled
    enable_forced_outages = snakemake.config["eva"]["forced_outages"]["enable"]
    # Check if outage inputs are available (not available for run_intraday_without_ramping)
    has_outage_inputs = hasattr(snakemake.input, "outage_scenarios") and hasattr(
        snakemake.input, "link_availability"
    )

    if enable_forced_outages and has_outage_inputs:
        # Load pre-generated outage scenarios from CSV
        outage_scenarios = pd.read_csv(
            snakemake.input.outage_scenarios,  # ty: ignore[possibly-missing-attribute]
            index_col=0,
            parse_dates=True,
        )
        outage_scenarios = align_scenario_to_snapshots(
            outage_scenarios, network, columns=network.generators.index
        )
        # Apply outage scenarios to network generators
        network, final_availability = apply_forced_outages(network, outage_scenarios)

        # Load pre-generated link availability from CSV
        link_availability = pd.read_csv(
            snakemake.input.link_availability,  # ty: ignore[possibly-missing-attribute]
            index_col=0,
            parse_dates=True,
        )
        link_availability = align_scenario_to_snapshots(
            link_availability, network, columns=network.links.index
        )
        # Apply link availability to network links
        network, final_link_availability = apply_forced_outages_to_links(
            network, link_availability
        )
    else:
        # No forced outages - all generators and links always available
        outage_scenarios = pd.DataFrame(
            1.0, index=network.snapshots, columns=network.generators.index
        )
        link_availability = pd.DataFrame(
            1.0, index=network.snapshots, columns=network.links.index
        )

    # Run intra-day market using rolling horizon
    res = run_market_model(
        n=network,
        storage_plans_csv=snakemake.input.storage_plans,  # ty: ignore[possibly-missing-attribute]
        market_interconnection=snakemake.params.market_interconnection,
        climate_year=snakemake.wildcards.cy,  # ty: ignore[possibly-missing-attribute]
        target_year=snakemake.wildcards.ty,  # ty: ignore[possibly-missing-attribute]
        config=snakemake.config,
        log=snakemake.log[0],
        status_log_csv=snakemake.output.intraday_rh_status_csv,  # ty: ignore[possibly-missing-attribute]
    )

    network_integer = pypsa.Network()
    if isinstance(res, pypsa.Network):
        # Export an empty integer network if model is linearized
        network = res
    elif isinstance(res, tuple):
        network, network_integer = res

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

    # Update wildcards in network meta before export to preserve all wildcards (like fo)
    network.meta["wildcards"] = dict(snakemake.wildcards)

    # Export the output network the intra-day
    network.export_to_netcdf(
        snakemake.output.intraday_network,  # ty: ignore[possibly-missing-attribute]
        compression=snakemake.config["compression"],
    )
    network.statistics.supply(
        comps=["Generator", "StorageUnit"], groupby=["carrier", "bus"]
    ).to_csv(snakemake.output.dispatch)  # ty: ignore[possibly-missing-attribute]
