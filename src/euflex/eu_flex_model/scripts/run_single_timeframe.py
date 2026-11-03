# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System

import logging
import pathlib

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
    run_market_model,
    sanitize_dual_names,
    write_model_to_lp_file,
)
from euflex.eu_flex_model.scripts.paths import get_path_config
from euflex.eu_flex_model.scripts.plan_maintenance import (
    apply_maintenance_schedules,
)

COMPRESSION = {"zlib": True, "complevel": 4}

if __name__ == "__main__":
    # Configure PyPSA logging level
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="run_single_timeframe",
    )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Retrieves the input network from snakemake (using actual timeseries)
    network = pypsa.Network(
        snakemake.input.network  # ty: ignore[possibly-missing-attribute]
    )

    run_name = snakemake.config["run"]["name"]
    path_config = get_path_config()
    run_dir = pathlib.Path(path_config.working_dir, run_name)

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
        / f"{snakemake.rule}_{target_year}_{climate_year}_actual_{iteration}_infeasible"
    )

    prepare_solver_instrumentation(network, snakemake, run_dir / "benchmarks")

    # Read maintenance plans and add to network
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

        # Disable PyPSA built-in ramping constraints by setting infinite ramp limits
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

    if enable_forced_outages:
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

    # Run single timeframe market model using rolling horizon
    # This combines all market features:
    # - Reserves (like day-ahead)
    # - Unit commitment (like day-ahead)
    # - Forced outages (like intraday)
    # - Maintenance planning (like both)
    # Using actual timeseries (no forecasts)
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
        is_day_ahead=True,  # Use day-ahead-like settings to get reserves and UC
        status_log_csv=getattr(
            snakemake.output, "market_rh_status_csv", pathlib.Path()
        ),
    )

    network_integer = pypsa.Network()
    if isinstance(res, pypsa.Network):
        # Export an empty integer network if model is linearized
        network = res
    elif isinstance(res, tuple):
        network, network_integer = res

    # Export the integer network from first step (not produced by without-ramping variant)
    market_network_integer = getattr(snakemake.output, "market_network_integer", None)
    if market_network_integer:
        network_integer.export_to_netcdf(
            market_network_integer, compression=COMPRESSION
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

    # Export the resulting network
    network.export_to_netcdf(
        snakemake.output.market_network,  # ty: ignore[possibly-missing-attribute]
        compression=COMPRESSION,
    )

    # Export dispatch data (not produced by without-ramping variant)
    dispatch_output = getattr(snakemake.output, "dispatch", None)
    if dispatch_output:
        network.statistics.supply(bus_carrier="AC", nice_names=False).to_csv(
            dispatch_output
        )
