# Check if dummy storage optimization is enabled
from snakemake.io import expand

from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    validate_curtailment_targets_per_bz,
)

DS_ENABLED = config["eva"]["dummy_storage_optimization"]["enable"]

STORAGE_RATIOS = (
    config["eva"]["dummy_storage_optimization"]["energy_power_ratios"]
    if DS_ENABLED
    else []
)

# Always use ct1, ct2, etc. format for curtailment targets (no underscore to avoid
# wildcard issues). The actual values are resolved in the Python script.
if DS_ENABLED:
    decimal_targets = config["eva"]["dummy_storage_optimization"]["curtailment_targets"]
    CURTAILMENT_TARGETS = [f"ct{i+1}" for i in range(len(decimal_targets))]
    # Store mapping for the Python script to use
    CURTAILMENT_TARGET_MAPPING = {
        f"ct{i+1}": val for i, val in enumerate(decimal_targets)
    }

    # Per-bidding-zone targets name rows of a CSV rather than giving a value
    # directly, so check here that every name resolves. Doing it while the
    # rules are being defined means a typo fails before any job starts,
    # rather than part-way through the assessment.
    if config["eva"]["dummy_storage_optimization"]["curtailment_targets_per_bz"]:
        validate_curtailment_targets_per_bz(decimal_targets)
else:
    CURTAILMENT_TARGETS = []

# Check if shadow prices extraction is enabled
SHADOW_PRICES_ENABLED = config["eva"]["shadow_prices"]["enable"]
# Constraint patterns to extract - passed to calculate_model_outputs rule params (see shadow_prices_constraints param)
SHADOW_PRICE_CONSTRAINTS = config["eva"]["shadow_prices"]["constraints"]

# Check if plot_model_outputs is enabled
PLOT_MODEL_OUTPUTS_ENABLED = config["plotting"]["enable"]


rule run_dummy_storage_optimization:
    """
    rule that runs a dummy storage optimization
    """
    params:
        shared_across_countries=config["eva"]["dummy_storage_optimization"][
            "shared_across_countries"
        ],
        minimize_system_costs=config["eva"]["dummy_storage_optimization"][
            "minimize_system_costs"
        ],
        export_models=config["eva"]["dummy_storage_optimization"]["export_models"],
        renewable_carriers=config["eva"]["base_network"]["renewable_carriers"],
        enable_load_spillage=config["eva"]["dummy_storage_optimization"][
            "enable_load_spillage"
        ],
        curtailment_targets_per_bz=config["eva"]["dummy_storage_optimization"][
            "curtailment_targets_per_bz"
        ],
        add_network_needs=config["eva"]["dummy_storage_optimization"]["network_needs"],
        conventional_floor_carriers=config["eva"]["dummy_storage_optimization"][
            "conventional_floor_carriers"
        ],
        market_interconnection=config["eva"]["market_interconnection"],
        eraa_year=config["eva"]["eraa_year"],
    input:
        network=(
            RESULTS + "networks/market_network_{ty}_{cy}.nc"
            if config["eva"]["perfect_foresight"]["enable"]
            else RESULTS + "networks/intraday_network_{ty}_{cy}.nc"
        ),
    output:
        network=RESULTS
        + "assessment/dummy_storage_optimization/network_{ty}_{cy}_{mh}_{ct}.nc",
        mathematical_model=RESULTS
        + "assessment/dummy_storage_optimization/model_{ty}_{cy}_{mh}_{ct}.lp",
    benchmark:
        "benchmarks/outputs/run_dummy_storage_optimization_{ty}_{cy}_{mh}_{ct}.tsv"
    log:
        python=logs("run_dummy_storage_optimization_{ty}_{cy}_{mh}_{ct}.log"),
        solver=logs("run_dummy_storage_optimization_{ty}_{cy}_{mh}_{ct}_solver.log"),
    script:
        "../scripts/run_dummy_storage_optimization.py"


rule calculate_model_outputs:
    """
    Calculate comprehensive model outputs including flexibility indicators, market outputs, and EVA outputs.

    This unified rule combines the functionality of calculate_flexibility_indicators,
    extract_market_outputs, and calculate_eva_outputs to create a single comprehensive
    dataset with all model outputs. Each output is tagged with a "kind" attribute to
    distinguish between "flexibility_indicator", "market_output", and "eva_output".

    Outputs include:
    - Flexibility indicators (production capacities, flexibility needs, etc.)
    - Market clearing prices per bidding zone
    - Total generation by technology per day per bidding zone
    - Total CO2 emissions per bidding zone
    - Net power transfer between bidding zones
    - Reserve capacity per generator
    - Reservoir storage levels from pre-optimization
    - EVA outputs (commissioned/decommissioned/mothballed capacity changes, NPV metrics)
    """
    params:
        storage_ratios=STORAGE_RATIOS,
        renewable_carriers=config["eva"]["base_network"]["renewable_carriers"],
        shadow_prices_enabled=SHADOW_PRICES_ENABLED,
        shadow_prices_constraints=SHADOW_PRICE_CONSTRAINTS,
    input:
        # In perfect foresight mode: use market networks for both
        # In standard EVA mode: use separate intraday and day-ahead networks
        intraday_network=(
            RESULTS + "networks/market_network_{ty}_{cy}.nc"
            if config["eva"]["perfect_foresight"]["enable"]
            else RESULTS + "networks/intraday_network_{ty}_{cy}.nc"
        ),
        day_ahead_network=(
            RESULTS + "networks/market_network_{ty}_{cy}.nc"
            if config["eva"]["perfect_foresight"]["enable"]
            else RESULTS + "networks/day_ahead_network_{ty}_{cy}.nc"
        ),
        intraday_network_without_ramping=(
            (
                expand(
                    RESULTS + "networks/market_network_without_ramping_{ty}_{cy}.nc",
                    ty=config["scenario"]["ty"],
                    cy=config["scenario"]["cy"],
                    run=config["run"]["name"],
                )
                if config["eva"]["ramping_shortfall"]["enable"]
                else []
            )
            if config["eva"]["perfect_foresight"]["enable"]
            else (
                expand(
                    RESULTS + "networks/intraday_network_without_ramping_{ty}_{cy}.nc",
                    ty=config["scenario"]["ty"],
                    cy=config["scenario"]["cy"],
                    run=config["run"]["name"],
                )
                if config["eva"]["ramping_shortfall"]["enable"]
                else []
            )
        ),
        pre_optimization_storage=lambda wildcards: expand(
            final_iteration_storage_plan(wildcards),
            run=config["run"]["name"],
        ),
        dummy_storage_optimizations=(
            expand(
                RESULTS
                + "assessment/dummy_storage_optimization/network_{ty}_{cy}_{mh}_{ct}.nc",
                ty=config["scenario"]["ty"],
                cy=config["scenario"]["cy"],
                mh=STORAGE_RATIOS,
                ct=CURTAILMENT_TARGETS,
                run=config["run"]["name"],
            )
            if DS_ENABLED
            else []
        ),
        capacity_csv_paths=lambda wildcards: expand(
            resources("capacities/capacity_{it}.csv"),
            run=config["run"]["name"],
            it=range(_EVA_TRIES + 1),
        ),
        unit_states_csv_paths=lambda wildcards: expand(
            resources("capacities/unit_states_{it}.csv"),
            run=config["run"]["name"],
            it=range(_EVA_TRIES + 1),
        ),
        base_network=lambda wildcards: resources(
            f"networks/base_network_{wildcards.ty}_{wildcards.cy}_actual_0.nc"
            if config["eva"]["perfect_foresight"]["enable"]
            else f"networks/base_network_{wildcards.ty}_{wildcards.cy}_forecast_0.nc"
        ),
        npv_data=resources(f"npv/npv_{_EVA_TRIES}.csv"),
    output:
        fn=RESULTS + "assessment/model_outputs_{ty}_{cy}.nc",
        shadow_prices=(
            RESULTS + "assessment/shadow_prices_{ty}_{cy}.nc"
            if SHADOW_PRICES_ENABLED
            else []
        ),
    benchmark:
        "benchmarks/outputs/calculate_model_outputs_{ty}_{cy}.tsv"
    log:
        logs("calculate_model_outputs_{ty}_{cy}.log"),
    script:
        "../scripts/calculate_model_outputs.py"


rule consolidate_model_outputs:
    """
    Consolidate comprehensive model outputs (flexibility indicators + market outputs) across all scenarios.
    """
    input:
        expand(
            RESULTS + "assessment/model_outputs_{ty}_{cy}.nc",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    output:
        fn=(RESULTS + "assessment/model_outputs.nc").format(run=config["run"]["name"]),
    benchmark:
        "benchmarks/outputs/consolidate_model_outputs.tsv"
    log:
        logs("consolidate_model_outputs.log").format(run=config["run"]["name"]),
    script:
        "../scripts/consolidate_flexibility_indicators.py"


if SHADOW_PRICES_ENABLED:

    rule consolidate_shadow_prices:
        """
        Consolidate shadow prices across all scenarios into a single file.
        """
        input:
            expand(
                RESULTS + "assessment/shadow_prices_{ty}_{cy}.nc",
                **config["scenario"],
                run=config["run"]["name"],
            ),
        output:
            fn=(RESULTS + "assessment/shadow_prices.nc").format(
                run=config["run"]["name"]
            ),
        benchmark:
            "benchmarks/outputs/consolidate_shadow_prices.tsv"
        log:
            logs("consolidate_shadow_prices.log").format(run=config["run"]["name"]),
        script:
            "../scripts/consolidate_flexibility_indicators.py"


if PLOT_MODEL_OUTPUTS_ENABLED:

    rule plot_model_outputs:
        """
        Generate summary figures and catalogue plots from the unified model outputs.
        """
        params:
            renewable_carriers=config["eva"]["base_network"]["renewable_carriers"],
        input:
            model_outputs=(RESULTS + "assessment/model_outputs.nc").format(
                run=config["run"]["name"]
            ),
            shadow_prices=(
                (RESULTS + "assessment/shadow_prices.nc").format(
                    run=config["run"]["name"]
                )
                if SHADOW_PRICES_ENABLED
                else []
            ),
            extra_targets=EXTRA_FINAL_TARGETS,
        output:
            base_case=(
                RESULTS + "assessment/plots/flexibility_summary_base_case.png"
            ).format(run=config["run"]["name"]),
            dummy_storage=(
                expand(
                    RESULTS
                    + "assessment/plots/flexibility_summary_dummy_storage_{mh}_{ct}.png",
                    mh=STORAGE_RATIOS,
                    ct=CURTAILMENT_TARGETS,
                    run=config["run"]["name"],
                )
                if DS_ENABLED
                else []
            ),
            comparison=(
                RESULTS + "assessment/plots/dummy_storage_capacity_vs_curtailment.png"
            ).format(run=config["run"]["name"]),
            catalogue_dir=directory(
                (RESULTS + "assessment/plots/catalogue").format(
                    run=config["run"]["name"]
                )
            ),
        benchmark:
            "benchmarks/outputs/plot_model_outputs.tsv"
        log:
            logs("plot_model_outputs.log").format(run=config["run"]["name"]),
        script:
            "../scripts/plot_model_outputs.py"

else:

    rule plot_model_outputs:
        """
        Placeholder rule when plotting is disabled. Outputs consolidated model outputs only.
        Any targets a wrapping workflow registered in EXTRA_FINAL_TARGETS are
        pulled in here too, so they are built before the run is considered done.
        """
        input:
            model_outputs=(RESULTS + "assessment/model_outputs.nc").format(
                run=config["run"]["name"]
            ),
            shadow_prices=(
                (RESULTS + "assessment/shadow_prices.nc").format(
                    run=config["run"]["name"]
                )
                if SHADOW_PRICES_ENABLED
                else []
            ),
            extra_targets=EXTRA_FINAL_TARGETS,
        output:
            touch(
                (RESULTS + "assessment/.plot_model_outputs_disabled").format(
                    run=config["run"]["name"]
                )
            ),
        log:
            logs("plot_model_outputs.log").format(run=config["run"]["name"]),
