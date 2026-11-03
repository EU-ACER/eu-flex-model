# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
# (Shared setup — imports, SCENARIO, helpers — lives in solve_eva_shared.smk)


def base_network(wcs):
    """Determine the input network for the build_base_network rule."""
    current_it = int(wcs.it)

    if current_it == 0:
        # Check if restart / warm-start mode is enabled
        if config["eva"]["restart_from"]["enable"]:
            return restart_base_network(wcs, ft="actual")

        # Check if FMC mode is enabled
        if config["eva"].get("fmc_only", {}).get("enable", False):
            # In FMC mode, use the final iteration networks from the parent EVA run
            parent_run = config["eva"]["fmc_only"]["parent_run"]

            # Get the parent run path (passed from CLI)
            parent_run_path = Path(config["eva"]["fmc_only"].get("parent_run_path"))

            if not parent_run_path.exists():
                raise ValueError(
                    f"FMC mode: Parent EVA run path does not exist: {parent_run_path}"
                )
            try:
                last_networks = find_iteration_networks(parent_run_path)
            except (FileNotFoundError, ValueError) as e:
                raise ValueError(
                    f"FMC mode: Failed to find final iteration networks from parent run {parent_run}: {e}"
                )

            # Get the network for the current (ty, cy) combination
            key = f"{wcs.ty}_{wcs.cy}"
            if key not in last_networks:
                raise ValueError(
                    f"FMC mode: No final iteration network found for TY={wcs.ty}, CY={wcs.cy} "
                    f"in parent run {parent_run}. Available: {list(last_networks.keys())}"
                )

            network_variants = last_networks[key]
            if "actual" not in network_variants:
                raise ValueError(
                    f"FMC mode: No actual network found for TY={wcs.ty}, CY={wcs.cy}"
                )

            return str(network_variants["actual"])

        # In sensitivity mode, the network is a folder of CSVs that
        # build_base_network reads instead of a pre-built snapshot
        if config["eva"]["sensitivity"]:
            return sensitivity_network(wcs)

        # A wrapping workflow may build the base network itself instead of
        # reading a pre-built snapshot; the platform builds it from the
        # database. See BASE_NETWORK_PROVIDER in setup.smk.
        if BASE_NETWORK_PROVIDER is not None:
            provided = BASE_NETWORK_PROVIDER(wcs)
            if provided:
                return provided

        run = get_parent_run(config["run"]["name"], config)
        path = config["eva"]["build_network"]["workflow_prebuilt"]["path"].format(
            run=run, cy=wcs.cy, ty=wcs.ty
        )

        input_dir = path_config.input_dir

        if input_dir:
            return Path(input_dir) / path

        # Fallback for development mode
        return find_project_root() / path
    else:
        prev_wcs = dict(wcs)
        prev_wcs["it"] = current_it - 1
        return resources("networks/base_network_{ty}_{cy}_actual_{it}.nc").format(
            **prev_wcs
        )


# Function def storage_plan(wcs) lives in solve_eva_shared.smk
# Function def maintenance_plan(wcs) lives in solve_eva_shared.smk
# Function def previous_unit_states(wcs) lives in solve_eva_shared.smk
# Function def unit_states_csv_input(wcs) lives in solve_eva_shared.smk


rule cleanup_previous_runs:
    message:
        "Cleaning files from previous run"
    output:
        touch(resources("cleanup_previous_runs.flag")),
    benchmark:
        "benchmarks/cleanup/cleanup_previous_runs.tsv"
    run:
        run_name = config["run"]["name"]
        print(f"Run name from config: {run_name}", file=sys.stderr)

        patterns = [
            "resources/networks/base_network_*.nc",
            "resources/networks/market_network_*.nc",
            "resources/networks/resampled_network_*.nc",
            "resources/npv/npv_*.csv",
            "resources/convergence/convergence_*.txt",
            "resources/capacities/capacity_*.csv",
            "resources/dispatch/dispatch_*.csv",
            "resources/cleanup/cleanup_iteration_*.touch",
        ]

        for pattern in patterns:
            print(f"Checking pattern: {pattern}", file=sys.stderr)
            for file in pathlib.Path().glob(pattern):
                print(f"Deleting {file}", file=sys.stderr)
                file.unlink()


rule build_base_network:
    params:
        scale_network=config["eva"]["build_network"]["scale_network"],
    input:
        base_network=base_network,
        unit_states_csv=unit_states_csv_input,
    output:
        network=resources("networks/base_network_{ty}_{cy}_actual_{it}.nc"),
    benchmark:
        "benchmarks/network_building/build_base_network_{ty}_{cy}_actual_{it}.tsv"
    log:
        logs("build_base_network_{ty}_{cy}_actual_{it}.log"),
    script:
        "../scripts/build_base_network.py"


rule resample_network:
    params:
        time_resolution=config["eva"]["storage_planning"]["po_resolution"],
    input:
        network=resources("networks/base_network_{ty}_{cy}_actual_{it}.nc"),
    output:
        network=resources("networks/resampled_network_{ty}_{cy}_actual_{it}.nc"),
    benchmark:
        "benchmarks/planning/resample_network_{ty}_{cy}_actual_{it}.tsv"
    log:
        logs("resample_network_{ty}_{cy}_actual_{it}.log"),
    script:
        "../scripts/resample_network.py"


rule plan_storage:
    params:
        solving=config_provider("solving"),
    input:
        network=resources("networks/resampled_network_{ty}_{cy}_actual_{it}.nc"),
        base_network=resources("networks/base_network_{ty}_{cy}_actual_{it}.nc"),
    output:
        storage_plans=resources("pre_optimization/storage_plan_{ty}_{cy}_{it}.csv"),
        mathematical_model=resources(
            "pre_optimization/preopt_model_{ty}_{cy}_forecast_{it}.lp"
        ),
    benchmark:
        "benchmarks/planning/plan_storage_{ty}_{cy}_{it}.tsv"
    log:
        logs("plan_storage_{ty}_{cy}_{it}.log"),
    script:
        "../scripts/plan_storage.py"


rule run_single_timeframe:
    params:
        market_interconnection=config_provider("eva", "market_interconnection"),
    input:
        network=resources("networks/base_network_{ty}_{cy}_actual_{it}.nc"),
        storage_plans=storage_plan,
        maintenance_plans=maintenance_plan,
        outage_scenarios=resources("outages/outage_scenario_{ty}_{fo}.csv"),
        link_availability=resources("outages/link_availability_{ty}_{fo}.csv"),
    output:
        market_network=resources(
            "networks/market_network_before_cap_update_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
        dispatch=resources("dispatch/dispatch_{ty}_{cy}_actual_{fo}_{it}.csv"),
        mathematical_model=resources(
            "market_models/market_model_{ty}_{cy}_actual_{fo}_{it}.lp"
        ),
        market_network_integer=resources(
            "networks/market_network_integer_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
        market_rh_status_csv=resources(
            "market_models/market_model_status_{ty}_{cy}_actual_{fo}_{it}.csv"
        ),
    benchmark:
        "benchmarks/market/run_single_timeframe_{ty}_{cy}_actual_{fo}_{it}.tsv"
    log:
        logs("run_single_timeframe_{ty}_{cy}_actual_{fo}_{it}.log"),
    script:
        "../scripts/run_single_timeframe.py"


rule run_single_timeframe_without_ramping:
    """
    Run single-timeframe dispatch without ramping constraints for ramping shortfall indicator.

    This rule runs the same market dispatch as run_single_timeframe but with ramping
    constraints disabled. The ramping shortfall indicator is calculated by comparing
    curtailment and ENS between the regular run (with ramping) and this run (without ramping).

    Only runs if ramping shortfall calculation is enabled in config (eva.ramping_shortfall.enable).
    Only runs once per scenario (ty, cy, fo) at the final iteration (after convergence).
    """
    params:
        market_interconnection=config_provider("eva", "market_interconnection"),
        disable_ramping=True,  # Flag to disable ramping constraints
    input:
        network=resources("networks/base_network_{ty}_{cy}_actual_{it}.nc"),
        storage_plans=storage_plan,
        maintenance_plans=maintenance_plan,
        convergence_check=resources("convergence/convergence_{it}.txt"),
        outage_scenarios=resources("outages/outage_scenario_{ty}_{fo}.csv"),
        link_availability=resources("outages/link_availability_{ty}_{fo}.csv"),
    output:
        market_network=resources(
            "networks/market_network_without_ramping_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
        mathematical_model=resources(
            "market_models/market_model_without_ramping_{ty}_{cy}_actual_{fo}_{it}.lp"
        ),
        market_rh_status_csv=resources(
            "market_models/market_model_without_ramping_status_{ty}_{cy}_actual_{fo}_{it}.csv"
        ),
    benchmark:
        "benchmarks/outputs/run_single_timeframe_without_ramping_{ty}_{cy}_actual_{fo}_{it}.tsv"
    log:
        logs("run_single_timeframe_without_ramping_{ty}_{cy}_actual_{fo}_{it}.log"),
    script:
        "../scripts/run_single_timeframe.py"


rule check_infeasibilities:
    params:
        infeasibility_threshold=config_provider("eva", "infeasibility_threshold"),
        fail_on_infeasibility=config_provider("eva", "fail_on_infeasibility"),
    input:
        market_status=resources(
            "market_models/market_model_status_{ty}_{cy}_actual_{fo}_{it}.csv"
        ),
    output:
        report=resources("market_models/infeasibility_report_{ty}_{cy}_{fo}_{it}.yaml"),
    benchmark:
        "benchmarks/validation/check_infeasibilities_{ty}_{cy}_{fo}_{it}.tsv"
    log:
        logs("check_infeasibilities_{ty}_{cy}_{fo}_{it}.log"),
    script:
        "../scripts/check_infeasibilities.py"


rule update_prices:
    input:
        market_networks_before_update=expand(
            resources(
                "networks/market_network_before_cap_update_{ty}_{cy}_actual_{fo}_{{it}}.nc"
            ),
            **SCENARIO,
        ),
    output:
        market_networks=expand(
            resources("networks/market_network_{ty}_{cy}_actual_{fo}_{{it}}.nc"),
            **SCENARIO,
        ),
        marginal_price_comparison_market=expand(
            resources("plots/marginal_price_comparison_market_{cy}_{{it}}.png"),
            **SCENARIO,
        ),
        market_price_cap_timeseries=expand(
            resources("price_caps/market_price_cap_{ty}_{cy}_actual_{fo}_{{it}}.csv"),
            **SCENARIO,
        ),
        price_cap_evolution_market=resources(
            "plots/price_cap_evolution_market_{it}.png"
        ),
    benchmark:
        "benchmarks/capacity_expansion/update_prices_{it}.tsv"
    log:
        logs("update_prices_{it}.log"),
    script:
        "../scripts/update_price_cap.py"


rule cleanup_before_cap_update:
    """Stub the _before_cap_update networks immediately after update_prices completes."""
    input:
        price_cap_evolution_market=resources(
            "plots/price_cap_evolution_market_{it}.png"
        ),
    output:
        touch(resources("cleanup/cleanup_before_cap_update_{it}.touch")),
    benchmark:
        "benchmarks/cleanup/cleanup_before_cap_update_{it}.tsv"
    run:
        if config["run"]["keep_files"]:
            return
        for f in pathlib.Path().glob(
            f"resources/networks/market_network_before_cap_update_*_{wildcards.it}.nc"
        ):
            mtime = f.stat().st_mtime
            f.write_bytes(b"")
            os.utime(f, (mtime, mtime))


rule compute_npv:
    input:
        market_networks=expand(
            resources("networks/market_network_{ty}_{cy}_actual_{fo}_{{it}}.nc"),
            **SCENARIO,
        ),
        market_price_cap_timeseries=expand(
            resources("price_caps/market_price_cap_{ty}_{cy}_actual_{fo}_{{it}}.csv"),
            **SCENARIO,
        ),
        infeasibility_reports=expand(
            resources("market_models/infeasibility_report_{ty}_{cy}_{fo}_{{it}}.yaml"),
            **SCENARIO,
        ),
    output:
        npv=resources("npv/npv_{it}.csv"),
        npv_computation_log=resources("npv/npv_computation_log_{it}.csv"),
    benchmark:
        "benchmarks/capacity_expansion/compute_npv_{it}.tsv"
    log:
        logs("compute_npv_{it}.log"),
    script:
        "../scripts/compute_npv.py"


checkpoint update_capacity:
    input:
        networks_actual=expand(
            resources("networks/base_network_{ty}_{cy}_actual_{{it}}.nc"),
            **SCENARIO_NO_FO,
        ),
        npv=resources("npv/npv_{it}.csv"),
        previous_unit_states=previous_unit_states,
    output:
        capacities=resources("capacities/capacity_{it}.csv"),
        unit_states=resources("capacities/unit_states_{it}.csv"),
        convergence_check=resources("convergence/convergence_{it}.txt"),
    benchmark:
        "benchmarks/capacity_expansion/update_capacity_{it}.tsv"
    log:
        logs("update_capacity_{it}.log"),
    script:
        "../scripts/update_capacity.py"


rule cleanup_iteration:
    """Delete intermediate network files from the previous iteration once the current one completes."""
    input:
        convergence_check=resources("convergence/convergence_{it}.txt"),
    output:
        touch(resources("cleanup/cleanup_iteration_{it}.touch")),
    benchmark:
        "benchmarks/cleanup/cleanup_iteration_{it}.tsv"
    run:
        prev_it = int(wildcards.it) - 1
        if prev_it < 0:
            return

        prev_convergence = pathlib.Path(
            f"resources/convergence/convergence_{prev_it}.txt"
        )
        prev_is_final = (
            prev_convergence.exists()
            and prev_convergence.read_text().strip() != "Running"
        )

        if config["run"]["keep_files"]:
            return

        delete_patterns = [
            f"resources/networks/resampled_network_*_{prev_it}.nc",
            f"resources/networks/market_network_before_cap_update_*_{prev_it}.nc",
            f"resources/networks/market_network_integer_*_{prev_it}.nc",
        ]
        # base_network_*_0.nc is needed by calculate_model_outputs for baseline capacity
        if prev_it > 0:
            delete_patterns.append(f"resources/networks/base_network_*_{prev_it}.nc")
        if not prev_is_final:
            delete_patterns += [
                f"resources/networks/market_network_*_{prev_it}.nc",
            ]

        for pattern in delete_patterns:
            for f in pathlib.Path().glob(pattern):
                f.unlink(missing_ok=True)


rule plot_process_validation:
    params:
        solving=config_provider("solving"),
    input:
        storage_plans=storage_plan,
        market_network=resources(
            "networks/market_network_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
        market_network_integer=resources(
            "networks/market_network_integer_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
        outage_scenarios=resources("outages/outage_scenario_{ty}_{fo}.csv"),
        link_availability=resources("outages/link_availability_{ty}_{fo}.csv"),
    output:
        precondition_market=resources(
            "plots/precondition_market_{ty}_{cy}_{fo}_{it}.png"
        ),
        network_flows_market=resources(
            "plots/network_flows_market_{ty}_{cy}_{fo}_{it}.png"
        ),
        network_flows_market_hour_12=resources(
            "plots/network_flows_market_one_timestamp_{ty}_{cy}_{fo}_{it}.png"
        ),
        dispatch_validation_integer_vs_linear=(
            resources(
                "plots/dispatch_validation_integer_vs_linear_{ty}_{cy}_{fo}_{it}.png"
            )
            if not config["solving"]["options"]["linearized_unit_commitment"]
            else []
        ),
        reserves_validation_market=resources(
            "plots/reserves_validation_market_{ty}_{cy}_{fo}_{it}.png"
        ),
        soc_validation_market=resources(
            "plots/soc_validation_market_{ty}_{cy}_{fo}_{it}.png"
        ),
        soc_cumsum_validation_market=resources(
            "plots/soc_cumsum_validation_market_{ty}_{cy}_{fo}_{it}.png"
        ),
        reserve_revenue_validation_market=resources(
            "plots/reserve_revenue_validation_market_{ty}_{cy}_{fo}_{it}.png"
        ),
        hydro_constraints_validation_market=resources(
            "plots/hydro_constraints_validation_market_{ty}_{cy}_{fo}_{it}.png"
        ),
        generator_dispatch_and_availability=resources(
            "plots/generator_dispatch_and_availability_{ty}_{cy}_{fo}_{it}.png"
        ),
        outage_scenarios_heatmap=resources(
            "plots/outage_scenarios_heatmap_{ty}_{cy}_{fo}_{it}.png"
        ),
        generator_availability_profile=resources(
            "plots/generator_availability_profile_{ty}_{cy}_{fo}_{it}.png"
        ),
        link_availability_profile=resources(
            "plots/link_availability_profile_{ty}_{cy}_{fo}_{it}.png"
        ),
        flag=resources("plots/process_validation_{ty}_{cy}_{fo}_{it}.touch"),
        dsr_validation_market_implicit=resources(
            "plots/dsr_validation_market_implicit_{ty}_{cy}_{fo}_{it}.png"
        ),
        dsr_validation_market_explicit=resources(
            "plots/dsr_validation_market_explicit_{ty}_{cy}_{fo}_{it}.png"
        ),
        must_run_dispatch_validation_market=resources(
            "plots/must_run_dispatch_market_{ty}_{cy}_{fo}_{it}.png"
        ),
    threads: 1
    resources:
        mem_mb=20_000,
    benchmark:
        "benchmarks/validation/plot_process_validation_{ty}_{cy}_{fo}_{it}.tsv"
    log:
        logs("plot_process_validation_{ty}_{cy}_{fo}_{it}.log"),
    script:
        "../scripts/plot_process_validation.py"


rule run_integration_tests:
    input:
        storage_plans=storage_plan,
        market_network=resources(
            "networks/market_network_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
    output:
        test_results_json=resources(
            "integration_tests/test_results_{ty}_{cy}_{fo}_{it}.json"
        ),
        test_report_txt=resources(
            "integration_tests/test_report_{ty}_{cy}_{fo}_{it}.txt"
        ),
    benchmark:
        "benchmarks/validation/run_integration_tests_{ty}_{cy}_{fo}_{it}.tsv"
    log:
        logs("run_integration_tests_{ty}_{cy}_{fo}_{it}.log"),
    script:
        "../scripts/run_integration_tests.py"


# _EVA_TRIES and def eva_input(wildcards) live in solve_eva_shared.smk
# Function def process_validation_plots(wildcards) lives in solve_eva_shared.smk
# Function def integration_test_outputs(wildcards) lives in solve_eva_shared.smk
# Function def iteration_cleanup_flags(wildcards) lives in solve_eva_shared.smk


def ramping_shortfall_networks(wildcards):
    """
    Return no-ramping market networks if ramping shortfall calculation is enabled.

    Only runs for the final/converged iteration by searching backwards from _EVA_TRIES
    to find the first iteration with a non-"Running" status (Converged, Aborted, or No EVA).
    Uses SCENARIO to include forced outage scenarios.
    """
    if not config["eva"]["ramping_shortfall"]["enable"]:
        return []

    # Find the final iteration by examining convergence files
    check = checkpoints.update_capacity
    run_name = config["run"]["name"]

    # Start from _EVA_TRIES and work backwards to find the final iteration
    final_it = None
    for it in range(_EVA_TRIES, -1, -1):
        try:
            fn = check.get(it=it, run=run_name).output.convergence_check
            if os.path.exists(fn):
                with open(fn) as f:
                    status = f.read().strip()
                # Found a non-Running iteration
                if status != "Running":
                    final_it = it
                    break
        except:
            continue

    # If we found a final iteration, request its outputs
    if final_it is not None:
        return expand(
            resources(
                "networks/market_network_without_ramping_{ty}_{cy}_actual_{fo}_{it}.nc"
            ),
            **SCENARIO,
            it=final_it,
        )
    else:
        return []


def final_market_networks(wildcards):
    return expand(
        resources("networks/market_network_{ty}_{cy}_actual_{fo}_{it}.nc"),
        **SCENARIO,
        it=_EVA_TRIES,
    )


rule finalize_networks:
    input:
        convergence_check=eva_input,
        process_validation_plots=process_validation_plots,
        integration_test_reports=integration_test_outputs,
        market_networks_without_ramping=ramping_shortfall_networks,
        iteration_cleanup_flags=iteration_cleanup_flags,
        market_networks=final_market_networks,
    output:
        market_networks=expand(
            RESULTS + "networks/market_network_{ty}_{cy}.nc", **SCENARIO_NO_FO
        ),
        market_networks_without_ramping=(
            expand(
                RESULTS + "networks/market_network_without_ramping_{ty}_{cy}.nc",
                **SCENARIO_NO_FO,
            )
            if config["eva"]["ramping_shortfall"]["enable"]
            else []
        ),
    benchmark:
        "benchmarks/outputs/finalize_networks.tsv"
    log:
        logs("finalize_networks.log"),
    script:
        "../scripts/finalize_networks.py"
