# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
# (Shared setup — imports, SCENARIO, helpers — lives in solve_eva_shared.smk)


def forecast_timeseries_inputs():
    """Return the forecast timeseries inputs for the build_base_network rule.

    Forecast timeseries come from the workflow's own CSV inputs unless a
    wrapping workflow produces them itself; the platform writes them out of
    the database alongside the network. See FORECAST_TIMESERIES_INPUTS in
    setup.smk.
    """
    if FORECAST_TIMESERIES_INPUTS is not None:
        provided = FORECAST_TIMESERIES_INPUTS()
        if provided:
            return provided

    build_network = config["eva"]["build_network"]
    if build_network["forecast_timeseries_source"] == "csv":
        return {
            "forecast_p_max_pu": str(
                resolve_input_path(build_network["forecast_p_max_pu_path"])
            ),
            "forecast_p_set": str(
                resolve_input_path(build_network["forecast_p_set_path"])
            ),
        }

    return {}


def base_network(wcs):
    """Determine the input network for the build_base_network rule."""
    current_it = int(wcs.it)

    if current_it == 0:
        # Check if restart / warm-start mode is enabled
        if config["eva"]["restart_from"]["enable"]:
            return restart_base_network(wcs, ft=wcs.ft)

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
        return resources("networks/base_network_{ty}_{cy}_{ft}_{it}.nc").format(
            **prev_wcs
        )


# Function def storage_plan(wcs) lives in solve_eva_shared.smk
# Function def maintenance_plan(wcs) lives in solve_eva_shared.smk
# Function def previous_unit_states(wcs) lives in solve_eva_shared.smk
# Function def unit_states_csv_input(wcs) lives in solve_eva_shared.smk


def _prev_clustering_mode(wcs):
    """Return clustering_mode file for iteration it-1 when clustering is enabled and it > 0.

    This forces determine_clustering_mode (and therefore cluster_weeks and
    extract_chunk_revenues) to run before the next iteration starts.
    """
    if not config["eva"]["advanced_clustering"]["enable"]:
        return []
    it = int(wcs.it)
    if it == 0:
        return []
    return resources(f"clustering/clustering_mode_{it-1}.txt")


def _prev_convergence_check(wcs):
    """Return the convergence file for iteration it-1, or [] for it == 0.

    Used by determine_clustering_mode to detect a ClusteredConverged status
    from the previous iteration and force a validation full run.
    """
    it = int(wcs.it)
    if it == 0:
        return []
    return resources(f"convergence/convergence_{it-1}.txt")


rule build_base_network:
    params:
        scale_network=config["eva"]["build_network"]["scale_network"],
    input:
        **forecast_timeseries_inputs(),
        base_network=base_network,
        unit_states_csv=unit_states_csv_input,
        clustering_mode=_prev_clustering_mode,
    output:
        network=resources("networks/base_network_{ty}_{cy}_{ft}_{it}.nc"),
    benchmark:
        "benchmarks/network_building/build_base_network_{ty}_{cy}_{ft}_{it}.tsv"
    log:
        logs("build_base_network_{ty}_{cy}_{ft}_{it}.log"),
    script:
        "../scripts/build_base_network.py"


rule resample_network:
    params:
        time_resolution=config["eva"]["storage_planning"]["po_resolution"],
    input:
        network=resources("networks/base_network_{ty}_{cy}_{ft}_{it}.nc"),
    output:
        network=resources("networks/resampled_network_{ty}_{cy}_{ft}_{it}.nc"),
    benchmark:
        "benchmarks/planning/resample_network_{ty}_{cy}_{ft}_{it}.tsv"
    log:
        logs("resample_network_{ty}_{cy}_{ft}_{it}.log"),
    script:
        "../scripts/resample_network.py"


rule plan_storage:
    params:
        solving=config_provider("solving"),
    input:
        network=resources("networks/resampled_network_{ty}_{cy}_forecast_{it}.nc"),
        base_network=resources("networks/base_network_{ty}_{cy}_forecast_{it}.nc"),
    output:
        storage_plans=resources("pre_optimization/storage_plan_{ty}_{cy}_{it}.csv"),
        mathematical_model=resources(
            "pre_optimization/preopt_model_{ty}_{cy}_forecast_{it}.lp"
        ),
    log:
        logs("plan_storage_{ty}_{cy}_forecast_{it}.log"),
    benchmark:
        "benchmarks/planning/plan_storage_{ty}_{cy}_forecast_{it}.tsv"
    script:
        "../scripts/plan_storage.py"


rule run_day_ahead:
    params:
        market_interconnection=config_provider("eva", "market_interconnection"),
    input:
        network=resources("networks/base_network_{ty}_{cy}_forecast_{it}.nc"),
        storage_plans=storage_plan,
        maintenance_plans=maintenance_plan,
    output:
        day_ahead_network=resources(
            "networks/day_ahead_network_before_cap_update_{ty}_{cy}_forecast_{it}.nc"
        ),
        mathematical_model=resources(
            "market_models/day_ahead_model_{ty}_{cy}_forecast_{it}.lp"
        ),
        day_ahead_network_integer=resources(
            "networks/day_ahead_network_integer_{ty}_{cy}_forecast_{it}.nc"
        ),
        day_ahead_rh_status_csv=resources(
            "market_models/day_ahead_network_model_status_{ty}_{cy}_forecast_{it}.csv"
        ),
    benchmark:
        "benchmarks/market/run_day_ahead_{ty}_{cy}_forecast_{it}.tsv"
    log:
        day_ahead_log=logs("run_day_ahead_{ty}_{cy}_forecast_{it}.log"),
    script:
        "../scripts/run_day_ahead.py"


rule run_intraday:
    params:
        solving=config_provider("solving"),
        market_interconnection=config_provider("eva", "market_interconnection"),
    input:
        day_ahead_network=intraday_da_baseline_input,
        network=resources("networks/base_network_{ty}_{cy}_actual_{it}.nc"),
        storage_plans=storage_plan,
        maintenance_plans=maintenance_plan,
        outage_scenarios=resources("outages/outage_scenario_{ty}_{fo}.csv"),
        link_availability=resources("outages/link_availability_{ty}_{fo}.csv"),
    output:
        intraday_network=resources(
            "networks/intraday_network_before_cap_update_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
        dispatch=resources("dispatch/dispatch_{ty}_{cy}_actual_{fo}_{it}.csv"),
        mathematical_model=resources(
            "market_models/intraday_model_{ty}_{cy}_actual_{fo}_{it}.lp"
        ),
        intraday_rh_status_csv=resources(
            "market_models/intraday_model_status_{ty}_{cy}_actual_{fo}_{it}.csv"
        ),
    benchmark:
        "benchmarks/market/run_intraday_{ty}_{cy}_actual_{fo}_{it}.tsv"
    log:
        logs("run_intraday_{ty}_{cy}_actual_{fo}_{it}.log"),
    script:
        "../scripts/run_intra_day.py"


rule run_intraday_without_ramping:
    """
    Run intraday dispatch without ramping constraints for ramping shortfall indicator.

    This rule runs the same intraday dispatch as run_intraday but with ramping
    constraints disabled. The ramping shortfall indicator is calculated by comparing
    curtailment and ENS between the regular run (with ramping) and this run (without ramping).

    Only runs if ramping shortfall calculation is enabled in config (eva.ramping_shortfall.enable).
    Only runs once per scenario (ty, cy, fo) at the final iteration (after convergence).

    Includes forced outage scenarios to match the regular intraday run.
    """
    params:
        market_interconnection=config_provider("eva", "market_interconnection"),
        disable_ramping=True,  # Flag to disable ramping constraints
    input:
        day_ahead_network=resources(
            "networks/day_ahead_network_{ty}_{cy}_forecast_{it}.nc"
        ),
        network=resources("networks/base_network_{ty}_{cy}_actual_{it}.nc"),
        storage_plans=storage_plan,
        maintenance_plans=maintenance_plan,
        convergence_check=resources("convergence/convergence_{it}.txt"),
        outage_scenarios=resources("outages/outage_scenario_{ty}_{fo}.csv"),
        link_availability=resources("outages/link_availability_{ty}_{fo}.csv"),
    output:
        intraday_network=resources(
            "networks/intraday_network_without_ramping_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
        dispatch=resources(
            "dispatch/dispatch_without_ramping_{ty}_{cy}_actual_{fo}_{it}.csv"
        ),
        mathematical_model=resources(
            "market_models/intraday_model_without_ramping_{ty}_{cy}_actual_{fo}_{it}.lp"
        ),
        intraday_rh_status_csv=resources(
            "market_models/intraday_model_without_ramping_status_{ty}_{cy}_actual_{fo}_{it}.csv"
        ),
    benchmark:
        "benchmarks/outputs/run_intraday_without_ramping_{ty}_{cy}_actual_{fo}_{it}.tsv"
    log:
        logs("run_intraday_without_ramping_{ty}_{cy}_actual_{fo}_{it}.log"),
    script:
        "../scripts/run_intra_day.py"


rule check_infeasibilities:
    params:
        infeasibility_threshold=config_provider("eva", "infeasibility_threshold"),
        fail_on_infeasibility=config_provider("eva", "fail_on_infeasibility"),
    input:
        day_ahead_status=check_infeasibilities_day_ahead_status_input,
        intraday_status=check_infeasibilities_intraday_status_input,
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
        intraday_networks_before_update=intraday_networks_before_update_input,
        day_ahead_networks_before_update=day_ahead_networks_before_update_input,
        week_clusters=update_prices_week_clusters_input,
    output:
        **(
            {
                "chunk_revenues": expand(
                    resources("clustering/chunk_revenues_{ty}_{cy}_{{it}}.csv"),
                    **SCENARIO_NO_FO,
                )
            }
            if config["eva"]["advanced_clustering"]["enable"]
            else {}
        ),
        intraday_networks=expand(
            resources("networks/intraday_network_{ty}_{cy}_actual_{fo}_{{it}}.nc"),
            **SCENARIO,
        ),
        day_ahead_networks=expand(
            resources("networks/day_ahead_network_{ty}_{cy}_forecast_{{it}}.nc"),
            **SCENARIO_NO_FO,
        ),
        marginal_price_comparison_day_ahead=expand(
            resources("plots/marginal_price_comparison_day_ahead_{cy}_{{it}}.png"),
            **SCENARIO_NO_FO,
        ),
        marginal_price_comparison_intraday=expand(
            resources("plots/marginal_price_comparison_intraday_{cy}_{{it}}.png"),
            **SCENARIO,
        ),
        day_ahead_price_cap_timeseries=expand(
            resources("price_caps/day_ahead_price_cap_{ty}_{cy}_forecast_{{it}}.csv"),
            **SCENARIO_NO_FO,
        ),
        intraday_price_cap_timeseries=expand(
            resources(
                "price_caps/intraday_price_cap_{ty}_{cy}_actual_{fo}_{{it}}.csv"
            ),
            **SCENARIO,
        ),
        price_cap_evolution_day_ahead=resources(
            "plots/price_cap_evolution_day_ahead_{it}.png"
        ),
        price_cap_evolution_intraday=resources(
            "plots/price_cap_evolution_intraday_{it}.png"
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
        price_cap_evolution_day_ahead=resources(
            "plots/price_cap_evolution_day_ahead_{it}.png"
        ),
        price_cap_evolution_intraday=resources(
            "plots/price_cap_evolution_intraday_{it}.png"
        ),
    output:
        touch(resources("cleanup/cleanup_before_cap_update_{it}.touch")),
    benchmark:
        "benchmarks/cleanup/cleanup_before_cap_update_{it}.tsv"
    run:
        if config["run"]["keep_files"]:
            return
        for pattern in [
            f"resources/networks/day_ahead_network_before_cap_update_*_{wildcards.it}.nc",
            f"resources/networks/intraday_network_before_cap_update_*_{wildcards.it}.nc",
            f"resources/networks/centroid_intraday_network_before_cap_update_*_{wildcards.it}.nc",
        ]:
            for f in pathlib.Path().glob(pattern):
                mtime = f.stat().st_mtime
                f.write_bytes(b"")
                os.utime(f, (mtime, mtime))


# week_clusters is not indexed by ft (forecast/actual) because DA and ID stages
# must solve the exact same chronological weeks. Clustering on forecast (DA) revenues
# selects the representative weeks; the same mapping is then applied when slicing
# both forecast and actual base networks in slice_network_to_centroid.
# Separating ft would break DA/ID coupling: ID relies on DA commitments and SOC
# outputs from the identical week.
checkpoint cluster_weeks:
    input:
        chunk_revenues=expand(
            resources("clustering/chunk_revenues_{{ty}}_{cy}_{{it}}.csv"),
            cy=config["scenario"]["cy"],
        ),
    output:
        week_clusters=resources("clustering/week_clusters_{ty}_{it}.csv"),
    log:
        logs("cluster_weeks_{ty}_{it}.log"),
    script:
        "../scripts/cluster_weeks.py"


rule slice_network_to_centroid:
    input:
        network=resources("networks/base_network_{ty}_{cy}_{ft}_{it}.nc"),
        week_clusters=lambda wcs: resources(
            f"clustering/week_clusters_{wcs.ty}_{int(wcs.it)-1}.csv"
        ),
    output:
        network=resources(
            "networks/centroid_base_network_{ty}_{cy}_{ft}_{cluster_id}_{it}.nc"
        ),
    log:
        logs("slice_network_to_centroid_{ty}_{cy}_{ft}_{cluster_id}_{it}.log"),
    script:
        "../scripts/slice_network_to_centroid.py"


rule run_intraday_centroid:
    params:
        solving=config_provider("solving"),
        market_interconnection=config_provider("eva", "market_interconnection"),
    input:
        day_ahead_network=resources(
            "networks/centroid_da_network_before_cap_update_{ty}_{cy}_forecast_{cluster_id}_{it}.nc"
        ),
        network=resources(
            "networks/centroid_base_network_{ty}_{cy}_actual_{cluster_id}_{it}.nc"
        ),
        storage_plans=storage_plan,
        maintenance_plans=maintenance_plan,
        outage_scenarios=resources("outages/outage_scenario_{ty}_{fo}.csv"),
        link_availability=resources("outages/link_availability_{ty}_{fo}.csv"),
    output:
        intraday_network=resources(
            "networks/centroid_intraday_network_before_cap_update_{ty}_{cy}_actual_{fo}_{cluster_id}_{it}.nc"
        ),
        dispatch=resources(
            "dispatch/dispatch_centroid_{ty}_{cy}_actual_{fo}_{cluster_id}_{it}.csv"
        ),
        mathematical_model=resources(
            "market_models/centroid_id_model_{ty}_{cy}_actual_{fo}_{cluster_id}_{it}.lp"
        ),
        intraday_rh_status_csv=resources(
            "market_models/centroid_id_model_status_{ty}_{cy}_actual_{fo}_{cluster_id}_{it}.csv"
        ),
    benchmark:
        "benchmarks/market/run_intraday_centroid_{ty}_{cy}_actual_{fo}_{cluster_id}_{it}.tsv"
    log:
        logs("run_intraday_centroid_{ty}_{cy}_actual_{fo}_{cluster_id}_{it}.log"),
    script:
        "../scripts/run_intra_day.py"


rule run_day_ahead_centroid:
    params:
        market_interconnection=config_provider("eva", "market_interconnection"),
    input:
        network=resources(
            "networks/centroid_base_network_{ty}_{cy}_forecast_{cluster_id}_{it}.nc"
        ),
        storage_plans=storage_plan,
        maintenance_plans=maintenance_plan,
    output:
        day_ahead_network=resources(
            "networks/centroid_da_network_before_cap_update_{ty}_{cy}_forecast_{cluster_id}_{it}.nc"
        ),
        mathematical_model=resources(
            "market_models/centroid_da_model_{ty}_{cy}_forecast_{cluster_id}_{it}.lp"
        ),
        day_ahead_network_integer=resources(
            "networks/centroid_da_network_integer_{ty}_{cy}_forecast_{cluster_id}_{it}.nc"
        ),
        day_ahead_rh_status_csv=resources(
            "market_models/centroid_da_model_status_{ty}_{cy}_forecast_{cluster_id}_{it}.csv"
        ),
    benchmark:
        "benchmarks/market/run_day_ahead_centroid_{ty}_{cy}_forecast_{cluster_id}_{it}.tsv"
    log:
        logs("run_day_ahead_centroid_{ty}_{cy}_forecast_{cluster_id}_{it}.log"),
    script:
        "../scripts/run_day_ahead.py"


rule reconstruct_da_status:
    input:
        week_clusters=lambda wcs: resources(
            f"clustering/week_clusters_{wcs.ty}_{int(wcs.it)-1}.csv"
        ),
        centroid_statuses=reconstruct_da_status_inputs,
    output:
        status_csv=resources(
            "market_models/day_ahead_network_model_status_reconstructed_{ty}_{cy}_forecast_{it}.csv"
        ),
    log:
        logs("reconstruct_da_status_{ty}_{cy}_{it}.log"),
    script:
        "../scripts/reconstruct_da_status.py"


rule reconstruct_id_status:
    input:
        week_clusters=lambda wcs: resources(
            f"clustering/week_clusters_{wcs.ty}_{int(wcs.it)-1}.csv"
        ),
        centroid_statuses=reconstruct_id_status_inputs,
    output:
        status_csv=resources(
            "market_models/intraday_model_status_reconstructed_{ty}_{cy}_actual_{fo}_{it}.csv"
        ),
    log:
        logs("reconstruct_id_status_{ty}_{cy}_{fo}_{it}.log"),
    script:
        "../scripts/reconstruct_da_status.py"


def _clustering_inputs(wcs):
    """Return week_clusters CSVs as inputs when clustering is enabled, else empty dict."""
    if not config["eva"]["advanced_clustering"]["enable"]:
        return {}
    return {
        "week_clusters": expand(
            resources("clustering/week_clusters_{ty}_{{it}}.csv"),
            ty=config["scenario"]["ty"],
        ),
    }


rule compute_npv:
    input:
        unpack(_clustering_inputs),
        intraday_networks=intraday_networks_for_npv,
        day_ahead_networks=day_ahead_network_inputs,
        day_ahead_price_cap_timeseries=expand(
            resources("price_caps/day_ahead_price_cap_{ty}_{cy}_forecast_{{it}}.csv"),
            **SCENARIO_NO_FO,
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
        networks_forecast=expand(
            resources("networks/base_network_{ty}_{cy}_forecast_{{it}}.nc"),
            **SCENARIO_NO_FO,
        ),
        networks_actual=expand(
            resources("networks/base_network_{ty}_{cy}_actual_{{it}}.nc"),
            **SCENARIO_NO_FO,
        ),
        npv=resources("npv/npv_{it}.csv"),
        previous_unit_states=previous_unit_states,
        prev_convergence_check=_prev_convergence_check,
    output:
        capacities=resources("capacities/capacity_{it}.csv"),
        unit_states=resources("capacities/unit_states_{it}.csv"),
        convergence_check=resources("convergence/convergence_{it}.txt"),
    log:
        logs("update_capacity_{it}.log"),
    benchmark:
        "benchmarks/capacity_expansion/update_capacity_{it}.tsv"
    script:
        "../scripts/update_capacity.py"


rule determine_clustering_mode:
    input:
        unpack(_clustering_inputs),
        convergence_check=resources("convergence/convergence_{it}.txt"),
        prev_convergence_check=_prev_convergence_check,
    output:
        clustering_mode=resources("clustering/clustering_mode_{it}.txt"),
    log:
        logs("determine_clustering_mode_{it}.log"),
    run:
        it = int(wildcards.it)
        enable = config["eva"]["advanced_clustering"]["enable"]
        freq = config["eva"]["advanced_clustering"]["full_run_frequency"]
        max_it = config["eva"]["max_it"]

        prev_was_clustered_converged = False
        if it > 0 and input.prev_convergence_check:
            prev_conv_path = pathlib.Path(str(input.prev_convergence_check))
            if prev_conv_path.exists():
                prev_was_clustered_converged = (
                    prev_conv_path.read_text().strip() == "ClusteredConverged"
                )

        if (
            not enable
            or it == 0
            or it % freq == 0
            or it >= max_it - 1
            or prev_was_clustered_converged
        ):
            mode = "full"
        else:
            mode = "clustered"
        with open(output.clustering_mode, "w") as f:
            f.write(mode)


def _clustering_mode_for_cleanup(wcs):
    """Ensure determine_clustering_mode runs for every completed iteration.

    Wiring clustering_mode_{it}.txt into cleanup_iteration guarantees the file
    is produced even at the final iteration, when no it+1 exists to request it
    via _prev_clustering_mode.
    """
    if not config["eva"]["advanced_clustering"]["enable"]:
        return []
    return resources(f"clustering/clustering_mode_{wcs.it}.txt")


rule cleanup_iteration:
    """Delete intermediate network files from the previous iteration once the current one completes."""
    input:
        convergence_check=resources("convergence/convergence_{it}.txt"),
        clustering_mode=_clustering_mode_for_cleanup,
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

        stub_patterns = [
            f"resources/networks/resampled_network_*_{prev_it}.nc",
            f"resources/networks/day_ahead_network_integer_*_{prev_it}.nc",
            f"resources/networks/centroid_base_network_*_{prev_it}.nc",
            f"resources/networks/centroid_da_network_integer_*_{prev_it}.nc",
            f"resources/networks/centroid_da_network_before_cap_update_*_{prev_it}.nc",
            f"resources/networks/centroid_intraday_network_before_cap_update_*_{prev_it}.nc",
        ]
        # base_network_*_0.nc is needed by calculate_model_outputs for baseline capacity
        if prev_it > 0:
            stub_patterns.append(f"resources/networks/base_network_*_{prev_it}.nc")
        if not prev_is_final:
            stub_patterns += [
                f"resources/networks/day_ahead_network_*_{prev_it}.nc",
                f"resources/networks/intraday_network_*_{prev_it}.nc",
            ]

            # Replace each file with a zero-byte stub that preserves the original mtime.
            # Snakemake considers a rule done when its outputs exist and are not older than
            # their inputs; preserving mtime means Snakemake won't re-run the rule on restart.
        for pattern in stub_patterns:
            for f in pathlib.Path().glob(pattern):
                mtime = f.stat().st_mtime
                f.write_bytes(b"")
                os.utime(f, (mtime, mtime))


rule plot_process_validation:
    params:
        solving=config_provider("solving"),
    input:
        storage_plans=storage_plan,
        day_ahead_network=resources(
            "networks/day_ahead_network_{ty}_{cy}_forecast_{it}.nc"
        ),
        day_ahead_network_integer=resources(
            "networks/day_ahead_network_integer_{ty}_{cy}_forecast_{it}.nc"
        ),
        intraday_network=resources(
            "networks/intraday_network_{ty}_{cy}_actual_{fo}_{it}.nc"
        ),
        outage_scenarios=resources("outages/outage_scenario_{ty}_{fo}.csv"),
        link_availability=resources("outages/link_availability_{ty}_{fo}.csv"),
    output:
        precondition_day_ahead=resources(
            "plots/precondition_day_ahead_{ty}_{cy}_{fo}_{it}.png"
        ),
        precondition_intraday=resources(
            "plots/precondition_intraday_{ty}_{cy}_{fo}_{it}.png"
        ),
        interlinkage_day_ahead_intraday=resources(
            "plots/interlinkage_day_ahead_intraday_{ty}_{cy}_{fo}_{it}.png"
        ),
        network_flows_day_ahead=resources(
            "plots/network_flows_day_ahead_{ty}_{cy}_{fo}_{it}.png"
        ),
        network_flows_intraday=resources(
            "plots/network_flows_intraday_{ty}_{cy}_{fo}_{it}.png"
        ),
        network_flows_day_ahead_hour_12=resources(
            "plots/network_flows_day_ahead_one_timestamp_{ty}_{cy}_{fo}_{it}.png"
        ),
        network_flows_intraday_hour_12=resources(
            "plots/network_flows_intraday_one_timestamp_{ty}_{cy}_{fo}_{it}.png"
        ),
        forecast_error_load=resources(
            "plots/forecast_error_load_{ty}_{cy}_{fo}_{it}.png"
        ),
        forecast_error_solar=resources(
            "plots/forecast_error_solar_{ty}_{cy}_{fo}_{it}.png"
        ),
        forecast_error_wind=resources(
            "plots/forecast_error_wind_{ty}_{cy}_{fo}_{it}.png"
        ),
        forecast_error_ror=resources("plots/forecast_error_ror_{ty}_{cy}_{fo}_{it}.png"),
        forecast_error_hydro_inflows=resources(
            "plots/forecast_error_hydro_inflows_{ty}_{cy}_{fo}_{it}.png"
        ),
        dispatch_validation_integer_vs_linear=(
            resources(
                "plots/dispatch_validation_integer_vs_linear_{ty}_{cy}_{fo}_{it}.png"
            )
            if not config["solving"]["options"]["linearized_unit_commitment"]
            else []
        ),
        marginal_price_validation_da_vs_id=resources(
            "plots/marginal_price_validation_da_vs_id_{ty}_{cy}_{fo}_{it}.png"
        ),
        reserves_validation_day_ahead=resources(
            "plots/reserves_validation_day_ahead_{ty}_{cy}_{fo}_{it}.png"
        ),
        soc_validation_day_ahead=resources(
            "plots/soc_validation_day_ahead_{ty}_{cy}_{fo}_{it}.png"
        ),
        soc_validation_intraday=resources(
            "plots/soc_validation_intraday_{ty}_{cy}_{fo}_{it}.png"
        ),
        soc_cumsum_validation_day_ahead=resources(
            "plots/soc_cumsum_validation_day_ahead_{ty}_{cy}_{fo}_{it}.png"
        ),
        soc_cumsum_validation_intraday=resources(
            "plots/soc_cumsum_validation_intraday_{ty}_{cy}_{fo}_{it}.png"
        ),
        reserve_revenue_validation_day_ahead=resources(
            "plots/reserve_revenue_validation_day_ahead_{ty}_{cy}_{fo}_{it}.png"
        ),
        hydro_constraints_validation_day_ahead=resources(
            "plots/hydro_constraints_validation_day_ahead_{ty}_{cy}_{fo}_{it}.png"
        ),
        hydro_constraints_validation_intraday=resources(
            "plots/hydro_constraints_validation_intraday_{ty}_{cy}_{fo}_{it}.png"
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
        reserves_interlinkage_p_limits=resources(
            "plots/reserves_interlinkage_p_limits_{ty}_{cy}_{fo}_{it}.png"
        ),
        dsr_validation_day_ahead_implicit=resources(
            "plots/dsr_validation_day_ahead_implicit_{ty}_{cy}_{fo}_{it}.png"
        ),
        dsr_validation_day_ahead_explicit=resources(
            "plots/dsr_validation_day_ahead_explicit_{ty}_{cy}_{fo}_{it}.png"
        ),
        dsr_validation_intraday_implicit=resources(
            "plots/dsr_validation_intraday_implicit_{ty}_{cy}_{fo}_{it}.png"
        ),
        dsr_validation_intraday_explicit=resources(
            "plots/dsr_validation_intraday_explicit_{ty}_{cy}_{fo}_{it}.png"
        ),
        must_run_dispatch_validation_day_ahead=resources(
            "plots/must_run_dispatch_day_ahead_{ty}_{cy}_{fo}_{it}.png"
        ),
        must_run_dispatch_validation_intraday=resources(
            "plots/must_run_dispatch_intraday_{ty}_{cy}_{fo}_{it}.png"
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
        day_ahead_network=resources(
            "networks/day_ahead_network_{ty}_{cy}_forecast_{it}.nc"
        ),
        intraday_network=resources(
            "networks/intraday_network_{ty}_{cy}_actual_{fo}_{it}.nc"
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
    Return no-ramping networks if ramping shortfall calculation is enabled.

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
                "networks/intraday_network_without_ramping_{ty}_{cy}_actual_{fo}_{it}.nc"
            ),
            **SCENARIO,
            it=final_it,
        )
    else:
        return []


def final_intraday_networks(wildcards):
    return expand(
        resources("networks/intraday_network_{ty}_{cy}_actual_{fo}_{it}.nc"),
        **SCENARIO,
        it=_EVA_TRIES,
    )


def final_day_ahead_networks(wildcards):
    return expand(
        resources("networks/day_ahead_network_{ty}_{cy}_forecast_{it}.nc"),
        **SCENARIO_NO_FO,
        it=_EVA_TRIES,
    )


rule finalize_networks:
    input:
        convergence_check=eva_input,
        process_validation_plots=process_validation_plots,
        integration_test_reports=integration_test_outputs,
        intraday_networks_without_ramping=ramping_shortfall_networks,
        iteration_cleanup_flags=iteration_cleanup_flags,
        intraday_networks=final_intraday_networks,
        day_ahead_networks=final_day_ahead_networks,
    output:
        intraday_networks=expand(
            RESULTS + "networks/intraday_network_{ty}_{cy}.nc", **SCENARIO_NO_FO
        ),
        day_ahead_networks=expand(
            RESULTS + "networks/day_ahead_network_{ty}_{cy}.nc", **SCENARIO_NO_FO
        ),
        intraday_networks_without_ramping=(
            expand(
                RESULTS + "networks/intraday_network_without_ramping_{ty}_{cy}.nc",
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
