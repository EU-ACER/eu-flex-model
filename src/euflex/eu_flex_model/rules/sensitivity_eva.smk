from snakemake.io import expand


rule prepare_sensitivity:
    input:
        base_networks=expand(
            resources("networks/base_network_{ty}_{cy}_actual_0.nc"),
            **config["scenario"],
            run=config["run"]["name"],
        ),
    output:
        "sensitivity_export_{run}.flag".replace("{run}", config["run"]["name"]),
    log:
        logs("prepare_sensitivity.log").format(run=config["run"]["name"]),
    script:
        "../scripts/prepare_sensitivity.py"
