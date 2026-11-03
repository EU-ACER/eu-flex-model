from snakemake.io import expand
import json


rule generate_all_maintenance_plans:
    input:
        expand(
            resources("pre_optimization/maintenance_plan_{ty}_actual_0.csv"),
            ty=config["scenario"]["ty"],
        ),
    output:
        "maintenance_plans_generated_{run}.flag".replace("{run}", config["run"]["name"]),
    benchmark:
        "benchmarks/planning/generate_all_maintenance_plans.tsv"
    log:
        logs("generate_maintenance_plans_{run}.log").format(run=config["run"]["name"]),
    run:
        metadata = {
            "plan_paths": [str(p) for p in input],
            "ty": config["scenario"]["ty"],
        }
        with open(output[0], "w") as f:
            json.dump(metadata, f, indent=2)
