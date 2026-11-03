# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
#
# The complete EVA rule graph, and the targets that drive it.
#
# Included by both Snakefiles after setup.smk (and, for the platform, after it
# has set its extension points), so that the platform runs exactly these rules
# rather than a copy of them.
from snakemake.io import expand


include: "common.smk"
include: "solve_eva_shared.smk"


# Perfect foresight solves the whole horizon at once and has its own rule graph
if config["eva"]["perfect_foresight"]["enable"]:

    include: "solve_eva_pf.smk"

else:

    include: "solve_eva.smk"


include: "flexibility_assessment.smk"
include: "sensitivity_eva.smk"
include: "generate_maintenance_plans.smk"


rule solve_eva_networks:
    input:
        # In perfect foresight mode: only market networks
        # In standard EVA mode: both day-ahead and intraday networks
        market_networks=(
            expand(
                RESULTS + "networks/market_network_{ty}_{cy}.nc",
                **config["scenario"],
                run=config["run"]["name"],
            )
            if config["eva"]["perfect_foresight"]["enable"]
            else []
        ),
        intraday_networks=(
            []
            if config["eva"]["perfect_foresight"]["enable"]
            else expand(
                RESULTS + "networks/intraday_network_{ty}_{cy}.nc",
                **config["scenario"],
                run=config["run"]["name"],
            )
        ),
        day_ahead_networks=(
            []
            if config["eva"]["perfect_foresight"]["enable"]
            else expand(
                RESULTS + "networks/day_ahead_network_{ty}_{cy}.nc",
                **config["scenario"],
                run=config["run"]["name"],
            )
        ),


rule run_flexibility_assessment:
    input:
        data=expand(RESULTS + "assessment/model_outputs.nc", run=config["run"]["name"]),
        shadow_prices=(
            expand(RESULTS + "assessment/shadow_prices.nc", run=config["run"]["name"])
            if config["eva"]["shadow_prices"]["enable"]
            else []
        ),
    shell:
        "echo 'Flexibility assessment data is ready at {input.data}'"
