# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
#
# Shared workflow setup: output path providers, wildcard constraints, and the
# extension points a wrapping workflow may override.
#
# Both Snakefiles include this file first — the model workflow's own, and the
# platform's, which layers database and benchmarking rules on top of these same
# rules. Everything here must therefore be independent of which of the two is
# driving the run.
import os

from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_shadow,
    path_provider,
)

run = config["run"]
RDIR = ""
RESULTS = os.path.join("results/")
shadow_config = get_shadow(run)

# Shared resources are not supported in this workflow (always false)
logs = path_provider("logs/", RDIR, False, [])
resources = path_provider("resources/", RDIR, False, [])


wildcard_constraints:
    fo=r"fo[0-9]+",
    ty=r"[0-9]+",
    cy=r"[0-9]+",
    ft=r"(actual|forecast)",
    it=r"[0-9]+",
    cluster_id=r"[0-9]+",


# --------------------------------------------------------------------------
# Extension points
# --------------------------------------------------------------------------
# A wrapping workflow assigns these after including this file and before
# including model.smk. The model workflow leaves them at their defaults, which
# is what makes it runnable on its own.

#: Callable ``(wildcards) -> str | None`` returning the path of a base network
#: built by the wrapping workflow, or a falsy value to fall back to the
#: configured pre-built snapshot. The platform points this at its
#: build_network_from_database rule.
BASE_NETWORK_PROVIDER = None

#: Callable ``() -> dict`` returning extra ``input:`` entries for
#: build_base_network that carry forecast timeseries, or an empty dict to fall
#: back to the configured CSV paths.
FORECAST_TIMESERIES_INPUTS = None

#: Additional files that must exist before a run counts as finished. They are
#: attached to the final plotting rule, so they are built as part of the run
#: rather than needing a separate target. The platform uses this for its
#: benchmark summary.
EXTRA_FINAL_TARGETS = []
