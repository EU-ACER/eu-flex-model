# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
#
# SPDX-License-Identifier: MIT


import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from snakemake.api import Workflow
from snakemake.common import SNAKEFILE_CHOICES
from snakemake.dag import DAG
from snakemake.io import Wildcards
from snakemake.jobs import Job
from snakemake.logging import LoggerManager
from snakemake.logging import logger as snakemake_logger
from snakemake.script import Snakemake
from snakemake.settings.types import (
    ConfigSettings,
    DAGSettings,
    OutputSettings,
    ResourceSettings,
    StorageSettings,
    WorkflowSettings,
)
from snakemake.utils import update_config

from euflex import eu_flex_model
from euflex.eu_flex_model.scripts.paths import get_path_config

logger = logging.getLogger(__name__)


def get_parent_run(run: str, config: dict[str, Any]) -> str:
    """
    Function to get the optional parent run name.

    Parameters
    ----------
    run: str
        The actual run name.
    config: dict
        The configuration dictionary.

    Returns
    -------
    str: The parent run name if given, otherwise returns the run name as is.
    """
    return config["run"]["parent"] or run


def get_run_path(
    fn: str,
    dir: str,
    rdir: str,
    shared_resources: str | bool,
    exclude_from_shared: list,
) -> str:
    """
    Dynamically provide paths based on shared resources and filename.

    Use this function for snakemake rule inputs or outputs that should be
    optionally shared across runs or created individually for each run.

    Parameters
    ----------
    fn : str
        The filename for the path to be generated.
    dir : str
        The base directory.
    rdir : str
        Relative directory for non-shared resources.
    shared_resources : str or bool
        Specifies which resources should be shared.
        - If string is "base", special handling for shared "base" resources (see notes).
        - If random string other than "base", this folder is used instead of the `rdir` keyword.
        - If boolean, directly specifies if the resource is shared.
    exclude_from_shared: list
        List of filenames to exclude from shared resources. Only relevant if shared_resources is "base".

    Returns
    -------
    str
        Full path where the resource should be stored.

    Notes
    -----
    Special case for "base" allows no wildcards other than "technology", "year"
    and "scope" and excludes filenames starting with "networks/elec" or
    "add_electricity". All other resources are shared.
    """

    if shared_resources == "base":
        pattern = r"\{([^{}]+)\}"
        existing_wildcards = set(re.findall(pattern, fn))
        irrelevant_wildcards = {"technology", "year", "scope", "kind"}
        no_relevant_wildcards = not existing_wildcards - irrelevant_wildcards
        not_shared_rule = (
            not fn.endswith("elec.nc")
            and not fn.startswith("add_electricity")
            and not any(fn.startswith(ex) for ex in exclude_from_shared)
        )
        is_shared = no_relevant_wildcards and not_shared_rule
        shared_files = (
            "networks/base_s_{clusters}.nc",
            "regions_onshore_base_s_{clusters}.geojson",
            "regions_offshore_base_s_{clusters}.geojson",
            "busmap_base_s_{clusters}.csv",
            "linemap_base_s_{clusters}.csv",
            "cluster_network_base_s_{clusters}",
            "profile_{clusters}_",
            "build_renewable_profile_{clusters}",
            "regions_by_class_{clusters}",
            "availability_matrix_",
            "determine_availability_matrix_",
            "solar_thermal",
        )
        if any(prefix in fn for prefix in shared_files) or is_shared:
            is_shared = True
        rdir = "" if is_shared else rdir
    elif isinstance(shared_resources, str):
        rdir = shared_resources + "/"
    elif isinstance(shared_resources, bool):
        rdir = "" if shared_resources else rdir
    else:
        raise ValueError(
            "shared_resources must be a boolean, str, or 'base' for special handling."
        )

    return f"{rdir}{dir}{fn}"


def path_provider(
    dir: str,
    rdir: str,
    shared_resources: list | str | bool,
    exclude_from_shared: list[str],
) -> partial:
    """
    Returns a partial function that dynamically provides paths based on shared
    resources and the filename.

    Returns
    -------
    partial function
        A partial function that takes a filename as input and
        returns the path to the file based on the shared_resources parameter.
    """
    return partial(
        get_run_path,
        dir=dir,
        rdir=rdir,
        shared_resources=shared_resources,
        exclude_from_shared=exclude_from_shared,
    )


def get_shadow(run):
    """
    Returns 'shallow' or None depending on the user setting.
    """
    shadow_config = run.get("use_shadow_directory", True)
    if shadow_config:
        return "shallow"
    return None


def set_scenario_config(snakemake):
    scenario = snakemake.config["run"].get("scenarios", {})
    if scenario.get("enable") and "run" in snakemake.wildcards.keys():
        try:
            with open(scenario["file"]) as f:
                scenario_config = yaml.safe_load(f)
        except FileNotFoundError:
            # fallback for mock_snakemake, relative to the model package root
            model_dir = Path(eu_flex_model.__path__[0])
            with open(model_dir / scenario["file"]) as f:
                scenario_config = yaml.safe_load(f)
        run_name = get_parent_run(snakemake.wildcards.run, snakemake.config)
        update_config(snakemake.config, scenario_config[run_name])


def get_snakemake(
    globals_scope: dict,
    rule: str,
    configfiles: list[Path] | None = None,
    workdir: str | Path | None = None,
    **wildcards: Any,
) -> Snakemake:
    """
    Retrieve or create a Snakemake mock object from a given global scope.

    Parameters
    ----------
    globals_scope : dict
        Dictionary of global variables to check for an existing Snakemake object.
    rule : str
        Name of the Snakemake rule to mock. Default is None.
    configfiles : list of Path, optional
        List of configuration files to use with the Snakemake mock. Default is None.
    workdir : str or Path, optional
        Working directory for resolving input/output paths. Use this to point
        to a previous run's output directory (e.g., "var/20260130_eva-temporal-test_ef3cfc16").
        Defaults to ``get_path_config().working_dir`` (typically ``<project_root>/var``).
    **wildcards : Any
        Additional wildcards arguments to pass to the mock_snakemake function.

    Returns
    -------
    Snakemake
        A mocked Snakemake object, either retrieved from the global scope
        or created using the provided parameters.
    """

    # Check if snakemake is in the provided global scope
    if "snakemake" in globals_scope:
        return globals_scope["snakemake"]

    # Resolve working directory: fall back to path_config.working_dir (var/)
    if workdir is None:
        workdir = get_path_config().working_dir

    # Prepare arguments for mock_snakemake
    mock_args: dict[str, Any] = {}

    # Add rule if provided
    if rule is not None:
        mock_args["rulename"] = rule

    # Add configfiles if provided
    if configfiles is not None:
        mock_args["configfiles"] = configfiles

    # workdir is always set (either explicitly or from path_config)
    mock_args["workdir"] = workdir

    # Add any additional kwargs
    mock_args.update(wildcards)

    # Create mock with prepared arguments
    return mock_snakemake(**mock_args)


def mock_snakemake(
    rulename: str,
    root_dir: str | Path | None = None,
    configfiles: list[Path] | None = None,
    workdir: str | Path | None = None,
    **wildcards: Any,
) -> Snakemake:
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename : str
        Name of the rule for which the Snakemake object should be generated.
    root_dir : str or Path, optional
        Directory to search for the Snakefile. Defaults to the model package
        directory, which is where the model workflow's own Snakefile lives.
    configfiles : list, optional
        List of configuration files to use for the mock. Default is None.
    workdir : str or Path, optional
        Working directory for resolving input/output paths. When specified,
        relative paths (e.g., resources/networks/...) are resolved relative
        to this directory instead of the current working directory.
        Useful for pointing to a previous run's output directory.
    **wildcards : dict
        Wildcards required for the rule.

    Returns
    -------
    Snakemake
        A mocked Snakemake object for the specified rule.
    """

    script_dir = Path(__file__).parent.resolve()
    root_dir = Path(root_dir or eu_flex_model.__path__[0]).resolve()

    user_in_script_dir = False
    if workdir is not None:
        workdir = Path(workdir).resolve()
    else:
        user_in_script_dir = Path.cwd().resolve() == script_dir
        if user_in_script_dir:
            os.chdir(root_dir)
        elif Path.cwd().resolve() != root_dir:
            logger.info(
                "Not in scripts or root directory, will assume this is a separate workdir"
            )
            workdir = Path.cwd()

    snakefile = Path()
    try:
        for p in SNAKEFILE_CHOICES:
            p = root_dir / p
            if os.path.exists(p):
                snakefile = p
                break
        if configfiles is None:
            # Resolve the bundled defaults from the package rather than from
            # root_dir, which a caller may point at the project root or at a
            # wrapping workflow. Only the model layer is loaded here; a run
            # that needs a platform overlay has to pass configfiles itself.
            path_config = get_path_config()
            configfiles = [path_config.config_dir / path_config.DEFAULT_CONFIG_FILENAME]
        elif isinstance(configfiles, str):
            configfiles = [Path(configfiles)]

        resource_settings = ResourceSettings()
        config_settings = ConfigSettings(configfiles=configfiles)
        workflow_settings = WorkflowSettings()
        storage_settings = StorageSettings()
        dag_settings = DAGSettings(rerun_triggers=set())
        # Keywords, not positionals: Snakemake has added fields in the middle
        # of this signature before (logger_manager, between workflow_settings
        # and storage_settings), which silently shifts every later argument.
        workflow = Workflow(
            logger_manager=LoggerManager(snakemake_logger, OutputSettings()),
            config_settings=config_settings,
            resource_settings=resource_settings,
            workflow_settings=workflow_settings,
            storage_settings=storage_settings,
            dag_settings=dag_settings,
            storage_provider_settings=dict(),
            overwrite_workdir=workdir,
        )
        workflow.include(snakefile)

        if configfiles:
            for f in configfiles:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Config file {f} does not exist.")
                workflow.configfile(f)

        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = DAG(workflow, rules=[rule])
        job = Job(rule, dag, Wildcards(fromdict=wildcards))

        def make_accessable(*ios):
            base = Path(workdir) if workdir else Path.cwd()
            for io in ios:
                for i, _ in enumerate(io):
                    io[i] = str(base / io[i]) if not os.path.isabs(io[i]) else io[i]

        make_accessable(job.input, job.output, job.log)
        # Create Snakemake object with job details
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            workflow.config,
            rulename,
            None,
        )

        # Ensure log and output directories exist
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    finally:
        if user_in_script_dir:
            os.chdir(script_dir)
    return snakemake


#: Name of the CSV holding per-bidding-zone curtailment targets, under ``data/``
#: in the checkout the workflow reads its inputs from.
CURTAILMENT_TARGETS_CSV = "ds_curtailment_targets_by_bz.csv"


def curtailment_targets_csv() -> Path:
    """
    Locate the per-bidding-zone curtailment targets CSV.

    Resolved against the input directory rather than the working directory, so
    that it is still found when the workflow runs from elsewhere with
    ``--directory``.

    Returns
    -------
    Path
        Path to the CSV, whether or not it exists.
    """
    return get_path_config().input_dir / "data" / CURTAILMENT_TARGETS_CSV


def validate_curtailment_targets_per_bz(decimal_targets: list[float]) -> None:
    """
    Check that the per-bidding-zone curtailment targets CSV covers every target.

    Per-bidding-zone mode reads the target values from a CSV keyed by ``ct1``,
    ``ct2`` and so on, one per entry in ``curtailment_targets``. Checking that
    up front, while the rules are being defined, fails a misconfigured run
    before any job starts rather than part-way through the assessment.

    Parameters
    ----------
    decimal_targets : list of float
        The configured ``eva.dummy_storage_optimization.curtailment_targets``.
        Only its length matters here; the values come from the CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing.
    ValueError
        If ``curtailment_targets`` is empty, or the CSV lacks the required
        columns or any of the ``ct1``..``ctN`` variables.
    """
    import pandas as pd

    if not decimal_targets:
        raise ValueError(
            "eva.dummy_storage_optimization.curtailment_targets_per_bz is enabled "
            "but curtailment_targets is empty. Give one entry per target, for "
            "example: curtailment_targets: [0.05, 0.1, 0.15]"
        )

    csv_path = curtailment_targets_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Per-bidding-zone curtailment targets are enabled but "
            f"{csv_path} does not exist.\n"
            f"Point EUFLEX_INPUT_DIR at the checkout holding data/, write the "
            f"CSV with columns BZ, variable, value, or set "
            f"eva.dummy_storage_optimization.curtailment_targets_per_bz to false."
        )

    targets = pd.read_csv(csv_path)

    required_columns = {"BZ", "variable", "value"}
    missing_columns = required_columns - set(targets.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing the columns {sorted(missing_columns)}. "
            f"Expected {sorted(required_columns)}, found {sorted(targets.columns)}."
        )

    available = set(targets["variable"].unique())
    required = [f"ct{i + 1}" for i in range(len(decimal_targets))]
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(
            f"{csv_path} has no rows for {missing}. "
            f"curtailment_targets lists {len(decimal_targets)} target(s), so the "
            f"CSV needs {required}; it has {sorted(available)}."
        )
