# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System

import json
import logging
import shutil
from pathlib import Path

import pypsa

from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
)
from euflex.eu_flex_model.scripts.paths import (
    find_project_root,
    get_path_config,
)

logger = logging.getLogger(__name__)
path_config = get_path_config()


def export_networks_to_csv(
    network_files: list[Path],
    case_name: str,
    tys: list[int],
    cys: list[int],
    project_root: Path,
) -> list[Path]:
    """
    Export network files to CSV folders organized by ty/cy combinations.

    Parameters
    ----------
    network_files : list[Path]
        List of network file paths to export
    case_name : str
        Base case name (e.g., 'base_eva-micro-test')
    tys : list[int]
        List of target years
    cys : list[int]
        List of climate years
    project_root : Path
        Project root directory

    Returns
    -------
    list[Path]
        List of CSV folder paths that were created
    """

    sensitivity_dir = path_config.sensitivity_dir
    if sensitivity_dir:
        base_sensitivity = Path(sensitivity_dir).resolve()
    else:
        base_sensitivity = (project_root / "var" / "workflow" / "sensitivity").resolve()

    csv_paths = []

    for ty in tys:
        for cy in cys:
            # Find matching network file
            network_file = None
            for nf in network_files:
                if f"_{ty}_{cy}_actual_0" in str(nf):
                    network_file = nf
                    break

            if not network_file or not network_file.exists():
                logger.warning(f"Network file not found for ty={ty}, cy={cy}")
                continue

            # Create CSV directory
            if sensitivity_dir:
                csv_path = Path(sensitivity_dir) / case_name / f"ty_{ty}_cy_{cy}"
            else:
                csv_path = (
                    project_root
                    / "var"
                    / "workflow"
                    / "sensitivity"
                    / case_name
                    / f"ty_{ty}_cy_{cy}"
                )

            # Security: Validate path is contained within expected base directory
            csv_path_resolved = csv_path.resolve()
            if not csv_path_resolved.is_relative_to(base_sensitivity):
                logger.error(
                    f"Security violation: resolved path {csv_path_resolved} escapes base directory {base_sensitivity}"
                )
                raise ValueError(
                    "Invalid case name: resolved path escapes base directory"
                )

            csv_path_resolved.mkdir(parents=True, exist_ok=True)

            # Load and export network
            logger.info(f"Exporting ty={ty}, cy={cy} to {csv_path_resolved}")
            n = pypsa.Network(str(network_file))
            n.export_to_csv_folder(str(csv_path_resolved))

            csv_paths.append(csv_path_resolved)

    if not csv_paths:
        raise RuntimeError("No networks were successfully exported")

    return csv_paths


if __name__ == "__main__":
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="prepare_sensitivity",
    )

    configure_logging(snakemake)

    # Get parameters from snakemake
    network_files = [
        Path(f)
        for f in snakemake.input.base_networks  # ty: ignore[possibly-missing-attribute]
    ]
    config = snakemake.config
    output_flag = Path(snakemake.output[0])

    # Extract ty/cy from config
    tys = config["scenario"]["ty"]
    cys = config["scenario"]["cy"]

    # Determine config name from run name
    config_name = config["run"]["parent"]

    # Determine case name
    case_name = f"base_{config_name}"

    path_config = get_path_config()
    # Get project root (script is in src/workflow/scripts)
    project_root = find_project_root()

    sensitivity_dir = path_config.sensitivity_dir
    if sensitivity_dir:
        case_path = Path(sensitivity_dir) / case_name
        expected_base = Path(sensitivity_dir).resolve()
    else:
        case_path = project_root / "var" / "workflow" / "sensitivity" / case_name
        expected_base = (project_root / "var" / "workflow" / "sensitivity").resolve()

    # Security: Validate path is contained within expected base directory
    case_path_resolved = case_path.resolve()
    try:
        # Try to validate even if path doesn't exist yet by checking parent
        if case_path.exists():
            path_to_check = case_path_resolved
        else:
            # Check parent path if case_path does not exist
            path_to_check = case_path.parent.resolve()

        if not path_to_check.is_relative_to(expected_base):
            logger.error(
                f"Security violation: resolved path {case_path_resolved} escapes base directory {expected_base}"
            )
            raise ValueError("Invalid run.parent: resolved path escapes base directory")
    except (OSError, ValueError) as e:
        logger.error(f"Security violation: could not validate path {case_path}: {e}")
        raise ValueError("Invalid run.parent: path validation failed")

    # Clean up existing case folder
    if case_path.exists():
        logger.info(f"Deleting existing sensitivity case: {case_path}")
        shutil.rmtree(case_path)

    # Export networks to CSV
    csv_paths = export_networks_to_csv(network_files, case_name, tys, cys, project_root)

    # Describe the export in the flag file, so that whatever drives the run
    # can report where the CSVs landed without re-deriving the paths.
    metadata = {
        "case_name": case_name,
        "case_path": str(case_path),
        "csv_paths": [str(p) for p in csv_paths],
        "tys": tys,
        "cys": cys,
        "preset": config["run"]["parent"].replace("eva-", ""),
    }

    output_flag.parent.mkdir(parents=True, exist_ok=True)
    with open(output_flag, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Successfully exported {len(csv_paths)} scenario(s) to CSV")
