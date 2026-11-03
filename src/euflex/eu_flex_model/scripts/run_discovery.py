# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
"""
Discovery of networks produced by a completed EVA run.

Fixed-market-clearing (FMC) runs start from the final iteration networks of an
earlier EVA run rather than from a base network. The helpers here locate those
networks on disk and index them by scenario, which the ``solve_eva`` rules use
to resolve their inputs.

This is a filesystem scan of a run directory, with no dependency on the CLI
that produced it.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_networks_dir(eva_run_path: Path) -> Path:
    """
    Find the networks directory within the EVA run path.

    Searches for a 'networks' directory in common locations within an
    EVA run output directory, such as 'resources/networks' or 'networks'.

    Parameters
    ----------
    eva_run_path : Path
        The path to the root directory of a completed EVA run.

    Returns
    -------
    Path
        The path to the found networks directory.

    Raises
    ------
    FileNotFoundError
        If the `eva_run_path` does not exist or if a networks directory
        cannot be found in the expected locations.
    """
    if not eva_run_path.exists():
        raise FileNotFoundError(f"EVA run path not found: {eva_run_path}")

    possible_dirs = [
        eva_run_path / "resources" / "networks",
        eva_run_path / "networks",
    ]
    for networks_dir in possible_dirs:
        if networks_dir.exists():
            return networks_dir

    raise FileNotFoundError(
        f"Networks directory not found in {eva_run_path}. "
        "Expected either resources/networks or networks subdirectory."
    )


def _iter_base_network_files(
    networks_dir: Path,
) -> Iterator[tuple[str, str, str, int, Path]]:
    """
    Yield the parsed metadata of every well-formed base network file.

    Scans `networks_dir` for files matching ``base_network_{ty}_{cy}_{ft}_{it}.nc``.
    Shared by both the FMC and restart discovery paths, which differ only in how
    they aggregate these tuples (latest-per-variant vs. grouped-by-iteration).

    Parameters
    ----------
    networks_dir : Path
        The directory containing the network files.

    Yields
    ------
    tuple[str, str, str, int, Path]
        ``(ty, cy, ft, it, path)`` for each file whose name parses; malformed
        filenames (wrong prefix, non-numeric iteration) are silently skipped.
    """
    for network_path in networks_dir.glob("base_network_*_*_*_*.nc"):
        parts = network_path.stem.split("_")
        if len(parts) < 6 or parts[0] != "base" or parts[1] != "network":
            continue
        ty, cy, ft, it_str = parts[2], parts[3], parts[4], parts[5]
        try:
            it = int(it_str)
        except ValueError:
            continue  # Ignore malformed filenames
        yield ty, cy, ft, it, network_path


def _parse_network_files(
    networks_dir: Path,
) -> dict[tuple[str, str, str], tuple[int, Path]]:
    """
    Parse network filenames to find the latest iteration for each variant.

    Identifies the file with the highest iteration number for each unique
    combination of target year, climate year, and forecast type.

    Parameters
    ----------
    networks_dir : Path
        The directory containing the network files.

    Returns
    -------
    dict[tuple[str, str, str], tuple[int, Path]]
        A dictionary mapping a key ``(ty, cy, ft)`` to a tuple containing
        the highest iteration number and the corresponding file ``Path``.

    Raises
    ------
    ValueError
        If no network files are found in the directory or if no valid
        network filenames can be parsed.
    """
    if not list(networks_dir.glob("base_network_*_*_*_*.nc")):
        raise ValueError(
            f"No base_network files found in {networks_dir}. "
            f"The EVA run may not have completed successfully."
        )

    network_info: dict[tuple[str, str, str], tuple[int, Path]] = {}
    for ty, cy, ft, it, network_path in _iter_base_network_files(networks_dir):
        key = (ty, cy, ft)
        if key not in network_info or it > network_info[key][0]:
            network_info[key] = (it, network_path)

    if not network_info:
        raise ValueError(
            f"Could not parse any valid network files in {networks_dir}. "
            "Expected format: base_network_{ty}_{cy}_{ft}_{it}.nc"
        )

    return network_info


def _organize_networks_by_scenario(
    network_info: dict[tuple[str, str, str], tuple[int, Path]],
) -> dict[str, dict[str, Path]]:
    """
    Organize parsed network info into a nested dictionary by scenario.

    Parameters
    ----------
    network_info : dict[tuple[str, str, str], tuple[int, Path]]
        A dictionary mapping a tuple of (target year, climate year, forecast type)
        to a tuple containing the iteration number and the file path.

    Returns
    -------
    dict[str, dict[str, Path]]
        A nested dictionary where outer keys are scenario strings (e.g., "2030_2013")
        and the inner dictionary maps the forecast type (e.g., "actual", "forecast")
        to the corresponding network Path.
    """
    result: dict[str, dict[str, Path]] = {}
    for (ty, cy, ft), (_, path) in network_info.items():
        key = f"{ty}_{cy}"
        if key not in result:
            result[key] = {}
        result[key][ft] = path
    return result


def find_iteration_networks(eva_run_path: Path) -> dict[str, dict[str, Path]]:
    """
    Find the iteration networks from a completed EVA run.

    Scans the resources/networks directory of an EVA run to find the highest
    iteration number for updated networks, which are the final converged networks.

    Parameters
    ----------
    eva_run_path : Path
        Path to the EVA run directory (e.g., var/results/20251105_eva-quick-test_abc12345)

    Returns
    -------
    dict[str, dict[str, Path]]
        Nested dict mapping (ty, cy) -> {"forecast": Path, "actual": Path}
        The forecast network is used for day-ahead, actual for intraday

    Raises
    ------
    FileNotFoundError
        If the EVA run path doesn't exist or no networks are found
    ValueError
        If no iteration networks are found in the expected location
    """
    networks_dir = _find_networks_dir(eva_run_path)
    network_info = _parse_network_files(networks_dir)
    return _organize_networks_by_scenario(network_info)


def _parse_restart_network_files(
    networks_dir: Path,
) -> dict[str, dict[int, dict[str, Path]]]:
    """
    Group network files by scenario and iteration.

    Unlike :func:`_parse_network_files`, which collapses straight to the highest
    iteration per forecast type, this keeps every iteration so that a *common*
    iteration across forecast types can be selected afterwards. Zero-byte files
    are ignored: a run killed mid-write leaves a truncated network behind, and
    restarting from it would fail deep inside the workflow instead of here.

    Parameters
    ----------
    networks_dir : Path
        The directory containing the ``base_network_{ty}_{cy}_{ft}_{it}.nc`` files.

    Returns
    -------
    dict[str, dict[int, dict[str, Path]]]
        Mapping ``"{ty}_{cy}" -> iteration -> forecast type -> path``.
    """
    by_scenario: dict[str, dict[int, dict[str, Path]]] = {}

    for ty, cy, ft, it, network_path in _iter_base_network_files(networks_dir):
        if network_path.stat().st_size == 0:
            continue  # Truncated or placeholder file
        by_scenario.setdefault(f"{ty}_{cy}", {}).setdefault(it, {})[ft] = network_path

    return by_scenario


def find_restart_networks(
    eva_run_path: Path,
    required_fts: tuple[str, ...] = ("actual", "forecast"),
) -> dict[str, tuple[int, dict[str, Path]]]:
    """
    Find the latest iteration of each scenario that can seed a warm-started run.

    For every ``(ty, cy)`` scenario the highest iteration is selected for which
    *all* of ``required_fts`` are present. Requiring a complete set matters when
    a run dies partway through an iteration: the forecast network for iteration
    N may exist while the actual one does not, and mixing iterations across
    forecast types would silently start the new run from an inconsistent state.

    Parameters
    ----------
    eva_run_path : Path
        Path to the root directory of the parent EVA run.
    required_fts : tuple[str, ...]
        Forecast types that must all be present for an iteration to qualify.
        Perfect-foresight runs only produce ``actual`` networks.

    Returns
    -------
    dict[str, tuple[int, dict[str, Path]]]
        Mapping ``"{ty}_{cy}" -> (iteration, {forecast type: path})``.

    Raises
    ------
    FileNotFoundError
        If `eva_run_path` or its networks directory does not exist.
    ValueError
        If no network files exist, or none of them form a complete iteration.
    """
    networks_dir = _find_networks_dir(eva_run_path)
    by_scenario = _parse_restart_network_files(networks_dir)

    if not by_scenario:
        raise ValueError(
            f"No usable base_network files found in {networks_dir}. "
            "The parent run may not have produced any complete iteration yet."
        )

    restart_networks: dict[str, tuple[int, dict[str, Path]]] = {}
    for scenario, iterations in by_scenario.items():
        complete = [
            it
            for it, variants in iterations.items()
            if all(ft in variants for ft in required_fts)
        ]
        if complete:
            latest = max(complete)
            restart_networks[scenario] = (latest, iterations[latest])

    if not restart_networks:
        raise ValueError(
            f"No complete iteration found in {networks_dir}. "
            f"Every iteration is missing at least one of the required "
            f"forecast types {list(required_fts)}."
        )

    return restart_networks
