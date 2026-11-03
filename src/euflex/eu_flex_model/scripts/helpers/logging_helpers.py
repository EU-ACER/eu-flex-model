# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
#
# SPDX-License-Identifier: MIT


import logging
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import IO

from snakemake.script import Snakemake

from euflex import eu_flex_model
from euflex.eu_flex_model.scripts.extensions import make_log_handler
from euflex.eu_flex_model.scripts.log_formatting import RuleTagFormatter
from euflex.eu_flex_model.scripts.paths import get_path_config

logger = logging.getLogger(__name__)


def warnings_to_file(
    message: str,
    category: type,
    filename: str,
    lineno: int,
    file: IO[str] | None = None,
    line: str | None = None,
    warnings_log_path: Path | None = None,
    warning_categories: tuple | None = None,
) -> None:
    """
    Custom warnings handler that writes specific warnings to file.

    Parameters
    ----------
    message : str
        The warning message.
    category : type
        The warning category class.
    filename : str
        The source file where the warning was raised.
    lineno : int
        The line number in the source file.
    file : file-like object, optional
        File to write to (defaults to stderr for non-captured warnings).
    line : str, optional
        The source code line.
    warnings_log_path : Path, optional
        Path to the warnings log file (required for captured warnings).
    warning_categories : tuple, optional
        Tuple of warning categories to capture (required).
    """
    if warning_categories and issubclass(category, warning_categories):
        with open(warnings_log_path, "a") as f:
            # Format: category: filename:lineno: message
            warning_text = warnings.formatwarning(
                message, category, filename, lineno, line
            )
            f.write(warning_text)
    else:
        # For other warnings, use the default behavior
        if file is None:
            file = sys.stderr
        try:
            file.write(
                warnings.formatwarning(message, category, filename, lineno, line)
            )
        except OSError:
            pass  # Ignore errors writing to stderr


def setup_deprecations_log(deprecations_log_path: Path) -> None:
    """
    Configure Python warnings to redirect specific warning categories to a log file.

    This function sets up a custom warnings handler that captures FutureWarning
    and DeprecationWarning messages and writes them to a dedicated log file,
    while keeping them off the console output.

    Parameters
    ----------
    deprecations_log_path : Path
        Path to the warnings log file where captured warnings will be written.
    """

    # Ensure the directory exists
    deprecations_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Categories to capture
    warning_categories = (FutureWarning, DeprecationWarning)

    # Create and set the warnings handler using partial to bind the log path and categories
    warnings.showwarning = partial(
        warnings_to_file,
        warnings_log_path=deprecations_log_path,
        warning_categories=warning_categories,
    )

    # Ensure these warning categories are always shown (not filtered)
    for category in warning_categories:
        warnings.filterwarnings("default", category=category)


def configure_logging(snakemake: Snakemake, skip_handlers: bool = False):
    """
    Configure the basic behavior for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """

    kwargs = snakemake.config["logging"].copy()
    kwargs.setdefault("level", "INFO")

    logging.getLogger("pypsa").setLevel(logging.WARNING)
    logging.getLogger("pypsa.optimization.constraints").setLevel(logging.DEBUG)
    logging.getLogger("linopy").setLevel(logging.WARNING)
    logging.getLogger("linopy.solvers").setLevel(logging.ERROR)
    logging.getLogger("highspy").setLevel(logging.WARNING)
    logging.getLogger("gribapi.bindings").setLevel(logging.WARNING)
    logging.getLogger("findlibs").setLevel(logging.WARNING)

    if skip_handlers is False:
        fallback_path = Path(eu_flex_model.__path__[0]).joinpath(
            "logs", f"{snakemake.rule}.log"
        )
        logfile = snakemake.log.get(
            "python", snakemake.log[0] if snakemake.log else fallback_path
        )

        rule: str = snakemake.rule

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(RuleTagFormatter(rule))

        handlers: list[logging.Handler] = [
            # Prefer the 'python' log, otherwise take the first log for each
            # Snakemake rule
            logging.FileHandler(logfile),
            stream_handler,
        ]

        # A wrapping distribution may add its own handler (the platform ships
        # workflow logs to Splunk in RFC5424 format); see extensions.py.
        extension_handler = make_log_handler(snakemake)
        if extension_handler is not None:
            handlers.append(extension_handler)

        kwargs.update({"handlers": handlers})
    logging.basicConfig(**kwargs)

    # Setup a function to handle uncaught exceptions and include them with their stacktrace into logfiles
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Log the exception
        logger = logging.getLogger()
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception

    # Setup deprecations log file
    # Use the run directory if available, otherwise use a general logs directory
    run_id: str = snakemake.config.get("run", {}).get("name", "-")
    working_dir = get_path_config().working_dir

    if run_id != "-":
        # Construct path to deprecations log in the run's var directory
        deprecations_log_path = working_dir / run_id / "deprecations.log"
    else:
        # Fallback: use a logs directory in the working dir
        deprecations_log_path = working_dir / "logs" / "deprecations.log"

    setup_deprecations_log(deprecations_log_path)
