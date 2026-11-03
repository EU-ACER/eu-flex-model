# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
"""
Process-level extension points for the EU-Flex model workflow.

The model workflow is self-contained: every extension point defined here has a
working default, so ``snakemake`` runs the full assessment with nothing else
installed. A wrapping distribution — ``eu-flex-platform`` — plugs its own
behaviour in by pointing an environment variable at an import target *before*
invoking Snakemake.

Environment variables are used in preference to an in-process ``register()``
call because Snakemake executes each rule in a separate Python process (and,
on a cluster, on a separate machine). Anything registered in the scheduler
process would not reach the scripts; environment variables survive fork, spawn
and job submission alike.

Extension points
----------------
``EUFLEX_LOG_HANDLER_FACTORY``
    ``"module:callable"``. Called as ``callable(snakemake)`` and expected to
    return a :class:`logging.Handler` (or ``None``), which is added to the
    root logger alongside the workflow's own file and stream handlers. The
    platform uses this to ship RFC5424 syslog records to Splunk.

``EUFLEX_SOLVER_INSTRUMENTATION``
    ``"module"``. The module must expose two callables:

    ``prepare_instrumentation(network, snakemake, benchmark_root)``
        Called once per solve script, after the network is loaded, to record
        whatever the instrumentation needs on ``network.meta``.

    ``run_optimization(optimize_callable, optimize_kwargs, network, module)``
        Called in place of ``optimize_callable(**optimize_kwargs)`` and must
        return the same ``(status, condition)`` pair.

    The platform uses this to wrap every solve in its solver profiler.

``EUFLEX_FLOW_BASED_DATA_PROVIDER``
    ``"module:callable"``. Called before the workflow registers flow-based
    market coupling data from its own Excel/CSV inputs, and takes precedence
    over them: returning ``True`` means the extension has registered the data
    itself and the built-in path is skipped. The platform uses this to load
    PTDFs and RAMs from the database.

A misconfigured extension is a hard error rather than a silent fallback: an
unreachable target means the wrapping distribution is broken, and quietly
running without it would drop production logging or profiling without a trace.
"""

import importlib
import logging
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

ENV_LOG_HANDLER_FACTORY = "EUFLEX_LOG_HANDLER_FACTORY"
ENV_SOLVER_INSTRUMENTATION = "EUFLEX_SOLVER_INSTRUMENTATION"
ENV_FLOW_BASED_DATA_PROVIDER = "EUFLEX_FLOW_BASED_DATA_PROVIDER"


class ExtensionLoadError(RuntimeError):
    """Raised when a configured extension point cannot be imported."""


@cache
def _import_module(module_name: str) -> ModuleType:
    """
    Import a module by name, reporting the extension variable on failure.

    Parameters
    ----------
    module_name : str
        Dotted module path.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    ExtensionLoadError
        If the module cannot be imported.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ExtensionLoadError(
            f"Could not import extension module '{module_name}': {exc}"
        ) from exc


@cache
def _load_callable(spec: str) -> Callable[..., Any]:
    """
    Resolve a ``"module:attribute"`` specification to a callable.

    Parameters
    ----------
    spec : str
        Import target in ``"package.module:attribute"`` form.

    Returns
    -------
    Callable
        The resolved callable.

    Raises
    ------
    ExtensionLoadError
        If the specification is malformed, or the target cannot be imported or
        is not callable.
    """
    if ":" not in spec:
        raise ExtensionLoadError(
            f"Invalid extension specification '{spec}': "
            "expected the form 'package.module:attribute'"
        )

    module_name, _, attribute = spec.partition(":")
    module = _import_module(module_name)

    try:
        target = getattr(module, attribute)
    except AttributeError as exc:
        raise ExtensionLoadError(
            f"Extension module '{module_name}' has no attribute '{attribute}'"
        ) from exc

    if not callable(target):
        raise ExtensionLoadError(f"Extension target '{spec}' is not callable")

    return target


def make_log_handler(snakemake: Any) -> logging.Handler | None:
    """
    Build the extension-provided logging handler, if one is configured.

    Parameters
    ----------
    snakemake : Any
        The rule's Snakemake object, passed straight through to the factory so
        it can name records after the running rule.

    Returns
    -------
    logging.Handler or None
        The handler to attach to the root logger, or ``None`` when no factory
        is configured or the factory declines to build one.
    """
    spec = os.environ.get(ENV_LOG_HANDLER_FACTORY)
    if not spec:
        return None

    return _load_callable(spec)(snakemake)


def _solver_instrumentation() -> ModuleType | None:
    """
    Return the configured solver-instrumentation module, or ``None``.

    Returns
    -------
    ModuleType or None
        The imported instrumentation module when
        ``EUFLEX_SOLVER_INSTRUMENTATION`` is set, otherwise ``None``.
    """
    module_name = os.environ.get(ENV_SOLVER_INSTRUMENTATION)
    if not module_name:
        return None

    return _import_module(module_name)


def prepare_solver_instrumentation(
    network: Any, snakemake: Any, benchmark_root: Path
) -> None:
    """
    Give the instrumentation extension a chance to annotate the network.

    A no-op when no extension is configured, which is the case for every run of
    the standalone model workflow.

    Parameters
    ----------
    network : Any
        The PyPSA network about to be solved. The extension may write to
        ``network.meta``.
    snakemake : Any
        The rule's Snakemake object, giving the extension access to the rule
        name and the run configuration.
    benchmark_root : Path
        Directory under which the extension should place any per-rule output
        it writes. It is not created here — the extension creates whatever it
        actually uses.
    """
    module = _solver_instrumentation()
    if module is None:
        return

    module.prepare_instrumentation(network, snakemake, benchmark_root)


def run_optimization(
    *,
    optimize_callable: Callable[..., tuple[Any, Any]],
    optimize_kwargs: dict[str, Any],
    network: Any,
    module: str,
) -> tuple[Any, Any]:
    """
    Run an optimisation, routing it through the instrumentation extension.

    With no extension configured this is exactly
    ``optimize_callable(**optimize_kwargs)``.

    Parameters
    ----------
    optimize_callable : Callable
        The solve entry point, returning a ``(status, condition)`` pair.
    optimize_kwargs : dict of str to Any
        Keyword arguments for ``optimize_callable``.
    network : Any
        The network being solved, for the extension to read metadata from.
    module : str
        Label identifying the calling step, used by the extension to tag its
        records (e.g. ``"plan_storage"``).

    Returns
    -------
    tuple
        The ``(status, condition)`` pair returned by the solve.
    """
    instrumentation = _solver_instrumentation()
    if instrumentation is None:
        return optimize_callable(**optimize_kwargs)

    return instrumentation.run_optimization(
        optimize_callable=optimize_callable,
        optimize_kwargs=optimize_kwargs,
        network=network,
        module=module,
    )


def register_flow_based_data_extension(
    network: Any,
    config: dict[str, Any],
    climate_year: int,
    target_year: int,
    eraa_year: int,
) -> bool:
    """
    Offer an extension the chance to supply flow-based market coupling data.

    Called by :func:`fbmc_ntc_helpers.register_market_interconnection` before
    it loads PTDFs and RAMs from the workflow's own Excel/CSV inputs. An
    extension that returns ``True`` has already registered the data on the
    network and the built-in path is skipped.

    Parameters
    ----------
    network : Any
        The PyPSA network to register flow-based data on.
    config : dict of str to Any
        The full run configuration, for the extension to read its own settings
        from.
    climate_year : int
        Climate year the data is requested for.
    target_year : int
        Target year the data is requested for.
    eraa_year : int
        ERAA study year the data is requested for.

    Returns
    -------
    bool
        ``True`` if an extension registered the data, ``False`` if the caller
        should fall back to the workflow's own inputs.
    """
    spec = os.environ.get(ENV_FLOW_BASED_DATA_PROVIDER)
    if not spec:
        return False

    return bool(
        _load_callable(spec)(
            network=network,
            config=config,
            climate_year=climate_year,
            target_year=target_year,
            eraa_year=eraa_year,
        )
    )
