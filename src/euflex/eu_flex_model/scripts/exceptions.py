"""Custom exceptions and helpers for EU-Flex workflow."""

from __future__ import annotations

import logging
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

_IIS_SOLVERS = {"gurobi", "xpress"}


def _build_xpress_row_to_label(model: Any) -> dict[int, int]:
    """
    Build the correct LP-row-position → linopy-label map for an Xpress model.

    linopy writes one LP row per non-masked constraint entry, ordered by
    constraint-name insertion order and then sorted by label value within each
    constraint.  Masked entries (label == -1) are skipped.  Xpress then
    numbers these rows sequentially from 0.

    ``linopy._compute_infeasibilities_xpress`` uses ``enumerate`` over the
    Xpress constraint list and treats the enumerate index as a linopy label,
    which is wrong whenever any labels are masked.  This function reconstructs
    the true row-position → linopy-label mapping to allow correct look-ups.

    Parameters
    ----------
    model : linopy.Model
        The linopy model (``n.model``) whose LP was passed to Xpress.

    Returns
    -------
    dict[int, int]
        Maps each Xpress row position (0-based) to the corresponding linopy
        constraint label.
    """
    mapping: dict[int, int] = {}
    row = 0
    for name in model.constraints:
        labels_flat = model.constraints[name].data.labels.values.flatten()
        for lbl in sorted(int(l) for l in labels_flat if int(l) >= 0):
            mapping[row] = lbl
            row += 1
    return mapping


def _print_xpress_iis(model: Any, log: logging.Logger) -> None:
    """
    Print an IIS report for an Xpress-solved linopy model using correct labels.

    Works around a bug in ``linopy._compute_infeasibilities_xpress`` where
    Xpress row positions are used as linopy constraint labels, producing
    incorrect constraint names when masked rows are present in the model.

    Parameters
    ----------
    model : linopy.Model
    log : logging.Logger
    """
    solver_model = getattr(model, "solver_model", None)
    if solver_model is None:
        log.debug("No Xpress solver_model available.")
        return

    # Compute IIS – IISAll also prints the summary table to stdout.
    iis_summary_buf = StringIO()
    try:
        with redirect_stdout(iis_summary_buf), redirect_stderr(iis_summary_buf):
            try:
                solver_model.IISAll()
            except AttributeError:
                solver_model.iisall()
    except Exception as exc:
        log.warning("IISAll() failed: %s", exc)
        return

    summary = iis_summary_buf.getvalue().strip()
    if summary:
        log.error("Xpress IIS summary:\n%s", summary)

    num_iis = getattr(getattr(solver_model, "attributes", None), "numiis", 0)
    if num_iis == 0:
        log.info("No IIS found by Xpress.")
        return

    row_to_label = _build_xpress_row_to_label(model)

    all_labels: set[int] = set()
    for iis_num in range(1, num_iis + 1):
        try:
            row_indices, *_ = solver_model.getIISData(iis_num)
        except AttributeError:
            # Old API: getiisdata fills lists in-place with constraint objects.
            row_objects: list[Any] = []
            solver_model.getiisdata(iis_num, row_objects, [], [], [], [], [], [], [])
            all_cons = list(solver_model.getConstraint())
            con_to_pos = {c: i for i, c in enumerate(all_cons)}
            row_indices = [con_to_pos[c] for c in row_objects if c in con_to_pos]

        for row_idx in row_indices:
            lbl = row_to_label.get(int(row_idx))
            if lbl is not None:
                all_labels.add(lbl)

    if not all_labels:
        log.warning("IIS found but no linopy constraint labels could be mapped.")
        return

    constraint_lines_buf = StringIO()
    try:
        with (
            redirect_stdout(constraint_lines_buf),
            redirect_stderr(constraint_lines_buf),
        ):
            model.constraints.print_labels(sorted(all_labels))
    except Exception as exc:
        log.warning("Failed to print IIS constraint labels: %s", exc)
        return

    constraint_text = constraint_lines_buf.getvalue().strip()
    if constraint_text:
        log.error("Solver xpress IIS constraints:\n%s", constraint_text)


class InfeasibilityError(Exception):
    """
    Represent infeasibility detected during workflow optimisation.

    Parameters
    ----------
    message : str, optional
        Custom error message. A descriptive message is generated when omitted.
    threshold : float, optional
        Threshold that was exceeded (0-1) for aggregated infeasibility checks.
    combined_rate : float, optional
        Combined infeasibility rate (0-1) observed across models.
    models : dict, optional
        Per-model statistics captured by the infeasibility check.
    report_path : str, optional
        Path to a detailed infeasibility report on disk.
    source : str, optional
        Name of the workflow step that triggered the infeasibility.
    details : dict, optional
        Additional context about the failure (e.g., solver status, snapshots).
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        threshold: float | None = None,
        combined_rate: float | None = None,
        models: dict[str, Any] | None = None,
        report_path: str | None = None,
        source: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.threshold = threshold
        self.combined_rate = combined_rate
        self.models = models or {}
        self.report_path = report_path
        self.source = source
        self.details = details or {}

        if message is None:
            if threshold is not None and combined_rate is not None:
                message = (
                    "Infeasibility threshold exceeded: "
                    f"{combined_rate:.1%} > {threshold:.1%}."
                )
                if report_path:
                    message += f" See report at {report_path}."
            elif source:
                message = f"Optimization infeasible in {source}."
            else:
                message = "Optimization infeasible."

        super().__init__(message)


def maybe_print_solver_infeasibilities(
    model: Any | None,
    solver_name: str | None,
    logger: logging.Logger | None = None,
) -> None:
    """Print an IIS report when supported by the active solver."""

    if model is None or not solver_name:
        return

    solver_key = solver_name.lower()
    if solver_key not in _IIS_SOLVERS:
        return

    log = logger or logging.getLogger(__name__)

    # For Gurobi, if status is "infeasible_or_unbounded", we need to re-solve
    # with DualReductions=0 to determine which one it is before computing IIS
    if solver_key == "gurobi":
        try:
            # Check if we have a gurobi model
            if hasattr(model, "solver_model"):
                gurobi_model = model.solver_model
                if hasattr(gurobi_model, "Status"):
                    # Status 4 = INF_OR_UNBD
                    if gurobi_model.Status == 4:
                        log.info(
                            "Model is infeasible_or_unbounded, re-solving with DualReductions=0 to determine which..."
                        )
                        gurobi_model.setParam("DualReductions", 0)
                        gurobi_model.optimize()
                        log.info(f"After re-solve: Status = {gurobi_model.Status}")

                    # Status 3 = INFEASIBLE - compute IIS
                    if gurobi_model.Status == 3:
                        log.info("Computing IIS for infeasible model...")
                        gurobi_model.computeIIS()
                        log.info("IIS computed successfully")
        except Exception as exc:
            log.warning(f"Failed to re-solve infeasible_or_unbounded model: {exc}")

    # Xpress: use our corrected IIS reporter that properly maps LP row positions
    # to linopy constraint labels (linopy bug: _compute_infeasibilities_xpress uses
    # enumerate position as label, which is wrong when masked rows are skipped in the LP).
    if solver_key == "xpress":
        _print_xpress_iis(model, log)
        return

    buffer = StringIO()

    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            model.print_infeasibilities()
    except AttributeError:
        log.debug(
            "Solver %s supports infeasibility reports but print_infeasibilities() is unavailable.",
            solver_name,
        )
        return
    except Exception as exc:  # pragma: no cover - unexpected solver failure
        log.warning(
            "Failed to capture infeasibility report from solver %s: %s",
            solver_name,
            exc,
        )
        return

    report = buffer.getvalue().strip()
    if not report:
        log.debug("Solver %s returned an empty infeasibility report.", solver_name)
        return

    cleaned_lines: list[str] = []
    prefixes = ("ERROR:", "WARNING:", "INFO:", "DEBUG:")

    for raw_line in report.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        lowered = line.lower()
        cleaned = line
        for prefix in prefixes:
            if lowered.startswith(prefix.lower()):
                parts = line.split(":", maxsplit=2)
                cleaned = parts[-1].strip() if len(parts) > 1 else ""
                break

        cleaned_lines.append(cleaned)

    formatted_report = "\n".join(cleaned_lines)
    log.error("Solver %s IIS report:\n%s", solver_name, formatted_report)
