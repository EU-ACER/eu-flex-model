#!/usr/bin/env python3
"""Check infeasibility rates from market model status logs."""

import logging
from pathlib import Path

import pandas as pd
import yaml

from euflex.eu_flex_model.scripts.exceptions import InfeasibilityError
from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
    set_scenario_config,
)

logger = logging.getLogger(__name__)


def check_model_infeasibilities(
    status_csv_path: Path,
    model_name: str,
    threshold: float = 0.1,
) -> dict:
    """
    Check infeasibilities in a market model status CSV.

    Parameters
    ----------
    status_csv_path : Path
        Path to the status CSV file
    model_name : str
        Name of the model (e.g., "day_ahead", "intraday")
    threshold : float
        Maximum allowed fraction of infeasible horizons

    Returns
    -------
    dict
        Dictionary with infeasibility statistics
    """
    # Check if file exists
    if not status_csv_path.exists():
        return {
            "model": model_name,
            "total_horizons": 0,
            "infeasible_count": 0,
            "infeasible_rate": 0.0,
            "exceeds_threshold": False,
            "infeasible_horizons": [],
            "status": "missing",
        }

    df = pd.read_csv(status_csv_path)

    total_horizons = len(df)
    infeasible_count = int((df["status"] != "ok").sum())
    infeasible_rate = float(
        infeasible_count / total_horizons if total_horizons > 0 else 0.0
    )

    # Get details of infeasible horizons
    infeasible_horizons = df[df["status"] != "ok"][
        ["horizon_index", "snapshot_start", "snapshot_end", "status", "condition"]
    ].to_dict("records")

    return {
        "model": model_name,
        "total_horizons": total_horizons,
        "infeasible_count": infeasible_count,
        "infeasible_rate": infeasible_rate,
        "exceeds_threshold": bool(infeasible_rate > threshold),
        "infeasible_horizons": infeasible_horizons,
        "status": "checked",
    }


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="check_infeasibilities",
    )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Parse Snakemake inputs and parameters
    threshold = snakemake.params.infeasibility_threshold
    fail_on_infeasibility = snakemake.params.fail_on_infeasibility

    output_report = Path(
        snakemake.output.report  # ty: ignore[possibly-missing-attribute]
    )

    logger.info(f"Checking infeasibilities with threshold: {threshold}")
    logger.info(f"Fail on infeasibility: {fail_on_infeasibility}")

    # Check both market models
    results = {
        "threshold": threshold,
        "fail_on_infeasibility": fail_on_infeasibility,
        "models": {},
    }

    # Check if perfect foresight mode (only market_status input)
    perfect_foresight = snakemake.config["eva"]["perfect_foresight"]["enable"]

    if perfect_foresight:
        # Perfect foresight mode: only one market model status
        market_status = Path(
            snakemake.input.market_status  # ty: ignore[possibly-missing-attribute]
        )
        market_result = check_model_infeasibilities(market_status, "market", threshold)
        results["models"]["market"] = market_result
        logger.info(
            f"Market: {market_result['infeasible_count']}/{market_result['total_horizons']} "
            f"infeasible ({market_result['infeasible_rate']:.1%})"
        )
    else:
        # Standard EVA mode: separate day-ahead and intraday models
        # In clustered iterations the input may be a list of centroid status CSVs;
        # concatenate them into a single combined path before checking.

        def resolve_status_input(raw_input, label: str):
            """Return a single Path to a (possibly concatenated) status CSV, or None."""
            if raw_input is None:
                return None
            paths = [raw_input] if isinstance(raw_input, str) else list(raw_input)
            existing = [p for p in paths if Path(p).exists()]
            if not existing:
                return None
            if len(existing) == 1:
                return Path(existing[0])
            # Multiple centroid CSVs: concatenate into a temp file
            import tempfile

            frames = [pd.read_csv(p) for p in existing]
            combined = pd.concat(frames, ignore_index=True)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{label}_combined.csv", delete=False
            )
            combined.to_csv(tmp.name, index=False)
            tmp.close()
            logger.info(
                f"{label}: concatenated {len(existing)} status CSVs -> {tmp.name}"
            )
            return Path(tmp.name)

        def zero_infeasibility_result(model_name: str) -> dict:
            """Return a zero-count result for a model with no status files."""
            return {
                "model": model_name,
                "total_horizons": 0,
                "infeasible_count": 0,
                "infeasible_rate": 0.0,
                "exceeds_threshold": False,
                "infeasible_horizons": [],
                "status": "missing",
            }

        da_status_raw = (
            snakemake.input.day_ahead_status
        )  # ty: ignore[possibly-missing-attribute]
        id_status_raw = (
            snakemake.input.intraday_status
        )  # ty: ignore[possibly-missing-attribute]

        day_ahead_status = resolve_status_input(da_status_raw, "da")
        intraday_status = resolve_status_input(id_status_raw, "id")

        # Check day-ahead model
        if day_ahead_status is None:
            da_result = zero_infeasibility_result("day_ahead")
            logger.info(
                "Day-ahead: no centroid status files for this (ty, cy) -- skipping"
            )
        else:
            da_result = check_model_infeasibilities(
                day_ahead_status, "day_ahead", threshold
            )
            logger.info(
                f"Day-ahead: {da_result['infeasible_count']}/{da_result['total_horizons']} "
                f"infeasible ({da_result['infeasible_rate']:.1%})"
            )
        results["models"]["day_ahead"] = da_result

        # Check intraday model
        if intraday_status is None:
            id_result = zero_infeasibility_result("intraday")
            logger.info(
                "Intraday: no centroid status files for this (ty, cy) -- skipping"
            )
        else:
            id_result = check_model_infeasibilities(
                intraday_status, "intraday", threshold
            )
            logger.info(
                f"Intraday: {id_result['infeasible_count']}/{id_result['total_horizons']} "
                f"infeasible ({id_result['infeasible_rate']:.1%})"
            )
        results["models"]["intraday"] = id_result

    # Overall assessment
    any_exceeded = any(
        model["exceeds_threshold"] for model in results["models"].values()
    )
    results["any_threshold_exceeded"] = any_exceeded

    # Calculate combined statistics
    total_horizons = sum(m["total_horizons"] for m in results["models"].values())
    total_infeasible = sum(m["infeasible_count"] for m in results["models"].values())
    combined_rate = total_infeasible / total_horizons if total_horizons > 0 else 0.0

    results["combined"] = {
        "total_horizons": total_horizons,
        "infeasible_count": total_infeasible,
        "infeasible_rate": float(combined_rate),
    }

    # Write report
    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Infeasibility report written to {output_report}")

    # Log summary
    if any_exceeded:
        logger.error(
            f"INFEASIBILITY THRESHOLD EXCEEDED! "
            f"Combined rate: {combined_rate:.1%} "
            f"(threshold: {threshold:.1%})"
        )
        if fail_on_infeasibility:
            raise InfeasibilityError(
                threshold=threshold,
                combined_rate=combined_rate,
                models=results["models"],
                report_path=str(output_report.resolve()),
            )
    else:
        logger.info(
            f"Infeasibility check passed. Combined rate: {combined_rate:.1%} "
            f"(threshold: {threshold:.1%})"
        )
