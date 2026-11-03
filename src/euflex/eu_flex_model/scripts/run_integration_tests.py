"""
Run integration tests for EVA workflow validation.

This script performs automated tests on day-ahead and intraday network results
to validate the EVA workflow execution. It checks for:
1. Market state of charge within pre-optimization limits
2. Intraday dispatch within day-ahead commitment limits
3. Price-flow consistency (power flows from lower to higher price zones)
"""

import json
import logging
import pathlib
from typing import Any

import pandas as pd
import pypsa

from euflex.eu_flex_model.scripts.helpers.logging_helpers import (
    configure_logging,
)
from euflex.eu_flex_model.scripts.helpers.snakemake_helpers import (
    get_snakemake,
    set_scenario_config,
)

logger = logging.getLogger(__name__)


def assert_soc_within_preopt_limits(
    n: pypsa.Network, storage_plans: pd.DataFrame, tolerance: float = 0.1
) -> dict[str, list[tuple[str, pd.Timestamp, float, float]]]:
    """
    Assert that market state of charge stays within pre-optimization limits.

    Only checks at rolling horizon boundaries (i.e., every horizon - overlap hours),
    excluding the initial timestamp (t=0) which is the initial condition.
    With default horizon=72 and overlap=24, this checks at t=48h, 96h, 144h, etc.

    Parameters
    ----------
    n : pypsa.Network
        The solved network (day-ahead or intraday).
    storage_plans : pd.DataFrame
        The pre-optimization storage plan with SOC set points.
    tolerance : float
        Relative tolerance for comparison (default 0.1 = 10%).

    Returns
    -------
    dict[str, list[tuple[str, pd.Timestamp, float, float]]]
        Dictionary mapping storage units to list of violations.
        Each violation is a tuple of (unit, timestamp, actual_soc, limit).
    """
    violations = {}

    # Get rolling horizon configuration from network meta
    horizon = n.meta.get("solving", {}).get("options", {}).get("horizon", 72)
    overlap = n.meta.get("solving", {}).get("options", {}).get("overlap", 24)
    boundary_interval = horizon - overlap  # Default: 72 - 24 = 48 hours

    for storage_unit in n.storage_units.index:
        if storage_unit not in storage_plans.columns:
            continue

        # Get actual SOC from network
        actual_soc = n.storage_units_t.state_of_charge.get(storage_unit)
        if actual_soc is None:
            continue

        # Get pre-opt set points
        preopt_soc = storage_plans[storage_unit]

        # Only check at rolling horizon boundaries
        # Get all set point times and filter to boundary intervals
        set_point_times = preopt_soc.dropna().index

        # Find timestamps that align with rolling horizon boundaries
        # Start from the first timestamp and check every boundary_interval hours
        # Skip the very first timestamp (t=0) as it's the initial condition
        if len(set_point_times) > 0:
            first_time = set_point_times[0]
            boundary_times = []
            for t in set_point_times:
                hours_diff = (t - first_time).total_seconds() / 3600
                # Check if this is at a boundary (48, 96, 144, etc.)
                # Skip t=0 (initial condition)
                if hours_diff > 0 and hours_diff % boundary_interval == 0:
                    boundary_times.append(t)
        else:
            boundary_times = []

        unit_violations = []
        for t in boundary_times:
            if t in actual_soc.index:
                actual = actual_soc.loc[t]
                limit = preopt_soc.loc[t]

                # Check if actual SOC is below the minimum limit with relative tolerance
                # The pre-opt SOC acts as a minimum requirement (lower bound)
                if actual < limit * (1 - tolerance):
                    unit_violations.append((storage_unit, t, actual, limit))

        if unit_violations:
            violations[storage_unit] = unit_violations

    return violations


def assert_intraday_dispatch_within_dayahead_limits(
    n_da: pypsa.Network, n_id: pypsa.Network, tolerance: float = 0.1
) -> dict[str, list[tuple[pd.Timestamp, float, float, float]]]:
    """
    Assert that intraday dispatch stays within day-ahead commitment limits.

    Parameters
    ----------
    n_da : pypsa.Network
        The day-ahead network.
    n_id : pypsa.Network
        The intraday network.
    tolerance : float
        Tolerance for comparison (default 0.1 = 10% in per-unit).

    Returns
    -------
    dict[str, list[tuple[pd.Timestamp, float, float, float]]]
        Dictionary mapping generators to list of violations.
        Each violation is a tuple of (timestamp, dispatch_pu, min_limit, max_limit).
    """
    violations = {}

    # Get committable generators
    committable_gens = n_da.generators.index[n_da.generators.committable]

    for gen in committable_gens:
        if gen not in n_id.generators.index:
            continue

        # Get p_nom for normalization
        p_nom = n_id.generators.p_nom.get(gen, 0)
        if p_nom == 0:
            continue

        # Get intraday dispatch in per-unit
        id_dispatch = n_id.generators_t.p.get(gen)
        if id_dispatch is None:
            continue
        id_dispatch_pu = id_dispatch / p_nom

        # Get intraday limits (these come from day-ahead commitments)
        p_min_pu = n_id.get_switchable_as_dense("Generator", "p_min_pu")[gen]
        p_max_pu = n_id.get_switchable_as_dense("Generator", "p_max_pu")[gen]

        # Find violations
        gen_violations = []
        for t in id_dispatch.index:
            dispatch = id_dispatch_pu.loc[t]
            min_limit = p_min_pu.loc[t]
            max_limit = p_max_pu.loc[t]

            # Only check violations if limits are actually constraining (non-zero range)
            # Skip if the limits define a feasible range
            if max_limit - min_limit < 1e-2:
                # Limits are essentially equal, skip this constraint
                continue

            # Check if dispatch violates limits with 10% tolerance
            if dispatch < min_limit - tolerance or dispatch > max_limit + tolerance:
                gen_violations.append((t, dispatch, min_limit, max_limit))

        if gen_violations:
            violations[gen] = gen_violations

    return violations


def assert_price_flow_consistency(
    n: pypsa.Network,
    tolerance: float = 0.05,
    absolute_tolerance: float = 1.0,
    flow_threshold: float = 0.1,
) -> list[tuple[pd.Timestamp, str, str, float, float, float]]:
    """
    Assert that power flows from lower price zones to higher price zones.

    Parameters
    ----------
    n : pypsa.Network
        The solved network.
    tolerance : float
        Relative price tolerance (default 0.05 = 5%).
    absolute_tolerance : float
        Absolute price tolerance in €/MWh (default 1.0).
        Used when prices are near zero.
    flow_threshold : float
        Minimum flow in MW to check (default 0.1).
        Flows below this are considered numerical noise.

    Returns
    -------
    list[tuple[pd.Timestamp, str, str, float, float, float]]
        List of violations where each tuple contains:
        (timestamp, from_bus, to_bus, from_price, to_price, flow_MW).
    """
    violations = []

    # Check each link (cross-border connection)
    for link in n.links.index:
        bus0 = n.links.loc[link, "bus0"]
        bus1 = n.links.loc[link, "bus1"]

        # Get flows and prices
        flows = n.links_t.p0.get(link)
        if flows is None:
            continue

        prices0 = n.buses_t.marginal_price.get(bus0)
        prices1 = n.buses_t.marginal_price.get(bus1)

        if prices0 is None or prices1 is None:
            continue

        # Check each timestamp
        for t in flows.index:
            if t not in prices0.index or t not in prices1.index:
                continue

            flow = flows.loc[t]
            price0 = prices0.loc[t]
            price1 = prices1.loc[t]

            # Skip negligible flows (numerical noise)
            if abs(flow) < flow_threshold:
                continue

            # Determine if prices are "effectively equal" using both tolerances
            price_diff = abs(price0 - price1)
            avg_price = (abs(price0) + abs(price1)) / 2

            # Use absolute tolerance when prices are near zero
            if avg_price < absolute_tolerance:
                prices_equal = price_diff < absolute_tolerance
            else:
                # Use relative tolerance for normal price ranges
                prices_equal = price_diff / avg_price < tolerance

            # Skip checking if prices are effectively equal
            if prices_equal:
                continue

            # Positive flow means power flows from bus0 to bus1
            # This should only happen if price0 <= price1 (within tolerance)
            if flow > flow_threshold and price0 > price1 * (1 + tolerance):
                violations.append((t, bus0, bus1, price0, price1, flow))
            # Negative flow means power flows from bus1 to bus0
            # This should only happen if price1 <= price0 (within tolerance)
            elif flow < -flow_threshold and price1 > price0 * (1 + tolerance):
                violations.append((t, bus1, bus0, price1, price0, -flow))

    return violations


def run_integration_tests(
    n_da: pypsa.Network,
    n_id: pypsa.Network,
    storage_plans_da: pd.DataFrame,
    storage_plans_id: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """
    Run all integration tests and return results.

    Parameters
    ----------
    n_da : pypsa.Network
        Day-ahead network.
    n_id : pypsa.Network
        Intraday network.
    storage_plans_da : pd.DataFrame
        Day-ahead storage plans.
    storage_plans_id : pd.DataFrame
        Intraday storage plans.
    config : dict
        Configuration dictionary with test parameters.

    Returns
    -------
    dict
        Test results with violations and statistics.
    """
    results: dict[str, Any] = {
        "passed": True,
        "tests": {},
    }

    # Get tolerances from config
    market_interconnection = config["eva"]["market_interconnection"]
    test_config = config["eva"]["integration_tests"]
    soc_tolerance = test_config["soc_tolerance"]
    dispatch_tolerance = test_config["dispatch_tolerance"]
    price_flow_tolerance = test_config["price_flow_tolerance"]

    # Test 1: Market state of charge within pre-opt limits
    logger.info("Testing market SOC within pre-opt limits...")
    soc_violations_da = assert_soc_within_preopt_limits(
        n_da, storage_plans_da, soc_tolerance
    )
    soc_violations_id = assert_soc_within_preopt_limits(
        n_id, storage_plans_id, soc_tolerance
    )

    results["tests"]["soc_within_limits"] = {
        "name": "SOC Limits",
        "description": "Market state of charge within pre-optimization limits",
        "passed": not (soc_violations_da or soc_violations_id),
        "day_ahead_violations": soc_violations_da,
        "intraday_violations": soc_violations_id,
    }

    if soc_violations_da or soc_violations_id:
        results["passed"] = False

    # Test 2: Intraday dispatch within day-ahead limits
    logger.info("Testing intraday dispatch within day-ahead limits...")
    dispatch_violations = assert_intraday_dispatch_within_dayahead_limits(
        n_da, n_id, dispatch_tolerance
    )

    results["tests"]["dispatch_within_limits"] = {
        "name": "Dispatch Limits",
        "description": "Intraday dispatch within day-ahead commitment limits",
        "passed": not dispatch_violations,
        "violations": dispatch_violations,
    }

    if dispatch_violations:
        results["passed"] = False

    # Test 3: Price-flow consistency, only run if market interconnection is not FBMC
    # (https://www.jao.eu/sites/default/files/2021-05/CWE%20FB%20plain%20Explanatory%20Material_v1.0.pdf)
    if market_interconnection != "fbmc":
        logger.info("Testing price-flow consistency...")
        price_flow_violations_da = assert_price_flow_consistency(
            n_da, price_flow_tolerance
        )
        price_flow_violations_id = assert_price_flow_consistency(
            n_id, price_flow_tolerance
        )

        results["tests"]["price_flow_consistency"] = {
            "name": "Price-Flow Consistency",
            "description": "Power flows from lower to higher price zones",
            "passed": not (price_flow_violations_da or price_flow_violations_id),
            "day_ahead_violations": price_flow_violations_da,
            "intraday_violations": price_flow_violations_id,
        }

        if price_flow_violations_da or price_flow_violations_id:
            results["passed"] = False
    else:
        logger.info(
            "Skipping price-flow consistency test for FBMC market interconnection"
        )
        results["tests"]["price_flow_consistency"] = {
            "name": "Price-Flow Consistency",
            "description": "Power flows from lower to higher price zones",
            "passed": True,
            "day_ahead_violations": {},
            "intraday_violations": {},
        }

    return results


def format_test_report(results: dict) -> str:
    """Format test results into a human-readable report."""
    report = []
    failure_error_message = "✗ FAILED - Violations found:"
    report.append("=" * 80)
    report.append("EVA INTEGRATION TEST REPORT")
    report.append("=" * 80)
    report.append("")

    overall_status = "PASSED" if results["passed"] else "FAILED"
    report.append(f"Overall Status: {overall_status}")
    report.append("")

    # Test 1: SOC within limits
    report.append("1. Market State of Charge Within Pre-optimization Limits")
    report.append("-" * 60)
    soc_test = results["tests"]["soc_within_limits"]

    if soc_test["passed"]:
        report.append("✓ PASSED - All storage units within pre-optimization limits")
    else:
        report.append(failure_error_message)

        if soc_test["day_ahead_violations"]:
            report.append("\n  Day-ahead violations:")
            for unit, violations in soc_test["day_ahead_violations"].items():
                report.append(f"    {unit}:")
                for _, t, actual, limit in violations[:3]:
                    report.append(
                        f"      {t}: actual={actual:.2f} MW > limit={limit:.2f} MW"
                    )
                if len(violations) > 3:
                    report.append(
                        f"      ... and {len(violations) - 3} more violations"
                    )

        if soc_test["intraday_violations"]:
            report.append("\n  Intraday violations:")
            for unit, violations in soc_test["intraday_violations"].items():
                report.append(f"    {unit}:")
                for _, t, actual, limit in violations[:3]:
                    report.append(
                        f"      {t}: actual={actual:.2f} MW > limit={limit:.2f} MW"
                    )
                if len(violations) > 3:
                    report.append(
                        f"      ... and {len(violations) - 3} more violations"
                    )

    report.append("")

    # Test 2: Dispatch within limits
    report.append("2. Intraday Dispatch Within Day-ahead Commitment Limits")
    report.append("-" * 60)
    dispatch_test = results["tests"]["dispatch_within_limits"]

    if dispatch_test["passed"]:
        report.append("✓ PASSED - All generators within commitment limits")
    else:
        report.append(failure_error_message)
        for gen, violations in dispatch_test["violations"].items():
            report.append(f"  {gen}:")
            for t, dispatch, min_lim, max_lim in violations[:3]:
                report.append(
                    f"    {t}: dispatch={dispatch:.4f} pu not in [{min_lim:.4f}, {max_lim:.4f}]"
                )
            if len(violations) > 3:
                report.append(f"    ... and {len(violations) - 3} more violations")

    report.append("")

    # Test 3: Price-flow consistency
    report.append("3. Price-Flow Consistency")
    report.append("-" * 60)
    price_test = results["tests"]["price_flow_consistency"]

    if price_test["passed"]:
        report.append("✓ PASSED - All flows consistent with prices")
    else:
        report.append(failure_error_message)

        if price_test["day_ahead_violations"]:
            report.append("\n  Day-ahead violations:")
            for t, from_bus, to_bus, from_price, to_price, flow in price_test[
                "day_ahead_violations"
            ][:3]:
                report.append(
                    f"    {t}: {from_bus} ({from_price:.2f} €/MWh) -> {to_bus} ({to_price:.2f} €/MWh), flow={flow:.2f} MW"
                )
            if len(price_test["day_ahead_violations"]) > 3:
                report.append(
                    f"    ... and {len(price_test['day_ahead_violations']) - 3} more violations"
                )

        if price_test["intraday_violations"]:
            report.append("\n  Intraday violations:")
            for t, from_bus, to_bus, from_price, to_price, flow in price_test[
                "intraday_violations"
            ][:3]:
                report.append(
                    f"    {t}: {from_bus} ({from_price:.2f} €/MWh) -> {to_bus} ({to_price:.2f} €/MWh), flow={flow:.2f} MW"
                )
            if len(price_test["intraday_violations"]) > 3:
                report.append(
                    f"    ... and {len(price_test['intraday_violations']) - 3} more violations"
                )

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


if __name__ == "__main__":
    snakemake = get_snakemake(
        globals_scope=globals(),
        rule="run_integration_tests",
        configfiles=[
            pathlib.Path("config", "test", "config.eva-regional-test.yaml"),
        ],
        it=1,
        run="eva-low-span",
        cy=2013,
        ty=2025,
    )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load networks and storage plans
    logger.info("Loading networks and storage plans...")
    n_da = pypsa.Network(
        snakemake.input.day_ahead_network  # ty: ignore[possibly-missing-attribute]
    )
    n_id = pypsa.Network(
        snakemake.input.intraday_network  # ty: ignore[possibly-missing-attribute]
    )
    # Storage plan is the same for both day-ahead and intraday (based on forecast)
    storage_plans = pd.read_csv(
        snakemake.input.storage_plans,  # ty: ignore[possibly-missing-attribute]
        index_col=0,
        parse_dates=True,
    )
    storage_plans_da = storage_plans
    storage_plans_id = storage_plans

    # Run tests
    logger.info("Running integration tests...")
    results = run_integration_tests(
        n_da, n_id, storage_plans_da, storage_plans_id, snakemake.config
    )

    # Generate report
    report = format_test_report(results)
    logger.info(f"\n{report}")

    # Save results
    with open(
        snakemake.output.test_results_json,  # ty: ignore[possibly-missing-attribute]
        "w",
    ) as f:
        json.dump(results, f, indent=2, default=str)

    with open(
        snakemake.output.test_report_txt,  # ty: ignore[possibly-missing-attribute]
        "w",
    ) as f:
        f.write(report)

    # Optionally fail the workflow if tests failed
    if (
        not results["passed"]
        and snakemake.config["eva"]["integration_tests"]["fail_on_error"]
    ):
        raise RuntimeError("Integration tests failed. See report for details.")

    # Log final status clearly
    if results["passed"]:
        logger.info("Integration Tests: PASSED ✓")
    else:
        logger.warning("Integration Tests: FAILED ✗")

    logger.info(
        f"Integration test report saved to {snakemake.output.test_report_txt}"  # ty: ignore[possibly-missing-attribute]
    )
