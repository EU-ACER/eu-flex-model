"""
Generic, non-database data helpers.

Pure data-shape utilities with no database dependency, shared by both the
database-backed and Excel/CSV-based data processing paths.
These functions are extracted here so that scripts like calculate_model_outputs.py
can use CSV-based data loading and filtering without importing SQL-specific classes.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from euflex.eu_flex_model.scripts.paths import get_path_config

logger = logging.getLogger(__name__)


class DateFormat(str, Enum):
    """
    Enumeration of ALL date format strings found across the EU-Flex Platform codebase.

    This enumeration catalogs every date formatting pattern currently used throughout
    the codebase. It serves as:
    1. A comprehensive reference of existing date format conventions
    2. Documentation of where each format is used
    3. A foundation for future date format standardization efforts

    The formats are organized by category and documented with their current usage locations.

    Attributes
    ----------
    ISO_DATETIME : str
        ISO 8601 datetime format: "2026-07-22T14:30:00"
        Usage: datetime.now().isoformat() - no format string needed
    ISO_DATETIME_SECONDS : str
        ISO 8601 datetime with explicit seconds precision: "2026-07-22T14:30:00"
        Format string: datetime.now().isoformat(timespec="seconds")
    ISO_DATETIME_UTC : str
        ISO 8601 datetime with UTC 'Z' suffix: "2026-07-22T14:30:00Z"
        Format string: "%Y-%m-%dT%H:%M:%SZ"
    DISPLAY_DATETIME_FULL : str
        Human-readable datetime with seconds: "2026-07-22 14:30:00"
        Format string: "%Y-%m-%d %H:%M:%S"
    DISPLAY_DATETIME_SHORT : str
        Human-readable datetime without seconds: "2026-07-22 14:30"
        Format string: "%Y-%m-%d %H:%M"
    COMPACT_DATE : str
        Compact date without separators: "20260722"
        Format string: "%Y%m%d"
    COMPACT_DATETIME : str
        Compact datetime without separators: "20260722143000"
        Format string: "%Y%m%d%H%M%S"
    CSV_EXPORT_DATETIME : str
        European-style datetime for CSV exports: "22/07/2026 14:30:00"
        Format string: "%d/%m/%Y %H:%M:%S"
    LOG_TIMESTAMP_WEEKDAY : str
        Log timestamp with weekday name: "Mon Jan 01 14:30:00 2026"
        Format string: "%a %b %d %H:%M:%S %Y"
        Usage: Parsing Snakemake log timestamps
    ISO_DATE : str
        ISO 8601 date only: "2026-07-22"
        Format string: "%Y-%m-%d"
    PLOT_DAY_MONTH : str
        Day-month for plot axes: "15-01"
        Format string: "%d-%m"
    PLOT_DAY_MONTH_TIME : str
        Day-month with time for plot axes: "15-01 14:30"
        Format string: "%d-%m %H:%M"
    PLOT_MONTH_DAY : str
        Month-day for plot axes: "01-15"
        Format string: "%m-%d"
    PLOT_MONTH_DAY_TIME : str
        Month-day with time for plot axes: "01-15 14:30"
        Format string: "%m-%d %H:%M"
    PLOT_DAY_ABBR_MONTH : str
        Day and abbreviated month for plot axes: "15 Jan"
        Format string: "%d %b"
    PLOT_DAY_ABBR_MONTH_YEAR : str
        Day, abbreviated month and year for plot axes: "15 Jan 2026"
        Format string: "%d %b %Y"
    """

    # ISO 8601 formats (for storage, APIs, logging)
    ISO_DATETIME = "iso"  # Use .isoformat() method directly
    ISO_DATETIME_SECONDS = "iso_seconds"  # Use .isoformat(timespec="seconds")
    ISO_DATETIME_UTC = "%Y-%m-%dT%H:%M:%SZ"
    ISO_DATE = "%Y-%m-%d"

    # Display formats (for user-facing output)
    DISPLAY_DATETIME_FULL = "%Y-%m-%d %H:%M:%S"
    DISPLAY_DATETIME_SHORT = "%Y-%m-%d %H:%M"

    # Compact formats (for database IDs, file naming)
    COMPACT_DATE = "%Y%m%d"
    COMPACT_DATETIME = "%Y%m%d%H%M%S"

    # Specialized formats
    CSV_EXPORT_DATETIME = "%d/%m/%Y %H:%M:%S"
    LOG_TIMESTAMP_WEEKDAY = "%a %b %d %H:%M:%S %Y"  # For parsing Snakemake logs

    # Plotting/visualization formats (for matplotlib axis labels)
    PLOT_DAY_MONTH = "%d-%m"  # e.g., "15-01"
    PLOT_DAY_MONTH_TIME = "%d-%m %H:%M"  # e.g., "15-01 14:30"
    PLOT_MONTH_DAY = "%m-%d"  # e.g., "01-15"
    PLOT_MONTH_DAY_TIME = "%m-%d %H:%M"  # e.g., "01-15 14:30"
    PLOT_DAY_ABBR_MONTH = "%d %b"  # e.g., "15 Jan"
    PLOT_DAY_ABBR_MONTH_YEAR = "%d %b %Y"  # e.g., "15 Jan 2026"

    @staticmethod
    def format_duration(duration_seconds: int | str | None) -> str:
        """
        Format duration in seconds to HH:MM:SS format for display.

        Parameters
        ----------
        duration_seconds : int | str | None
            Duration in seconds (can be int, string representation of int, or None)

        Returns
        -------
        str
            Formatted duration as HH:MM:SS, or "-" if not available

        Examples
        --------
        >>> DateFormat.format_duration(3665)
        '01:01:05'
        >>> DateFormat.format_duration(90)
        '00:01:30'
        >>> DateFormat.format_duration(None)
        '-'
        """
        if duration_seconds is None:
            return "-"

        try:
            total_seconds = int(duration_seconds)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except (ValueError, TypeError):
            return "-"


def drop_feb_29(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Drop Feb 29 entries from a DatetimeIndex.

    Network snapshots and any timeseries derived from or aligned to them are
    kept leap-day-free (e.g. 8760 hours for an hourly full year) so they stay
    alignable across years regardless of whether the source year is a leap
    year.
    """
    return index[~((index.month == 2) & (index.day == 29))]


def climate_year_to_ws_number(climate_year: int) -> str:
    """
    Convert climate year to WS number for capacity_factors table.

    Parameters
    ----------
    climate_year : int
        Climate year (1982-2019)

    Returns
    -------
    str
        WS number (e.g., 2013 -> "WS32")

    Raises
    ------
    ValueError
        If climate year is outside valid range
    """
    if climate_year < 1982 or climate_year > 2019:
        raise ValueError(
            f"Climate year {climate_year} not supported. Must be 1982-2019"
        )
    return f"WS{climate_year - 1981}"


def create_datetime_from_eraa(
    df: pd.DataFrame, climate_year: int | None = None
) -> pd.DataFrame:
    """
    Create datetime snapshot column from ERAA date_id and hour_id columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date_id' and 'hour_id' columns in ERAA format.
    climate_year : int, optional
        Climate year to use for datetime conversion. If provided, replaces the year
        in date_id with climate_year for proper timeseries alignment.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'snapshot' datetime column using climate year.
    """
    df = df.copy()

    # Handle potential NULL values and convert to proper format
    if df["date_id"].isnull().any() or df["hour_id"].isnull().any():
        logger.warning(
            "Found NULL values in date_id or hour_id columns. Dropping rows with NULL timestamps."
        )
        df = df.dropna(subset=["date_id", "hour_id"])

    if len(df) == 0:
        logger.warning("No valid timestamp data remaining after dropping NULL values.")
        df["snapshot"] = pd.Series([], dtype="datetime64[ns]")
        return df

    # Convert to integers first to handle decimal values
    try:
        date_id_clean = df["date_id"].astype(float).astype(int).astype(str)
        hour_id_clean = df["hour_id"].astype(float).astype(int).astype(str).str.zfill(6)

        # If climate_year is provided, replace the year in date_id with climate_year
        if climate_year is not None:
            # Extract month and day from original date_id (format: YYYYMMDD)
            month_day = date_id_clean.str[-4:]  # Last 4 digits (MMDD)

            # Always drop Feb 29 so every climate year produces a series of
            # consistent length, and so a leap target year never creates a
            # Feb 29 date that fails to parse against a non-leap climate year
            feb29 = month_day == "0229"
            if feb29.any():
                logger.warning(
                    f"Dropping {int(feb29.sum())} rows dated Feb 29 for "
                    f"climate year {climate_year} (Feb 29 is always excluded)"
                )
                df = df.loc[~feb29.values]
                date_id_clean = date_id_clean[~feb29]
                hour_id_clean = hour_id_clean[~feb29]
                month_day = month_day[~feb29]

            # Create new date_id with climate_year
            date_id_clean = str(climate_year) + month_day

        # Combine date and hour into datetime string
        datetime_str = date_id_clean + hour_id_clean

        # Convert to datetime
        df["snapshot"] = pd.to_datetime(
            datetime_str, format=DateFormat.COMPACT_DATETIME.value
        )

    except (ValueError, TypeError) as e:
        # A positional fallback would silently misalign every timeseries
        # (e.g. broadcasting a single weekly value across the whole year),
        # so fail loudly instead
        logger.error(f"Error parsing datetime from date_id and hour_id: {e}")
        raise ValueError(
            "Failed to parse datetimes from date_id/hour_id columns; "
            "check the source data for invalid dates."
        ) from e

    return df


def load_csv_with_fallback(csv_path: str | None) -> pd.DataFrame:
    """
    Load a CSV file, resolving a relative path against the input directory.

    Parameters
    ----------
    csv_path : str | None
        Path to CSV file (relative or absolute). If None, returns empty DataFrame.

    Returns
    -------
    pd.DataFrame
        Loaded CSV data, or empty DataFrame if file not found or csv_path is None.
    """
    if not csv_path:
        return pd.DataFrame()

    filepath = Path(csv_path)
    if not filepath.is_absolute():
        # Against the input directory, which is where data/ lives, rather than
        # against this file's position in the package: the package sits at a
        # different depth in the public repository than it does in the
        # platform one, and the working directory need not be the checkout.
        filepath = get_path_config().input_dir / filepath

    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read CSV file '{filepath}': {e}")
        return pd.DataFrame()


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """
    Apply multiple column filters to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    filters : dict[str, Any]
        Dictionary mapping column names to filter values.
        - Scalar values: exact match
        - Lists: membership check (df[col].isin(values))

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    df = df.copy()
    for col, value in filters.items():
        if isinstance(value, list):
            df = df[df[col].isin(value)]
        else:
            df = df[df[col] == value]
    return df


def normalize_value_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'value' column to numeric, handling 'NULL' strings and coercion errors.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'value' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with numeric 'value' column; rows with NaN values removed.
    """
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[df["value"].notna()]


_NECP_REQUIRED_COLS = ["country_short", "variable2", "value", "unit"]
_CO2_REQUIRED_COLS = ["country_short", "value", "unit"]


def load_necp_targets(csv_path: str | None, target_year: int) -> pd.DataFrame:
    """
    Load and filter NECP RES generation targets from a CSV file.

    Parameters
    ----------
    csv_path : str | None
        Path to the NECP targets CSV file.
    target_year : int
        Target year to filter records by.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["country_short", "variable2", "value", "unit"]``,
        or an empty DataFrame with those columns if no data is found.
    """
    df = load_csv_with_fallback(csv_path)
    if not df.empty:
        df = apply_filters(
            df,
            {
                "year": target_year,
                "kpi": "Electricity generation",
                "variable1": "Renewable",
                "variable2": ["Wind", "Solar", "Hydro", "Bio", "Other Renewable"],
                "variable3": "Total",
            },
        )
        df = normalize_value_column(df)
        if not df.empty:
            df = df[_NECP_REQUIRED_COLS].reset_index(drop=True)
    if df.empty:
        df = pd.DataFrame(columns=pd.Index(_NECP_REQUIRED_COLS))
    return df


def load_co2_targets(csv_path: str | None, target_year: int) -> pd.DataFrame:
    """
    Load and filter CO2 emission targets from a CSV file.

    Parameters
    ----------
    csv_path : str | None
        Path to the CO2 targets CSV file.
    target_year : int
        Target year to filter records by.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["country_short", "value", "unit"]``,
        or an empty DataFrame with those columns if no data is found.
    """
    df = load_csv_with_fallback(csv_path)
    if not df.empty:
        df = apply_filters(
            df,
            {
                "category": "Emissions",
                "kpi": "GHG emissions",
                "sector": "Total",
                "variable1": "Total",
            },
        )
        df = normalize_value_column(df)
        if not df.empty and "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year"].notna()].copy()
            df["year"] = df["year"].astype(int)
            df = df[df["year"] == target_year]
        if not df.empty:
            df = df[_CO2_REQUIRED_COLS].reset_index(drop=True)
    if df.empty:
        df = pd.DataFrame(columns=pd.Index(_CO2_REQUIRED_COLS))
    return df


def decommissionable_by_carrier(
    carrier: pd.Series, config: dict[str, Any], component_type: str
) -> pd.Series:
    """
    Flag units whose carrier is listed in ``eva.base_network.decommissioned_carriers``.

    Parameters
    ----------
    carrier : pd.Series
        Carrier of each unit.
    config : dict
        Configuration dictionary containing ``eva.base_network.decommissioned_carriers``.
    component_type : str
        Either "Generator" or "StorageUnit".

    Returns
    -------
    pd.Series
        Boolean mask, True for units with a decommissioned carrier.
    """
    decommissioned = config["eva"]["base_network"]["decommissioned_carriers"]
    return carrier.isin(decommissioned.get(component_type, []))


def existing_capacity_bounds(
    p_nom: pd.Series, decommissionable: pd.Series
) -> pd.DataFrame:
    """
    Capacity bounds of existing units.

    Decommissionable units may be reduced down to zero (``p_nom_min = 0``), all
    other existing units are fixed at ``p_nom``. No existing unit may grow
    beyond ``p_nom``.

    Parameters
    ----------
    p_nom : pd.Series
        Installed capacity of each existing unit.
    decommissionable : pd.Series
        Boolean mask of decommissionable units.

    Returns
    -------
    pd.DataFrame
        Columns ``p_nom_decommissionable``, ``p_nom_min`` and ``p_nom_max``.
    """
    return pd.DataFrame(
        {
            "p_nom_decommissionable": decommissionable,
            "p_nom_min": p_nom.where(~decommissionable, 0.0),
            "p_nom_max": p_nom,
        }
    )
