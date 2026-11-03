import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from euflex.eu_flex_model.scripts.extensions import (
    register_flow_based_data_extension,
)
from euflex.eu_flex_model.scripts.helpers.date_helpers import (
    DateFormat,
    climate_year_to_ws_number,
    create_datetime_from_eraa,
    drop_feb_29,
)
from euflex.eu_flex_model.scripts.paths import (
    get_path_config,
    resolve_input_path,
)

logger = logging.getLogger(__name__)


# TODO: the two assignment helpers below can be removed once the pre-builts are updated/removed. The
#  types are read from the database.
def _assign_core_region(n: pypsa.Network) -> None:
    """
    Assign European core region to the network.

    The Core Capacity Calculation Region (Core CCR) is a specific area
    in the European electricity grid that uses Flow-Based Market Coupling
    (FBMC) for capacity calculation. This function assigns the Core CCR
    to the network by setting the `core_region` attribute of the network.


    Parameters
    ----------
    n : pypsa.Network
    """
    # Core countries are Austria, Belgium, Croatia, the Czech Republic,
    # France, Germany, Hungary, the Netherlands, Poland, Romania,
    # Slovakia, and Slovenia
    core_buses = [
        "AT00",
        "BE00",
        "HR00",
        "CZ00",
        "FR00",
        "DE00",
        "HU00",
        "NL00",
        "PL00",
        "RO00",
        "SK00",
        "SI00",
    ]
    is_core = n.buses.index.isin(core_buses)
    n.buses["core_region"] = is_core


def _assign_nordic_region(n: pypsa.Network) -> None:
    """
    Assign Nordic region to the network.

    The Nordic region is defined as a part of the countries Denmark, Finland,
    Norway, and Sweden. This function assigns the Nordic region to the
    network by setting the `nordic_region` attribute of the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object
    """
    nordic_buses = [
        "NOS1",
        "NOS2",
        "NOS3",
        "NOM1",
        "NON1",
        "FI00",
        "SE01",
        "SE02",
        "SE03",
        "SE04",
        "DKE1",
    ]
    is_nordic = n.buses.index.isin(nordic_buses)
    n.buses["nordic_region"] = is_nordic


def _assign_link_type(n: pypsa.Network) -> pypsa.Network:
    """
    Assign the link type. Options are 'core', 'nordic', 'ahc', or 'ntc'.

    - 'core': Links between two core region buses
    - 'nordic': Links between two nordic region buses
    - 'ahc': Links between core and non-core/non-nordic buses, between nordic and non-core/non-nordic buses, or between core and nordic buses (Advanced Hybrid Coupling)
    - 'ntc': Links between two buses outside both nordic and core regions (Net Transfer Capacity)

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object

    Returns
    -------
    pypsa.Network
        The network with link types assigned
    """

    # Get core buses from the network
    core_buses = set(n.buses.index[n.buses.core_region])

    # Get nordic buses from the network
    if "nordic_region" in n.buses.columns:
        nordic_buses = set(n.buses.index[n.buses.nordic_region])
    else:
        nordic_buses = set()

    # Initialize type column if it doesn't exist
    if "type_coupling" not in n.links.columns:
        n.links["type_coupling"] = pd.Series("ntc", index=n.links.index, dtype=str)

    # Assign link types based on bus0 and bus1 region status
    for link_idx in n.links.index:
        bus0 = n.links.loc[link_idx, "bus0"]
        bus1 = n.links.loc[link_idx, "bus1"]

        bus0_is_core = bus0 in core_buses
        bus1_is_core = bus1 in core_buses
        bus0_is_nordic = bus0 in nordic_buses
        bus1_is_nordic = bus1 in nordic_buses

        if bus0_is_core and bus1_is_core:
            # 1) Both buses are in core region
            n.links.at[link_idx, "type_coupling"] = "core"
        elif bus0_is_nordic and bus1_is_nordic:
            # 2) Both buses are in nordic region
            n.links.at[link_idx, "type_coupling"] = "nordic"
        elif (bus0_is_core and not bus1_is_core and not bus1_is_nordic) or (
            bus1_is_core and not bus0_is_core and not bus0_is_nordic
        ):
            # 3) One bus is core, the other is outside both core and nordic
            n.links.at[link_idx, "type_coupling"] = "ahc"
        elif (bus0_is_nordic and not bus1_is_nordic and not bus1_is_core) or (
            bus1_is_nordic and not bus0_is_nordic and not bus0_is_core
        ):
            # 4) One bus is nordic, the other is outside both core and nordic
            n.links.at[link_idx, "type_coupling"] = "ahc"
        elif (bus0_is_core and bus1_is_nordic) or (bus1_is_core and bus0_is_nordic):
            # 6) One bus is core, the other is nordic
            n.links.at[link_idx, "type_coupling"] = "ahc"
        else:
            # 5) Both buses are outside nordic and core regions
            n.links.at[link_idx, "type_coupling"] = "ntc"

    return n


def _adapt_network_for_fbmc(n: pypsa.Network, type_ccr: str = "core") -> pypsa.Network:
    """
    Adapt the network for FBMC by removing links between core BZs and adding
    CORE/NORDIC bus with links from core BZs to CORE/NORDIC bus.

    This function:
    1. Removes all links between core/nordic region buses (type_coupling == 'core'/'nordic')
    2. Creates a virtual CORE/NORDIC bus
    3. Adds links from each core/nordic bus to the CORE/NORDIC bus with infinite capacity

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to adapt

    Returns
    -------
    pypsa.Network
        The adapted network with CORE bus structure
    """
    # Find and remove core-to-core links
    links_to_adapt = n.links[n.links.type_coupling == type_ccr].index.tolist()

    if len(links_to_adapt) > 0:
        logger.info(
            f"Removing {len(links_to_adapt)} {type_ccr}-to-{type_ccr} links: {links_to_adapt}"
        )
        n.links.drop(links_to_adapt, inplace=True)

        # Also remove from time-varying data if it exists
        for attr in ["p_max_pu", "p_min_pu"]:
            if hasattr(n.links_t, attr):
                links_t_attr = getattr(n.links_t, attr)
                for link in links_to_adapt:
                    if link in links_t_attr.columns:
                        links_t_attr.drop(columns=[link], inplace=True)

    # Get core buses
    if type_ccr == "core":
        buses_to_adapt = n.buses[n.buses.core_region].index.tolist()
        is_nordic = False
        is_core = True
    elif type_ccr == "nordic":
        buses_to_adapt = n.buses[n.buses.nordic_region].index.tolist()
        is_nordic = True
        is_core = False
    else:
        logger.error(f"Unknown type_ccr: {type_ccr}. Use 'core' or 'nordic'.")
        return n

    if len(buses_to_adapt) > 0:
        logger.info(
            f"Creating {type_ccr.upper()} bus and connecting {len(buses_to_adapt)} {type_ccr} buses"
        )

        # Add virtual bus - use strategic positioning closer to regional center
        if type_ccr == "core":
            # Position CORE virtual bus in Central Europe (Frankfurt/Main area)
            # This is more representative of the European Core region
            core_bus_x = 8.7  # Frankfurt longitude
            core_bus_y = 50.1  # Frankfurt latitude
        elif type_ccr == "nordic":
            # Position NORDIC virtual bus in Scandinavia (Stockholm area)
            # This better represents the Nordic region
            core_bus_x = 18.1  # Stockholm longitude
            core_bus_y = 59.3  # Stockholm latitude
        else:
            # Fallback to original method for other regions
            core_bus_x = n.buses.loc[buses_to_adapt, "x"].mean()
            core_bus_y = n.buses.loc[buses_to_adapt, "y"].mean()

        if "nordic_region" in n.buses.columns:
            n.add(
                "Bus",
                type_ccr.upper(),
                country=type_ccr.upper(),
                carrier="AC",
                x=core_bus_x,
                y=core_bus_y,
                core_region=is_core,
                nordic_region=is_nordic,
            )
        else:
            n.add(
                "Bus",
                type_ccr.upper(),
                country=type_ccr.upper(),
                carrier="AC",
                x=core_bus_x,
                y=core_bus_y,
                core_region=is_core,
            )

        # Add links from each core bus to CORE bus (bidirectional)
        # Use a very high number instead of infinity for numerical stability
        very_high_capacity = 1e6  # 1 million MW

        for bus in buses_to_adapt:
            # Link from core bus to CORE bus
            link_name = f"{bus}-{type_ccr.upper()}"
            n.add(
                "Link",
                link_name,
                bus0=bus,
                bus1=type_ccr.upper(),
                carrier="AC",
                p_nom=very_high_capacity,
                p_max_pu=1.0,
                p_min_pu=-1.0,
                marginal_cost=0.0,
            )

            # Set type_coupling for the new link
            n.links.loc[link_name, "type_coupling"] = type_ccr

            logger.debug(
                f"Added link {link_name} with capacity {very_high_capacity} MW"
            )

    return n


def _adapt_network_for_evolved_fb(n: pypsa.Network, eraa_year: int) -> pypsa.Network:
    """
    Adapt the network to account for the evolved Flow-Based (FB) structure.

    This function modifies the PyPSA network by introducing evolved FB nodes and
    links between specific countries depending on the specified ERAA year.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object to be adapted.
    eraa_year : int
        The ERAA year to decide the configuration of FB pairs. Supported values are 2023 and 2024.

    Returns
    -------
    pypsa.Network
        The modified PyPSA network with added evolved FB nodes and links.
    """

    # Countries between which the evolved FB is applied
    evolved_fb_pairs = [
        {"bus0": "BE00", "bus1": "DE00", "capacity": 1000, "name": "ALEGRO"}
    ]

    if eraa_year == 2024:
        evolved_fb_pairs += [
            {"bus0": "NON1", "bus1": "FI00", "capacity": 150, "name": "NON1-FI00"},
            {"bus0": "SE03", "bus1": "SE04", "capacity": 1200, "name": "SE03-SE04"},
            {"bus0": "SE03", "bus1": "FI00", "capacity": 1200, "name": "SE03-FI00"},
        ]

    # For each pair, add three virtual nodes,
    # disconnected from the rest of the network
    for pair in evolved_fb_pairs:
        bus0 = pair["bus0"]
        bus1 = pair["bus1"]
        capacity = pair["capacity"]
        name = pair["name"]

        # Check if bu0 and bus1 exist in the network
        if bus0 in n.buses.index and bus1 in n.buses.index:
            logger.info(f"Adding evolved FB nodes and link {name} to the network")

            # Create virtual buses for the evolved FB
            # Position virtual buses in appropriate regional areas
            if (
                "NON1" in name  # ty: ignore[unsupported-operator]
                or "FI00" in name  # ty: ignore[unsupported-operator]
                or "SE" in name  # ty: ignore[unsupported-operator]
            ):
                # Nordic interconnections - position in Nordic area (southern Scandinavia)
                virtual_x = 12.0  # Southern Scandinavia longitude
                virtual_y = 56.0  # Southern Scandinavia latitude
            else:
                # Core region interconnections (like ALEGRO) - position in Central Europe
                virtual_x = 8.0  # Central Europe longitude
                virtual_y = 50.0  # Central Europe latitude

            for i in range(3):
                if eraa_year == 2024:
                    n.add(
                        "Bus",
                        f"{name}_{str(i)}",
                        core_region=False,
                        nordic_region=False,
                        country="virtual",
                        x=virtual_x,
                        y=virtual_y,
                    )
                else:
                    n.add(
                        "Bus",
                        f"{name}_{str(i)}",
                        core_region=False,
                        country="virtual",
                        x=virtual_x,
                        y=virtual_y,
                    )
            # Add three links between the buses
            for i in range(3):
                if i == 0:
                    link_name = name
                else:
                    link_name = f"{name}_{str(i)}-{name}_00{str((i + 1) % 3)}"
                n.add(
                    "Link",
                    link_name,
                    bus0=f"{name}_{str(i)}",
                    bus1=f"{name}_{str((i + 1) % 3)}",
                    p_nom=capacity,
                    p_max_pu=1.0,
                    p_min_pu=-1.0,
                    type_coupling="evfb",
                )
    return n


def adapt_network_for_market_coupling(
    n: pypsa.Network, market_interconnection: str, eraa_year: int
) -> pypsa.Network:
    """Makes the necessary adaptations of the network depending on the market coupling type."""

    # Assign core region to buses
    _assign_core_region(n)

    if eraa_year == 2024:
        _assign_nordic_region(n)

    # Assign link types based on core region
    n = _assign_link_type(n)

    # If market interconnection is FBMC, adapt the network structure
    if market_interconnection == "fbmc":
        n = _adapt_network_for_fbmc(n, "core")
        if eraa_year == 2024:
            n = _adapt_network_for_fbmc(n, "nordic")
        n = _adapt_network_for_evolved_fb(n, eraa_year)

    return n


def register_market_interconnection(
    n: pypsa.Network,
    market_interconnection: str,
    climate_year: int,
    target_year: int,
    config: dict[str, Any],
    eraa_year: int = 2024,  # Default to 2024 if not specified
) -> None:
    """
    Register market interconnection constraints for the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    market_interconnection : str
        The type of market interconnection model to be used. Supported values
        include "fbmc" for flow-based market coupling (with integrated NTCs)
        and "physical" for physical interconnection.
    climate_year : int
        The climate year for which the flow-based data is relevant.
    target_year : int
        The target year for which the flow-based data is relevant.
    eraa_year : int
        ERAA data (year of the study)
    config : dict, Optional
        Configuration options

    Notes
    -----
    Flow-based data comes from the workflow's own Excel/CSV inputs unless an
    extension supplies it instead — the platform loads it from the database
    through the ``EUFLEX_FLOW_BASED_DATA_PROVIDER`` extension point. This
    currently only supports ERAA 2024 and ERAA 2025 data.
    """

    # Add market interconnection constraints
    if market_interconnection == "fbmc":
        setattr(n, "_market_interconnection", "fbmc")
        setattr(n, "_eraa_year", eraa_year)

        registered_by_extension = register_flow_based_data_extension(
            network=n,
            config=config,
            climate_year=int(climate_year),
            target_year=int(target_year),
            eraa_year=int(eraa_year),
        )

        if not registered_by_extension:
            logger.info("Registering flow-based data from workflow inputs")
            register_flow_based_data(
                n=n,
                cy=climate_year,
                ty=target_year,
                eraa_year=eraa_year,
            )

    elif market_interconnection == "physical" or market_interconnection == "ntc":
        # Use setattr to dynamically add the attribute
        setattr(n, "_market_interconnection", market_interconnection)
    else:
        raise ValueError(
            f"Unknown market interconnection type: {market_interconnection}"
        )


def reshape(
    df: pd.DataFrame,
    climate_year: int,
    name: str,
    full_domain_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Reshape and process a DataFrame based on domain assignment and climate year.

    This function performs several transformations on the input DataFrame:
    1. Resets a specific index level
    2. Unstacks the DataFrame
    3. Reindexes based on domain assignment values
    4. Sets a time-based index for the entire climate year
    5. Stacks the DataFrame
    6. Removes NaN values
    7. Sets index names

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be reshaped.
    climate_year : int
        The reference climate year to set the date range.
    name : str
        The name to be assigned to the second index level after stacking.
    full_domain_series : pandas.Series, optional
        A series containing domain assignment values used for reindexing.

    Returns
    -------
    pandas.DataFrame
        The reshaped and processed DataFrame with a MultiIndex.
    """
    df = df.reset_index(2, drop=True).unstack(1)
    data_range = pd.date_range(
        f"{climate_year}-01-01", f"{climate_year}-12-31 23:00", freq="h"
    )

    if full_domain_series is not None:
        logger.info(
            f"Reshaping DataFrame with domain assignment for climate year {climate_year}"
        )
        df = df.reindex(full_domain_series.values)
        df.index = data_range
        df = df.stack(1).dropna()
    else:
        logger.info(
            f"Reshaping DataFrame without domain assignment for climate year {climate_year}"
        )
        # Since Nordic has only one domain, replicate across all hours of the
        # year. Drop Feb 29 so this stays 8760 hours even when `climate_year`
        # is a leap year, matching n.snapshots (which is always leap-day-free)
        # instead of hardcoding 8760 against a `data_range` that can be 8784.
        nordic_range = drop_feb_29(data_range)
        df = pd.concat([df] * len(nordic_range), ignore_index=True)
        df.index = nordic_range
        df = df.stack(1)

    df.index.names = ["snapshot", name]
    return df


def load_full_year_flow_based_data(
    buses: list[str], climate_year: int, target_year: int, eraa_year: int = 2024
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process flow-based domain data from an Excel file for a specific climate and target year.

    This function reads flow-based domain data from an ERAA (European Resource Adequacy Assessment)
    Excel file, processing Power Transfer Distribution Factors (PTDF) and Remaining Available Margin
    (RAM) data for specified buses and years.

    Parameters
    ----------
    buses : List[str]
        A list of bus names or identifiers to filter the data.

    climate_year : int
        The reference climate year for data processing.
        Must be within the range of available weather scenarios (1982-2017).

    target_year : int
        The target year for which the flow-based domain data is being analyzed.

    eraa_year : int, optional
        The ERAA (European Resource Adequacy Assessment) year of the dataset.
        Currently supports 2023 and 2024. Default is 2024.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing four pandas DataFrames:
        - pt_core: Core Power Transfer Distribution Factors
        - pt_ahc: Adjusted Hourly Commitment Power Transfer Distribution Factors
        - pt_dc: DC (Direct Current) Power Transfer Distribution Factors
        - ram: Remaining Available Margin

    Raises
    ------
    ValueError
        If the specified climate year is not available in the WS (Weather Scenario) mapping.

    Warning
    -------
    Logs a warning if an unsupported ERAA year is provided.

    Notes
    -----
    - The function supports data processing for ERAA years 2023 and 2024.
    - Data is reshaped and filtered based on the specified buses and years.
    - The Excel file is expected to have specific sheet and column naming conventions.
    """

    pt_core = pd.DataFrame()
    pt_ahc = pd.DataFrame()
    pt_dc = pd.DataFrame()
    ram = pd.DataFrame()

    # Validate ERAA year
    if eraa_year not in [2023, 2024]:
        logger.warning("Unsupported ERAA year. Use 2023 or 2024.")
        return pt_core, pt_ahc, pt_dc, ram

    path_config = get_path_config()
    input_dir = path_config.input_dir
    if input_dir is None:
        raise OSError(
            "EUFLEX_INPUT_DIR environment variable is not set. Please set it to the input directory path."
        )
    excel_file_path = pathlib.Path(
        pathlib.Path(input_dir),
        "data",
        f"FB-Domain-CORE_Full_ERAA{str(eraa_year)}.xlsx",
    )
    excel_file = pd.ExcelFile(excel_file_path)

    if eraa_year == 2023:
        domain_assign = pd.read_excel(
            excel_file, sheet_name="FB Domain Assignment", index_col=0
        ).iloc[:, 4:]
        domain_assign.columns = [
            int(str(i).split("_")[1]) for i in domain_assign.columns
        ]
        full_domain_assign = domain_assign[climate_year]

        pt_core = reshape(
            pd.read_excel(
                excel_file, sheet_name="PTDF", index_col=[0, 1, 2], header=[0, 1]
            )["PTDF_SZ"],
            climate_year,
            "CNEC",
            full_domain_assign,
        )
        pt_ahc = reshape(
            pd.read_excel(
                excel_file, sheet_name="PTDF", index_col=[0, 1, 2], header=[0, 1]
            )["PTDF*_AHC,SZ"],
            climate_year,
            "CNEC",
            full_domain_assign,
        )
        pt_dc = reshape(
            pd.read_excel(
                excel_file, sheet_name="PTDF", index_col=[0, 1, 2], header=[0, 1]
            )["PTDF_EvFB"],
            climate_year,
            "CNEC",
            full_domain_assign,
        )

        ram_sheets = [s for s in excel_file.sheet_names if "RAM" in s]
        ram_years = pd.Series(
            [int(s.split("_")[1]) for s in ram_sheets], index=ram_sheets
        )
        ram_sheet = ram_years[ram_years <= target_year].idxmax()

        ram = pd.read_excel(excel_file, sheet_name=ram_sheet, index_col=0, skiprows=3)
        ram.columns = ram.columns.astype(int)
        ram = ram.T.reindex(full_domain_assign.values)
        ram.index = pd.date_range(
            f"{climate_year}-01-01", f"{climate_year}-12-31 23:00", freq="h"
        )
        ram = ram.stack()
        ram.index.names = ["snapshot", "CNEC"]

        pt_ahc.columns.name = "Link"
        pt_ahc = pt_ahc.loc[:, pt_ahc.sum() != 0]
        pt_ahc = pt_ahc[
            [
                i
                for i in pt_ahc.columns
                if i.split("-")[0][:4] in buses and i.split("-")[1][:4] in buses
            ]
        ]
        pt_ahc = pt_ahc.T.groupby(pt_ahc.columns.str[:9]).sum().T

    elif eraa_year == 2024:
        # 1. Map WS columns to actual years
        ws_lookup = {f"WS{i + 1}": 1982 + i for i in range(36)}
        year_lookup = {v: k for k, v in ws_lookup.items()}
        ws_column = year_lookup.get(climate_year)
        if ws_column is None:
            raise ValueError(f"Climate year {climate_year} is not in WS mapping.")

        # 2. Read domain assignment
        domain_assign_raw = pd.read_excel(excel_file, sheet_name="FB Domain Assignment")
        ws_cols = [
            col for col in domain_assign_raw.columns if str(col).startswith("WS")
        ]
        domain_assign = domain_assign_raw.loc[
            domain_assign_raw["Year"] == target_year, ws_cols
        ].copy()
        domain_assign.columns = [ws_lookup[str(c)] for c in domain_assign.columns]

        # 3. Map domain labels to numeric values
        full_domain_assign = domain_assign[climate_year]
        full_domain_assign.index = pd.RangeIndex(
            1, len(full_domain_assign) + 1, name="Time_step"
        )

        # 5. Extract PTDF_SZ data and reshape
        pt_core = reshape(
            pd.read_excel(
                excel_file,
                sheet_name=f"PTDF_{target_year}",
                index_col=[0, 1, 2],
                header=[0, 1],
            )["PTDF_SZ"],
            climate_year,
            "CNEC",
            full_domain_assign,
        )

        pt_ahc = reshape(
            pd.read_excel(
                excel_file,
                sheet_name=f"PTDF_{target_year}",
                index_col=[0, 1, 2],
                header=[0, 1],
            )["PTDF*_AHC,SZ"],
            climate_year,
            "CNEC",
            full_domain_assign,
        )
        pt_dc = reshape(
            pd.read_excel(
                excel_file,
                sheet_name=f"PTDF_{target_year}",
                index_col=[0, 1, 2],
                header=[0, 1],
            )["PTDF_EvFB"],
            climate_year,
            "CNEC",
            full_domain_assign,
        )

        ram_sheets = [s for s in excel_file.sheet_names if "RAM" in s]
        ram_years = pd.Series(
            [int(s.split("_")[1]) for s in ram_sheets], index=ram_sheets
        )
        ram_sheet = ram_years[ram_years <= target_year].idxmax()

        ram = pd.read_excel(excel_file, sheet_name=ram_sheet, index_col=0, skiprows=1)
        ram = ram.T.reindex(full_domain_assign.values)
        ram.index = pd.date_range(
            f"{climate_year}-01-01", f"{climate_year}-12-31 23:00", freq="h"
        )
        ram = ram.stack()
        ram.index.names = ["snapshot", "CNEC"]

        pt_ahc.columns.name = "Link"
        pt_ahc = pt_ahc.select_dtypes(include="number")
        pt_ahc = pt_ahc.loc[:, pt_ahc.sum() != 0]
        pt_ahc = pt_ahc[
            [
                i
                for i in pt_ahc.columns
                if i.split("-")[0][:4] in buses and i.split("-")[1][:4] in buses
            ]
        ]
        pt_ahc = pt_ahc.T.groupby(pt_ahc.columns.str[:9]).sum().T

    # Adapt names of pt_core
    pt_core.columns.name = "Link"
    pt_dc.columns.name = "Link"
    pt_core = pt_core.rename(columns={name: f"{name}-CORE" for name in pt_core.columns})

    return pt_core, pt_ahc, pt_dc, ram


def load_full_year_flow_based_nordic_data(
    buses: list[str], climate_year: int, target_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the data for FBMC of NORDIC from the Excel file. Only available for eraa_year 2024."""
    path_config = get_path_config()
    input_dir = path_config.input_dir
    if input_dir is None:
        raise OSError(
            "EUFLEX_INPUT_DIR environment variable is not set. Please set it to the input directory path."
        )
    excel_file_path = pathlib.Path(input_dir, "data", "FB-Domain-NORDIC_ERAA2024.xlsx")
    excel_file = pd.ExcelFile(excel_file_path)

    # 1. Map WS columns to actual years
    ws_lookup = {f"WS{i + 1}": 1982 + i for i in range(36)}
    year_lookup = {v: k for k, v in ws_lookup.items()}
    ws_column = year_lookup.get(climate_year)

    if ws_column is None:
        raise ValueError(f"Climate year {climate_year} is not in WS mapping.")

    # 2. Read domain assignment from Excel file
    domain_assign_raw = pd.read_excel(excel_file, sheet_name="FB Domain Assignment")
    year_data = domain_assign_raw[domain_assign_raw["Year"] == target_year]

    if year_data.empty:
        raise ValueError(
            f"No domain assignment data found for target year {target_year}"
        )

    # Extract domain assignment for the specific climate year
    full_domain_assign = year_data[ws_column].reset_index(drop=True)
    full_domain_assign.name = "domain_assignment"
    full_domain_assign.index = pd.RangeIndex(
        1, len(full_domain_assign) + 1, name="Time_step"
    )

    # 5. Extract PTDF_SZ data and reshape
    pt_nordic = reshape(
        pd.read_excel(
            excel_file,
            sheet_name=f"PTDF_{target_year}",
            index_col=[0, 1, 2],
            header=[0, 1],
        )["PTDF_SZ"],
        climate_year=climate_year,
        name="CNEC",
    )

    pt_ahc = reshape(
        pd.read_excel(
            excel_file,
            sheet_name=f"PTDF_{target_year}",
            index_col=[0, 1, 2],
            header=[0, 1],
        )["PTDF*_AHC,SZ"],
        climate_year=climate_year,
        name="CNEC",
    )
    pt_dc = reshape(
        pd.read_excel(
            excel_file,
            sheet_name=f"PTDF_{target_year}",
            index_col=[0, 1, 2],
            header=[0, 1],
        )["PTDF_EvFB"],
        climate_year=climate_year,
        name="CNEC",
    )

    ram_sheets = [s for s in excel_file.sheet_names if "RAM" in s]
    ram_years = pd.Series([int(s.split("_")[1]) for s in ram_sheets], index=ram_sheets)
    ram_sheet = ram_years[ram_years <= target_year].idxmax()

    ram = pd.read_excel(excel_file, sheet_name=ram_sheet, index_col=0, skiprows=1)
    ram = ram.T.reindex(full_domain_assign.values)
    ram.index = pd.date_range(
        f"{climate_year}-01-01", f"{climate_year}-12-31 23:00", freq="h"
    )
    ram = ram.stack()
    ram.index.names = ["snapshot", "CNEC"]

    pt_ahc.columns.name = "Link"
    pt_ahc = pt_ahc.select_dtypes(include="number")
    pt_ahc = pt_ahc.loc[:, pt_ahc.sum() != 0]
    pt_ahc = pt_ahc[
        [
            i
            for i in pt_ahc.columns
            if i.split("-")[0][:4] in buses and i.split("-")[1][:4] in buses
        ]
    ]
    pt_ahc = pt_ahc.T.groupby(pt_ahc.columns.str[:9]).sum().T

    # Adapt names of pt_core
    pt_nordic.columns.name = "Link"
    pt_dc.columns.name = "Link"
    pt_nordic = pt_nordic.rename(
        columns={name: f"{name}-NORDIC" for name in pt_nordic.columns}
    )

    return pt_nordic, pt_ahc, pt_dc, ram


def register_flow_based_data(
    n: pypsa.Network,
    cy: int,
    ty: int,
    eraa_year: int = 2023,
) -> None:
    """
    Register flow-based data for the given network and snapshots.

    Parameters
    ----------
    n : pypsa.Network
        The network to register flow-based data for.
    cy : int
        The climate year.
    ty : int
        The target year.
    eraa_year : int
        The ERAA year (e.g., 2023, 2024).
    """
    ptdf_core_all, ptdf_ahc_all, ptdf_dc_all, ram_all = load_full_year_flow_based_data(
        buses=list(n.buses.index),
        climate_year=int(cy),
        target_year=int(ty),
        eraa_year=int(eraa_year),
    )

    ptdf_core = ptdf_core_all.loc[
        n.snapshots, ptdf_core_all.columns.isin(n.links.index)
    ]

    ptdf_ahc = ptdf_ahc_all.loc[n.snapshots]
    ptdf_dc = ptdf_dc_all.loc[n.snapshots, ptdf_dc_all.columns.isin(n.links.index)]
    ram = ram_all.loc[n.snapshots]

    links_direction = n.links.bus0 + "-" + n.links.bus1
    links_reverse = n.links.bus1 + "-" + n.links.bus0
    rename_reverse_to_direction = dict(zip(links_reverse, links_direction))
    ptdf_ahc_directed = pd.concat(
        [
            ptdf_ahc.loc[:, ptdf_ahc.columns.isin(links_direction)],
            -ptdf_ahc.loc[:, ptdf_ahc.columns.isin(links_reverse)].rename(
                columns=rename_reverse_to_direction
            ),
        ],
        axis=1,
    )

    # rename CNECs
    ptdf_core.index = ptdf_core.index.map(
        lambda x: (x[0], f"core_{x[1]}") if isinstance(x, tuple) else f"core_{x}"
    )
    ptdf_ahc_directed.index = ptdf_ahc_directed.index.map(
        lambda x: (x[0], f"core_{x[1]}") if isinstance(x, tuple) else f"core_{x}"
    )
    ptdf_dc.index = ptdf_dc.index.map(
        lambda x: (x[0], f"core_{x[1]}") if isinstance(x, tuple) else f"core_{x}"
    )
    ram.index = ram.index.map(
        lambda x: (x[0], f"core_{x[1]}") if isinstance(x, tuple) else f"core_{x}"
    )

    if eraa_year == 2024:
        pt_nordic_all, pt_ahc_nordic_all, pt_dc_nordic_all, ram_nordic_all = (
            load_full_year_flow_based_nordic_data(
                buses=list(n.buses.index), climate_year=int(cy), target_year=int(ty)
            )
        )
        ptdf_nordic = pt_nordic_all.loc[
            n.snapshots, pt_nordic_all.columns.isin(n.links.index)
        ]
        ptdf_ahc_nordic = pt_ahc_nordic_all.loc[n.snapshots]
        ptdf_dc_nordic = pt_dc_nordic_all.loc[
            n.snapshots, pt_dc_nordic_all.columns.isin(n.links.index)
        ]
        ram_nordic = ram_nordic_all.loc[n.snapshots]
        ptdf_ahc_nordic_directed = pd.concat(
            [
                ptdf_ahc_nordic.loc[:, ptdf_ahc_nordic.columns.isin(links_direction)],
                -ptdf_ahc_nordic.loc[
                    :, ptdf_ahc_nordic.columns.isin(links_reverse)
                ].rename(columns=rename_reverse_to_direction),
            ],
            axis=1,
        )

        # Rename Nordic CNECs with "nordic_" prefix while preserving MultiIndex
        ptdf_nordic_renamed = ptdf_nordic.copy()
        ptdf_nordic_renamed.index = ptdf_nordic_renamed.index.map(
            lambda x: (
                (x[0], f"nordic_{x[1]}") if isinstance(x, tuple) else f"nordic_{x}"
            )
        )

        ptdf_ahc_nordic_directed_renamed = ptdf_ahc_nordic_directed.copy()
        ptdf_ahc_nordic_directed_renamed.index = (
            ptdf_ahc_nordic_directed_renamed.index.map(
                lambda x: (
                    (x[0], f"nordic_{x[1]}") if isinstance(x, tuple) else f"nordic_{x}"
                )
            )
        )

        ptdf_dc_nordic_renamed = ptdf_dc_nordic.copy()
        ptdf_dc_nordic_renamed.index = ptdf_dc_nordic_renamed.index.map(
            lambda x: (
                (x[0], f"nordic_{x[1]}") if isinstance(x, tuple) else f"nordic_{x}"
            )
        )

        ram_nordic_renamed = ram_nordic.copy()
        ram_nordic_renamed.index = ram_nordic_renamed.index.map(
            lambda x: (
                (x[0], f"nordic_{x[1]}") if isinstance(x, tuple) else f"nordic_{x}"
            )
        )

        # Merge Nordic and Core dataframes while preserving MultiIndex structure
        # Align columns first to ensure consistent structure
        all_links = set(ptdf_core.columns) | set(ptdf_nordic_renamed.columns)
        ptdf_core_aligned = ptdf_core.reindex(columns=all_links, fill_value=np.nan)
        ptdf_nordic_aligned = ptdf_nordic_renamed.reindex(
            columns=all_links, fill_value=np.nan
        )

        all_ahc_links = set(ptdf_ahc_directed.columns) | set(
            ptdf_ahc_nordic_directed_renamed.columns
        )
        ptdf_ahc_directed_aligned = ptdf_ahc_directed.reindex(
            columns=all_ahc_links, fill_value=np.nan
        )
        ptdf_ahc_nordic_directed_aligned = ptdf_ahc_nordic_directed_renamed.reindex(
            columns=all_ahc_links, fill_value=np.nan
        )

        all_dc_links = set(ptdf_dc.columns) | set(ptdf_dc_nordic_renamed.columns)
        ptdf_dc_aligned = ptdf_dc.reindex(columns=all_dc_links, fill_value=np.nan)
        ptdf_dc_nordic_aligned = ptdf_dc_nordic_renamed.reindex(
            columns=all_dc_links, fill_value=np.nan
        )

        # Concatenate aligned dataframes
        ptdf_fb_region = pd.concat(
            [ptdf_core_aligned, ptdf_nordic_aligned], axis=0, sort=False
        )
        ptdf_ahc_directed_merged = pd.concat(
            [ptdf_ahc_directed_aligned, ptdf_ahc_nordic_directed_aligned],
            axis=0,
            sort=False,
        )
        ptdf_dc = pd.concat(
            [ptdf_dc_aligned, ptdf_dc_nordic_aligned], axis=0, sort=False
        )
        ram = pd.concat([ram, ram_nordic_renamed], axis=0, sort=False)

        ptdf_ahc_directed = ptdf_ahc_directed_merged
    else:
        ptdf_fb_region = ptdf_core

    ptdf_fb_region = xr.DataArray(ptdf_fb_region).unstack("dim_0")
    ptdf_ahc_directed = xr.DataArray(ptdf_ahc_directed).unstack("dim_0")
    ptdf_dc = xr.DataArray(ptdf_dc).unstack("dim_0")
    ram = xr.DataArray(ram).unstack("dim_0")

    # Use setattr to dynamically add attributes to the network
    setattr(n, "_ptdf_fb_region", ptdf_fb_region)
    setattr(n, "_ptdf_ahc_directed", ptdf_ahc_directed)
    setattr(n, "_ptdf_dc", ptdf_dc)
    setattr(n, "_ram", ram)


def run_fbmc_pipeline(
    n: pypsa.Network,
    cy: int,
    raw_domain_assign_data: pd.DataFrame,
    raw_ptdf_data: pd.DataFrame,
    raw_ram_data: pd.DataFrame,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Run the shared FBMC processing pipeline on pre-loaded, pre-patched raw tables.

    Shared by both the database-backed and CSV-fallback FBMC processing paths.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    cy : int
        Climate year (integer, not WS format).
    raw_domain_assign_data : pd.DataFrame
        Domain assignment table, already filtered and patched.
    raw_ptdf_data : pd.DataFrame
        PTDF table, already filtered and patched.
    raw_ram_data : pd.DataFrame
        RAM table, already filtered and patched.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
        ptdf_fb_region, ptdf_ahc_directed, ptdf_dc, ram — as DataArrays.
    """
    # Process domain assignment data
    logger.info("Processing domain assignment data...")

    # Use create_datetime_from_eraa to create proper datetime snapshots
    raw_domain_assign_data = create_datetime_from_eraa(raw_domain_assign_data, cy)

    # Set index to snapshot for easy filtering and alignment
    domain_assign_filtered = raw_domain_assign_data.set_index("snapshot")

    # The domain assignment value is in the 'value' column, not 'domain_id'
    full_domain_assign = domain_assign_filtered["value"]

    # Process PTDF data
    logger.info("Processing PTDF data...")

    def _reshape_ptdf_data(
        data, ptdf_type, full_domain_assign, cy, n, column_type="Link"
    ):
        """Reshape PTDF data following the Excel pattern."""
        available_regions = data["region"].unique()
        ptdf_region = []
        all_columns = set()

        for region in available_regions:
            region_data = data[
                (data["ptdf_type"] == ptdf_type) & (data["region"] == region)
            ].copy()

            if region_data.empty:
                continue

            # Filter domain assignment
            filtered_domain_assign = full_domain_assign[
                full_domain_assign.isin(region_data["fb_domain_id"])
            ]

            if len(filtered_domain_assign) == 0:
                continue

            # Create initial pivot: index=[domain, cnec], columns=Link/market_node, values=value
            region_pivot = region_data.pivot_table(
                index=["fb_domain_id", "fb_ptdf_ram_cnec_id"],
                columns=column_type,
                values="value",
            )

            # Excel pattern: unstack(1) - unstack the CNEC level
            region_unstacked = region_pivot.unstack(1)

            # Excel pattern: reindex with domain assignment values
            region_reindexed = region_unstacked.reindex(
                filtered_domain_assign.values, fill_value=np.nan
            )

            # Reset index to integers (needed because domain IDs create duplicate index)
            region_reindexed = region_reindexed.reset_index(drop=True)

            # Excel pattern: set datetime index for FULL YEAR (8760 hours)
            # This ensures we have all snapshots that might be in n.snapshots
            full_year_range = pd.date_range(
                f"{cy}-01-01", f"{cy}-12-31 23:00", freq="h"
            )
            # Reindex to full year if needed (pad with last values)
            if len(region_reindexed) < len(full_year_range):
                region_reindexed = region_reindexed.reindex(
                    range(len(full_year_range)), method="ffill"
                )
            region_reindexed.index = full_year_range

            # Excel pattern: stack(1) - move CNEC level from columns to index, then dropna
            region_reshaped = region_reindexed.stack(1, future_stack=True).dropna()

            region_reshaped.index.names = ["snapshot", "CNEC"]
            region_reshaped.columns.name = column_type

            ptdf_region.append(region_reshaped)
            all_columns = all_columns | set(region_reshaped.columns)

        # Align all regions to same columns and concatenate
        final_ptdf_region = []
        for ptdf in ptdf_region:
            ptdf_aligned = ptdf.reindex(columns=all_columns, fill_value=np.nan)
            final_ptdf_region.append(ptdf_aligned)

        if final_ptdf_region:
            combined_ptdf = pd.concat(final_ptdf_region, axis=0, sort=False)
            # Filter to network snapshots and links (Excel pattern)
            if column_type == "Link":
                combined_ptdf = combined_ptdf.loc[
                    n.snapshots, combined_ptdf.columns.isin(n.links.index)
                ]
            else:
                combined_ptdf = combined_ptdf.loc[n.snapshots]
            return xr.DataArray(combined_ptdf).unstack("dim_0")
        else:
            # Return empty DataArray with proper structure
            empty_series = pd.Series(dtype=float)
            empty_series.index = pd.MultiIndex.from_tuples(
                [], names=["snapshot", "CNEC"]
            )
            return xr.DataArray(empty_series).unstack("dim_0")

    # Process region-specific PTDF_SZ data (core and nordic)
    ptdf_fb_region_final = _reshape_ptdf_data(
        raw_ptdf_data, "PTDF_SZ", full_domain_assign, cy, n, "Link"
    )

    # Process AHC PTDF data with directed links
    logger.info("Processing AHC PTDF data...")

    # Process AHC data separately because it needs directed link transformation
    # Note: Database uses "PTDF*_AHC,SZ" format (with asterisk and comma)
    def _reshape_ahc_ptdf_data(data, full_domain_assign, cy, n):
        """Reshape AHC PTDF data and apply directed link transformation."""
        available_regions = data["region"].unique()
        ptdf_region = []

        for region in available_regions:
            region_data = data[
                (data["ptdf_type"] == "PTDF*_AHC,SZ") & (data["region"] == region)
            ].copy()

            if region_data.empty:
                continue

            # Filter domain assignment
            filtered_domain_assign = full_domain_assign[
                full_domain_assign.isin(region_data["fb_domain_id"])
            ]

            if len(filtered_domain_assign) == 0:
                continue

            # Create initial pivot
            region_pivot = region_data.pivot_table(
                index=["fb_domain_id", "fb_ptdf_ram_cnec_id"],
                columns="Link",
                values="value",
            )

            # Unstack and reindex
            region_unstacked = region_pivot.unstack(1)
            region_reindexed = region_unstacked.reindex(
                filtered_domain_assign.values, fill_value=np.nan
            )
            region_reindexed = region_reindexed.reset_index(drop=True)

            # Create full year range
            full_year_range = pd.date_range(
                f"{cy}-01-01", f"{cy}-12-31 23:00", freq="h"
            )
            if len(region_reindexed) < len(full_year_range):
                region_reindexed = region_reindexed.reindex(
                    range(len(full_year_range)), method="ffill"
                )
            region_reindexed.index = full_year_range

            # Stack and dropna - keep as DataFrame
            region_reshaped = region_reindexed.stack(1, future_stack=True).dropna()
            region_reshaped.index.names = ["snapshot", "CNEC"]
            region_reshaped.columns.name = "Link"

            ptdf_region.append(region_reshaped)

        if ptdf_region:
            # Concatenate all regions
            ahc_df = pd.concat(ptdf_region, axis=0, sort=False)

            # Apply directed link transformation (Excel pattern)
            links_direction = n.links.bus0 + "-" + n.links.bus1
            links_reverse = n.links.bus1 + "-" + n.links.bus0
            rename_reverse_to_direction = dict(zip(links_reverse, links_direction))

            ptdf_ahc_directed = pd.concat(
                [
                    ahc_df.loc[:, ahc_df.columns.isin(links_direction)],
                    -ahc_df.loc[:, ahc_df.columns.isin(links_reverse)].rename(
                        columns=rename_reverse_to_direction
                    ),
                ],
                axis=1,
            )

            # Filter to network snapshots only (Excel pattern: line 31 was ptdf_ahc_all.loc[n.snapshots])
            # Note: Excel didn't filter to n.links.index here, just used links_direction/links_reverse filter above
            ptdf_ahc_directed = ptdf_ahc_directed.loc[n.snapshots]

            # Convert to xarray
            return xr.DataArray(ptdf_ahc_directed).unstack("dim_0")
        else:
            # Empty structure with network snapshots (Excel pattern)
            empty_data = xr.DataArray(
                np.empty((0, len(n.snapshots), 0), dtype=float),
                dims=["Link", "snapshot", "CNEC"],
                coords={
                    "Link": [],
                    "snapshot": n.snapshots,
                    "CNEC": [],
                },
            )
            return empty_data

    ptdf_ahc_directed_final = _reshape_ahc_ptdf_data(
        raw_ptdf_data, full_domain_assign, cy, n
    )

    # Process DC/Evolved FB PTDF data
    logger.info("Processing DC/Evolved FB PTDF data...")
    ptdf_dc_final = _reshape_ptdf_data(
        raw_ptdf_data, "PTDF_EvFB", full_domain_assign, cy, n, "Link"
    )

    # Process RAM data
    logger.info("Processing RAM data...")

    def _reshape_ram_data(data, full_domain_assign, cy, n):
        """Reshape RAM data following the Excel pattern."""
        available_regions = data["region"].unique()
        ram_region = []
        all_cnecs = set()

        for region in available_regions:
            region_data = data[data["region"] == region].copy()

            if region_data.empty:
                continue

            # Filter domain assignment
            filtered_domain_assign = full_domain_assign[
                full_domain_assign.isin(region_data["fb_domain_id"])
            ]

            if len(filtered_domain_assign) == 0:
                continue

            # Create pivot: index=domain, columns=cnec, values=value
            region_pivot = region_data.pivot_table(
                index="fb_domain_id", columns="fb_ptdf_ram_cnec_id", values="value"
            )

            # Reindex with domain assignment values
            region_reindexed = region_pivot.reindex(
                filtered_domain_assign.values, fill_value=np.nan
            )

            # Reset index to integers (needed because domain IDs create duplicate index)
            region_reindexed = region_reindexed.reset_index(drop=True)

            # Set datetime index for FULL YEAR (8760 hours)
            full_year_range = pd.date_range(
                f"{cy}-01-01", f"{cy}-12-31 23:00", freq="h"
            )
            # Reindex to full year if needed
            if len(region_reindexed) < len(full_year_range):
                region_reindexed = region_reindexed.reindex(
                    range(len(full_year_range)), method="ffill"
                )
            region_reindexed.index = full_year_range

            # Stack to create MultiIndex, then dropna (Excel pattern)
            region_reshaped = region_reindexed.stack().dropna()
            region_reshaped.index.names = ["snapshot", "CNEC"]

            ram_region.append(region_reshaped)
            all_cnecs = all_cnecs | set(region_reshaped.index.get_level_values("CNEC"))

        # Combine all regions
        if ram_region:
            combined_ram = pd.concat(ram_region, axis=0, sort=False)
            # Filter to network snapshots (Excel pattern: line 33 was ram_all.loc[n.snapshots])
            combined_ram = combined_ram.loc[n.snapshots]
            return xr.DataArray(combined_ram).unstack("dim_0")
        else:
            # Return empty DataArray with proper structure
            empty_series = pd.Series(dtype=float)
            empty_series.index = pd.MultiIndex.from_tuples(
                [], names=["snapshot", "CNEC"]
            )
            return xr.DataArray(empty_series).unstack("dim_0")

    ram_final = _reshape_ram_data(raw_ram_data, full_domain_assign, cy, n)

    return ptdf_fb_region_final, ptdf_ahc_directed_final, ptdf_dc_final, ram_final


def patch_fbmc_domain_assignment(
    raw_domain_assign_data: pd.DataFrame,
    raw_ptdf_data: pd.DataFrame,
    raw_ram_data: pd.DataFrame,
    ty: int,
    ws_climate_year: str,
    eraa_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply temporary patches to FBMC domain assignment and related data.

    This function applies several data transformations to work around database
    inconsistencies in the FBMC (Flow-Based Market Coupling) data:
    1. Add missing Nordic domain assignment data
    2. Standardize region names (nordics -> nordic)
    3. Rename CNEC IDs to avoid duplication across regions
    4. Create Link identifiers for topology matching

    TODO: THIS IS TO BE ADAPTED IN THE DATABASE AND THE CODE BELOW CAN BE REMOVED.

    Parameters
    ----------
    raw_domain_assign_data : pd.DataFrame
        Raw domain assignment data from database
    raw_ptdf_data : pd.DataFrame
        Raw PTDF (Power Transfer Distribution Factor) data
    raw_ram_data : pd.DataFrame
        Raw RAM (Remaining Available Margin) data
    ty : int
        Target year
    ws_climate_year : str
        Climate year for weather scenarios
    eraa_year : int
        ERAA dataset year

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Patched versions of (domain_assign_data, ptdf_data, ram_data)
    """
    # 1) Considering that nordics has only one domain
    version = raw_domain_assign_data["version"].iloc[0]
    date_range = pd.date_range(
        start=f"{ty}-01-01 00:00:00", end=f"{ty}-12-31 23:00:00", freq="h"
    )
    value = "nordic_ptdf"
    hour_id = ["00000" if h.hour == 0 else f"{h.hour:02d}0000" for h in date_range]

    new_data = pd.DataFrame(
        {
            "id": [None] * len(date_range),
            "date_id": date_range.strftime(DateFormat.COMPACT_DATE.value),
            "hour_id": hour_id,
            "target_year": ty,
            "region": "nordics",
            "ws_cy": ws_climate_year,
            "value": value,
            "eraa_year": eraa_year,
            "version": version,
        }
    )

    raw_domain_assign_data = pd.concat(
        [raw_domain_assign_data, new_data], ignore_index=True
    )

    raw_ptdf_data.loc[raw_ptdf_data["region"] == "nordics", "fb_domain_id"] = value

    # 2) Standardize region names: rename 'nordics' to 'nordic'
    raw_ptdf_data["region"] = raw_ptdf_data["region"].str.replace(
        "nordics", "nordic", case=False
    )
    raw_ram_data["region"] = raw_ram_data["region"].str.replace(
        "nordics", "nordic", case=False
    )

    # 3) Rename CNEC ids to avoid duplication
    raw_ptdf_data["fb_ptdf_ram_cnec_id"] = (
        raw_ptdf_data["region"] + "_" + raw_ptdf_data["fb_ptdf_ram_cnec_id"].astype(str)
    )
    raw_ram_data["fb_ptdf_ram_cnec_id"] = (
        raw_ram_data["region"] + "_" + raw_ram_data["fb_ptdf_ram_cnec_id"].astype(str)
    )

    # 4) Create links to adapt to topology and NTCs tables
    raw_ptdf_data["market_node_to"] = raw_ptdf_data["market_node_to"].fillna(
        raw_ptdf_data["region"].str.upper()
    )
    raw_ptdf_data["Link"] = (
        raw_ptdf_data["market_node_from"].astype(str)
        + "-"
        + raw_ptdf_data["market_node_to"].astype(str)
    )

    return raw_domain_assign_data, raw_ptdf_data, raw_ram_data


FBMC_CSV_FILENAMES = ("domain_assignments.csv", "ptdf.csv", "ram.csv")


def resolve_fbmc_csv_dir(eraa_year: int, config: dict[str, Any]) -> pathlib.Path:
    """
    Resolve the directory holding FBMC CSV fallback files for ``eraa_year``.

    Checks ``eva.fbmc_csv_fallback_dir`` first (a publicly bundled data
    directory, e.g. for the public release), and uses it only if all three
    expected files are present under its ``eraa{eraa_year}`` subdirectory.
    Otherwise falls back to the working-directory cache populated by
    ``SqlDataProcessor.export_fbmc_csv_fallbacks``.

    Parameters
    ----------
    eraa_year : int
        ERAA year.
    config : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    pathlib.Path
        Directory to read the FBMC CSV fallback files from.
    """
    configured_dir = config["eva"]["fbmc_csv_fallback_dir"]
    if configured_dir:
        public_dir = resolve_input_path(configured_dir) / f"eraa{eraa_year}"
        if all((public_dir / name).exists() for name in FBMC_CSV_FILENAMES):
            return public_dir

    return get_path_config().working_dir / "db_fallbacks" / f"eraa{eraa_year}"


def process_fbmc_data_from_csvs(
    n: pypsa.Network,
    cy: int,
    ty: int,
    eraa_year: int,
    config: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Process FBMC data from CSV fallback files instead of the database.

    Reads the full raw tables previously exported to CSV as a database
    fallback, filters them to the requested ``ty`` / ``cy`` combination, and
    runs the same processing pipeline (``run_fbmc_pipeline``) as the
    database-backed path.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object.
    cy : int
        Climate year.
    ty : int
        Target year.
    eraa_year : int
        ERAA year.
    config : dict[str, Any]
        Configuration dictionary, used to check ``eva.fbmc_csv_fallback_dir``
        for a publicly bundled data directory before falling back to the
        working-directory cache (see ``resolve_fbmc_csv_dir``).

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
        ptdf_fb_region, ptdf_ahc_directed, ptdf_dc, ram — as DataArrays.
    """
    out_dir = resolve_fbmc_csv_dir(eraa_year, config)
    logger.info("Loading FBMC data from CSV fallbacks at %s", out_dir)

    climate_year = climate_year_to_ws_number(cy)

    raw_domain_assign_data = pd.read_csv(out_dir / "domain_assignments.csv")
    raw_ptdf_data = pd.read_csv(out_dir / "ptdf.csv")
    raw_ram_data = pd.read_csv(out_dir / "ram.csv")

    # Filter full-table CSVs down to the requested ty/cy before processing
    raw_domain_assign_data = raw_domain_assign_data[
        (raw_domain_assign_data["target_year"] == ty)
        & (raw_domain_assign_data["ws_cy"] == climate_year)
        & (raw_domain_assign_data["eraa_year"] == eraa_year)
    ]
    raw_ptdf_data = raw_ptdf_data[
        (raw_ptdf_data["target_year"] == ty) & (raw_ptdf_data["eraa_year"] == eraa_year)
    ]
    raw_ram_data = raw_ram_data[
        (raw_ram_data["target_year"] == ty) & (raw_ram_data["eraa_year"] == eraa_year)
    ]

    raw_domain_assign_data, raw_ptdf_data, raw_ram_data = patch_fbmc_domain_assignment(
        raw_domain_assign_data,
        raw_ptdf_data,
        raw_ram_data,
        ty,
        climate_year,
        eraa_year,
    )

    return run_fbmc_pipeline(n, cy, raw_domain_assign_data, raw_ptdf_data, raw_ram_data)
