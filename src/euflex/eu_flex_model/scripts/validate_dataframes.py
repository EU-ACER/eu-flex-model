import logging
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Field
from pandera.typing import Series  # ty: ignore[possibly-missing-import]


class ComponentSchema(pa.DataFrameModel):
    @staticmethod
    def _validate_pu_series(series: Any) -> bool:
        for val in series.dropna():
            if isinstance(val, (list | np.ndarray)):
                val = pd.Series(val)  # Convert list/array to Series
            if isinstance(val, pd.Series):
                if not val.dropna().between(-1, 1).all():
                    return False
            elif isinstance(val, (int | float | np.number)):
                if not -1 <= val <= 1:
                    return False
            else:
                return False
        return True

    @staticmethod
    def _is_valid_pu(value: Any) -> bool:
        if pd.api.types.is_scalar(value):
            if pd.isna(value):
                return True
        else:
            # For array-like, treat as valid if all elements are NA
            arr = np.asarray(value)
            if arr.size == 0 or np.all(pd.isna(arr)):
                return True
        if isinstance(value, (int | float | np.number)):
            return not np.isnan(value) if not np.isinf(value) else True
        if isinstance(value, (list | tuple | np.ndarray)):
            try:
                arr = np.asarray(value, dtype=float)
                return len(arr) > 0 and all(
                    not np.isnan(val) or np.isinf(val) for val in arr
                )
            except (ValueError, TypeError):
                return False
        return False

    @staticmethod
    def _check_min_max(min_value: Any, max_value: Any) -> bool:
        # Convert list or array to Series if needed
        if isinstance(min_value, (list | np.ndarray)):
            min_value = pd.Series(min_value)
        if isinstance(max_value, (list | np.ndarray)):
            max_value = pd.Series(max_value)

        # Both are Series
        if isinstance(min_value, pd.Series) and isinstance(max_value, pd.Series):
            if len(min_value) != len(max_value):
                return False
            return (max_value >= min_value).all()

        # One is Series, one is scalar
        if isinstance(min_value, pd.Series) and np.isscalar(max_value):
            return (max_value >= min_value).all()
        if isinstance(max_value, pd.Series) and np.isscalar(min_value):
            return (max_value >= min_value).all()

        # Both are scalars
        if np.isscalar(min_value) and np.isscalar(max_value):
            return max_value >= min_value

        return False  # Any unsupported case

    @classmethod
    def validate_component(cls, df: pd.DataFrame) -> pd.DataFrame:
        try:
            return cls.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            logging.error(f"{cls.__name__} validation failed:\n{exc.failure_cases}")
            raise  # Let the calling code handle the exception properly

    @classmethod
    def apply_defaults(
        cls,
        df: pd.DataFrame,
        eva_defaults: pd.Series | None = None,
        target_year: int | None = None,
        earliest_ty: int | None = None,
        carrier_efficiency_defaults: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Apply default values to DataFrame before validation.

        Base implementation is a no-op. Subclasses should override
        to apply component-specific defaults using the centralized
        apply_*_defaults() functions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to apply defaults to
        eva_defaults : pd.Series, optional
            EVA cone parameter defaults (carrier-specific)
        target_year : int, optional
            Target year for the simulation
        earliest_ty : int, optional
            Earliest target year from scenario config
        carrier_efficiency_defaults : dict[str, float], optional
            Per-carrier efficiency fallback (``eva.base_network.efficiency``),
            used ahead of the blanket 1.0 default for carriers with no
            technical-parameter data of their own (e.g. Biofuel).

        Returns
        -------
        pd.DataFrame
            DataFrame with defaults applied
        """
        return df

    @classmethod
    def validate_with_defaults(
        cls,
        df: pd.DataFrame,
        eva_defaults: pd.Series | None = None,
        target_year: int | None = None,
        earliest_ty: int | None = None,
        carrier_efficiency_defaults: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Validate DataFrame, applying defaults only if initial validation fails.

        Attempts strict validation first (without any defaults). If validation
        fails, applies defaults and re-validates. This ensures we only modify
        data when necessary.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        eva_defaults : pd.Series, optional
            EVA cone parameter defaults (carrier-specific)
        target_year : int, optional
            Target year for the simulation. Used for build_year default.
        earliest_ty : int, optional
            Earliest target year from scenario config. Used for aggregated assets
            to set consistent commissioning_year across all TYs.
        carrier_efficiency_defaults : dict[str, float], optional
            Per-carrier efficiency fallback (``eva.base_network.efficiency``),
            passed through to ``apply_defaults``.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame (with defaults applied only if needed)

        Raises
        ------
        pa.errors.SchemaErrors
            If validation fails even after applying defaults
        """
        try:
            # First attempt: strict validation without defaults
            return cls.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            logging.debug(
                f"{cls.__name__} validation failed (ahead of applying defaults):\n"
                f"{exc.failure_cases}"
            )
            # Apply defaults and retry
            df = cls.apply_defaults(
                df, eva_defaults, target_year, earliest_ty, carrier_efficiency_defaults
            )
            try:
                return cls.validate(df, lazy=True)
            except pa.errors.SchemaErrors as exc:
                logging.error(
                    f"{cls.__name__} validation failed (after applying defaults):\n"
                    f"{exc.failure_cases}"
                )
                raise


class GeneratorSchema(ComponentSchema):
    # ─── REQUIRED columns ─────────────────────────────────────────
    # Note: 'name' is in the CSV but as the index column, not a separate column
    # When read with index_col=0, it becomes the index, so we don't validate it here

    bus: Series[str] = Field(
        nullable=False,
        description="Name of bus to which generator is connected",
        str_length={"min_value": 1},
        str_matches=r"^[\w\-]+$",
    )

    # ─── Nominal power columns  ──────────────────────────────────
    p_nom: Series[float] | None = Field(
        description="Nominal power capacity [MW]",
        default=0.0,
        nullable=True,
        ge=0.0,
        le=100_000_000,  # Allow very large values for load shedding/spillage
        coerce=True,  # Allow int to float conversion
    )

    p_nom_extendable: Series[bool] | None = Field(
        description="Whether capacity can be extended in Economic Viability Assessment (EVA)",
        default=False,
        nullable=True,
    )

    p_nom_min: Series[float] | None = Field(
        description="Minimum nominal power if extendable [MW]",
        default=0.0,
        nullable=True,
        ge=0.0,
    )

    p_nom_max: Series[float] | None = Field(
        description="Maximum nominal power if extendable [MW]",
        default=np.inf,
        nullable=True,
        ge=0.0,
    )

    p_min_pu: Series[Any] | None = Field(
        description="Minimum active power per unit of p_nom",
        default=0.0,
        nullable=True,
    )

    p_max_pu: Series[Any] | None = Field(
        description="Maximum active power per unit of p_nom",
        default=1.0,
        nullable=True,
    )

    # ─── Energy sum attributes validation ──────────────────────────
    e_sum_min: Series[float] | None = Field(
        description="The minimum total energy produced during a single optimization horizon",
        default=-np.inf,
        nullable=True,
    )

    e_sum_max: Series[float] | None = Field(
        description="The maximum total energy produced during a single optimization horizon",
        default=np.inf,
        nullable=True,
    )

    # ─── Economic attributes validation ────────────────────────────
    marginal_cost: Series[float] | None = Field(
        description="Marginal cost of production [EUR/MWh] (can be negative for load spillage)",
        default=0.0,
        nullable=True,
        ge=-10_000,  # Allow negative values for load spillage
        le=30_000,
    )

    lifetime: Series[float] | None = Field(
        description="Lifetime of the generator [years]",
        default=40.0,
        nullable=False,
        ge=0,
        coerce=True,  # Allow int to float or inf
    )

    @pa.check("lifetime", element_wise=True)
    def validate_lifetime_finite(cls, value):
        """Reject infinite lifetimes to trigger defaults application."""
        return np.isfinite(value)

    build_year: Series[int] | None = Field(
        description="Year when generator was built/commissioned",
        nullable=True,
        coerce=True,
    )

    decommissioning_year: Series[int] = Field(
        description="Year when generator will be decommissioned (computed from build_year + lifetime if missing)",
        nullable=False,
        coerce=True,
    )

    capital_cost: Series[float] | None = Field(
        description="Capital cost of extending capacity [EUR/MW]",
        default=0.0,
        nullable=True,
        ge=0.0,
    )

    # ─── Technical attributes validation ──────────────────────────
    efficiency: Series[float] | None = Field(
        description="Energy conversion efficiency", default=1.0, nullable=True
    )

    committable: Series[bool] | None = Field(
        description="Whether generator can be committed/decommitted",
        default=False,
        nullable=True,
    )

    start_up_cost: Series[float] | None = Field(
        description="Cost of starting up a committable generator [EUR]",
        default=0.0,
        nullable=True,
        ge=0.0,
    )

    shut_down_cost: Series[float] | None = Field(
        description="Cost of shutting down a committable generator [EUR]",
        default=0.0,
        nullable=True,
        ge=0.0,
    )

    min_up_time: Series[int] | None = Field(
        description="Minimum up time of committable generator [hours]",
        default=0,
        nullable=True,
        ge=0,
    )

    min_down_time: Series[int] | None = Field(
        description="Minimum down time of committable generator [hours]",
        default=0,
        nullable=True,
        ge=0,
    )

    up_time_before: Series[int] | None = Field(
        description="Number of snapshots that the committable generator is online before "
        "the start of the optimization [hours]",
        default=1,
        nullable=True,
        ge=0,
    )

    down_time_before: Series[int] | None = Field(
        description="Number of snapshots that the committable generator is offline before "
        "the start of the optimization [hours]",
        default=0,
        nullable=True,
        ge=0,
    )

    # ─── Carrier and technology attributes validation ─────────────
    carrier: Series[str] | None = Field(
        description="Energy carrier (e.g., fuel type)",
        nullable=True,
        isin=[
            # PyPSA-Eur naming conventions
            "DC",
            "AC",
            "oil",
            "CCGT",
            "biomass",
            "lignite",
            "OCGT",
            "nuclear",
            "coal",
            "geothermal",
            "hydro",
            "offwind-dc",
            "solar",
            "onwind",
            "offwind-ac",
            "offwind-float",
            "solar-hsat",
            "ror",
            "PHS",
            "battery",
            "H2",
            "",
            "load shedding",
            "load spillage",
            # ERAA/EU-Flex naming conventions
            "Gas",
            "Oil",
            "Coal",
            "Hard coal",
            "Lignite",
            "Nuclear",
            "Geothermal",
            "Biofuel",
            "Small biomass",
            "Solar (PV)",
            "Solar (Thermal)",
            "Solar PV rooftop",
            "Solar PV farm",
            "Wind Onshore",
            "Wind Offshore",
            "Hydro - Run of River (Turbine)",
            "Marine",
            "DSR",
            "DSR_explicit",
            "DSR_implicit",
            "Electrolyser",
            "Non-market batteries",
            "Others (non-RES)",
            "Not defined or splitting not known RES",
            "periphery_gen",
            "FuelCell",
        ],
        str_length={"min_value": 1},
    )

    # ─── Carrier and technology attributes validation ─────────────
    dsr_type: Series[str] | None = Field(
        description="Type of e-DSR generator",
        default="load_shedding",
        nullable=True,
        # Note: isin validation done via custom check below (allows nullable)
    )

    @pa.check("dsr_type", element_wise=True)
    def validate_dsr_type(cls, value):
        """Validate dsr_type is in allowed values, but allow NaN/None."""
        if pd.isna(value) or value == "":
            return True  # Allow null/empty values
        return value in ["load_shedding", "load_shifting"]

    # ─── Ramp rates validation ────────────────────────────────────
    ramp_limit_up: Series[float] | None = Field(
        description="Ramp up limit per unit of p_nom per hour",
        default=np.nan,
        nullable=True,
        ge=0.0,
    )

    ramp_limit_down: Series[float] | None = Field(
        description="Ramp down limit per unit of p_nom per hour",
        default=np.nan,
        nullable=True,
        ge=0.0,
    )

    ramp_limit_start_up: Series[float] | None = Field(
        description="Ramp limit for start-up per unit of p_nom",
        default=1.0,
        nullable=True,
        ge=0.0,
    )

    ramp_limit_shut_down: Series[float] | None = Field(
        description="Ramp limit for shut-down per unit of p_nom",
        default=1.0,
        nullable=True,
        ge=0.0,
    )

    # ─── EVA Cost Parameters (for sensitivity CSV mode) ─────────────
    lifetime_extendable: Series[bool] | None = Field(
        description="Whether this asset is eligible for lifetime extension",
        nullable=True,
    )

    lifetime_extension_years: Series[float] | None = Field(
        description="Years of lifetime extension available",
        nullable=True,
        ge=0.0,
    )

    Mothballing_Cost: Series[float] | None = Field(
        description="CAPEX for mothballing [EUR/MW]",
        nullable=True,
        ge=0.0,
    )

    hurdle_rate_demothballing: Series[float] | None = Field(
        description="Hurdle rate for demothballing decision [fraction]",
        nullable=True,
        ge=0.0,
        le=1.0,
    )

    hurdle_rate_lifetime_extension: Series[float] | None = Field(
        description="Hurdle rate for lifetime extension decision [fraction]",
        nullable=True,
        ge=0.0,
        le=1.0,
    )

    # ─── Custom validation ────────────────────────────────────────
    @pa.check("p_min_pu", element_wise=True)
    def validate_p_min_pu(cls, value):
        return cls._is_valid_pu(value)

    @pa.check("p_max_pu", element_wise=True)
    def validate_p_max_pu(cls, value):
        return cls._is_valid_pu(value)

    @pa.check("p_min_pu", element_wise=False)
    def check_p_min_pu(cls, series: pd.Series) -> bool:
        return cls._validate_pu_series(series)

    @pa.check("p_max_pu", element_wise=False)
    def check_p_max_pu(cls, series: pd.Series) -> bool:
        return cls._validate_pu_series(series)

    @pa.dataframe_check
    def min_max_e_sum(cls, df: pd.DataFrame) -> Series[bool]:
        # Only check rows where both columns exist and are not null
        if "e_sum_min" not in df or "e_sum_max" not in df:
            return Series([True] * len(df), index=df.index)
        mask = df["e_sum_min"].notnull() & df["e_sum_max"].notnull()
        result = Series([True] * len(df), index=df.index)
        result[mask] = df.loc[mask, "e_sum_min"] <= df.loc[mask, "e_sum_max"]
        return result

    @pa.dataframe_check
    def min_max_pu(cls, df: pd.DataFrame) -> Series[bool]:
        return df.apply(
            lambda row: cls._check_min_max(row["p_min_pu"], row["p_max_pu"]), axis=1
        )

    @classmethod
    def apply_defaults(
        cls,
        df: pd.DataFrame,
        eva_defaults: pd.Series | None = None,
        target_year: int | None = None,
        earliest_ty: int | None = None,
        carrier_efficiency_defaults: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Apply generator-specific defaults."""
        return apply_generator_defaults(
            df, eva_defaults, target_year, earliest_ty, carrier_efficiency_defaults
        )

    class Config:
        """Base configuration class."""

        coerce = True  # Allow type coercion (int->float, etc.)


class StorageUnitSchema(ComponentSchema):
    # ─── REQUIRED columns ─────────────────────────────────────────
    # Note: 'name' is in the CSV but as the index column, not a separate column
    # When read with index_col=0, it becomes the index, so we don't validate it here

    bus: Series[str] = Field(
        nullable=False,
        description="Bus to which storage unit is connected",
        str_length={"min_value": 1},
        str_matches=r"^[\w\-]+$",
    )

    # ─── Power capacity and extension ─────────────────────────────
    p_nom: Series[float] | None = Field(
        description="Nominal power capacity [MW]",
        default=0.0,
        nullable=True,
        ge=0.0,
        le=100_000,
    )

    p_nom_mod: Series[float] | None = Field(
        description="Nominal power per module [MW]",
        default=0.0,
        nullable=True,
        ge=0.0,
    )

    p_nom_extendable: Series[bool] | None = Field(
        description="Whether power capacity can be extended",
        default=False,
        nullable=True,
    )

    p_nom_min: Series[float] | None = Field(
        description="Minimum nominal power if extendable [MW]",
        default=0.0,
        nullable=True,
        ge=0.0,
    )

    p_nom_max: Series[float] | None = Field(
        description="Maximum nominal power if extendable [MW]",
        default=np.inf,
        nullable=True,
        ge=0.0,
    )

    # ─── Dispatch characteristics ─────────────────────────────────
    p_min_pu: Series[Any] | None = Field(
        description="Minimum dispatch per unit of p_nom (can be negative for charging)",
        default=-1.0,
        nullable=True,
    )

    p_max_pu: Series[Any] | None = Field(
        description="Maximum dispatch per unit of p_nom",
        default=1.0,
        nullable=True,
    )

    p_set: Series[Any] | None = Field(
        description="Fixed dispatch value [MW]",
        default=0.0,
        nullable=True,
    )

    q_set: Series[Any] | None = Field(
        description="Fixed reactive power [MVar]",
        default=0.0,
        nullable=True,
    )

    sign: Series[float] | None = Field(
        description="Sign convention for dispatch (usually +1)",
        default=1.0,
        nullable=True,
    )

    control: Series[str] | None = Field(
        description='Control mode ("PQ", "PV", or "Slack")',
        default="PQ",
        nullable=True,
        isin=["PQ", "PV", "Slack"],
    )

    # ─── Energy capacity and SOC settings ─────────────────────────
    max_hours: Series[float] | None = Field(
        description="Energy-to-power ratio [h]",
        default=1.0,
        nullable=True,
        ge=0.0,
    )

    state_of_charge_initial: Series[float] | None = Field(
        description="Initial energy state of charge [MWh]",
        default=0.0,
        nullable=True,
        ge=0.0,
    )

    state_of_charge_initial_per_period: Series[bool] | None = Field(
        description="Use initial SOC at each investment period",
        default=False,
        nullable=True,
    )

    state_of_charge_set: Series[Any] | None = Field(
        description="Fixed state of charge [MWh] at each snapshot",
        default=np.nan,
        nullable=True,
    )

    cyclic_state_of_charge: Series[bool] | None = Field(
        description="Whether SOC is cyclic over optimization horizon",
        default=False,
        nullable=True,
    )

    cyclic_state_of_charge_per_period: Series[bool] | None = Field(
        description="Whether SOC is cyclic within each investment period",
        default=True,
        nullable=True,
    )

    # ─── Losses and efficiencies ─────────────────────────────────
    efficiency_store: Series[Any] | None = Field(
        description="Charging efficiency [pu]",
        default=1.0,
        nullable=True,
        ge=0.0,
        le=1.0,
    )

    efficiency_dispatch: Series[Any] | None = Field(
        description="Discharging efficiency [pu]",
        default=1.0,
        nullable=True,
        ge=0.0,
        le=1.0,
    )

    standing_loss: Series[Any] | None = Field(
        description="Standing energy loss per hour [pu]",
        default=0.0,
        nullable=True,
        ge=0.0,
        le=1.0,
    )

    inflow: Series[Any] | None = Field(
        description="External natural inflow to storage [MW]",
        default=0.0,
        nullable=True,
    )

    # ─── Economic attributes ─────────────────────────────────────
    marginal_cost: Series[Any] | None = Field(
        description="Marginal cost of dispatch [EUR/MWh]",
        default=0.0,
        nullable=True,
        ge=0.0,
        le=20_000,
    )

    marginal_cost_quadratic: Series[Any] | None = Field(
        description="Quadratic marginal cost term [EUR/MWh²]",
        default=0.0,
        nullable=True,
    )

    marginal_cost_storage: Series[Any] | None = Field(
        description="Marginal cost of storing energy [EUR/MWh/h]",
        default=0.0,
        nullable=True,
    )

    capital_cost: Series[float] | None = Field(
        description="Capital cost per MW [EUR/MW]",
        default=0.0,
        nullable=True,
        ge=0.0,
    )

    spill_cost: Series[Any] | None = Field(
        description="Cost for spilling unused energy [EUR/MWh]",
        default=0.0,
        nullable=True,
    )

    lifetime: Series[float] | None = Field(
        description="Lifetime [years]",
        default=20.0,
        nullable=False,
        ge=0.0,
    )

    @pa.check("lifetime", element_wise=True)
    def validate_lifetime_finite(cls, value):
        """Reject infinite lifetimes to trigger defaults application."""
        return np.isfinite(value)

    build_year: Series[int] | None = Field(
        description="Year of construction",
        default=0,
        nullable=True,
    )

    decommissioning_year: Series[int] = Field(
        description="Year when storage unit will be decommissioned (computed from build_year + lifetime if missing)",
        nullable=False,
        coerce=True,
    )

    # ─── Carrier and status ───────────────────────────────────────
    carrier: Series[str] | None = Field(
        description="Energy carrier type",
        default="",
        nullable=True,
        str_length={"min_value": 1},
        isin=[
            # PyPSA-Eur naming conventions
            "battery",
            "hydro",
            "PHS",
            "H2",
            "CAES",
            "flywheel",
            "",
            # ERAA/EU-Flex naming conventions
            "Batteries",
            "DSR_implicit",
            "Hydro - Reservoir",
            "Hydro - Pondage",
            "Hydro - Pump Storage Open Loop (Pumping)",
            "Hydro - Pump Storage Closed Loop (Pumping)",
            # Present in the database from TY2035 onwards only, so absent from
            # the earlier target years' networks.
            "Hydrogen",
        ],
    )

    active: Series[bool] | None = Field(
        description="Whether storage unit is included in OPF",
        default=True,
        nullable=True,
    )

    # ─── EVA Cost Parameters (for sensitivity CSV mode) ─────────────
    # NOTE: Mothballing parameters (hurdle_rate_demothballing, etc.) are excluded
    # because mothballing only applies to generators, not storage units.
    lifetime_extendable: Series[bool] | None = Field(
        description="Whether this asset is eligible for lifetime extension",
        nullable=True,
    )

    lifetime_extension_years: Series[float] | None = Field(
        description="Years of lifetime extension available",
        nullable=True,
        ge=0.0,
    )

    hurdle_rate_lifetime_extension: Series[float] | None = Field(
        description="Hurdle rate for lifetime extension decision [fraction]",
        nullable=True,
        ge=0.0,
        le=1.0,
    )

    @pa.dataframe_check
    def check_p_min_max_pu(cls, df: pd.DataFrame) -> Series[bool]:
        # Only check if both columns exist as scalars in the dataframe
        # (they might be in separate time-series files like storage_units-p_max_pu.csv)
        if "p_min_pu" not in df.columns or "p_max_pu" not in df.columns:
            return Series([True] * len(df), index=df.index)
        result = Series(True, index=df.index)
        mask = df["p_min_pu"].notnull() & df["p_max_pu"].notnull()
        result[mask] = df.loc[mask, "p_min_pu"] <= df.loc[mask, "p_max_pu"]
        return result

    @pa.dataframe_check
    def check_initial_soc_within_capacity(cls, df: pd.DataFrame) -> Series[bool]:
        # Only check if state_of_charge_initial column exists
        if "state_of_charge_initial" not in df.columns:
            return Series([True] * len(df), index=df.index)
        cap = df["p_nom"] * df["max_hours"]
        return df["state_of_charge_initial"] <= cap

    @classmethod
    def apply_defaults(
        cls,
        df: pd.DataFrame,
        eva_defaults: pd.Series | None = None,
        target_year: int | None = None,
        earliest_ty: int | None = None,
        carrier_efficiency_defaults: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Apply storage unit-specific defaults."""
        return apply_storage_defaults(
            df, eva_defaults, target_year, earliest_ty, carrier_efficiency_defaults
        )

    class Config:
        """Base configuration class."""

        coerce = True  # Allow type coercion (int->float, etc.)


# ─── Centralized Default/Fallback Functions ─────────────────────────────────
# These functions apply default values to DataFrames AFTER data is read
# but BEFORE validation. This centralizes fallback logic that was previously
# scattered throughout data loading code.


def fill_commissioning_decommissioning_lifetime(
    df: pd.DataFrame,
    default_lifetime: float,
    earliest_ty: int | None = None,
    target_year: int | None = None,
    component_name: str = "assets",
) -> pd.DataFrame:
    """
    Fill missing commissioning/decommissioning/lifetime values consistently.

    This ensures all three fields are defined for NPV calculations. Lifetime
    takes precedence over decommissioning_year when both are provided.

    Precedence logic:
    - Step 0: All three present → recalculate decommissioning_year from lifetime + build_year
    - Step 1: build_year + decommissioning_year present, lifetime missing → calculate lifetime
    - Step 2: Fill remaining missing lifetime with default
    - Step 3: Fill missing build_year and/or decommissioning_year using lifetime

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential lifetime/build_year/decommissioning_year columns
    default_lifetime : float
        Default lifetime in years to use when lifetime is missing
    earliest_ty : int, optional
        Earliest target year from scenario config. Used for aggregated assets
        to set consistent commissioning_year across all TYs.
    target_year : int, optional
        Current target year. Used as fallback if earliest_ty not provided.
    component_name : str, optional
        Component name for logging (e.g., "generators", "storage units")

    Returns
    -------
    pd.DataFrame
        DataFrame with lifetime, build_year, and decommissioning_year filled
    """
    df = df.copy()

    # Ensure all three columns exist (create if missing)
    if "lifetime" not in df.columns:
        df["lifetime"] = pd.NA
    if "build_year" not in df.columns:
        df["build_year"] = pd.NA
    if "decommissioning_year" not in df.columns:
        df["decommissioning_year"] = pd.NA

    # Identify missing values (before applying defaults)
    # Treat 0 as missing for build_year and decommissioning_year (invalid year values)
    mask_missing_lifetime = df["lifetime"].isna()

    # For infinite check, only check non-NA values to avoid numpy ufunc errors
    def is_infinite_or_not_finite(x):
        if pd.isna(x):
            return False
        return x == np.inf or not np.isfinite(x)

    mask_infinite_lifetime = df["lifetime"].apply(is_infinite_or_not_finite)
    mask_missing_build = df["build_year"].isna() | (df["build_year"] == 0)
    mask_missing_decom = df["decommissioning_year"].isna() | (
        df["decommissioning_year"] == 0
    )

    # Step 0: If all three values present → lifetime + build_year take precedence, recalculate decommissioning_year
    mask_all_present = (
        ~mask_missing_lifetime
        & ~mask_infinite_lifetime
        & ~mask_missing_build
        & ~mask_missing_decom
    )
    if mask_all_present.any():
        df.loc[mask_all_present, "decommissioning_year"] = (
            df.loc[mask_all_present, "build_year"]
            + df.loc[mask_all_present, "lifetime"]
        ).astype(int)
        logging.info(
            f"Recalculated decommissioning_year from lifetime + build_year for {mask_all_present.sum()} {component_name} (all three values provided, lifetime takes precedence)"
        )

    # Step 1: Calculate lifetime from build_year + decommissioning_year if lifetime is missing
    mask_calc_lifetime = (
        (mask_missing_lifetime | mask_infinite_lifetime)
        & ~mask_missing_build
        & ~mask_missing_decom
    )
    if mask_calc_lifetime.any():
        df.loc[mask_calc_lifetime, "lifetime"] = (
            df.loc[mask_calc_lifetime, "decommissioning_year"]
            - df.loc[mask_calc_lifetime, "build_year"]
        )
        logging.info(
            f"Calculated lifetime from build_year + decommissioning_year for {mask_calc_lifetime.sum()} {component_name}"
        )

    # Step 2: Fill remaining missing lifetime with default and handle infinite lifetimes
    mask_missing_lifetime = df["lifetime"].isna()
    mask_infinite_lifetime = df["lifetime"].apply(is_infinite_or_not_finite)
    mask_needs_default = mask_missing_lifetime | mask_infinite_lifetime

    if mask_needs_default.any():
        df.loc[mask_needs_default, "lifetime"] = default_lifetime
        if mask_missing_lifetime.any():
            logging.info(
                f"Applied default lifetime={default_lifetime}y to {mask_missing_lifetime.sum()} {component_name} with missing lifetime"
            )
        if mask_infinite_lifetime.any():
            logging.info(
                f"Replaced infinite lifetime with default={default_lifetime}y for {mask_infinite_lifetime.sum()} {component_name} (required for NPV calculation)"
            )

    # Step 3: Fill missing build_year and decommissioning_year
    # Re-calculate masks after filling lifetime
    mask_missing_build = df["build_year"].isna() | (df["build_year"] == 0)
    mask_missing_decom = df["decommissioning_year"].isna() | (
        df["decommissioning_year"] == 0
    )

    # Case 1: Has build, missing decom → decom = build + lifetime
    mask1 = ~mask_missing_build & mask_missing_decom
    if mask1.any():
        df.loc[mask1, "decommissioning_year"] = (
            df.loc[mask1, "build_year"] + df.loc[mask1, "lifetime"]
        ).astype(int)
        logging.info(
            f"Calculated decommissioning_year from build_year and lifetime for {mask1.sum()} {component_name}"
        )

    # Case 2: Has decom, missing build → build = decom - lifetime
    mask2 = mask_missing_build & ~mask_missing_decom
    if mask2.any():
        df.loc[mask2, "build_year"] = (
            df.loc[mask2, "decommissioning_year"] - df.loc[mask2, "lifetime"]
        ).astype(int)
        logging.info(
            f"Calculated build_year from decommissioning_year and lifetime for {mask2.sum()} {component_name}"
        )

    # Case 3: Missing both build and decom → use earliest_ty
    mask3 = mask_missing_build & mask_missing_decom
    if mask3.any():
        if earliest_ty is not None:
            df.loc[mask3, "build_year"] = int(earliest_ty)
            df.loc[mask3, "decommissioning_year"] = (
                earliest_ty + df.loc[mask3, "lifetime"]
            ).astype(int)
            logging.info(
                f"Set build_year={earliest_ty} and calculated decommissioning_year for {mask3.sum()} {component_name} (aggregated assets)"
            )
        elif target_year is not None:
            # Fallback to target_year if earliest_ty not provided
            df.loc[mask3, "build_year"] = int(target_year)
            df.loc[mask3, "decommissioning_year"] = (
                target_year + df.loc[mask3, "lifetime"]
            ).astype(int)
            logging.warning(
                f"earliest_ty not provided, using target_year={target_year} as build_year for {mask3.sum()} {component_name}"
            )
        else:
            raise ValueError(
                f"Cannot determine build_year for {mask3.sum()} {component_name}. "
                "Either earliest_ty or target_year must be provided."
            )

    return df


def apply_generator_defaults(
    df: pd.DataFrame,
    eva_defaults: pd.Series | None = None,
    target_year: int | None = None,
    earliest_ty: int | None = None,
    carrier_efficiency_defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Apply default values to generator DataFrame.

    This function centralizes all fallback logic for generators, replacing
    scattered .fillna() calls in data loading code.

    Parameters
    ----------
    df : pd.DataFrame
        Generator DataFrame with potential missing values
    eva_defaults : pd.Series, optional
        EVA cone parameter defaults indexed by column name, with carrier mapping.
        If provided, used for carrier-specific defaults before scalar fallbacks.
    target_year : int, optional
        Target year for the simulation. Used to set build_year default.
    earliest_ty : int, optional
        Earliest target year from scenario config. Used for aggregated assets
        to set consistent commissioning_year across all TYs.
    carrier_efficiency_defaults : dict[str, float], optional
        Per-carrier efficiency fallback (``eva.base_network.efficiency``), used
        ahead of the blanket 1.0 default for carriers with no technical-parameter
        data of their own (e.g. Biofuel, which has no matching row anywhere in
        the source technical-parameters table).

    Returns
    -------
    pd.DataFrame
        Generator DataFrame with defaults applied
    """
    df = df.copy()

    # Fleet size default
    if "n_units" in df.columns:
        missing = df["n_units"].isna().sum()
        if missing > 0:
            logging.info(f"Applying default n_units=100 to {missing} generators")
        df["n_units"] = df["n_units"].fillna(100).astype(int)

    # Apply EVA defaults if provided (carrier-specific)
    if eva_defaults is not None and "carrier" in df.columns:
        eva_columns = [
            col for col in eva_defaults.index if col in df.columns and col != "carrier"
        ]
        for col in eva_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df["carrier"].map(eva_defaults[col]))

    # Fill commissioning/decommissioning/lifetime using schema default
    df = fill_commissioning_decommissioning_lifetime(
        df,
        default_lifetime=GeneratorSchema.to_schema().columns["lifetime"].default,
        earliest_ty=earliest_ty,
        target_year=target_year,
        component_name="generators",
    )

    # Capital cost default
    if "capital_cost" in df.columns:
        missing = df["capital_cost"].isna().sum()
        if missing > 0:
            logging.warning(
                f"Missing capital_cost for {missing} generators; defaulting to 100 EUR/kW"
            )
        df["capital_cost"] = df["capital_cost"].fillna(0.0)

    # CAPEX default (used before renaming to capital_cost)
    if "CAPEX" in df.columns:
        missing = df["CAPEX"].isna().sum()
        if missing > 0:
            logging.warning(
                f"Missing CAPEX for {missing} generators; defaulting to 0 EUR/kW"
            )
        df["CAPEX"] = df["CAPEX"].fillna(0.0)

    # Efficiency default: replace 0 or NaN with the carrier-specific default
    # (if any), else 1.0, clip to avoid division by zero
    if "efficiency" in df.columns:
        df["efficiency"] = df["efficiency"].replace(0, pd.NA)
        if carrier_efficiency_defaults and "carrier" in df.columns:
            df["efficiency"] = df["efficiency"].fillna(
                df["carrier"].map(carrier_efficiency_defaults)
            )
        df["efficiency"] = df["efficiency"].fillna(1.0)
        df["efficiency"] = df["efficiency"].clip(lower=0.01)

    # Ramp limit defaults (1.0 = no limit)
    for col in ["ramp_limit_up", "ramp_limit_down"]:
        if col in df.columns:
            df[col] = df[col].fillna(1.0)

    # Marginal cost default
    if "marginal_cost" in df.columns:
        df["marginal_cost"] = df["marginal_cost"].fillna(0.0).clip(lower=0.0)

    # Time column defaults (ensure integer)
    time_columns = ["min_up_time", "min_down_time"]
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Planned outage defaults
    if "planned_outage_min_hours" in df.columns:
        df["planned_outage_min_hours"] = df["planned_outage_min_hours"].fillna(24)
    if "planned_outage_annual_windows" in df.columns:
        df["planned_outage_annual_windows"] = df[
            "planned_outage_annual_windows"
        ].fillna(1)

    # DSR type default - only for DSR generators (carrier contains 'DSR')
    if "dsr_type" in df.columns and "carrier" in df.columns:
        is_dsr = df["carrier"].str.contains("DSR", case=False, na=False)
        missing_dsr_type = is_dsr & (df["dsr_type"].isna() | (df["dsr_type"] == ""))
        if missing_dsr_type.any():
            dsr_type_default = GeneratorSchema.to_schema().columns["dsr_type"].default
            df.loc[missing_dsr_type, "dsr_type"] = dsr_type_default
            logging.info(
                f"Applied default dsr_type='{dsr_type_default}' to {missing_dsr_type.sum()} DSR generators"
            )

    return df


def apply_storage_defaults(
    df: pd.DataFrame,
    eva_defaults: pd.Series | None = None,
    target_year: int | None = None,
    earliest_ty: int | None = None,
    carrier_efficiency_defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Apply default values to storage unit DataFrame.

    This function centralizes all fallback logic for storage units, replacing
    scattered .fillna() calls in data loading code.

    Note: Intermediate capacity columns (charging_capacity, discharge_capacity,
    storage_capacity) are handled during preprocessing since they're used for
    p_nom/p_pu calculations and don't persist to the final DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Storage unit DataFrame with potential missing values
    eva_defaults : pd.Series, optional
        EVA cone parameter defaults indexed by column name, with carrier mapping.
        If provided, used for carrier-specific defaults before scalar fallbacks.
    target_year : int, optional
        Target year for the simulation. Used to set build_year default.
    earliest_ty : int, optional
        Earliest target year from scenario config. Used for aggregated assets
        to set consistent commissioning_year across all TYs.
    carrier_efficiency_defaults : dict[str, float], optional
        Per-carrier charge/discharge efficiency fallback (``eva.base_network.efficiency``),
        used ahead of the blanket 1.0 default for carriers with no technical-parameter
        data of their own (e.g. pumped hydro). This is a per-direction efficiency,
        applied identically to both efficiency_store and efficiency_dispatch.

    Returns
    -------
    pd.DataFrame
        Storage unit DataFrame with defaults applied
    """
    df = df.copy()

    # Apply EVA defaults if provided (carrier-specific)
    if eva_defaults is not None and "carrier" in df.columns:
        eva_columns = [
            col for col in eva_defaults.index if col in df.columns and col != "carrier"
        ]
        for col in eva_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df["carrier"].map(eva_defaults[col]))

    # Fill commissioning/decommissioning/lifetime using schema default
    df = fill_commissioning_decommissioning_lifetime(
        df,
        default_lifetime=StorageUnitSchema.to_schema().columns["lifetime"].default,
        earliest_ty=earliest_ty,
        target_year=target_year,
        component_name="storage units",
    )

    # Capital cost default
    if "capital_cost" in df.columns:
        missing = df["capital_cost"].isna().sum()
        if missing > 0:
            logging.warning(
                f"Missing capital_cost for {missing} storage units; defaulting to 100 EUR/kW"
            )
        df["capital_cost"] = df["capital_cost"].fillna(0.0)

    # CAPEX default (used before renaming to capital_cost)
    if "CAPEX" in df.columns:
        missing = df["CAPEX"].isna().sum()
        if missing > 0:
            logging.warning(
                f"Missing CAPEX for {missing} storage units; defaulting to 0 EUR/kW"
            )
        df["CAPEX"] = df["CAPEX"].fillna(0.0)

    # p_min_pu default for storage: 0 means no charging, should be -1.0
    if "p_min_pu" in df.columns:
        df["p_min_pu"] = pd.to_numeric(df["p_min_pu"], errors="coerce")
        # Where p_min_pu is 0 (no charging data), default to -1.0 (symmetric charging)
        df.loc[df["p_min_pu"] == 0, "p_min_pu"] = -1.0

    # Efficiency defaults: replace 0 or NaN with the carrier-specific default
    # (if any), else 1.0, clip to avoid division by zero
    for col in ("efficiency_store", "efficiency_dispatch"):
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)
            if carrier_efficiency_defaults and "carrier" in df.columns:
                df[col] = df[col].fillna(df["carrier"].map(carrier_efficiency_defaults))
            df[col] = df[col].fillna(1.0)
            df[col] = df[col].clip(lower=0.01)

    return df
