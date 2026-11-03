# Apply pypsa meshed_threshold patch when helpers package is imported
from euflex.eu_flex_model.scripts.helpers.pypsa_helpers import (  # noqa: F401
    _patch_pypsa_meshed_threshold,
)

_patch_pypsa_meshed_threshold()
