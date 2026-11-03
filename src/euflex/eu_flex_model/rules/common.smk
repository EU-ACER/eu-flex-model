# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-FileCopyrightText: Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
#
# SPDX-License-Identifier: MIT
#
# Config access helpers shared by the rule files.
#
# Rules read configuration through config_provider rather than indexing config
# directly, so that a value is looked up when the rule's parameters are
# resolved rather than when the rule is defined.
from functools import partial


def get_config(config, keys, default=None):
    """Retrieve a nested value from a dictionary using a tuple of keys."""
    value = config
    for key in keys:
        if isinstance(value, list):
            value = value[key]
        else:
            value = value.get(key, default)
        if value == default:
            return default
    return value


def config_provider(*keys, default=None):
    """Return a callable that resolves a nested config value for a rule.

    Usage in Snakemake rules would look something like:
    params:
        my_param=config_provider("key1", "key2", default="some_default_value")
    """
    return partial(_getter, keys=keys, default=default)


def _getter(wildcards, keys, default):
    """Look up a config value, ignoring the wildcards Snakemake passes in."""
    return get_config(config, keys, default)
