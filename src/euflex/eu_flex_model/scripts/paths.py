# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
"""
Directory resolution for the EU-Flex model workflow.

The workflow reads its inputs from, and writes its scratch data to, four
directories. Each one is resolved in the same order:

1. an override set in-process through :meth:`PathConfig.set_override`,
2. the corresponding ``EUFLEX_*`` environment variable,
3. a default derived from the checkout the workflow is running against.

Only the directories the model itself reads live here. A wrapping
distribution adds its own — logs, results, locks and so on — by subclassing
:class:`PathConfig`.
"""

import enum
import os
from pathlib import Path

from euflex import eu_flex_model

#: Files that mark the root of a checkout. The model package sits at a
#: different depth in the public ``eu-flex-model`` repository than it does in
#: the ``eu-flex-platform`` one, and the platform may take the model from an
#: installed package rather than from its own tree, so all three are checked.
_PROJECT_ROOT_MARKERS: tuple[str, ...] = (
    "src/euflex/eu_flex_model/config/config.default.yaml",
    "src/euflex/eu_flex_model/config/config.default.yaml",
    "src/euflex/workflow/config/config.platform.yaml",
)


def _safe_exists(path: Path) -> bool:
    """
    Check whether a path exists, treating an unreadable path as absent.

    Parameters
    ----------
    path : Path
        Path to test.

    Returns
    -------
    bool
        ``True`` if the path exists and could be read, ``False`` otherwise.
    """
    try:
        return path.exists()
    except OSError:
        return False


def find_project_root() -> Path:
    """
    Find the checkout root by searching upward from the working directory.

    Returns
    -------
    Path
        The first ancestor of the working directory holding one of
        :data:`_PROJECT_ROOT_MARKERS`, or the working directory itself when
        none does.
    """
    current = Path.cwd().resolve()

    for path in [current, *current.parents]:
        for marker in _PROJECT_ROOT_MARKERS:
            if _safe_exists(path / marker):
                return path

    return current


class PathConfigurationError(Exception):
    """Raised when path configuration fails or is invalid."""


class EnvVarEnum(str, enum.Enum):
    """
    Environment variables that select the model workflow's directories.

    Attributes
    ----------
    ENV_CONFIG_FILES : str
        Directory holding the configuration files.
    ENV_SENSITIVITY_DATA : str
        Directory holding sensitivity-mode input data.
    ENV_INPUT_DATA : str
        Checkout holding ``data/``, the pre-built networks in particular.
    ENV_WORKING_DIR : str
        Directory for per-run scratch data.
    """

    ENV_CONFIG_FILES: str = "EUFLEX_CONFIG_DIR"
    ENV_SENSITIVITY_DATA: str = "EUFLEX_SENSITIVITY_DIR"
    ENV_INPUT_DATA: str = "EUFLEX_INPUT_DIR"
    ENV_WORKING_DIR: str = "EUFLEX_WORKING_DIR"


class PathConfig:
    """
    Resolves the directories the model workflow reads and writes.

    Resolved paths are cached, so a directory keeps the value it was first
    resolved to for the lifetime of the instance. Call
    :func:`reset_path_config` to start again.
    """

    #: Name of the configuration file expected in ``config_dir``. The platform
    #: subclass points this at its own overlay.
    DEFAULT_CONFIG_FILENAME = "config.default.yaml"

    def __init__(self):
        """Initialise with no overrides and an empty cache."""
        self._cli_overrides: dict[str, Path | None] = {}
        self._resolved_paths: dict[str, Path] = {}

    def set_override(self, key: str, path: Path | None) -> None:
        """
        Override one directory, taking precedence over the environment.

        Parameters
        ----------
        key : str
            Internal key of the directory, as used by :meth:`_resolve_path`.
        path : Path or None
            The directory to use, or ``None`` to drop the override.
        """
        self._cli_overrides[key] = path
        self._resolved_paths.pop(key, None)

    def _get_env_path(self, env_var: str) -> Path | None:
        """
        Read a directory from the environment.

        Parameters
        ----------
        env_var : str
            Environment variable to read.

        Returns
        -------
        Path or None
            The resolved path, or ``None`` when the variable is unset or empty.
        """
        value = os.getenv(env_var)
        if value:
            return Path(value).expanduser().resolve()

        return None

    def _resolve_path(
        self,
        key: str,
        env_var: str,
        fallback: Path,
    ) -> Path:
        """
        Resolve one directory and cache the result.

        Parameters
        ----------
        key : str
            Internal key, used for the override lookup and the cache.
        env_var : str
            Environment variable to consult.
        fallback : Path
            Default used when neither an override nor the environment applies.

        Returns
        -------
        Path
            The resolved absolute path.
        """
        if key in self._resolved_paths:
            return self._resolved_paths[key]

        override = self._cli_overrides.get(key)
        if override is not None:
            self._resolved_paths[key] = override.resolve()
            return self._resolved_paths[key]

        env_path = self._get_env_path(env_var)
        if env_path is not None:
            self._resolved_paths[key] = env_path
            return env_path

        self._resolved_paths[key] = fallback.expanduser().resolve()
        return self._resolved_paths[key]

    @property
    def config_dir(self) -> Path:
        """Directory holding the configuration files."""
        return self._resolve_path(
            "config",
            EnvVarEnum.ENV_CONFIG_FILES,
            Path(eu_flex_model.__path__[0], "config"),
        )

    @property
    def sensitivity_dir(self) -> Path:
        """Directory holding sensitivity-mode input data."""
        return self._resolve_path(
            "sensitivity",
            EnvVarEnum.ENV_SENSITIVITY_DATA,
            find_project_root() / "var" / "workflow" / "sensitivity",
        )

    @property
    def input_dir(self) -> Path:
        """Checkout holding ``data/``, the pre-built networks in particular."""
        return self._resolve_path(
            "input", EnvVarEnum.ENV_INPUT_DATA, find_project_root()
        )

    @property
    def working_dir(self) -> Path:
        """Directory for per-run scratch data."""
        return self._resolve_path(
            "working",
            EnvVarEnum.ENV_WORKING_DIR,
            find_project_root() / "var",
        )

    def get_all_paths(self) -> dict[str, Path]:
        """
        Return every directory this configuration resolves.

        Returns
        -------
        dict of str to Path
            Mapping of directory name to resolved absolute path.
        """
        return {
            "config_dir": self.config_dir,
            "sensitivity_dir": self.sensitivity_dir,
            "input_dir": self.input_dir,
            "working_dir": self.working_dir,
        }


# Global instance
_global_config: PathConfig | None = None


def get_path_config() -> PathConfig:
    """
    Get the global :class:`PathConfig` instance.

    Returns
    -------
    PathConfig
        The process-wide singleton.
    """
    global _global_config
    if _global_config is None:
        _global_config = PathConfig()
    return _global_config


def reset_path_config() -> None:
    """Discard the global :class:`PathConfig` instance (mainly for testing)."""
    global _global_config
    _global_config = None


def resolve_input_path(path: str | Path) -> Path:
    """
    Locate an input file or directory named by a path in the configuration.

    An absolute path, or a relative one that exists from the working
    directory, is used as given. Any other relative path is taken
    from the input directory, like the pre-built networks, so that a
    configuration can name files in the checkout and still work when the
    workflow runs from elsewhere with ``--directory``.

    Parameters
    ----------
    path : str or Path
        Path as written in the configuration.

    Returns
    -------
    Path
        Where to read the input from, whether or not it exists.
    """
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path

    return get_path_config().input_dir / path
