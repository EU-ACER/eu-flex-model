# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System


import re
from pathlib import Path
from typing import Any


def sanitize_filename(value: object) -> str:
    """Return a filesystem-safe string."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def with_gurobi_exports(
    solver_name: str,
    solver_options: dict[str, Any] | None,
    export_dir: str | Path,
    stem: str,
    export_log: bool = True,
    export_model: bool = False,
) -> dict[str, Any]:
    """Add optional Gurobi LogFile and ResultFile options without mutating input options."""
    options = dict(solver_options or {})

    if solver_name != "gurobi":
        return options

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    if export_log:
        options.setdefault("LogFile", str(export_path / f"{stem}.log"))

    if export_model:
        options.setdefault("ResultFile", str(export_path / f"{stem}.mps"))

    return options
