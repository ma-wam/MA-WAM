"""Repository-relative path helpers shared by MA-WAM entry points."""

from pathlib import Path
from typing import Mapping, Union


def data_path_from_config(
    config: Mapping[str, object], project_root: Union[str, Path]
) -> str:
    """Return the dataset directory specified by a Phase-1 configuration."""
    root = Path(project_root)
    if "data_root" in config:
        data_root = Path(str(config["data_root"])).expanduser()
        if not data_root.is_absolute():
            data_root = root / data_root
        return str(data_root / str(config["env_name"]) / str(config["data_split"]))
    return str(
        root
        / "diffuser"
        / "datasets"
        / "data"
        / str(config["env_type"])
        / str(config["env_name"])
        / str(config["data_split"])
    )
