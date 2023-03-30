import os
from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml(path: os.PathLike) -> Dict[str, Any]:
    with open(path, "r") as fin:
        cfg = yaml.safe_load(fin)
    return cfg


def convert_to_paths(paths: Dict[str, os.PathLike]) -> Dict[str, Path]:
    return {k: Path(v) for k, v in paths.items()}
