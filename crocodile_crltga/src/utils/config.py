import json
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["config_path"] = str(Path(path).resolve())
    return config
