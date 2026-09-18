import os
from pathlib import Path
import yaml

def load_paths_config():
    """Loading configuration files"""
    root = Path(__file__).resolve().parents[1]
    default_path = root / "utils/paths.default.yaml"
    local_path = root / "utils/paths.local.yaml"

    with open(default_path, "r") as f:
        raw = os.path.expandvars(f.read())
        config = yaml.safe_load(raw)

    if local_path.exists():
        with open(local_path, "r") as f:
            local_raw = os.path.expandvars(f.read())
            local_cfg = yaml.safe_load(local_raw)
            config.update(local_cfg)

    # Expand ~ in any path values
    for k, v in config.items():
        config[k] = Path(os.path.expanduser(v)).resolve()

    return config