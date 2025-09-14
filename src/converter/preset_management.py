import json
import os
from pathlib import Path
from typing import Dict, List, Optional

PRESETS_DIR = "presets"


def get_preset_path(preset_name: str) -> Path:
    """Constructs the full path for a given preset name."""
    return Path(PRESETS_DIR) / f"{preset_name}.json"


def get_available_presets() -> List[str]:
    """Returns a list of available preset names without the .json extension."""
    if not os.path.exists(PRESETS_DIR):
        return []
    presets = sorted([p.stem for p in Path(PRESETS_DIR).glob("*.json") if p.is_file()])
    return presets


def save_preset(preset_name: str, params: Dict) -> None:
    """Saves a dictionary of parameters to a JSON file."""
    if not preset_name:
        return
    os.makedirs(PRESETS_DIR, exist_ok=True)
    filepath = get_preset_path(preset_name)
    with open(filepath, "w") as f:
        json.dump(params, f, indent=2)


def load_preset(preset_name: str) -> Optional[Dict]:
    """Loads a preset JSON file into a dictionary."""
    filepath = get_preset_path(preset_name)
    if not filepath.exists():
        return None
    with open(filepath, "r") as f:
        return json.load(f)
