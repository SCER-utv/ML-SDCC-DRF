import json
import os

_cached_config = None


# Loads basic system configuration from the local config.json file
def load_config():
    global _cached_config

    # Return the cached configuration if already loaded to avoid redundant disk reads
    if _cached_config is not None:
        return _cached_config

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    config_path = os.path.join(root_dir, 'config', 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f" [CRITICAL ERROR] Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f" [CRITICAL ERROR] Invalid JSON format in {config_path}: {e}")

    # Attach the root directory for potential local path resolutions
    config['_root_dir'] = root_dir

    _cached_config = config
    return config