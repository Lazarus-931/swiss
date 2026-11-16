"""Refresh tests/model_config.json from Hugging Face: python tests/fetch_configs.py
[REPO ...]."""

import json
import sys
import urllib.request
from pathlib import Path

PATH = Path(__file__).parent / "model_config.json"

if __name__ == "__main__":
    configs = json.loads(PATH.read_text())
    for repo in sys.argv[1:] or list(configs):
        with urllib.request.urlopen(
            f"https://huggingface.co/{repo}/resolve/main/config.json",
        ) as response:
            configs[repo] = json.loads(response.read())
    PATH.write_text(json.dumps(configs, indent=1) + "\n")
