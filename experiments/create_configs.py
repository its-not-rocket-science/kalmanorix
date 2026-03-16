"""
Create example configuration files for Milestone 2.1 experiments.
"""

from pathlib import Path
from config import create_config_dir

if __name__ == "__main__":
    config_dir = Path("experiments/configs")
    create_config_dir(config_dir)
    print(f"Created example configurations in {config_dir}")
    for path in config_dir.glob("*.yaml"):
        print(f"  {path.name}")
