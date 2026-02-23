"""
CLI entry point for prisme.

Usage:
    prisme input=/path/to/video.mp4 output=/path/to/output.mp4
    prisme --config-name example
"""

import hydra
from omegaconf import DictConfig

from prisme.runner import run


@hydra.main(version_base=None, config_path="configs", config_name="example")
def main(cfg: DictConfig) -> None:
    """
    Run the prisme pipeline with the given Hydra config.

    Args:
        cfg: Hydra config object populated from configs/example.yaml and CLI overrides.

    """
    run(cfg)


if __name__ == "__main__":
    main()
