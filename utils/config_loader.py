"""
utils/config_loader.py
──────────────────────
Load and expose project config from configs/config.yaml.
"""

import yaml
from pathlib import Path
from utils.logger import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load YAML config file and return as a nested dict.

    Args:
        config_path: relative or absolute path to config.yaml

    Returns:
        dict: full configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Config loaded from: {path.resolve()}")
    return cfg


def get_corpus_config(cfg: dict) -> dict:
    return cfg.get("corpus", {})


def get_retrieval_config(cfg: dict) -> dict:
    return cfg.get("retrieval", {})


def get_generation_config(cfg: dict) -> dict:
    return cfg.get("generation", {})


def get_confidence_config(cfg: dict) -> dict:
    return cfg.get("confidence", {})
