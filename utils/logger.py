import os
import warnings
from typing import Any, Dict, Optional

import hydra
from lightning.fabric.loggers.logger import Logger
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem


def get_logger(cfg: Dict[str, Any]) -> Optional[Logger]:
    logger = None
    if cfg.metric.log_level > 0:
        if "tensorboard" in cfg.metric.logger._target_.lower():
            root_dir = os.path.join("logs", "runs", cfg.root_dir)
            if root_dir != cfg.metric.logger.root_dir:
                warnings.warn(
                    "The specified root directory for the TensorBoardLogger is different from the experiment one, "
                    "so the logger one will be ignored and replaced with the experiment root directory",
                    UserWarning,
                )
            if cfg.run_name != cfg.metric.logger.name:
                warnings.warn(
                    "The specified name for the TensorBoardLogger is different from the `run_name` of the experiment, "
                    "so the logger one will be ignored and replaced with the experiment `run_name`",
                    UserWarning,
                )
            cfg.metric.logger.root_dir = root_dir
            cfg.metric.logger.name = cfg.run_name
        logger = hydra.utils.instantiate(cfg.metric.logger, _convert_="all")
    return logger


def get_log_dir(root_dir: str, run_name: str) -> str:
    """Return and, if necessary, create the log directory.

    Args:
        root_dir (str): the root directory of the experiment.
        run_name (str): the name of the experiment.

    Returns:
        The log directory of the experiment.
    """
    save_dir = os.path.join("logs", "runs", root_dir, run_name)
    fs = get_filesystem(root_dir)
    try:
        listdir_info = fs.listdir(save_dir)
        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if _is_dir(fs, d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            version = 0
        else:
            version = max(existing_versions) + 1
        log_dir = os.path.join(save_dir, f"version_{version}")
    except OSError:
        warnings.warn("Missing logger folder: %s" % save_dir, UserWarning)
        log_dir = os.path.join(save_dir, f"version_{0}")

    os.makedirs(log_dir, exist_ok=True)
    return log_dir
