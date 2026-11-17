"""Compose the original single-stage 32k recipes; only paths are portable."""

import importlib, json, os, sys, types
from pathlib import Path
from omegaconf import OmegaConf
from paths import N, ROOT


def native_ppo():
    name = "mimic_lite_learning"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(N / "projects/mimic-lite/mimic_lite_learning")]
        sys.modules[name] = package
    return importlib.import_module(name + ".ppo")


def data_config(data):
    root = Path(os.environ.get("G1_TRAIN_DATA", ROOT / "data/training")).resolve()
    cfg = json.loads((ROOT / "configs/training/mixture.json").read_text())
    if data == "L":
        cfg["motion_cfgs"] = {"lafan": cfg["motion_cfgs"]["lafan"]}
        cfg["motion_cfgs"]["lafan"]["weight"] = 1.0
    elif data != "M":
        raise ValueError(data)
    for v in cfg["motion_cfgs"].values():
        v["path"] = str(root / v["path"])
    return cfg


def ppo_config(data):
    native_ppo()
    cfg = OmegaConf.create(json.loads((ROOT / "configs/training/ppo.json").read_text()))
    cfg.task.command.motion_cfgs = data_config(data)["motion_cfgs"]
    return cfg


def sm_training_config():
    return json.loads((ROOT / "configs/training/supermimic.json").read_text())


def sm_config():
    from supermimic.learner import SuperMimicConfig

    return SuperMimicConfig(**sm_training_config()["algo"])
