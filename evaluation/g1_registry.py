"""Cold G1-only registry: avoid downloading unrelated H2 assets at import time.

The original G1 config class, complete G1_CFG, physics, policy runtime and
scorer are imported unchanged. Only discovery of unrelated robots is omitted.
"""

import importlib, sys, types
from pathlib import Path


def register():
    name = "sim2real.config.robots"
    if name in sys.modules:
        return
    import sim2real

    package = types.ModuleType(name)
    package.__path__ = [str(Path(sim2real.__file__).parent / "config/robots")]
    sys.modules[name] = package
    base = importlib.import_module(name + ".base")
    g1 = importlib.import_module(name + ".g1")
    package.RobotCfg = base.RobotCfg
    package.G1_CFG = g1.G1_CFG

    def get_robot_cfg(robot):
        if str(robot).strip().lower() != "g1":
            raise ValueError("This repository evaluates G1 only")
        return g1.G1_CFG

    package.get_robot_cfg = get_robot_cfg
