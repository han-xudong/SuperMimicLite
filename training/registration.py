"""G1-only registration and recipe preparation; never launches a runner."""

import importlib
from ppo_command_owner import load_native_command
from robot import native_asset


def register_native_G1():
    load_native_command()
    for module in ["actions", "motion", "observations", "rewards", "terminations"]:
        importlib.import_module("mimic_lite.tasks." + module)
    native_asset()
