"""Original native command, with one explicit cold-path storage dispatch.

The method's code object is untouched. Its private globals copy substitutes
only the dataset factory; native modules and their globals remain unchanged.
All native transforms, clocks, rewind, reward and reset bodies are retained.
"""

import importlib, sys, types
from paths import N
from host_reference_bank import native_loader_with_host_bank


def load_native_command():
    for name, folder in [
        ("mimic_lite", N / "projects/mimic-lite/mimic_lite"),
        ("mimic_lite.tasks", N / "projects/mimic-lite/mimic_lite/tasks"),
    ]:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(folder)]
            sys.modules[name] = package
    return importlib.import_module("mimic_lite.tasks.command")


def command_class():
    original = load_native_command().RobotTracking
    namespace = dict(original._initialize.__globals__)
    namespace["load_motion_dataset_collection"] = native_loader_with_host_bank
    method = types.FunctionType(
        original._initialize.__code__,
        namespace,
        original._initialize.__name__,
        original._initialize.__defaults__,
        original._initialize.__closure__,
    )
    method.__kwdefaults__ = original._initialize.__kwdefaults__

    class RobotTrackingHostBank(original, namespace="supermimiclite"):
        _initialize = method

    return RobotTrackingHostBank


RobotTrackingHostBank = command_class()
