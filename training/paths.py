"""Repository-relative owners; no historical machine paths or experiment imports."""

import importlib.util, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = ROOT / "third_party/active-adaptation"


def leaf(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
