"""Opt-in plugin load for the unmodified author's `python -m` child command."""

import os, sys

# GLFW probes its shared library in `python -c` children. Never recursively load
# sim2real there: doing so would start another GLFW probe and deadlock imports.
if (
    os.environ.get("G1_LEADERBOARD_REGISTER") == "1"
    and sys.argv[0] == "-m"
    and "--policy-config" in sys.argv
):
    from g1_registry import register

    register()
    import g1_observations  # noqa: F401 -- registers the explicit observation plugin
