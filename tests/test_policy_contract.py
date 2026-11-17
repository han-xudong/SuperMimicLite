"""Check the joint/history failure boundary that invalidated the first PPO evaluation."""

import json
from pathlib import Path
import numpy as np
import pytest
from observation_core import PPOCore, SMCore, Reference

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    n = 12
    p = np.zeros((n, 3), np.float32)
    R = np.repeat(np.eye(3, dtype=np.float32)[None], n, 0)
    q = np.repeat(np.arange(29, dtype=np.float32)[None], n, 0)
    z = np.zeros_like(p)
    qd = np.zeros_like(q)
    return Reference(p, R, z, z, q, qd), (p[0], R[0], z[0], z[0], q[0], qd[0])


def test_ppo_uses_declared_joint_order():
    ref, state = fixture()
    order = list(reversed(range(29)))
    core = PPOCore(order)
    first = core.compute(state, np.zeros(29, np.float32), ref)
    assert not first["policy"].any() and not first["command"].any()
    second = core.compute(state, np.arange(29, dtype=np.float32), ref)
    np.testing.assert_array_equal(second["policy"][42 : 42 + 29], state[4][order])
    np.testing.assert_array_equal(second["command"][72:101], state[4][order])
    assert second["policy"].shape == (535,) and second["command"].shape == (304,)


def test_order_is_bijective():
    with pytest.raises(AssertionError):
        PPOCore([0] * 29)


def test_history_reset_and_end_clamp():
    ref, state = fixture()
    for core in [PPOCore(range(29)), SMCore(np.zeros(29, np.float32))]:
        first = core.compute(state, np.zeros(29, np.float32), ref)
        for i in range(80):
            out = core.compute(state, np.zeros(29, np.float32), ref)
        assert all(np.isfinite(x).all() for x in out.values())
        core.reset()
        again = core.compute(state, np.zeros(29, np.float32), ref)
        for key in first:
            np.testing.assert_array_equal(first[key], again[key])


def test_deploy_mapping_has_same_29_named_joints():
    import yaml

    for cell in ["P-L", "P-M", "S-L", "S-M"]:
        cfg = yaml.safe_load((ROOT / "configs/deploy" / f"{cell}.yaml").read_text())
        assert len(set(cfg["policy_joint_names"])) == 29
        assert set(cfg["policy_joint_names"]) == set(cfg["joint_names_simulation"])
        assert len(cfg["action_scale"]) == 29


def test_fixed_public_grid():
    spec = json.loads((ROOT / "manifests/motiondecode.json").read_text())
    assert sum(r["count"] for r in spec.values()) == 228
    for split, row in spec.items():
        names = (
            (ROOT / "manifests/evaluation" / split / "motion-list.txt")
            .read_text()
            .splitlines()
        )
        assert len(names) == len(set(names)) == row["count"]
