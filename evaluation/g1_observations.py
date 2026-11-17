"""Explicit observation registry extension, not a physics/scoring monkeypatch."""

import numpy as np
from sim2real.rl_policy.observations.base import Observation
from observation_core import quat_matrix, Reference, PPOCore, SMCore


class NativeView(Observation, namespace="g1_leaderboard"):
    def __init__(self, env, method, key, policy_order, default_q):
        super().__init__(env)
        self.key = key
        if not hasattr(env, "g1_native_core"):
            env.g1_native_core = PPOCore(policy_order) if method == "P" else SMCore(default_q)
            env.g1_native_method = method
            env.g1_native_order = list(policy_order)
            env.g1_native_token = None
            sp = self.state_processor
            length = int(sp.motion_length)
            data = sp.motion_dataset.get_slice(sp.motion_ids, np.array([0]), np.arange(length))
            idx = sp.motion_body_names.index("pelvis")
            env.g1_native_reference = Reference(
                data.body_pos_w[0, :, idx],
                quat_matrix(data.body_quat_w[0, :, idx]),
                data.body_lin_vel_w[0, :, idx],
                data.body_ang_vel_w[0, :, idx],
                data.joint_pos[0],
                data.joint_vel[0],
            )
        assert env.g1_native_method == method and env.g1_native_order == list(policy_order)

    def reset(self):
        self.env.g1_native_core.reset()
        self.env.g1_native_token = None

    def update(self, data):
        sp = self.state_processor
        # IntegratedSim2Sim advances this once per completed policy step.
        token = int(self.env.total_inference_cnt)
        if self.env.g1_native_token == token:
            return
        R = quat_matrix(sp.root_quat_w)
        state = (
            sp.root_pos_w.copy(),
            R,
            sp.root_lin_vel_w.copy(),
            R @ sp.root_ang_vel_b,
            sp.joint_pos.copy(),
            sp.joint_vel.copy(),
        )
        for x in state:
            assert np.isfinite(x).all(), "nonfinite native observation state"
        out = self.env.g1_native_core.compute(
            state, np.asarray(data["action"], np.float32), self.env.g1_native_reference
        )
        assert all(np.isfinite(x).all() for x in out.values()), "nonfinite native inputs"
        self.env.g1_native_inputs = out
        self.env.g1_native_token = token

    def compute(self):
        return self.env.g1_native_inputs[self.key].copy()
