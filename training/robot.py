"""Materialize the actual native mode15 EntityCfg, without a Simulation."""

from dataclasses import dataclass
import re

import numpy as np
import torch

from paths import N, leaf


def native_asset():
    import active_adaptation as aa

    if aa.get_backend() is None:
        aa.set_backend("mjlab")
    if aa.get_backend() != "mjlab":
        raise ValueError("This integration is pinned to the native mjlab owner")
    module = leaf(
        "recipe_native_g1_asset", N / "projects/mimic-lite/mimic_lite/assets/g1.py"
    )
    return module._build_g1_cfg(15, "mjlab")


def make_scene(num_envs, device="cpu"):
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    asset = native_asset()

    class NativeScene(Scene):
        def compile(self):
            model = super().compile()
            nominal_mujoco().apply(model)
            return model

    # Same TerrainEntityCfg and solver overrides as native MjlabBackendEnv.
    return NativeScene(
        SceneCfg(
            num_envs=num_envs,
            env_spacing=2.5,
            entities={"robot": asset.config},
            terrain=TerrainEntityCfg(terrain_type="plane"),
        ),
        device=device,
    )


def nominal_mujoco():
    from mjlab.sim import MujocoCfg

    return MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)


def floor_ids(model):
    import mujoco

    return [
        i
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        and not model.geom(i).name.startswith("robot/")
    ]


def clean(name):
    return name.removeprefix("robot/")


@dataclass
class RobotBinding:
    model: object
    default_q: torch.Tensor
    scale: torch.Tensor
    joint_ids: list
    qpos_ids: list
    qvel_ids: list
    actuator_ids: list
    body_ids: dict
    collision_geom_ids: list
    rows: list


def bind(model, device="cpu"):
    names = {clean(model.joint(i).name): i for i in range(model.njnt)}
    if not set(G1_JOINT_NAMES) <= names.keys():
        raise ValueError("mode15 must expose all original29 SM joints")
    asset = native_asset().config
    defaults = []
    rows = []
    jids, aids = [], []
    for name in G1_JOINT_NAMES:
        jid = names[name]
        matches = [i for i in range(model.nu) if int(model.actuator_trnid[i, 0]) == jid]
        if len(matches) != 1:
            raise ValueError((name, "expected one native position actuator", matches))
        aid = matches[0]
        if not np.allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]):
            raise ValueError(
                (name, "non-unit actuator gear needs an explicit contract")
            )
        kp = float(model.actuator_gainprm[aid, 0])
        kd = -float(model.actuator_biasprm[aid, 2])
        if kp <= 0 or not np.isclose(model.actuator_biasprm[aid, 1], -kp):
            raise ValueError((name, "not a native affine position actuator"))
        if not model.actuator_forcelimited[aid]:
            raise ValueError((name, "native effort limit is not finite"))
        low, high = map(float, model.actuator_forcerange[aid])
        if not np.isclose(-low, high) or high <= 0:
            raise ValueError((name, "asymmetric effort limit"))
        default = 0.0
        for pattern, value in asset.init_state.joint_pos.items():
            if re.fullmatch(pattern, name):
                default = float(value)
        jids.append(jid)
        aids.append(aid)
        defaults.append(default)
        rows.append(
            dict(
                joint=name,
                joint_id=jid,
                actuator_id=aid,
                Kp=kp,
                Kd=kd,
                effort=high,
                scale=0.25 * high / kp,
                default_q=default,
                limits=model.jnt_range[jid].tolist(),
            )
        )
    body_ids = {clean(model.body(i).name): i for i in range(1, model.nbody)}
    collisions = [
        i
        for i in range(model.ngeom)
        if model.geom(i).name.startswith("robot/")
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    return RobotBinding(
        model,
        torch.tensor(defaults, dtype=torch.float32, device=device),
        torch.tensor([r["scale"] for r in rows], dtype=torch.float32, device=device),
        jids,
        [int(model.jnt_qposadr[j]) for j in jids],
        [int(model.jnt_dofadr[j]) for j in jids],
        aids,
        body_ids,
        collisions,
        rows,
    )


G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
