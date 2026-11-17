# Contains modified Uni-SuperMimic code under Apache-2.0; see LICENSE and NOTICE.
# G1 integration also uses MimicLite and Active Adaptation; see THIRD_PARTY.md.

"""MimicLite G1 physics and the SuperMimic training environment."""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace as NS
import torch
from unilab.managers import EventTermCfg, ManagerRng, SceneEntityCfg
from supermimic.geometry import quat_to_mat
from supermimic.contracts import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from supermimic.terms import (
    MotionCommand,
    MotionCommandCfg,
    JointAction,
    JointActionCfg,
    ContextRandomization,
    JointOffsets,
    make_current_info,
    make_transition_info,
    supermimic_reward,
    supermimic_termination,
    supermimic_timeout,
    supermimic_clip_end,
)
from robot import make_scene, bind, nominal_mujoco, floor_ids, G1_JOINT_NAMES


class G1Physics:
    def __init__(self, num_envs, device="cuda:0"):
        device = str(device)
        from active_adaptation.envs.backends.mjlab.adapter import (
            MjlabSceneAdapter,
            MjlabSimAdapter,
        )
        from mjlab.sim import Simulation, SimulationCfg
        from unisim.backend.mjwarp.randomization import DERIVED_FIELDS

        self.num_envs, self.device = (num_envs, torch.device(device))
        self.scene = make_scene(num_envs, device)
        model = self.scene.compile()
        self.binding = bind(model, device)
        self.sim = Simulation(
            num_envs=num_envs,
            model=model,
            device=device,
            cfg=SimulationCfg(nconmax=128, njmax=640, mujoco=nominal_mujoco()),
        )
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = self.scene.entities["robot"]
        self.native_scene = MjlabSceneAdapter(self.scene, self.sim)
        self.native_sim = MjlabSimAdapter(self.sim, scene=self.scene)
        self.sim.expand_model_fields(("geom_friction", "body_ipos", *DERIVED_FIELDS))
        self.joint_order = torch.tensor(
            [self.robot.joint_names.index(n) for n in G1_JOINT_NAMES], device=device
        )
        self.ctrl_ids = torch.tensor(self.binding.actuator_ids, device=device)
        self.geom_ids = torch.tensor(self.binding.collision_geom_ids, device=device)
        self.floor_ids = torch.tensor(floor_ids(model), device=device)
        self.torso_ids = torch.tensor(
            [self.binding.body_ids["torso_link"]], device=device
        )
        self.default_ipos = torch.tensor(
            model.body_ipos.copy(), device=device, dtype=torch.float32
        )
        self.default_friction = torch.tensor(
            model.geom_friction.copy(), device=device, dtype=torch.float32
        )
        self.sim.create_graph()
        self.native_sim.forward()

    def read(self):
        data = self.robot.data
        return (
            data.root_link_pos_w - self.scene.env_origins,
            quat_to_mat(data.root_link_quat_w),
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_w,
            data.joint_pos[:, self.joint_order],
            data.joint_vel[:, self.joint_order],
        )

    def write(self, root_state, position, velocity, reset_mask):
        ids = reset_mask.nonzero().flatten()
        self.robot.reset(ids)
        self.robot.write_root_link_pose_to_sim(root_state[ids, :7], env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(root_state[ids, 7:13], env_ids=ids)
        self.robot.write_joint_state_to_sim(
            position[ids], velocity[ids], joint_ids=self.joint_order, env_ids=ids
        )

    def set_target(self, value):
        self.sim.data.ctrl[:, self.ctrl_ids] = value

    def step(self):
        for _ in range(4):
            self.native_scene.zero_external_wrenches()
            self.native_sim.step()
            self.native_scene.update(0.005)
        self.native_sim.forward()

    def sync_reset(self, recompute=False):
        if recompute:
            from mjlab.managers.event_manager import RecomputeLevel

            self.sim.recompute_constants(RecomputeLevel.set_const)
        self.native_sim.forward()
        self.scene.update(0.0)


class G1Entity:
    def __init__(self, physics, ground=False):
        self.physics, self.ground = (physics, ground)
        self.joint_names = tuple(G1_JOINT_NAMES)
        self.data = NS(
            default_joint_pos=lambda device: physics.binding.default_q.to(
                device
            ).expand(physics.num_envs, -1)
        )

    def find_bodies(self, name, preserve_order=True):
        if name != "torso_link":
            raise ValueError(name)
        return (self.physics.torso_ids, [name])

    def bind_geom_friction_write(self, geom_ids, **kwargs):
        ids = self.physics.floor_ids if self.ground else self.physics.geom_ids
        return (ids, self.physics.default_friction[ids].clone())

    def bind_body_ipos_write(self, body_ids, **kwargs):
        return (body_ids, self.physics.default_ipos[body_ids].clone())

    def write_geom_friction_to_sim(self, values, geom_ids, env_ids, **kwargs):
        self.physics.sim.model.geom_friction[env_ids[:, None], geom_ids] = values

    def write_body_ipos_to_sim(self, values, body_ids, env_ids, **kwargs):
        self.physics.sim.model.body_ipos[env_ids[:, None], body_ids] = values

    record_startup_body_ipos_to_sim = write_body_ipos_to_sim

    def write_motion_state_tensor_to_sim(self, **kwargs):
        self.physics.write(**kwargs)

    def set_full_width_joint_position_target(self, value):
        self.physics.set_target(value)


class StateReadPlan:
    ready = True

    def __init__(self, physics):
        self.physics = physics

    def refresh(self):
        pass

    def body_tensor_view(self, robot, names):
        from supermimic.geometry import mat_to_quat

        p, R, v, w, _, _ = self.physics.read()
        return NS(
            pos_w=(p + self.physics.scene.env_origins)[:, None],
            quat_w=mat_to_quat(R)[:, None],
            lin_vel_w=v[:, None],
            ang_vel_w=w[:, None],
        )

    def joint_tensor_view(self, robot):
        *_, q, qd = self.physics.read()
        return NS(joint_pos=q, joint_vel=qd)


class G1Scene(dict):
    def __init__(self, physics):
        super().__init__(robot=G1Entity(physics), ground=G1Entity(physics, True))
        self.read_plan = StateReadPlan(physics)
        self.physics = physics

    def env_origins(self, device):
        return self.physics.scene.env_origins.to(device)


@dataclass
class StepState:
    obs: dict
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict
    final_observation: object = None


class G1Env:
    def __init__(self, collection, physics=None, seed=0, eval_mode=False):
        self.collection = collection
        self.num_envs, self.device = (collection.num_envs, collection.device)
        self.physics = (
            physics if physics is not None else G1Physics(self.num_envs, self.device)
        )
        self.rng = ManagerRng(
            device=self.device,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        self.scene = G1Scene(self.physics)
        self.command = MotionCommand(
            MotionCommandCfg(eval_mode=eval_mode, goal_frames=10), self, collection
        )
        self.command_manager = NS(get_term=lambda name: self.command)
        self.action = JointAction(JointActionCfg(), self)
        self.action_manager = NS(action=self.action.raw_action)
        self.termination_manager = NS(
            terminated=torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        )
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)
        self.startup_context = ContextRandomization(
            EventTermCfg(
                func=ContextRandomization,
                mode="startup",
                params={"ground_asset_cfg": SceneEntityCfg("ground")},
            ),
            self,
        )
        self.reset_context = ContextRandomization(
            EventTermCfg(
                func=ContextRandomization,
                mode="reset",
                params={"ground_asset_cfg": SceneEntityCfg("ground")},
            ),
            self,
        )
        self.offsets = JointOffsets(EventTermCfg(func=JointOffsets, mode="reset"), self)
        self.startup_context(self, None)
        self.physics.sync_reset(recompute=True)
        self.command.bind_read_phase()
        self.state = None
        self.init_state()

    @property
    def supermimic_mdp(self):
        return self.command.mdp

    def current_info(self):
        c = self.command
        return make_current_info(c.state(), c.f, c.xi, c.a_prev, c.ep_step)

    def reset(self, env_ids=None, **kwargs):
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids
        )
        self.reset_context(self, ids)
        self.offsets(self, ids)
        self.command.reset(ids)
        self.action.reset(ids)
        self.physics.sync_reset()
        info = {SUPERMIMIC_STATE_KEY: self.current_info()}
        if self.state is not None:
            self.state.info.update(info)
        return ({}, info)

    def reset_to_frames(self, frames):
        self.command.request_frames(frames)
        return self.reset()

    def init_state(self):
        if self.state is None:
            _, info = self.reset()
            self.state = StepState(
                {},
                self.reward_buf.clone(),
                self.reset_buf.clone(),
                self.reset_buf.clone(),
                info,
            )
        return self.state

    @torch.no_grad()
    def step(self, actions):
        if actions.shape != (self.num_envs, 29) or not bool(
            torch.isfinite(actions).all()
        ):
            raise ValueError("Invalid G1 actions")
        c = self.command
        before = self.current_info()
        before_state = tuple((t.clone() for t in c.state()))
        self.action.process_actions(actions)
        self.action.apply_actions()
        self.physics.step()
        after_state = tuple((t.clone() for t in c.state()))
        if not all((bool(torch.isfinite(t).all()) for t in after_state)):
            raise FloatingPointError(
                "Nonfinite native physics; learner update rejected"
            )
        self.reward_buf = supermimic_reward(self)
        hard = supermimic_termination(self)
        timeout = supermimic_timeout(self) | supermimic_clip_end(self)
        self.termination_manager.terminated = hard
        self.reset_buf = hard | timeout
        c._update_command()
        transition = make_transition_info(
            before_state,
            after_state,
            actions,
            before["a_prev"],
            before["xi"],
            self.reward_buf,
            before["f"],
            before["ep_step"],
            hard,
            timeout,
        )
        transition.update(c.last_episode_stats)
        ids = self.reset_buf.nonzero().flatten()
        if ids.numel():
            self.reset(ids)
        self.state = StepState(
            {},
            self.reward_buf,
            hard,
            timeout,
            {
                SUPERMIMIC_TRANSITION_KEY: transition,
                SUPERMIMIC_STATE_KEY: self.current_info(),
            },
        )
        return self.state

    def close(self):
        pass
