# Adapted from Uni-SuperMimic, revision faf7818e8abac3ec6abb8f00e6e904d3577f800b.
# Copyright 2026 UniSuperMimic Authors. Apache-2.0; see LICENSE and NOTICE.
# Modified for SuperMimicLite's native G1 integration.

"""G1 motion commands, actions, and domain randomization."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
import torch
from unilab.base.reset_selectors import resolve_reset_env_ids
from unilab.managers import (
    ActionTerm,
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    EventTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)
from .geometry import flatten_state, mat_to_quat, quat_to_mat, so3_exp
from .contracts import State
from .task import MDP, NUM_J, XI_DIM, MAX_STEP_REWARD
from kinematics import G1Kinematics
from reference import MixtureMotion

if TYPE_CHECKING:
    from g1_env import G1Env, G1Entity

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
BIN_FRAMES = 50


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    """G1 motion-command and episode settings."""

    entity_name: str = "robot"
    eval_mode: bool = False
    gamma: float = 0.99
    reset_noise: bool = True
    max_episode_steps: int = 500
    goal_frames: int = 1
    context_dim: int = XI_DIM
    resampling_time_range: tuple[float, float] = (1000000000.0, 1000000000.0)

    def build(self, env: G1Env) -> MotionCommand:
        return MotionCommand(self, env, env.collection)


class MotionCommand(CommandTerm):
    """Reference motion, shared simulation context, and episode bookkeeping."""

    def __init__(self, cfg, env, collection):
        self._validate_cfg(cfg)
        CommandTerm.__init__(self, cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.kin = G1Kinematics(env.physics.binding.model, self._device)
        self.motion = MixtureMotion(collection, self.kin)
        self.default_q = self.robot.data.default_joint_pos(self._device).clone()
        self.mdp = MDP(self.kin, self.motion, self.default_q[0].clone())
        self.gamma, self.eval_mode = (float(cfg.gamma), cfg.eval_mode)
        self.reset_noise = cfg.reset_noise and (not cfg.eval_mode)
        self.max_episode_steps = 10**9 if cfg.eval_mode else cfg.max_episode_steps
        self._env_origins = env.scene.env_origins(self._device)
        self.f = torch.zeros(self.num_envs, dtype=torch.int64, device=self._device)
        self.xi = torch.zeros(self.num_envs, 33, device=self._device)
        self.a_prev = torch.zeros(self.num_envs, 29, device=self._device)
        self.ep_step = torch.zeros_like(self.f)
        self.start_bin = torch.zeros_like(self.f)
        self.ep_disc_ret = torch.zeros(self.num_envs, device=self._device)
        self.ep_ret = torch.zeros_like(self.ep_disc_ret)
        self.bin_G = torch.zeros(self.motion.num_bins, device=self._device)
        self.bin_w_avg = torch.full_like(self.bin_G, 1 / self.motion.num_bins)
        self.last_episode_stats = {}
        self._requested_frames = None
        self._read_phase_bound = False
        self.reset_draw_counts = torch.zeros(
            len(collection.datasets), device=self._device, dtype=torch.long
        )
        self.source_frame_counts = torch.zeros_like(self.reset_draw_counts)
        self.source_done_counts = torch.zeros_like(self.reset_draw_counts)
        self.source_hard_counts = torch.zeros_like(self.reset_draw_counts)

    cfg: MotionCommandCfg

    @staticmethod
    def _validate_cfg(cfg: MotionCommandCfg) -> None:
        if not isinstance(cfg.entity_name, str) or not cfg.entity_name:
            raise ValueError("MotionCommandCfg entity_name must be non-empty")
        if not isinstance(cfg.eval_mode, bool) or not isinstance(cfg.reset_noise, bool):
            raise TypeError("MotionCommandCfg eval_mode and reset_noise must be bool")
        if isinstance(cfg.gamma, bool) or not isinstance(cfg.gamma, (int, float)):
            raise TypeError("MotionCommandCfg gamma must be a real number")
        if not math.isfinite(cfg.gamma) or not 0.0 < cfg.gamma < 1.0:
            raise ValueError("MotionCommandCfg gamma must be in (0, 1)")
        if isinstance(cfg.max_episode_steps, bool) or cfg.max_episode_steps <= 0:
            raise ValueError("MotionCommandCfg max_episode_steps must be positive")
        if isinstance(cfg.goal_frames, bool) or cfg.goal_frames <= 0:
            raise ValueError("MotionCommandCfg goal_frames must be positive")
        if cfg.context_dim != XI_DIM:
            raise ValueError(f"SuperMimic requires context_dim={XI_DIM}")

    @property
    def tensor_body_names(self) -> tuple[str, ...]:
        return ("pelvis",)

    @property
    def command(self) -> torch.Tensor:
        return self.goal(self.cfg.goal_frames)

    def bind_read_phase(self) -> None:
        read_plan = self._env.scene.read_plan
        if read_plan is None:
            raise RuntimeError(
                "SuperMimic requires the compiled scene tensor read plan"
            )
        read_plan.refresh()
        self._read_phase_bound = True

    def state(self) -> State:
        """Read the current pelvis and joint state in the SuperMimic convention."""
        read_plan = getattr(self._env.scene, "read_plan", None)
        if read_plan is None or not read_plan.ready:
            if self._read_phase_bound:
                raise RuntimeError(
                    "SuperMimic requires a refreshed scene tensor read phase"
                )
            return self.mdp.ref_state(self.f)
        body = read_plan.body_tensor_view(self.robot, self.tensor_body_names)
        joint = read_plan.joint_tensor_view(self.robot)
        return (
            body.pos_w[:, 0] - self._env_origins,
            quat_to_mat(body.quat_w[:, 0]),
            body.lin_vel_w[:, 0],
            body.ang_vel_w[:, 0],
            joint.joint_pos,
            joint.joint_vel,
        )

    def observation(self) -> torch.Tensor:
        return self.mdp.obs(self.state(), self.a_prev, self.xi, self.f)

    def goal(self, num_frames: int) -> torch.Tensor:
        return self.mdp.goal(self.f, num_frames)

    def request_frames(self, frames: torch.Tensor) -> None:
        """Set per-environment frames for the next full reset."""
        if not isinstance(frames, torch.Tensor):
            raise TypeError("SuperMimic reset frames must be a torch.Tensor")
        if frames.shape != (self.num_envs,):
            raise ValueError(
                f"SuperMimic reset frames must have shape {(self.num_envs,)}"
            )
        if frames.dtype not in (torch.int32, torch.int64):
            raise TypeError("SuperMimic reset frames must use an integer dtype")
        frames = frames.to(device=self._device, dtype=torch.int64).clone()
        if bool((frames < 0).any()) or bool((frames >= self.motion.num_frames).any()):
            raise ValueError("SuperMimic reset frames are outside the motion library")
        self._requested_frames = frames

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, reset_mask: torch.Tensor) -> None:
        if bool(reset_mask.any()):
            raise RuntimeError(
                "SuperMimic commands resample only through the reset lifecycle"
            )

    def refresh_reset(self, reset_mask: torch.Tensor) -> None:
        del reset_mask

    def post_compute_reset(self) -> None:
        return

    def _draw_frames(self, ids):
        if self._requested_frames is not None:
            assert ids.numel() == self.num_envs
            frames, self._requested_frames = (self._requested_frames, None)
            return frames
        current_clip = self.motion.clip_id[self.f[ids]]
        sample = self.motion.collection.sample_motion(
            ids,
            terminated_t=self.f[ids] - self.motion.clip_start[current_clip],
            rewind_mask=torch.zeros(ids.numel(), device=self._device, dtype=torch.bool),
            rewind_steps=torch.zeros(
                ids.numel(), device=self._device, dtype=torch.long
            ),
        )
        frames = self.motion.clip_start[sample.motion_id] + sample.start_t
        self.reset_draw_counts += torch.bincount(
            self.motion.clip_source[sample.motion_id],
            minlength=self.reset_draw_counts.numel(),
        )
        return frames

    def reset(self, env_ids=None, *, publish_metrics=True):
        del publish_metrics
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self._device,
            owner="MixtureCommand.reset",
        )
        reset_mask = self._reset_mask_buffer.resolve(ids, owner="MixtureCommand.reset")
        if ids.numel() == 0:
            return {}
        frames = self._draw_frames(ids)
        bins = self.motion.frame_bin[frames]
        full_frames = self.f.clone()
        full_frames[ids] = frames
        p, rotation, v, w, q, qd = (
            value.clone() for value in self.motion.ref_state(full_frames)
        )
        if self.reset_noise:
            selected_p, selected_R, selected_v, selected_w, selected_q, _ = (
                value[ids].clone() for value in (p, rotation, v, w, q, qd)
            )
            count = ids.numel()
            selected_p += self._rng.uniform(
                torch.tensor([-0.05, -0.05, -0.01], device=self._device),
                torch.tensor([0.05, 0.05, 0.01], device=self._device),
                (count, 3),
            )
            rotation_noise = self._rng.uniform(
                torch.tensor([-0.1, -0.1, -0.2], device=self._device),
                torch.tensor([0.1, 0.1, 0.2], device=self._device),
                (count, 3),
            )
            selected_R = so3_exp(rotation_noise) @ selected_R
            selected_v += self._rng.uniform(
                torch.tensor([-0.5, -0.5, -0.2], device=self._device),
                torch.tensor([0.5, 0.5, 0.2], device=self._device),
                (count, 3),
            )
            selected_w += self._rng.uniform(
                torch.tensor([-0.52, -0.52, -0.78], device=self._device),
                torch.tensor([0.52, 0.52, 0.78], device=self._device),
                (count, 3),
            )
            selected_q += self._rng.uniform(-0.1, 0.1, (count, NUM_J))
            p[ids], rotation[ids], v[ids], w[ids], q[ids] = (
                selected_p,
                selected_R,
                selected_v,
                selected_w,
                selected_q,
            )
        q = torch.clamp(q, self.kin.q_lo, self.kin.q_hi)
        root_state = torch.cat((p + self._env_origins, mat_to_quat(rotation), v, w), -1)
        self.robot.write_motion_state_tensor_to_sim(
            root_state=root_state, position=q, velocity=qd, reset_mask=reset_mask
        )
        self.f[ids] = frames
        self.ep_step[ids] = 0
        self.a_prev[ids] = 0.0
        self.start_bin[ids] = bins
        self.ep_disc_ret[ids] = 0.0
        self.ep_ret[ids] = 0.0
        self.time_left[ids] = self._resampling_time_range[0]
        self.command_counter[ids] += 1
        return {}

    def _update_command(self):
        reward = cast("G1Env", self._env).reward_buf
        self.ep_disc_ret.add_(self.gamma ** self.ep_step.float() * reward)
        self.ep_ret.add_(reward)
        self.f.add_(1)
        self.ep_step.add_(1)
        self.a_prev.copy_(self._env.action_manager.action)
        if self.eval_mode:
            self.last_episode_stats = {}
        else:
            self.last_episode_stats = update_start_bin_returns(
                self.bin_G,
                self.start_bin,
                self.ep_step,
                self.ep_ret,
                self.ep_disc_ret,
                self._env.termination_manager.terminated,
                self._env.reset_buf,
                self.gamma,
            )
        source = self.motion.clip_source[self.motion.clip_id[self.f - 1]]
        self.source_frame_counts += torch.bincount(
            source, minlength=self.reset_draw_counts.numel()
        )
        self.source_done_counts += torch.bincount(
            source[self._env.reset_buf], minlength=self.reset_draw_counts.numel()
        )
        self.source_hard_counts += torch.bincount(
            source[self._env.termination_manager.terminated],
            minlength=self.reset_draw_counts.numel(),
        )


def update_start_bin_returns(
    bin_G: torch.Tensor,
    start_bin: torch.Tensor,
    ep_step: torch.Tensor,
    ep_ret: torch.Tensor,
    ep_disc_ret: torch.Tensor,
    hard: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Update completed-episode returns and return done-row logging statistics."""
    if not bool(done.any()):
        return {}
    lengths = ep_step[done].float()
    bootstrap = torch.where(
        hard[done], 0.0, gamma**lengths * ep_ret[done] / lengths / (1.0 - gamma)
    )
    returns = ep_disc_ret[done] + bootstrap
    bins = start_bin[done]
    bin_G.index_put_((bins,), 0.8 * bin_G[bins] + 0.2 * returns)
    return {
        "ep_len": lengths,
        "ep_ret_norm": ep_ret[done] / (lengths * MAX_STEP_REWARD),
    }


def make_current_info(
    state: State,
    frames: torch.Tensor,
    context: torch.Tensor,
    previous_action: torch.Tensor,
    ep_step: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Materialize the learner's current-state input without backend views."""
    return {
        "s": flatten_state(state).detach().clone(),
        "f": frames.detach().clone(),
        "xi": context.detach().clone(),
        "a_prev": previous_action.detach().clone(),
        "ep_step": ep_step.detach().clone(),
    }


def make_transition_info(
    state: State,
    next_state: State,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    context: torch.Tensor,
    reward: torch.Tensor,
    frames: torch.Tensor,
    ep_step: torch.Tensor,
    hard: torch.Tensor,
    timeout: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Materialize one pre-reset transition, including both done causes."""
    return {
        "s": flatten_state(state).detach().clone(),
        "s1": flatten_state(next_state).detach().clone(),
        "a": action.detach().clone(),
        "a_prev": previous_action.detach().clone(),
        "xi": context.detach().clone(),
        "r": reward.detach().clone(),
        "f": frames.detach().clone(),
        "ep_step": ep_step.detach().clone(),
        "term": hard.detach().clone(),
        "done": (hard | timeout).detach().clone(),
    }


def joint_action_target(
    default_q: torch.Tensor,
    context: torch.Tensor,
    alpha: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Build natural-joint-order position targets from the shared context."""
    return default_q + context[:, 1:30] + alpha * action


@dataclass(kw_only=True)
class JointActionCfg(ActionTermCfg):
    """Natural-joint-order SuperMimic position-action configuration."""

    entity_name: str = "robot"
    command_name: str = "motion"

    def build(self, env: G1Env) -> JointAction:
        return JointAction(self, env)


class JointAction(ActionTerm):
    """Apply default pose, episode joint offsets, and per-joint action scaling."""

    cfg: JointActionCfg

    def __init__(self, cfg: JointActionCfg, env: G1Env):
        super().__init__(cfg, env)
        self._command = _command(env, cfg.command_name)
        if self._command.robot is not self._entity:
            raise ValueError(
                "SuperMimic action and command must control the same entity"
            )
        self._raw_actions = torch.zeros(
            (self.num_envs, NUM_J), dtype=torch.float32, device=env.device
        )
        self._target = torch.zeros_like(self._raw_actions)
        self._alpha = env.physics.binding.scale.to(env.device).clone()

    @property
    def action_dim(self) -> int:
        return NUM_J

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def tensor_target(self) -> torch.Tensor:
        return self._target

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)

    def apply_actions(self) -> None:
        self._target.copy_(
            joint_action_target(
                self._command.default_q,
                self._command.xi,
                self._alpha,
                self._raw_actions,
            )
        )
        cast("G1Entity", self._entity).set_full_width_joint_position_target(
            self._target
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self._env.device,
            owner="JointAction.reset",
        )
        self._raw_actions[ids] = 0.0


def supermimic_reward(env: G1Env, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    return command.mdp.reward(
        command.state(), command.f + 1, env.action_manager.action, command.a_prev
    )


def supermimic_termination(env: G1Env, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    return command.mdp.termination(command.state(), command.f + 1)


def supermimic_timeout(env: G1Env, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    return command.ep_step + 1 >= command.max_episode_steps


def supermimic_clip_end(env: G1Env, command_name: str = "motion") -> torch.Tensor:
    command = _command(env, command_name)
    clip_end = command.motion.clip_end[command.motion.clip_id[command.f]]
    return command.f + 1 >= clip_end


class ContextRandomization(ManagerTermBase):
    """Sample startup friction/CoM or replay those values on later resets."""

    uses_startup_rows = True

    def __init__(self, cfg: EventTermCfg, env: G1Env):
        super().__init__(env)
        if cfg.mode not in ("startup", "reset"):
            raise ValueError("ContextRandomization requires startup or reset mode")
        self._startup = cfg.mode == "startup"
        self._command_name = _event_command_name(cfg)
        self._randomize_friction = cfg.params.get("randomize_friction", True)
        if not isinstance(self._randomize_friction, bool):
            raise TypeError("ContextRandomization randomize_friction must be bool")
        asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("ContextRandomization asset_cfg must be SceneEntityCfg")
        self._robot = cast("G1Entity", env.scene[asset_cfg.name])
        self._geom_ids = None
        self._friction_defaults = None
        if self._randomize_friction:
            self._geom_ids, self._friction_defaults = (
                self._robot.bind_geom_friction_write(
                    asset_cfg.geom_ids, term_name="supermimic_context"
                )
            )
        torso_ids, _ = self._robot.find_bodies("torso_link", preserve_order=True)
        self._torso_ids, self._ipos_defaults = self._robot.bind_body_ipos_write(
            torso_ids, term_name="supermimic_context"
        )
        ground_cfg = cfg.params.get("ground_asset_cfg")
        if ground_cfg is not None and (not isinstance(ground_cfg, SceneEntityCfg)):
            raise TypeError(
                "ContextRandomization ground_asset_cfg must be SceneEntityCfg or None"
            )
        self._ground = (
            None
            if ground_cfg is None or not self._randomize_friction
            else cast("G1Entity", env.scene[ground_cfg.name])
        )
        self._ground_geom_ids = None
        self._ground_friction_defaults = None
        if self._ground is not None and ground_cfg is not None:
            self._ground_geom_ids, self._ground_friction_defaults = (
                self._ground.bind_geom_friction_write(
                    ground_cfg.geom_ids, term_name="supermimic_ground_context"
                )
            )

    def __call__(
        self,
        env: G1Env,
        env_ids: torch.Tensor | None,
        command_name: str = "motion",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        ground_asset_cfg: SceneEntityCfg | None = None,
        randomize_friction: bool = True,
    ) -> None:
        del command_name, asset_cfg, ground_asset_cfg, randomize_friction
        ids = resolve_reset_env_ids(
            env_ids,
            num_envs=env.num_envs,
            device=env.device,
            owner="ContextRandomization",
        )
        command = _command(env, self._command_name)
        if self._startup:
            expected = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
            if not torch.equal(ids, expected):
                raise ValueError(
                    "SuperMimic startup context requires every environment row"
                )
            if command.eval_mode or not self._randomize_friction:
                command.xi[:, 0] = 1.0
            else:
                command.xi[:, 0] = env.rng.uniform(0.3, 1.2, (env.num_envs,))
            if command.eval_mode:
                command.xi[:, 30:33] = 0.0
            else:
                command.xi[:, 30:33] = env.rng.uniform(
                    torch.tensor([-0.025, -0.05, -0.05], device=env.device),
                    torch.tensor([0.025, 0.05, 0.05], device=env.device),
                    (env.num_envs, 3),
                )
        if self._randomize_friction:
            assert self._geom_ids is not None and self._friction_defaults is not None
            friction = _selected_defaults(self._friction_defaults, ids)
            friction[..., 0] = command.xi[ids, :1]
            self._robot.write_geom_friction_to_sim(
                friction,
                geom_ids=self._geom_ids,
                env_ids=ids,
                term_name="supermimic_context",
            )
        ipos = _selected_defaults(self._ipos_defaults, ids)
        ipos += command.xi[ids, 30:33].unsqueeze(1)
        if self._startup:
            self._robot.record_startup_body_ipos_to_sim(
                ipos,
                body_ids=self._torso_ids,
                env_ids=ids,
                term_name="supermimic_context",
            )
        else:
            self._robot.write_body_ipos_to_sim(
                ipos,
                body_ids=self._torso_ids,
                env_ids=ids,
                term_name="supermimic_context",
            )
        if self._ground is not None:
            assert self._ground_friction_defaults is not None
            ground_friction = _selected_defaults(self._ground_friction_defaults, ids)
            ground_friction[..., 0] = 1.0 if command.eval_mode else 0.3
            self._ground.write_geom_friction_to_sim(
                ground_friction,
                geom_ids=self._ground_geom_ids,
                env_ids=ids,
                term_name="supermimic_ground_context",
            )


class JointOffsets(ManagerTermBase):
    """Sample reset joint offsets into the command's only simulation context."""

    def __init__(self, cfg: EventTermCfg, env: G1Env):
        super().__init__(env)
        if cfg.mode != "reset":
            raise ValueError("JointOffsets requires reset mode")
        self._command_name = _event_command_name(cfg)

    def __call__(
        self,
        env: G1Env,
        env_ids: torch.Tensor | None,
        command_name: str = "motion",
    ) -> None:
        del command_name
        ids = resolve_reset_env_ids(
            env_ids, num_envs=env.num_envs, device=env.device, owner="JointOffsets"
        )
        command = _command(env, self._command_name)
        if command.eval_mode:
            command.xi[ids, 1:30] = 0.0
        else:
            command.xi[ids, 1:30] = env.rng.uniform(-0.01, 0.01, (ids.numel(), NUM_J))


def _event_command_name(cfg: EventTermCfg) -> str:
    command_name = cfg.params.get("command_name", "motion")
    if not isinstance(command_name, str) or not command_name:
        raise ValueError("SuperMimic event command_name must be non-empty")
    return command_name


def _selected_defaults(defaults: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    values = defaults.to(device=env_ids.device, dtype=torch.float32)
    if values.ndim == 2:
        return values.expand((env_ids.numel(), *values.shape)).clone()
    if values.ndim == 3:
        return values.index_select(0, env_ids).clone()
    raise ValueError(f"SuperMimic event defaults have invalid shape {values.shape}")


def _command(env: G1Env, command_name: str) -> MotionCommand:
    try:
        command = env.command_manager.get_term(command_name)
    except KeyError as exc:
        raise KeyError(f"SuperMimic command term '{command_name}' not found") from exc
    if not isinstance(command, MotionCommand):
        raise TypeError(
            f"Command term '{command_name}' is {type(command).__name__}, expected MotionCommand"
        )
    return command
