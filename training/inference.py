"""Inference-only native actor construction, without an optimizer or environment."""

from types import SimpleNamespace
import torch
from torch import nn
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from active_adaptation.learning.modules.vecnorm import VecNorm
from recipe_config import native_ppo


class PPOInference(nn.Module):
    def __init__(self, checkpoint, cfg):
        super().__init__()
        holder = nn.Module()
        holder.cfg = SimpleNamespace(**cfg["algo"])
        holder.device = "cpu"
        holder.action_dim = 29
        holder.actor_residual_gate = None
        self.actor = native_ppo().PPOPolicy._build_actor(
            holder, list(cfg["algo"]["actor_in_keys"])
        )
        self.actor(
            TensorDict(
                {"command": torch.zeros(1, 304), "policy": torch.zeros(1, 535)},
                batch_size=[1],
            )
        )
        self.actor.load_state_dict(checkpoint["policy"]["actor"], strict=True)
        self.norms = nn.ModuleDict(
            {
                key: VecNorm(dim, decay=0.9999)
                for key, dim in [("command", 304), ("policy", 535)]
            }
        )
        for key, norm in self.norms.items():
            norm.load_state_dict(
                {
                    k.removeprefix(key + "."): v
                    for k, v in checkpoint["policy"]["vecnorms"].items()
                    if k.startswith(key + ".")
                },
                strict=True,
            )
        self.eval().requires_grad_(False)

    def forward(self, policy, command):
        td = TensorDict(
            {"policy": policy[None], "command": command[None]}, batch_size=[1]
        )
        with VecNorm.freeze(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for k, norm in self.norms.items():
                td[k] = norm(td[k])
            self.actor(td)
        return td["action"][0]


class SMInference(nn.Module):
    def __init__(self, checkpoint, cfg):
        super().__init__()
        from supermimic import networks

        networks.AMP_DTYPE = torch.bfloat16 if cfg["amp"] == "bf16" else None
        self.hist = int(cfg["hist"])
        self.goal_dim = 73 * int(cfg["goal_frames"])
        self.policy = networks.Policy(
            99,
            self.goal_dim,
            hist=self.hist,
            z_dim=int(cfg["z_dim"]),
            num_joints=29,
            hidden=tuple(cfg["policy_hidden"]),
        )
        self.policy.load_state_dict(checkpoint["pi"], strict=True)
        self.eval().requires_grad_(False)

    def forward(self, proprioception_history, goal):
        return self.policy(
            proprioception_history.reshape(1, self.hist, 99), goal[None]
        )[0]
