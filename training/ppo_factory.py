"""Construct the MimicLite PPO environment and policy."""

import importlib

import hydra
from torchrl.envs.transforms import Compose, InitTracker, StepCounter, TransformedEnv

from active_adaptation.envs import _EnvBase
from active_adaptation.utils.string import resolve_matching_names


def make_env_policy(task_cfg, algo_cfg, seed, headless, device):
    importlib.import_module("active_adaptation.envs.backends.mjlab")
    policy_cfg = hydra.utils.instantiate(algo_cfg)
    policy_cls = policy_cfg.get_class()
    in_keys = policy_cfg.in_keys
    if not isinstance(in_keys, str):
        in_keys = list(in_keys)
    _, in_keys = resolve_matching_names(
        in_keys, task_cfg.observation.keys(), preserve_order=True
    )
    for key in list(task_cfg.observation):
        if not str(key).endswith("_") and str(key) not in in_keys:
            task_cfg.observation.pop(key)
    policy_cfg.in_keys = in_keys
    algo_cfg.in_keys = in_keys
    env_cls = _EnvBase.registry[task_cfg.get("env_class", "MjlabBackendEnv")]
    base_env = env_cls(task_cfg, str(device), headless=headless)
    transform = Compose(InitTracker(), StepCounter())
    env = TransformedEnv(base_env, transform)
    env.set_seed(seed)
    policy = policy_cls(
        policy_cfg,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device,
        env,
    )
    if hasattr(policy, "make_tensordict_primer"):
        transform.append(policy.make_tensordict_primer())
        env = TransformedEnv(env.base_env, transform)
    return env, policy
