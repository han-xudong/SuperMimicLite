# Adapted from MimicLite scripts/train.py, revision 94dd77a827c2b28d6b69efc3291e392e0f07c906.
# Modified for SuperMimicLite; upstream licensing is recorded in THIRD_PARTY.md.

"""PPO rollout and update loop for the MimicLite recipes."""

import torch
import wandb
import json
import logging
import time
import datetime
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict import TensorDict
import active_adaptation as aa
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.learning.modules.vecnorm import VecNorm
from registration import register_native_G1

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def run(cfg: DictConfig, out):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    register_native_G1()
    assert aa.get_backend() == "mjlab" and (not aa.is_distributed())
    from ppo_observe import make_env_policy
    from active_adaptation.utils.helpers import EpisodeStats

    num_envs = cfg.task.num_envs
    frames_per_batch = num_envs * cfg.algo.train_every
    total_iters = int(cfg.total_iters)
    cfg.task.total_iters = total_iters
    env, policy = make_env_policy(
        cfg.task, cfg.algo, seed=cfg.seed, headless=cfg.headless, device=cfg.device
    )
    policy: PPOBase
    requires_rollout_value = bool(getattr(policy, "requires_rollout_value", True))
    env.total_iters = total_iters
    policy.total_iters = total_iters
    checkpoint_interval = cfg.checkpoint_interval
    upload_interval = cfg.upload_interval
    log_interval = cfg.task.max_episode_length // cfg.algo.train_every + 1
    logging.info(f"Log interval: {log_interval} steps")
    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str, *, upload_to_wandb: bool = True):
        assert run is not None
        run_dir = Path(run.dir)
        stage_dir = Path.cwd()
        ckpt_path = run_dir / f"{checkpoint_name}.pt"
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        state_dict["env"] = env.state_dict()
        state_dict["cfg"] = cfg
        torch.save(state_dict, ckpt_path)
        if upload_to_wandb:
            run.save(str(ckpt_path), policy="now", base_path=run.dir)
        latest_link = run_dir / "checkpoint_latest.pt"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_path.name)
        stage_latest_link = stage_dir / "checkpoint_latest.pt"
        if stage_latest_link.exists() or stage_latest_link.is_symlink():
            stage_latest_link.unlink()
        stage_latest_link.symlink_to(ckpt_path.resolve())
        logging.info(
            f"Saved checkpoint to {ckpt_path}" + (" (wandb)" if upload_to_wandb else "")
        )
        return str(ckpt_path)

    assert env.training

    def should_save(i):
        if not aa.is_main_process():
            return False
        return i % checkpoint_interval == 0 or i % upload_interval == 0

    carry = env.reset()
    next_saved_keys = [
        "done",
        "terminated",
        "truncated",
        "discount",
        "reward",
        "stats",
        "is_init",
        "adapt_hx",
        "episode_id",
    ]
    next_saved_keys.extend(policy.get_next_saved_keys())
    next_saved_keys = list(dict.fromkeys(next_saved_keys))
    start_iter = 0
    env_frames = start_iter * frames_per_batch
    if hasattr(policy.cfg, "stages"):
        stages = policy.cfg.stages
    else:
        stages = ("",)
    metrics_path = None
    if aa.is_main_process():
        run = wandb.init(
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
            id=cfg.wandb.get("id", None),
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()
        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        run_idx = run.id if run.name is None else run.name.split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)
        run_dir = Path(run.dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_save_path = run_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)
        run.save(str(cfg_save_path), policy="now")
        run.save(str(run_dir / "config.yaml"), policy="now")
        if cfg.wandb.mode == "disabled":
            hydra_output_dir = out
            hydra_output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = hydra_output_dir / "metrics.jsonl"
            OmegaConf.save(cfg, hydra_output_dir / "cfg.yaml")
            logging.info(f"Disabled-W&B metrics will be written to {metrics_path}")
    for stage in stages:
        policy.on_stage_start(stage, env)
        rollout_policy = policy.get_rollout_policy("train")
        with (
            torch.inference_mode(),
            set_exploration_type(ExplorationType.RANDOM),
            VecNorm.freeze(),
        ):
            tmp_carry = rollout_policy(carry.clone(True))
            if tmp_carry.device.type == "cuda":
                torch.cuda.empty_cache()
            tmp_td, _ = env.step_and_maybe_reset(tmp_carry.clone(False))
            tmp_td["next"] = tmp_td["next"].select(*next_saved_keys, strict=False)
        data_buf: TensorDict = (
            tmp_td.unsqueeze(-1).expand(env.num_envs, cfg.algo.train_every).clone()
        )
        del tmp_td, tmp_carry
        if env.device.type == "cuda":
            torch.cuda.empty_cache()
        progress = range(start_iter, total_iters)
        if aa.is_main_process():
            progress = tqdm(progress, desc=stage)
        for i in progress:
            if should_save(i):
                should_upload = i % upload_interval == 0
                checkpoint_name = (
                    f"checkpoint_{i}" if should_upload else "checkpoint_temp"
                )
                ckpt_path = save(policy, checkpoint_name, upload_to_wandb=should_upload)
                if ckpt_path is not None:
                    print(f"Latest checkpoint: {ckpt_path}")
            rollout_start = time.perf_counter()
            with ScopedTimer("rollout") as rollout_timer:
                with (
                    torch.inference_mode(),
                    set_exploration_type(ExplorationType.RANDOM),
                ):
                    if hasattr(env, "set_progress"):
                        env.set_progress(i)
                    for step in range(cfg.algo.train_every):
                        with ScopedTimer("policy_inference"):
                            carry = rollout_policy(carry)
                        td, carry = env.step_and_maybe_reset(carry)
                        td["next"] = td["next"].select(*next_saved_keys, strict=False)
                        data_buf[:, step] = td
                    if requires_rollout_value:
                        if hasattr(policy, "compute_rollout_values"):
                            policy.compute_rollout_values(data_buf, carry.copy())
                        else:
                            policy.critic(data_buf)
                            values = data_buf["state_value"]
                            last_value = policy.compute_value(carry.copy())[
                                "state_value"
                            ]
                            next_values = torch.cat(
                                [values[:, 1:], last_value.unsqueeze(1)], dim=1
                            )
                            data_buf["next", "state_value"] = torch.where(
                                data_buf["next", "done"], values, next_values
                            )
            rollout_time = rollout_timer.last_time
            episode_stats.add(data_buf)
            env_frames += data_buf.numel()
            info = {}
            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()
            with ScopedTimer("training") as training_timer:
                info.update(policy.train_op(data_buf))
            training_time = training_timer.last_time
            info.update(env.extra)
            info.update(env.stats_ema)
            if hasattr(policy, "step_schedule"):
                policy.step_schedule(i / total_iters)
            info["env_frames"] = env_frames * aa.get_world_size()
            info["performance/rollout_fps"] = (
                data_buf.numel() / rollout_time * aa.get_world_size()
            )
            info["performance/rollout_time"] = rollout_time
            info["performance/training_time"] = training_time
            info["performance/iter_time"] = time.perf_counter() - rollout_start
            if aa.is_main_process() and run is not None:
                run.log(info, step=i)
                if metrics_path is not None:
                    metrics_record = {"iteration": i}
                    for key, value in info.items():
                        if isinstance(value, torch.Tensor):
                            if value.numel() != 1:
                                continue
                            value = value.detach().item()
                        elif hasattr(value, "item"):
                            try:
                                value = value.item()
                            except (TypeError, ValueError):
                                continue
                        if isinstance(value, (bool, int, float, str)) or value is None:
                            metrics_record[key] = value
                    with metrics_path.open("a") as metrics_file:
                        metrics_file.write(
                            json.dumps(metrics_record, sort_keys=True) + "\n"
                        )
    if aa.is_main_process():
        ckpt_path = save(policy, f"checkpoint_{total_iters}")
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")
    return
