"""Finite-gradient and optimizer-step monitoring for PPO."""

import json, time, torch
from ppo_factory import make_env_policy as build_env_policy

POLICY = None
COUNTS = {"actor": 0, "critic": 0}
OUT = None


def finite(v):
    if torch.is_tensor(v):
        return bool(v.isfinite().all()) if v.is_floating_point() else True
    if isinstance(v, dict):
        return all(finite(x) for x in v.values())
    if isinstance(v, (tuple, list)):
        return all(finite(x) for x in v)
    return True


def make_env_policy(*args, **kwargs):
    global POLICY
    env, policy = build_env_policy(*args, **kwargs)
    POLICY = policy
    state = policy.state_dict()
    torch.save(
        {k: state[k] for k in ["actor", "critic", "vecnorms"]}, OUT / "fresh-initial.pt"
    )
    for name, opt in [("actor", policy.opt_policy), ("critic", policy.opt_critic)]:

        def before(opt, a, k):
            grads = [
                p.grad
                for g in opt.param_groups
                for p in g["params"]
                if p.grad is not None
            ]
            if not grads or not all(bool(g.isfinite().all()) for g in grads):
                raise FloatingPointError("Nonfinite learner gradient")

        def after(opt, a, k, name=name):
            COUNTS[name] += 1

        opt.register_step_pre_hook(before)
        opt.register_step_post_hook(after)
    original = policy.train_op

    def observe(data):
        if not finite(data.to_dict()):
            raise FloatingPointError("Nonfinite rollout entered learner")
        result = original(data)
        if not finite(result) or not finite(policy.state_dict()):
            raise FloatingPointError("Nonfinite learner state")
        payload = {
            "iteration": int(policy.state_dict()["last_iter"]) + 1,
            "optimizer_steps": COUNTS,
            "finite": True,
            "time": time.time(),
        }
        (OUT / "progress.json").write_text(json.dumps(payload) + "\n")
        return result

    policy.train_op = observe
    return env, policy
