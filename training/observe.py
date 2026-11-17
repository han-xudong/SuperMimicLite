"""Finite-gradient and optimizer-step monitoring for SuperMimic."""

import json
import time
import torch
from supermimic.learner import SuperMimicLearner


class ObservedLearner(SuperMimicLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = {"wm": 0, "pi": 0, "critic": 0}
        self.last_check = 0.0
        self.last_finite = {}
        self.observer_handles = []
        for name, optimizer in (
            ("wm", self.opt_wm),
            ("pi", self.opt_pi),
            ("critic", self.opt_v),
        ):

            def before(opt, args, kwargs, name=name):
                grads = [
                    p.grad
                    for g in opt.param_groups
                    for p in g["params"]
                    if p.grad is not None
                ]
                if not grads or not bool(
                    torch.stack([g.norm() for g in grads]).isfinite().all()
                ):
                    raise FloatingPointError(
                        f"Nonfinite or absent {name} gradients before optimizer step"
                    )

            def after(opt, args, kwargs, name=name):
                self.counts[name] += 1

            self.observer_handles.extend(
                (
                    optimizer.register_step_pre_hook(before),
                    optimizer.register_step_post_hook(after),
                )
            )

    def publish(self, name, values):
        if not all(bool(v.isfinite().all()) for v in values):
            raise FloatingPointError(f"Nonfinite {name} update return")
        self.last_finite[name] = [float(v) for v in values]
        if time.time() - self.last_check < 5:
            return
        params = [
            p for module in (self.wm, self.pi, self.critic) for p in module.parameters()
        ]
        if not bool(torch.stack([p.detach().norm() for p in params]).isfinite().all()):
            raise FloatingPointError("Nonfinite learner parameter")
        free, total = (
            torch.cuda.mem_get_info(self.device)
            if self.device.type == "cuda"
            else (None, None)
        )
        result = {
            "time": time.time(),
            "iteration": self.iteration,
            "optimizer_steps": self.counts,
            "finite": True,
            "last_update": self.last_finite,
            "free_bytes": free,
            "total_bytes": total,
        }
        path = self.log_dir / "observer.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
        tmp.replace(path)
        if free is not None and free < 1945 * 1024**2:
            raise MemoryError("GPU headroom below 1945MiB; original recipe not reduced")
        self.last_check = time.time()

    def wm_update(self):
        result = super().wm_update()
        self.publish("wm", result)
        return result

    def policy_update(self):
        result = super().policy_update()
        self.publish("policy", result)
        return result
