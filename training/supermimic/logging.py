"""Console and TensorBoard metrics for SuperMimic training."""

import time


class SuperMimicLogger:
    def __init__(
        self,
        *,
        env_name,
        backend,
        max_iterations,
        wm_only_iters,
        start_iteration,
        num_envs,
        log_dir,
        log_backend,
    ):
        self.max_iterations = max_iterations
        self.writer = None
        if log_backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        elif log_backend != "none":
            raise ValueError(f"Unsupported log backend: {log_backend}")
        self.label = f"{env_name}: {num_envs} environments, {max_iterations} iterations"
        self.started = None

    def start(self, status=""):
        print(self.label, flush=True)
        self.log_status(status)

    def start_training_timer(self):
        self.started = time.monotonic()

    def log_status(self, message):
        if message:
            print(message, flush=True)

    def log_buffer_fill(self, size, capacity):
        self.log_status(f"Replay: {size}/{capacity} steps")

    def log_save(self, path):
        self.log_status(f"Saved {path}")

    def log_step(self, iteration, **metrics):
        if self.writer is not None:
            for name, value in metrics.items():
                if value is not None:
                    self.writer.add_scalar(name, value, iteration)
        if iteration % 50 == 0 or iteration == self.max_iterations:
            values = " ".join(
                f"{k}={metrics[k]:.5g}"
                for k in ("wm_loss", "sim_rew", "imag_rew")
                if metrics.get(k) is not None
            )
            self.log_status(f"{iteration}/{self.max_iterations} {values}")

    def finish(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
