"""NumPy deployment adapter; its native-oracle contract is tested separately.

Only policy input construction lives here. Author physics/scorer are unchanged.
Canonical state is root xyz, rotation, world linear/angular, canonical29 q/qd.
"""

import numpy as np


def quat_matrix(q):
    q = np.asarray(q, dtype=np.float32)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


def heading(R, projected=False):
    if projected:
        x, z = R[..., :, 0], R[..., :, 2]
        xy = np.where(
            (np.linalg.norm(x[..., :2], axis=-1) > 0.1)[..., None],
            x[..., :2],
            np.where((x[..., 2] < 0)[..., None], z[..., :2], -z[..., :2]),
        )
        a = np.arctan2(xy[..., 1], xy[..., 0])
        c, s = np.cos(a), np.sin(a)
    else:
        c, s = R[..., 0, 0], R[..., 1, 0]
        norm = np.sqrt(c * c + s * s + np.float32(1e-9))
        c, s = c / norm, s / norm
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(R.shape)


class Reference:
    def __init__(self, p, R, v, w, q, qd):
        self.p, self.R, self.v, self.w, self.q, self.qd = [
            np.asarray(x, np.float32) for x in (p, R, v, w, q, qd)
        ]
        self.length = len(self.p)

    def goal(self, frame):
        f = min(max(int(frame), 0), self.length - 1)
        future = np.minimum(f + np.arange(1, 11), self.length - 1)
        ht = heading(self.R[f]).T
        dp = np.einsum("ij,kj->ki", ht, self.p[future] - self.p[f])
        ori = (ht @ self.R[future])[..., :, :2].reshape(10, 6)
        rt = self.R[future].transpose(0, 2, 1)
        lv = np.einsum("kij,kj->ki", rt, self.v[future])
        av = np.einsum("kij,kj->ki", rt, self.w[future])
        return np.concatenate(
            (dp, ori, lv, av, self.q[future], self.qd[future] * np.float32(0.1)), -1
        ).reshape(730)


class PPOCore:
    def __init__(self, order):
        assert sorted(order) == list(range(29))
        self.order = list(order)
        self.reset()

    def reset(self):
        self.history = None
        self.previous_actions = np.zeros((3, 29), np.float32)
        self.tick = 0

    def compute(self, state, previous_action, ref):
        p, R, v, w, q, qd = state
        vals = [R.T @ w, -R[2, :], q[self.order], qd[self.order]]
        vals[1] = vals[1] / max(float(np.linalg.norm(vals[1])), 1e-6)
        if self.history is None:
            self.history = [np.repeat(x[None], 17, 0) for x in vals]
        else:
            self.history = [
                np.concatenate((x[None], h[:-1]), 0) for h, x in zip(self.history, vals)
            ]
            self.previous_actions = np.concatenate(
                (np.asarray(previous_action, np.float32)[None], self.previous_actions[:-1]), 0
            )
        if self.tick == 0:
            out = dict(policy=np.zeros(535, np.float32), command=np.zeros(304, np.float32))
        else:
            f = min(self.tick - 1, ref.length - 1)
            future = np.clip(f + np.array([-8, -4, -2, 0, 1, 2, 3, 4]), 0, ref.length - 1)
            anchor = ref.p[f].copy()
            anchor[2] = 0
            local = np.einsum(
                "ij,kj->ki", heading(ref.R[f], projected=True).T, ref.p[future] - anchor
            )
            relative = (R.T @ ref.R[future])[:, :2, :].reshape(-1)
            command = np.concatenate(
                (local.reshape(-1), relative, ref.q[future][:, self.order].reshape(-1))
            )
            policy = np.concatenate(
                [h[[0, 1, 2, 3, 4, 8, 16]].reshape(-1) for h in self.history]
                + [self.previous_actions.reshape(-1)]
            )
            out = dict(policy=policy.astype(np.float32), command=command.astype(np.float32))
        self.tick += 1
        return out


class SMCore:
    def __init__(self, default_q):
        self.default_q = np.asarray(default_q, np.float32)
        self.reset()

    def reset(self):
        self.history = None
        self.tick = 0

    def compute(self, state, previous_action, ref):
        p, R, v, w, q, qd = state
        f = min(self.tick, ref.length - 1)
        obs = np.concatenate(
            (
                q - self.default_q,
                qd,
                R.T @ w,
                -R[2, :],
                previous_action,
                (R.T @ ref.R[f])[:, :2].reshape(6),
            )
        ).astype(np.float32)
        assert obs.shape == (99,)
        self.history = (
            np.repeat(obs[None], 10, 0)
            if self.history is None
            else np.concatenate((self.history[1:], obs[None]), 0)
        )
        self.tick += 1
        return dict(proprioception_history=self.history.reshape(990).copy(), goal=ref.goal(f))
