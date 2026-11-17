"""Independent quaternion/matrix arithmetic; does not import author scorer."""

import re
import numpy as np

TRACK = (
    "pelvis",
    "torso_link",
    ".*_hip_yaw_link",
    ".*_knee_link",
    ".*_toe_link",
    ".*_shoulder_yaw_link",
    ".*_elbow_link",
    ".*_wrist_yaw_link",
)
RETURN = (
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
)


def conjugate(q):
    out = q.copy()
    out[..., 1:] *= -1
    return out


def multiply(a, b):
    w = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, v), -1)


def rotation(q):
    w = q[..., 0]
    v = q[..., 1:]
    x, y, z = np.moveaxis(v, -1, 0)
    cross = np.stack(
        (np.zeros_like(x), -z, y, z, np.zeros_like(x), -x, -y, x, np.zeros_like(x)), -1
    ).reshape(q.shape[:-1] + (3, 3))
    return (
        (2 * w * w - 1)[..., None, None] * np.eye(3, dtype=q.dtype)
        + 2 * v[..., :, None] * v[..., None, :]
        + 2 * w[..., None, None] * cross
    )


def yaw(q):
    mat = rotation(q)
    x = mat[..., :, 0]
    z = mat[..., :, 2]
    fallback = np.where((x[..., 2] < 0)[..., None], z[..., :2], -z[..., :2])
    xy = np.where((np.linalg.norm(x[..., :2], axis=-1) > 0.1)[..., None], x[..., :2], fallback)
    angle = np.arctan2(xy[..., 1], xy[..., 0])
    out = np.zeros(q.shape, dtype=q.dtype)
    out[..., 0] = np.cos(angle / 2)
    out[..., 3] = np.sin(angle / 2)
    return out


def angle(q):
    return 2 * np.arctan2(np.linalg.norm(q[..., 1:], axis=-1), np.maximum(np.abs(q[..., 0]), 1e-9))


def rotate(q, v):
    return np.einsum("...ij,...j->...i", rotation(q), v)


def local(p, q, anchor):
    origin = p[:, anchor].copy()
    origin[:, 2] = 0
    h = conjugate(yaw(q[:, anchor]))
    return rotate(h[:, None, :], p - origin[:, None, :]), multiply(h[:, None, :], q)


def root_error(data):
    p = np.asarray(data["robot_root_pos_w"], np.float64)
    q = np.asarray(data["robot_root_quat_w"], np.float64)
    rp = np.asarray(data["motion_root_pos_w"], np.float64)
    rq = np.asarray(data["motion_root_quat_w"], np.float64)
    qs = [q[0] / np.linalg.norm(q[0]), rq[0] / np.linalg.norm(rq[0])]
    yaws = [
        np.arctan2(2 * (v[0] * v[3] + v[1] * v[2]), 1 - 2 * (v[2] ** 2 + v[3] ** 2)) for v in qs
    ]
    theta = yaws[1] - yaws[0]
    c, s = np.cos(theta), np.sin(theta)
    t = np.asarray(data["motion_t"])
    idx = int(np.flatnonzero(t < int(data["motion_length"]) - 1)[-1])
    displacement = p[idx, :2] - p[0, :2]
    xy = (
        np.array(
            [c * displacement[0] - s * displacement[1], s * displacement[0] + c * displacement[1]]
        )
        + rp[0, :2]
        - rp[idx, :2]
    )
    err = np.array([*xy, p[idx, 2] - rp[idx, 2]])
    return dict(
        root_final_error_norm=float(np.linalg.norm(err)),
        root_final_error_xy_norm=float(np.linalg.norm(xy)),
        root_final_error_z_abs=float(abs(err[2])),
    )


def compute(data):
    t0 = np.asarray(data["motion_t"], np.int32)
    selected = np.r_[0, 1 + np.flatnonzero(np.diff(t0) != 0)]
    t = t0[selected]
    length = int(data["motion_length"])
    denom = max(1, length - 1)
    names = list(map(str, data["body_names"]))
    anchor = names.index("pelvis")
    torso = names.index("torso_link")
    ids = [i for pattern in TRACK for i, n in enumerate(names) if re.fullmatch(pattern, n)]
    assert len(set(ids)) == len(ids) == 14
    ri = [names.index(n) for n in RETURN]
    wi = [names.index(n) for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    p, q, rp, rq = [
        np.asarray(data[key], np.float32)[selected]
        for key in (
            "robot_body_pos_w",
            "robot_body_quat_w",
            "motion_body_pos_w",
            "motion_body_quat_w",
        )
    ]
    posfail = np.abs(p[:, anchor, 2] - rp[:, anchor, 2]) > 0.3
    orifail = np.abs(rotation(q[:, anchor])[:, 2, 2] - rotation(rq[:, anchor])[:, 2, 2]) > 0.8
    failed = np.flatnonzero(posfail | orifail)
    end = int(failed[0]) if len(failed) else len(t)
    reason = ("anchor_pos_z" if posfail[end] else "anchor_ori") if len(failed) else "motion_end"
    terminated = bool(len(failed))
    pre = max(1, end)
    if not terminated:
        assert t[-1] == denom
    pl, ql = local(p, q, anchor)
    rpl, rql = local(rp, rq, anchor)
    bodypos = np.linalg.norm(pl - rpl, axis=-1)
    bodyori = angle(multiply(conjugate(rql), ql))
    translation = p[:, torso].copy()
    translation[:, 2] = rp[:, torso, 2]
    delta = yaw(multiply(q[:, torso], conjugate(rq[:, torso])))
    targetp = translation[:, None, :] + rotate(delta[:, None, :], rp - rp[:, torso, None, :])
    targetq = multiply(delta[:, None, :], rq)
    psq = np.mean(np.sum((targetp[:, ri] - p[:, ri]) ** 2, axis=-1), axis=-1)
    osq = np.mean(angle(multiply(conjugate(targetq[:, ri]), q[:, ri])) ** 2, axis=-1)
    reward = np.exp(-psq / 0.3**2) + np.exp(-osq / 0.4**2)
    result = dict(
        progress=float(t[end] if terminated else t[-1]) / denom,
        termination_motion_t=int(t[end] if terminated else t[-1]),
        termination_reason=reason,
        terminated=int(terminated),
        local_body_tracking_error=float(bodypos[:pre, ids].mean()),
        local_body_orientation_error=float(bodyori[:pre, ids].mean()),
        wrist_tracking_error=float(bodypos[:pre, wi].mean()),
        wrist_orientation_error=float(bodyori[:pre, wi].mean()),
        normalized_tracking_return=float(np.sum(reward[:end], dtype=np.float64) / denom),
        mean_tracking_reward=float(np.sum(reward, dtype=np.float64) / denom),
    )
    result.update(root_error(data))
    return result
