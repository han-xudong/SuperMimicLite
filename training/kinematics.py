"""Differentiable full-body mode15 FK, including fixed links and joint anchors."""

import copy
import torch
from supermimic.geometry import axis_angle_to_mat, quat_to_mat
from robot import G1_JOINT_NAMES, TRACKED_BODIES
from robot import clean


class G1Kinematics:
    def __init__(
        self, model, device="cpu", tracked=TRACKED_BODIES, dtype=torch.float32
    ):
        import mujoco

        self.device = torch.device(device)
        self.body_names = [clean(model.body(i).name) for i in range(1, model.nbody)]
        self.num_bodies = len(self.body_names)
        self.tracked_idx = [self.body_names.index(name) for name in tracked]
        self.parent = [int(model.body_parentid[i]) - 1 for i in range(1, model.nbody)]
        self.joint_idx, axes, anchors = [], [], []
        roots = []
        for body in range(1, model.nbody):
            count = int(model.body_jntnum[body])
            if count > 1:
                raise ValueError("mode15 body has multiple joints")
            joint = int(model.body_jntadr[body]) if count else -1
            if joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                roots.append(body - 1)
                self.joint_idx.append(-2)
            elif joint >= 0:
                if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("Only native mode15 hinge joints are supported")
                self.joint_idx.append(
                    G1_JOINT_NAMES.index(clean(model.joint(joint).name))
                )
            else:
                self.joint_idx.append(-1)
            axes.append(model.jnt_axis[joint].tolist() if joint >= 0 else [0, 0, 1])
            anchors.append(model.jnt_pos[joint].tolist() if joint >= 0 else [0, 0, 0])
        if len(roots) != 1 or self.body_names[roots[0]] != "pelvis":
            raise ValueError("Expected the native free pelvis root")
        self.root_index = roots[0]
        self.axes = torch.tensor(axes, device=device, dtype=dtype)
        self.anchors = torch.tensor(anchors, device=device, dtype=dtype)
        self.off_p = torch.tensor(model.body_pos[1:], device=device, dtype=dtype)
        self.off_R = quat_to_mat(
            torch.tensor(model.body_quat[1:], device=device, dtype=dtype)
        )
        jmap = {clean(model.joint(i).name): i for i in range(model.njnt)}
        self.q_lo = torch.tensor(
            [model.jnt_range[jmap[n], 0] for n in G1_JOINT_NAMES],
            device=device,
            dtype=dtype,
        )
        self.q_hi = torch.tensor(
            [model.jnt_range[jmap[n], 1] for n in G1_JOINT_NAMES],
            device=device,
            dtype=dtype,
        )

    def to(self, device):
        new = copy.copy(self)
        new.device = torch.device(device)
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(new, name, value.to(device))
        return new

    def forward(self, p, R, v, w, q, qd, all_bodies=False):
        ps, Rs, vs, ws = [], [], [], []
        for i in range(self.num_bodies):
            if i == self.root_index:
                pi, Ri, vi, wi = p, R, v, w
            else:
                j = self.joint_idx[i]
                parent = self.parent[i]
                if parent < 0:
                    pp, Rp, vp, wp = (
                        torch.zeros_like(p),
                        torch.eye(3, device=p.device, dtype=p.dtype).expand_as(R),
                        torch.zeros_like(v),
                        torch.zeros_like(w),
                    )
                else:
                    pp, Rp, vp, wp = ps[parent], Rs[parent], vs[parent], ws[parent]
                base_p = pp + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
                base_R = Rp @ self.off_R[i]
                if j >= 0:
                    joint_R = axis_angle_to_mat(self.axes[i], q[..., j])
                    Ri = base_R @ joint_R
                    anchor = base_p + (base_R @ self.anchors[i].unsqueeze(-1)).squeeze(
                        -1
                    )
                    pi = anchor - (Ri @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                    axis_w = (base_R @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                    joint_w = axis_w * qd[..., j : j + 1]
                    wi = wp + joint_w
                    vi = (
                        vp
                        + torch.cross(wp, pi - pp, dim=-1)
                        + torch.cross(joint_w, pi - anchor, dim=-1)
                    )
                else:
                    pi, Ri, wi = base_p, base_R, wp
                    vi = vp + torch.cross(wp, pi - pp, dim=-1)
            ps.append(pi)
            Rs.append(Ri)
            vs.append(vi)
            ws.append(wi)
        ids = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([ps[i] for i in ids], -2),
            torch.stack([Rs[i] for i in ids], -3),
            torch.stack([vs[i] for i in ids], -2),
            torch.stack([ws[i] for i in ids], -2),
        )
