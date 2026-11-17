"""Native motion loading and G1 reference features."""

from pathlib import Path
from paths import N, leaf
import torch
import active_adaptation as aa

if aa.get_backend() is None:
    aa.set_backend("mjlab")

from any4hdmi import FullMotionDataset


# Import the pinned data owners without registering unrelated H2/AsRo assets.
_tasks = N / "projects/mimic-lite/mimic_lite/tasks"
_multi = leaf("supermimiclite_multi_dataset", _tasks / "multi_dataset.py")
_motion = leaf("supermimiclite_motion_loader", _tasks / "motion.py")
normalize_motion_cfgs = _multi.normalize_motion_cfgs
load_motion_dataset_collection = _multi.load_motion_dataset_collection
create_dataset_from_path = _motion.create_dataset_from_path
from supermimic.geometry import mat_to_6d, quat_to_mat, yaw_mat
from robot import G1_JOINT_NAMES


def load_collection(data, num_envs, device="cpu"):
    from recipe_config import data_config

    config = data_config(data)
    configs = normalize_motion_cfgs(config["motion_cfgs"])
    assert all(item.full_motion for item in configs), (
        "G1 recipes require full-motion datasets"
    )
    collection = load_motion_dataset_collection(
        configs,
        create_dataset_fn=create_dataset_from_path,
        target_fps=50,
        num_envs=num_envs,
        body_names=["pelvis"],
        joint_names=G1_JOINT_NAMES,
        windowed_next_window_device=config.get(
            "windowed_next_window_device", "current"
        ),
        windowed_pin_window_load=config.get("windowed_pin_window_load", True),
    )
    return place_collection(collection, device)


def place_collection(collection, device):
    """Use native moves in metadata-coherent order, retaining packed aliases.

    Native MultiMotionDataset.to packs fields before moving each child's
    TensorClass device metadata. A CPU-constrained child consequently copies
    assigned CUDA slices back to CPU, then .to(CUDA) allocates a second copy.
    Moving children first means the unchanged native pack assigns CUDA views
    to CUDA-owned TensorClasses. No field, precision, sampling or RNG changes.
    """
    target = torch.device(device)
    if collection.device != target:
        for dataset in collection.datasets:
            dataset.to(target)
    return collection.to(target)


class _ClipIndex:
    def __init__(self, motion):
        self.motion = motion

    def __getitem__(self, frames):
        return torch.bucketize(frames.contiguous(), self.motion.ends, right=True)


class _BinIndex:
    def __init__(self, motion):
        self.motion = motion

    def __getitem__(self, frames):
        clip = self.motion.clip_id[frames]
        local = torch.minimum(
            frames - self.motion.clip_start[clip], self.motion.clip_len[clip] - 1
        )
        return self.motion.clip_bin_start[clip] + local // 50


class _Field:
    def __init__(self, motion, key):
        self.motion, self.key = motion, key

    def __getitem__(self, frames):
        return self.motion.field(self.key, frames)


class MixtureMotion:
    def __init__(self, collection, kin):
        assert all(isinstance(item, FullMotionDataset) for item in collection.datasets)
        assert (
            collection.body_names == ["pelvis"]
            and collection.joint_names == G1_JOINT_NAMES
        )
        self.collection, self.kin, self.device = collection, kin, collection.device
        # A single cold-path binding to the existing MimicLite resident cache.
        # Hot paths neither inspect assets nor discover private backend fields.
        self.storage = collection._resident_data
        assert self.storage is not None
        self._root_p = self.storage.body_pos_w[:, 0]
        self._root_quat = self.storage.body_quat_w[:, 0]
        self._root_v = self.storage.body_lin_vel_w[:, 0]
        self._root_w = self.storage.body_ang_vel_w[:, 0]
        self._q, self._qd = self.storage.joint_pos, self.storage.joint_vel
        self.storage_clip_start = collection.starts.clone()
        self.clip_len = collection.ends - collection.starts
        # Any4 samples every source frame, including the final one. Native
        # SuperMimic reward/termination then read f+1. Reserve one logical-only
        # sentinel per clip and map it back to that clip's last stored frame,
        # exactly as Any4.get_slice's end clamp. Sampling probabilities, starts,
        # real clip lengths and evaluator denominators remain unchanged.
        self.clip_start = collection.starts + torch.arange(
            collection.num_motions, device=self.device
        )
        self.clip_end = self.clip_start + self.clip_len - 1
        self.ends = self.clip_end + 2
        assert bool((self.clip_len >= 2).all())
        self.num_frames = collection.num_steps + collection.num_motions
        self.clip_id, self.frame_bin = _ClipIndex(self), _BinIndex(self)
        self.names, source_ids = [], []
        for source_id, (source, dataset) in enumerate(
            zip(collection.motion_cfgs, collection.datasets, strict=True)
        ):
            base = Path(source.path)
            self.names.extend(
                source.name + "::" + path.relative_to(base / "motions").as_posix()
                for path in dataset.motion_paths
            )
            source_ids.extend([source_id] * dataset.num_motions)
        assert len(set(self.names)) == len(self.names) == collection.num_motions
        self.clip_source = torch.tensor(
            source_ids, device=self.device, dtype=torch.long
        )
        bins_per_clip = (self.clip_len + 49) // 50
        self.clip_bin_start = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.num_bins = int(bins_per_clip.sum())
        for key in (
            "root_p",
            "root_R",
            "root_v",
            "root_w",
            "q",
            "qd",
            "root_v_local",
            "root_w_local",
            "root_gz",
            "body_p",
            "body_R",
            "body_v",
            "body_w",
        ):
            setattr(self, key, _Field(self, key))

    def to(self, device):
        if torch.device(device) != self.device:
            raise ValueError(
                "Bind the actual cache to its final device at the cold-path owner"
            )
        return self

    def ref_state(self, frames):
        frames = self._cache_frames(frames)
        p = self._root_p[frames].float()
        quaternion = self._root_quat[frames].float()
        # FP16 storage needs canonical SO(3) reconstruction, not a nonunit matrix.
        quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rotation = quat_to_mat(quaternion)
        return (
            p,
            rotation,
            self._root_v[frames].float(),
            self._root_w[frames].float(),
            self._q[frames].float(),
            self._qd[frames].float(),
        )

    def field(self, key, frames):
        if key == "root_p":
            return self._root_p[self._cache_frames(frames)].float()
        if key == "q":
            return self._q[self._cache_frames(frames)].float()
        if key == "qd":
            return self._qd[self._cache_frames(frames)].float()
        state = self.ref_state(frames)
        if key == "root_R":
            return state[1]
        if key == "root_v":
            return state[2]
        if key == "root_w":
            return state[3]
        if key == "root_v_local":
            return (state[1].transpose(-1, -2) @ state[2].unsqueeze(-1)).squeeze(-1)
        if key == "root_w_local":
            return (state[1].transpose(-1, -2) @ state[3].unsqueeze(-1)).squeeze(-1)
        if key == "root_gz":
            return state[1][..., 2, 2]
        bodies = self.kin.forward(*state)
        return bodies[("body_p", "body_R", "body_v", "body_w").index(key)]

    def _cache_frames(self, frames):
        clip = self.clip_id[frames]
        local = torch.minimum(frames - self.clip_start[clip], self.clip_len[clip] - 1)
        return self.storage_clip_start[clip] + local

    def future_frames(self, frames, num_frames):
        end = self.clip_end[self.clip_id[frames]]
        future = frames.unsqueeze(-1) + torch.arange(
            1, num_frames + 1, device=frames.device
        )
        return torch.minimum(future, end.unsqueeze(-1))

    def goal(self, frames, num_frames):
        # Exact original MotionLib goal formula, with stable cache-backed fields.
        future = self.future_frames(frames, num_frames)
        heading_t = yaw_mat(self.root_R[frames]).transpose(-1, -2).unsqueeze(1)
        delta = (
            heading_t
            @ (self.root_p[future] - self.root_p[frames].unsqueeze(1)).unsqueeze(-1)
        ).squeeze(-1)
        rotation = mat_to_6d(heading_t @ self.root_R[future])
        features = torch.cat(
            (
                delta,
                rotation,
                self.root_v_local[future],
                self.root_w_local[future],
                self.q[future],
                self.qd[future] * 0.1,
            ),
            -1,
        )
        return features.reshape(frames.shape[0], -1)
