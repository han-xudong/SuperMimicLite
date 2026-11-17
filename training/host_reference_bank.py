"""Bounded reference transfer, retaining native sampling/get_slice arithmetic.

Only cold-path storage ownership changes. All floating fields stay in the
native FP16 CPU bank; each native get_slice transfers only its requested rows.
The native source/frame sampler still executes on the requested device with
its original integer metadata and RNG.
"""

import copy
import torch


class HostFields:
    def __init__(self, source, device):
        assert source.device.type == "cpu"
        self.source = source
        self.device = torch.device(device)
        self.motion_id = source.motion_id.to(self.device)
        self.step = source.step.to(self.device)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        # Native FullMotionDataset computes/clamps the index before this call.
        return self.source[index.to("cpu")].to(self.device)


def place_host_bank(collection, device):
    """Copy mutable metadata only; do not mutate the source collection."""
    from any4hdmi.dataset.full import FullMotionDataset

    target = torch.device(device)
    result = copy.copy(collection)
    result.device = target
    result.datasets = []
    for original in collection.datasets:
        assert isinstance(original, FullMotionDataset)
        child = copy.copy(original)
        child.device = target
        child.data = HostFields(original.data, target)
        for name in ("starts", "ends", "lengths", "_env_motion_id", "_env_motion_len"):
            setattr(child, name, getattr(original, name).to(target).clone())
        result.datasets.append(child)
    for name, value in vars(collection).items():
        if isinstance(value, torch.Tensor):
            setattr(result, name, value.to(target).clone())
    result._resident_data = None
    result._resident_route_starts = None
    result._resident_route_ends = None
    return result


class PreparedHostBank:
    """Native RobotTracking's loader return protocol, including .to(device)."""

    def __init__(self, collection):
        self.collection = collection

    def to(self, device):
        return place_host_bank(self.collection, device)


def native_loader_with_host_bank(*args, **kwargs):
    import reference

    return PreparedHostBank(reference.load_motion_dataset_collection(*args, **kwargs))
