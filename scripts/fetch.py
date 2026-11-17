"""Download checksum-pinned models, motion data, and assets from Hugging Face."""

import argparse
import shutil
from pathlib import Path
from _common import ROOT, SPLITS, artifacts, load, sha


def models(checkpoints=False):
    from huggingface_hub import hf_hub_download

    spec = load(ROOT / "manifests/artifacts.json")
    for row in spec["files"]:
        if row["directory"] == "checkpoints" and not checkpoints:
            continue
        out = ROOT / row["directory"]
        out.mkdir(exist_ok=True)
        p = out / row["name"]
        if p.exists():
            if sha(p) != row["sha256"]:
                raise ValueError(f"Existing artifact mismatch: {p}")
            continue
        cached = hf_hub_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            revision=spec["revision"],
            filename=row["directory"] + "/" + row["name"],
        )
        shutil.copy2(cached, p)
    artifacts()
    if checkpoints:
        artifacts("checkpoints")


def datasets():
    from data_store import download, fetch_files, install_verified, specification

    spec = specification()
    root = ROOT / "data/motiondecode"
    record = spec["evaluation_sources"]
    install_verified(download(record), root / "sources.json", record["sha256"])
    for split in SPLITS:
        out = root / split
        expected = load(ROOT / "manifests/evaluation" / split / "checksums.json")
        source = {key: spec[key] for key in ("repo_id", "repo_type", "revision")}
        source["subfolder"] = spec["hosted_evaluation"][split]
        fetch_files(source, out, expected)
        if (out / "motion-list.txt").read_bytes() != (
            ROOT / "manifests/evaluation" / split / "motion-list.txt"
        ).read_bytes():
            raise ValueError("Motion ordering changed")


def asset(training=False):
    from huggingface_hub import snapshot_download

    key = "training_asset" if training else "evaluation_asset"
    pin = load(ROOT / "manifests/upstream.json")[key]
    cache = ROOT / ".cache" / ("training" if training else "evaluation") / "hf/hub"
    root = Path(
        snapshot_download(pin["repo_id"], revision=pin["revision"], cache_dir=cache)
    )
    # The original author's hf://...@main locator is resolved in our isolated cache.
    refs = root.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(pin["revision"])
    print(f"{key}: {pin['revision']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["models", "data", "asset", "all"])
    p.add_argument("--checkpoints", action="store_true")
    p.add_argument("--training-asset", action="store_true")
    a = p.parse_args()
    if a.what in ["models", "all"]:
        models(a.checkpoints)
    if a.what in ["data", "all"]:
        datasets()
    if a.what in ["asset", "all"]:
        asset(a.training_asset)


if __name__ == "__main__":
    main()
