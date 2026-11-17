"""Prepare the exact motion bank from hosted data, upstream sources and local CSVs."""

import argparse, gzip, json, shutil
import os, subprocess, tempfile
from pathlib import Path
from _common import ROOT, load, sha


def load_admission(root):
    from data_store import download, specification

    path = root / "admission.json.gz"
    record = specification()["admission"]
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download(record), path)
    if sha(path) != record["sha256"]:
        raise ValueError(f"Training admission metadata mismatch: {path}")
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def expected_files(admission, selected):
    cfg = load(ROOT / "configs/training/mixture.json")["motion_cfgs"]
    expected = {}
    for name in selected:
        source = admission["sources"][name]
        expected.update(
            {
                p: admission["file_records"][p]["sha256"]
                for p in source["selected_files"]
            }
        )
        expected[source["root"] + "/manifest.json"] = source["manifest_sha256"]
        if cfg[name].get("filenames_path"):
            expected[source["root"] + "/" + cfg[name]["filenames_path"]] = source[
                "filter_sha256"
            ]
    return expected


def selection_summary(admission, selected):
    groups = [admission["sources"][name]["selected_files"] for name in selected]
    paths = set().union(*map(set, groups))
    frames = {p: admission["file_records"][p]["control_frames"] for p in paths}
    return {
        "files": len(paths),
        "frames_50Hz": sum(frames.values()),
        "source_entries": sum(map(len, groups)),
        "source_frames_50Hz": sum(frames[p] for group in groups for p in group),
    }


def verify_source_selection(root, name, source, cfg):
    base = root / source["root"]
    if sha(base / "manifest.json") != source["manifest_sha256"]:
        raise ValueError(f"Training manifest mismatch: {name}")
    if cfg.get("filenames_path"):
        filter_path = base / cfg["filenames_path"]
        if sha(filter_path) != source["filter_sha256"]:
            raise ValueError(f"Training filter mismatch: {name}")
        names = [
            line.strip()
            for line in filter_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        actual = {
            (Path(source["root"]) / "motions" / value).as_posix() for value in names
        }
    else:
        actual = {
            p.relative_to(root).as_posix() for p in (base / "motions").rglob("*.npz")
        }
    if actual != set(source["selected_files"]):
        raise ValueError(f"Training selection differs from the admitted source: {name}")


def verify(root, data):
    admission = load_admission(root)
    selected = ["lafan"] if data == "L" else list(admission["sources"])
    paths = set()
    cfg = load(ROOT / "configs/training/mixture.json")["motion_cfgs"]
    for name in selected:
        source = admission["sources"][name]
        verify_source_selection(root, name, source, cfg[name])
        paths.update(source["selected_files"])
    for i, relative in enumerate(sorted(paths)):
        record = admission["file_records"][relative]
        p = root / relative
        if (
            not p.is_file()
            or p.stat().st_size != record["bytes"]
            or sha(p) != record["sha256"]
        ):
            raise ValueError(f"Training data mismatch: {p}")
        if i % 5000 == 0:
            print(f"verified {i}/{len(paths)}", flush=True)
    counts = selection_summary(admission, selected)
    assert (counts["files"], counts["frames_50Hz"]) == (
        (40, 441140) if data == "L" else (131373, 50712626)
    )
    assert counts["source_frames_50Hz"] == (
        441140 if data == "L" else admission["global_frame_count"]
    )
    print(json.dumps({"passed": True, "data": data, **counts}))


def prepare(root, data, sonic_csv=None, workers=8):
    from data_store import download, fetch_files, install_verified, specification

    admission = load_admission(root)
    selected = ["lafan"] if data == "L" else list(admission["sources"])
    expected = expected_files(admission, selected)
    spec = specification()
    sonic = {p: h for p, h in expected.items() if p.startswith("sonic/motions/")}
    if any(not (root / p).is_file() for p in sonic):
        if sonic_csv is None:
            raise FileNotFoundError(
                "Mixed requires licensed BONES-SEED G1 CSVs from "
                "https://huggingface.co/datasets/bones-studio/seed. "
                "Accept the upstream terms and download/extract g1.tar.gz, then "
                "run prepare --data M --sonic-csv /path/to/g1/csv."
            )
        if not sonic_csv.is_dir() or workers < 1:
            raise ValueError(
                "Provide an existing --sonic-csv directory and positive --workers"
            )
        converter = ROOT / "environments/train/.venv/bin/python"
        if not converter.is_file():
            raise FileNotFoundError(
                "Run uv sync --project environments/train --locked before CSV conversion"
            )
        asset = load(ROOT / "manifests/upstream.json")["training_asset"]
        with tempfile.TemporaryDirectory(
            prefix="sonic-conversion-", dir=root.parent
        ) as folder:
            subprocess.run(
                [
                    str(converter),
                    "-m",
                    "any4hdmi.scripts.preprocess.sonic",
                    "--csv-dir",
                    str(sonic_csv),
                    "--out-dir",
                    folder,
                    "--mjcf-repo",
                    asset["repo_id"],
                    "--mjcf-revision",
                    asset["revision"],
                    "--mjcf-path",
                    "g1-mode_13_15.xml",
                    "--workers",
                    str(workers),
                ],
                env=dict(
                    os.environ,
                    HF_HOME=str(ROOT / ".cache/training/hf"),
                    HF_HUB_OFFLINE="1",
                ),
                check=True,
            )
            for name, digest in sonic.items():
                install_verified(
                    Path(folder) / name.removeprefix("sonic/"), root / name, digest
                )
    for name, record in spec["metadata"].items():
        if name in expected:
            install_verified(download(record), root / name, expected[name])
    for base in {admission["sources"][name]["root"] for name in selected}:
        files = {
            p.removeprefix(base + "/"): digest
            for p, digest in expected.items()
            if p.startswith(base + "/") and p not in spec["metadata"]
        }
        if base in spec["hosted_training"]:
            source = {key: spec[key] for key in ("repo_id", "repo_type", "revision")}
            source["subfolder"] = spec["hosted_training"][base]
            hosted_files = {
                name: digest
                for name, digest in files.items()
                if not any(
                    (base + "/" + name).startswith(prefix)
                    for prefix in spec["excluded_motion_prefixes"]
                )
            }
            fetch_files(source, root / base, hosted_files)
            files = {
                name: digest
                for name, digest in files.items()
                if name not in hosted_files
            }
        source = spec["training_sources"][base]
        if files and "repo_id" in source:
            fetch_files(source, root / base, files)
    verify(root, data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "verify"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", choices=["L", "M"], default="M")
    p.add_argument(
        "--sonic-csv", type=Path, help="Licensed BONES-SEED g1/csv directory for Mixed"
    )
    p.add_argument("--workers", type=int, default=8, help="CSV conversion workers")
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == "verify":
        verify(root, a.data)
        return
    prepare(root, a.data, a.sonic_csv, a.workers)


if __name__ == "__main__":
    main()
