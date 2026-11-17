"""Portable paths and pinned artifact checks, shared by command entry points."""

import hashlib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ("P-L", "P-M", "S-L", "S-M")
SPLITS = ("locomotion", "manipulation", "ground", "dance")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def artifacts(directory="models"):
    rows = [
        r
        for r in load(ROOT / "manifests/artifacts.json")["files"]
        if r["directory"] == directory
    ]
    for r in rows:
        p = ROOT / directory / r["name"]
        if not p.is_file():
            raise FileNotFoundError(f"{p}: run uv run scripts/fetch.py models")
        if p.stat().st_size != r["bytes"] or sha(p) != r["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {p}")
    return rows


def author_env():
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        G1_LEADERBOARD_REGISTER="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HOME=str(ROOT / ".cache/evaluation/hf"),
        ANY4HDMI_QPOS_CACHE_ROOT=str(ROOT / ".cache/evaluation/any4"),
        ANY4HDMI_CACHE_BUILD_NUM_WORKERS="0",
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        SIM2REAL_ORT_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        map(str, [ROOT / "evaluation", ROOT / "third_party/sim2real"])
    )
    return env
