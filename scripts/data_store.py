"""Fetch verified files from their pinned source repositories."""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from _common import ROOT, load, sha


def specification():
    return load(ROOT / "manifests/datasets.json")


def download(record):
    spec = specification()
    p = Path(
        hf_hub_download(
            repo_id=spec["repo_id"],
            repo_type=spec["repo_type"],
            revision=spec["revision"],
            filename=record["path"],
        )
    )
    if p.stat().st_size != record["bytes"] or sha(p) != record["sha256"]:
        raise ValueError(f"Metadata checksum mismatch: {record['path']}")
    return p


def install_verified(source, target, digest):
    source, target = Path(source), Path(target)
    if target.exists():
        if not target.is_file() or sha(target) != digest:
            raise ValueError(f"Existing data differs: {target}")
        return
    if sha(source) != digest:
        raise ValueError(f"Source checksum mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, out)
            out.close()
            if sha(temporary) != digest:
                raise ValueError(f"Copied data mismatch: {target}")
            try:
                target.hardlink_to(temporary)
            except FileExistsError:
                if not target.is_file() or sha(target) != digest:
                    raise ValueError(f"Existing data differs: {target}")
        finally:
            temporary.unlink(missing_ok=True)


def fetch_files(source, destination, expected):
    destination = Path(destination).resolve()
    for name, digest in expected.items():
        target = destination / name
        if Path(name).is_absolute() or not target.resolve().is_relative_to(destination):
            raise ValueError(f"Invalid source path: {name}")
        if target.exists():
            if not target.is_file() or sha(target) != digest:
                raise ValueError(f"Existing data differs: {target}")
            continue
        cached = hf_hub_download(filename=name, **source)
        install_verified(cached, target, digest)
