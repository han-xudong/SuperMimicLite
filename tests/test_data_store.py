"""Pinned source downloads must preserve validated local data."""

import hashlib

import pytest
import data_store


def source_file(tmp_path, monkeypatch, content=b"motion-bytes"):
    cached = tmp_path / "cached.npz"
    cached.write_bytes(content)
    monkeypatch.setattr(data_store, "hf_hub_download", lambda **kwargs: cached)
    return hashlib.sha256(content).hexdigest()


def test_fetches_missing_files_and_reuses_verified_files(tmp_path, monkeypatch):
    digest = source_file(tmp_path, monkeypatch)
    root = tmp_path / "bank"
    data_store.fetch_files({}, root, {"motions/a.npz": digest})
    target = root / "motions/a.npz"
    inode = target.stat().st_ino
    data_store.fetch_files({}, root, {"motions/a.npz": digest})
    assert target.read_bytes() == b"motion-bytes" and target.stat().st_ino == inode


def test_changed_local_data_is_not_overwritten(tmp_path, monkeypatch):
    digest = source_file(tmp_path, monkeypatch)
    root = tmp_path / "bank"
    root.mkdir()
    target = root / "a.npz"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing data differs"):
        data_store.fetch_files({}, root, {"a.npz": digest})
    assert target.read_bytes() == b"changed"


def test_bad_source_does_not_leave_a_partial_file(tmp_path, monkeypatch):
    source_file(tmp_path, monkeypatch, b"wrong")
    root = tmp_path / "bank"
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        data_store.fetch_files(
            {}, root, {"a.npz": hashlib.sha256(b"expected").hexdigest()}
        )
    assert not root.exists()


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside"])
def test_rejects_paths_outside_bank(tmp_path, monkeypatch, name):
    digest = source_file(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Invalid source path"):
        data_store.fetch_files({}, tmp_path / "bank", {name: digest})


def test_rejects_symlink_outside_bank(tmp_path, monkeypatch):
    digest = source_file(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "bank"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Invalid source path"):
        data_store.fetch_files({}, root, {"link/a.npz": digest})


def test_partial_mirror_fetches_excluded_clips_only_from_original_source(
    tmp_path, monkeypatch
):
    import training_data

    base = "public/any4hdmi-g1-extreme-demo"
    payloads = {
        "manifest.json": b"canonical manifest for the complete collection",
        "motions/mimickit/kick.npz": b"public clip",
        "motions/sonic/roll.npz": b"source-only clip",
    }
    hashes = {
        name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()
    }
    admission = {
        "sources": {
            "extreme_demo": {
                "root": base,
                "selected_files": [
                    base + "/" + name for name in payloads if name.endswith(".npz")
                ],
                "manifest_sha256": hashes["manifest.json"],
            }
        },
        "file_records": {
            base + "/" + name: {"sha256": digest} for name, digest in hashes.items()
        },
    }
    spec = {
        "repo_id": "project/data",
        "repo_type": "dataset",
        "revision": "release",
        "training_sources": {
            base: {
                "repo_id": "original/data",
                "repo_type": "dataset",
                "revision": "original",
            }
        },
        "hosted_training": {base: "training/" + base},
        "excluded_motion_prefixes": [base + "/motions/sonic/"],
        "metadata": {},
    }
    calls = []

    def remote_file(**kwargs):
        name = kwargs["filename"]
        excluded = name.startswith("motions/sonic/")
        assert kwargs["repo_id"] == ("original/data" if excluded else "project/data")
        assert kwargs["revision"] == ("original" if excluded else "release")
        assert kwargs.get("subfolder") == (None if excluded else "training/" + base)
        cached = tmp_path / ("cached-" + name.replace("/", "-"))
        cached.write_bytes(payloads[name])
        calls.append(name)
        return cached

    root = tmp_path / "bank"
    verified = []

    def verify_bank(path, data):
        assert path == root and data == "M"
        for name, content in payloads.items():
            assert (root / base / name).read_bytes() == content
        verified.append(True)

    monkeypatch.setattr(training_data, "load_admission", lambda root: admission)
    monkeypatch.setattr(training_data, "verify", verify_bank)
    monkeypatch.setattr(data_store, "specification", lambda: spec)
    monkeypatch.setattr(data_store, "hf_hub_download", remote_file)
    training_data.prepare(root, "M")
    training_data.prepare(root, "M")
    assert sorted(calls) == sorted(payloads) and verified == [True, True]
