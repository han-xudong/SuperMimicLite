"""Reject changed filters and extra motions before constructing a training env."""

import hashlib
import pytest
from training_data import selection_summary, verify_source_selection


def test_overlapping_sources_count_unique_frames_separately():
    admission = {
        "sources": {
            "main": {"selected_files": ["a", "b"]},
            "subset": {"selected_files": ["b", "c"]},
        },
        "file_records": {
            p: {"control_frames": n} for p, n in [("a", 10), ("b", 20), ("c", 30)]
        },
    }
    assert selection_summary(admission, ["main", "subset"]) == {
        "files": 3,
        "frames_50Hz": 60,
        "source_entries": 4,
        "source_frames_50Hz": 80,
    }


def make_source(tmp_path):
    base = tmp_path / "source"
    (base / "motions").mkdir(parents=True)
    (base / "manifest.json").write_text("{}")
    (base / "motions/a.npz").write_bytes(b"not-loaded-by-this-membership-test")
    return base, {
        "root": "source",
        "manifest_sha256": hashlib.sha256(b"{}").hexdigest(),
        "filter_sha256": None,
        "selected_files": ["source/motions/a.npz"],
    }


def test_unfiltered_source_rejects_extra_motion(tmp_path):
    base, source = make_source(tmp_path)
    verify_source_selection(tmp_path, "example", source, {})
    (base / "motions/extra.npz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="selection differs"):
        verify_source_selection(tmp_path, "example", source, {})


def test_filtered_source_rejects_modified_filter(tmp_path):
    base, source = make_source(tmp_path)
    filter_path = base / "selected.txt"
    filter_path.write_text("a.npz\n")
    source["filter_sha256"] = hashlib.sha256(filter_path.read_bytes()).hexdigest()
    cfg = {"filenames_path": "selected.txt"}
    verify_source_selection(tmp_path, "example", source, cfg)
    filter_path.write_text("extra.npz\n")
    with pytest.raises(ValueError, match="filter mismatch"):
        verify_source_selection(tmp_path, "example", source, cfg)
