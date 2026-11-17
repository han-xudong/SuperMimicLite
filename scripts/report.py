"""Recompute every raw trajectory independently; aggregate only matched protocols."""

import argparse, json, sys
from pathlib import Path
import numpy as np
from _common import ROOT, CELLS, SPLITS, load, sha

sys.path.insert(0, str(ROOT / "evaluation"))
from independent_metrics import compute


def audit(folder):
    folder = folder.resolve()
    splits = {}
    allrows = []
    expected = load(ROOT / "manifests/motiondecode.json")
    for split in SPLITS:
        payload = load(folder / split / "tracking_metrics.json")
        bypolicy = {}
        for row in payload["rows"]:
            path = Path(row["path"]).resolve()
            if not path.is_relative_to(folder):
                raise ValueError(f"Raw outside input root: {path}")
            with np.load(path, allow_pickle=False) as f:
                data = {k: f[k] for k in f.files}
            assert int(data["seed"]) == 0
            mt = data["motion_t"]
            assert np.array_equal(
                mt, np.arange(1, int(data["motion_length"]), dtype=mt.dtype)
            )
            for key, value in data.items():
                if value.dtype.kind in "fc":
                    assert np.isfinite(value).all(), (path, key)
            metrics = compute(data)
            for key, value in metrics.items():
                if isinstance(value, str) or key in [
                    "terminated",
                    "termination_motion_t",
                ]:
                    assert row[key] == value, (path, key)
                else:
                    assert (
                        abs(float(row[key]) - value)
                        <= 2e-5 + abs(float(row[key])) * 2e-6
                    ), (
                        path,
                        key,
                    )
            cell = row["policy"]
            assert cell in CELLS
            bypolicy.setdefault(cell, []).append(metrics)
            allrows.append(
                {
                    "split": split,
                    "cell": cell,
                    "motion_index": row["motion_index"],
                    "trajectory": str(path.relative_to(folder)),
                    "sha256": sha(path),
                    **metrics,
                }
            )
        assert set(bypolicy) == set(CELLS)
        splits[split] = {}
        for cell, rows in bypolicy.items():
            assert len(rows) == expected[split]["count"]
            value = {
                k: float(np.mean([r[k] for r in rows]))
                for k, v in rows[0].items()
                if isinstance(v, (int, float))
                and k not in ["terminated", "termination_motion_t"]
            }
            value.update(
                count=len(rows), completed=sum(not r["terminated"] for r in rows)
            )
            splits[split][cell] = value
    assert len(allrows) == 912
    primary = {}
    for cell in CELLS:
        primary[cell] = {
            k: (splits["locomotion"][cell][k] + splits["manipulation"][cell][k]) / 2
            for k in splits["locomotion"][cell]
            if k not in ["count", "completed"]
        }
        for key in [
            "root_final_error_norm",
            "root_final_error_xy_norm",
            "root_final_error_z_abs",
        ]:
            primary[cell][key] = splits["locomotion"][cell][key]
        primary[cell].update(
            body_mm=primary[cell]["local_body_tracking_error"] * 1000,
            wrist_mm=primary[cell]["wrist_tracking_error"] * 1000,
            progress_pct=primary[cell]["progress"] * 100,
        )
    return {
        "passed": True,
        "raw_count": 912,
        "splits": splits,
        "primary": primary,
    }, allrows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    aggregate, rows = audit(a.input)
    a.out.mkdir(parents=True, exist_ok=False)
    for name, value in [("aggregate.json", aggregate), ("per-motion.json", rows)]:
        (a.out / name).write_text(json.dumps(value, indent=2) + "\n")
    table = [
        "# Four-model reproduction",
        "",
        "Locomotion80 and Manipulation48 receive equal weight. Root uses Locomotion80 only.",
        "",
        "| Model | Body mm ↓ | Wrist mm ↓ | Root m ↓ | No failure / 228 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in CELLS:
        r = aggregate["primary"][cell]
        count = sum(aggregate["splits"][s][cell]["completed"] for s in SPLITS)
        table.append(
            f"| {cell} | {r['body_mm']:.2f} | {r['wrist_mm']:.2f} | {r['root_final_error_norm']:.3f} | {count}/228 |"
        )
    (a.out / "RESULTS.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
