"""Run the unchanged author evaluator with native policy I/O adapters."""

import argparse, json, subprocess, sys
from pathlib import Path
import yaml
from _common import ROOT, CELLS, SPLITS, artifacts, author_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cells", choices=CELLS, nargs="+", default=list(CELLS))
    p.add_argument("--splits", choices=SPLITS, nargs="+", default=list(SPLITS))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    artifacts()
    assert a.workers > 0
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    deploy = out / "deploy"
    deploy.mkdir()
    cmd = [
        sys.executable,
        str(
            ROOT
            / "third_party/sim2real/scripts/tracking_experiment/run_motiondecode_eval.py"
        ),
        "--datasets-root",
        str(ROOT / "data/motiondecode"),
        "--output-dir",
        str(out / "evaluation"),
        "--max-workers",
        str(a.workers),
        "--splits",
        *a.splits,
    ]
    for cell in a.cells:
        cfg = yaml.safe_load((ROOT / "configs/deploy" / f"{cell}.yaml").read_text())
        cfg["model_path"] = str(ROOT / "models" / f"{cell}.onnx")
        target = deploy / f"{cell}.yaml"
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        cmd += ["--policy", cell + "=" + str(target)]
    if a.dry_run:
        cmd += ["--dry-run"]
    with (out / "author.log").open("x") as f:
        result = subprocess.run(
            cmd, env=author_env(), stdout=f, stderr=subprocess.STDOUT
        )
    (out / "exit.json").write_text(
        json.dumps(
            {
                "actual_cli": result.returncode,
                "command": cmd,
                "cells": a.cells,
                "splits": a.splits,
                "seed": 0,
                "protocol": "author CPU ONNX + CPU MuJoCo; frame0; fullclip; unchanged scorer",
            },
            indent=2,
        )
        + "\n"
    )
    if result.returncode:
        raise SystemExit(
            f"Author evaluator failed ({result.returncode}); inspect {out}/author.log. No automatic retry."
        )
    print(f"Completed: {out}")


if __name__ == "__main__":
    main()
