"""One whole fixed motion per model: real inference and CPU physics, fresh outputs."""

import argparse, json, subprocess, sys
from pathlib import Path
import yaml
from _common import ROOT, CELLS, artifacts, author_env, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cells", choices=CELLS, nargs="+", default=list(CELLS))
    a = p.parse_args()
    artifacts()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    split = ROOT / "data/motiondecode/locomotion"
    motion = split / (split / "motion-list.txt").read_text().splitlines()[0]
    results = {}
    for cell in a.cells:
        cfg = yaml.safe_load((ROOT / "configs/deploy" / f"{cell}.yaml").read_text())
        cfg["model_path"] = str(ROOT / "models" / f"{cell}.onnx")
        target = out / f"{cell}.yaml"
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        traj = out / f"{cell}.npz"
        cmd = [
            sys.executable,
            "-m",
            "sim2real.sim_env.integrated_sim2sim",
            "--robot",
            "g1",
            "--inference-backend",
            "onnx-cpu",
            "--policy-config",
            str(target),
            "--motion-path",
            str(motion),
            "--headless",
            "--run-once",
            "--initial-pause-s",
            "0",
            "--trajectory-output",
            str(traj),
            "--seed",
            "0",
        ]
        with (out / f"{cell}.log").open("x") as f:
            rc = subprocess.run(
                cmd, env=author_env(), stdout=f, stderr=subprocess.STDOUT
            ).returncode
        results[cell] = {"actual_cli": rc, "motion": str(motion.relative_to(ROOT))}
        if rc:
            raise SystemExit(f"{cell} failed: {out}/{cell}.log")
        results[cell]["trajectory_sha256"] = sha(traj)
    (out / "COMPLETE.json").write_text(
        json.dumps({"passed": True, "results": results}, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
