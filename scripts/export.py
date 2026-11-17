"""Export complete native policies and frozen normalization, then compare CPU outputs."""

import argparse, json, os, sys
from pathlib import Path
from _common import ROOT, CELLS, artifacts, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", choices=CELLS, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    artifacts("checkpoints")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = list(
        map(str, [ROOT / "training", ROOT / "third_party/active-adaptation"])
    )
    import numpy as np, onnxruntime as ort, torch
    from omegaconf import OmegaConf
    from inference import PPOInference, SMInference
    from recipe_config import ppo_config, sm_training_config

    torch.set_num_threads(2)
    saved = torch.load(
        ROOT / "checkpoints" / f"{a.cell}.pt",
        map_location="cpu",
        weights_only=a.cell[0] == "S",
    )
    if a.cell[0] == "P":
        assert saved["policy"]["last_iter"] == 31999
        cfg = ppo_config(a.cell[-1])
        model = PPOInference(saved, OmegaConf.to_container(cfg, resolve=True))
        config_path = ROOT / "configs/training/ppo.json"
        names = ["policy", "command"]
        shapes = [535, 304]
    else:
        assert saved["iteration"] == 32000
        model = SMInference(saved, sm_training_config()["algo"])
        config_path = ROOT / "configs/training/supermimic.json"
        names = ["proprioception_history", "goal"]
        shapes = [99 * model.hist, model.goal_dim]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            tuple(torch.zeros(k) for k in shapes),
            str(a.out),
            input_names=names,
            output_names=["action"],
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        for path in [a.out, ROOT / "models" / f"{a.cell}.onnx"]
    ]
    rng = np.random.default_rng(31)
    maxerr = 0.0
    for k in range(80):
        values = [
            rng.normal(size=n).astype(np.float32)
            * (0 if k == 0 else 0.3 if k < 40 else 1.0)
            for n in shapes
        ]
        with torch.inference_mode():
            native = model(*map(torch.from_numpy, values)).numpy()
        for session in sessions:
            actual = session.run(["action"], dict(zip(names, values)))[0]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, native, atol=1e-4, rtol=1e-4)
            maxerr = max(maxerr, float(np.max(np.abs(actual - native))))
    assert not torch.cuda.is_initialized()
    a.out.with_suffix(".parity.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cell": a.cell,
                "samples": 80,
                "training_config_sha256": sha(config_path),
                "max_abs_error": maxerr,
                "sha256": sha(a.out),
                "native_and_release_onnx_compared": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{a.cell}: complete native policy, frozen normalization, 80 CPU parity cases passed"
    )


if __name__ == "__main__":
    main()
