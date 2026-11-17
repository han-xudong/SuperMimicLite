"""Four independent fresh recipes. Existing outputs cannot be overwritten."""

import argparse, dataclasses, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    recipes = json.loads((ROOT / "configs/training/recipes.json").read_text())
    p.add_argument("--cell", choices=list(recipes), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    sys.path.append(str(ROOT / "scripts"))
    from training_data import verify

    recipe = recipes[a.cell]
    verify(a.data.resolve(), recipe["data"])
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        G1_TRAIN_OUTPUT=str(out),
        G1_TRAIN_DATA=str(a.data.resolve()),
        WANDB_MODE="disabled",
    )
    sys.path[:0] = [
        str(ROOT / "training"),
        str(ROOT / "third_party/active-adaptation"),
    ]
    import active_adaptation as aa

    aa.set_backend("mjlab")
    import torch
    from recipe_config import ppo_config, sm_config, sm_training_config

    start = time.monotonic()
    if recipe["method"] == "ppo":
        from omegaconf import OmegaConf
        from ppo_runner import run
        import ppo_observe

        ppo_observe.OUT = out
        cfg = ppo_config(recipe["data"])
        (out / "config.json").write_text(
            json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
        )
        os.chdir(out)
        run(cfg, out)
        assert ppo_observe.COUNTS == {"actor": 1280000, "critic": 1280000}
        torch.save(
            {
                "actor": ppo_observe.POLICY.opt_policy.state_dict(),
                "critic": ppo_observe.POLICY.opt_critic.state_dict(),
            },
            out / "optimizer-final-external.pt",
        )
    else:
        from reference import load_collection
        from g1_env import G1Env, G1Physics

        from observe import ObservedLearner

        cfg = sm_config()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        num_envs = sm_training_config()["num_envs"]
        collection = load_collection(a.cell[-1], num_envs, "cuda:0")
        env = G1Env(collection, physics=G1Physics(num_envs, collection.device))
        learner = ObservedLearner(env, cfg, out / "training", backend="MimicLite/mjlab")
        torch.save(
            {k: getattr(learner, k).state_dict() for k in ["wm", "pi", "critic"]},
            out / "fresh-initial.pt",
        )
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        learner.train()
        assert learner.iteration == 32000
        assert learner.counts == {"wm": 320000, "pi": 31500, "critic": 630000}
        env.close()
    (out / "COMPLETE.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cell": a.cell,
                "iterations": 32000,
                "seed": 0,
                "wall_seconds": time.monotonic() - start,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
