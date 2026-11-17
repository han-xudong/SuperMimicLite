# Training

## Recipes

| Cell | Method | Data | Environments | Iterations |
|---|---|---|---:|---:|
| P-L | PPO | LaFAN40 | 8192 | 32000 |
| P-M | PPO | Mixed | 8192 | 32000 |
| S-L | SuperMimic | LaFAN40 | 4096 | 32000 |
| S-M | SuperMimic | Mixed | 4096 | 32000 |

All recipes use seed 0 and fresh initialization. The released runs used an
RTX 5090D. Their frame counts and optimizer steps are recorded in
[`reference/training.json`](../reference/training.json).

The four selections are defined in [`recipes.json`](../configs/training/recipes.json).
PPO and SuperMimic each share one algorithm configuration; dataset paths and
sampling weights are defined in [`mixture.json`](../configs/training/mixture.json).

## Prepare

```bash
uv run scripts/bootstrap.py --training
uv sync --project environments/train --locked
uv run scripts/fetch.py asset --training-asset
uv run scripts/training_data.py prepare --root data/training --data L
```

Preparation uses pinned revisions and verifies every file hash. The
[project data hub](https://huggingface.co/datasets/han-xudong/SuperMimicLite/tree/main/training/public)
hosts 3,692 exact training files: LaFAN40, 100STYLE, ground motions, the selected
AMASS-hard motions, extreme motions, real-pico, and three non-SONIC extreme
demos. Their original source terms and attribution are retained.

SONIC motions are not hosted, including twelve extreme-demo clips and the two
high-dynamic sequences containing SONIC segments. The complete recipes retain
these motions and their original sampling weights; preparing the public portion
alone does not replace the full Mixed bank. The hub publishes selection lists,
file hashes, and source revisions for the complete bank.

For Mixed (`P-M` / `S-M`), obtain BONES-SEED's `g1.tar.gz` from the
[official repository](https://huggingface.co/datasets/bones-studio/seed) after
accepting its license, then extract the `g1/csv` directory. Convert and prepare:

```bash
uv run scripts/training_data.py prepare --root data/training --data M --sonic-csv /path/to/g1/csv --workers 8
```

Conversion uses the locked Python 3.12 training environment and the pinned G1
asset. It stages converted files separately, installs only the admitted motions,
and checks every hash. Existing matching data is reused. Source licenses,
including BONES-SEED's prohibition on public mirrors, continue to apply.

LaFAN40 contains 40 files and 441,140 control frames. Mixed contains 131,373
unique files and 50,712,626 unique frames. Counting memberships in the ten
source groups gives 131,897 entries and 51,098,520 frames. The loader preserves
these overlapping groups and their sampling weights.

## Run

```bash
bash scripts/train.sh --cell S-L --data data/training --out outputs/train-S-L-01
```

Choose a new output directory for each run. The launcher verifies the selected
files, manifests and hashes before constructing the environment.

PPO uses the MimicLite `roa_student` recipe. SuperMimic uses the integrated
world-model learner under `training/supermimic/`, a 2048 × 3 world model,
policy layers 2048 / 1024 / 512 / 256, history 10 and ten targets over 0.2 s.
Its schedule retains warmup 64, 500 world-model-only iterations, horizon 32,
UTD 0.4 and Adam at 2e-4. Complete values are in
[`ppo.json`](../configs/training/ppo.json) and
[`supermimic.json`](../configs/training/supermimic.json).

Training uses Python 3.12 / CUDA 12.8; evaluation uses the separate Python 3.10
CPU environment. Neither launcher reduces the recipe to fit a smaller device.

## Export

Download the original native checkpoints from Hugging Face, then export:

```bash
uv run scripts/fetch.py models --checkpoints
uv run --project environments/train python scripts/export.py --cell S-M --out outputs/S-M.onnx
```

The exporter builds the policy from the current recipe, strictly loads the
checkpoint, and compares native, new ONNX and released ONNX outputs on 80 inputs.
Released checkpoints provide policy weights; Uni artifacts
also contain the world model. They omit a complete optimizer resume state.
To reproduce the checkpoint learning curves, retain intermediate checkpoints
from a new training run.
