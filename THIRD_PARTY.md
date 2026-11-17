# Third-party sources and licenses

The root [Apache-2.0 license](LICENSE) covers SuperMimicLite's original
contributions. It does not relicense upstream material, model assets, released
weights, or motion data.

| Component | Revision | Use |
|---|---|---|
| [MimicLite](https://github.com/EGalahad/mimic-lite) | `94dd77a827c2b28d6b69efc3291e392e0f07c906` | G1 assets, tracking task and PPO policy; adapted training loop in `training/ppo_runner.py` |
| [Active Adaptation](https://github.com/Agent-3154/active-adaptation) | `b6a6b3f101344dfe8d5bdd6a06e921f4333127eb` | Simulation framework and environment construction |
| [Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | `faf7818e8abac3ec6abb8f00e6e904d3577f800b` | Learner, networks and replay adapted into `training/supermimic/`; G1 task integrated with MimicLite |
| [sim2real](https://github.com/EGalahad/sim2real) | `287ac157e6a7143340da7a9fc2fa5ff552031bc8` | CPU evaluator and deployment interface |
| [Any4 HDMI](https://github.com/EGalahad/any4hdmi) | Pinned separately in the training and evaluation locks | Motion loading and caches |
| [G1 assets](https://huggingface.co/elijahgalahad/g1_xmls) | `manifests/upstream.json` | Training and evaluation robot models |

## Included code

Uni-SuperMimic uses Apache-2.0; its complete license text is retained as the
root [LICENSE](LICENSE), and its original notice is reproduced in [NOTICE](NOTICE).
The [MIT license and copyright notice](LICENSES/MIT-supermimic.txt) for its
SuperMimic reference implementation are also retained. Adapted files in
`training/supermimic/` identify their origin and modifications.

`training/ppo_runner.py` adapts MimicLite's `scripts/train.py`. The G1 bindings,
task configuration, and motion handling also incorporate or modify material
from MimicLite and Active Adaptation.
At the pinned revisions above, MimicLite, Active Adaptation, and sim2real do
not include a repository-wide license. Active Adaptation has MIT notices in
some individual files; these are not a license for its entire repository.
The project's Apache-2.0 grant does not cover those upstream portions or
establish permission to redistribute them. Their licensing requires
clarification from the respective authors.

External framework checkouts are fetched by `scripts/bootstrap.py` and are
not distributed in this Git repository.

## Models and motion data

Model weights are hosted on [Hugging Face](https://huggingface.co/han-xudong/SuperMimicLite).
The [data hub](https://huggingface.co/datasets/han-xudong/SuperMimicLite)
hosts 3,692 exact training files from the public non-SONIC portions of the pinned
Any4 HDMI sources: LaFAN40, 100STYLE, ground motions, selected AMASS-hard motions,
extreme motions, real-pico, and three extreme-demo clips. These are byte-identical
mirrors of the existing G1 trajectories; no additional retargeting is performed.
Selection metadata, hashes, source revisions, and per-collection attribution are
retained. Public availability is not an additional redistribution license, and
the project's Apache-2.0 license does not apply to these motion files.

Original 100STYLE motion capture is by Ian Mason, Sebastian Starke, and Taku
Komura under [CC BY 4.0](https://www.ianxmason.com/100style/). Its source manifest
records Axellwppr MotionDataset and Any4 HDMI processing. The source
[LAFAN1 terms](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
retain CC BY-NC-ND 4.0, and [AMASS](https://amass.is.tue.mpg.de/license.html)
retains its non-commercial research terms and redistribution restrictions.
Other Any4 HDMI training repositories do not provide a separate data license
at the pinned revisions; their original source terms continue to apply. No
additional licensing authorization is asserted for the mirrored trajectories.

SONIC and SONIC-derived trajectories are excluded from this project's public
mirror, including twelve `extreme-demo` clips and the two high-dynamic sequences
containing SONIC segments. They remain in the complete local training bank.

The data hub also mirrors the fixed 228-motion evaluation subset of
[ChingMu MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode),
using the four G1 subsets pinned in [`motiondecode.json`](manifests/motiondecode.json).
The original 120 Hz trajectories, motion order, per-file hashes, source selection
manifests, attribution and ChingMu non-commercial access terms are retained
without changes. The dataset card's `source_datasets` metadata records the
direct and original sources; the model card's `datasets` metadata links the
released policies to this project's training-data repository.

Training data includes [Motion Data by Bones Studio](https://bones.studio/).
Use of the underlying dataset is subject to the
[BONES Motion Capture Dataset License Agreement](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md).
Mixed training uses licensed BONES-SEED G1 CSVs converted locally; its source
license prohibits public mirrors and redistribution of the motion data.
