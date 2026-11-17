# Evaluation

Evaluation uses CPU ONNX Runtime and MuJoCo 3.10 with the pinned
[sim2real evaluator](https://github.com/EGalahad/sim2real/tree/287ac157e6a7143340da7a9fc2fa5ff552031bc8).

| Subset | Motions | Metrics |
|---|---:|---|
| Locomotion | 80 | Body, wrist and root errors |
| Manipulation | 48 | Body and wrist errors |
| Ground | 60 | Completion and tracking errors |
| Dance | 40 | Completion and tracking errors |

All policies use seed 0, frozen normalization and complete clips at 50 Hz.
Motion content and ordering are pinned in `manifests/evaluation/`. The four
[models](https://huggingface.co/han-xudong/SuperMimicLite) and
[228 motion files](https://huggingface.co/datasets/han-xudong/SuperMimicLite/tree/main/evaluation)
are public. `uv run scripts/fetch.py all` downloads both from this project's
Hugging Face repositories at fixed revisions. For evaluation data alone, run:

```bash
uv run scripts/fetch.py data
```

The downloader verifies every file hash and the fixed motion order. The source
G1 trajectories remain at 120 Hz; the evaluator uses 50 Hz policy frames.
The original MotionDecode attribution, ChingMu terms, selection manifests and
four upstream source revisions are retained alongside the mirrored files.

## Policy inputs

PPO consumes 535 policy and 304 command values. SuperMimic consumes 990 history
and 730 goal values; its ONNX includes the encoder and decoder. Joint history,
preview and actions follow each policy's declared 29-joint order.

Deployment uses the evaluator's direct PD controller and nominal physics.
PPO's randomized training delay and alpha are not applied during evaluation.

## Scoring

The first failure occurs when pelvis height error exceeds 0.30 m, or the
absolute difference between the robot and reference pelvis rotation-matrix
R22 entries exceeds 0.8. Body and wrist errors use the pre-failure interval.
Completion means reaching the clip's end before either condition. Average
progress is reported separately from completion count.

The main body and wrist scores average the Locomotion80 and Manipulation48
subset means equally. Root error uses Locomotion80 only. Ground and Dance
remain separate subsets.

The learning curve uses the evaluator's two-term tracking return, whose
maximum is 2. It contains 64 measured points: four policies at 16 checkpoints,
every 2k iterations through 32k. Its score definition differs from the paper's
Appendix-F return.

## Recompute results

```bash
uv run scripts/evaluate.py --out outputs/benchmark-01 --workers 4
uv run scripts/report.py --input outputs/benchmark-01/evaluation --out outputs/report-01
```

The report independently recomputes metrics from all 912 raw trajectories and
rejects incomplete output. Reference [aggregate](../reference/aggregate.json)
and [per-motion](../reference/per-motion.json) results are retained for comparison.

These are simulation results from one seed. Training and evaluation data may
overlap, and the complete PPO and SuperMimic recipes use different sample
counts, networks and observations.
