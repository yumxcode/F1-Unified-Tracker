# F1-Unified-Tracker

Unified whole-body motion tracking for the Weichai **F1 29-DoF** humanoid: one
policy, one set of weights, tracking **walking, running and dancing**, with an
explicit **stand → skill → stand** entry/exit protocol and a frozen, source-
calibrated quality standard.

## What is in this repo

| Path | Purpose |
|---|---|
| `whole_body_tracking/` | BeyondMimic-style tracking training (IsaacLab / rsl_rl), extended with a multi-clip motion library, skill + contact-phase observations, flight/stance rewards and idle transition augmentation |
| `whole_body_tracking/motions/f1_unified_library.npz` | 17-clip, 50 Hz motion library: 8 walk (2 UMR ACCAD), 4 run, 4 dance, 1 idle; 139,930 frames (46.6 min) |
| `tools/` | Retargeting curation, contact-consistent repair, clip/library builders, source metrics |
| `docs/STANDARDS.md` | Frozen Tier-A retargeting and Tier-B sim2sim gates, with source-data derivation and change-control rules |
| `whole_body_tracking/scripts/play_unified.py` | Sim2sim playback from a standing start to a stop; entry/in-motion/exit metrics |

## Method

1. **Contact-consistent retargeting repair.** GMR/UMR map the human foot pitch
   onto a rigid 20 cm F1 sole, which digs the toe into the floor and keeps the
   swing foot on the ground. `tools/repair/contact_repair.py` re-solves
   whole-body IK per frame with flat feet, stance-foot locking at contact
   onset, swing clearance, a per-joint velocity clamp and a final ground-lift
   pass. Contact is taken from the **source mocap** (LAFAN velocity 0.002
   m/frame), not from the retargeted foot height.
2. **Frozen quality gates.** Tier A (FK, per clip): penetration, self-collision,
   joint limits/velocity, root height, continuity, flat-foot, arm quality,
   stance openness, foot skating. Tier B (sim2sim): success, falls, tracking
   error, contact mismatch, slip, flight fraction, action rate, entry/exit.
   Thresholds are derived from `tools/source_metrics.py` and frozen in
   `docs/STANDARDS.md`; post-hoc lowering is prohibited.
3. **Unified tracker.** A multi-clip `MotionCommand` samples clips and time
   bins adaptively, with a 15 % idle share. The policy observes
   `[ref joint pos/vel, anchor pose, base velocity, joint state, last action,
   skill one-hot, reference contact phase]` (167-D) and is trained with the
   BeyondMimic tracking reward plus a CLF-RL flight contact penalty and
   stance-foot anti-skate terms. Idle prefixes/suffixes are built into every
   clip, so the policy learns to start from a stand and stop.
4. **Sim2sim protocol.** `play_unified.py` starts from the default stand, plays
   `3 s idle → 0.5 s settle → motion → 2 s stop hold`, never loops, and reports
   entry, in-motion and exit success separately.

## Train (gradmotion, 1×4090D 24 GB)

```bash
gm-run F1-Unified-Tracker/whole_body_tracking/scripts/entry_f1_train.py \
  --task=Tracking-Flat-F1-v0 \
  --motion_file ../motions/f1_unified_library.npz \
  --headless --num_envs 4096 --max_iterations 8000
```

The entry script `pip install -e source/whole_body_tracking --no-deps`, then
launches `rsl_rl/train.py`.

## Evaluate

```bash
python whole_body_tracking/scripts/play_unified.py \
  --library whole_body_tracking/motions/f1_unified_library.npz \
  --checkpoint trained_models/unified_model_7999.pt \
  --clip walk1_subject1_walk --headless --metrics out/walk1.npz
```

## Data and licence

- **LAFAN1** (Ubisoft La Forge) — CC BY-NC-ND 4.0; used for internal,
  non-commercial research. Derived motion is not redistributed publicly.
- **UMR** retargeting (local `UMR_weilai`) for the two ACCAD walk clips.
- **GMR** (MIT) for the LAFAN BVH clips.

## Status

- [x] F1 29-DoF asset audit and PD gains
- [x] 17-clip Tier-A-passing motion library (8 walk / 4 run / 4 dance / 1 idle)
- [x] Multi-clip command, skill/contact obs, flight/stance rewards
- [x] Sim2sim protocol with joint-order self-check
- [ ] Training on gradmotion (TASK_20261123_325) and Tier-B evaluation
- [ ] Final report and videos
