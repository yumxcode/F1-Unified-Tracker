# F1-Unified-Tracker

Unified whole-body motion tracking for the Weichai **F1 29-DoF** humanoid: one
policy, one set of weights, tracking **walking, running and dancing**, with an
explicit **stand → skill → stand** entry/exit protocol and a frozen, source-
calibrated quality standard.

## What is in this repo

| Path | Purpose |
|---|---|
| `whole_body_tracking/` | BeyondMimic-style tracking training (IsaacLab / rsl_rl), extended with a multi-clip motion library, skill + contact-phase observations, flight/stance rewards and idle transition augmentation |
| `whole_body_tracking/motions/f1_unified_library.npz` | **13-clip**, 50 Hz library: 6 walk, 4 run, 2 dance, 1 idle; 37,189 frames (743.8 s) |
| `whole_body_tracking/motions/f1_run_library.npz` | 4-run + idle specialist library (242 s) |
| `trained_models/v5c2_15000.pt` | Final unified policy (15k iterations, aligned contact labels, 4096 envs, 4090D 24 GB) |
| `tools/` | Retargeting curation, contact-consistent repair, clip/library builders, source metrics, Tier-B evaluation |
| `docs/STANDARDS.md` | Frozen Tier-A (A1–A12) and Tier-B (B1–B11) gates, source-data derivation, change control |
| `docs/REPORT.md` | Experiment report: data, method, per-gate results, limitations |
| `videos/` | Sim2sim clips: idle / walk / run / dance (v5b_15000) |
| `whole_body_tracking/scripts/play_unified.py` | Sim2sim playback from a standing start to a stop; entry/in-motion/exit metrics |

## Method

1. **Contact-consistent retargeting repair.** GMR/UMR map the human foot pitch
   onto a rigid 20 cm F1 sole. `tools/repair/contact_repair.py` re-solves
   whole-body IK per frame with flat feet, stance-foot locking, swing clearance,
   a velocity clamp and a **local ground correction**: a floating stance foot
   lowers the base; a penetrating stance foot raises the base by exactly its
   deficit; a penetrating swing foot lifts only its own target. The old global
   post-lift (which floated the stance foot) is disabled. Contact is taken from
   the **source mocap** (`v < 0.02 m/frame` **and** `z < 0.12 m`).
2. **Frozen quality gates.** Tier A (FK, per clip): A1–A12, including the new
   **A12 stance-foot grounded rate ≥95 %** (source stance frames, sole z ≤2 cm).
   Tier B (sim2sim): success, falls, tracking error, contact mismatch, slip,
   flight fraction, action rate, entry/exit. Thresholds are derived from
   `tools/source_metrics.py` and frozen in `docs/STANDARDS.md` (v1.1).
3. **Unified tracker.** Multi-clip `MotionCommand` with skill-balanced sampling
   (walk/run/dance equal, 15 % idle). The policy observes 167-D
   `[ref joint pos/vel, anchor pose, base velocity, joint state, last action,
   skill one-hot, reference contact phase]` and is trained with the BeyondMimic
   tracking reward plus a CLF-RL flight contact penalty and stance-foot
   anti-skate terms. Idle prefixes/suffixes are built into every clip.
4. **Sim2sim protocol.** `play_unified.py` starts from the reference stand,
   plays the clip once (idle entry → skill → idle exit), never loops, and reports
   entry / in-motion / exit separately with B1–B11 gates (`tools/eval_unified.py`).

## Train (gradmotion, 1×4090D 24 GB)

```bash
gm-run F1-Unified-Tracker/whole_body_tracking/scripts/entry_f1_train.py \
  --task=Tracking-Flat-F1-v0 \
  --motion_file motions/f1_unified_library.npz \
  --headless --num_envs 4096 --max_iterations 15000
```

## Evaluate

```bash
python whole_body_tracking/scripts/play_unified.py \
  --library whole_body_tracking/motions/f1_unified_library.npz \
  --checkpoint trained_models/v5b_15000.pt \
  --clip walk1_subject1_walk --headless --record out.mp4

python tools/eval_unified.py \
  --library whole_body_tracking/motions/f1_unified_library.npz \
  --checkpoint trained_models/v5b_15000.pt --out_dir eval/v5b
```

## Results (v5b_15000, see `docs/REPORT.md`)

- **Tier A**: 12/14 clips pass A1–A12 (walk 6/6, run 4/4, dance 2/4); the two
  dance failures are A8 wrist ROM 1.56 % < 2 % and A12 94.8 % < 95 %.
- **Tier B, full 13 clips (v5c2, STANDARDS v1.2, 3 seeds x 2 replays)**:
  idle 100 % (all B1–B11, B3 median 28.7 mm); walk 88.9 % success but
  B2 11.1 %, B3 185 mm, B6 21.6 %, B7 0.226 m/s, entry 33.3 %/exit 72.2 %;
  run 16.7 %; dance 50 %. See `docs/REPORT.md` §3.7 for the standard→code map.
- **Run (jog, ≤1.2 m/s, A2c 10k + 2.0 s blend, 3 seeds x 2)**: success 100 %,
  fall 0 %, entry/exit 100 %, stop drift 2.2 cm — **B1/B2/B10 pass**; B3 128 mm,
  B5 0.80 m/s, B6 17.8 %, B7 0.111 m/s, B8 still fail. Fast run 1.65–2.1 m/s
  remains infeasible.
- **Contact-label alignment fix**: the BVH offsets of 8/12 clips were wrong
  (window[0] not used); fixed and asserted (BVH vs FK stance agreement ≥90%).
  Re-evaluating v5b on aligned labels drops walk B6 35.7→18.4 % and B7
  1.13→0.23 m/s; the old B6/B7 attribution is void.
- **Run**: after label alignment, v5c2/v9c pass the slowest clip only (25 %);
  fast run remains an open limitation, see `docs/REPORT.md` §3.3/§3.5.

## Data and licence

- **LAFAN1** (Ubisoft La Forge) — CC BY-NC-ND 4.0; internal, non-commercial
  research. Derived motion is not redistributed publicly.
- **UMR** retargeting (local `UMR_weilai`) for the two ACCAD walk clips.
- **GMR** (MIT) for the LAFAN BVH clips.

## Status

- [x] F1 29-DoF asset audit and PD gains
- [x] Contact-consistent repair + A12 stance-grounded gate; 12/14 Tier-A clips
- [x] 13-clip library + 4-run library
- [x] Unified policy v5b_15000; sim2sim from stand to stop
- [x] Tier-B per-gate evaluation and report/videos
- [ ] Run: jog (≤1.2 m/s) passes B1/B2/B10; fast clips and strict B3–B8 remain open
