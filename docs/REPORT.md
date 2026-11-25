# F1 29-DoF 统一运控：行走 / 跑步 / 跳舞 —— 实验报告

日期：2026-11-24  
代码/数据/模型仓库：`yumxcode/F1-Unified-Tracker`（交付时设为 PRIVATE）  
标准：`docs/STANDARDS.md` v1.1（Tier-A A1–A12，Tier-B B1–B11）  
策略：单一统一策略（一套权重，skill one-hot + contact-phase 条件化）；最终模型 `trained_models/v5c2_15000.pt`（15k，对齐接触标签）

---

## 0. 结论（先读这一节）

**已交付并验证**

1. **数据管线**：LAFAN1（walk/run/dance）+ UMR ACCAD walk → 重定向 → **接触一致修复** → 冻结 Tier-A 门 → 50 Hz 动作库。最终 **12/14 段通过 A1–A12**（walk 6/6、run 4/4、dance 2/4），失败段记录了失败向量（§2）。
2. **参考悬浮根因已修**：`contact_repair` 原先用“抬全身”补摆动腿穿地，导致 run 的源 stance 帧支撑脚 sole z p50=4.8 cm、60% 帧 >3 cm。改为**局部修摆动腿 / 局部降 root**（`--no_post_lift`）+ 源接触 `v<0.02 且 z<0.12` 后，A12 贴地率 95.5–100%（p50≈1 mm）。新增 **A12 支撑脚贴地率 ≥95%** 门（STANDARDS v1.1，§6 有变更记录）。
3. **统一算法**：BeyondMimic 多片段跟踪 + skill one-hot + 参考 contact-phase 条件化 + 飞行接触惩罚/支撑足速度/滑步奖励 + idle 起步/停止增广 + skill-balanced 采样。24 GB 4090D 上 4096 envs 可训练。
4. **sim2sim 管线**：MuJoCo 独立回放（PhysX↔npz 关节置换、actuator 映射、armature/damping、timestep 5 ms、参考帧初始状态、entry/exit 窗口）；**从站立起步→技能→停止**；输出 B1–B11 逐门 pass/fail。
5. **最终模型 v5c2_15000（对齐标签，15k）**：全 13 段 idle 100%/tierB 全过；walk 83.3%（1 跌）；**run 25%**（run1_subject2 通过 body 0.0875，其余 3 段跌）；dance 50%。视频：`videos/*_v5c2_15000.mp4`。

**未达到 / 需继续**

- **对齐标签重训后的最好结果**：v5c2 统一 idle tierB 全过；walk 83.3%（1 跌）；run 25%（最慢段通过）；dance 50%。严格 Tier-B（B3/B5/B6/B7/B8）未过；fast run 仍跌。

- **Tier-B 仅 idle 全绿**：walk/dance 在 4 段协议无跌倒，但 B3/B5/B6/B7/B8 未过；**接触标签错位已修**（对齐后 walk B6 35.7%→18.4%、B7 1.13→0.23 m/s），旧 B6/B7 归因作废；run 仍跌倒。
- **长时程**：全 11 段（最长 162 s）walk 83.3%、dance 50%、run 0%；>60 s 漂移累积导致跌倒。
- **B8 dance 飞行门**：STANDARDS 写“walk/dance 飞行=0”，但 LAFAN dance 源本身有 8.1% 飞行；该门按现有数据必然失败，需走 §6 决定“选零飞行舞蹈窗口”或“改为与源相对误差 ≤20%”。本报告按现门记为 fail，不静默改门。
- **run 专项（未通过）**：4-run specialist 三次尝试均未稳定通过：v7（0.8s blend，10k）4/4 跌；v8（2.0s blend，5.5k）仅 run1_subject2 无跌倒（body 0.087）；v8（10k）回退到 4/4 跌。run 是当前数据/机器人下的开放限制，见 §3.3 与 §5。
- **策略是可复现基线，不是 Tier-B 全绿交付**。

---

## 1. 方法

### 1.1 统一多片段跟踪

- 参考状态：anchor-relative body poses/velocities + 全局 root 位姿（BeyondMimic compact reward）。
- 条件化：`skill_onehot`（idle/walk/run/dance）+ `contact_phase`（左右脚接触 + 飞行）。
- 奖励：`motion_body_pos/ori/lin_vel/ang_vel`、`motion_global_anchor_pos/ori`、`action_rate_l2`、`joint_limit`、`undesired_contacts`、**`flight_contact_penalty`**、**`stance_foot_velocity`**、**`foot_slippage`**。
- 采样：skill-balanced（walk/run/dance 等概率）+ 15% idle；adaptive bin 采样。
- 起步/停止：每个 clip 前后各 2 s idle + 0.8 s blend；`play_unified` 从 idle 前缀起步、在 idle 后缀停止，entry/exit 独立窗口。

### 1.2 数据与 Tier-A

- LAFAN1 14 段（walk/run/dance）GMR 重定向 + `contact_repair`；UMR ACCAD walk B1/B2（run 不合规，未入库）。
- 修复：mink 全身 IK、双脚平贴、stance 足锁定、swing 抬脚、速度 clamp、**局部地面修正**（支撑脚 >8 mm 时降 root，摆动脚 <5 mm 时只抬该脚；不再全局抬身）。
- 门：A1–A12 冻结；A12 新增（源 stance 帧支撑脚 sole z ≤2 cm ≥95%）。
- 窗口：`--scan_windows` 返回所有 ≥min_window 候选，batch 对每个 slice 跑**完整 A1–A12**，选最长全门窗口。

### 1.3 训练

- 平台：gradmotion 4090D 24 GB，IsaacLab 2.1，4096 envs，PPO（rsl_rl）。
- 主要 run：v1 8k、v2 15k、v4 6k（平台回收）、v5 5.5k（平台回收）、**v5b 15k（完成，22114 s）**。
- 账号：号池共享，按用户指导核查他人任务（id71/id72 有 `upt-*` 活跃任务，id73 空闲）；v5b 与 run specialist 在 id73 训练。

---

## 2. Tier-A 结果（10/14）

| 段 | gait | window | pass | 失败门（若 fail） |
|---|---|---|---|---|
| walk1_subject1 | walk | 2605–3708 | ✅ | — |
| walk1_subject2 | walk | 0–860 | ✅ | — |
| walk1_subject5 | walk | 3001–4860 | ✅ | — |
| walk2_subject1 | walk | 3124–4097 | ✅ | — |
| walk3_subject1 | walk | 3832–4976 | ✅ | — |
| walk4_subject1 | walk | 0–4856 | ✅ | — |
| run1_subject2 | run | 0–1546 | ✅ | — |
| run2_subject4 | run | 2499–3521 | ✅ | — |
| run1_subject5 | run | 3158–5145 | ✅ | 仅 stance deficit 抬 root 恢复（A12 96.6/97.4%） |
| run2_subject1 | run | 4202–5939 | ✅ | 仅 stance deficit 抬 root 恢复（A12 97.1/97.0%） |
| dance1_subject3 | dance | 186–1880 | ✅ | — |
| dance2_subject1 | dance | 1406–2633 | ✅ | — |
| dance1_subject1 | dance | — | ❌ | 全候选窗口 A8 腕 ROM 1.56% < 2.0% |
| dance1_subject2 | dance | — | ❌ | A12 右 94.8% < 95%（ground_rounds 8 仍差） |

A12 贴地率（通过段）：95.5–100%，p50 0.001 m。  
动作库：13 clips（6 walk / 4 run / 2 dance / 1 idle），37189 帧 @50 Hz，743.8 s。

---

## 3. sim2sim Tier-B 结果（v5b_15000）

### 3.1 四段协议（idle/walk/run/dance，站立起步→技能→停止）

| clip | 秒 | success | falls | body err (rel) | joint err | B6 mismatch | B7 slip | exit drift | tierB | 未过门 |
|---|---|---|---|---|---|---|---|---|---|---|
| idle_stand | 10.0 | ✅ | 0 | 0.0295 | 0.0878 | 0.0% | 0.001 | 0.000 | ✅ | — |
| walk1_subject1 | 42.3 | ✅ | 0 | 0.0712 | 0.1705 | 18.4% | 0.230 | 0.018 | ❌ | B3,B5,B6,B7,B8,B10 |
| run1_subject2 | 57.1 | ❌ | 1 | 0.2964 | 0.2596 | 31.4% | 0.291 | 0.156 | ❌ | B1,B2,B3,B5,B6,B7,B8,B10 |
| dance1_subject3 | 62.0 | ✅ | 0 | 0.0632 | 0.1778 | 22.0% | 0.062 | 0.012 | ❌ | B3,B5,B6,B8,B10 |

B1–B11：idle 全过；walk/dance 无跌倒但 B3/B5/B6/B8（walk 另 B7）未过；run 跌倒。  
B3 = 绝对 body error ≤35 mm；B5 = body 速度误差 ≤0.30 m/s；B6 = 接触不一致 ≤1%；B7 = 滑步 ≤0.10 m/s；B8 = 飞行占比。

### 3.2 全 11 段（最长 162 s）

| skill | n | success | fall | body err | mismatch | slip | tierB |
|---|---|---|---|---|---|---|---|
| idle | 1 | 100% | 0% | 0.0295 | 0.0% | 0.001 | ✅ |
| walk | 6 | 66.7% | 33.3% | 0.174 | 18.4% | 0.230 | ❌ |
| run | 4 | 0% | 100% | 0.638 | 31.4% | 0.291 | ❌ |
| dance | 2 | 50% | 50% | 0.391 | 22.0% | 0.062 | ❌ |

**判读**：策略在 idle 全门通过；walk/dance 短片段无跌倒但严格门未过；run 短片段跌倒；长片段（>60 s）漂移累积。B6 接触不一致在 20–36%，说明参考/机器人足底高度仍有系统差（policy 浮动或接触阈值口径），需要进一步校准。

### 3.3 run 专项尝试（4-run specialist，均未通过）

| 模型 | run1_subject2 | run1_subject5 | run2_subject1 | run2_subject4 |
|---|---|---|---|---|
| v7（0.8 s blend，10k） | 跌倒 | 跌倒 | 跌倒 | 跌倒 |
| v8（2.0 s blend，5.5k） | **无跌倒**（body 0.087） | 跌倒 | 跌倒 | 跌倒 |
| v8（2.0 s blend，10k） | 跌倒 | 跌倒 | 跌倒 | 跌倒 |

v7 5.5k 的训练 episode 平均长度仅 81/500 步（anchor_pos 终止 35.7/episode），说明 run 参考在训练中也难以跟踪；v8 在 5.5k 曾有 1/4 通过，10k 回退，训练不稳定。结论：**run 在当前 F1 29-DoF + LAFAN fast-run 数据下未学会**；不是 sim2sim 或账号问题。

### 3.5 对齐标签重训（v5c2 / v9c，2026-11-26）

用对齐标签重训：

| 模型 | idle | walk | run | dance |
|---|---|---|---|---|
| v5c2 统一（15k，对齐） | 100%（tierB ✅） | 83.3%（5/6，walk2_subject1 跌） | 25%（run1_subject2 通过 body 0.0875；其余 3 段跌） | 50%（dance1_subject3 通过） |
| v9c run specialist（10k，对齐） | — | — | 25%（run1_subject2 通过 body 0.091；其余 3 段跌） | — |

结论：标签对齐把 run 从 0/4 提到 1/4（最慢的 run1_subject2，1.45 m/s），walk 从 66.7% 提到 83.3%；但 fast run（1.64–1.9 m/s）仍跌，Tier-B 严格门（B3/B5/B6/B7/B8）仍未过。run 与严格 Tier-B 是当前数据/机器人下的开放限制。

### 3.6 run 难度分层（per-clip，v5c2 / v9c）

| clip | 根速度 mean/p95 (m/s) | 飞行占比 | 时长 | v9c fall_step | v5c2 fall_step | 结果 |
|---|---|---|---|---|---|---|
| run1_subject2 | 1.17 / 3.15 | 28.2% | 51.5 s | — | — | **通过**（body 0.091/0.088） |
| run1_subject5 | 2.07 / 4.84 | 30.3% | 66.2 s | 208 (4.2 s) | 304 (6.1 s) | 跌 |
| run2_subject1 | 1.65 / 4.63 | 18.7% | 57.9 s | 154 (3.1 s) | 201 (4.0 s) | 跌 |
| run2_subject4 | 1.24 / 3.86 | 12.7% | 34.1 s | 153 (3.1 s) | 153 (3.1 s) | 跌（entry 段） |

判读：唯一通过的是最慢的 run1_subject2；run2_subject4 虽慢（1.24）但在 idle→run 过渡后 3.1 s 跌倒，说明**入口加速**也是独立失败模式；fast run 1.65–2.07 在 3–6 s 跌。下一步 A/B 单变量：A 用 ≤1.2 m/s jog 窗口（4 段，A1–A12 全过）+ episode 20 s；B 保持 fast run + episode 20 s（只改训练）。A/B 任务 TASK_20261126_023/024 在训。

### 3.4 接触标签对齐后的复评（2026-11-25，零训练成本）

发现并修复了一个数据缺陷：`build_all_clips.py` 的 BVH 偏移对 8/12 个 clip 默认取 0（或沿用 3918），而 curated CSV 是窗口切片，导致库内 `foot_contact` 与参考动作相位错位——污染 `contact_phase` 观测与 `flight_contact_penalty`/`stance_foot_velocity`/`foot_slippage` 三项奖励，以及 B6/B7/B8 评测。修复：偏移一律取 `curation_report.json` 的 `window[0]`，并加“源 stance 帧 BVH 标签 vs FK sole-z 一致率 ≥90%”断言；重建 13-clip/4-run 库。

用同一 v5b_15000 在对齐标签上复评（**不重训**）：

| 指标 | 错位标签 | 对齐标签 |
|---|---|---|
| walk B6 mismatch | 35.7% | **18.4%** |
| walk B7 slip | 1.130 m/s | **0.230 m/s** |
| dance B6 mismatch | 19.8% | 22.0% |
| dance B7 slip | 0.095 m/s | 0.062 m/s |
| run B6 mismatch | 31.4% | 31.4%（仍跌倒） |

结论：错位标签显著虚高了 walk 的 B6/B7，旧归因（“B6 源于足底高度口径/策略浮动”、“v7 0.8 s blend 过陡”）**作废**；对齐后 walk 的 B6/B7 明显回落但仍未过 1%/0.10 门，dance 的 B6 未改善。v5b 是在错位标签上训练的，需用对齐标签重训（v5c）后再判；run v9（对齐标签）在训。B8 已按 STANDARDS §6 修订为“与源飞行占比相对误差 ≤20%”（v1.2），因为 LAFAN walk/dance 源本身有 1.4–6.3%/5.7–28.7% 飞行，旧“=0”门与数据矛盾。

---

## 4. 复现

```bash
# 数据
python tools/batch_curate.py                 # 14 clip -> repair -> A1-A12 -> windows
python tools/build_all_clips.py              # pass=true 的 clip -> npz
python tools/build_library.py --clips data/clips/*.npz --out whole_body_tracking/motions/f1_unified_library.npz

# 训练（gradmotion 24G）
gm-run F1-Unified-Tracker/whole_body_tracking/scripts/entry_f1_train.py \
  --task=Tracking-Flat-F1-v0 --motion_file motions/f1_unified_library.npz \
  --headless --num_envs 4096 --max_iterations 15000

# sim2sim（MuJoCo）
python whole_body_tracking/scripts/play_unified.py \
  --library whole_body_tracking/motions/f1_unified_library.npz \
  --checkpoint trained_models/v5b_15000.pt --clip walk1_subject1_walk \
  --headless --json_out out.json --record out.mp4

# Tier-B 汇总
python tools/eval_unified.py --library whole_body_tracking/motions/f1_unified_library.npz \
  --checkpoint trained_models/v5b_15000.pt --out_dir eval/v5b_15000_full
```

---

## 5. 限制与下一步

1. **B6 接触不一致（20–36%）**：核对机器人足底高度口径（0.03 阈值）与参考接触标签；检查 policy 是否在支撑相浮动。
2. **B3/B5**：绝对 body error >35 mm、速度误差 >0.30 m/s；需要更多训练或更紧的跟踪奖励。
3. **B10 停止漂移**：walk 0.018（4 段）/0.034（全段）已接近 5 cm；run 0.156 未过。
4. **B8 dance 飞行门**：源 dance 8.1% 飞行，需 §6 决策（零飞行窗口 vs 相对误差门）。
5. **run**：4 段 run 通过 A1–A12，但 4-run specialist 三次训练均未稳定通过（v7 10k 4/4 跌；v8 5.5k 1/4；v8 10k 4/4 跌）。下一步需换路线：更慢的 run/jog 数据、从 v5b 微调（低 LR）、更长 episode 或课程式加速；不是继续堆同一配置。
6. **长时程**：训练 episode 10 s，策略未学会 >60 s 漂移校正；需要更长 episode（20–30 s）或课程式延长。
7. **账号/额度**：共享号池 id71/id72 有他人 `upt-*` 活跃任务，id73 已用满额度（447 训练 11 min 即 status=6）；按用户指导只用空闲账号，待额度重置/新账号重试 4-run specialist。
