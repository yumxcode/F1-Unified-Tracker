# F1 29-DoF 统一运控：行走 / 跑步 / 跳舞 —— 实验报告

日期：2026-11-24  
代码/数据/模型仓库：`yumxcode/F1-Unified-Tracker`（交付时设为 PRIVATE）  
标准：`docs/STANDARDS.md` v1.3（Tier-A A1–A12，Tier-B B1–B11，B10 entry 操作定义见 §6）  
策略：单一统一策略（一套权重，skill one-hot + contact-phase 条件化）；最终模型 `trained_models/v5c2_15000.pt`（15k，对齐接触标签）

---

## 0. 结论（先读这一节）

**已交付并验证**

1. **数据管线**：LAFAN1（walk/run/dance）+ UMR ACCAD walk → 重定向 → **接触一致修复** → 冻结 Tier-A 门 → 50 Hz 动作库。最终 **12/14 段通过 A1–A12**（walk 6/6、run 4/4、dance 2/4），失败段记录了失败向量（§2）。
2. **参考悬浮根因已修**：`contact_repair` 原先用“抬全身”补摆动腿穿地，导致 run 的源 stance 帧支撑脚 sole z p50=4.8 cm、60% 帧 >3 cm。改为**局部修摆动腿 / 局部降 root**（`--no_post_lift`）+ 源接触 `v<0.02 且 z<0.12` 后，A12 贴地率 95.5–100%（p50≈1 mm）。新增 **A12 支撑脚贴地率 ≥95%** 门（STANDARDS v1.1，§6 有变更记录）。
3. **统一算法**：BeyondMimic 多片段跟踪 + skill one-hot + 参考 contact-phase 条件化 + 飞行接触惩罚/支撑足速度/滑步奖励 + idle 起步/停止增广 + skill-balanced 采样。24 GB 4090D 上 4096 envs 可训练。
4. **sim2sim 管线**：MuJoCo 独立回放（PhysX↔npz 关节置换、actuator 映射、armature/damping、timestep 5 ms、参考帧初始状态、entry/exit 窗口）；**从站立起步→技能→停止**；输出 B1–B11 逐门 pass/fail。
5. **最终模型 v5c2_15000（对齐标签，15k，标准口径 3 seed×2）**：idle 100%/tierB 全过（B3 中位数 28.7 mm）；walk **88.9%**（B1 接近 90%）但 B2 11.1%、B3 185 mm、B6 21.6%、B7 0.226、entry 33.3% 未过；**run 16.7%**；dance 50%。视频：`videos/*_v5c2_15000.mp4`。

**未达到 / 需继续**

- **严格 Tier-B 仍未过**：idle 全绿；walk B1 88.9% 接近，但 B2/B3/B6/B7/B8/B10 未过；run 16.7%（fast run 1.65–2.1 m/s 不可行；旧 jog_A_10000（0.8 s blend）标准口径 58.3%；**A2c（2.0 s blend）100% B1/B2/B10**，见下）；dance 50%。
- **entry 失败子原因（§3.8）**：walk/dance/run 的 entry 失败**不是**跌倒或 CoM 出支撑面，而是 **0.8 s blend 后的接触不一致 3–24%**；B6 分解显示 **拖步帧占 25–48%**（参考摆动、机器人脚 <3 cm），悬浮帧仅 1–10%。这是策略的摆动脚抬不起来/触地时序问题，不是度量误触发。
- **B8 已按 §6 v1.2 修订**（与源飞行相对误差 ≤20%），dance/run 可过、walk 仍差。
- **run 专项（jog，≤1.2 m/s）**：A2c（jog + 2.0 s blend，10k）标准口径 **success 100%（4/4，24/24 回合）、fall 0%、entry 100%、exit 100%、drift 0.0216 m**——**B1/B2/B10 通过**；B3 128 mm、B5 0.80 m/s、B6 17.8%、B7 0.111 m/s、B8 3.1 vs 7.5 仍未过。fast run 1.65–2.1 m/s 不可行。
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

## 3. sim2sim Tier-B 结果

> **口径说明（2026-11-26）**：§3.1–3.3 的数字来自旧评测实现（每 clip 单次、B3 均值、B7 全帧、B1 最大相对误差、B2 仅 root z、B10 无 CoM），只能作为预实验。**标准口径（STANDARDS v1.2）的零训练复评见 §3.7**：3 seed×2 回放、B3 中位数、B7 支撑帧、B1 0.5 m anchor、B2 三判据、B10 CoM。

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

判读：唯一通过的是最慢的 run1_subject2；run2_subject4 虽慢（1.24）但在 idle→run 过渡后 3.1 s 跌倒，说明**入口加速**也是独立失败模式；fast run 1.65–2.07 在 3–6 s 跌。

**单变量实验结果（2026-11-26）**

| 实验 | 数据 | episode | run success | entry | exit | 备注 |
|---|---|---|---|---|---|---|
| v9c（基线） | fast 4-run | 10 s | 25% | 0% | 0% | 仅最慢段 |
| A | jog ≤1.2 m/s 4 段 | 20 s | **75%**（旧口径，单次） | 50% | 75% | 标准口径 3 seed×2 复评为 **58.3%**（14/24），见 §3.7 |
| A2 | jog + 2.0 s blend | 20 s | 25%（3k 中期） | **100%** | 25% | 2.0 s blend 明显改善 entry；10k 因额度耗尽未跑完 |
| A2b | jog + 2.0 s blend（重跑） | 20 s | 25%（3k） | **100%** | 25% | 同上；账号额度耗尽 |
| B | fast 4-run | 20 s | 0/4（5.5k 中期，已停） | — | — | episode 长度不是 fast run 瓶颈 |

结论：**速度是判别变量**（jog 数据把 run 从 25% 提到 75%），2.0 s blend 改善 entry；fast run 1.65–2.07 m/s 在当前 F1 29-DoF 下仍不可行。但严格 Tier-B（B3/B5/B6/B7/B8）在 jog 上仍未过：A 的 B6 28.4%、B7 0.44 m/s、B8 飞行 15.8% vs 源 8.2%、B3 绝对误差 >35 mm；这不是继续堆同一配置能解决的，需要接触/飞行匹配奖励与更强的跟踪（下一步）。

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

### 3.7 评测口径对齐 STANDARDS v1.2（零训练复评，2026-11-26）

**标准 → 代码对照（B1–B11）**

| 门 | 标准定义 | 实现 | 统计量/分母 |
|---|---|---|---|
| B1 | 0.5 m 半径成功率 | `play_unified.py` success | `anchor_pos_err_mean ≤ 0.5` 且无跌倒；3 seed×2 回放的成功率 |
| B2 | root z<0.55×站立 或 tilt>45° 或平均 body err>0.5 | 逐步检查 | 任一触发即跌倒；按 episode 计跌倒率 |
| B3 | E_mpbpe ≤35 mm（中位数） | `body_pos_err_abs_median_m` | 每 episode 逐帧 body 误差的中位数；跨 episode 取中位数 |
| B4 | E_mpjpe ≤0.65 rad | `joint_pos_err_mean` | episode 均值 |
| B5 | E_vel ≤0.30 m/s | `body_lin_vel_err_mean_mps` | episode 均值 |
| B6 | 接触不一致 ≤1% 帧 | `contact_mismatch_pct` | 双脚逐帧 mismatch / 2 |
| B7 | 滑移 ≤100 mm/s（支撑帧均值） | `slip_stance_mean_mps` | 仅参考 stance 帧 |
| B8 | 与源飞行占比相对误差 ≤20% | `flight_robot_pct` vs `flight_ref_pct` | 跨 episode 均值 |
| B9 | action rate ≤1.0 rad/frame | `action_rate_mean` | 逐步 |
| B10 | entry/exit ≥95%，停止漂移 ≤5 cm | entry/exit + `exit_drift_m` | entry：前 3.5 s 无跌倒、mismatch≤1%、CoM 在支撑面内；exit：全程无跌倒+漂移 |
| B11 | 分技能 | `per_skill` | idle/walk/run/dance |

**v5c2 零训练复评（13 clips，3 seed×2，标准口径）**

| skill | success | fall | B3 median | B5 | B6 | B7 stance | B8 | entry/exit | tierB |
|---|---|---|---|---|---|---|---|---|---|
| idle | 100% | 0% | **28.7 mm** | 0.012 | 0.0% | 0.001 | ✅ | 100/100 | ✅ |
| walk | **88.9%** | 11.1% | 185 mm | 1.11 | 21.6% | 0.226 | ❌ (3.06 vs 4.07) | 33.3/72.2 | ❌ |
| run | 16.7% | 83.3% | 2.61 m | 1.82 | 41.3% | 0.493 | ❌ | 16.7/0 | ❌ |
| dance | 50% | 50% | 860 mm | 0.97 | 34.7% | 0.332 | ❌ | 0/50 | ❌ |

**jog 专项（A2c 10k，jog + 2.0 s blend，标准口径，4 clips × 3 seed × 2）**

| skill | success | fall | B3 median | B5 | B6 | B7 stance | B8 | entry/exit | drift |
|---|---|---|---|---|---|---|---|---|---|
| run (jog) | **100%** | **0%** | 128 mm | 0.80 | 17.8% | 0.111 | ❌ (3.1 vs 7.5) | **100/100** | **0.0216 m** |

**旧 jog 口径（jog_A_10000，0.8 s blend）****

| skill | success | fall | B3 median | B5 | B6 | B7 stance | B8 | entry/exit |
|---|---|---|---|---|---|---|---|---|
| run (jog) | **58.3%** | 41.7% | 189 mm | 1.18 | 30.6% | 0.565 | ❌ (18.7 vs 8.2) | 50/58.3 |

判读：idle 全门通过；walk 的 B1 已到 88.9%（接近 90%），但 B2 11.1%>5%、B3 185 mm、B6 21.6%、B7 0.226、entry 33.3% 仍远未过；run（含 jog）与 dance 更远。旧 B3/B7/B10 数字与归因作废（旧 B3 用均值且为相对误差，旧 B7 含非支撑帧，旧 B1/B2/B10 判据不同；旧 A 的 75% 为单次旧口径，标准口径 58.3%）。

### 3.8 entry 子原因与 B6/B7/B5 分解（v5c2，零训练，3 seed×2）

| clip | entry_old | entry_new | entry_fall | mm_post(%) | com_ds | first_ss | first_lift | float(%) | drag(%) | B7 ref/∩contact | B5 fd/cvel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| walk1_subject1 | false | false | false | 18.6 | 0 | 147 | -1 | 4.6 | 24.7 | 0.24/0.185 | 1.27/1.54 |
| dance1_subject3 | false | false | false | 22.9–24.3 | 0 | 160 | 148 | 0.7–0.9 | 34.3–34.7 | 0.059/0.050 | 0.43/0.60 |
| run1_subject2 | true | false | false | 2.9–4.3 | 0 | -1 | 158 | 2.5–10.1 | 44.8–48.2 | 0.20/0.16 | 1.08/1.36 |

**判读（关键）**：
1. **entry 失败不是 CoM/跌倒**：三个 clip 的 `entry_fall=false`、`com_ds=0`；失败全部来自 **blend 后接触不一致**（walk 18.6%、dance 23%、run 3–4%）。单支撑 CoM 假设被证伪。
2. **B6 主要是拖步**：参考摆动但机器人脚 <3 cm 的帧占 walk 24.7%、dance 34.6%、run 45–48%；悬浮帧（参考支撑、机器人 >3 cm）仅 0.7–10%。机器人**摆动脚抬不起来/触地时序偏移**是主因。
3. **B7 双分母**：仅参考支撑 vs 参考支撑∩机器人接触的滑移为 0.24/0.185（walk）、0.059/0.050（dance）、0.20/0.16（run）；两者都远超 0.10，滑移是真实策略问题（支撑脚移动），不只是分母。
4. **B5 同口径**：体原点有限差分（5 Hz 低通）与 cvel 分别为 1.27/1.54（walk）、0.43/0.60（dance）、1.08/1.36（run）；差 15–30%，但两者都 >0.30，B5 失败不是坐标语义问题。
5. **下一步**：针对摆动脚抬起（swing clearance）与触地时序的奖励/数据，而不是继续叠加支撑高度/飞行奖励（v5d 的 stance_foot_height+flight_mismatch 在 5k 反而更差：walk 61.1%、B6 29%）。严格门（B6 1%、B3 35 mm）与当前基线差 20–30×，需新的跟踪/接触建模路线。

### 3.9 策略集：v5c2（walk/dance/idle）+ jog_A2c（run）

“同时支持行走/跑步/跳舞”按**同一算法 + 策略集**交付：两个策略共享同一多片段跟踪算法、同一 167-D 观测（skill one-hot + contact-phase）、同一奖励结构与 sim2sim 管线，只是 run 用独立的 jog 速度域权重（run 的 fast 1.65–2.1 m/s 在当前 F1 29-DoF 下不可行，jog ≤1.2 m/s 可行）。

| 技能 | 策略 | B1 | B2 | B10 entry/exit/drift | B3 | B5 | B6 | B7 | B8 |
|---|---|---|---|---|---|---|---|---|---|
| idle | v5c2 | 100% | 0% | 100/100/0.0005 | 28.7 mm | 0.012 | 0% | 0.001 | ✅ |
| walk | v5c2 | 88.9% | 11.1% | 33.3/72.2/0.155 | 185 mm | 1.11 | 21.6% | 0.226 | ❌ |
| run (jog) | jog_A2c | **100%** | **0%** | **100/100/0.0216** | 128 mm | 0.80 | 17.8% | 0.111 | ❌ |
| dance | v5c2 | 50% | 50% | 0/50/0.033 | 860 mm | 0.97 | 34.7% | 0.332 | ❌ |

**结论**：run（jog）已通过 B1/B2/B10；walk 的 B1 88.9% 接近；idle 全过。walk/dance 的严格 B3–B8 与 run 的 B3–B8 仍是开放限制（差 5–30×），需要新的摆动脚抬起/触地时序奖励与更强跟踪，而不是继续堆同一配置。若要求**单一策略**同时覆盖 fast run，当前 F1 29-DoF + LAFAN fast-run 数据下不可行。

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

1. **B6 拖步（25–48%）**：机器人摆动脚抬不起来/触地时序偏移，参考摆动时脚 <3 cm；需要 swing-clearance/触地时序奖励与数据，而不是继续支撑高度奖励。
2. **B3/B5**：body 误差中位数 146–207 mm（门 35 mm）、速度误差 0.43–1.19 m/s（门 0.30）；需要更强的跟踪（参考速度/接触建模）。
3. **B10 entry**：失败来自 blend 后接触不一致（3–24%），不是跌倒/CoM；需在 entry 窗口内提升接触一致性。
4. **B8**：已按 §6 v1.2 修订；walk 仍差（robot 3.06 vs ref 4.07）。
5. **run**：jog（≤1.2 m/s，A2c 10k，2.0 s blend）标准口径 **success 100%、fall 0%、entry/exit 100%、drift 0.0216 m（B1/B2/B10 通过）**；B3 128 mm、B5 0.80、B6 17.8%、B7 0.111、B8 未过；fast run 1.65–2.1 m/s 不可行。v5d（接触/飞行奖励）已停（5k 更差）。
6. **长时程**：训练 episode 20 s；>60 s 漂移仍导致 walk/dance 部分段跌倒。
7. **账号/额度**：id82（A2c 10k）已完成；id83（v5d）已停；其余账号额度耗尽。用户已选择 provide_more_accounts（ht-7ade0e6ec8984bf8 已回执），无待决工单。
