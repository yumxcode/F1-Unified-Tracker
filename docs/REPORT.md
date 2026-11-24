# F1 29-DoF 统一运控：行走 / 跑步 / 跳舞 —— 实验报告

日期：2026-11-24  
代码/数据/模型仓库：`yumxcode/F1-Unified-Tracker`（交付时设为 PRIVATE）  
标准：`docs/STANDARDS.md` v1.1（Tier-A A1–A12，Tier-B B1–B11）  
策略：单一统一策略（一套权重，skill one-hot + contact-phase 条件化）；最终模型 `trained_models/v5b_15000.pt`（15k iter）

---

## 0. 结论（先读这一节）

**已交付并验证**

1. **数据管线**：LAFAN1（walk/run/dance）+ UMR ACCAD walk → 重定向 → **接触一致修复** → 冻结 Tier-A 门 → 50 Hz 动作库。最终 **12/14 段通过 A1–A12**（walk 6/6、run 4/4、dance 2/4），失败段记录了失败向量（§2）。
2. **参考悬浮根因已修**：`contact_repair` 原先用“抬全身”补摆动腿穿地，导致 run 的源 stance 帧支撑脚 sole z p50=4.8 cm、60% 帧 >3 cm。改为**局部修摆动腿 / 局部降 root**（`--no_post_lift`）+ 源接触 `v<0.02 且 z<0.12` 后，A12 贴地率 95.5–100%（p50≈1 mm）。新增 **A12 支撑脚贴地率 ≥95%** 门（STANDARDS v1.1，§6 有变更记录）。
3. **统一算法**：BeyondMimic 多片段跟踪 + skill one-hot + 参考 contact-phase 条件化 + 飞行接触惩罚/支撑足速度/滑步奖励 + idle 起步/停止增广 + skill-balanced 采样。24 GB 4090D 上 4096 envs 可训练。
4. **sim2sim 管线**：MuJoCo 独立回放（PhysX↔npz 关节置换、actuator 映射、armature/damping、timestep 5 ms、参考帧初始状态、entry/exit 窗口）；**从站立起步→技能→停止**；输出 B1–B11 逐门 pass/fail。
5. **最终模型 v5b_15000**：4 段协议（idle/walk/run/dance，站立起步→技能→停止）中 **idle 全门通过（tierB=true）**；walk/dance **无跌倒、success=true**，但 B3/B5/B6/B7/B8 未过；**run 仍跌倒**。全 11 段：idle 100%、walk 83.3%、run 0%、dance 50%。视频：`videos/*_v5b_15000.mp4`。

**未达到 / 需继续**

- **Tier-B 仅 idle 全绿**：walk/dance 在 4 段协议无跌倒，但 B3（绝对 body error >35 mm）、B5（速度误差）、B6（接触不一致 19.8–35.7% vs ≤1%）、B7（滑步 0.10–1.13 m/s）、B8（飞行）未过；run 在 57 s 全段跌倒。
- **长时程**：全 11 段（最长 162 s）walk 83.3%、dance 50%、run 0%；>60 s 漂移累积导致跌倒。
- **B8 dance 飞行门**：STANDARDS 写“walk/dance 飞行=0”，但 LAFAN dance 源本身有 8.1% 飞行；该门按现有数据必然失败，需走 §6 决定“选零飞行舞蹈窗口”或“改为与源相对误差 ≤20%”。本报告按现门记为 fail，不静默改门。
- **run 专项**：4-run specialist（TASK_20261124_447，id73）训练 11 min 后被平台额度回收（status=6）；id71/id72 有他人 `upt-*` 任务，按用户指导不使用。run 仍是未完成项，待额度重置/新账号重试。
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
| walk1_subject1 | 42.3 | ✅ | 0 | 0.0731 | 0.1705 | 35.7% | 1.130 | 0.018 | ❌ | B3,B5,B6,B7,B8 |
| run1_subject2 | 57.1 | ❌ | 1 | 0.2964 | 0.2596 | 31.4% | 0.291 | 0.156 | ❌ | B1,B2,B3,B5,B6,B7,B8,B10 |
| dance1_subject3 | 62.0 | ✅ | 0 | 0.0628 | 0.1778 | 19.8% | 0.095 | 0.012 | ❌ | B3,B5,B6,B8 |

B1–B11：idle 全过；walk/dance 无跌倒但 B3/B5/B6/B8（walk 另 B7）未过；run 跌倒。  
B3 = 绝对 body error ≤35 mm；B5 = body 速度误差 ≤0.30 m/s；B6 = 接触不一致 ≤1%；B7 = 滑步 ≤0.10 m/s；B8 = 飞行占比。

### 3.2 全 11 段（最长 162 s）

| skill | n | success | fall | body err | mismatch | slip | tierB |
|---|---|---|---|---|---|---|---|
| idle | 1 | 100% | 0% | 0.0295 | 0.0% | 0.001 | ✅ |
| walk | 6 | 83.3% | 16.7% | 0.1492 | 27.8% | 0.580 | ❌ |
| run | 2 | 0% | 100% | 0.5151 | 36.8% | 0.426 | ❌ |
| dance | 2 | 50% | 50% | 0.3881 | 33.1% | 0.317 | ❌ |

**判读**：策略在 idle 全门通过；walk/dance 短片段无跌倒但严格门未过；run 短片段跌倒；长片段（>60 s）漂移累积。B6 接触不一致在 20–36%，说明参考/机器人足底高度仍有系统差（policy 浮动或接触阈值口径），需要进一步校准。

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
5. **run 数据**：4 段 run 通过 A1–A12（run1_subject5/run2_subject1 用“仅 stance deficit 抬 root”的局部修复恢复）。
6. **长时程**：训练 episode 10 s，策略未学会 >60 s 漂移校正；需要更长 episode（20–30 s）或课程式延长。
7. **账号/额度**：共享号池 id71/id72 有他人 `upt-*` 活跃任务，id73 已用满额度（447 训练 11 min 即 status=6）；按用户指导只用空闲账号，待额度重置/新账号重试 4-run specialist。
