# F1 29-DoF 统一运控：行走 / 跑步 / 跳舞 —— 实验报告

日期：2026-11-24  
代码/数据/模型仓库：`yumxcode/F1-Unified-Tracker`（交付时设为 PRIVATE）  
标准：`docs/STANDARDS.md` v1.1（Tier-A A1–A12，Tier-B B1–B11）  
策略：单一统一策略（一套权重，skill one-hot + contact-phase 条件化）

---

## 0. 结论（先读这一节）

**已交付并验证**

1. **数据管线**：LAFAN1（walk/run/dance）+ UMR ACCAD walk → 重定向 → **接触一致修复** → 冻结 Tier-A 门 → 50 Hz 动作库。最终 **10/14 段通过 A1–A12**（walk 6/6、run 2/4、dance 2/4），失败段记录了失败向量（见 §2）。
2. **参考悬浮根因已修**：`contact_repair` 原先用“抬全身”补摆动腿穿地，导致 run 的源 stance 帧支撑脚 sole z p50=4.8 cm、60% 帧 >3 cm。改为**局部修摆动腿 / 局部降 root**（`--no_post_lift`）+ 源接触 `v<0.02 且 z<0.12` 后，A12 贴地率 95.5–100%（p50≈1 mm）。新增 **A12 支撑脚贴地率 ≥95%** 门（STANDARDS v1.1，§6 有变更记录）。
3. **统一算法**：BeyondMimic 多片段跟踪 + skill one-hot + 参考 contact-phase 条件化 + 飞行接触惩罚/支撑足速度/滑步奖励 + idle 起步/停止增广 + skill-balanced 采样。24 GB 4090D 上 4096 envs 可训练。
4. **sim2sim 管线**：MuJoCo 独立回放（PhysX↔npz 关节置换、actuator 映射、armature/damping、timestep 5 ms、参考帧初始状态、entry/exit 窗口）；**从站立起步→技能→停止**；输出 B1–B11 逐门 pass/fail。
5. **短片段协议（4 段，10–62 s，站立起步/停止）**：v5_5500 **4/4 success、0 跌倒**，body error 0.064–0.072，joint error 0.21–0.23 rad。视频：`videos/{idle_stand,walk1_subject1_walk,run1_subject2_run,dance1_subject3_dance}_v5_5500.mp4`。

**未达到 / 需继续**

- **Tier-B 全门未通过**：短片段 B6（接触不一致 ≤1%）为 3.9–20.3%、B10（entry/exit ≥95% 且停止漂移 ≤5 cm）未过（walk 停止漂移 0.137 m）；全 11 段长片段（最长 162 s）仅 5/11 success，长时程漂移导致跌倒。策略是**可复现基线**，不是 Tier-B 全绿交付。
- **run 专项**：run specialist 在共享账号上被平台回收/额度中断两次（063/070/071 于 11:46 同批 status=6；217 在他人占用账号上按用户指导停止）。统一策略 v5_5500 的 run 短片段无跌倒，但全 57 s 片段失败。
- **B8 dance 飞行门**：STANDARDS 写“walk/dance 飞行=0”，但 LAFAN dance 源本身有 8.1% 飞行；该门按现有数据必然失败，需走 §6 决定“选零飞行舞蹈窗口”或“改为与源相对误差 ≤20%”。本报告按现门如实记为 fail，不静默改门。
- **v5b（TASK_20261124_218，15k，id73 空闲账号）**：训练中，完成后替换/补充本报告的模型列。

---

## 1. 方法

### 1.1 统一多片段跟踪

- 参考状态：anchor-relative body poses/velocities + 全局 root 位姿（BeyondMimic compact reward）。
- 条件化：`skill_onehot`（idle/walk/run/dance）+ `contact_phase`（左右脚接触 + 飞行）。
- 奖励：`motion_body_pos/ori/lin_vel/ang_vel`、`motion_global_anchor_pos/ori`、`action_rate_l2`、`joint_limit`、`undesired_contacts`、**`flight_contact_penalty`**（飞行相位足底力惩罚）、**`stance_foot_velocity`**、**`foot_slippage`**。
- 采样：skill-balanced（walk/run/dance 等概率）+ 15% idle；adaptive bin 采样。
- 起步/停止：每个 clip 前后各 2 s idle + 0.8 s blend；`play_unified` 从 idle 前缀起步、在 idle 后缀停止，entry/exit 独立窗口。

### 1.2 数据与 Tier-A

- LAFAN1 14 段（walk/run/dance）GMR 重定向 + `contact_repair`；UMR ACCAD walk B1/B2（run 不合规，未入库）。
- 修复：mink 全身 IK、双脚平贴、stance 足锁定、swing 抬脚、速度 clamp、**局部地面修正**（支撑脚 >8 mm 时降 root，摆动脚 <5 mm 时只抬该脚；不再全局抬身）。
- 门：A1–A12 冻结；A12 新增（源 stance 帧支撑脚 sole z ≤2 cm ≥95%）。
- 窗口：`--scan_windows` 返回所有 ≥min_window 候选，batch 对每个 slice 跑**完整 A1–A12**，选最长全门窗口。

### 1.3 训练

- 平台：gradmotion 4090D 24 GB，IsaacLab 2.1，4096 envs，PPO（rsl_rl）。
- 主要 run：v1 8k、v2 15k、v4 6k（平台回收）、v5 5.5k（平台回收）、v5b 15k（在训）。
- 账号：号池共享，按用户指导核查他人任务（id71/id72 有 `upt-*` 活跃任务，id73 空闲）；v5b 在 id73 训练。

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
| run1_subject5 | run | — | ❌ | 108 个 ground 违规帧，无 ≥600 全门窗口 |
| run2_subject1 | run | — | ❌ | 91 个 ground 违规帧，无 ≥600 全门窗口 |
| dance1_subject3 | dance | 186–1880 | ✅ | — |
| dance2_subject1 | dance | 1406–2633 | ✅ | — |
| dance1_subject1 | dance | — | ❌ | 全候选窗口 A8 腕 ROM 1.56% < 2.0% |
| dance1_subject2 | dance | — | ❌ | A12 右 94.8% < 95%（ground_rounds 8 仍差） |

A12 贴地率（通过段）：95.5–100%，p50 0.001 m。  
动作库：11 clips（6 walk / 2 run / 2 dance / 1 idle），30425 帧 @50 Hz，608.5 s。

---

## 3. sim2sim Tier-B 结果（v5_5500）

### 3.1 短片段协议（4 段，站立起步→技能→停止）

| clip | 秒 | success | falls | body err | joint err | B6 mismatch | B7 slip | exit drift | B10 |
|---|---|---|---|---|---|---|---|---|---|
| idle_stand | 10.0 | ✅ | 0 | 0.0637 | 0.2245 | 3.9% | 0.074 | 0.042 | ❌ entry 0% |
| walk1_subject1 | 42.3 | ✅ | 0 | 0.0715 | 0.2260 | 13.1% | 0.084 | 0.137 | ❌ exit 0% |
| run1_subject2 | 57.1 | ✅ | 0 | 0.0687 | 0.2191 | 20.3% | 0.135 | 0.098 | ❌ |
| dance1_subject3 | 62.0 | ✅ | 0 | 0.0713 | 0.2135 | 20.3% | 0.121 | 0.056 | ❌ |

B1–B11：B1/B2/B3/B4/B9/B11 通过；**B5/B6/B10 失败**；B8 见 §0。

### 3.2 全 11 段（最长 162 s）

| skill | n | success | fall | body err | mismatch | slip | tierB |
|---|---|---|---|---|---|---|---|
| idle | 1 | 100% | 0% | 0.0246 | 0.0% | 0.001 | ✅ |
| walk | 6 | 50% | 50% | 0.309 | 33.5% | 0.647 | ❌ |
| run | 2 | 0% | 100% | 0.545 | 37.0% | 0.532 | ❌ |
| dance | 2 | 50% | 50% | 0.400 | 30.7% | 0.362 | ❌ |

**判读**：策略在 10–62 s 短片段上三技能均无跌倒，但长时程（>60 s）漂移累积导致跌倒；B6 接触不一致在长片段后半段升高（run 末段 ~40%），说明参考/机器人足底高度仍有系统差（policy 浮动或接触阈值口径）。这不是“通过 Tier-B”的结果。

---

## 4. 复现

```bash
# 数据
python tools/batch_curate.py                 # 14 clip -> repair -> A1-A12 -> windows
python tools/build_all_clips.py              # pass=true 的 clip -> npz
python tools/build_library.py --clips data/clips/*.npz --out whole_body_tracking/motions/f1_unified_library.npz

# 训练（gradmotion 24G；entry 自动 merge 分片库）
gm-run F1-Unified-Tracker/whole_body_tracking/scripts/entry_f1_train.py \
  --task=Tracking-Flat-F1-v0 --motion_file motions/f1_unified_library.npz \
  --headless --num_envs 4096 --max_iterations 15000

# sim2sim（MuJoCo）
python whole_body_tracking/scripts/play_unified.py \
  --library whole_body_tracking/motions/f1_unified_library.npz \
  --checkpoint trained_models/v5_5500.pt --clip walk1_subject1_walk \
  --headless --json_out out.json --record out.mp4

# Tier-B 汇总
python tools/eval_unified.py --library whole_body_tracking/motions/f1_unified_library.npz \
  --checkpoint trained_models/v5_5500.pt --out_dir eval/v5_5500_full
```

---

## 5. 限制与下一步

1. **长时程稳定性**：训练 episode 10 s，策略未学会 >60 s 的漂移校正；需要更长 episode（20–30 s）或课程式延长。
2. **B6 接触不一致**：需要核对机器人足底高度口径（0.03 阈值）与参考接触标签；短片段 3.9–20.3%，长片段 30%+。
3. **B10 停止漂移**：walk 0.137 m；需要加强 idle 出口训练（更长 idle 后缀、停止奖励）。
4. **B8 dance 飞行门**：源 dance 8.1% 飞行，需 §6 决策（零飞行窗口 vs 相对误差门）。
5. **run 数据**：2 段 run 通过 A1–A12；run1_subject5/run2_subject1 的落地穿地需要更强的局部修（踝目标钳制/更多 ground rounds），不是降门。
6. **账号/额度**：共享号池被他人 `upt-*` 任务占用；按用户指导只用空闲账号（id73），并记录账号核查。
