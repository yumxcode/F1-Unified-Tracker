# F1 29-DoF 运动重定向与 sim2sim 验收标准（v1，冻结）

版本：v1.4  
冻结日期：2026-11-27（v1.0 冻结于 2026-11-23；v1.1 新增 A12；v1.2 修订 B8；v1.3 明确 B10 entry 操作定义；v1.4 评测器恢复，B2 主表不变）  
适用对象：Weichai F1 29-DoF（URDF `SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf`）  
适用数据：LAFAN1 walk/run/sprint/dance 的 GMR/UMR 重定向结果，以及由这些动作训练出的统一跟踪策略的 MuJoCo sim2sim 回放。

本文件是**验收门限的唯一来源**。评测前冻结；评测后只允许修数据（重定向/接触修复），不允许改门限。确需改门限时，必须先给出新的源数据推导、预注册并记入 `task_journal` decision，再重新评测；不得在看到失败值后同轮下调。

---

## 0. 为什么有两层门

| 层 | 测什么 | 何时测 | 失败含义 |
|---|---|---|---|
| **Tier A 重定向门** | 参考动作本身在 F1 运动学上是否可行（FK，无物理） | 训练前，逐 clip/窗口 | 数据不可用，策略再好也学不到接触一致的动作 |
| **Tier B sim2sim 门** | 策略在 MuJoCo 部署镜像里能否稳定跟踪 | 训练后，从站立起步到停止 | 策略不可部署/不可交付 |

两层不能互相替代：Tier A 全过不代表动态可执行；Tier B 通过也不能掩盖参考数据的滑步/穿地。

---

## 1. 度量定义（单位与口径）

所有 FK 量基于 `GMR/assets/f1/f1_mocap.xml`，足底用 4 个角点（`FOOT_CORNERS`，半径 2 mm 的角球），`sole_min_z = min(corner_z) - 0.002`。接触判定：`sole_min_z < 0.03 m`；支撑相：连续接触 ≥5 帧。

| # | 度量 | 定义 | 单位 |
|---|---|---|---|
| A1 | 穿地 | `min_t sole_min_z(t)` | m |
| A2 | 自碰撞 | 非相邻 link 对的最小距离（capsule 球近似） | m |
| A3 | 关节限位 | 超出 URDF 限位的帧数 | 个 |
| A4 | 关节速度 | `max |q̇_j| / URDF velocity limit` | 倍 |
| A5 | 根高度 | `min/max base_link z` | m |
| A6 | 连续性 | 四元数逐帧点积 >0.1；数值有限 | bool |
| A7 | 平贴率 | 支撑帧中 `|sole pitch|≤10° 且 |sole roll|≤8°` 的比例 | % |
| A8 | 手臂质量 | 任一臂关节处于限位 0.05 rad 内的帧比例；腕关节 ROM 占限位范围比例 | % |
| A9 | 站姿开度 | 双支撑帧 `ankle_gap / hip_width` 的均值与 P5；双腿 hip roll 同内收（>3°）帧比例 | — / % |
| A10 | 滑步 | 接触帧足底水平速度 `<10 cm/s` 的比例 | % |
| A11 | 双支撑占比 | 双支撑帧 / 总帧 | % |
| A12 | 支撑脚贴地率 | 源 stance 帧中支撑脚 sole z ≤0.02 m 的比例 | % |

> **A7 不是从人体数据推导的。** 人脚在支撑相天然 toe-down 20–25°（`data/source_metrics.json`：walk stance pitch mean −14…−25°，`flat_pct_10deg` 中位数 2.5%）。F1 是 20.4 cm 刚性脚板，平贴是机器人特有的接触要求，90% 目标按 F1 dance v5 的实测平贴率（90.5/93.3%）校准，属于机器人规格门，不是源数据分位数门。

---

## 2. Tier A：重定向门（冻结）

| 门 | walk | run | dance | 依据 |
|---|---|---|---|---|
| A1 穿地 | ≥ −0.015 m | ≥ −0.015 m | ≥ −0.015 m | PHUMA non-penetration 1 cm + F1 角球半径 2 mm + 3 mm 余量 |
| A2 自碰撞 | ≥ 0.005 m | ≥ 0.005 m | ≥ 0.005 m | RoboForge/GMR：自穿插必须为 0；5 mm 为 capsule 近似余量 |
| A3 关节限位 | 0 | 0 | 0 | 硬约束 |
| A4 关节速度 | ≤ 1.5× | ≤ 1.5× | ≤ 1.5× | URDF 速度限的 1.5 倍工程余量（与历史 checker 一致） |
| A5 根高度 | 0.35–1.0 m | 0.35–1.0 m | 0.35–1.0 m | F1 站立 0.588 m，下蹲/腾空带 |
| A6 连续性 | pass | pass | pass | 四元数连续、有限 |
| A7 平贴率 | ≥ 90% | ≥ 90% | ≥ 90% | F1 刚性脚板规格门（见 §1 注） |
| A8 手臂饱和 / 腕 ROM | ≤10% / ≥0.5% | ≤10% / ≥0.5% | ≤10% / ≥2% | 饱和门是主门（限位 0.05 rad 内）；腕 ROM 是防“完全冻结”的次门。dance 用腕表达动作，沿 v5 的 2%；walk/run 的腕不承担功能，0.5% 足以检出完全冻结（0%），见 §3 注 |
| A9 站姿开度 mean | ≥ 0.90 | ≥ 0.85 | ≥ 1.10 | 源数据均值下界（见 §3） |
| A9 站姿开度 P5 | ≥ 0.63 | ≥ 0.73 | ≥ 0.59 | 源数据 per-clip P5 的最小值（见 §3） |
| A9 双腿同内收 | ≤10% | ≤10% | ≤5% | F1 hip roll 不对称限位；dance 更严（交叉步语义） |
| A10 滑步 | ≥90% | ≥90% | ≥90% | PHUMA non-skating 10 cm/s；F1 刚性脚板 |
| A11 双支撑占比 | 报告 | 报告 | 报告 | 只报告，不 gate；用于判断接触检测是否被穿地污染 |
| A12 支撑脚贴地率 | ≥95% | ≥95% | ≥95% | 源 stance 帧中支撑脚 sole z ≤0.02 m 的比例；2026-11-24 预注册新增，防“抬全身补穿地”导致参考悬浮（见 §6 变更记录） |

**窗口选择规则（预注册）**：对每个 LAFAN 源 clip，先对完整修复序列跑 `tools_check_motion_f1.py --scan_windows`，取**最长零违规窗口**；最短长度 300 帧（10 s @30 fps）。只有窗口通过全部 Tier A 门且窗口覆盖率 ≥10% 才进入动作库。报告必须同时给出：
1. 全片失败向量（哪些门失败、差多少）；
2. 选中窗口的 `[start,end)`、帧数、覆盖率；
3. 窗口级 pass/fail。

不得把窗口 pass 表述为全片 pass。

---

## 3. A9 门限的源数据推导（冻结记录）

脚本：`tools/source_metrics.py` → `data/source_metrics.json`。  
源数据：LAFAN1 本地 `downloads/lafan1/`（77 段 BVH，30 fps），接触判定为 LAFAN 官方规则（逐帧足速 <0.02 m/frame）。  
髋宽：`|LeftUpLeg − RightUpLeg|` 的逐帧距离（≈0.211 m）。

| gait | 源 clip 数 | per-clip P5 最小值 | per-clip P5 中位数 | 源均值范围 | 冻结 P5 门 | 冻结 mean 门 |
|---|---|---|---|---|---|---|
| walk | 12 | **0.6239** | 0.9736 | 1.318–2.552 | **0.63** | **0.90** |
| run | 6 | **0.7075** | 1.0162 | 1.533–2.528 | **0.73** | **0.85** |
| dance | 8 | **0.5597** | 0.7351 | 1.643–2.371 | **0.59** | **1.10** |

推导规则：
- **P5 门 = 该 gait 所有源 clip 的 per-clip P5 的最小值**（允许“不差于源数据最差 clip 的 P5”）。这是源数据分位数门，不是拍脑袋常数，也不是 dance v5 的 0.70。
- **mean 门 = 源均值的下界向下取整到 0.05**（walk 1.318→0.90 留 32% 余量；run 1.533→0.85 留 45% 余量；dance 1.643→1.10 留 33% 余量）。余量用于吸收 F1 髋宽/腿长与人的形态差异。
- dance v5 报告中的 0.70 是“dance2s1 源 P5 0.53 × 1.3”的单 clip 校准；本表用 8 个 dance clip 的最小 P5 0.5597，更保守也更可复现。

**A8 腕 ROM 的 per-gait 依据**：A8 的主门是“任一臂关节处于限位 0.05 rad 内的帧比例 ≤10%”（防饱和）；腕 ROM 是次门，用于检出腕关节被 IK 完全钉死（ROM=0%）。dance v5 的腕 ROM 实测 54.5/11.5/50.3/79.6%，2% 门有大量余量；walk/run 的重定向腕 ROM 实测 0.37–1.2%，功能上由肩/肘承担摆臂，0.5% 门仍能检出 0% 冻结。该差异在评测前写入本表，属于预注册的 per-gait 规格，不是看到失败后的下调。

**冻结声明**：以上数值在 2026-11-23 依据 `data/source_metrics.json` 冻结。此前曾出现“看到 dance P5=0.5613 后把门限 0.65→0.55、看到 run flat 85.76 后把 90→85”的事后下调；该做法已撤销，相关 pass 不作数。本文件之后的任何门限变更必须走 §6 的变更流程。

---

## 4. Tier B：sim2sim 门（冻结）

部署镜像：MuJoCo（`whole_body_tracking/.../assets/f1/mjcf/f1_flat.xml`），PD 增益按 `F1_29DOF/f1_kp_kd.json`，armature 由 play 脚本注入，控制 50 Hz、物理 1 ms（decimation 4）。每个 clip 至少 3 个随机种子 × 2 次完整回放；策略从**默认站立位姿**开始，不 reset 到参考帧。

| 门 | 阈值 | 依据 |
|---|---|---|
| B1 成功率 | ≥90%（0.5 m 半径，sim2sim） | GMR/ASAP 协议；walk/run/dance 同门 |
| B2 跌倒率 | ≤5% | root z <0.55×站立高度 或 root tilt >45° 或平均 body error >0.5 m |
| B3 E_mpbpe | ≤35 mm（中位数） | GMT/UMR/ASAP “good” band |
| B4 E_mpjpe | ≤0.65 rad | 同上 |
| B5 E_vel | ≤0.30 m/s | GMT band |
| B6 接触不一致 | ≤1% 帧 | HuB contact mismatch |
| B7 滑移 | ≤100 mm/s（支撑帧均值） | HuB slip |
| B8 飞行占比 | walk/run/dance 均与源飞行占比相对误差 ≤20% | CLF-RL / 本文新增；源值见 `source_metrics.json` 与 clip `foot_contact`；2026-11-25 v1.2 修订 |
| B9 平滑 | action rate ≤1.0 rad/frame | HuB 0.51 vs 8.02 |
| B10 起步/停止 | entry ≥95%，exit ≥95%，2 s 停止保持 root 漂移 ≤5 cm；entry 操作定义见 §6 v1.3 | 契约第 6 条；HuB idle entry/exit + GMR start-frame |
| B11 分技能 | walk/run/dance 分别报告，不允许只报合并均值 | 契约第 2 条 |

**起步/停止协议（固定）**：
1. 机器人置于默认站立位姿（`f1_stand.json`），3.0 s idle；
2. 0.5 s settle：参考保持在 clip 的 idle 前缀最后一帧，re-anchor 到当前根位姿（yaw 对齐、保高）；
3. 播放参考：idle 前缀 → 0.8 s 混合 → 动作 → 0.8 s 混合到站立 → 2.0 s 停止保持；
4. 记录 entry 成功（前 3.5 s 无跌倒/接触不一致/CoM 出支撑面）、exit 成功（停止保持 2 s 无跌倒且 root 漂移 ≤5 cm）、动作段成功率；
5. 三个数分开报告；合并成功率不得掩盖 entry/exit 失败。

---

## 5. 报告模板（每个 clip 一行）

```
clip | gait | window[start,end) | frames | coverage% | A1..A12 | overall_A
     | SR_sim2sim% | falls% | E_mpbpe | E_mpjpe | E_vel | mismatch% | slip_mm/s
     | flight% (ref/policy) | act_rad/frame | entry% | exit% | drift_cm | overall_B
```

全片失败向量与窗口 pass 必须同时出现；不得只报窗口。

---

## 6. 变更流程

> **变更记录 2026-11-26（v1.3，B10 entry 操作定义）**：entry 成功 = 前 3.5 s 无跌倒 **且** 0.8 s blend 后（帧 140–175）接触不一致 ≤1% **且** CoM 仅在**双支撑相**出支撑面计失败。单支撑相 CoM 在支撑脚外属正常步态，不再计失败；blend 段参考接触正在切换，接触不一致从 blend 结束后统计。此定义不改变 B10 的 95%/5 cm 门限，只明确统计窗口与分母。子原因（跌倒/接触不一致/CoM）在评测 JSON 中分别记录。

> **变更记录 2026-11-27（v1.4，评测器恢复，门限不变）**：标准评测器曾因 `git reset --hard` 丢失，按 v1.3 定义恢复并提交（commit 5704988）。恢复口径严格对齐本文件主表：① **B2 主表不变**——root z<0.55×站立 **或** root tilt>45° 即跌倒（不放宽为“持续”）；② entry 窗口由 `idle_mask` 推导（blend_end = idle_end×1.4；原库等价帧 140–175，slow07 为 200–235），固定 140–175 在时间缩放库上会落在 blend 内造成空判；③ success = B1 anchor（anchor_pos_err_mean<0.5 m）且无跌倒；④ B3 中位数、B7 仅参考支撑帧、entry 子原因（fall/mismatch_post/com_out_ds/first_ss/first_lift）、fall_step/fall_time/fall_ref_phase 照旧。**不构成放水**：B2 主表门限未改；恢复评测器后 v5c2 walk 由旧评测器的 88.9%/11.1% 变为 75.0%/25.0%（更严），walk 专项 97.2%/2.8% 与 jog 100%/0% 在冻结口径下复现。旧评测器数字与恢复口径不可直接比较；恢复口径回归见 REPORT §3.10。

> **变更记录 2026-11-25（v1.2）**：B8 由“walk/dance 飞行=0；run 相对误差 ≤20%”改为“walk/run/dance 均与源飞行占比相对误差 ≤20%”。起因：LAFAN 源 walk 1.4–6.3%、dance 5.7–28.7% 本身有飞行相位，旧门按现数据必然 fail，且与“跟踪源动作”的目标矛盾；新门衡量策略是否复现源的空中相位占比，而非禁止飞行。A1–A12 与 B1–B7/B9–B11 不变。

> **变更记录 2026-11-24（v1.1）**：新增 A12（支撑脚贴地率，walk/run/dance 均 ≥95%）。起因：`contact_repair` 的 post-lift 为修摆动腿穿地把全身 root 抬高，run 的源 stance 帧支撑脚 sole z p50 达 4.8 cm、60% 帧 >3 cm；旧 Tier A 的 A7/A10 只在 sole z<0.03 的帧上统计，看不见该缺陷。A12 要求源 stance 帧 ≥95% 的支撑脚贴地（≤0.02 m），并据此把 repair 改为局部修摆动腿/局部降 root（`--no_post_lift`），不再全局抬身。A12 是新增门，不修改 A1–A11 的任何门限。

1. 提出变更：写明门名、旧值、新值、理由。
2. 给出源数据推导（数据集、clip 数、分位数、归一化、脚本输出路径）。
3. 在 `task_journal` 记 decision（source=self 或 drift），说明是预注册还是事后；事后变更需说明为何不构成对已评测结果的放水。
4. 重新评测所有受影响 clip，保留旧/新两套结果。
5. 更新本文件版本号与冻结日期。

没有 1–5 的门限变更一律视为无效，对应 pass 不计入验收。
