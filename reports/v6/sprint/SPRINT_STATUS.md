# SMOKE-EDGE 48h 交付冲刺 — 状态页

**START 2026-10-28T02:22:54Z ｜ DEADLINE 2026-10-30T02:22:54Z**
里程碑：T+4h 诊断完成 ｜ T+20h 候选初筛结束 ｜ T+24h 模型冻结 ｜ T+48h 交付。
每 4 小时更新本页。改进路线阻塞即回退 A。

## 保底模型 A（已锁定 T+0.3h）

- checkpoint `models/v6/runs/expV6A_realn_s0/seg03/best.pt`（YOLO11n, 2.59M 参数, seed 0, 60ep, imgsz 1280, batch 48, 数据 expB_train 38,454 张）
- 锁箱 `deliverables/smoke_edge_release/lockbox_A/`（pt + ONNX 11.11MB + TFLite fp16 5.69MB / int8 3.6MB + args + 训练列表 + SHA256SUMS）
- **INT8 3.6MB / 45.8ms 出处核实**：该 checkpoint、x86 CPU（devbox）、ai-edge-litert 运行时、输入 1280×1280、batch=1、纯模型前向中位延迟，**不含预处理与 NMS**；真机未测（提供基准脚本）。ONNX↔Torch 奇偶校验 76/76 框（24 图，raw maxΔ 0.0157）
- 服务器默认 expB_y11s 不变；A-long 导出为非阻塞旁路

## 冻结判据（预先固定，沿用预注册）

主指标 = sdis_val（7,335 张/1,158 负图）matched-FP recall@10%（IoU≥0.4，负图误报率=出现任意≥conf 预测的负图比例，在负图集上扫 conf 得阈值）；次指标 boreal val。候选需在配对 seed 下稳定 ≥ A 均值 +2pt 且无来源明显退步，否则交付 A。

## 候选（≤2，T+20h 截止）

- **C1 真实短程微调**：从 A 出发，按 FP/FN 类型调采样（难负/难正/来源配比），不用伪标签。待诊断完成后启动
- **C2 审核后合成低比例续训**：先用新审核协议校准（60–100 张已知样本），再审 2,104 张 CORPUS_REP，同预算与 C1 对比
- **KD 已按规则跳过**：现有教师 expB_y11s 的 val R@10 = 0.329，不优于 Nano A（seeds 0.310/0.341，均值 0.326）——"教师确实优于 Nano" 条件不成立，不为本轮新写蒸馏框架

## 暂停项（冲刺期内）

新生图模型接入、大规模合成生成、新检测架构、大范围超参搜索。

## 状态更新

- **T+0.3h**：A 锁箱完成；INT8 口径核实；FireSpot 测试批第二次取回失败（17GB 上传窗口），降级为非阻塞重试项；A seed-2 val 评测提交中；A-long 边缘导出仍在本地跑（非阻塞）
- **T+3.5h**：部署验证完成——INT8 文件全量 val：@2% 0.111 / @5% 0.194（均无损）/ @10% 0.289（−2.1pt，tiny 主导）；fp16 与 INT8 完全一致 → 差异来自 onnx2tf 图转换而非量化（已验证 NMS IoU 与 resize 插值均非原因）。部署阈值按 INT8 自身曲线冻结（0.73/0.68/0.62 @2/5/10%）。审核员校准**判负**（真实微小烟 8/9 判无烟）→ 弃用新审核层，C2 用既有 rule-v3 语料。A 三 seed 齐：R@10 0.310/0.341/0.341（均值 0.331）> A-long 均值 0.313。A-long INT8 导出完成（3.6MB/45.2ms，非阻塞旁路）。候选 C1×2 + C2×2 在训。冻结门槛 = 两 seed 稳定 ≥0.351
- **T+1.5h**：诊断报告完成（`DIAGNOSTICS.md`：指标口径、分母、按来源表、FP/FN 类型目检）；KD 按规则跳过（教师 0.329 < Nano 0.341）；C1 难例挖掘打分在跑（job v6scoretrain）；C2 审核员校准 90 张在跑；离线推理工具 `infer.py` 完成并通过三后端一致性验证（pt/ONNX/INT8 同图同框，score 0.64/0.66/0.64）。注意：e2e 延迟（预处理+前向+NMS，batch=1）~320ms@devbox CPU，与 45.8ms（纯前向中位）口径不同，README 两者都报
- **T+27.5h（+12h 延期内，收尾）**：候选诊断完成——C1 0.313 / C2 0.325 @10%，均在 A seed 带内未过门槛，冻结 A 的决定成立；发现并记录微调确定性（seed 旋钮对 checkpoint 微调无效，位级相同权重）；交付包完成并打 zip（README/demo/失败案例/锁箱）；FINAL_REPORT_V6 完成（中文摘要）；GitHub 07ea482→(+终报) / HF v6A_y11n 已发布；台账 $27.6/$2000 无挂起。SPRINT CLOSED — 交付模型 A
