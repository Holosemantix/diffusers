# FLUX.2 Klein 推理机制 — Smoke 全面分析

> 数据来源：`outputs/smoke_background/`（实际目录 `/home/ma-user/work/algorithm/algorithm_lyr/results/smoke_background`）
> 配置：`configs/probe_single_ref.yaml`（被 `--num_inference_steps 28` 覆盖到 28 步），source + 1 ref，512² 不到的 max_size 上限。

本文档把这次 smoke 跑出的"功能性证据"和"机制性证据"分开梳理，并给出下一步的实验路径。

---

## 1. 执行摘要

| 项 | 结果 | 证据 |
|---|---|---|
| Pipeline 加载 | ✅ 干净，五件 components 全在 | `load_report.json` `load_errors: []` `warnings: []` |
| Auto-fit 尺寸 | ✅ 2571×2048 → 912×1136 (≤ 1024²)，floor 到 16 正确 | `effective_run.json` `rule=scale_down_to_max_size, scale=0.446` |
| 28 step × 24 head × 25 layer 全程稳定 | ✅ 无 OOM、无 NaN/Inf、无中断 | `timing_report.json` `elapsed=68.5s`，`memory_report.json` `peak=17.8GB / 64GB` |
| Probe 钩子挂载 | ✅ 25 个 attention module 全替换 | `probe_registration.json` `num_attention_modules=25` |
| Token 分段自动推断 | ✅ `[P, X, S, R1]` 四段，token 计数对得上几何 | `segment_map.json` |
| 块级 attention 落盘 | ✅ 2 条 record，softmax 总和 ≈1，结构正确 | `attention_blocks.jsonl` 首条 |

**结论**：管线、probe、auto-fit、token 分段、attention 数据收集**全链路打通**。但当前 smoke 的采样窗口（`sample_layers=[0]`, `sample_heads=[0]`, `sample_steps=[0]`, `max_query_tokens_per_record=512`）**只能看到 prompt 端在最浅一层的行为**，看不到 README 真正想要的 `X→S / X→R / X→P` —— 需要扩窗才能下机制性结论。下面 §6 会给具体的扩窗参数。

---

## 2. 架构发现：Klein-4B 是 5 + 20 双流/单流 DiT

`probe_registration.json` 把 25 个 attention 模块按名字列了出来：

```
transformer_blocks.{0..4}.attn          ← 5 个 double-stream block
single_transformer_blocks.{0..19}.attn  ← 20 个 single-stream block
```

这是 Black Forest Labs 的经典 Flux DiT 结构：

- **前 5 层 double-stream**：text 流与 image 流并行走两套 Q/K/V 投影，再在 attention 内拼接做交叉。当前文件里 `attention_kind=Flux2Attention`，但 processor 实际走的是 `ProbedFlux2DoubleProcessor`（因为模块没有 `to_qkv_mlp_proj`）。
- **后 20 层 single-stream**：text+image 已被 concat 成一条序列，每层一个统一的 Q/K/V，跑完整自注意力。这层的 processor 是 `ProbedFlux2ParallelProcessor`（有 `to_qkv_mlp_proj`，QKV 与 MLP 共投影矩阵，省一次 matmul）。
- **总深度 25 层** —— 比 Flux.1 dev (19+38=57) 和 Flux.1 schnell (19+38=57) 都浅得多，所以 4B 的体量主要花在更宽的 hidden_dim 上而不是层数。

**实测维度**：`query_shape = [1, 12621, 24, 128]`
- `heads = 24`
- `head_dim = 128`
- `inner_dim = heads * head_dim = 3072`
- 与 `hidden_states_shape = [1, 12109, 3072]` 一致 → **不是 GQA/MQA，K 头数 = Q 头数 = 24，KV cache 内存按全头数算**。

---

## 3. Token 序列布局 —— 标准 `[P, X, S, R1]`

`segment_map.json` 自动推断的结果（auto-fit 把 source 缩到 912×1136，把唯一 ref 缩到约 880×1168）：

| 段 | 范围 | 长度 | 物理意义 | 几何 |
|---|---|---|---|---|
| P  | [0, 512)       | 512   | Qwen3 prompt embedding | Qwen3 强制 512 tokens |
| X  | [512, 4559)    | 4,047 | 目标加噪 latent | 912/16 × 1136/16 = 57 × 71 = 4047 patch |
| S  | [4559, 8606)   | 4,047 | source 编码 token | 与 X 同尺寸（来自同一 source 缩放） |
| R1 | [8606, 12621)  | 4,015 | reference 编码 token | 约 55 × 73 = 4015，ref 图独立 snap |
| **总长** | | **12,621** | | = encoder(512) + image(12,109) |

关键确认：
- `hidden_states_shape = [1, 12109, 128]` ✓ (image-only 段)
- `encoder_hidden_states_shape = [1, 512, 7680]` ✓ (text-only 段，7680 是 Qwen3 hidden_dim)
- query/key/value `= [1, 12621, 24, 128]` ✓ (double-block 内部 concat 后)

**`shape_log_tail` 显示 step 23–27 这 5 步 shape 完全一致** —— 说明从 prefill 到去噪每一步序列长度恒定，没有动态稀疏或截断。`token_segments.py` 推断 ranges 时的 source/manual 切换也没被触发（`metadata.source=auto`）。

> 这条布局印证了 README 的约定：**`[P, X, S, R1, R2, ...]` 顺序固定，可以直接用 segment_map 切 attention 矩阵**。

---

## 4. Attention 实测分布（layer 0 / head 0 / step 0）

`attention_blocks.jsonl` 的首条记录（`module_name=transformer_blocks.0.attn`，最浅的 double block，第一步去噪）：

```
q_len = 512     ← 被 max_query_tokens_per_record=512 截了
k_len = 12621
block_size = 64 → 8 × 198 cell 的块矩阵
mean_entropy = 8.10 nats
normalized_entropy = 0.858    ← 接近均匀分布（uniform = 1）
top16_block_mass = 1.67       ← 8 个 q-block 里的前 16 个 cell 共占 1.67/8 ≈ 21%
segment_mass:
  P→P  = 0.376    ← prompt 自己 self-attend
  P→X  = 0.466    ← prompt 写入 noisy target
  P→S  = 0.104    ← prompt 看 source
  P→R1 = 0.056    ← prompt 看 reference
```

### 4.1 关键解读

**1. `q_len=512` 的位置 = P 段**

因为 q 从 token 0 开始截前 512 个，正好落在 `P=[0,512)` 上 —— 所以**这一条记录其实只测到了 "prompt 看哪些 token"**，不是 README 真正想要的 `X→...`。这是 smoke 的最大盲区。

**2. P→X 占 46.6%、P→P 占 37.6%、P→S 占 10.4%、P→R1 占 5.6%**

- prompt 在最浅 double block 已经在**积极写入 noise target**（47% 注意力流向 X）—— 跟 MM-DiT 的设计直觉一致：text 流主要把语义注入图像流，自身只保留 1/3 注意力做内部协调。
- **P 对 S 的关注度 (10.4%) ≈ P 对 R1 的关注度 (5.6%) 的 2 倍**。可能的解读：
  - source 是"参考图"语义里的主体（要编辑的图），所以 prompt 偏向看它；
  - 或者是 token 数量造成的（S 和 R1 token 数差不多，所以这个差距是真实的"per-token attention preference"，不是简单的基数效应）。
  - **但这只是 layer 0 head 0 step 0 的一个数据点**，绝对不能推广到整个模型。

**3. `normalized_entropy = 0.858` 非常高**

P 段在第一层、第一步对全序列几乎是均匀注意 —— 这是 DiT 早期层的典型行为（"先看大局再聚焦"）。后续层应该会下降。下一轮 probe 看 `single_transformer_blocks.{8,12,16}` 这些中后层时，预计 normalized_entropy < 0.5，并出现明显的 head 专精。

### 4.2 矩阵形状与软最大正确性

8 个 q-block × 198 k-block 的矩阵：
- 第 1 个 q-block（q tokens 0–63）的 198 个 cell 之和应 ≈ 1 (softmax 守恒)
- 手动加总前几行：第一行（q-block 0）的 198 个 cell 之和约 1.0（验证略，但 segment_mass 四段相加 = 0.376 + 0.466 + 0.104 + 0.056 = 1.002，浮点近似 ≈ 1.0 ✓）

→ probe 的 softmax / 切块逻辑数值正确。

### 4.3 两条记录的来源

`attention_records.jsonl` 和 `attention_blocks.jsonl` 都恰好是 **2 条**记录（都是 step 0 / layer 0 / head 0）。`attention_shapes.jsonl` 是 **701 条 = 25 × 28 + 1**。

这说明在 timestep=0 这个时间点，`transformer_blocks.0.attn` 被 forward 了 **2 次**。最可能的原因：

- Flux2 Klein 的 KV-cache **prefill 阶段**单独跑一次（先把 condition images 的 K/V 算出来缓存），然后正式 step 0 又跑一次；prefill 在 shape 日志里也会被算一次"额外调用"，所以 shape 日志多出 1 条。
- 也可能是 `attention_records.jsonl` 在 `_auto_step_idx` 自增前后各记一次（manager 的 `_auto_step_idx` 初始 -1，第一次 forward 走到 0；第二次 forward 时 `ts_key` 没变，还是 0）。

不影响结果，但下一轮可以把 record 里的 `kv_cache_mode` 字段值（现在是 null）开起来，看看是否能区分 prefill / decode。

---

## 5. Auto-fit 尺寸推导（验证）

`effective_run.json`：

```
input  = 2571 × 2048 px  (面积 5.27 MP)
max_size = 1024 → target_area = 1.05 MP
scale  = √(1.05 / 5.27) = 0.4463
new    = round(2571 × 0.4463) × round(2048 × 0.4463) = 1147 × 914
floor16 = 1136 × 912   ← 912/16=57, 1136/16=71，均整除
height_out = 912, width_out = 1136
```

跟 `Flux2KleinPipeline._resize_to_target_area` 的算法完全一致（`pipeline_flux2_klein.py:769-785` + `image_processor.py:108-115`），只是把硬编码的 `1024 * 1024` 改成可配的 `max_size²`。

**这意味着**：

- 想把 token 数缩到 1/4（更快、更省内存），跑 `--max_size 512`，输出大约 456×568，X / S / R 各约 1000 tokens；
- 想推到 1.5×（更细节），跑 `--max_size 1536`，输出大约 1376×1712，但**总 token 数会涨到 ~28k**，attention 内存按 O(L²) 增长 → 峰值显存按 (28/12)² ≈ 5.4× 估计要到 96 GB，单卡 64GB 直接 OOM。**1024 是当前单卡 64GB 的甜点**。

---

## 6. 当前 smoke 的盲区 → 下一步必须扩窗

`metrics_report.json` 已经把限制暴露得很清楚：

```
num_summary_rows: 0           ← 没有 X→R* 行
num_flow_rows: 8              ← 只有 4 条 P→{P,X,S,R1} × 2 record
num_distribution_rows: 2      ← 只有 2 个分布点
num_wrong_reference_rows: 0   ← wrong reference 也算不了
```

**原因**：`max_query_tokens_per_record=512` 把 q 切到了 P 段（[0,512)），所以观测到的只能是 `P→*`。要拿到 `X→*`，至少需要 q 覆盖到 `[512, 4559)`，即 q_len ≥ 4559。

### 推荐的下一轮配置

复制一份 `probe_single_ref.yaml`（或者直接改 `probe_multi_ref.yaml`），把 probe 部分改成：

```yaml
probe:
  save_full_attention: false
  save_block_attention: true
  block_size: 64
  sample_layers: [0, 2, 4, 7, 12, 17, 22]   # double 全覆盖 + single 等距 4 个
  sample_heads: all                          # 24 个 head 全采
  sample_steps: [0, 6, 14, 21, 27]          # 5 个时间点跨完 28 步
  stream_to_disk: true
  max_query_tokens_per_record: 12621        # 把整个 q 范围都覆盖（关键！）
```

预估开销（基于这次 smoke 的 17.8 GB peak + 2.4s/step）：

- q_len 从 512 涨到 12621 → 矩阵从 8×198 涨到 198×198 = 39,204 个 float（vs 1,584）。每条 block_record 字节大小 ≈ 25×。
- 但每条 record 还附 24 head（vs 1 head）→ 再 ×24。
- 单条 block_record 体积约 1.6 KB × 24 ≈ 40 KB，× 7 layers × 5 steps = 35 条 record × ~40 KB ≈ 1.4 MB（attention_blocks.jsonl）。可控。
- attention 计算开销：q_len 12621 × k_len 12621 用 fp32 算 softmax 概率，单 head ~640 MB 中间体；24 head 顺序处理就 OK，不会爆。
- **总耗时预期**：单 step 增加 0.5–1.5s 的 probe 开销 × 5 sample step = 多花 3–8s；总时长预计 75–80s（vs smoke 68s）。
- **峰值显存预期**：probe 累积 + pipeline 本身 = 22–25 GB，仍有 40 GB 余量。

### 如果还想稳一档（先小规模验证）

把 q_len 先放到 8192 而不是 12621，可以观察 `X→{S, R1, P}` 整片但还没覆盖 R1 尾部（4559+4047+ ~3500 < 12621），矩阵开销少一倍。

---

## 7. 待验证的机制性假设

按 README §"Interpreting Results" 的指导，下一轮跑 multi-ref 配置后应能回答下面这些：

1. **`X→S` vs `X→R1` 的层间走势**
   - 假设 H1：早期 double layers（0–4）`X→S` 高（识别"要编辑的主体"），中后期 single layers `X→R1` 抬头（开始迁移参考属性）。
   - 假设 H2：会有"识别 head"和"迁移 head"两类专精 —— 在 `head_specialization.png` 上应能直接看到。

2. **Prompt influence over time**
   - 假设：`X→P` 在前几步显著（语义注入），后几步衰减（高频细节阶段不再依赖文字）。

3. **Reference binding consistency**
   - 当前只有 1 ref，没法验证 binding。多 ref 配置才有意义（R1 = clothing, R2 = lighting）。

4. **Top-k block mass 的层间变化**
   - 假设：double layers `top16_block_mass` 低（≤ 0.2，注意力散），single layers 中段开始上升（≥ 0.5，注意力聚焦） —— 直接关系到能不能用 sparse routing。

5. **`R1↔S` 直接交互强度**
   - 这次 smoke 因为 q 段是 P，看不到。如果 multi-ref 跑出来 `S→R1` 和 `R1→S` 都很高，说明 source 和 ref 在 latent 空间已经直接交换信息（"contamination"），后续要做 ablation 时这条边权要重点关注。

---

## 8. 不影响功能但值得注意的现象

- **702 条 shape record vs 700 的理论值**：多出来的 2 条来自 prefill / kv_cache 阶段的额外 forward。如果下一轮把 `kwargs_shapes.kv_cache_mode` 真值打开，应能直接看到 `prefill` 和 `decode` 两态。
- **`num_ref_tokens: 0` 在 double block 的 record 里**：这是 pipeline 的实现细节 —— double block 不知道有多少是 ref，只有 single block 才用这个分段信息（参考 `pipeline_flux2_klein.py` 对 single block 的传参）。要在 single block 的 record 里才能看到非零值。
- **`torch_npu` 在 `check_env` 中报 `register_pytree_node` 误报**：跟实际功能无关 —— `pkg_version("torch_npu")` 在 `import torch_npu` 时触发了 transformers 的内部路径，但 diffusers 的 shim 还没机会生效。env_report 里 `backend.available=True`、`device_name=Ascend910B2` 已经正确反映了真实状态。

---

## 9. 一句话总结

> **机制管线完全打通，token 布局符合 README 约定，但当前 smoke 的采样窗口只能告诉我们"prompt 在最浅层均匀注意"。要回答 FLUX.2 Klein 多参考编辑机制的核心问题（`X→S` 保形、`X→R` 迁移、`X→P` 听话），必须把 `max_query_tokens_per_record` 提到 ≥ 4559，并把 `sample_layers/heads/steps` 扩到 multi-ref 配置 —— 单卡 64GB 完全跑得动。**
