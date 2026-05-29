# FLUX.2 Klein 单 Ref 推理注意力系统分析与泛化验证计划（Priority 0）

> 版本：2026-05-29  
> 分支：`codex/flux2-attention-probe`  
> 目的：把现有 full-attention teacher 数据整理成一份中文、分层、可复现、可执行的系统分析报告，并补充“任务 / prompt / 分辨率 / 输入图数量”变化下的下一步机制验证计划。  
> 当前结论范围：仅覆盖 **single source + single reference**、seed=0、1248×832、28-step、prompt=`Change the background of the first image to that of the second image.` 的 FLUX.2-klein-4B 推理样本。本文明确区分：
>
> 1. 当前背景替换任务下已经被数据支持的结论；
> 2. 可能跨任务稳定的普适机制假设；
> 3. 需要下一轮实验验证的开放问题；
> 4. 每份数据/表/图对应的生成命令与输出目录。

---

## 0. 核心修正：不能把当前报告写成“FLUX.2 的一般规律”

当前数据的任务非常具体：

```text
Change the background of the first image to that of the second image.
```

也就是说，当前样本中：

- 第一张图承担 **被编辑主体 / source content / foreground preservation** 的角色；
- 第二张图承担 **background donor / reference background** 的角色；
- prompt 明确写了 `first image` 与 `second image`，所以 text tokens 不只是语义条件，还在做 **图像角色绑定**。

因此，当前 attention 分布并不能直接推广成“所有编辑任务都这样”。如果 prompt 改成：

```text
Change the background of the second image to that of the first image.
```

或者任务变成参考打光、添加/删除物体、修改衣服、添加饰品、多参考图分别提供身份/服装/背景/光照，那么 `X->R1`、`X->S`、`X->P`、`S<->R1`、`R1->X` 的强度、位置、时间窗口和 top heads 都可能变化。

本报告的正确目标是：

> 从当前背景替换样本中抽取机制候选，然后设计一组任务变化实验，区分“背景替换特异性模式”和“跨任务稳定的推理机制骨架”。

---

## 1. 原报告为什么不够系统

原有报告中有很多有效结论，但组织方式不够系统，主要问题有六个：

1. **任务特异性没有被充分标注**：当前 prompt 是背景替换，并且带明确 first/second image 角色绑定。报告里虽然描述了 source/ref，但没有把“背景替换”作为解释所有 attention 现象的前提。
2. **语义角色与物理 segment 混在一起**：`S`、`R1` 是输入槽位；但在不同 prompt 下，谁是 target content carrier、谁是 attribute donor 可能会变化。
3. **light run 与 heavy run 的角色没有被清楚分离**：light 是 scout，用来定位候选机制；heavy 是 full teacher，用来校准 light 偏差并生成 atlas。两者不能直接平均值对比，必须先做 heavy slice。
4. **“稀疏”这个词有歧义**：当前 block size=256，本质是 1D strip-level block sparsity，不是真正 object/region-level 的 2D 空间稀疏。
5. **策略结论和统计标签有潜在冲突**：atlas 里大量 `X->X` / `S->S` / `R1->R1` 因为 segment dominance 被标成 SPARSE，但这不等价于第一阶段工程实现里就应该裁掉自注意力。
6. **缺少可复现命令链**：读者看到 `priority0_systematic`、`compare_light_heavy_v2`、`priority0_sparse_atlas` 的结果，但不知道它们分别由哪些命令生成。

因此本文按如下顺序重构：

**数据与符号 → 数据生成命令 → 当前背景替换机制 → 任务特异性 → 普适机制假设 → 泛化实验矩阵 → 稀疏策略 → 机制验证计划。**

---

## 2. 数据与符号定义

### 2.1 当前数据集

当前可用的主数据来自 Priority 0 heavy_v2 full-attention teacher：

| 项 | 值 |
|---|---|
| Model | FLUX.2-klein-4B |
| Backend | NPU |
| 当前任务 | single source + single reference 背景替换 |
| Prompt | `Change the background of the first image to that of the second image.` |
| 输出尺寸 | 1248×832 |
| Seed | 0 |
| Denoising steps | 28 |
| Attention layers | 25 = 5 double-stream + 20 single-stream |
| Heads | 24 / layer |
| Head-moments | 25 × 24 × 28 = 16800 |
| Block size | 256 tokens |
| Block matrix | 50×50 |
| Records | 701 block matrices；step0/layer0 有 duplicate/prefill，已平均 |

### 2.2 物理 segment

当前序列被分成四段：

| 符号 | 范围 | 长度 | 物理含义 |
|---|---:|---:|---|
| `P` | `[0, 512)` | 512 | prompt/text tokens |
| `X` | `[512, 4568)` | 4056 | noisy target latent/image tokens |
| `S` | `[4568, 8624)` | 4056 | source image tokens |
| `R1` | `[8624, 12680)` | 4056 | reference image tokens |

本文所有 edge 都按 **query → key/value** 解释。例如：

- `X->R1`：target/noisy tokens 读取 reference tokens；
- `X->P`：target/noisy tokens 读取 prompt tokens；
- `R1->X`：reference tokens 读取 target tokens，在后续 residual 更新中可表现为 reference-to-target writeback 相关路径；
- `S->R1` / `R1->S`：source/reference conditioning tokens 之间直接互读。

### 2.3 必须新增的语义角色层

后续跨任务分析时，不能只看 `S` / `R1` 这种物理 segment。必须额外引入 **semantic role**：

| 语义角色 | 含义 | 当前背景替换任务中的对应 |
|---|---|---|
| `T` | target content carrier，被编辑对象/主体来源 | 第一张图，当前主要对应 `S` |
| `D_bg` | background donor，背景来源 | 第二张图，当前主要对应 `R1` |
| `D_light` | lighting donor，光照/氛围来源 | 未测试 |
| `D_cloth` | clothing donor，衣服/纹理来源 | 未测试 |
| `D_acc` | accessory donor，饰品来源 | 未测试 |
| `D_obj` | object donor，新增物体来源 | 未测试 |
| `R_wrong` | 当前任务中不应被读取的 reference | 多参考时定义 |

泛化验证时要同时报告两套 edge：

1. **物理 edge**：`X->S`, `X->R1`, `S->R1`；
2. **语义 edge**：`X->T`, `X->D_bg`, `X->D_light`, `X->D_cloth`, `X->R_wrong`。

只有这样才能回答：模型是否学到“背景 donor”这个语义角色，还是只偏向某个输入槽位，比如总是偏向 `R1` 或总是偏向 `S`。

---

## 3. 数据生成与后处理命令

> 这一节用于保证报告可复现。所有命令都假设从 `flux2_attention_probe/` 目录执行。实际 source/ref 图片路径没有提交到仓库，因此用环境变量占位。

### 3.1 通用环境变量

```bash
cd /home/ag/projects_anguo/diffusers/flux2_attention_probe

export SOURCE=/path/to/first_image.png
export REF1=/path/to/second_image.png
export PROMPT="Change the background of the first image to that of the second image."

export ASCEND_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD/../src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 3.2 Light_new 机制探针数据

用途：快速 scout，采样少量 layer/head/step，验证 hook 与机制方向。

输入配置：

```yaml
config: configs/probe_mechanism_light.yaml
sample_layers: [0, 4, 14, 24]
sample_heads: [0, 4, 8, 12, 16, 20]
sample_steps: [0, 14, 27]
block_size: 256
max_query_tokens_per_record: 12621  # 旧配置，下一轮应改成 all
```

运行命令：

```bash
python3 scripts/run_probe.py \
  --config configs/probe_mechanism_light.yaml \
  --source "$SOURCE" \
  --refs "$REF1" \
  --prompt "$PROMPT" \
  --num_inference_steps 28 \
  --seed 0 \
  --backend npu \
  --output_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new
```

汇总命令：

```bash
python3 scripts/summarize_probe.py \
  --input_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new \
  --config configs/probe_mechanism_light.yaml
```

主要输出：

```text
/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new/
  attention_blocks.jsonl
  attention_records.jsonl
  segment_flow_by_step_layer_head.parquet
  distribution_metrics.parquet
  segment_map.json
  probe_registration.json
  metrics_report.json
  figures/
```

注意：`light_new` 沿用了旧的 `max_query_tokens_per_record=12621`，而当前 1248×832 输入实际 token 总长为 12680。由于 `block_size=256` 下二者都对应 50 个 query blocks，因此 segment-level 结论基本不受影响；下一轮 light 必须改为 `max_query_tokens_per_record: all`。

### 3.3 Heavy_v2 full-attention teacher 数据

用途：主证据。覆盖所有 attention layers、所有 heads、所有 denoising steps，但只保存 block-level summary，不保存 full `N×N` attention tensor。

输入配置：

```yaml
config: configs/probe_mechanism_heavy.yaml
sample_layers: all
sample_heads: all
sample_steps: all
block_size: 256
max_query_tokens_per_record: all
save_full_attention: false
save_block_attention: true
```

运行命令：

```bash
python3 scripts/run_probe.py \
  --config configs/probe_mechanism_heavy.yaml \
  --source "$SOURCE" \
  --refs "$REF1" \
  --prompt "$PROMPT" \
  --num_inference_steps 28 \
  --seed 0 \
  --backend npu \
  --output_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy
```

汇总命令：

```bash
python3 scripts/summarize_probe.py \
  --input_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy \
  --config configs/probe_mechanism_heavy.yaml
```

主要输出：

```text
/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy/
  attention_blocks.jsonl
  attention_records.jsonl
  attention_shapes.jsonl
  attention_summary.parquet
  segment_flow_by_step_layer_head.parquet
  distribution_metrics.parquet
  segment_map.json
  effective_run.json
  probe_registration.json
  metrics_report.json
  figures/
```

### 3.4 Light vs Heavy 对比数据

用途：先从 heavy 中切出和 light 完全相同的 layer/head/step 子集，验证 light 统计是否可靠；再用 heavy full 分析 light 采样偏差。

运行命令：

```bash
python3 scripts/compare_light_heavy.py \
  --light_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new \
  --heavy_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy \
  --output_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy/compare_light_heavy_v2
```

主要输出：

```text
/home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy/compare_light_heavy_v2/
  compare_light_heavy_v2_report.md
  run_metadata_comparison.csv
  light_vs_heavy_slice_edge_metrics.csv
  light_vs_heavy_slice_edge_diff.csv
  key_mechanism_replication.csv
  sampling_bias_summary.csv
  heavy_full_edge_summary.csv
  heavy_full_top_layer_head.csv
  sparse_heads_top20.csv
  figures/
```

仓库内复制/保存路径：

```text
flux2_attention_probe/analysis_assets/compare_light_heavy_v2/
flux2_attention_probe/data/mechanism_single_ref_heavy/compare_light_heavy_v2/
```

### 3.5 Priority0 systematic anatomy 数据

用途：把 heavy_v2 的 full block-level attention 进一步整理成系统解剖报告，包括 step/layer/head/edge、空间 block matrix、spatial overlay、sparsity distribution 等。

当前仓库中已经提交了输出：

```text
flux2_attention_probe/reports/priority0_systematic_report.md
flux2_attention_probe/data/priority0_systematic/
```

但需要诚实说明：**当前提交中没有找到完整、独立、可直接复现 `priority0_systematic` 目录的生成脚本/命令。** 该目录显然来自 heavy_v2 的 `attention_blocks.jsonl` 与相关后处理，但精确命令需要补录。后续应把这一步固定成脚本，例如：

```bash
# TODO: 将当前 priority0_systematic 的临时后处理整理成正式脚本
python3 scripts/build_priority0_systematic.py \
  --heavy_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy \
  --light_dir /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light_new \
  --output_dir /home/ag/projects_anguo/results/attention_i2i/priority0_systematic
```

在补齐脚本前，本报告引用 `priority0_systematic` 时应视为 **已生成分析产物**，但不能视为完全可复现步骤。

### 3.6 Sparse atlas 数据

用途：把 heavy_v2 / systematic outputs 转成 sparse / dense / fallback 决策表。

当前脚本 `scripts/build_sparse_atlas.py` 使用硬编码路径：

```python
HEAVY_DIR = /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_heavy
SYSTEMATIC_DIR = /home/ag/projects_anguo/results/attention_i2i/priority0_systematic
LIGHT_DIR = /home/ag/projects_anguo/results/attention_i2i/mechanism_single_ref_light
OUT_DIR = /home/ag/projects_anguo/results/attention_i2i/priority0_sparse_atlas
```

运行命令：

```bash
python3 scripts/build_sparse_atlas.py
```

主要输出：

```text
/home/ag/projects_anguo/results/attention_i2i/priority0_sparse_atlas/
  atlas_report.md
  atlas_decision_table.csv
  sparse_candidate_edges.csv
  dense_required_edges.csv
  fallback_required_edges.csv
  density_sweep.csv
  rerun_requirements.md
  figures/
```

仓库内复制/保存路径：

```text
flux2_attention_probe/reports/priority0_sparse_atlas_report.md
flux2_attention_probe/data/priority0_sparse_atlas/
```

### 3.7 复现状态表

| 数据/报告 | 生成命令状态 | 备注 |
|---|---|---|
| `mechanism_single_ref_light_new` | 已记录 | `run_probe.py + probe_mechanism_light.yaml` |
| `mechanism_single_ref_heavy` | 已记录 | `run_probe.py + probe_mechanism_heavy.yaml` |
| `compare_light_heavy_v2` | 已记录 | `compare_light_heavy.py` |
| `priority0_systematic` | **待补录** | 输出已提交，但生成脚本/命令未完整提交 |
| `priority0_sparse_atlas` | 已记录 | `build_sparse_atlas.py`，但脚本当前使用硬编码路径，建议改成 CLI 参数 |

---

## 4. 当前背景替换任务下的已验证结论

### 4.1 全局信息流

Heavy full 的全局平均说明，自段注意仍然是主体，跨段注意是少数功能 head 的专门行为。

| Edge | Heavy full mean | 当前背景替换任务下的解释 |
|---|---:|---|
| `P->P` | 0.654 | prompt 自保持最强 |
| `S->S` | 0.585 | source 自保持 |
| `X->X` | 0.576 | target 自保持 |
| `R1->R1` | 0.568 | reference 自保持 |
| `X->P` | 0.273 | target 持续读取 prompt，负责“只改背景”和 first/second 角色绑定 |
| `R1->X` | 0.179 | reference-to-target writeback 明显 |
| `S->X` | 0.156 | source-to-target writeback 明显 |
| `X->R1` | 0.097 | target 直接读取 reference background，但全局均值不高 |
| `X->S` | 0.088 | target 读取 source，偏 source preservation |
| `S->R1` | 0.108 | source/reference 直接融合 |
| `R1->S` | 0.102 | reference/source 直接融合 |

关键解释：

1. `X->R1` 的全局均值不高，但最强 head 可以接近 0.7 到 0.8，说明 reference transfer 是 **head-specialized routing**，不是全层均匀注入。
2. `X->P` 比 `X->R1` 更高，说明 prompt 在背景替换中不仅描述任务，还持续约束“谁的背景换到谁身上”。
3. `R1->X` / `S->X` 说明 target 不只主动读取条件图；conditioning tokens 也参与向 target 的 writeback 路径。
4. `S<->R1` 说明 source 和 background reference 会直接融合。多 reference 场景下，这可能演化成 reference contamination。

### 4.2 当前任务的阶段分工

#### Early：step 0–5

主要特征：

- `X->P` 高，prompt 注入强；
- `X->S` 较高，source preservation 强；
- `X->R1` 尚未成为主导；
- double-stream 层 entropy 较低，attention 更集中。

背景替换解释：模型先建立“保留第一张图主体结构”的约束，再逐步引入第二张图背景。

#### Mid：step 10–15

主要特征：

- `X->R1` 进入稳定高值区；
- `X->P` 仍保持强约束；
- `L14/H12`、`L19/H0` 等 reference-transfer heads 活跃。

背景替换解释：这是 background donor 信息进入 target 的主要窗口。

#### Late：step 20–27

主要特征：

- `X->R1` 下降；
- `S->R1` / `R1->S` 保持或增强；
- `S->X` / `R1->X` 增强；
- prompt control 仍存在并在最后几步略有回升。

背景替换解释：late stage 不再大量 direct target-to-reference，而是转向 background/source 融合后回写 target。

### 4.3 当前任务下的关键机制 head

| 机制 | Edge | Head | 证据摘要 | 当前解释 |
|---|---|---|---|---|
| Reference transfer top | `X->R1` | `L19/H0` | heavy full top，mean≈0.689 | 第二张图背景进入 target 的强路径 |
| Reference transfer stable | `X->R1` | `L14/H12` | light 发现 + heavy 复现，rank=2 | 稳定 background transfer head |
| Double-stream alignment | `X->R1` / `X->S` | `L4/H4` | all-step rank略低但 late rank高 | source/background 对齐或区域绑定 |
| Prompt control top | `X->P` | `L12/H17` | mean≈0.989 | first/second 角色和任务指令强绑定 |
| Prompt control stable | `X->P` | `L14/H20` | light 发现 + heavy 复现 | 中期 prompt control |
| Writeback top | `R1->X`, `S->X` | `L5/H22` | `R1->X`≈0.949, `S->X`≈0.795 | reference/source 向 target 回写 |
| Fusion top | `S<->R1` | `L10/H1` | `S->R1`≈0.777, `R1->S`≈0.829 | source/background 直接融合 |
| Late fusion candidate | `R1->S` | `L24/H8` | late rank 高 | late-stage 融合/回写候选 |

---

## 5. 哪些是背景替换特异性，哪些可能更普适

### 5.1 背景替换特异性模式

以下结论很可能依赖当前 prompt 和任务：

1. **`X->S` early 高**：因为任务要求保留第一张图主体和非背景结构。如果任务是“把第二张图改成第一张图背景”，且 target 语义角色反转，则物理 `X->S` / `X->R1` 的解释会变化。
2. **`X->R1` 表示 background transfer**：当前 `R1` 刚好是 background donor。若任务改成打光或衣服，`X->R1` 可能变成 lighting transfer 或 clothing transfer，空间分布和 timestep 都会不同。
3. **`S<->R1` fusion 可能与背景融合有关**：背景替换需要把 source foreground 与 reference background 边界融合，因此 fusion/writeback 明显。删除物体或纯打光任务可能出现不同 fusion 模式。
4. **`X->P` 的高值可能部分来自 first/second 角色绑定**：当前 prompt 明确说 first image / second image。若 prompt 改成更隐式的表达，`X->P` 可能增强或削弱。

### 5.2 可能跨任务稳定的普适机制骨架

以下是假设，不是已经完全证明的定律。它们需要下一轮实验验证：

| 编号 | 普适机制假设 | 需要验证什么 |
|---|---|---|
| U1 | target 自保持始终存在 | `X->X` 在不同任务/分辨率/参考数下是否稳定主导 |
| U2 | prompt control 不只是 early bias | `X->P` 是否在背景、衣服、打光、增删改任务中都持续到 mid/late |
| U3 | attribute donor 有 direct transfer 路径 | 不同任务中是否都存在 `X->D_attr` 的强 head，只是 donor 角色不同 |
| U4 | late writeback 普遍存在 | `D_attr->X` 是否在 late window 变强，尤其背景/衣服/饰品 |
| U5 | fusion 在多条件任务中增强 | 多参考图时 `R_i<->R_j` 是否显著上升，并导致 wrong-reference contamination |
| U6 | top head 可能变，但 layer family 可能稳定 | 例如 prompt control 是否总在 L9–L15，reference/attribute transfer 是否常在 L14–L20 |
| U7 | 稀疏性依赖任务空间尺度 | 背景/打光更 diffuse，饰品/衣服局部区域更 sparse |
| U8 | role-normalized pattern 比 slot-normalized pattern 更稳定 | 语义 `X->D_bg` 是否比物理 `X->R1` 更可泛化 |

---

## 6. 不同任务下的 attention 预期变化

下面是基于当前数据提出的 **可检验预测**。这些不是结论，而是下一轮实验的 hypothesis。

### 6.1 反向背景替换

Prompt：

```text
Change the background of the second image to that of the first image.
```

关键问题：这不是简单改几个词，而是语义角色反转。需要两种设置：

1. **只改 prompt，不换输入槽位**：测试模型是否能仅凭 prompt 把 semantic target/donor 反过来；
2. **prompt + source/ref 槽位同时交换**：测试物理 slot bias 与语义 role binding 是否一致。

预期：

- 如果模型真的按语义角色工作，那么 role-normalized 的 `X->D_bg` 仍应在 mid window 上升；
- 但物理 edge 可能从 `X->R1` 转到 `X->S`，取决于谁被定义为 background donor；
- `X->P` 可能更高，因为 prompt 需要覆盖 pipeline 默认的 source/ref 先验；
- 若模型强依赖输入槽位，则即使 prompt 反转，attention 仍可能偏向原 `R1`。

### 6.2 参考打光 / 光照迁移

Prompt 示例：

```text
Change the lighting of the first image to match the second image.
```

预期：

- `X->P` 仍强，因为 lighting 是抽象属性，需要 prompt 解释“只改光照”；
- `X->D_light` 可能比背景替换更 diffuse，因为光照是全局属性，不局限于背景区域；
- `D_light->X` late writeback 可能更明显，但 block/top-k concentration 不一定高；
- `S<->R1` fusion 可能弱于背景替换，因为不需要大范围空间边界拼接。

稀疏含义：光照任务不一定适合超低密度 sparse mask，可能需要更高密度或 fallback。

### 6.3 衣服 / 饰品参考

Prompt 示例：

```text
Change the clothes of the person in the first image to match the second image.
Add the glasses from the second image to the person in the first image.
```

预期：

- `X->D_cloth` / `X->D_acc` 应该在对应局部区域增强；
- 需要 2D tile 才能判断是否真的集中在衣服/饰品区域；
- `X->S` 对 face/body identity preservation 仍应强；
- `D_attr->X` late writeback 应该存在，尤其局部纹理/形状回写；
- 多参考时 wrong-reference activation 风险较高，例如衣服 head 错读背景 ref。

稀疏含义：衣服/饰品可能比背景和光照更适合 region-level sparse，但当前 1D block resolution 不够。

### 6.4 添加物体

Prompt 示例：

```text
Add the object from the second image into the first image.
```

预期：

- `X->D_obj` 在 object insertion 区域增强；
- `X->P` 强，因为 prompt 决定“添加”而不是替换；
- `X->X` / local self-attention 可能更重要，因为需要在 target 中创造新结构；
- `D_obj->X` late writeback 应强；
- 如果没有显式位置描述，attention 可能更 diffuse。

### 6.5 删除物体

Prompt 示例：

```text
Remove the object from the first image.
```

预期：

- 如果没有 reference donor，`X->R1` 应该显著下降或不稳定；
- `X->P` 可能更强，因为删除动作由 prompt 主导；
- `X->S` 对非删除区域 preservation 很重要；
- `X->X` 和 local inpainting-like self-attention 可能增强；
- 若提供 reference background，则 `X->D_bg` / `D_bg->X` 会回升。

### 6.6 多参考图任务

Prompt 示例：

```text
Use the identity from the first reference, clothes from the second reference, and background from the third reference.
```

预期：

- `X->R_i` 应按区域/属性分工，而不是所有 refs 均匀读；
- `R_i->R_j` 会出现，需要判断是有益融合还是 contamination；
- prompt role binding 更重要，`X->P` 可能上升；
- wrong-reference activation 是核心指标：衣服区域是否错误读背景 ref，背景区域是否错误读身份 ref。

稀疏含义：多参考 sparse routing 必须是 role-aware，而不能只保留 top attention ref。

---

## 7. 下一轮实验矩阵

### 7.1 实验维度

| 维度 | 水平 | 目的 |
|---|---|---|
| Task | background, reverse background, lighting, clothing, accessory, add object, remove object | 区分任务特异性与普适机制 |
| Prompt | 明确 first/second、反向 first/second、role named、弱提示/隐式提示 | 测试 prompt role binding |
| Resolution | 512, 768, 1024, optionally 1536 | 测试分辨率一致性和 block/tile scaling |
| Input count | source+1ref, source+2refs, source+3refs | 测试多参考绑定与 contamination |
| Seed | 0,1,2 | 测试 head/edge 稳定性 |
| Tiling | 1D block256, 2D tile8×8 | 测试真实空间稀疏 |
| Image pair | 同一图对正反向、不同图对重复 | 区分图像内容偏差与机制规律 |

### 7.2 P0：背景替换 role symmetry 实验

目标：验证当前结论到底是 background-transfer 机制，还是 source/ref slot bias。

| 组 | 输入槽位 | Prompt | 目的 |
|---|---|---|---|
| BG-A | source=A, ref=B | Change background of first to second | 复现当前任务 |
| BG-B | source=A, ref=B | Change background of second to first | 只改 prompt，测 role override |
| BG-C | source=B, ref=A | Change background of first to second | 交换槽位，测 slot symmetry |
| BG-D | source=B, ref=A | Change background of second to first | prompt/slot 双反转 |

必须报告：

- physical edges：`X->S`, `X->R1`, `S->R1`, `R1->S`；
- semantic edges：`X->T`, `X->D_bg`, `D_bg->X`, `T->D_bg`；
- 输出是否真的改变了正确图片的背景；
- `X->P` 是否在 role conflict 情况下升高。

### 7.3 P1：分辨率一致性实验

```yaml
max_size: [512, 768, 1024]
seed: [0, 1, 2]
task: background A->B
max_query_tokens_per_record: all
```

指标：segment flow Pearson/Spearman、top head rank overlap、step curve correlation、tile/block top-k overlap、semantic edge stability。

### 7.4 P2：任务类型泛化实验

| 任务 | Prompt 示例 | 重点 edge |
|---|---|---|
| 背景替换 | Change background of first to second | `X->D_bg`, `D_bg->X`, `X->P` |
| 光照迁移 | Change lighting of first to match second | `X->D_light`, `D_light->X`, `X->P` |
| 衣服参考 | Change clothes of first to match second | `X->D_cloth`, `D_cloth->X`, `X->S` |
| 饰品添加 | Add glasses/accessory from second to first | `X->D_acc`, `D_acc->X`, `X->P` |
| 添加物体 | Add object from second into first | `X->D_obj`, `D_obj->X`, `X->X` |
| 删除物体 | Remove object from first | `X->P`, `X->S`, `X->X` |

每个任务至少跑：

```yaml
seeds: [0, 1, 2]
max_size: [768, 1024]
probe: expanded_light
```

对关键任务再跑 heavy / 2D tile。

### 7.5 P3：多参考绑定实验

推荐设置：

```text
source: target person/image
R1: identity or face reference
R2: clothing reference
R3: background or lighting reference
prompt: Use R1 for identity, R2 for clothing, R3 for background.
```

核心指标：

| 指标 | 定义 |
|---|---|
| correct_ref_mass | 目标区域读正确 reference 的 mass |
| wrong_ref_mass | 目标区域读错误 reference 的 mass |
| binding_ratio | correct_ref_mass / (correct_ref_mass + wrong_ref_mass) |
| inter_ref_mass | `R_i->R_j` 的总量 |
| contamination score | 错误 reference 在非对应区域的激活 |
| prompt_binding_mass | `X->P` 中与 role words 对应 token 的注意力 |

---

## 8. 稀疏策略：从“当前 atlas”升级为“任务条件化 atlas”

### 8.1 当前 atlas 的正确解读

当前 atlas 把每个 `(layer, head, edge)` 分成三类：

| 类别 | 含义 | 当前数量 | 策略 |
|---|---|---:|---|
| SPARSE | 高集中、位置较稳定，可尝试 sparse mask | 5143 / 53.6% | 候选，不等于直接上线 |
| DENSE | 分布 diffuse 或需要全局覆盖 | 4432 / 46.2% | 保持 dense |
| FALLBACK | 中等集中但 step variance 高 | 25 / 0.3% | 先 sparse，retained mass 不够则回退 dense |

必须强调：这是 **背景替换任务下的 atlas**。下一步应该生成：

```text
atlas(task, prompt_role, resolution, num_refs, seed)
```

然后分析哪些 sparse decision 是稳定的，哪些只对某个任务成立。

### 8.2 第一阶段不要做的事

1. 不要直接把所有 `SPARSE` 标签上线为 kernel mask。
2. 不要对 self-attention 做大范围裁剪。
3. 不要把背景替换中的 `X->R1` 直接当成所有任务的 reference-transfer mask。
4. 不要对 diffuse fusion head 做超低密度 sparse。
5. 不要只看 all-step mean；late-specific head 会被低估。

### 8.3 第一阶段可以做的事

| 优先级 | Target | 当前任务下的依据 | 泛化验证要求 |
|---|---|---|---|
| P0 | `L14/H12 X->R1` | light 发现 + heavy 复现 | 在 role-normalized `X->D_bg` 中是否仍强 |
| P0 | `L14/H20 X->P` | light 发现 + heavy 复现 | 不同 prompt 是否仍承担 role binding |
| P1 | `L19/H0 X->R1` | heavy full top | 背景反向/多图中是否转向正确 donor |
| P1 | `L12/H17 X->P` | prompt-control top | 是否在打光/衣服/增删改中仍强 |
| P2 | `L5/H22 R1->X/S->X` | writeback top | 不同 attribute donor 是否都有 writeback |
| P2 | `L10/H1 S<->R1` | fusion top但 diffuse | 先 ablation，不先 sparse |
| P2 | `L24/H8 R1->S` | late-specific | 仅 step-conditioned fallback |

---

## 9. 2D tile 是必要条件，不是可选优化

当前 `block_size=256` 的问题非常大：

- image token grid 是 78×52；
- 一个 block 大约覆盖 4.9 行 token × 52 列 token；
- 像素上约 78px 高 × 832px 宽；
- 它几乎横跨整张图宽度。

这意味着当前 sparse atlas 只能说明 **1D strip-level 上集中**，不能说明它精确定位到背景、衣服、脸、饰品或物体。

下一步必须做 2D tile rerun：

| 项 | 当前 | 建议 |
|---|---|---|
| block 形式 | 1D strip | 2D tile |
| block_size | 256 | 64 |
| tile shape | N/A | 8×8 tokens |
| pixel patch | 约 78×832 px | 约 128×128 px |
| segment aware | 弱 | 必须强制 segment-aware |

建议 pilot：

```yaml
tile_shape: [8, 8]
sample_steps: [0, 5, 10, 15, 20, 25, 27]
tasks: [background, lighting, clothing/accessory]
max_size: 1024
seeds: [0]
```

---

## 10. 机制验证：从统计相关走向因果验证

### 10.1 Ablation 不能只在当前任务上做

如果只在当前背景替换 prompt 上 ablate `L14/H12 X->R1`，最多只能说明它对当前 background transfer 有影响。要证明机制更普适，需要跨任务验证。

### 10.2 当前任务 ablation 分组

| 组 | 操作 | 要验证的问题 |
|---|---|---|
| A | block `L14/H12 X->R1` | light 发现的 reference-transfer 是否有因果作用 |
| B | block `L19/H0 X->R1` | heavy top reference-transfer 是否更关键 |
| C | block `L14/H20 X->P` | light prompt-control 是否影响 prompt adherence |
| D | block `L12/H17 X->P` | heavy top prompt-control 是否主控 first/second binding |
| E | block `L5/H22 R1->X/S->X` | writeback 是否是 reference 进入 target 的主路径 |
| F | block `L10/H1 S<->R1` | fusion 是否必要，是否影响背景融合边界 |
| G | A+B+E 组合 | 同时切 direct transfer 与 writeback 后 reference 是否显著消失 |

### 10.3 跨任务 ablation 分组

| 任务 | Ablation target | 预期影响 |
|---|---|---|
| 反向背景 | role-normalized `X->D_bg` top heads | 错误背景来源减少或输出失败 |
| 光照迁移 | `X->D_light`, `D_light->X`, `X->P` | 光照相似度下降，主体保持可能不变 |
| 衣服参考 | `X->D_cloth`, `D_cloth->X` | 衣服纹理/颜色迁移下降 |
| 饰品添加 | `X->D_acc`, `D_acc->X`, `X->P` | 饰品缺失或位置错误 |
| 添加物体 | `X->D_obj`, `D_obj->X`, `X->X` | 新物体形状/位置失败 |
| 删除物体 | `X->P`, local `X->X`, `X->S` | 删除失败或 source preservation 变差 |
| 多参考 | wrong-ref high heads | reference contamination 降低或正确绑定提升 |

### 10.4 评价指标

| 维度 | 指标 |
|---|---|
| 输出图像变化 | pixel diff、LPIPS、DINO/CLIP image similarity |
| prompt adherence | CLIP text-image score、人工检查是否执行正确任务 |
| source preservation | foreground/identity/layout similarity |
| attribute fidelity | 背景/衣服/光照/饰品与 donor 的相似度 |
| wrong-reference contamination | 非目标 reference 的视觉属性是否泄漏 |
| hidden deviation | per-layer hidden L2 / cosine drift |
| attention compensation | ablate 后其他 head/edge 是否 mass 上升 |
| sparse quality | retained mass、fallback rate、输出退化程度 |

---

## 11. 建议的下一轮执行顺序

### Step 1：先补背景替换 role symmetry

```yaml
tasks:
  - background_first_to_second
  - background_second_to_first
slot_orders:
  - source=A, ref=B
  - source=B, ref=A
seeds: [0, 1, 2]
max_size: 1024
probe: expanded_light
max_query_tokens_per_record: all
```

### Step 2：补分辨率一致性

```yaml
max_size: [512, 768, 1024]
seeds: [0, 1, 2]
tasks: [background_first_to_second]
probe: expanded_light
```

### Step 3：补任务类型

```yaml
tasks:
  - lighting_transfer
  - clothing_transfer
  - accessory_addition
  - object_addition
  - object_removal
seeds: [0, 1, 2]
max_size: [768, 1024]
probe: expanded_light
```

### Step 4：对关键任务做 2D tile pilot

```yaml
tasks:
  - background_first_to_second
  - lighting_transfer
  - clothing_transfer
  - accessory_addition
sample_steps: [0, 5, 10, 15, 20, 25, 27]
tile_shape: [8, 8]
seed: 0
max_size: 1024
```

### Step 5：多参考图绑定实验

```yaml
num_refs: [2, 3]
roles:
  R1: identity / face
  R2: clothing / accessory
  R3: background / lighting
seeds: [0, 1, 2]
probe: expanded_light + selected heavy/tile
```

### Step 6：最后做 causal ablation

只对满足以下条件的 head 做 ablation：

1. 在当前任务中 mass 高；
2. 跨 seed 稳定；
3. 在 role-normalized edge 中解释清楚；
4. 2D tile 或 region 指标支持其空间解释；
5. 不是明显 diffuse 的 global fusion head，或者有 fallback 策略。

---

## 12. 报告格式建议：以后每个任务都按同一模板输出

每个新任务报告都应包含以下固定表格，避免再次变成零散观察。

### 12.1 Metadata

| 字段 | 值 |
|---|---|
| task_name | |
| prompt | |
| source/ref roles | |
| resolution | |
| seed | |
| num_refs | |
| segment map | |
| probe config | |
| run command | |
| output dir | |

### 12.2 Semantic role map

| Physical segment | Semantic role | Expected contribution |
|---|---|---|
| S | T / D_x | |
| R1 | D_bg / D_light / D_cloth / ... | |
| R2 | ... | |

### 12.3 Core flow table

| Edge type | Physical edge | Semantic edge | Mean | Early | Mid | Late | Interpretation |
|---|---|---|---:|---:|---:|---:|---|

### 12.4 Top heads

| Mechanism | Semantic edge | Physical edge | Layer | Head | Mean | Peak step | Entropy | Tile concentration | Stable? |
|---|---|---|---:|---:|---:|---:|---:|---:|---|

### 12.5 Task-specific vs universal verdict

| Observation | Task-specific? | Universal candidate? | Evidence | Next test |
|---|---|---|---|---|

### 12.6 Sparse decision

| Candidate | Decision | Density | Fallback? | Why | Ablation needed? |
|---|---|---:|---|---|---|

---

## 13. 最终结论

当前数据支持的最稳妥结论是：

1. **背景替换任务下，attention 不是均匀把 reference 注入 target，而是由少数 head 执行 reference transfer、prompt control、writeback 和 fusion。**
2. **当前最强机制包括 `L19/H0 X->R1`、`L14/H12 X->R1`、`L12/H17 X->P`、`L14/H20 X->P`、`L5/H22 R1/S->X`、`L10/H1 S<->R1`。**
3. **这些 head 的功能解释强烈依赖当前 prompt：`first image` 是 target/source，`second image` 是 background donor。prompt 或任务改变后，物理 edge 可能变化。**
4. **下一步不能只继续堆当前 prompt 的 heavy run，而要系统改变任务、prompt 方向、分辨率和输入图数量。**
5. **真正可泛化的结论应该写成 semantic-role-normalized 形式，例如 `X->D_bg`、`X->D_light`、`D_attr->X`，而不是只写 `X->R1`。**
6. **Sparse routing 应从当前 atlas 升级为 task-conditioned / role-conditioned atlas。**
7. **2D tile 是必要条件，因为当前 block size=256 只能提供粗粒度 strip-level sparsity。**
8. **复现实验必须记录完整命令链；当前 light/heavy/compare/sparse-atlas 命令已补齐，`priority0_systematic` 的生成命令仍需把临时后处理固化成正式脚本。**

一句话总结：

> 当前背景替换样本已经暴露出 FLUX.2 Klein 的机制骨架：prompt 负责角色绑定和约束，target 在中期读取 attribute donor，late 阶段通过 donor/source writeback 和 fusion 完成整合。但这个骨架是否普适，必须通过反向 prompt、不同编辑任务、不同分辨率和多参考图实验来验证；后续 sparse attention 不能按物理 `R1` 写死，而应按语义角色和任务条件动态生成 routing / fallback 策略。
