# FLUX.2 Klein 单 Ref 推理注意力系统分析（Priority 0）

> 版本：2026-05-28  
> 分支：`codex/flux2-attention-probe`  
> 目的：把现有 full-attention teacher 数据重新整理成一份中文、分层、可执行的系统分析报告。  
> 结论范围：仅覆盖当前 **single source + single reference**、seed=0、1248×832、28-step 的 FLUX.2-klein-4B 推理样本。它可以指导下一轮 sparse routing / ablation 设计，但不能直接等价于最终 sparse inference 结论。

---

## 0. 为什么需要重写这份报告

原有报告中有很多有效结论，但组织方式不够系统，主要问题有四个：

1. **叙事层级混在一起**：机制发现、工程 sanity check、light/heavy 对比、sparse atlas、block-size 局限、下一步实验建议混在同一条时间线里，读者很难判断哪些是“已验证结论”，哪些只是“下一步假设”。
2. **light run 与 heavy run 的角色没有被清楚分离**：light 是 scout，用来定位候选机制；heavy 是 full teacher，用来校准 light 偏差并生成 atlas。两者不能直接平均值对比，必须先做 heavy slice。
3. **“稀疏”这个词有歧义**：当前 block size=256，本质是 1D strip-level block sparsity，不是真正 object/region-level 的 2D 空间稀疏。报告里虽然提到了这个限制，但没有提前作为解释前提。
4. **策略结论和统计标签有潜在冲突**：atlas 里大量 `X->X` / `S->S` / `R1->R1` 因为 segment dominance 被标成 SPARSE，但策略层又建议 self-attention 先保持 dense。正确理解应该是：统计分类说明它们“集中”，不等于第一阶段工程实现里就应该裁掉自注意力。

因此本文按如下顺序重构：

**数据与符号 → 数据可信度 → 全局信息流 → 机制分工 → 稀疏性来源 → 稀疏策略 → 局限 → 下一步实验。**

---

## 1. 数据与符号定义

### 1.1 当前数据集

当前可用的主数据来自 Priority 0 heavy_v2 full-attention teacher：

| 项 | 值 |
|---|---|
| Model | FLUX.2-klein-4B |
| Backend | NPU |
| 输入 | single source + single reference |
| Prompt | `Change the background of the first image to that of the second image.` |
| 输出尺寸 | 1248×832 |
| Seed | 0 |
| Denoising steps | 28 |
| Attention layers | 25 = 5 double-stream + 20 single-stream |
| Heads | 24 / layer |
| Head-moments | 25 × 24 × 28 = 16800 |
| Block size | 256 tokens |
| Block matrix | 50×50 |
| Records | 701 block matrices；step0/layer0 有一次 duplicate/prefill，已平均 |

### 1.2 Token segment

序列被分成四段：

| 符号 | 范围 | 长度 | 含义 |
|---|---:|---:|---|
| `P` | `[0, 512)` | 512 | prompt/text tokens |
| `X` | `[512, 4568)` | 4056 | noisy target latent/image tokens |
| `S` | `[4568, 8624)` | 4056 | source image tokens |
| `R1` | `[8624, 12680)` | 4056 | reference image tokens |

本文所有 edge 都按 **query → key/value** 解释。例如：

- `X->R1`：target/noisy tokens 主动读取 reference tokens；这是直接 reference transfer。
- `X->P`：target/noisy tokens 读取 prompt tokens；这是 prompt control。
- `R1->X`：reference tokens 读取 target tokens；在推理更新里表现为 reference-to-target writeback 相关路径。
- `S->R1` / `R1->S`：source/reference conditioning tokens 之间直接融合。

---

## 2. 数据可信度与 light/heavy 关系

### 2.1 Heavy v2 是当前主证据

Heavy v2 的意义是：在同一输入、同一 seed、同一分辨率下，覆盖所有层、所有 head、所有 timestep。它是当前机制判断和 sparse atlas 的主依据。

### 2.2 Light run 是 scout，不是完整图谱

Light_new 只采样：

```yaml
sample_layers: [0, 4, 14, 24]
sample_heads:  [0, 4, 8, 12, 16, 20]
sample_steps:  [0, 14, 27]
```

它的作用是快速定位明显机制，如 `L14/H12 X->R1` 和 `L14/H20 X->P`，但它天然会漏掉未采样 layer/head，例如 `L19/H0`、`L12/H17`、`L5/H22`、`L10/H1`。

### 2.3 Light 与 heavy slice 的工程一致性很好

为了验证 light 不是 hook 或统计错误，必须从 heavy_v2 里切出完全相同的 layer/head/step 子集，再和 light_new 对齐比较。这个 heavy slice 与 light_new 几乎完全一致：核心 edge 的 mean abs diff 约为 0 到 0.0003，Pearson 接近 1.0。

结论：

- light_new 的统计管线可信；
- light_new 的问题不是数值错误，而是采样覆盖不足；
- heavy_v2 才能回答“完整 sparse routing 应该保留哪些 head/edge”。

### 2.4 q_len drift 需要修正，但对当前 block-level 结论影响很小

Light_new 元数据里 `q_len=12621`，heavy_v2 是 `12680`。这是旧 light 配置留下的 `max_query_tokens_per_record` 漂移。由于当前 `block_size=256` 时二者都解析成 50 个 query blocks，所以 segment-level mass 差异可以忽略。但下一轮必须统一使用：

```yaml
max_query_tokens_per_record: all
```

---

## 3. 全局信息流：target 不是简单地“均匀读 reference”

Heavy full 的全局平均说明，自段注意仍然是主体，跨段注意是少数功能 head 的专门行为。

| Edge | Heavy full mean | 机制解释 |
|---|---:|---|
| `P->P` | 0.654 | prompt 自保持最强 |
| `S->S` | 0.585 | source 自保持 |
| `X->X` | 0.576 | target 自保持 |
| `R1->R1` | 0.568 | reference 自保持 |
| `X->P` | 0.273 | target 持续读取 prompt，是最强 target 外部条件路径 |
| `R1->X` | 0.179 | reference-to-target writeback 明显 |
| `S->X` | 0.156 | source-to-target writeback 明显 |
| `X->R1` | 0.097 | target 直接读取 reference，但全局均值不高 |
| `X->S` | 0.088 | target 读取 source，偏 early/source preservation |
| `S->R1` | 0.108 | source/reference 直接融合 |
| `R1->S` | 0.102 | reference/source 直接融合 |

关键解释：

1. `X->R1` 的全局均值并不高，但最强 head 可以接近 0.7 到 0.8。这说明 reference transfer 是 **head-specialized routing**，不是全层均匀注入。
2. `X->P` 比 `X->R1` 更高，说明 prompt 在编辑过程中不是 early-only，而是贯穿中后期。
3. `R1->X` / `S->X` 说明 target 不只主动读取条件图；条件图 token 也参与向 target 的回写路径。
4. `S<->R1` 说明 source/reference 在 conditioning token 内部会直接融合。多 reference 场景下，这可能演化成 reference contamination，需要重点验证 `R_i<->R_j`。

---

## 4. Timestep 分工：early / mid / late 三阶段

从 heavy step 曲线看，推理过程可以分成三个阶段。

### 4.1 Early：step 0–5

主要特征：

- `X->P` 最高，prompt 注入强；
- `X->S` 较高，source preservation 强；
- `X->R1` 尚未成为主导；
- double-stream 层的 entropy 较低，attention 更集中。

这阶段更像是在建立 target 的语义方向和 source 结构约束。

### 4.2 Mid：step 10–15

主要特征：

- `X->R1` 进入稳定高值区；
- `X->P` 仍然保持强约束；
- `L14/H12`、`L19/H0` 这类 reference-transfer head 活跃；
- 部分 prompt-control head 仍高度集中。

这阶段是 reference transfer 最清楚的窗口。

### 4.3 Late：step 20–27

主要特征：

- `X->R1` 下降；
- `S->R1` / `R1->S` 上升或保持；
- `S->X` / `R1->X` 明显增强；
- prompt control 仍然存在，并在最后几步略有回升。

这说明 late stage 不是继续大量 direct target-to-reference，而是转向 **conditioning fusion + writeback**。

稀疏策略含义：

- `X->R1` mask 应重点覆盖 mid window；
- `S/R1->X` writeback mask 应覆盖 late window；
- `S<->R1` fusion 不能只看 all-step 平均，因为某些 head late-rank 很高但 all-step rank 不高；
- prompt control 不能只在 early 保留，late 也要保留关键路径。

---

## 5. Layer 分工：几类机制分别落在不同层段

### 5.1 Double-stream layer 0–4

特点：

- early denoising 更稀疏；
- `L4/H4` 是 light 发现的强 `X->R1` / `X->S` head；
- source preservation 与 early reference alignment 更明显；
- 但不是所有 double-stream head 都可 sparse，`L4/H1` 等组合被 dense-required 表标为必须 dense。

### 5.2 Single-stream early layer 5–8

特点：

- heavy full 发现 `L5/H22` 是最强 writeback head；
- `R1->X` 峰值极高，`S->X` 也很强；
- light 原始采样没有覆盖 layer 5，因此之前低估了 early writeback。

### 5.3 Single-stream middle layer 9–15

特点：

- `X->P` prompt control 在 layer 9–10 和 layer 12 附近极强；
- `L12/H17 X->P` 是全量最强 prompt-control head；
- `L14/H12 X->R1` 是 light 发现并被 heavy 复现的稳定 reference-transfer head；
- `L10/H1 S<->R1` 是最强 fusion head。

### 5.4 Single-stream mid-late layer 16–20

特点：

- `X->R1` reference transfer 在 layer 16–19 达到最强区域；
- `L19/H0 X->R1` 是 heavy full 的 top reference-transfer head；
- 这是 light 原采样没有覆盖到的关键盲区。

### 5.5 Late layer 21–24

特点：

- layer 23–24 的 `S/R1->X` writeback 很强；
- layer 24 的 `S<->R1` late fusion 明显；
- 但 late layer 的 entropy 普遍更高，attention 更像全局融合，不适合简单 top-k 一刀切。

策略含义：late layer 不应默认强稀疏，除非后续 2D tile 和 ablation 证明可以保真。

---

## 6. 机制目录：每个机制的证据与用途

### 6.1 Prompt control：`X->P`

代表 head：

| Head | 角色 | 证据 |
|---|---|---:|
| `L12/H17` | heavy full top prompt-control | mean ≈ 0.989，peak segment mass ≈ 99.7% |
| `L14/H20` | light 发现且 heavy 复现 | heavy full rank=10，light value ≈ 0.886 |
| `L24/H20` | late prompt-control 候选 | late 存在但 all-step rank 不高 |

解释：prompt 不是只负责 early semantic direction，而是在中后期维持“只改背景”等编辑约束。`X->P` 的稀疏性非常强，尤其 `L12/H17` 甚至表现为接近单 block 控制。

策略：

- `L12/H17 X->P` 是第一批 sparse mask / ablation 的 P1 目标；
- `L14/H20 X->P` 是 P0 目标，因为它由 light 发现且 heavy 复现；
- 不建议完全移除 prompt path，否则容易破坏 prompt adherence。

### 6.2 Reference transfer：`X->R1`

代表 head：

| Head | 角色 | 证据 |
|---|---|---:|
| `L19/H0` | heavy full top reference-transfer | mean ≈ 0.689 |
| `L14/H12` | light 发现且 heavy 复现 | rank=2，mean ≈ 0.677 |
| `L4/H4` | double-stream reference/source alignment | all-step rank=11，late rank=7 |

解释：target 直接读取 reference 的确存在，但集中在少数 head。它不是所有层都均匀发生。

策略：

- `L14/H12`：P0 ablation，因为最稳定、可由 light/heavy 双重验证；
- `L19/H0`：P1 ablation，因为是 heavy full top，但 light 原本没采到；
- `L4/H4`：不能忽略，它可能连接 source preservation 与 reference alignment。

### 6.3 Source preservation：`X->S`

代表现象：

- step0 的 `X->S` 高于中后期；
- top layers 包含 L4、L1、L2、L5 等；
- `L4/H4` 同时强 `X->S` 和 `X->R1`。

解释：source preservation 更偏 early，它可能先锁定 source 结构，再逐步让 reference 信息进入 target。

策略：

- 不要在 early window 粗暴裁掉 `X->S`；
- 如果做 sparse，应保留 `X->S` 的 early source-preservation heads；
- ablation 时应把 `X->S` 与 `X->R1` 区分，否则可能把 source preservation 误判成 reference transfer。

### 6.4 Writeback：`S/R1 -> X`

代表 head：

| Head | Edge | 角色 | 证据 |
|---|---|---|---:|
| `L5/H22` | `R1->X`, `S->X` | early writeback top | `R1->X` mean ≈ 0.949，`S->X` mean ≈ 0.795 |
| `L24/H20` | `R1->X` | late writeback | light 发现，但 heavy full rank 不是 top |
| `L24/H6/H15/H21` | `R1->X` | late writeback family | heavy top list 中多次出现 |

解释：reference influence 不只通过 `X->R1` 进入 target。`R1->X` 和 `S->X` 是另一条重要路径，尤其 late stage。只 ablate `X->R1` 不能证明 reference 已被完全切断。

策略：

- `L5/H22 R1->X/S->X` 必须进入 P2 或 P1.5 ablation；
- late `L24` writeback 可以先 fallback，不要直接强稀疏；
- 如果做 causal ablation，至少要比较三组：只 block `X->R1`、只 block `R1->X`、同时 block 二者。

### 6.5 Source/reference fusion：`S<->R1`

代表 head：

| Head | Edge | 角色 | 证据 |
|---|---|---|---:|
| `L10/H1` | `S->R1`, `R1->S` | heavy full top fusion | `S->R1` mean ≈ 0.777，`R1->S` mean ≈ 0.829 |
| `L24/H8` | `R1->S` late | light 发现的 late fusion candidate | late-rank 较高，但 all-step rank 低 |

解释：source 和 reference 会在 conditioning token 内互相融合。这是后续多 reference contamination 的风险来源。

策略：

- 当前不建议直接 sparsify fusion edge；
- fusion 的 block 分布更 diffuse，当前 1D block resolution 也不足以解析空间区域；
- 多 reference 实验必须重点看 `R_i->R_j` 和 wrong-reference activation。

### 6.6 Self-attention：`X->X` / `S->S` / `R1->R1` / `P->P`

自段注意占全局质量的主要部分，很多 self edge 在 atlas 中被标成 SPARSE，因为它们 segment-dominant、集中度高。

但策略层要保守：

- 自注意力承担基本结构维持；
- 它的“集中”不等价于“可无损裁剪”；
- 第一阶段不要把 self-attention 当主要 sparse target；
- 可以后续单独做 self-attention local-window / diagonal-band ablation。

---

## 7. 稀疏性来源：哪些是真正适合 sparse routing 的信号

当前数据说明，稀疏性主要来自三类 head：

1. **自段 dominant head**：如 `X->X` / `S->S` / `R1->R1`，segment mass 很高，但不一定适合第一阶段裁剪。
2. **功能专精 cross-segment head**：如 `L19/H0 X->R1`、`L12/H17 X->P`、`L5/H22 R1->X`。这是最有价值的 sparse routing 目标。
3. **step-specific late head**：all-step 平均可能不高，但在 late window 很重要，如 `L24/H8 R1->S`。这类应使用 fallback 或 step-conditioned policy。

稀疏指标不能单独使用。一个 head 是否适合 sparse，应同时看：

| 指标 | 作用 | 风险 |
|---|---|---|
| segment mass | 判断该 edge 是否真的重要 | 高 mass 可能 diffuse，未必可 sparse |
| entropy | 判断分布是否集中 | 低 entropy 可能只是 self-attention |
| top-k block mass | 判断少量 block 是否覆盖大部分 attention | 当前 block 是 1D strip，空间解释粗 |
| step stability | 判断 mask 是否能跨 step 复用 | late-specific head 会被 all-step 平均低估 |
| mechanism role | 判断裁剪后会破坏什么能力 | 必须结合 prompt/reference/source 语义 |

---

## 8. Sparse atlas 的正确解读

Atlas 把每个 `(layer, head, edge)` 分成三类：

| 类别 | 含义 | 当前数量 | 策略 |
|---|---|---:|---|
| SPARSE | 高集中、位置较稳定，可尝试 sparse mask | 5143 / 53.6% | 候选，不等于直接上线 |
| DENSE | 分布 diffuse 或需要全局覆盖 | 4432 / 46.2% | 保持 dense |
| FALLBACK | 中等集中但 step variance 高 | 25 / 0.3% | 先 sparse，retained mass 不够则回退 dense |

建议把 atlas 当成 **teacher-derived decision table**，而不是最终 kernel 配置。实际推理策略应分三层：

1. **Mechanism whitelist**：优先只对已理解的机制 head 做 sparse。
2. **Runtime retained-mass check**：如果保留 mass 低于阈值，回退 dense。
3. **Ablation confirmation**：只有通过 causal ablation 的 head 才能进入真实 sparse inference。

---

## 9. 当前最合理的稀疏策略草案

### 9.1 第一阶段：只测试高置信 cross-segment head

| 优先级 | Target | 建议 |
|---|---|---|
| P0 | `L14/H12 X->R1` | light 发现 + heavy 复现；测试 1–3% density |
| P0 | `L14/H20 X->P` | light 发现 + heavy 复现；测试 0.5–1% density |
| P1 | `L19/H0 X->R1` | heavy full top；测试 1–3% density |
| P1 | `L12/H17 X->P` | prompt-control top；可测试 0.5% density |
| P2 | `L5/H22 R1->X/S->X` | writeback top；测试 1–3%，并与 X->R1 ablation 组合 |
| P2 | `L10/H1 S<->R1` | fusion top，但 diffuse；先 ablation，不先 sparse |
| P2 | `L24/H8 R1->S` | late-specific；仅 step-conditioned fallback |

### 9.2 第一阶段不要做的事

1. 不要直接对所有 `SPARSE` 标签上线 mask。
2. 不要对 self-attention 做大范围裁剪。
3. 不要把 `S<->R1` fusion 当成普通 sparse candidate。
4. 不要用统一 threshold 跨 layer/head/step。
5. 不要只看 all-step mean，late-specific head 会被低估。

### 9.3 推荐的 expanded light probe

下一轮 light-efficient probe 应覆盖 heavy full 暴露出的盲区：

```yaml
sample_layers: [0, 1, 2, 4, 5, 9, 10, 14, 23, 24]
sample_heads:  [0, 1, 3, 4, 8, 10, 12, 15, 16, 19, 20, 21, 22]
sample_steps:  [0, 7, 14, 21, 27]
block_size:    256
max_query_tokens_per_record: all
```

成本约为 `10 × 13 × 5 = 650` 条 probe records，比 heavy full 便宜约 10 倍，但能覆盖当前 7 个核心 edge 的主要 head。

---

## 10. 当前 block-size 的关键局限

当前 `block_size=256` 的问题非常大：

- image token grid 是 78×52；
- 一个 block 大约覆盖 4.9 行 token × 52 列 token；
- 像素上约 78px 高 × 832px 宽；
- 它几乎横跨整张图宽度。

因此当前 sparse atlas 只能说明 **哪些 head/edge 在 1D strip-level 上集中**，不能说明它们已经精确定位到脸、手、背景区域或某个物体。

下一步必须做 2D tile rerun：

| 项 | 当前 | 建议 |
|---|---|---|
| block 形式 | 1D strip | 2D tile |
| block_size | 256 | 64 |
| tile shape | N/A | 8×8 tokens |
| pixel patch | 约 78×832 px | 约 128×128 px |
| segment aware | 弱 | 必须强制 segment-aware |

建议 pilot steps：`[0, 5, 10, 15, 20, 25, 27]`。

---

## 11. Causal ablation 设计建议

### 11.1 Ablation 目标

Ablation 不是为了验证“某个 mass 很高”，而是验证：切掉某条路径后，输出图、hidden state、reference fidelity、prompt adherence、source preservation 是否发生可解释变化。

### 11.2 分组设计

| 组 | 操作 | 要验证的问题 |
|---|---|---|
| A | block `L14/H12 X->R1` | light 发现的 reference-transfer 是否有因果作用 |
| B | block `L19/H0 X->R1` | heavy top reference-transfer 是否更关键 |
| C | block `L14/H20 X->P` | light prompt-control 是否影响 prompt adherence |
| D | block `L12/H17 X->P` | heavy top prompt-control 是否主控 prompt adherence |
| E | block `L5/H22 R1->X/S->X` | writeback 是否是 reference 进入 target 的主路径 |
| F | block `L10/H1 S<->R1` | fusion 是否必要，是否影响 source/ref 混合 |
| G | A+B+E 组合 | 同时切 direct transfer 与 writeback 后 reference 是否显著消失 |

### 11.3 指标建议

| 维度 | 指标 |
|---|---|
| 输出图像 | pixel diff、LPIPS、DINO/CLIP image similarity |
| prompt adherence | CLIP text-image score、人工检查是否仍“只改背景” |
| reference fidelity | reference crop/feature similarity |
| source preservation | source foreground/identity/布局保持度 |
| hidden deviation | per-layer hidden L2 / cosine drift |
| attention compensation | 是否出现其他 head/edge mass 上升补偿 |

### 11.4 Seed stability

在 ablation 前，先跑 seeds `0/1/2` 的 expanded light probe。只有 seed-stable 的 head 才适合做 sparse kernel 设计；seed-specific 的 head 更适合作为 fallback 或 diagnostic signal。

---

## 12. 最终结论

1. **当前分析已经能证明 FLUX.2 Klein 推理 attention 存在天然稀疏结构**，但它不是全局统一稀疏，而是机制化、head-specialized、step-conditioned。
2. **最重要的机制不是单一路径**：reference 既通过 `X->R1` direct transfer，也通过 `R1->X` writeback，还可能经过 `S<->R1` fusion 间接影响 target。
3. **Prompt control 是强机制**：`X->P` 不仅强，而且持续到中后期，尤其 `L12/H17` 和 `L14/H20`。
4. **Light run 是可靠 scout，但不是完整 atlas**：它发现了 `L14/H12` 和 `L14/H20`，但漏掉 `L19/H0`、`L12/H17`、`L5/H22`、`L10/H1`。
5. **Sparse atlas 可作为候选表，但不能直接变成最终 sparse inference 配置**：必须经过 2D tile rerun、seed stability、causal ablation。
6. **第一阶段 sparse/ablation 应聚焦 cross-segment 功能 head**，不要先动大范围 self-attention，也不要直接稀疏 diffuse fusion head。

一句话总结：

> 当前数据支持一种“机制感知的条件稀疏注意力”路线：对 `X->R1`、`X->P`、`R1/S->X` 这类高度专精 head 做小密度 sparse mask；对 late fusion、self-attention 和 diffuse head 保持 dense 或 fallback；在进入真实 sparse inference 前，用 2D tile 与 causal ablation 验证空间保真和因果作用。
