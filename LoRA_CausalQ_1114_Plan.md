# LoRA_CausalQ_1114_Plan

> 日期：2026-11-14。用途：放入 CausalQ_DG 工作空间，作为 Codex 分阶段实施指导书。
>
> 项目：[Ealea1028/CausalQ_DG](https://github.com/Ealea1028/CausalQ_DG)。
>
> 方案核对基线：`db388c954a24755d4999de90c6ed8b0de8c35fb6`。这是编写方案时的本机代码 SHA，不是未来实验发布 SHA。
>
> 状态：新研究设计，尚未实现或执行。文中的新文件、接口及命令模式均为待实施契约；历史结果不能填入本轮结果栏。

## 0. 给执行本方案的 Codex

本轮研究问题：在保留 CausalQ 定义与训练语义的前提下，仅给 DINO 骨干加入极少量 LoRA 参数，检验其能否改善分割、Anchor/Residual 跨风格稳定性以及 Query 分工；同时比较 DINOv2 和 DINOv3 两条独立骨干实验线。

执行规则：

1. 开始前阅读 `README.md`、`AGENTS.md`、`docs/METHOD.md`、`docs/AUTODL_NEXT.md` 和本方案；先核对当前提交、未提交文件及最新验收记录。
2. 一次只实施一个阶段。历史 Phase28–32、Phase33–38 均已结项，不重开旧研究，不覆盖其配置、验收清单或结论。
3. 本机负责代码、配置、CPU 合成测试、文档和返回结果分析；AutoDL 负责真实权重、真实数据、GPU 验证、训练、推理、特征提取和显存测量。
4. 当下一项验证需要 GPU 或远程数据时，更新 `docs/AUTODL_NEXT.md`，交付该阶段的真实完整 SHA、环境及可复制命令，在本机停止，等待实例结果。
5. 本方案交付本身只新增指导书。正式实施时才切换项目活动阶段并更新交接文档；不能仅因读到本方案就启动 28 个正式实验。
6. 不改旧公开 API 的默认行为，不删除失败测试，不复制旧 DAFormer/MRM 或 QK Adapter 工程。主实现沿用本仓库 EoMT/DINO/CausalQ；HPI 提供优化配方与既有 B 系列参考。
7. 每个阶段返回证据后继续预定下一阶段；负性能、方差排序不符合预期不构成删除实验的理由。工程错误须修复并记录，不能伪装为科学负结果。

## 1. 已核实的历史依据与本轮改变

### 1.1 DINOv3 强线：Phase28–32 的 UA、CA

直接参考以下真实文件，不凭实验名称重新猜测定义：

- `20261020_Diagnostic_Plan.md`。
- `configs/diagnostic_20261020/formal/ua_seed0.yaml`、`ca_seed0.yaml`。
- `analysis/diagnostic_final_report_20261028/comparison.csv`、`REPORT.md`。
- `experiments/DIAGNOSTIC_20261020/phase29_freeze.json`、`phase30_freeze.json`、`phase32_target_freeze.json`。

| 历史 DINOv3-L/16 模型 | Query 参数化 | 一致性 | 源域 mIoU % | Cityscapes mIoU % |
| --- | --- | --- | ---: | ---: |
| U0 | `q=a+e` | 无 | 73.8862 | 65.2091 |
| UA | `q=a+e` | Anchor CQE，权重 1 | 73.8529 | 66.0753 |
| C0 | `q=a+e-mean_R(e)` | 无 | 73.8542 | 65.6598 |
| CA | `q=a+e-mean_R(e)` | Anchor CQE，权重 1 | 73.8365 | 66.3220 |

这些是已知目标反馈下的历史单种子结果，说明选择 UA/CA 作为参考的依据，不能证明新 LoRA 必然受益。

本轮保留其 Query、解码器、效应、损失、数据划分和几何定义，但统一采用第 6 节的 HPI 优化配方与第 8 节的新 batch 合同。因此必须重训本轮无 LoRA 对照；不能把历史 66.0753/66.3220 当作新配方下的配对基线。

### 1.2 B 系列：保留已存在的实验含义

参考 `20261031_CausalQ_HPI_DINOv2_Plan.md`、`configs/hpi_dinov2_20261031/formal/`、`causalq/models/hpi_anchor_only.py`、`hpi_query_route.py` 和 `analysis/hpi_dinov2_final/REPORT.md`。

- B0 是 U0 的别名，不重复训练。
- B1：去掉可学习 Query residual，保留重复 Anchor 的 R=2 读出。
- B2：B1 加一个类别 Anchor 路由模块。
- B3：B2 加弱类别空间定位，`lambda_loc=0.1`。
- B4/B5 已有独立含义，本轮不增加其预测效应不变性与正贡献损失。

旧 DINOv2 B2/B3 目标 mIoU 为 62.1193/62.1149%，未呈现逐项叠加的单调改善。本轮必须保留负结果；不得把 B2/B3 重命名为不同 LoRA rank 或不同注入位置。

### 1.3 文献来源与明确的适用范围

- [HPI 论文，arXiv:2609.05864v1，§4.1](https://arxiv.org/html/2609.05864v1#S4.SS1)：AdamW、基础学习率 `1e-4`、1500 次线性 warm-up；论文正文写 batch2、12k iterations。
- [HPI 官方 G→C 配置，固定提交 6512993679a682422f82f675f4c6e9a2ccce4c9b](https://github.com/MosukFate/HPI/blob/6512993679a682422f82f675f4c6e9a2ccce4c9b/configs/hpi_clip_vit-l_1e-4_20k-g2c-512.py)：实际 runner 为 40000；AdamW `weight_decay=1e-5`，poly power0.9、`min_lr=1e-6`、warm-up ratio `1e-6`、clip norm1。
- [LoRA 原文](https://arxiv.org/abs/2106.09685)：冻结预训练权重，训练低秩增量。本方案的 rank、层位置和缩放是预先指定的工程设计，不是 HPI 已验证的最优 LoRA 配方。
- [DINOv2 官方特征实现，固定版本](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py)：参考 CLS/Patch 分离和末层归一化语义。

本研究采用 HPI 官方配置的 40k 优化时间表，同时披露与论文 12k 的差异。只迁移适用于当前模型的优化规则；不移入 VLM、文本提示、Mask2Former 或 HPI 全模型，也不宣称复现 HPI 的完整 SOTA 结果。

## 2. 本机、GitHub、AutoDL 的明确边界

| 地点 | 应做的工作 | 不应在这里做的工作 |
| --- | --- | --- |
| 本机 `D:\CrossDomainProject\CausalQ_Project` | 编辑源代码与 YAML；CPU 合成测试；统计/PCA 工具实现；生成清单与交接；核对和分析返回的 JSON/CSV/图 | 真实 DINO 大权重加载、全量数据扫描、GPU smoke、训练、真实图像推理、显存测量 |
| GitHub repo | 保存代码、配置、文档、依赖锁、实验合同与小型汇总；连接本机和实例 | 保存数据集、预训练权重、模型/优化器检查点、全量特征缓存、完整 run 目录 |
| AutoDL RTX 6000D / 用户标称 84GB | 检查实际硬件与可用显存；加载两种骨干；数据检查；GPU forward/backward；训练、源诊断、目标评估、特征提取 | 手工修改源码、临时改学习率/损失、在旧运行目录混跑新协议 |

建议独立路径：

```text
本机分支：codex/lora-causalq-1114
研究 ID：lora_causalq_1114
实例代码：/root/autodl-tmp/CausalQ_DG_lora1114_<release_sha前7位>
实例数据：/root/autodl-tmp/datasets
实例预训练权重：/root/autodl-tmp/pretrained
实例输出：/root/autodl-tmp/outputs/CausalQ_DG/lora_causalq_1114
Git 内元数据：experiments/LORA_CAUSALQ_1114/
本机汇总分析：analysis/lora_causalq_1114/
```

每次交付遵循“本机实现与测试 → 提交并推送 GitHub → AutoDL fetch → detached checkout 精确 SHA → 实例执行 → 回传小型证据 → 本机验收”。

- 以真正可读取的远端提交验证发布完成；方案基线 SHA、文档 SHA 不能代替实现 SHA。
- 每阶段使用独立代码检出与 run 路径。不可在运行中切换其代码 SHA。
- Git 工作区必须干净；输出位于代码目录外。发现实例源码修改先保留并回报，不用 `reset --hard` 覆盖。
- 原数据清单和权重身份能复用时直接复用；只做新环境所必需的检查，不重跑已验收的旧研究。
- 实例可能已有 DINOv2 环境。DINOv3 需要的 Transformers 等依赖须另核对，不能因旧 DINOv2 已通过就声称 DINOv3 可加载；必要时创建独立、可锁定的新环境。

## 3. 双骨干与特征 forward 契约

### 3.1 固定两条独立实验线

“双线”是两个独立分割模型与实验矩阵，不是把两个 DINO 输出融合成一个模型。

| 字段 | D3 线 | D2 线 |
| --- | --- | --- |
| 真实模型 | DINOv3 ViT-L/16 | DINOv2 ViT-L/14 的仓库既有 HPI patch16 适配版本 |
| 仓库标识 | `dinov3_vitl16` | `dinov2_vitl14_hpi_p16` |
| 主干宽度/深度 | 1024 / 24 | 1024 / 24 |
| Patch size | 16 | 16；必须披露由原生 patch14 转换 |
| 解码器读取点 | 第6/12/18/24个 block | 第6/12/18/24个 block |
| 当前代码索引 | HF hidden states `[6,12,18,24]` | 零基 block `[5,11,17,23]` |
| Query 输入 | 当前末层 Patch tokens，保留原规范 | 末层 LayerNorm 后 Patch tokens |

优先复用已验收的权重身份：D2 见 `experiments/HPI_DINOV2_20261031/weight_identity.json`；D3 见旧 UA/CA 的权重 manifest。AutoDL 核对完整文件/张量摘要及模型配置，不能只对比文件名。

D2 保留 FP32 bicubic 14→16 patch kernel、37→32 位置网格转换，保留 LayerNorm、LayerScale、CLS 和既有运行时位置插值；严格加载。转换只在实例按固定配方执行一次。不能改成原生 patch14 后继续使用同一实验 ID。

### 3.2 同时返回 Patch 和 CLS，兼容原接口

当前 `DINOv3Features` 只有 `patch_map`、`patch_tokens`、`intermediate_maps`；D2 也复用该结构。**当前代码尚未暴露 CLS，必须在实施阶段补齐。**

推荐保持前三个字段及位置参数不变，在末尾添加有默认值的字段，或新增显式扩展接口：

```text
patch_tokens       [B, Hp*Wp, 1024]，保持旧 Query 使用的归一化语义
patch_map          [B, 1024, Hp, Wp]，与 patch_tokens 数值对应
intermediate_maps  4 × [B, 1024, Hp, Wp]，保持各骨干原先的归一化语义
cls_token          [B, 1024]，同一真实 forward 的末层 CLS
register_tokens    可选；单独保存，不属于 patch_tokens
grid_size          可选元数据，或从 patch_map 可靠推导
```

具体约束：

1. D3 从真实输出提取 CLS；根据模型 metadata 识别 CLS/register/patch 的位置与数量，不硬编码“从 token1 开始全部是 Patch”。
2. D2 从 `norm(x)` 的 CLS 位置取 CLS，Patch 取其余真实 Patch；四个中间层继续使用原先 raw post-block 特征。
3. CLS 不能用 Patch 均值、分类 logits、Query 或零张量代替。记录 CLS 是否经过末层 norm。
4. 兼容旧的三位置字段构造、旧 `forward(images)`、旧 checkpoint 及未开启 LoRA 的行为。
5. `hpi_query_route.py` 等重新组装特征对象的地方须传递 CLS/元数据，不能悄悄丢失新字段。路由仅改变约定的 Patch 路径；CLS 保留骨干原始输出。
6. 返回特征默认保持可微；仅在显式评估/导出时 detach。新增 CLS 返回不意味着新增 CLS 监督、CLS 分割路径或 CLS 损失。
7. 验证 batch1/batch多样本、512方图、非方图、原生1024×2048；`N=Hp*Wp`，pad 和有效像素 mask 一致。

DINO 仍完成图像 patch embedding、位置编码/旋转位置处理、Transformer 和特征输出；CausalQ 仍是其后的 Query/分割逻辑。LoRA 只是指定线性层的低秩增量。

## 4. 极轻量 LoRA：只增加一个可控因素

### 4.1 首轮固定配置

对指定线性变换使用：

```text
y = W0 x + (alpha_lora / r) B(Ax)
W0 冻结；A、B 可训练；A 随机初始化，B 全零初始化。
```

| 字段 | 本轮值 |
| --- | --- |
| 注入位置 | 两种骨干最后6个 block：零基18–23，即第19–24层 |
| 目标投影 | self-attention 的 Q、V |
| rank | `r=4` |
| `alpha_lora` | 4，缩放为1；与 CausalQ `alpha` 完全不同 |
| LoRA dropout | 0.0 |
| LoRA bias | 不新增；原 bias 冻结 |
| 其他骨干参数 | Patch embed、位置编码、CLS/register 参数、K、output projection、MLP、LayerNorm/LayerScale 全冻结 |
| 可训练范围 | LoRA A/B + 该实验原有可训练 CausalQ/decoder/router 参数 |
| 训练方式 | 从相同预训练骨干、新头和零增量 LoRA 开始，端到端训练；不从历史 UA/CA 或本轮已训练头继续 |

在两条线的实际 Q/V 都为1024→1024时，LoRA 新增参数为：

```text
6 blocks × 2 projections × (1024×4 + 4×1024) = 98,304
```

报告实际统计值、占骨干总参数的比例和模型总可训练参数。98,304 是该结构下的推算，不代替实例检查；极轻量指额外骨干训练参数，不暗示 decoder 也只有这些参数。

不在首轮同时扫 rank、解冻 LayerNorm、注入 MLP、增加蒸馏或新正则。若后续需要 rank/位置消融，另立阶段并在新目标反馈前冻结。

### 4.2 适配真实 QKV 布局

- D2 当前是 fused `attn.qkv`。保留原 QKV 权重和输出排列，只为 Q、V 两个分片添加独立低秩增量，K 分片严格为零增量。
- D3 按实际 Transformers 模块结构定位独立 Q/V 或 fused QKV；先导出匹配清单，再注入，禁止不经检查地套用 `q_proj/v_proj` 字符串。
- 验收应恰好匹配6个 block、12个 Q/V 目标、24个 A/B 矩阵；目标缺失、重复、注入 K/MLP 或参数覆盖不符直接失败。
- 保留原 attention、位置处理、SDPA 调用和特征 hook 语义，不用通用语言模型包装器替换 DINO forward。
- 默认不 merge。若提供推理 merge，单独验证 merge/unmerge 等价、不可重复叠加，且不改原始权重文件。

### 4.3 必须修复冻结包装器与梯度边界

当前两个 backbone wrapper 在 `freeze=True` 时会用 `torch.no_grad()` 包住整段 forward。只给 LoRA 设置 `requires_grad=True` 不足以恢复这条计算图。

必须区分：

```text
base_weights_frozen = true       # 原始 W0 不更新
adapters_trainable = true        # LoRA A/B 更新
forward_requires_grad = true    # 训练时适配层及后续路径建图
```

要求：

1. 先冻结 W0，再只打开 LoRA 和该变体允许的头参数；不要 `backbone.requires_grad_(True)` 全解冻。
2. LoRA 训练路径不得被 wrapper、caller、feature producer 的 `no_grad/inference_mode/detach` 截断。
3. 第一个 LoRA 之前的纯冻结前缀可无梯度执行；从首个适配层开始到输出需保留计算图。下游未训练的线性层仍可能必须对其输入求导。
4. 骨干 eval 与 autograd 是不同开关。可保留预训练骨干的确定性 eval 行为，同时让 LoRA 求梯度；头的 dropout0.1 维持原训练行为。
5. 若使用 activation checkpointing，采用真实支持冻结输入梯度场景的方案，优先 `use_reentrant=False`，并验证 LoRA 非零梯度和恢复一致性；不能为了凑梯度把图像设为 trainable。
6. 零初始化 B 时，第一次 backward 的 A 梯度可为零，B 应获得有效梯度；在实际多次更新后再确认 A/B 都有学习信号。不得要求第0步所有 LoRA 参数梯度非零。
7. W0 在训练前后摘要保持一致；LoRA 在工程更新后发生变化。CausalQ 的初始 `alpha=0` 保留，不为方便观察梯度改成非零。

### 4.4 LoRA 训练不使用历史冻结特征共享路径

`CohortRunner` / `feature_pipeline` 的历史设计共享 no-grad DINO 特征，只适合 L0 冻结骨干实验。

- L1 每个实验独立维护 LoRA、优化器、RNG、游标和 forward。不同实验的适配后特征禁止共享。
- 同一更新内，Full/Anchor 使用同一视图的一次骨干输出；原图和 photo 各有自己的可微骨干输出。
- 可共享不可变数据/增强结果；可在确有收益且校验通过时共享第一 LoRA 之前的冻结前缀，但首版默认不实现额外缓存。
- 不能预计算第19–24层特征供后续多次 optimizer.step 使用；不能跨更新保留旧计算图。
- 正式默认单实验进程顺序执行，先用 batch/流水线填充算力。多进程 DataLoader 不等于同时启动多个共享可训练骨干的实验。
- 推理阶段可以无梯度；不同 LoRA 检查点仍需独立适配后 forward。L0 可共享相同冻结骨干输出，但须证明预测与独立运行相同。

## 5. 不破坏 CausalQ 的数学和训练语义

### 5.1 Query 与效应保持原定义

```text
U 组：q[c,r] = a[c] + e[c,r]
C 组：q[c,r] = a[c] + e[c,r] - mean_r(e[c,r])
R = 2；static；cross_attention_layers = 0
temperature = 0.07；聚合 = logsumexp
alpha_query_init = 0.0，之后可训练

E_full   = alpha_query * F(q, features)
E_anchor = alpha_query * F(repeat(a, R=2), features)
E_res    = FP32(E_full) - FP32(E_anchor)
z        = z_base + E_full
```

`E_res` 是输出空间的 Residual Contribution，不是把 e 单独送入 head 的输出，也不是独立 residual 的因果效应。不能通过相减 BF16 最终 logits 来替代已有 pre-addition 分量接口。

Anchor 路径必须重复两次 Anchor 并用相同聚合；单槽替换会改变 logsumexp 常数。LoRA 不改变这些公式、Query 参数命名、类别隔离或 alpha 的含义。

### 5.2 UA/CA 保留原 Anchor CQE

对每个源图及其同几何 photo：

```text
L_CE = CE(z_original, y) + CE(z_photo, y)
L_anchor = SmoothL1(N(E_anchor_photo), stopgrad(N(E_anchor_original)))
L_UA_or_CA = L_CE + 1.0 * L_anchor
```

N 维持原实现：FP32、非ignore有效区域、每图每类空间 L2 归一化，SmoothL1 beta1、空间求和、GT-present 类平均；不是只在对应类别 GT 区域算一致性。

- 原图参考仅对 CQE detach，原图 CE 仍对其 LoRA、decoder、Query 求梯度。
- photo 侧 Anchor 路径保持可微，使 CQE 可更新其 LoRA 及共享投影/Anchor。
- 两个 CE 权重各1，不能改成两个视图整体除2。batch 内先按原单图定义算损失，再对源图求平均。
- 默认使用原图 CE backward → 保存已 detach 参考 → photo 的 CE+CQE backward；完成整个有效 batch 后统一 clip/step。两视图之间不执行 optimizer.step。
- 若合并两视图骨干 forward，必须证明其梯度、dropout/RNG 和 loss 与约定执行方式一致；不能直接复制旧的两次 backward 导致共享图被提前释放。
- 不在 UA/CA 增加预测 KL、Full/Residual CQE、B3定位、B4/B5或新 CLS loss。

U 组存在 `a→a+t, e→e−t` 的分解不唯一性。保留 Anchor 范数、residual 均值范数、raw/centered residual 范数与 q 范数；不加隐式重中心化或正交投影以“修正”结果。

### 5.3 B1/B2/B3 的准确边界

- B1：公共初始化后将 e 清零并冻结，依旧重复 Anchor 两次、保留 R=2 logsumexp。
- B2：B1 + 现有8头、route width256、gate width32、gamma初值0.1的 `Aᵀ(AV)` 路由。同一注入增量写入末层 normalized Patch 与 raw layer24 的既有路径。
- B3：B2 + 原定位损失。使用有效面积归一化的 Patch 软标签，保留边界混合、忽略padding/255；对匹配 logits 在类别维归一化，原图/photo各算一次取平均，权重0.1。
- `lambda_presence=0`；B1–B3 没有 UA/CA CQE。B2→B3 只新增定位监督。
- LoRA 与路由注入是不同增量：前者在 DINO attention 内，后者是既有 CausalQ Patch 路由。不得把路由 delta 改称 Query residual。

## 6. 优化器和学习率：明确采用 HPI 官方配方

### 6.1 全部本轮实验统一使用

| 字段 | 固定值 | 来源/说明 |
| --- | --- | --- |
| optimizer | AdamW | HPI 论文和官方配置 |
| base LR | `1e-4` | LoRA、decoder、Query、现有router均为1倍基础率 |
| weight decay | `1e-5` | HPI 官方 G→C 配置；不同于旧研究的0.01 |
| norm decay | 0 | 对本模型真实 norm 参数显式分组，沿用 HPI norm 排除规则 |
| 其他 bias/标量 decay | 基础 `1e-5`，除非属于 norm | 不擅自扩展为“所有 bias 和 alpha 均无 decay” |
| betas / eps | `(0.9,0.999)` / `1e-8` | 官方配置未覆盖，显式固定采用的 AdamW 默认值 |
| LR policy | poly，power0.9 | HPI 官方配置 |
| min LR | `1e-6` | HPI 官方配置 |
| warm-up | linear，1500 optimizer updates | HPI 论文和官方配置 |
| warm-up ratio | `1e-6` | HPI 官方配置 |
| 总更新数 | 每模型40000 | HPI 官方该配置 runner；也保留历史最终40k的命名 |
| gradient clip | 全部可训练参数合并 norm2，max1 | HPI 官方配置；每个有效batch只裁剪一次 |
| precision | BF16 autocast；LoRA/头主权重与AdamW状态FP32 | 本轮运行合同，实例验证 |

HPI 的10/12/20倍学习率属于它特定的 temperature、SCC gate 和 beta 参数。本模型没有相同模块；不把这些倍数任意套到 CausalQ temperature、alpha、gamma 或 LoRA。

LoRA 初始化、分组规则是本方案新增映射，不宣称 HPI 自身采用 LoRA。不开启自动线性 LR 随 batch 放大。

### 6.2 时间表和恢复不得产生歧义

令 t 为当前更新前的绝对零基 optimizer step，T=40000：

```text
poly(t) = 1e-6 + (1e-4 - 1e-6) * (1 - t/T)^0.9
warm(t) = 1e-6 + (1 - 1e-6) * t/1500       if 0 <= t < 1500
          1                              if t >= 1500
lr(t) = poly(t) * warm(t)
```

这是按 HPI/MMCV 的“全程poly × warm-up系数”实现；不是 warm-up结束再重启一个40k poly。记录首步、1499、1500、20000、39999和恢复点的实际 LR；首步极小 LR 来自官方 warm-up ratio，不能静默换成常见的0.001。

调度只随 optimizer.step 前进，不随 photo forward、microbatch、DataLoader packet 或 LoRA 模块数前进。保存 optimizer group、scheduler 状态与绝对更新数；恢复不能再次 warm-up。

## 7. 正式实验矩阵：28 个模型，全部配对

### 7.1 命名

```text
D3_<variant>_L0_S0     DINOv3，无 LoRA
D3_<variant>_L1_S0     DINOv3，固定 LoRA
D2P16_<variant>_L0_S0  DINOv2 HPI-p16，无 LoRA
D2P16_<variant>_L1_S0  DINOv2 HPI-p16，固定 LoRA
```

`L0/L1` 仅指 LoRA关闭/开启，不代表 CausalQ 层数。`B0=U0` 不单独训练。

| variant | e 参数化 | 路由 | 定位 | Anchor CQE | 两骨干均跑 L0/L1 | 主要作用 |
| --- | --- | --- | --- | --- | --- | --- |
| U0 | 非中心化，可学习 | 无 | 无 | 无 | 是 | 原 CausalQ + LoRA 的最基本配对 |
| UA | 非中心化，可学习 | 无 | 无 | 1.0 | 是 | 历史 D3 强线迁移及LoRA收益 |
| C0 | 中心化，可学习 | 无 | 无 | 无 | 是 | CA 的必要匹配对照 |
| CA | 中心化，可学习 | 无 | 无 | 1.0 | 是 | 历史 D3 强线及中心化交互 |
| B1 | e=0 且冻结 | 无 | 无 | 无 | 是 | B2 的必要容量/参数化对照 |
| B2 | 同 B1 | 有 | 无 | 无 | 是 | 既有 Query 路由效果 |
| B3 | 同 B1 | 有 | 0.1 | 无 | 是 | 定位监督的额外效果 |

总计 `2 backbone × 7 variant × 2 LoRA = 28` 个 fresh seed0 模型；无LoRA14个、有LoRA14个。B1和C0不能省去，否则 B2或CA会同时改变两个因素。

### 7.2 初始化与公平性

- 每个模型从固定预训练骨干和新的公共头开始，均训练40000更新；不接着另一个实验训练。
- 同骨干、同variant的 L0/L1 公共头逐张量相同；L1初始骨干增量为0，初始 Patch/CLS/logits应等价。
- U0/UA/C0/CA 共享原始 a/e 初始化张量，再通过既定前向决定是否中心化；B1–B3在相同公共初始化后清零冻结e。
- 路由、LoRA初始化各使用独立 RNG，不改变公共头、数据、风格与dropout的序列。B2/B3路由初始状态相同。
- 两骨干统一样本/增强流和头构造；记录是否实现公共头张量一致。不将两种预训练表征称为同一初始特征。
- L0和L1使用相同有效batch、更新数、LR、精度和采样规则；不同物理microbatch只能是已验证的执行差异。
- 不能复用旧训练分数来节省本轮 L0，旧优化配方及batch不一致。

### 7.3 预声明对比

每条骨干分别报告：

1. 七个 `variant_L1 − variant_L0`：LoRA 对对应方法的增量。
2. L0和L1各自的 `UA−U0`、`CA−C0`、`CA−UA`、`C0−U0`。
3. L0和L1各自的 `B1−U0`、`B2−B1`、`B3−B2`。
4. LoRA×Anchor约束交互：`(UA_L1−U0_L1)−(UA_L0−U0_L0)`；CA/C0对应交互同报。
5. 中心化×Anchor约束交互：`(CA−C0)−(UA−U0)`，分别给L0/L1。
6. 同variant、同LoRA状态的 `D3−D2P16`；仅解释为这两个完整预训练骨干及适配实现的差异。

所有对比同时包含源/目标mIoU、方差、幅值和责任指标。seed0只能给探索性点估计；图像bootstrap不能替代训练种子不确定性。

默认不扩张到 B4/B5、多rank或seed1/2。若预算需要缩减，应在本轮目标评估前形成新版本并明确哪些对比不再可识别，不能事后只保留最优组。

## 8. 训练协议与 RTX 6000D 算力利用

### 8.1 科学配方与 batch 合同

| 项目 | 固定要求 |
| --- | --- |
| 源域 | 原冻结 GTA5：22469 train / 2497 holdout |
| 主目标域 | 原冻结 Cityscapes val500；本轮不扩展BDD/Mapillary |
| 类别 | 19 trainIds，ignore255；原数据清单SHA |
| 训练几何 | 512×512，scale[0.5,2.0]，flip0.5；原min_valid_fraction0.01/crop_attempts10 |
| 输入归一化 | 本仓库既有ImageNet RGB口径；不改CLIP mean/std |
| 两视图 | 原图+同几何photometric |
| photo参数 | brightness/contrast/saturation[0.7,1.3]；gamma[0.8,1.2]；temperature[0.9,1.1]；grayscale概率0.1 |
| 解码器 | 原256通道、dropout0.1和四层读取 |
| 有效batch | 每次更新8个源图，每图两个视图，`B_eff_source=8` |
| 正式更新 | 每模型40000；固定最终检查点，不挑best |
| 恢复 | 每1000更新原子检查点；源诊断按图像恢复 |

**本方案主动把历史有效batch1改为8以提高算力利用。** 这会改变优化轨迹和训练样本暴露量，并非无损执行优化；因此新矩阵全部重训且统一batch8。HPI的基础LR保持1e-4。

每模型源图暴露量 `40000×8=320000`，两个视图合计640000；28模型合计1,120,000次更新、8,960,000源图暴露、17,920,000视图。与历史batch1模型不能直接归因比较。

允许 `microbatch_source × accumulation = 8`，例如8×1、4×2、2×4、1×8。分别记录 source batch、view batch、有效batch，不能把16个视图写成16个独立源样本。

累积时每个单图两视图损失除以有效源图数，整个batch后一次clip/step。不得更改CE视图权重或因OOM缩小有效batch。物理batch变化需通过等价性检查并写入execution profile；正式运行不能自动搜索新配方。

### 8.2 数据与执行流水线

1. 以12个spawn DataLoader workers、`persistent_workers=true`、`pin_memory=true`、`prefetch_factor=2`为初始候选；根据实例CPU核数、RAM和`/dev/shm`核对。必要时有界比较8/12/16，不能无上限堆workers。
2. 每个样本的几何/photo种子来自研究ID、seed、绝对样本位置、view的确定性规则；不能依赖worker到达顺序。采样器使用固定epoch/randperm规则。
3. CPU负责解码与确定性增强；GPU异步H2D，用有界双输入缓冲及CUDA事件管理生命周期。保持训练消费顺序，不能预取后改变已提交样本游标。
4. LoRA训练优先尝试microbatch4或8；有显存余量才增大物理batch，跨两个视图同时保图是否更快要端到端测量。
5. 使用经验证的SDPA/BF16。首版以eager为可靠基线；可选compile只有在兼容性和端到端计时通过后启用，编译成本单报。不得未经测量声称compile或大batch一定更快。
6. 训练循环不逐步 `.item()`、全参数哈希或写大量图片；日志每50更新聚合，完整哈希放在阶段边界/检查点处。
7. 内存不足时先降低物理microbatch、相应增加累积；再考虑仅对适配后block做activation checkpoint。保持有效batch8、分辨率、损失与总更新数。
8. 禁止通过空转kernel、无关任务或重复计算制造GPU占用率。L0允许有界冻结特征批处理；L1必须遵守第4.4节梯度边界。

### 8.3 实测验收，目标70%以上

实际GPU型号、driver、compute capability和可见显存以实例测量为准；不从“84GB”反推可用字节。历史RTX6000D环境只能提供起点，不能替代本次双骨干测量。

在正式训练前，对每种骨干的L1代表配置做一次有界源域工程定标，覆盖UA的两视图CQE和B3路由/定位。候选最多3条完整执行策略；每条暖机20更新后测量连续至少120秒，单候选总时长上限300秒。工程更新计数单列且丢弃状态，正式模型重新初始化。

先固定有效batch8，再比较物理batch、workers与缓冲。记录：

- `nvidia-smi` 每秒遥测的训练稳态GPU-Util：均值、中位数、p10/p90，以及≥60%/≥70%的时间比例。
- 数据等待、H2D、骨干forward/backward、头/损失、optimizer、checkpoint各耗时；真实source images/s、views/s和updates/s。
- 峰值allocated/reserved、设备可用显存、功耗、CPU/RAM、输入队列等待。
- 冷启动/编译、训练稳态、检查点保存、完整作业分别统计；训练稳态包括正常输入等待和同步，不能只挑GPU繁忙片段。

工程验收目标：稳态GPU活动中位数≥70%，p10≥60%，数据等待占比<10%，显存reserved≤实际可见显存的85%，无OOM/NaN；同时吞吐优于保守microbatch1对照。它衡量设备活动程度，不等于精确SM算术利用率。

按端到端source images/s选择；吞吐差≤5%时选更低显存/更稳定方案。若最多3候选仍未达到60–70%目标，标记`performance_target_met=false`，返回瓶颈与实测数据，本机修复后再发布；不得把“已配置12 workers”写成“已达到70%”。不要无限增加工程试跑或用新目标mIoU选执行策略。

历史Phase35/36稳态中位数只有68%/62%，因此本方案不能承诺旧流水线已经满足目标。正式训练开始后的首个稳态窗口及每段恢复后继续验证遥测；源诊断/目标推理也分别定标batch和吞吐，不能拿训练占用率代替评估测量。

训练ETA仅由本轮测得的各variant耗时推算：`Σ(40000 / updates_per_second_variant)`，另加诊断、PCA和评估。L0共享骨干时按整个cohort真实墙钟计时，不把旧冻结骨干速度当L1训练速度。

## 9. 每个模型必报的结果与统计定义

### 9.1 分割指标

源域最终诊断覆盖2497张holdout，长边1024、保比例resize、pad16。五视图中的original forward同时产生源域mIoU，避免再做一遍同样原图推理。恢复记录须标记original是否已经计入，不能重复累计。

目标域在28个最终模型、完整配置和源诊断冻结后统一评估Cityscapes val500，原生1024×2048、整图BF16、argmax，无resize/crop/sliding/flip/TTA，无optimizer和参数更新。不能照搬会在训练中读取Cityscapes并保存best的HPI验证钩子。

保存19×19整数混淆矩阵、19类IoU、有效像素数、样本ID、覆盖数和预测/标签映射身份。以数据集累计混淆矩阵计算mIoU，不平均逐图mIoU：

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU = 对有效union类别的IoU取宏平均；缺失union按原评估器规则并明确记录
```

所有模型使用同一有效类别规则。JSON中同时保存[0,1]值与百分数；差值单位为百分点pp。目标未运行时为`null`，不填0或历史分数。

### 9.2 Anchor / Residual / Full 跨风格方差

复用Phase28/31定义与确定性种子：每图original+4个冻结photometric视图，共S=5；所有模型使用相同图像、几何、增强参数，eval、dropout关闭、零更新。风格生成使用独立RNG，不能消耗训练RNG。

对同一图像i、类别c和固定非ignore有效像素集合Ω，令E为第5节某一读出效应：

```text
mu(p)       = mean_s E_s(p)
Var_raw(E)  = mean_p mean_s (E_s(p) - mu(p))^2
RMS_s       = sqrt(mean_p E_s(p)^2)
E_norm_s    = E_s / RMS_s
Var_norm(E) = 对E_norm使用相同总体方差公式
```

- 方差分母为5，不用无偏分母4。另存original+photo1的两视图兼容方差，不能与五视图数值混用。
- 任一视图RMS≤1e-6，该图类归一化方差为`null`并计近零数；不通过epsilon除法制造“完美稳定”。
- 同时输出mean-abs、联合及逐视图RMS、signed mean、幅值变化、Anchor/Residual方差比及有效数。
- FP32效应图、FP64统计累计，保留逐图/逐类口径。主表先在每图内平均GT-present类别，再等权平均图像；all-class、逐类、像素加权和分位数另表标明。
- 保留空间相关、跨风格协方差与抵消指标，并验证 `Var(E_full)=Var(E_anchor)+Var(E_res)+2Cov(E_anchor,E_res)`。
- 方差下降同时伴随RMS下降只能说明可能存在幅值收缩；不能独立证明语义稳定或DG改善。

B1/B2/B3的e结构性缺失：`E_res=0`，原始方差/RMS为0，归一化方差和相关为`null`，`residual_status=structurally_absent`。不得把它们的0方差当作LoRA成功压制风格。

B2/B3在相同routed features上计算上述读出效应。它们不包含Anchor经路由改变base decoder的全部作用；报告必须命名为`readout_effect`，不能改称主路径总因果效应。

### 9.3 Query responsibility

在Query score原生Patch网格上使用已有含temperature0.07的分数，不再次除温度：

```text
rho[c,r,p] = softmax_r(score[c,r,p])
pi[c,r,p]  = softmax_(c,r)(score[c,r,p])
pi_class[c,p] = sum_r pi[c,r,p]
```

分别保存两类责任，不能把同类slot分工、跨类责任或router的Patch attention混成一个指标。

| 责任统计 | 要求 |
| --- | --- |
| 每类每slot平均rho | 原图及各风格；同类slot总和为1 |
| 像素责任熵 | 先算每像素熵再平均 |
| 使用熵/有效Query数 | 对数据集平均责任算熵及exp(H)，另报归一化熵 |
| top1份额 | 并列平分；保留原slot编号 |
| 风格变化 | 责任绝对变化、总体方差、JS divergence、top1切换率 |
| 跨类pi_class | 类别责任、熵、使用情况及风格JS，所有组统一报告 |
| B2/B3路由 | 独立输出attention、gate、gamma和注入RMS |

主slot统计在GT类别区域、nearest下采样到score网格的有效位置计算；all-valid补充独立保存。像素加权与每图等权分别输出；缺失类别记`null`并给计数，不插值放大Patch测量数。

B1–B3两个Anchor槽重复，rho恒为0.5，有效独立Query为1；高熵是结构退化，不是良好分工。跨模型slot可交换，不能默认UA的slot0等于CA的slot0。责任不是效应强度，必须结合alpha、RMS及分割结果解释。

### 9.4 路由组的主路径补充诊断

为兼容已有B2/B3分析，复用Git冻结的128源图清单，针对每图每个GT-present类别及五视图做on/off诊断：真正关闭该类路由写回与其additive读出，重新计算后续gate/base decoder；不能用Q=0向量冒充移除。

保持输入与骨干适配后特征相同，每视图仅一次DINO forward。保存正确类`p_on/p_off`、`tau=mean_GT_region(p_on−p_off)`、正/负/近零比例、跨风格方差。此项是128图补充，不能冒充2497图全量；不引入B4/B5训练损失。

本轮不自动重做旧2497图Query梯度普查。新增梯度验收聚焦LoRA和CausalQ计算图；若需要科学梯度分析，另行冻结源域小样本并保持`Anchor_total=via_query+direct`原分解。

## 10. 训练完成后的联合 PCA 与原始特征对比

### 10.1 工具目标和真实对比对象

为本轮实现独立、可恢复的联合PCA工具；仓库扫描未发现现成的受控PCA入口，不得在交付中声称已经存在。

每种骨干分别比较：

1. 固定预训练权重、无LoRA的原始DINO特征。
2. 本轮七种L1模型最终LoRA权重下的DINO特征。
3. 同variant的L0分割模型作为语义表现对照；其router/decoder输出如展示须另命名。

主PCA提取点在DINO输出、进入CausalQ Query/router之前。不能用B2/B3路由后的特征冒充“LoRA改变的骨干特征”。主结果使用末层Patch tokens；四个中间层和CLS另作分开的特征比较。

### 10.2 固定源域样本与一套共同坐标

- 在读取本轮图像/特征前，复用已有128图源清单作为PCA样本；另按固定seed20261114抽12图做密集可视化，保存清单SHA。所有模型使用同样的12图。
- 使用与第9节相同的源评估几何、original+4个photo、有效mask。每图每视图按固定坐标均匀抽32个有效Patch用于拟合；不足时取全部并保存计数。
- 每种骨干拟合集包括原始DINO和七个最终L1 DINO，各组等权。典型上限 `128×5×32×8=163840` 行；总上限200000行。对不满样本的图使用相同组间坐标，不允许某组挑自己的高响应Patch。
- 用FP32/FP64稳定计算共同均值和一次PCA/SVD；所有组投影 `Z=(X−mu_joint) @ V_joint`。`whiten=false`，默认不额外做逐模型z-score或L2归一化。
- 同一骨干的原始/L1、各variant及五视图必须使用同一均值、基底和颜色映射。禁止各自fit PCA后对比颜色。
- 为固定PCA符号，将每个PC绝对载荷最大维度置为正；报告解释方差与累计解释方差。
- D2与D3分别拟合，不因两者维度都是1024就把隐藏通道当作天然对齐。两条线的RGB颜色不能直接解释为同一种语义。
- 四个中间层分别拟合各自的联合PCA，不混合不同层数值尺度。CLS使用独立联合PCA；128×5图级向量足够，不能把CLS拼到Patch矩阵。

若联合样本偏向某组会影响主方向，应在报告中说明；可补充“仅原始DINO拟合、所有组投影”的参考视图，必须独立命名，不能替换主联合结果。

### 10.3 可视化与定量输出

每条骨干生成对齐面板：输入原图、photo、可用GT、原始DINO的PCA-RGB、对应最终LoRA特征的PCA-RGB、Patch变化热图、L0/L1分割结果。第一至三PC映射RGB，所有组共用联合投影分位数范围（如1%–99%）；保存范围，禁止每张图独立拉伸。

输出：

- 原始/L1的共同二维散点，按模型/风格着色；源GT类别着色可作解释，不参与PCA拟合或模型选择。
- 同坐标Patch的cosine similarity、相对L2位移、特征范数变化、跨风格特征方差，以及各PC解释方差。
- CLS向量的cosine/相对L2漂移和独立PCA散点；CLS向量不绘制成Patch热图。
- 至少对比U0、UA、CA、B2、B3，并交付七个L1完整数据；不只展示看起来最好的几张。
- 共享的PCA mean/components、样本及Patch索引、采样seed、特征层/归一化身份、模型/权重SHA、RGB范围、工具版本与摘要。

PCA只读：模型eval、无更新，不反写训练配置。提取前后校验模型状态/RNG；不以颜色“更干净”证明机制或泛化提高。

源PCA在源诊断阶段完成。目标PCA仅在所有模型/规则冻结后的最终目标阶段允许：可用固定、预声明Cityscapes无标签样本套用源域PCA基底，**不重新fit，不用目标GT或PCA观感调参**。目标PCA是补充项，不能代替必做源PCA。

### 10.4 计算与存储边界

真实特征提取在AutoDL执行；PCA拟合可在实例CPU/GPU完成，或将有界采样特征回传本机做CPU结果分析。全量dense特征和checkpoint留实例，不进入Git。

对L0原始骨干特征每条线只提取一次；L1按独立LoRA检查点提取。流式采样、及时释放GPU张量，不保留28模型的全部激活。Git只保存小型PCA元数据/统计/必要图；大数组以路径和SHA引用。

建议产物：`pca_manifest.json`、`pca_basis_<backbone>_<layer>.npz`、`feature_drift.csv`、`pca_explained_variance.csv`、`pca_samples.json`、`pca_panels/`、`index.html`。是否将basis入Git取决于实际大小，必须在metadata记录其SHA。

## 11. 检查点、恢复与证据身份

### 11.1 新恢复格式必须包含 LoRA

当前历史 `cohort_runner.head_state()` 会排除所有`backbone.*`键，直接复用会丢失LoRA。实现新研究专用schema或显式向后兼容扩展，保留旧研究行为。

每个正式检查点至少包含：

- 原始骨干文件/张量身份及转换配方，明确骨干W0不在compact checkpoint中。
- 完整LoRA A/B、目标路径、rank、scale、注入层与结构版本，包括`backbone.*`下的trainable键。
- 该variant的全部decoder/Query/router状态，以及B1–B3 residual结构性零的重建/验证合同。
- optimizer全部状态和精确参数组、scheduler、绝对step、有效batch/累积状态、输入流位置。
- Python/NumPy/Torch CPU/CUDA RNG、sampler epoch/offset、增强规则和独立初始化seed。
- 训练producer SHA、配置hash、环境锁、数据清单SHA、运行ID、完成标记和文件SHA256。

不要只保存`requires_grad`参数而遗漏会影响输出的buffer或结构状态。checkpoint按完整optimizer更新边界保存；若不保存microbatch中间梯度，恢复只能回到最后完整提交边界，未提交样本允许确定性重放。

采用临时文件→完整CPU快照→hash→原子rename/完成标记；每1000更新保存，不覆盖已验收的immutable segment。优化器更新游标与写日志游标分开，以完整检查点为已提交进度。

### 11.2 接受与拒绝恢复

仅在同study、backbone、base权重、LoRA布局、variant、科学配置、样本流和环境合同匹配时恢复。L0↔L1、D2↔D3、rank变动或损失变化不得静默加载；必须显式失败。

同SHA优先直接恢复；工程修复迁移SHA时保留原producer，记录新恢复SHA和差异，证明已提交状态/优化器/RNG/输入连续。不同GPU kernel导致的数值变化必须说明，不虚称逐位一致。

评估/PCA恢复以模型hash+数据ID+view+几何+工具版本为记录键，已完成记录不重计数；不能因部分包损坏重新训练模型。

## 12. 本机实施模块与必要测试

以下路径为建议，实施者可按现有结构调整，但交付时必须列出真实路径：

| 文件/目录 | 职责 |
| --- | --- |
| `causalq/models/lora.py` | 原线性层低秩增量、fused QV、注入清单和trainability审计 |
| 现有 `dinov2_wrapper.py`、`dinov3_wrapper.py`、`backbone_factory.py` | 可微LoRA路径、Patch/CLS统一返回、旧API兼容 |
| 现有 `hpi_query_route.py` | 扩展特征对象的CLS/metadata传递；保留原路由数学 |
| `causalq/protocols/lora_causalq_1114.py` | 矩阵、字段约束、阶段权限和predeclared contrasts |
| `causalq/runtime/lora_training.py` | 图像到LoRA到CausalQ训练、有效batch与恢复；不套用冻结特征头runner |
| `causalq/runtime/lora_prepare.py` | 实例权重/forward/backward/速度/恢复工程检查 |
| `causalq/analysis/joint_pca.py` | 联合拟合、统一投影/颜色、抽样与漂移统计 |
| `configs/lora_causalq_1114/` | 双骨干×七variant×L0/L1完整解析配置、execution profiles |
| `tools/run_lora_causalq_1114.py` | 分阶段prepare/train/diagnose/evaluate/status/package入口 |
| `tools/visualize_lora_joint_pca.py` | 受控提取与PCA入口，区分实例提取和本机分析 |
| `scripts/run_lora_causalq_1114.sh` | exact-SHA后台启动、锁、PID、日志、退出码、恢复 |
| `experiments/LORA_CAUSALQ_1114/` | 协议、freeze、release、environment和轻量acceptance |

每新增模块写聚焦测试，复用已有数学测试，不另写只复述实现的空测试。至少覆盖：

1. LoRA初始恒等、adapter关闭等价、fused Q/V排列和K零增量、注入目标与参数量、拒绝重复注入。
2. 真实结构的微型合成D2/D3：多batch/非方图、CLS与Patch分离、register排除、四层索引/归一化、旧三字段构造兼容；不得在本机下载真实大权重。
3. W0无梯度且不更新；B初始有效梯度；多步后A/B梯度；LoRA可经CE和photo侧CQE更新；原图CQE参考确实detach。
4. L0与L1零增量初始logits/效应一致；R=2 Anchor重复、C/U参数化、Residual差分、alpha初值、B1结构性零保持。
5. microbatch累积和完整有效batch在无dropout/受控RNG条件下梯度等价；生产dropout策略保持配对，不因packet重排变化。
6. LR边界、norm/其他参数分组、optimizer恰好覆盖全部且仅覆盖允许训练参数；恢复不重启warm-up。
7. compact checkpoint包含LoRA及输出相关状态；连续训练与断点训练一致；拒绝错误权重/variant/layout恢复。
8. 方差协方差恒等式、零能量null、责任归一化/并列、结构性零状态；尽量直接复用已通过的现有统计工具。
9. PCA共同basis投影、固定符号、共同颜色范围、采样组平衡、禁止逐组fit/混入CLS、只读状态不变。
10. 阶段权限：prepare不产生正式更新；训练/源诊断不读目标；target阶段拒绝未冻结模型；未实现模式明确失败。

每阶段本机改动后运行 `python -m pytest` 全量测试，记录实际解释器、CPU-only配置、结果和跳过原因。不能以文档模板或历史“754 passed”代替当前测试。

实例工程验收一次覆盖真实两骨干严格加载、Patch/CLS、LoRA梯度、零增量等价、少量更新、fresh-process恢复、BF16、目标尺寸的合成张量和定标。用源数据做工程测试，不读目标图/GT。数值容差必须在看返回误差前预设，不按结果放宽。

建议CPU FP32恒等/恢复 `atol=1e-6, rtol=1e-5`；实例BF16零增量对比优先要求同kernel精确一致；确有kernel差异时预设 `atol=1e-2, rtol=1e-2, relative_L2<=1e-3` 并输出max/mean误差。关键代数仍FP32验证≤1e-6。失败先定位，不直接提高容差；这些是本方案新工程标准，未声称已通过。

## 13. 分阶段实施与唯一下一步

阶段编号以当前仓库已完成Phase38为起点；若接手时已有后续阶段，先核对并顺延编号，保持本研究ID独立。每阶段的本机实现/CPU测试和实例执行属于同一个阶段，不跨阶段并行推进。训练阶段保存最终检查点并核对有限损失/状态/覆盖；完整源域mIoU统一在Phase45与五视图诊断合并计算，避免提前验证后再次重复原图推理。

| 阶段 | 本机与实例工作 | 本阶段结束条件 |
| --- | --- | --- |
| Phase39 | 本机：实现LoRA、Patch/CLS接口、协议与CPU合成测试；不训练 | 全量测试；默认旧API兼容；交付Phase40准备代码所需接口清单 |
| Phase40 | 本机：完成prepare/定标/恢复工程入口；AutoDL：真实双骨干验证和源域性能定标 | 权重/环境/LoRA匹配/梯度/恢复验收；固定batch8执行profile；0正式更新、无目标访问 |
| Phase41 | 本机：发布核心无LoRA合同；AutoDL：D3和D2的U0/UA/C0/CA L0，共8模型 | 各40k更新与最终检查点，320000正式更新；源mIoU待Phase45，不评目标 |
| Phase42 | 发布并执行D3的U0/UA/C0/CA L1，共4模型 | 各40k，160000新增更新；核对LoRA/W0状态；不评目标 |
| Phase43 | 发布并执行D2的U0/UA/C0/CA L1，共4模型 | 各40k，160000新增更新；不评目标 |
| Phase44 | 两骨干B1/B2/B3的L0/L1，共12模型；按骨干和variant顺序执行 | 各40k，480000新增更新；单个提交不同时加入多个新实验机制 |
| Phase45 | AutoDL：28模型全2497图五视图源诊断、B2/B3的128图补充、源PCA；本机分析 | 覆盖/原始定义/null/结果hash验收；冻结完整目标评估合同；0更新 |
| Phase46 | AutoDL：28模型固定最终Cityscapes500评估及可选无标签目标PCA；本机最终报告 | 所有预声明对比和负结果齐全；PCA图/元数据完整；0更新，结项 |

Phase44内部先复用已验证的B1、再接入B2、最后接入B3；每个模块独立提交和测试。新骨干上的路由接入须先CPU验证，不在一个实现提交中同时改LoRA、路由数学和定位损失。

工程/性能达标后不插入多轮20步/500步性能排名pilot。模型工程失败可修复；低mIoU、Anchor方差较大、B3不如B2都须按预定矩阵继续完整报告。

源诊断的逻辑模型视图数为`28×2497×5=349580`；目标推理为`28×500=14000`图次。L0可按同一骨干共享冻结原始特征，降低实际backbone调用；L1独立计算，报告逻辑模型图次和实际骨干图次，不能只报节省后的数字掩盖覆盖。

各阶段本机停止点必须写出：已完成内容、真实CPU测试、下一步需要实例的原因、完整实现SHA、唯一允许的实例模式、返回清单。禁止把Phase40–46串成未经阶段验收的无人审阅长任务。

## 14. AutoDL 交付命令与运行入口规范

### 14.1 每次发布必须先做实的事情

Codex应先实现本阶段入口并完成本机测试，再把真实40位SHA、已推送分支、精确环境路径、输出根和配置路径写入`docs/AUTODL_NEXT.md`。如使用sparse checkout，发布经验证的依赖闭包清单，包含必需代码/配置/协议/third_party，不默认下载旧图表与全部历史产物。

本文件不提供假SHA或声称现有旧runner可执行新模式。首次交付须给出以下三段真实可复制命令：

1. 拉取该阶段完整提交并校验工作区/环境/权重路径。
2. 仅启动本阶段后台任务，可重复调用以恢复，输出PID、日志和run目录。
3. 查看状态、退出码、完成文件，打包必要结果并显示导出路径/hash。

命令应在子Bash内使用严格失败处理，不在用户交互shell顶层设置会退出终端的状态。使用`python -m tools...`等已验证导入入口；不提供缺依赖、无模块导入或仅存在于未推送工作区的命令。

### 14.2 待实施的模式合同

```text
prepare                  Phase40，真实权重/梯度/恢复/定标，零正式更新
train-core-l0            Phase41，两个backbone的U0/UA/C0/CA，无LoRA
train-core-d3-l1         Phase42，D3核心4模型LoRA
train-core-d2-l1         Phase43，D2核心4模型LoRA
train-b-series           Phase44，B1/B2/B3双骨干LoRA配对
source-diagnostics       Phase45，全部最终模型源诊断与128图补充
joint-pca-source         Phase45，源域特征提取/联合PCA
target-evaluation        Phase46，仅冻结最终模型的Cityscapes500
joint-pca-target         Phase46，可选，只套用已冻结源PCA基底
status / package         只读状态或导出，不隐式触发训练/推理
```

这些名称是建议接口，当前尚不存在；实施交付必须替换为真实CLI及完整参数。launcher按阶段/资源互斥，防止双重启动和混写；恢复核对上次退出与已提交游标，不自动从头重训。

训练、源诊断和源PCA的dataset factory必须只接受源域；目标路径即使实例上已经存在，也不能在开发阶段被扫描或读取。监控/status不创建模型或加载数据。

### 14.3 每次返回的最小证据

```text
run_manifest.json           study / phase / variant / backbone / LoRA / seed
environment.json            Python、torch、Transformers、CUDA、driver、GPU、可见显存
resolved_config.yaml        真实运行的完整解析配置
lora_targets.json           注入层、Q/V路径、参数量、dtype、冻结/可训练清单
progress.json               committed updates、样本游标、PID、状态与退出码
training_summary.json       损失/学习率/最终checkpoint引用、hash、W0不变证据
source_summary.json         confusion、mIoU、2497覆盖及诊断统计/null计数
responsibility_summary.csv  slot / class / router分别命名
telemetry_summary.json      稳态利用率、吞吐、分项耗时、显存、定标选择
pca_manifest.json           在PCA阶段提供，连同basis引用和必要图
target_summary.json         只在最终目标阶段提供
SHA256SUMS                  返回文件校验及原始远程产物引用
```

运行manifest必须区分：原训练producer SHA、恢复/诊断/评估producer SHA、base权重身份、LoRA+head checkpoint身份、数据/增强合同。实例完成而尚未回传的证据应明确标为“远程已生成、待核对”，不能称为本机独立验收。

全量逐图诊断、dense特征、optimizer和checkpoint留AutoDL；本机返回分析所需JSON/CSV、小型矩阵/图及日志末尾即可。若需要独立重算，补充有界原始记录；不要求反复传完整run目录。

## 15. 最终报告与结项验收

最终`analysis/lora_causalq_1114/REPORT.md`至少包含：

- 28行主表：backbone、variant、LoRA、seed、源/目标mIoU、Anchor/Residual/Full raw与RMS-normalized方差、幅值、slot/class责任、参数量和显存/耗时。
- 所有第7.3节预声明对比，不删负结果；独立列出历史背景，披露batch、优化器和硬件的变化。
- 每类IoU和整数混淆矩阵重算核对；源2497、目标500及诊断五视图覆盖和缺失原因。
- W0冻结、LoRA98,304目标参数、Patch/CLS正确返回、梯度与恢复证据；如实际结构不同，说明差异及新版本合同。
- LoRA是否改善UA和CA；是否只改变幅值；Query责任是否塌缩或切换；B2/B3是否保留结构性零Residual的正确解释。
- 原始DINO与最终LoRA特征的共同PCA基底、图像对齐面板、定量漂移、解释方差和抽样身份；区分CLS、Patch与路由后特征。
- 每阶段GPU活动率和吞吐是否达到目标，训练/源诊断/目标推理分别披露；未达标如实写出，不以显存占用率替代算力指标。
- 单训练种子、历史目标反馈、D2 patch16转换和完整模型组合的归因限制。

模型验收的必要条件是工程正确、协议一致、覆盖完整、证据可追溯和结果真实；不要求LoRA必然提高mIoU或让Anchor方差小于Residual。若结果不支持假设，应正常结项并保留负结论，不自动加损失、改rank或追加目标调参。

## 16. 可直接交给 Codex 的启动指令

> 请以 `LoRA_CausalQ_1114_Plan.md` 为本轮唯一新研究指导书。先读项目必读文件并核对当前阶段和Git状态，保留全部旧研究记录。现在只开始Phase39：实现极轻量QV LoRA和两种DINO的Patch/CLS兼容接口，保持原CausalQ数学与默认API；使用CPU合成模型验证并运行全量pytest。不要在本机加载真实大权重或真实数据，不启动正式训练。完成后说明真实改动与测试，并交付下一阶段需要的AutoDL工程验证边界。后续每次只推进一个阶段，以真实完整Git SHA连接本机与AutoDL。

## 17. 本次方案文件的验证记录

本次仅新增本Markdown方案，未实现LoRA、未改既有代码/配置、未切换旧研究状态、未执行任何AutoDL实验。历史数据来自第1节所列仓库报告；HPI优化值已对照论文与固定版本官方配置。

本次验证：

- UTF-8、章节连续编号、Markdown代码围栏、无行尾空白、既有引用文件路径均已检查；28模型/总更新/样本暴露量与98,304个LoRA参数的计算一致。
- 本机全量测试：**754 passed，1 skipped，286.00秒**。跳过项为Windows不允许无特权符号链接的既有数据路径测试；没有删改测试。
- 实际本机解释器：`E:\Anaconda\envs\pytorch\python.exe`，Python3.10.20、torch2.10.0+cu128、pytest8.4.2；本次关闭GPU，只作既有CPU兼容测试，不代表本轮AutoDL环境已验收。项目声明的Python版本要求及实例环境须在Phase40单独满足并锁定。
- 执行命令：上述解释器运行 `-m pytest -p no:cacheprovider --basetemp=.test-tmp-strict-dg-lora1114-plan`；`PYTHONPATH`指向仓库`.test-deps-strict-dg`，`CUDA_VISIBLE_DEVICES`为空，OMP/MKL线程为1、MKL线程层为SEQUENTIAL。
- 未执行真实DINO加载、LoRA训练、GPU利用率测量或PCA提取；这些都是后续阶段的待验证工作。
