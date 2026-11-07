# CausalQ：HPI 去 VLM 骨架迁移与 Query 语义路由实验指导书

- 文件名：20261031_CausalQ_HPI_DINOv2_Plan.md
- 制定日期：2026-10-31（Asia/Shanghai）
- 修订版本：v3，2026-11-05，RTX 6000D/84GB 吞吐与 GPU 活动率验收版。
- 项目：https://github.com/Ealea1028/CausalQ_DG
- 研究标识：`hpi_dinov2_20261031`
- 建议开发分支：`codex/hpi-dinov2-20261031`
- 制定时本机代码基点：`fc0701aa461e87e685f6b2a57d96de6c89ac5ac9`。
- 状态：Phase33–36 已完成，当前执行 Phase37 的七组目标评估，继续使用冻结的 v3 batch32 双缓冲。
- 新设备：用户提供的 RTX 6000D / 84GB，22 核 CPU、110GB 内存；系统盘 30GB、数据盘初始 50GB，可扩容。

> 给接手 Codex 的执行指令：先粗读当前项目，再按本文件逐阶段实施。一次只推进一个阶段，每个新增机制分别提交。完成本机实现、CPU 测试、配置冻结和 GitHub 发布后，遇到真实权重、远程数据或 GPU 操作即停止，更新 `docs/AUTODL_NEXT.md`，给用户准确的 AutoDL 命令。用户回传结果后再推进下一阶段。不要只复述方案，也不要提前运行后续阶段或目标评估。

### 本次修订的执行口径

用户于2026-11-05明确选择：**保留原训练配方，增大骨干物理 batch，并加强多进程加载与异步预取**。每个模型有效训练 batch=1、两视图 CE 求和、AdamW/LR/40000更新、7组总280000正式更新均保留。DataLoader 的装载包、冻结骨干物理 batch 和每头有效 batch 是三个独立字段，不能混为一谈。

Phase33 的 `hardware_profile.json` 已实测：物理视图 batch32/64/128 分别约173.43/163.74/162.10视图每秒；12个spawn worker 已启用。因此 v3 不直接把64或128写成“更快”，改为按真实数据、共享骨干及全部头的完整流水线测量选择。当前 `feature_pipeline.py` 在窗口结束后等待消费流，下一窗口尚未与头更新真正重叠；声明两个缓存窗口不等于已实现异步双缓冲。

Phase34 性能修订和 Phase35 B1/B2/B3 均已验收。Phase36 复用已接受的权重、数据、环境和 v3 执行配置。运行 SHA `9243205c5da354e67c0998b35bb4f462276b8c7e` 对应已完成的 v2 基线。已运行任务保持其原身份，切换策略需从完整检查点进行显式恢复迁移，不在实例直接修改配置。

## 1. 本轮目标与边界

本轮将原冻结 DINOv3 ViT-L/16 替换成 **HPI 主结果 CLIP 分支使用的、由 DINOv2 ViT-L/14 权重适配而来的 DINOv2-L / patch16 骨干**，去掉全部 VLM 组件；保持 U0、UA 的其余科学配置与 Phase29–31 所用配方一致。

在新骨干上研究三个方向：

1. **弱语义定位**：给 Anchor 类别空间职责，避免要求 Anchor 单独完成完整精细分割。
2. **Query 引导的视觉特征注入**：让 Query 的更新进入主解码器所读取的视觉特征。
3. **预测效应不变性与正贡献约束**：约束关闭同一 Query 所造成的预测变化，并防止零效应塌缩。

本修订将高吞吐执行列为必须实现的工程目标：多进程成批准备数据、物理批次提取冻结骨干特征、同阶段多个独立头共享只读特征、跨窗口异步预取和批量诊断。保留各模型逻辑 batch=1 的优化配方；以端到端速度、数据等待和GPU活动率验收，不以分配更多显存或增加训练预算代替有效利用算力。

本轮不是 HPI 全模型复现，也不预先声称可达到其性能。保留现有 CausalQ 解码器、训练数据划分与训练预算，不同时引入 Mask2Former、VLM、文本提示、跨模型融合或新的骨干微调方法。

### 1.1 本机、GitHub、AutoDL 的分工

| 位置 | 执行内容 |
| --- | --- |
| 本机 Codex 工作区 | 阅读项目、编写模块和启动器、合成 CPU 测试、冻结配置/清单、分析回传结果、生成报告 |
| GitHub | 代码、配置、说明、版本化实验协议及精简结果元数据的唯一连接点 |
| 新 AutoDL 实例 | 环境安装、真实权重下载和转换、必要数据迁移、GPU 装载验证、正式训练、真实数据评估和显存测量 |
| 用户 | 在新实例执行交接命令，回传日志和精简导出包，提示 Codex 继续 |

约束：

- AutoDL 必须运行 Codex 交付的完整 40 位提交 SHA，不直接编辑源码或受控配置。
- 路径通过命令参数、环境变量或实例外部的运行记录传递，不在实例修改 YAML。
- 原始数据、预训练权重、检查点、全量逐图记录和完整运行目录不进入 Git。
- 本机不加载真实预训练权重，不在本机重做真实数据推理。
- 保留旧研究和用户未提交文件，不清理、覆盖或“顺手修复”无关工作。
- 用户已确认磁盘可以扩容；本方案按可扩容规划，不再把初始 50GB 当作不可改变的限制。扩容金额和实际容量仍由用户在 AutoDL 控制台操作。

## 2. 实施前只做一次项目粗读

必须先读：

- `README.md`、`AGENTS.md`、`docs/METHOD.md`、`docs/AUTODL_NEXT.md`。
- `analysis/diagnostic_final_report_20261028/REPORT.md`。
- `configs/strict_dg/formal/m0_seed0.yaml`。
- `configs/diagnostic_20261020/formal/ua_seed0.yaml`。
- `causalq/models/dinov3_wrapper.py`、`query_segmentor.py`、`baseline_segmentor.py`。
- `tools/train_formal_source.py`、`strict_source_runner.py`、`selective_cqe_source_update.py`、`deterministic_source_ce.py`。
- `causalq/analysis/selective_stability.py`、`diagnostic_summary.py`。
- `manifests/gta5_source_train.txt`、`gta5_source_val.txt`、`cityscapes_val500.json`。

先用 `git status --short` 判断用户改动；用 `rg` 定位相关实现，不全盘重读历史日志、旧压缩包或旧项目。

粗读必须得到以下结论：

- Phase28–32 的旧研究已结项，本轮是新的探索性研究。
- 旧 U0 是 M0 在后续诊断矩阵中的名称；其正式配置位于 Phase22 的 M0 YAML。UA 在 Phase29 训练，Phase31 做诊断。
- 旧 U0/UA 目标 mIoU 为 65.2091% / 66.0753%，只作历史背景，不能作为新骨干实验的基线结果复用。
- 现有非中心化结构为 `q[c,r] = a[c] + e[c,r]`，R=2。
- UA 是旧的 **Anchor 分量 CQE**，不是本方案新增的单 Query 预测效应损失。
- 旧诊断的 Residual Contribution 是 `E_full - E_anchor`，不是单独 residual 参数的因果效应。
- 现有主解码器先读取骨干特征；后续 Query 交互并未自动把更新送回该主解码器。

本机生成一份简短 `implementation_map.md`，记录复用点、修改点和历史配方来源。不要重跑旧阶段，不再设置 Anchor 原始方差必须小于 Residual 方差的训练门槛。

## 3. 骨架替换的精确定义

### 3.1 HPI 的真实实现差异

不能仅凭权重文件名推断运行时 patch size：

- HPI 主结果 CLIP 实现中，DINOv2 分支创建 `patch_size=16`、`embed_dim=1024`、`depth=24`、`num_heads=16` 的模型。
- 它加载 `dinov2_vitl14_pretrain.pth`，将 Patch 投影核由 14×14 插值至 16×16，并调整位置编码。
- HPI 的 EVA 实现保留 patch14。本轮不混用这两种实现。

**本轮选定 HPI-CLIP 路径的 DINOv2-L / patch16 适配版本**，配置名建议为 `dinov2_vitl14_hpi_p16`。这保留了原 512×512 训练裁剪、源域 pad16、目标域原生 1024×2048 输入，避免额外改变几何协议。

上游依据：

- [HPI CLIP 分支：骨干创建与权重插值](https://github.com/MosukFate/HPI/blob/6512993679a682422f82f675f4c6e9a2ccce4c9b/models/backbones/clip/models_hpi.py#L699)
- [HPI EVA 分支](https://github.com/MosukFate/HPI/blob/6512993679a682422f82f675f4c6e9a2ccce4c9b/models/backbones/eva_clip/eva_vit_model_hpi.py#L515)
- [HPI 的纯 DINO Transformer 实现](https://github.com/MosukFate/HPI/blob/6512993679a682422f82f675f4c6e9a2ccce4c9b/models/backbones/dino_v2.py)
- [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)

接手时若上游有更新，仍使用上述固定版本作为本轮参考；升级另立变更，不能随着 `main/master` 自动漂移。

### 3.2 保留与舍弃

| 组件 | 本轮处理 |
| --- | --- |
| DINOv2-L 视觉 Transformer | 保留，冻结 |
| 14→16 Patch 核与位置编码适配 | 按 HPI-CLIP 的确定性转换保留 |
| 现有 CausalQ 多层特征融合解码器 | 保留 |
| 现有 Query 读出、标量 alpha、R=2 聚合 | U0、UA 原样保留 |
| CLIP / EVA-CLIP 视觉与文本编码器 | 舍弃 |
| SHP 文本库、文本 Prompt、Context Decoder、Talk2DINO | 舍弃 |
| VLM↔DINO 跨分支融合 Adapter | 舍弃 |
| HPI 中依赖文本的分割头、文本投影 | 舍弃 |
| HPI 的 Query 读出/写回思想与空间门控 | 在 B2 独立实现为无文本模块 |
| Mask2Former、稀有类采样、HPI 训练/目标评估配方 | 本轮不引入 |

禁止通过实例完整克隆 HPI、安装其旧 MMCV 环境来实现本轮。应在本机整理必要的纯 PyTorch 骨干代码和许可证，提交到 CausalQ，使实例只拉取 CausalQ 发布快照。

### 3.3 骨干和权重契约

- 24 个 Transformer Block，宽度 1024，16 个 Attention Head，MLP ratio=4，无 register token，CLS 不进入密集特征图。
- 保留 DINOv2 自身的位置编码、LayerScale 和 LayerNorm 约定；不为了匹配接口改成 DINOv3 的位置编码。
- 模型构建 `img_size=512`、patch16；原权重中的空间位置网格插值为 32×32，CLS 位置向量不改。
- Patch 核使用 FP32 bicubic、`align_corners=False`，不额外做面积补偿、重归一化或随机初始化。
- 只允许转换 `pos_embed` 和 `patch_embed.proj.weight`；其他权重逐张量保持。
- 转换后 `strict=True` 加载。记录原文件 SHA256、转换配方、转换后文件 SHA256 和规范化 tensor-state digest。
- 冻结模型参数并保持 `eval()`；新路由模块放在骨干输出之后，允许骨干继续 `no_grad()`。
- 统一暴露 `hidden_size`、`patch_size`、`freeze`，以及 `patch_tokens/patch_map/intermediate_maps`。
- 保留原第 6/12/18/24 个 Block 后的特征读取点；显式映射为零基下标 5/11/17/23，不把 embedding 输出算一层。
- 中间特征是否经过最终 LayerNorm，应保持当前解码器的输入契约并写入冻结配置；当前旧 wrapper 的中间 hidden states 与最终 normalized patch tokens 必须区分。不能只因官方 DINOv2 接口默认 `norm=True` 就悄悄改掉全部中间特征。
- 旧 DINOv3 API 和旧配置继续可用；新增骨干选择器，不用重命名破坏旧类型。
- 可使用 PyTorch SDPA 等价实现减少注意力内存，必须保留原 qkv/投影/缩放/位置语义；用小张量 CPU 对照证明等价，不顺带添加 QK norm 或改变 Attention 温度。

## 4. 固定实验矩阵：7 组，而不是 8 组

研究内命名使用 `U0_D2P16`、`UA_D2P16`、`B1_D2P16` 至 `B5_D2P16`；论文表格可简写，但始终保留 backbone 字段。

**B0 就是 U0_D2P16 的别名，不另训一次。**

| 编号 | Query 参数化 | 特征注入 | 新语义定位 | 旧 Anchor CQE | 新预测效应不变性 | 正贡献目标 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 = U0_D2P16 | 非中心化 a+e，R=2 | 无 | 无 | 无 | 无 | 无 |
| UA_D2P16 | 与 U0 相同 | 无 | 无 | 原配方，权重1 | 无 | 无 |
| B1_D2P16 | 去掉可学习 e，重复 Anchor 保留 R=2 读出 | 无 | 无 | 无 | 无 | 无 |
| B2_D2P16 | 同 B1 | 单处 Query 路由 | 无 | 无 | 无 | 无 |
| B3_D2P16 | 同 B1 | 同 B2 | 弱类别空间定位 | 无 | 无 | 无 |
| B4_D2P16 | 同 B1 | 同 B2 | 同 B3 | 无 | 有 | 无 |
| B5_D2P16 | 同 B1 | 同 B2 | 同 B3 | 无 | 同 B4 | 有 |

主实验均 fresh seed0、40000 次优化器更新，共 **7×40000=280000 更新**。不从旧检查点微调，不从 B2 接着训练 B3，不把试运行状态当正式初始状态。

预先固定的对比：

- UA−U0：新骨干上旧 Anchor CQE 的效果。
- B1−U0：去掉可学习 Query Residual 的效果；包含预期的容量变化，不能解释成“证明风格特征有害”。
- B2−B1：Query 特征注入的整体效果。
- B3−B2：弱语义定位的效果。
- B4−B3：预测效应不变性的效果。
- B5−B4：正贡献目标的额外效果。
- B5−U0、B5−UA：完整新方案相对两条必要基线的变化。

不因中间源域方差排序不符合预期而删除后续组。失败仅指工程错误、非有限值、OOM、数据/配置身份不符等，负性能结果仍完整报告。

本轮默认一个种子以控制预算；不作显著性结论。多种子扩展需另行预先冻结，不能看目标结果后只给最高分方法补种子。

## 5. U0/UA 的科学配方与高吞吐执行契约

模型、损失、优化器、数据分布、每次更新的有效样本数和更新预算沿用旧配方。本次修订明确覆盖 v1 中 `num_workers=0`、冻结骨干两视图逐张执行、每个模型重复加载/前向骨干的保守执行限制；这些执行字段在新 U0/UA 及消融组中统一调整并记录。不能同时声称新运行 YAML 与旧配置逐字段相同。

| 项目 | 固定值 |
| --- | --- |
| 源数据 | GTA5，22469 train / 2497 holdout |
| 类别 | 19，ignore=255，沿用已有 trainIds |
| 训练输入 | 512×512，随机 scale [0.5,2.0]，水平翻转概率0.5 |
| Crop 策略 | min_valid_fraction=0.01，crop_attempts=10，沿用原实现 |
| 归一化 | 当前 ImageNet RGB mean/std，不换成 CLIP 归一化 |
| 视图 | original + photometric，几何一致；头的 CE/求梯度顺序保持，冻结骨干允许预先合批 |
| CE | 原图 CE + 1.0×干预图 CE，求和而非除以2 |
| Query | R=2，static，alpha_init=0，temperature=0.07，logsumexp |
| 中心化 | false；无中心化、正交、去相关硬约束 |
| 解码器 | channels=256，dropout=0.1，多层读取点6/12/18/24 |
| 优化器 | AdamW，lr=1e-4，weight_decay=0.01 |
| 调度 | poly power=0.9，绝对40000步；无额外 warmup |
| 梯度 | 两视图累积后 clip norm=1.0，再进行一次 optimizer.step |
| 逻辑训练批量 | 每模型每次 optimizer.step 仍为1个源图样本及其两个视图，accumulation=1 |
| 执行批量/加载器 | 已测基线为32个骨干视图、12个spawn worker；v3比较32/64/128，容量和瓶颈需要时扩至256，按端到端吞吐冻结 |
| DataLoader装载包 | v3默认8个源样本/包，每样本含2视图；仅用于装载与传输，头仍逐样本更新 |
| 预取 | persistent_workers=true、prefetch_factor=2；候选4；最多2个活跃GPU特征窗口，以CUDA事件管理复用 |
| 精度 | 冻结骨干 BF16；可训练权重与优化器状态保持 FP32，前向按原 BF16 AMP |
| 恢复 | 每1000更新原子保存；按绝对步数恢复 LR、采样器及全部 RNG |
| 检查点 | 固定最终40000步，不选 best |
| 源验证 | 最终完整2497张，长边1024、等比例 resize、pad16 |
| 目标评估 | 训练和源诊断全部冻结后，Cityscapes val500 原生1024×2048整图 BF16 |

干预参数保持：

```yaml
brightness: [0.7, 1.3]
contrast: [0.7, 1.3]
saturation: [0.7, 1.3]
gamma: [0.8, 1.2]
temperature: [0.9, 1.1]
grayscale_probability: 0.1
```

Fourier 参数可留在历史配置中，但本轮不启用 Fourier 训练。

UA 额外且仅额外包含：

```yaml
causal_query_effect:
  enabled: true
  lambda_cqe: 1.0
  effect_space: logits
  reference_stop_gradient: true
  normalization: l2_per_class_map
  present_classes_only: true
  valid_pixels_only: true
  loss: smooth_l1
  smooth_l1_beta: 1.0
  component: anchor
```

UA 继续使用“原图 Anchor 效应作 detached reference，干预侧 Anchor 路径求梯度”的旧更新语义。不能用 B4/B5 的损失替代它，不能在 UA 上额外叠加定位或注入。

旧代码中有研究标识、权重哈希和环境的硬编码保护。应新增明确支持本研究的入口或协议分支，保持旧数学及旧接口行为；不能修改旧研究常量、伪装成旧 study，或删除保护来绕过检查。

运行库因新显卡而改变时，建立新研究的 runtime lock，不再与旧 RTX4090D 的 environment 字典逐字段相等。新 U0/UA 之间必须使用同一环境；旧结果只作背景，不宣称硬件迁移后的逐位复现。

新 U0/UA 的初始可训练头必须逐张量相同。骨干和新模块构造用独立 RNG 作用域，避免其随机初始化改变公共头、采样、增强和 dropout 的随机序列。记录新头与历史头是否一致，不能伪造历史初始化哈希。

### 5.1 区分逻辑训练 batch 与冻结骨干的物理 batch

本轮优先使用以下流水线，而不是把7个实验全部改成 batch16 后仍训40000步：

1. CPU提前准备一段按绝对训练步编号的样本与干预参数。
   v3以8个源样本为一个DataLoader装载包，由worker完成几何、photometric和collate；同一包有16个视图。每个样本保留独立标签、绝对步号与参数摘要，包的维度不参与梯度归约。
2. 默认把16个后续逻辑步的 original/photo 共32个视图送入一次冻结 DINOv2 前向。
3. 将所需的4层特征和最终 token 保存在有限的 GPU 缓存窗口中。
4. 对窗口中的逻辑步逐个训练头：一个样本的两视图 CE 和对应辅助项，执行一次该模型自己的 AdamW 更新。
5. 消费窗口k时，预先提交窗口k+1的H2D和冻结骨干计算；最多两个活跃窗口。具体事件、缓存生命周期与验收见5.7，不能只在同一线程下一次迭代时才开始准备k+1。

骨干始终冻结、eval、无样本间归一化或训练态随机层，因而跨后续逻辑步提前提取特征在数学上不依赖尚未发生的头更新。批量 kernel、SDPA、编译可能带来浮点低位差异，必须记录；不声称与旧单图内核逐位一致。

这使 GPU 的重计算部分使用 batch32 甚至更大的物理批量，同时各模型仍是原来的 batch1/40000更新。若直接改成有效 batch16且仍40000步，源图曝光会变成16倍，不能再称为本轮的同配方比较；本轮不自动增加这条研究线。

训练用的特征生产器必须使用 `torch.no_grad()`。不要直接把 `inference_mode` 产生的特殊张量交给需要保存输入来计算权重梯度的头；如确需使用，必须在该作用域外复制为普通张量并计入复制成本。纯只读诊断/评估才可直接使用 `inference_mode`。

### 5.2 同阶段多个独立实验头共享一份冻结骨干

默认每个训练阶段使用一个协调进程和一个冻结骨干，按同一输入窗口推进下列模型组：

| 阶段 | 协同执行的独立实验头 | 共享范围 |
| --- | --- | --- |
| Phase34 | U0、UA | 原图/干预图的冻结骨干输出 |
| Phase35 | B1、B2、B3 | 同上 |
| Phase36 | B4、B5 | 同上 |

每组中的头、Anchor、Residual、路由、门控、优化器、调度器、梯度裁剪和 RNG 状态均独立。共享只发生在所有模型都不训练的骨干及其原始特征上；不得共享更新后的 P'、头 logits、可训练参数或 optimizer state，不是模型集成或联合损失训练。

具体执行为：对同一逻辑样本，分别运行各头自己的完整更新；不将多个模型的损失相加再做一次公共 backward，不给某头累积其他模型的梯度。路由模块不能原地修改共享 P；必要时创建私有更新张量。

在同一输入/骨干身份条件下，未优化执行的训练骨干图像前向量为 `7×40000×2=560000`；三组共享后是 `3×40000×2=240000`，减少约57%的重复骨干图像计算，再通过物理合批减少调用开销。这是计算量核算，不是已测得的墙钟加速承诺。优化器更新总数仍是280000。

头的逐样本 GPU 运算仍按确定顺序入队，不强制多个头并行backward。v3必须实现并有界比较“下一窗口骨干/H2D与当前窗口头更新重叠”，数值、RNG、生命周期与吞吐均通过后才冻结启用；若更慢则保留串行结果并报告瓶颈。多进程用于CPU DataLoader，不以多个独立CUDA训练进程重复加载/前向骨干来增加表面利用率。独立头的多流并行涉及dropout RNG隔离，首轮不作为默认实现。

### 5.3 输入流水线、RNG与恢复

- v3从已实测的12个CPU worker起步，有界比较8/12/16；prefetch_factor默认2、候选4，pin_memory=true、persistent_workers=true。按实际22核配额与等待比例选择，不盲目增加worker。
- DataLoader默认一次返回8个源样本的装载包，worker内collate；减少逐样本IPC、主进程拼接与反复pin_memory。包内及包间顺序保留，尾包不丢样本。DataLoader batch=8不等于模型有效batch=8。
- 进程加载器使用spawn上下文，worker只做CPU工作，启动后保持常驻；避免fork继承已初始化CUDA的状态。
- 主进程CPU计算线程默认4，每个worker的Torch/BLAS线程为1，避免多进程各自开满22线程。
- 骨干/特征缓存只放在GPU短窗口中，不把全部40000步的多层特征写到磁盘；固定缓存最多2个窗口。
- 数据流水线与解码缓存的总主机预算为24GiB，其中锁页内存不超过4GiB；预算是所有worker总和，不是每worker各分24GiB。
- 检查 `/dev/shm` 容量。进程加载器需要的共享内存不足时，采用线程解码加主进程显式变换的实现；不要在实例临时改大容器权限或让任务反复因共享内存不足崩溃。
- 数据传输使用 pinned buffer 与 `non_blocking=True`；由CUDA event保护缓存的生产/消费生命周期，不在每个小操作后 `cuda.synchronize()`。
- 采样ID顺序继续沿用 `randperm(seed+epoch)`。增强参数由“研究输入流版本、seed、绝对步数、sample ID、view”决定，不依赖worker数量、预取完成顺序或模型variant。
- 输入种子使用稳定SHA256派生并固定整数范围，不能使用进程间会变化的Python内置hash；具体算法写入协议并做恢复测试。
- 并行变换使用显式、局部 generator；不在线程中反复修改全局 Torch/NumPy RNG，也不允许 fork 的worker自行初始化CUDA。
- 所有7组使用相同的规范输入流，同一个逻辑步得到相同源图、crop/flip和photometric参数。输入随机性与各模型dropout、Query采样分离。
- 头的两视图更新仍遵守第5节和各损失的既定语义；B4/B5训练on/off保持共同随机数，不为了合批破坏dropout匹配。批量counterfactual优先用于dropout关闭的诊断。
- 本次显式输入流实现会改变对旧全局RNG调用次序的依赖；保持增强分布与新组间配对，记录此执行变化，不伪造旧逐步输入hash或声称历史轨迹逐位复现。

恢复窗口按绝对逻辑步对齐。若1000步检查点位于物理窗口中间，恢复时重建同一完整输入窗口，只跳过已经提交的头更新；不能因重分组改变剩余样本的增强，也不能重算已完成优化步。每个头有自己的恢复游标，某头已提交1000而另一头只提交0时，恢复不能让前者重复更新。

GPU日志按50逻辑步批量转到CPU。去掉每步对每个参数调用 `.item()`、逐参数梯度日志和整份冻结骨干的CPU哈希搬运；保留每次更新所需的梯度裁剪与非有限值保护。输入身份记录区分“ID/增强参数摘要”与“实际张量SHA”，不得把前者冒充后者。骨干身份在加载/恢复和最终关闭时核对，期间以冻结参数、优化器排除与版本计数检查保证只读。

每1000更新仍保存状态；可以用独立写线程写已经固定的CPU快照，禁止后台线程直接引用仍在改变的 `state_dict`。落盘与哈希完成后才写 complete 标记，最多保留一个待写快照，不能将尚未提交的状态当作可恢复检查点。

### 5.4 Phase33 已完成的硬件定标（历史契约）

下述是已验收的Phase33准备范围和当时模板，予以保留，不重跑。当前已冻结值为物理batch32、12个spawn worker、eager+SDPA、compile=false。v3需要的增量完整流水线测量按5.7执行，不能覆盖Phase33的原始 `hardware_profile.json`。

Phase33 的同一次 prepare 中完成，定标不读取目标数据、不比较mIoU、不进行正式优化更新：

- 骨干物理视图batch候选为8/16/32/64/128，默认32；候选按相同512×512输入、BF16和固定4层输出计时。
- 每候选2次warmup、5次计时；用CUDA event测GPU耗时，同时记端到端视图/秒。比较数据worker时另记录读取/等待时间。
- 选择实测最快且有余量的配置，不要求把候选全跑到OOM，也不以“最大可分配batch”为唯一依据。
- Phase33 骨干和缓存峰值先限制在实际可见显存的60%，给后续2–3个头、梯度与优化器留余量；完整任务峰值目标不超过实际可见显存的85%。以设备查询的bytes核算，不把84GB广告值直接当84GiB。
- 训练默认使用SDPA自动选择受支持的融合后端，保持原Attention数学；不能为提速关闭必要的确定性设置或悄悄改变温度。
- 冻结骨干的SDPA显式使用dropout_p=0.0、is_causal=false并保持原scale，不能认为仅调用eval就会替函数式Attention自动关闭dropout。
- 只对冻结骨干的固定512输入尝试一次 `torch.compile(mode="default", dynamic=False)`；首轮编译限时180秒，失败、超时、明显graph break或实测更慢就记录原因并用eager+SDPA，不让编译阻塞正式训练。
- 只读对照使用同一输入检查合批/编译与未编译路径的形状和数值误差，纳入原有骨干准备检查；提前冻结容差，不新增反复试跑。
- 编译/内核缓存放在数据盘并跨阶段复用。可训练头和带共同随机数的counterfactual路径首轮不编译，避免额外的随机性与重复编译问题。
- 不同shape的源域整图与目标原生整图使用eager+SDPA，避免每遇新尺寸重新编译；这些不是把训练分辨率调小。

硬件定标结果写入 `hardware_profile.json` 并回传；本机将其写入Git，所有后续组使用同一冻结执行配置。该选择仅依赖硬件吞吐/内存和数学正确性，不是模型选择。

下面的 `auto_once` 和候选选择仅由prepare动作执行。正式训练动作必须读取Git已冻结的实际批量、worker数、编译开关与内核策略，不在每个阶段重新自动择优或改变增强算法。

初始执行配置模板，实测字段由Phase33回传后冻结：

```yaml
profile: rtx6000d_throughput_v2
logical_batch_size: 1
optimizer_updates_per_variant: 40000
cohort_training: true
backbone_microbatch_views: 32
feature_queue_windows: 2
data_workers: 8
prefetch_factor: 2
pin_memory: true
persistent_workers: true
cpu_main_threads: 4
cpu_worker_threads: 1
host_pipeline_budget_gib: 24
pinned_memory_budget_gib: 4
prepare_backbone_memory_fraction: 0.60
total_gpu_memory_fraction: 0.85
attention_backend: sdpa_auto
compile_frozen_backbone: auto_once
compile_time_limit_seconds: 180
source_view_batch: 5
target_image_batch: 2
counterfactual_chunk: 4
metric_log_interval: 50
gpu_telemetry_interval_seconds: 2
```

PyTorch实现依据：[2.7数据加载器](https://docs.pytorch.org/docs/2.7/data.html)、[SDPA](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html)、[torch.compile](https://docs.pytorch.org/docs/2.7/generated/torch.compile.html)、[CUDA异步执行/流](https://docs.pytorch.org/docs/2.7/notes/cuda.html)。本节的模型组共享、预算和默认批量是本项目的工程设计，不是上述文档对该硬件的性能承诺。

### 5.5 源诊断、counterfactual与目标评估的批量化

- 源诊断默认把同一图像的5个风格视图合批提取特征，再供该阶段模型组的全部头消费；原图只计一次mIoU。
- 进一步合并多个图像时，按预处理后的精确H/W分桶。不能为了成批额外resize或补到别的尺寸：ViT的位置编码和padding参与的Attention可能因此改变。
- 128图主路径诊断在缓存特征上按4个关闭类别组成counterfactual小批次，只执行路由/门控/头；内存充足且计时更优可在prepare允许的范围内冻结为8。输入图像/视图维度和关闭类别维度分别记录，不能重复计数。
- Phase37 将7组头载入同一只读评估协调器，目标默认batch2个原生分辨率图像，提取一次骨干特征后分别执行7组头。输出仍是7份独立预测和混淆矩阵，禁止做概率平均或投票。
- 同一套500目标图像的骨干图像计算由3500次降为500次；这是只读计算复用，不能改动每组的配置重建、校验与覆盖。
- 原生目标batch的显存预估在Phase33使用合成1024×2048张量完成，不提前打开目标数据；最终读取目标数据仍只发生在Phase37。
- 源/目标评估可用1/2/4图像batch做有界内存和吞吐定标；同组7个头统一使用冻结策略。不能用改变滑窗、缩放或TTA来解决容量问题。

### 5.6 用实测吞吐验收，避免“满显存即满算力”

每2秒记录GPU活动率、显存、功耗，逐窗口记录数据等待/H2D/骨干/各头更新时间；报告：

- 实际物理骨干batch、各模型逻辑batch和各自优化更新数。
- 骨干视图/秒、模型组逻辑步/秒、各模型更新/秒、整个阶段墙钟时间。
- 每个独立实验头的前向/反向耗时和共享成本，不能把一个组的总吞吐冒充每个模型的吞吐。
- GPU活动率与功耗分布、峰值allocated/reserved显存、CPU/数据等待比例。
- 与短暂的单图eager执行基准相比的实测吞吐；不事先承诺固定倍数，也不声称NVML活动率等于Tensor Core利用率。

若持续低利用且数据等待占比较大，应先检查worker、磁盘、逐步CPU同步和输入预取；不要首先扩大科学训练batch。若合批/共享部分失败，只能在日志中明确记录临时单头回退及原因，不能把长期保守执行称为已完成高吞吐优化。运行正式模型后不得静默漂移硬件配置；变更通过新提交和明确的恢复迁移记录完成。

### 5.7 v3：完整流水线优化与GPU活动率验收

#### 5.7.1 已测事实与需要修改的实现

Phase33已确认的骨干单独计时如下；它不包含正式头更新、数据等待和检查点保存，不是完整训练吞吐：

| 物理视图batch | 视图/秒 | 骨干定标峰值reserved |
| --- | --- | --- |
| 32 | 173.43 | 约1.96GiB |
| 64 | 163.74 | 约3.25GiB |
| 128 | 162.10 | 约5.85GiB |

原始依据为Git中的 `experiments/HPI_DINOV2_20261031/hardware_profile.json`。单独增大骨干batch尚未带来加速；同样，配置中的12个worker已经是spawn多进程，不应再写成“当前没有开启多进程”。当前尚无完整正式训练GPU活动率的已接受统计，不能预先宣称其低于某个百分比。

本机实施重点固定为：

1. `input_stream.py`：新增装载包Dataset/collate接口，复用既有绝对步、ID顺序与局部generator；处理spawn/pickle、装载尾包和恢复窗口。
2. `feature_pipeline.py`：实现两个缓存槽位和可提前提交下一窗口的生产器，替换窗口末尾等待整个消费流的串行结构；输出完整时间戳和活跃槽位计数。
3. `cohort_runner.py` / `hpi_training.py`：保留每头逐样本优化，合并日志设备→CPU传输，移除逐条记录、逐窗口文件打开和非必要同步；统计每头更新速度。
4. 源诊断：按精确H/W分桶和有效内存预算批量执行只读头，最后一包覆盖不丢图；逐图19类统计不能改成混合图像的归约。

#### 5.7.2 多进程加载与传输

- 默认12个spawn worker，每worker Torch/BLAS线程为1，主进程最多4线程；worker不得构造模型或初始化CUDA。
- 装载包默认8个源样本，预取2包/worker；只有数据等待仍明显时，才比较预取4。包内 original/photo 共享几何；CPU预取使用绝对步编号，不能依赖完成顺序。
- worker完成crop/flip/photo与collate；主进程接收连续张量，使用pinned staging buffer与non_blocking H2D，避免每图单独cat、pin、传输。统计实际collate、主进程复制和pin耗时，不能把重新分配的普通CPU张量称作已锁页。
- 主机数据流水线总预算仍24GiB，锁页总预算仍4GiB；包含全部worker预取、装载包、staging buffer和待回收对象。按实际dtype/shape及队列深度核算，不仅乘worker个数。超预算的候选直接跳过。
- 检查共享内存；spawn共享内存不足时，显式记录原因并使用既有有界线程回退。不能将线程回退报告为多进程已启用。
- DataLoader自动装载批量只影响数据组织，送入头时拆成单个源图的两视图。各头每次optimizer.step仍只对应一个源样本，不做跨样本梯度累积。

依据：[PyTorch2.7 DataLoader](https://docs.pytorch.org/docs/2.7/data.html)；装载包大小、worker候选与内存预算是本项目的工程选择，不是文档对RTX6000D性能的承诺。

#### 5.7.3 真正的异步双缓冲

固定默认路径：CPU预取下一装载包 → copy stream异步H2D → 冻结骨干producer stream → 主消费流按样本/按头更新。窗口k+1可在窗口k的头更新期间生成；生成下一窗口无需等待当前头参数，因为骨干eval、no_grad且不含随机训练层。

- 每槽位有ready/consumed事件。消费者只等待所用窗口的ready；生产者只在复用同一槽位时等待其consumed，不能每窗口等待整个默认流。
- 使用wait_event和record_stream管理跨流依赖及缓存生命周期；CPU pinned buffer同样等对应H2D完成才复用。生产器启动时等待初始权重装载完成。
- 始终最多2个活跃GPU特征窗口，包括队列、调用方仍引用的旧窗口与尚未完成的CUDA操作；必须测试跨yield/恢复/异常退出，避免实际驻留3个窗口。
- 所有原始共享特征只读；梯度、优化器和dropout RNG仍由独立头管理。新增生产器不消耗训练dropout随机序列。
- 预取空队列、GPU内存上限、异常与正常结束均能关闭worker/线程并释放slot。显存上限保持实际可见显存的85%；禁止OOM后自动降低训练分辨率或修改有效batch。
- `cuda.synchronize()`只出现在必要的测量、日志收集、故障确认和提交边界；不能放在每次H2D、每个头或每个样本后破坏重叠。仍需保留有明确用途的数值保护，并说明其同步成本。

依据：[PyTorch2.7 CUDA流与内存生命周期](https://docs.pytorch.org/docs/2.7/notes/cuda.html)。是否实际加速由完整cohort实测决定，不能只凭增加stream对象认定已并行。

#### 5.7.4 有界选择更大的物理batch

1. 复用已验收的旧batch32/worker12/eager作为完整cohort对照；不重复权重转换、数据普查、compile试验或Phase33准备。
2. 比较多进程装载8/12/16 worker及预取2/4时，先做有界真实输入流测量，不运行所有参数的笛卡尔积。记录冷启动耗时，另取稳定区间；不能只用第一个耗时很长、后续队列已经填满的短样本判断总速度。
3. 骨干物理batch候选32/64/128；仅当当前瓶颈与显存预算支持时增加256。允许更大batch，但必须让完整训练更快。窗口大小随物理视图数变化，逻辑输入ID和增强不得变化。
4. 对完整cohort最多比较3个执行候选：旧串行基线、batch32的新双缓冲、经输入/容量测量选出的一个较大batch组合。每个候选预热2个完整物理窗口，再计时4个完整窗口，避免预热尚未填满队列就比较速度。每头全部候选的临时工程optimizer更新合计不超过1024；32/32/256三候选共960个tick/头，32/32/128则为576个tick/头。报告实际数量，数据ID均来自同一规范输入流。
5. 工程计时只用隔离的临时头/optimizer/RNG副本，从同一固定状态开始，执行真实两视图损失、反向、裁剪及optimizer.step。工程状态全部丢弃，不可推广为正式初始头或覆盖正式检查点。此作业0次正式更新、无mIoU和目标访问；临时工程更新与正式280000更新分别记账。
6. 最终按完整cohort源样本/秒及每头更新/秒选择，而非只按骨干视图/秒。根据多个完整计时窗口评估速度波动；速度差不超过5%的候选，优先较小内存与较少worker。GPU活动率作为诊断和验收指标，不独立决定配置。

`performance-check` 只在独立 v3 性能发布 SHA 执行；已完成的 v2 正式 SHA 不支持此动作。测量结果另存 `performance_v3.json`；回传验收后，新建对应v3执行profile和发布清单，保留原始硬件报告及Phase34旧冻结文件。

#### 5.7.5 日志、源诊断与工程验收

- 每50个逻辑tick将缓冲指标stack后批量D2H并写日志，CUDA事件在统一收集点读取；避免50步内每个历史记录各自`.cpu()`和`synchronize()`。保存、数据等待、GPU计算耗时分别报告。
- 每1000个正式更新仍原子提交。可用最多一个待写CPU快照的后台写线程；快照必须先固定，哈希/落盘完成才写complete。正式快照时间计入端到端吞吐，不能从报告中全部扣掉。
- 源诊断的5视图已共享骨干，但当前头逐视图执行且逐项转CPU；v3按预算批量执行只读头，在GPU上形成每图指标后分包回传。FP32效应/FP64累计、图像宏平均、原图mIoU只计一次及checkpoint身份保持。
- 测量GPU活动率每2秒采样；剔除初始化、编译、结束导出后，按持续训练稳态区间报告均值、中位数、P10/P90、低于50%的样本比例。区间至少120秒；有界工程计时不足120秒时延长遥测到正式运行，不扩大1024次临时更新预算，也不以少数2秒样本宣称稳定达到目标。
- 工程目标：稳态GPU活动率中位数达到80%，完整cohort吞吐相对旧执行方式提高至少10%，数据等待占墙钟时间低于10%，完整任务峰值reserved不超过实际显存85%。这些是待实测目标，不能写成已经通过；旧执行方式已满足某项时不要求重复增加无效计算。
- 若GPU活动率升高但吞吐下降，不接受该配置。若最快正确配置仍达不到目标，给出加载/CPU发射/头更新/骨干/保存的瓶颈占比，标记“目标未达成”并冻结可复现事实，不修改科学配方追求表面满载。
- 交付至少包括：逻辑batch=1、装载包大小、物理batch、真实loader模式、worker/prefetch、异步开关、工程与正式更新数、单头及cohort吞吐、数据等待、遥测分布和总墙钟时间。不得将骨干短基准或prepare全流程遥测称为正式训练利用率。

## 6. B1–B5 的模块定义

### 6.1 B1：删除可学习 Query Residual，保留读出形式

先按公共初始化规则创建相同的头，再把 `e[c,r]` 置零并冻结，或使用等价的不可训练零 buffer。两个读出槽位均为同一个 `a[c]`，继续执行原 R=2 logsumexp。

这样保留读出聚合中的 `log(2)` 常数和 alpha 语义，不同时改成 R=1 聚合。记录 `effective_independent_queries_per_class=1`。冻结零槽位要进入可恢复状态或可验证重建契约，不能被旧 runner 的状态集合检查误判为丢失参数。

B2–B5 的路由只使用 **19 个唯一类别 Anchor**，不把重复槽位当作38个独立 Query 导致注入翻倍。本轮不新增区域槽位、文本部件标签或不同 R 的搜索。

### 6.2 B2：一个无文本 Query 路由模块

位置固定在最终 normalized patch tokens 之后、主解码器之前。

设 P 为 N×1024 特征，Q 为19×1024类别 Anchor。采用8头、总路由维度256：

\[
A_h=\operatorname{Softmax}_{n}
\left(\frac{(LN(Q)W_{Q,h})(LN(P)W_{K,h})^\top}{\sqrt{32}}\right),
\quad U_h=A_h(PW_{V,h}),
\]
\[
\Delta P=\operatorname{Concat}_h(A_h^\top U_h)W_O.
\]

直接计算 `Aᵀ(AV)`，禁止显式构造 N×N 矩阵。值投影和输出投影不使用偏置，保证关闭全部路由时更新严格为零；新模块无 dropout。

首版门控固定为一种实现：

\[
g=\sigma\!\left(Conv_{3\times3}\bigl(GELU(Conv_{1\times1}[LN(P),LN(\Delta P)])\bigr)\right).
\]
\[
P'=P+\gamma g\odot\Delta P,\qquad \gamma=\sigma(b),\quad\gamma_{\mathrm{init}}=0.1.
\]

门控 bottleneck=32，输出每 Patch 一个标量。不要同时试 SAC/SCC、多种卷积组合或因果门控；本模块是受 HPI 启发的新设计，不声称逐模块复现 HPI。

主解码器的第24层输入替换为与最后层原输入规范一致的更新特征：

- 最终归一化 token 上计算更新。
- 向原第24层解码特征加同网格的更新；其余三层特征不变。
- Query 读出使用对应更新后的最终 token。
- 明确记录 raw intermediate 与 normalized token 的映射，不把某一路额外做 LN 当成隐含改动。
- `gamma` 和其他门控参数均不能全部零初始化；记录更新幅值，防止“模型运行了但注入分支没有梯度”。

所有可训练模块放在骨干 `no_grad()` 范围之外。主 CE 对公共头、Anchor、路由和门控正常求梯度。

### 6.3 B3：弱类别空间定位，不增加完整分割辅助头

对路由的 pre-softmax 匹配分数按 Head 求均值，再在类别维度 Softmax 得到 R[c,n]。不要直接把“在 Patch 维度归一化的 A”当作类别概率。

- 将像素 one-hot GT 以面积平均降到 Patch 网格。
- ignore/padding 不贡献面积；按每 Patch 的有效面积归一化，纯 ignore Patch 排除。
- 保留跨边界 Patch 的软类别占比。
- 损失为该软标签与 R 的交叉熵，先按有效 Patch 归约，再按图像归约。
- 原图和 photometric 各计算一次并取平均；`lambda_loc=0.1`。
- 梯度通过实际路由匹配路径进入 Anchor 与路由投影；不额外训练一个可绕过 Anchor 的大型分割辅助头。
- 类别存在性 BCE 可实现成独立选项，但本轮所有组 `lambda_presence=0`。这样 B3 只引入一种新监督，后续若要研究存在性监督需单独设计对照。
- 不将这种类别定位解释为已证明 Anchor 是纯本征特征。

### 6.4 B4：主路径的类别 Query 预测效应不变性

本轮的一个干预单元是“一个类别 Anchor 及其对应路径”，不是重复的两个虚拟槽位。对训练 GT 中存在的类别 c：

1. 保持输入、骨干特征、其他 Query 和随机 dropout 状态一致。
2. on：正常执行所有路径。
3. off,c：移除 c 的路由读出/写回贡献，并关闭原 Query 分类读出中 c 的 additive 分量；base 解码器保留，其输入随路由变化。
4. 重新计算后续门控与主解码器，测量完整运行路径的效应，不固定 full 模型的门控图冒充总效应。
5. 不把 `Q=0` 当作关闭操作，因为零向量仍可能产生非零的均匀注意力。

在 GT 区域 M_c 上定义：

\[
\tau_c^{(e)}
=\frac{1}{|M_c|}\sum_{n\in M_c}
\left[p_c^{on}(n,e)-p_c^{off,c}(n,e)\right].
\]

训练每张图均匀抽取一个 GT-present 类别，用独立 RNG；两风格使用同一个 c。对两个视图计算总体方差：

\[
\mathcal L_{\mathrm{inv}}
=\operatorname{Var}_{e\in\{original,photo\}}(\tau_c^{(e)}),
\quad \lambda_{\mathrm{inv}}=1.0.
\]

具体约定：

- CE 仍监督全部类别和全部有效像素。
- 训练时 off 分支 `stop_gradient=true`，on 分支两视图均可求梯度；这是明确声明的训练梯度代理，不声称等于完整 on−off 差值的全梯度。
- 诊断用实际 on/off 数值差，不混入训练的梯度约定。
- GT-present 类别等概率采样，权重不由模型学习，避免责任权重缩到零来逃避约束。
- Counterfactual 使用共同 dropout 随机数；额外 off forward 在独立 RNG 作用域执行，不消耗主训练随机序列。
- 原图和 photo 各只执行一次冻结骨干；on/off 共享其骨干特征。不能为每个 off 再运行整个骨干。
- 训练总目标为两视图 CE + B3定位项 + 本项。不要顺带加入旧 UA CQE 或预测 KL。
- 全部效应归约、Softmax 概率和方差计算使用 FP32；累计诊断使用 FP64。

### 6.5 B5：有停止点的正贡献目标

\[
\bar\tau_c=\tfrac12(\tau_c^{original}+\tau_c^{photo}),\qquad
\mathcal L_{\mathrm{use}}=[m-\bar\tau_c]_+,
\]
\[
m=0.02,\qquad\lambda_{\mathrm{use}}=0.1.
\]

只作用于 B4 已抽取的 GT-present 类别；与 B4 使用完全相同的 off 停梯度、采样和共同随机数协议。

- 不最大化绝对值，不奖励负贡献。
- 达到 margin 后不继续强迫放大。
- 同时报告 on/off 的正确类概率和负效应比例，检查效应增大是否仅来自 off 恶化。
- 上述权重与 margin 是本方案预先指定的首轮值，不是已验证最优值。不根据目标分数临时改动。

## 7. 结果标准与统一统计口径

### 7.1 每组必报

每个配置均输出：

- 源域 mIoU、目标域 mIoU，统一同时保存 [0,1] 与百分数显示；差值使用百分点。
- 19×19 整数混淆矩阵、19类 IoU、有效像素数、图像清单及覆盖计数。
- Anchor / Residual Contribution / Full 的跨风格原始方差、RMS归一化方差、平均绝对幅值、RMS。
- Query responsibility 均值、像素责任熵、使用熵、有效 Query 数、风格间 JS、责任变化与 top1 切换率。
- 精确配置、运行 SHA、权重身份、环境、总更新数、参数量、训练耗时与峰值显存。

目标域 mIoU 只在最后统一评估后填入；尚未评估时为 null，不能用历史 U0/UA 分数填充。

### 7.2 两种效应必须分开命名

**兼容旧诊断的读出效应：**

\[
E_F=\alpha\,\delta_Q,\quad E_A=\alpha\,\delta_{\mathrm{repeated\ anchor}},
\quad E_R=E_F-E_A.
\]

B2–B5 在相同的 routed feature 上计算这三项。它们只测 Query 读出贡献，不包含 Anchor 经路由改变 base 解码器的全部作用。不得将这组方差冠以“主路径总 Anchor 因果效应”。

**新增主路径干预效应：** 使用第6.4节的 \(\tau_c\)，另报均值、正/负/近零比例、跨风格方差和 on/off 概率。它是模型内部干预效应，不自动识别现实世界的因果因素。

另报 `injection_feature_update_rms`、`gamma`、门控均值及跨风格变化；不能把 \(\Delta P\) 改称旧 Query Residual。

### 7.3 原始方差与归一化方差

在同一图像、类别和固定有效像素集合 Ω 上，使用 original+4个冻结 photometric 视图：

\[
Var_{\rm raw}(E)=\frac1{|\Omega|}\sum_{n\in\Omega}
\frac1{5}\sum_e(E_e(n)-\bar E(n))^2,
\]
\[
RMS_e=\sqrt{\operatorname{mean}_{n\in\Omega}E_e(n)^2},\quad
\widetilde E_e=E_e/\max(RMS_e,10^{-6}).
\]

归一化方差用同样的总体方差公式计算。某一视图 RMS≤1e−6 时，归一化结果为 null 并记录近零原因，不能填零。

复用 Phase31 的确定性视图种子规则和原代码归约；采用新的独立研究元数据，不能修改旧研究文件以适配新模型。

主表报告 GT-present 类别的图像宏平均，并保留 all-class 补充、逐类结果和中位数。不要将图像宏平均与像素加权统计混合。

同时保留 Anchor/Residual 空间相关系数、跨风格协方差和相加抵消幅值，避免用单一归一化方差解释性能。

### 7.4 删除 Residual 后的统计解释

B1–B5 没有可学习 e：

- `E_R` 在结构上为零，原始方差、平均绝对幅值和 RMS 为0；记录 `residual_status=structurally_absent`。
- RMS归一化方差、与 Anchor 的相关系数为 null，不能除以 epsilon 后伪装成有效测量。
- 这不是“Residual 稳定性提升”，也不能与 U0/UA 的非零 Residual 当作同一学习结果比较。
- 旧 R=2 槽位责任恒为0.5，熵为 log(2)，有效独立 Query 数仍为1。标记 `duplicated_anchor_slots`，不能称为良好分工或高多样性。

为使所有组仍有有意义的责任分析，补充一种全组统一指标：

\[
\pi_{c,r}(n)=\operatorname{Softmax}_{(c,r)}s_{c,r}(n),\qquad
\pi_c(n)=\sum_r\pi_{c,r}(n).
\]

对19类的 \(\pi_c\) 统计类别责任、熵和跨风格 JS。它与旧“同类槽位内责任”分别输出，不能共用字段或直接比较熵值。另将路由 A 的 Patch 分配统计单独命名。

### 7.5 数据范围、几何与计算预算

- 源域 mIoU、读出效应统计和责任统计：全部2497张 source holdout，长边1024/pad16。
- 五视图诊断的 original forward 同时产生源域 mIoU，不再先做一次验证再重复原图诊断。
- 新主路径 on/off 干预诊断：预先按 seed20261031 从源 holdout 固定抽128张，所有配置用相同清单；评估其中每个 GT-present 类别与五视图。此项单独注明样本范围。
- Query 梯度全量诊断不作为本轮必跑门槛；若需补充解释，另行冻结小型源域样本，不自动重做旧全量梯度任务。
- 三组干预图：沿用旧固定随机抽样的18734、12488、08744；实施时先确认它们仍属于冻结 holdout。保存原图、实际输入、4个实际干预视图和参数/种子、接触表。只输出训练真实使用的方法；未用于训练的展示方法必须注明 diagnostic-only。
- 目标域：固定 Cityscapes val500，原生1024×2048、整图 BF16、argmax；无 resize、crop、sliding、flip、TTA。
- 目标域只计算最终分割指标，本轮不增加目标域风格诊断或用目标标签指导 Query。
- 所有7组最终状态冻结后统一评估，不据源方差排序跳过配置，不据目标结果改损失、选 checkpoint 或追加预算。

## 8. 数据与磁盘迁移方案

建议在新实例将数据盘扩至约200GB；这是一项容量规划，不是训练前的硬编码容量门槛。以旧实例必要目录的实际 `du` 加上环境、1份骨干缓存、7组头检查点、诊断和至少30GiB余量核算，空间充足则不必继续扩。

制定时旧 `diagnostic_disk_report.json` 记录整个 datasets 约74.8GB，包含旧实例的数据布局，不能当作精确的本轮迁移字节数。因此初始50GB不宜直接开始整批复制。

建议布局：

```text
/root/autodl-tmp/
  CausalQ_DG_hpi_<sha7>/        # 精简、只读代码快照
  envs/causalq-hpi/            # 新环境
  datasets/gta5/images/images/ # 24966张源图，维持旧 reader 布局
  datasets/gta5/labels_trainIds/
  datasets/cityscapes/         # 最终目标评估阶段才迁入val图像/标签
  pretrained/dinov2_vitl14/
  outputs/CausalQ_DG/hpi_dinov2_20261031/
  cache/                      # pip/torch等缓存
```

现在只迁移：

1. GTA5的 `images/images`。
2. 已接受的 `labels_trainIds`。
3. 如需历史结果分析，复用 Git 中精简摘要，无须迁移旧运行目录。

不迁移旧 DINOv3 权重、旧环境、GTA5 raw labels、数据压缩包的多份副本、旧中间检查点、完整 HPI 仓库或 VLM 权重。

目标阶段才迁移 Cityscapes 的 `leftImg8bit/val` 和 `gtFine/val`。不需要 Cityscapes train/test、BDD、Mapillary。旧实例的最终检查点与完整证据应留在原处或用户备份中，本方案不删除它们。

检查点只保存可训练状态、必要冻结头状态、optimizer/scheduler/RNG/sampler及骨干身份；冻结 DINOv2 权重单独共享一份。禁止每1000步重复写入整个约十亿字节级骨干。

## 9. 本机实施文件与阶段

以下是本轮接口契约。骨干、输入流、共享头协调器和 U0/UA 正式入口已在 Phase33–34 实现。Phase35 已完成 B1–B3，Phase36 发布 B4/B5。

建议文件：

- `causalq/models/dinov2_wrapper.py` 与必要的纯 PyTorch vendor 代码/许可证。
- `causalq/models/query_router.py`。
- `causalq/runtime/feature_pipeline.py`、`input_stream.py`、`cohort_runner.py`：有界冻结特征缓存、规范输入流、独立多头更新与恢复。
- `causalq/losses/query_localization.py`、`query_effect_invariance.py`。
- `causalq/protocols/hpi_dinov2_20261031.py`：本轮独立协议，不依赖旧接受报告的环境字典。
- `configs/hpi_dinov2_20261031/`：7份正式配置。
- `configs/hpi_dinov2_20261031/runtime_high_throughput.yaml`：受控执行参数；实际定标结果另存 `experiments/HPI_DINOV2_20261031/hardware_profile.json`。
- 新运行配置以受控runtime profile作为加载器与物理批量的唯一取值来源；不保留“YAML写worker0、实际运行worker8”的相互矛盾字段。旧配置仅作为科学配方来源和差异审计依据。
- `experiments/HPI_DINOV2_20261031/`：study freeze、runtime lock、转换配方、发布和结果摘要。
- `requirements/hpi_dinov2_cu128.txt`：精简环境依赖。
- `scripts/run_hpi_dinov2_20261031.sh`：统一后台启动器。
- `tools/run_hpi_dinov2_20261031.py`：prepare/train/source-report/target-eval/package等子命令。
- 必要公共运行逻辑移入 `causalq`，避免实例必须拉取整个旧 `tools` 或旧结果目录才能 import。
- 上述新模块各加聚焦 CPU 测试，并在本机运行完整 `python -m pytest`。
- 高吞吐执行单独提交，不与新的科学损失混在一次改动里。新增骨干可接受预提取特征的头入口，旧 `forward(images)` 保持行为兼容。
- CPU合成测试至少覆盖：共享特征不被原地修改、头之间梯度/优化器隔离、同一输入单独/协同更新等价、worker数不改变输入流、窗口中间及单头不同游标的恢复、批量诊断不丢图/不重复计数。
- v3补充覆盖：装载包尾包与物理窗口重组不改变绝对输入流；两个槽位跨yield始终有界；mock事件的生产/消费顺序、异常退出与worker关闭；日志合并不改变数值与更新计数。实际CUDA事件、异步生命周期、峰值内存和吞吐仍只在AutoDL验证，CPU测试不能冒充GPU并行已通过。
- 随机变换和干预模块新增可选显式generator/参数重放接口，旧默认调用保持兼容；不得通过改全局RNG实现并行预处理。
- 在 `.gitattributes` 为新增 Bash、配置、JSON及稀疏清单指定 LF；受控哈希基于 UTF-8/LF，避免 Windows 与 Linux 换行转换造成假不一致。
- `docs/AUTODL_NEXT.md`、`docs/METHOD.md` 和 README 只更新当前阶段入口，旧结果保留。

阶段安排：

| 阶段 | 本机工作 | AutoDL工作及停止点 |
| --- | --- | --- |
| Phase33（已完成） | 骨干/转换、输入流水线、共享骨干协调器、U0/UA配方与部署清单 | 已验收权重、环境及batch32/worker12定标，不重跑prepare |
| Phase34（已完成） | 保留已冻结科学配方；按5.7实现装载包、异步双缓冲、日志与诊断吞吐优化；独立提交并测试 | 一次有界性能验收后冻结v3执行profile；U0、UA各40000正式更新及联合源诊断 |
| Phase35（已完成） | 分别提交B1、B2、B3机制与测试，保持公共初始化 | B1/B2/B3共享冻结特征，各40000更新及联合源报告；不看目标 |
| Phase36（当前） | 分别提交B4、B5损失与测试，冻结on/off策略 | B4/B5共享冻结特征，各40000更新及联合源报告；不看目标 |
| Phase37 | 冻结7个最终checkpoint身份、所有配置和目标评估契约 | 一次最终目标评估7组，不再训练或重复已完成源诊断 |
| Phase38 | 汇总7组结果、预声明对比、限制与可视化 | 无GPU任务；结项 |

每个阶段结束就交接；不要把后续阶段的实施推迟到等待用户确认“是否继续”，也不要越过所需 AutoDL 结果自行假定通过。

Phase33准备已经完成。v3复用其已接受的架构、权重、环境、512形状与BF16依据，只对新增并行加载/双缓冲/日志路径做5.7的一次有界性能验收；隔离临时更新仅用于完整cohort计时，不进行mIoU、超参或模型选择。不重做独立8vs4+4恢复、全图标签普查、权重转换、compile和多轮readiness，也不未经完整流水线测量宣称已充分使用84GB显卡。

旧 deterministic CE 的 `fp32_log_softmax_gather_valid_mean_v1` 必须保留；不要重新引入此前在严格确定性下失败的 CUDA nll_loss2d 内核。

## 10. AutoDL 最小代码拉取：只取后续运行依赖

### 10.1 发布者必须先完成的工作

每次交接前，Codex 必须：

1. 提交当前阶段代码和受控元数据，发布到上述分支。
2. 提供该阶段真实 `EXPECTED_SHA`，不能用制定本方案的旧SHA代替。
3. 在 Git 中提供 `experiments/HPI_DINOV2_20261031/autodl_sparse_paths.txt`。
4. 用本机临时稀疏检出验证新入口的 import 依赖闭包，不读取旧 analysis、旧实验接受报告或未拉取 tools。
5. 更新当前阶段的实际命令和返回文件列表。

v3必须新增执行版本身份并保留旧Phase34冻结文件；不能仅在原 `runtime_high_throughput.yaml` 改worker/batch便绕过文件哈希。`performance-check` 发布前，本机先补齐实现、CPU测试、稀疏依赖闭包和真实SHA，随后才给出AutoDL命令。当前文档修订不构成新运行发布。

建议稀疏清单：

```text
/causalq/
/configs/hpi_dinov2_20261031/
/experiments/HPI_DINOV2_20261031/
/manifests/
/scripts/run_hpi_dinov2_20261031.sh
/tools/run_hpi_dinov2_20261031.py
/requirements/hpi_dinov2_cu128.txt
/pyproject.toml
/.gitattributes
/.gitignore
/AGENTS.md
/README.md
/docs/METHOD.md
/docs/AUTODL_NEXT.md
/20261031_CausalQ_HPI_DINOv2_Plan.md
```

如存在其他确实需要的单文件依赖，在本机更新清单；不改为全量拉取 analysis、artifacts、历史导出或所有实验日志。运行端无需完整 tests；完整测试由本机完成。

使用 Git 的 blob 过滤、浅克隆和 non-cone sparse-checkout，同时减少历史/文件传输和落盘。仅使用 sparse-checkout、先完整 clone 再删文件不能达到同样目的。[Git clone 文档](https://git-scm.com/docs/git-clone)、[稀疏检出文档](https://git-scm.com/docs/git-sparse-checkout)

### 10.2 在新实例执行的代码拉取模板

下面所有实例命令均包在子 Bash 中，避免 `set -e` 或 `exit` 关闭用户的交互终端。没有真实发布 SHA 时不要执行。

```bash
bash <<'BASH'
set -euo pipefail
EXPECTED_SHA=REPLACE_WITH_REAL_40_CHARACTER_RELEASE_SHA
RELEASE_REF=codex/hpi-dinov2-20261031
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo '请使用本阶段Codex交付的完整SHA'; exit 2;
}
CODE_DIR="/root/autodl-tmp/CausalQ_DG_hpi_${EXPECTED_SHA:0:7}"
FRESH_CLONE=0
if [[ ! -d "$CODE_DIR/.git" ]]; then
  test ! -e "$CODE_DIR"
  git clone --filter=blob:none --depth 1 --single-branch \
    --branch "$RELEASE_REF" --no-checkout \
    https://github.com/Ealea1028/CausalQ_DG.git "$CODE_DIR"
  FRESH_CLONE=1
fi
if [[ "$FRESH_CLONE" -eq 0 ]]; then
  test -z "$(git -C "$CODE_DIR" status --porcelain)"
fi
git -C "$CODE_DIR" fetch --depth 1 --filter=blob:none origin "$EXPECTED_SHA"
git -C "$CODE_DIR" show \
  "$EXPECTED_SHA:experiments/HPI_DINOV2_20261031/autodl_sparse_paths.txt" |
  git -C "$CODE_DIR" sparse-checkout set --no-cone --stdin
git -C "$CODE_DIR" checkout --detach "$EXPECTED_SHA"
test "$(git -C "$CODE_DIR" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$CODE_DIR" status --porcelain)"
du -sh "$CODE_DIR"
echo "CODE_DIR=$CODE_DIR"
BASH
echo '代码准备结束，终端保持打开。'
```

首次 `--no-checkout` 克隆后的工作树尚未展开，因此模板只对已有检出执行前置 clean 检查；正式展开后的 clean 检查始终保留。每个后续阶段都先用本节命令拉取新的发布 SHA，再运行该阶段动作。

## 11. 新实例环境、数据和权重命令

以下依赖于 Phase33 已发布的入口；各块中的 SHA 和路径应由 Codex 在实际交接时填好。

### 11.1 查看硬件与扩容后的空间

```bash
bash <<'BASH'
set -euo pipefail
nvidia-smi
df -hT /root/autodl-tmp
df -h /dev/shm
findmnt -T /root/autodl-tmp
mkdir -p /root/autodl-tmp/{datasets,pretrained,outputs,envs,cache}
BASH
echo '硬件与空间查看结束，终端保持打开。'
```

若控制台扩容尚未反映到 `df`，先按平台提供的方式完成挂载/扩容，不在不明确设备映射时自行格式化磁盘。

### 11.2 创建独立环境

优先保留旧训练配方的 PyTorch 主版本，使用支持新硬件的 CUDA12.8 wheel；不照搬 HPI 的 torch2.0.1/CUDA11.8/MMCV1.7.2 环境。

推荐首轮：Python3.12、torch2.7.0+cu128、torchvision0.22.0+cu128。实际 GPU compute capability、驱动和可用 kernel 在 Phase33 检查并写入 runtime lock；此处不从截图猜测具体 SM 号，也不保证旧环境可直接运行。

```bash
bash <<'BASH'
set -euo pipefail
EXPECTED_SHA=REPLACE_WITH_REAL_40_CHARACTER_RELEASE_SHA
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
CODE_DIR="/root/autodl-tmp/CausalQ_DG_hpi_${EXPECTED_SHA:0:7}"
ENV_DIR=/root/autodl-tmp/envs/causalq-hpi
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
export TORCH_HOME=/root/autodl-tmp/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/cache/torchinductor
export CUDA_CACHE_PATH=/root/autodl-tmp/cache/cuda
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create -y -p "$ENV_DIR" python=3.12 pip
fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r "$CODE_DIR/requirements/hpi_dinov2_cu128.txt"
"$PY" -m pip check
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())'
BASH
echo '环境准备结束，终端保持打开。'
```

精简 requirements 由本机实现者按实际 import 冻结，包含 numpy/Pillow/PyYAML 等真实运行依赖；不直接安装旧根 requirements 的全部研究/绘图工具，不下载 xformers、MMCV、VLM 或 Talk2DINO 来凑齐旧 HPI 依赖。

不向共享环境 editable-install 某一个 SHA 的代码目录，避免切换阶段后仍导入旧快照。启动器每次设定 `PYTHONPATH` 为当前精确代码目录，使用 `"$PY" -m tools.run_hpi_dinov2_20261031 ...`，并核对 `causalq.__file__` 位于本次检出内；子进程沿用同样约束。

版本安装依据：[PyTorch 历史版本命令](https://pytorch.org/get-started/previous-versions/)、[PyTorch2.7 对 Blackwell/CUDA12.8 的支持说明](https://pytorch.org/blog/pytorch-2-7/)。若新卡实际 kernel 不兼容，返回准备日志，由本机更新全组统一运行库，不能在实例随意升级或降级一组环境。

### 11.3 从旧实例迁移必要源数据

仅在旧实例仍可 SSH 访问时使用；由用户填实际地址和端口。不把 SSH 凭据写进 Git，不使用关闭主机验证的参数。

```bash
bash <<'BASH'
set -euo pipefail
OLD_SSH=root@REPLACE_WITH_OLD_AUTODL_HOST
OLD_SSH_PORT=REPLACE_WITH_OLD_SSH_PORT
[[ "$OLD_SSH" != *REPLACE* && "$OLD_SSH_PORT" =~ ^[0-9]+$ ]] || exit 2
NEW_DATA=/root/autodl-tmp/datasets
mkdir -p "$NEW_DATA/gta5/images/images" "$NEW_DATA/gta5/labels_trainIds"
rsync -a --partial --info=progress2 -e "ssh -p $OLD_SSH_PORT" \
  "$OLD_SSH:/root/autodl-tmp/datasets/gta5/images/images/" \
  "$NEW_DATA/gta5/images/images/"
rsync -a --partial --info=progress2 -e "ssh -p $OLD_SSH_PORT" \
  "$OLD_SSH:/root/autodl-tmp/datasets/gta5/labels_trainIds/" \
  "$NEW_DATA/gta5/labels_trainIds/"
du -sh "$NEW_DATA/gta5"
df -hT /root/autodl-tmp
BASH
echo '源数据迁移结束，终端保持打开。'
```

使用真实目录，不用软链接绕过旧 reader 的路径检查。迁移后按冻结 manifest 检查 ID 配对和数量，训练读取时检查标签范围；不重新设计 split，也不全量解码两万多张图来重复旧普查。

如旧实例已不可访问，用户可用持久存储或已有备份传入同样两个目录；不要虚构 AutoDL 内部迁移 CLI，也不要假定新实例已自动拥有旧数据。

### 11.4 仅下载一份必要预训练权重

```bash
bash <<'BASH'
set -euo pipefail
WEIGHT_DIR=/root/autodl-tmp/pretrained/dinov2_vitl14
WEIGHT="$WEIGHT_DIR/dinov2_vitl14_pretrain.pth"
mkdir -p "$WEIGHT_DIR"
if [[ ! -f "$WEIGHT" ]]; then
  curl -fL --retry 3 --retry-delay 2 --continue-at - \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth \
    -o "$WEIGHT.partial"
  mv "$WEIGHT.partial" "$WEIGHT"
fi
sha256sum "$WEIGHT"
ls -lh "$WEIGHT"
BASH
echo '权重下载结束，终端保持打开。'
```

第一次获得的真实权重 SHA 和转换结果由 Phase33 prepare 导出，回传后写入 Git 冻结，再发布正式训练 SHA。不能在文档里编造未知权重哈希，也不能让正式训练默默接受任意同名文件。

### 11.5 一次准备任务与回传

```bash
bash <<'BASH'
set -euo pipefail
EXPECTED_SHA=REPLACE_WITH_REAL_PHASE33_SHA
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
CODE_DIR="/root/autodl-tmp/CausalQ_DG_hpi_${EXPECTED_SHA:0:7}"
cd "$CODE_DIR"
export PY=/root/autodl-tmp/envs/causalq-hpi/bin/python
export CAUSALQ_DATA_ROOT=/root/autodl-tmp/datasets
export CAUSALQ_PRETRAINED_ROOT=/root/autodl-tmp/pretrained
export CAUSALQ_OUTPUT_ROOT=/root/autodl-tmp/outputs/CausalQ_DG/hpi_dinov2_20261031
export CAUSALQ_RUNTIME_PROFILE="$CODE_DIR/configs/hpi_dinov2_20261031/runtime_high_throughput.yaml"
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/cache/torchinductor
export CUDA_CACHE_PATH=/root/autodl-tmp/cache/cuda
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash scripts/run_hpi_dinov2_20261031.sh "$EXPECTED_SHA" prepare
BASH
echo '准备任务已提交，终端保持打开。'
```

`prepare` 应输出 runtime、官方/转换权重身份、source配对摘要、GPU形状/有限性/梯度/显存、hardware_profile、throughput_benchmark和精简 ZIP；不访问目标、不进行正式更新。硬件不支持可选编译时记录回退至eager+SDPA，数学/权重/数据出现明确错误才停止，由本机修复后用新SHA重试。

## 12. 正式启动、恢复、目标评估与交接

### 12.1 启动器的必须行为

`scripts/run_hpi_dinov2_20261031.sh EXPECTED_SHA ACTION` 的动作限定：

- `prepare`
- `train-baselines`：U0、UA作为一个共享骨干的独立头组。
- `train-b1-b3`：B1/B2/B3作为一个共享骨干的独立头组。
- `train-b4-b5`：B4/B5作为一个共享骨干的独立头组。
- `target-evaluation`：7组头共享只读骨干，需Git已冻结的七模型目标评估清单。
- `status`、`package`：只读状态/已有结果导出，不触发新推理。

长任务必须由启动器自身使用 nohup 或等价方式脱离终端，使用阶段级共享锁防止重复协调进程占GPU，立即返回PID、固定输出目录和launcher.log。协调进程内部允许本阶段2–3个独立头协同推进；不要把该锁误写成只允许实例驻留一个实验头。不要要求用户再手动套第二层nohup。

同 SHA 同 ACTION 可恢复：跳过完成组，从最后一个完整1000步状态恢复未完成组；保留 partial attempt，不能因重启而多计更新或选较好的 attempt。

恢复以各头自己的已提交游标为准；共享骨干的窗口可重建，但已提交头步不能重复执行。日志同时记录cohort tick及各头optimizer更新数，不把Phase34的40000个组时钟误记成仅40000次总更新。

输出示例布局：

```text
hpi_dinov2_20261031/
  phase33/PREPARE_<sha7>/
  phase34/BASELINES_<sha7>/
  phase35/B1_B3_<sha7>/
  phase36/B4_B5_<sha7>/
  phase37/TARGET_<sha7>/
```

每个目录包含 launcher.log、status.json、run_manifest.json、SHA256SUMS 和该阶段导出包。启动器日志必须打印最终实际目录，交接不靠用户猜时间戳。

另包含 `hardware_profile.json`、`throughput.jsonl`、`gpu_telemetry.csv`。显存报告区分模型组总峰值、共享骨干/缓存和各头参数/优化器占用，不把模型组全部显存算成每个头独占峰值。

### 12.2 正式训练命令模板

Phase34 使用 `ACTION=train-baselines`；Phase35/36 只替换为对应真实 SHA 和 ACTION。正式命令由 Codex 填写发布，不用未实现入口提前启动。

```bash
bash <<'BASH'
set -euo pipefail
EXPECTED_SHA=REPLACE_WITH_REAL_TRAINING_RELEASE_SHA
ACTION=train-baselines
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
CODE_DIR="/root/autodl-tmp/CausalQ_DG_hpi_${EXPECTED_SHA:0:7}"
cd "$CODE_DIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"
export PY=/root/autodl-tmp/envs/causalq-hpi/bin/python
export CAUSALQ_DATA_ROOT=/root/autodl-tmp/datasets
export CAUSALQ_PRETRAINED_ROOT=/root/autodl-tmp/pretrained
export CAUSALQ_OUTPUT_ROOT=/root/autodl-tmp/outputs/CausalQ_DG/hpi_dinov2_20261031
export CAUSALQ_RUNTIME_PROFILE="$CODE_DIR/configs/hpi_dinov2_20261031/runtime_high_throughput.yaml"
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/cache/torchinductor
export CUDA_CACHE_PATH=/root/autodl-tmp/cache/cuda
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash scripts/run_hpi_dinov2_20261031.sh "$EXPECTED_SHA" "$ACTION"
BASH
echo '正式任务已提交，终端保持打开。'
```

运行时强制加载该发布在Git中冻结的执行profile，按模型组共享骨干、合批提取和预取。逻辑batch仍为1；OOM不得自动改变逻辑batch、分辨率、精度或评估方式。v3的装载包大小、物理batch、worker、prefetch与异步策略由增量性能报告冻结为新执行版本；纯执行变更也必须新发布。若从已有正式状态迁移，另存旧/新SHA、原始配置身份、每头已提交游标及RNG/optimizer/输入流不变的迁移记录，旧检查点不可改写；启动器明确支持迁移后才执行，不隐式fresh重训或重计更新。

### 12.3 查看状态与返回文件

用户只需查看一次已启动任务，不通过重复启动探测状态：

```bash
bash <<'BASH'
OUT=REPLACE_WITH_EXACT_OUTPUT_DIRECTORY_PRINTED_BY_LAUNCHER
test -d "$OUT"
tail -n 80 "$OUT/launcher.log"
if [[ -f "$OUT/status.json" ]]; then cat "$OUT/status.json"; fi
if [[ -f "$OUT/export.zip" ]]; then
  ls -lh "$OUT/export.zip"
  sha256sum "$OUT/export.zip"
fi
BASH
echo '查看结束，终端保持打开。'
```

精简导出至少包含：

- config、metadata、formal_report、source_metrics、source_class_iou、19×19混淆矩阵。
- source_effect_summary、逐类效应表、readout责任与class责任表。
- B2–B5 的门控/注入幅值与128图主路径干预摘要。
- 数据覆盖轨迹、manifest身份、最终头checkpoint SHA、共享骨干身份。
- 新实例实际环境、峰值显存、耗时、失败状态、SHA256SUMS。
- 实际硬件profile、物理batch/worker、编译选择与回退原因、吞吐和GPU遥测摘要；完整遥测日志保留实例。
- 三组干预图单独一个小型图片包，避免每组重复导出相同图片。

回传摘要供本机分析；检查点与全量逐图明细留在实例。必要核对只做文件身份、覆盖、整数混淆矩阵归约和字段一致性，不再次加载 GPU 检查点重复推理。

### 12.4 最终目标阶段

Phase37 发布前，本机将7组最终头checkpoint SHA、训练配置、各组训练 SHA、骨干转换身份、源诊断和 Cityscapes500清单写入 `target_freeze.json`。目标评估使用新的只读运行 SHA，不能要求所有历史训练组具有同一 producer SHA；通过各组配置及已冻结身份重建。

此时才执行 Cityscapes val 的迁移：

```bash
bash <<'BASH'
set -euo pipefail
OLD_SSH=root@REPLACE_WITH_OLD_AUTODL_HOST
OLD_SSH_PORT=REPLACE_WITH_OLD_SSH_PORT
[[ "$OLD_SSH" != *REPLACE* && "$OLD_SSH_PORT" =~ ^[0-9]+$ ]] || exit 2
CITY=/root/autodl-tmp/datasets/cityscapes
mkdir -p "$CITY/leftImg8bit/val" "$CITY/gtFine/val"
rsync -a --partial --info=progress2 -e "ssh -p $OLD_SSH_PORT" \
  "$OLD_SSH:/root/autodl-tmp/datasets/cityscapes/leftImg8bit/val/" \
  "$CITY/leftImg8bit/val/"
rsync -a --partial --info=progress2 -e "ssh -p $OLD_SSH_PORT" \
  "$OLD_SSH:/root/autodl-tmp/datasets/cityscapes/gtFine/val/" \
  "$CITY/gtFine/val/"
BASH
echo '最终评估数据迁移结束，终端保持打开。'
```

随后使用第12.2节模板，填Phase37真实SHA和 `ACTION=target-evaluation`；该动作不含optimizer，不训练、不选模型。默认目标图像batch2，所有7个独立头共享每张图的冻结特征，报告全部7组，禁止做集成。

## 13. 最终报告与结项条件

最终 Markdown/CSV 至少包含下表字段：

| 模型 | backbone | seed | 参数量 | source mIoU% | target mIoU% | Anchor raw Var | Anchor RMS-norm Var | Residual raw Var | Residual RMS-norm Var | slot责任/熵 | class责任/熵与JS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0 | DINOv2-L HPI-p16 | 0 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| UA | 同上 | 0 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 | 实测 |
| B1–B5 | 同上 | 0 | 各组实测 | 各组实测 | 各组实测 | 各组实测 | 各组实测 | 结构性0 | null/无Residual | 重复槽位退化 | 各组实测 |

另附：

- 第4节全部预声明对比，包含负结果，不只列最高分。
- 平均绝对幅值/RMS、相关与抵消、主路径干预、门控幅值、逐类 IoU。
- GPU/依赖/耗时/峰值显存，恢复与失败记录。
- 硬件利用报告：逻辑训练batch与物理骨干batch分别列示，模型组共享的计算量、真实吞吐、数据等待、编译成本/回退、GPU活动率与功耗，不以满显存代替算力利用证据。
- 旧 DINOv3 U0/UA 作为另表历史背景，明确其硬件和骨干不同。
- 单种子、已知历史目标反馈、无 VLM、未复现 HPI 全模型等限制。

解释边界：

- 读出方差下降不充分证明泛化改善。
- Query 被移除后变化大，不充分证明它是本征因果因素。
- Attention 可视化不单独构成因果解释。
- B2–B5 的收益可归因于整个新增模块/损失组合；若要主张“收益只来自 Anchor 参数学习”，需要另立冻结其他权重及缓冲区、仅更新 Anchor 的实验，本轮不自动追加。
- 不将 B1–B5 的结构性零 Residual 或重复槽位高熵包装成解耦证据。

结项条件：7组完成40000步，源域标准统计完整，冻结后的7组目标评估完成，全部对比及限制已报告。结项后停止，不根据目标结果自动追加新机制。

## 14. 当前交接

**本轮只执行 Phase37。** Phase36 两个 ZIP、32 个文件、两头各 40000 更新及完整源域诊断已验收。B4/B5 源域 mIoU 为 72.4607%、72.4340%。 Phase35 主 ZIP、33 个文件、40000 个配对训练步、2497 张源域覆盖及 B2/B3 的 128 图主路径诊断均已核验。B1/B2/B3 的源域 mIoU 为 72.4013%、72.4495%、72.4771%，目标数据未访问。

AutoDL 使用 [当前交接 SHA 和命令](docs/AUTODL_NEXT.md) 运行 `target-evaluation`。七组固定最终头按已冻结清单评估 Cityscapes val500，原生 1024×2048 整图 BF16，每批两张图共享骨干特征。本阶段新增训练更新为 0。回传后进入 Phase38 汇总。

## 15. 本文件交付时的验证记录

### v3修订 — 2026-11-05

本次仅修改本方案、README、`docs/METHOD.md` 和 `docs/AUTODL_NEXT.md` 四份文档，没有更改可执行代码、已冻结的训练YAML、Phase33实测profile或Phase34冻结元数据。用户明确保留有效batch=1及原正式更新预算；v3性能目标与待实现内容分别标注。

UTF-8/LF、代码围栏、9段Bash语法、3段YAML解析、32/64/128实测吞吐引用及临时更新预算计算核对通过。运行项目独立环境的完整 `python -m pytest`：**729 passed、1 skipped、14 warnings，136.09秒**。跳过项为Windows非特权符号链接限制。

解释器为 `.venv/Scripts/python.exe`，Python3.12.4、PyTorch2.7.0+cpu；命令为 `python -m pytest --basetemp=.test-tmp-strict-dg-plan-v3`，CUDA_VISIBLE_DEVICES为空，OMP/MKL/OPENBLAS线程为1，MPLBACKEND=Agg。日志保留在本机忽略目录 `.test-tmp-strict-dg-hpi-validation/plan_v3_full.log`。该结果不代表v3异步执行、GPU活动率或加速目标已通过；这些仍需新增实现与AutoDL增量性能验收。

### v2历史交付记录

v2修订仅修改本指导书，没有修改模型或运行入口。UTF-8、代码围栏、9段Bash语法、3段YAML解析及高吞吐profile的一致性检查通过；原版本已在隔离小型Git仓库验证“no-checkout→按清单稀疏展开→detached SHA→clean”的顺序，并确认无关根文件及analysis未落盘。共享头、实际吞吐和GPU显存目前是待实现/实测的要求，不能记为本轮已测通过。

v2修订后再次执行当前项目的完整 `python -m pytest`，默认解释器为 `E:/Anaconda/python.exe`，仍因缺少 `torch` 在测试收集阶段失败（66项收集错误），不能记为通过。接手实施时应先准备符合项目要求的本机CPU测试环境，再运行测试；本轮不为文档交付修改用户的全局Python环境，也不把环境错误改写成模型测试结论。

