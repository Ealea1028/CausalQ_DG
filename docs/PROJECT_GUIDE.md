# CausalQ-DG 项目完整指导书

> 面向首次接触本项目的研究者、复现实验人员与后续开发者。本文说明项目要解决的问题、当前实现、实验流程、已验证结论、创新边界和下一步路线。

## 1. 文档状态与阅读约定

- 当前基准日期：2026-09-24。
- 当前 Phase 12 机制分析提交：`07ef4ce71d04cac432259f10d05693cdcecfdca8`。
- 当前已完成主实验：A0–A5。
- Phase 11 风格消融和 Phase 12 Query 数量消融均已完成。
- 本文中的“已实现”“已验证”只指已进入 Git 并有实验证据的内容。
- 本文中的“建议”“下一版”“拟议”均不是当前实现或现有实验结论。

项目最初的完整设想见 [`CausalQ_DG_Project_Plan.md`](../CausalQ_DG_Project_Plan.md)，当前实现定义见 [`METHOD.md`](METHOD.md)，下一次 AutoDL 操作以 [`AUTODL_NEXT.md`](AUTODL_NEXT.md) 为唯一执行入口。

## 2. 一句话理解本项目

CausalQ-DG 研究的是：在只用合成驾驶场景训练、直接迁移到未见过的真实城市时，能否让一组按语义类别组织的 Query 从冻结的 DINOv3 特征中提取更稳定的类别证据，并通过保持几何与标签不变的风格干预来提高语义分割的域泛化能力。

当前证据支持两点：

1. 类别分组、单向读取图像特征的 Query 分支有效；
2. photometric 与 Fourier 风格视图联合训练有效。

当前证据不支持“现有 CQE 定义能够提高目标域精度”：它显著压低了效应方差，却同时降低了分割性能。因此，现阶段最强方法是 A2（Query + Style），而不是 A4/A5。

## 3. 研究问题与实验边界

### 3.1 任务定义

任务是单源、无目标域训练数据的域泛化语义分割：

- 源域：GTA5；
- 主要验证目标域：Cityscapes validation；
- 计划中的扩展目标域：BDD100K、Mapillary；
- 类别空间：Cityscapes 19 类；
- 忽略标签：255；
- 主指标：目标域 mIoU。

训练阶段不能使用目标域图像、标签或统计量调参。Cityscapes 在当前工程中承担阶段验证角色，因此所有结论都应明确写成“GTA5 → Cityscapes”，不能泛化宣称已经覆盖 BDD100K 或 Mapillary。

### 3.2 为什么选择冻结的 DINOv3

冻结主干有三个目的：

- 将研究重点放在 Query 机制和训练约束，而不是大规模主干微调；
- 在 RTX 4090D 24 GB 上控制显存与实验成本；
- 让 A0–A5 的差异更容易归因于新增模块。

当前主实验采用 DINOv3 ViT-L/16，读取第 6、12、18、24 层特征。主干通过 Hugging Face Transformers 加载，权重固定并用 SHA-256 校验。

## 4. 两层因果叙事与表述边界

### 4.1 数据生成层

当前工作假设可简化为：

```text
D -> S -> X <- C -> Y
```

其中：

- `D`：域或环境；
- `S`：风格，如亮度、色调、纹理频谱；
- `C`：语义内容与场景结构；
- `X`：观测图像；
- `Y`：像素级语义标签。

有效的风格干预应改变 `S`，同时尽量保持 `C`、几何位置和 `Y` 不变。项目使用的 photometric 与 Fourier 变换是这种干预的代理，不等于真实世界中对风格变量的完美因果操纵。

### 4.2 模型内部机制层

模型变量包括冻结特征 `F`、类别 Query `Q`、基础预测和 Query 残差。Query 只读取图像 token，不反向写入基础特征，因此其输出贡献能够从总 logits 中结构化分离。

当前代码把关闭某类 Query 残差后的 logits 作为零消融反事实。这个差分是“模型内部、结构定义的零消融效应代理”，不能直接称为真实数据生成过程中的因果效应。

### 4.3 可以和不可以声称什么

可以声称：

- Query 分支对 logits 的增量可显式分离与消融；
- 风格变换保持空间对齐，可比较不同视图下的预测或 Query 贡献；
- CQE 明显降低了归一化 Query 效应图的跨风格方差；
- 当前 CQE 同时降低了 Cityscapes mIoU，说明“稳定”本身不足以保证“有用”。

不可以声称：

- 已识别真实世界的因果效应；
- 风格代理覆盖了所有域偏移；
- A4/A5 已经优于普通一致性或 A2；
- 尚未运行的 BDD100K、Mapillary 或多种子结果已经成立。

## 5. 当前模型结构

### 5.1 总体数据流

```text
输入图像 X
   |
   +--> 冻结 DINOv3 ViT-L/16 --> 多层 patch 特征 --> 基础解码器 --> L_base
                                  |
                                  +--> 图像 tokens
                                         |
类别分组 Query Bank ---------------------+--> 单向 Cross-Attention --> Q_ctx
                                                                       |
像素特征 --------------------------------------------------------------+--> Delta L_query

最终 logits：L = L_base + alpha * Delta L_query
```

### 5.2 冻结主干与基础解码器

`DINOv3Backbone`：

- 支持 ViT-B/16 与 ViT-L/16；
- 从模型元数据动态识别 prefix token 数量；
- 去掉 prefix token 后恢复二维 patch 网格；
- 输出多层 BCHW 特征及最终 patch tokens；
- 主实验中参数冻结。

`BaselineDecoder`：

- 拼接同分辨率的多层 ViT 特征；
- 使用 `1×1 Conv + GroupNorm + GELU` 融合；
- 再使用 `3×3 Conv + GroupNorm + GELU + Dropout`；
- 输出 19 类低分辨率 logits，再上采样至标签大小。

关键实现（[`causalq/models/dinov3_wrapper.py`](../causalq/models/dinov3_wrapper.py)）如下。这里最重要的不是 Hugging Face 的调用本身，而是冻结时使用 `no_grad`、动态去除 prefix tokens，并把 patch token 恢复成二维特征图：

```python
def forward(self, images: Tensor) -> DINOv3Features:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")

    grid_size = (
        images.shape[-2] // self.patch_size[0],
        images.shape[-1] // self.patch_size[1],
    )
    context = torch.no_grad() if self.freeze else nullcontext()
    with context:
        outputs = self.model(
            pixel_values=images,
            output_hidden_states=bool(self.intermediate_indices),
            return_dict=True,
        )

    patch_map, patch_tokens = self._to_patch_map(
        outputs.last_hidden_state, grid_size
    )
    return DINOv3Features(
        patch_map=patch_map,
        patch_tokens=patch_tokens,
        intermediate_maps=tuple(
            self._to_patch_map(outputs.hidden_states[i], grid_size)[0]
            for i in self.intermediate_indices
        ),
    )
```

基础解码器的实际融合入口（[`causalq/models/baseline_segmentor.py`](../causalq/models/baseline_segmentor.py)）很小，这保证 A0 的可解释性：

```python
class BaselineDecoder(nn.Module):
    def forward(self, feature_maps: Sequence[Tensor]) -> Tensor:
        fused = torch.cat(tuple(feature_maps), dim=1)
        return self.classifier(self.fuse(fused))


def forward(self, images: Tensor) -> Tensor:
    features = self.backbone(images)
    maps = features.intermediate_maps or (features.patch_map,)
    logits = self.decoder(maps)
    return F.interpolate(
        logits, size=images.shape[-2:], mode="bilinear", align_corners=False
    )
```

### 5.3 Grouped Causal Query Bank

每个类别维护一个共享锚点和多个可学习残差：

```text
q[c, r] = anchor[c] + residual[c, r]
```

当前设置：

- 类别数 `C = 19`；
- 每类 Query 数 `R = 3`；
- Query 总数 `57`。

这种组织方式让 Query 具有明确的类别归属，同时允许同一类别表示多个上下文或外观模式。

对应源码（[`causalq/models/query_segmentor.py`](../causalq/models/query_segmentor.py)）：

```python
class GroupedCausalQueryBank(nn.Module):
    def __init__(self, hidden_size: int, *, num_classes=19,
                 queries_per_class=3):
        super().__init__()
        self.class_anchors = nn.Parameter(
            torch.empty(num_classes, hidden_size)
        )
        self.query_residuals = nn.Parameter(
            torch.empty(num_classes, queries_per_class, hidden_size)
        )
        nn.init.normal_(self.class_anchors, std=0.02)
        nn.init.normal_(self.query_residuals, std=0.02)

    def forward(self, batch_size: int) -> Tensor:
        queries = self.class_anchors[:, None, :] + self.query_residuals
        return queries.unsqueeze(0).expand(batch_size, -1, -1, -1)
```

返回张量形状是 `B × C × R × D`。`C=19`、`R=3` 时，一张样本有 57 个 Query；Phase 12 只改变 `R`，其余协议保持不变。

### 5.4 单向 Query-to-Image Cross-Attention

Query 作为查询，图像 token 作为 key/value：

```text
Q_ctx = CrossAttention(Q, F_img, F_img)
```

当前采用 1 层、8 个注意力头，并接残差、归一化与前馈网络。关键结构约束是单向读取：图像特征不会被 Query 写回修改。这既控制实现复杂度，也保留基础分支与 Query 分支的可分离性。

核心层的真实调用明确展示了方向（`queries` 是 query，`image_tokens` 同时是 key/value）：

```python
def forward(self, queries: Tensor, image_tokens: Tensor) -> Tensor:
    attended, _ = self.attention(
        queries,
        image_tokens,
        image_tokens,
        need_weights=False,
    )
    queries = self.attention_norm(queries + attended)
    return self.ffn_norm(queries + self.ffn(queries))
```

模型中先把 `B×C×R×D` 展平为 `B×(C·R)×D`，注意力完成后再恢复类别和组维度：

```python
grouped_queries = self.query_bank(images.shape[0])
flat_queries = grouped_queries.flatten(1, 2)
contextual_queries = self.query_attention(
    flat_queries, features.patch_tokens
).reshape(
    images.shape[0], self.num_classes, self.queries_per_class,
    self.backbone.hidden_size,
)
```

### 5.5 Query Residual Head

像素特征和上下文化 Query 分别投影并归一化，计算温度缩放相似度；同一类别的 3 个 Query 用 `logsumexp` 聚合，形成每类残差图 `Delta L_query`。

最终输出为：

```text
L = L_base + alpha * Delta L_query
```

其中 `alpha` 是可学习标量，初始化为 0。这样 A1 在初始化时与 A0 完全一致，Query 分支只在训练中逐步获得影响力。

残差头的关键计算（同一源文件）为：

```python
pixel_features = F.normalize(
    self.feature_projection(image_tokens), dim=-1
)
query_features = F.normalize(
    self.query_projection(grouped_queries), dim=-1
)
scores = torch.einsum(
    "bnd,bcrd->bcrn", pixel_features, query_features
) / self.temperature
delta = torch.logsumexp(scores, dim=2)
return delta.reshape(batch_size, self.num_classes, *grid_size)
```

这段代码说明三个设计点：像素与 Query 先做 L2 归一化；温度 `0.07` 控制相似度尖锐程度；`logsumexp` 在同一类别的 `R` 个 Query 上聚合，而不是把不同类别混在一起。

## 6. 当前零消融与效应定义

对类别 `c` 做零消融时，代码将该类别的 Query 残差通道替换为基础 logits：

```text
L_factual[c] = L_base[c] + alpha * Delta L_query[c]
L_zero[c]    = L_base[c]
E_zero[c]    = L_factual[c] - L_zero[c]
             = alpha * Delta L_query[c]
```

这个实现有两个优点：

- 不需要重新运行 DINOv3；
- 消融量与 Query 分支在结构上完全对应。

对应的公共 API 是：

```python
def counterfactual_logits(self, class_index: int) -> Tensor:
    if not 0 <= class_index < self.logits.shape[1]:
        raise IndexError(f"class_index out of range: {class_index}")
    result = self.logits.clone()
    result[:, class_index] = self.base_logits[:, class_index]
    return result


def get_query_effect(
    self, images: Tensor | None = None, *,
    output: QuerySegmentorOutput | None = None,
) -> Tensor:
    if (images is None) == (output is None):
        raise ValueError("Provide exactly one of images or output")
    if output is None:
        output = self.forward_components(images)
    return output.scaled_delta_logits
```

`counterfactual_logits` 直接赋值为 `base_logits`，而不是用减法近似，避免浮点抵消误差。训练和分析代码通常直接读取 `scaled_delta_logits`，因此效应定义不会因为重复前向而改变。

但它也有一个已经被实验暴露出的缺点：让 `alpha * Delta L_query` 变小，同样可以让跨风格效应更“稳定”。因此，单独约束零消融效应一致性存在退化解，不能保证 Query 对正确预测具有充分贡献。

## 7. 风格干预

同一张源域图像产生空间对齐的视图：

1. `original`：原图；
2. `photometric`：亮度、对比度、饱和度、gamma、色温与低概率灰度变换；
3. `fourier`：混合低频幅度，保留相位以尽量维持几何与语义结构。

当前范围：

- 亮度、对比度、饱和度：`[0.7, 1.3]`；
- gamma：`[0.8, 1.2]`；
- 色温：`[0.9, 1.1]`；
- 灰度概率：`0.1`；
- Fourier mix strength：`[0.1, 0.35]`。

训练时三个视图顺序前向以节省显存。几何增强先统一完成，风格变换不改变标签坐标。

`StyleInterventionBank.forward` 只负责组织对齐视图；视图内部的 photometric 与 Fourier 变换均在原图的归一化/反归一化空间中完成：

```python
def forward(self, images: Tensor) -> StyleViews:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape Bx3xHxW")
    return StyleViews(
        original=images,
        photometric=self.photometric_view(images),
        fourier=self.fourier_view(images),
    )
```

Fourier 分支保留相位、只混合幅度：

```python
spectrum = torch.fft.fft2(unit, dim=(-2, -1))
amplitude = spectrum.abs()
phase = spectrum / amplitude.clamp_min(1e-8)
if unit.shape[0] > 1:
    donor = amplitude.roll(shifts=1, dims=0)
else:
    donor = amplitude.roll(shifts=1, dims=1)
strength = self._sample(self.fourier_mix_strength, unit)
mixed_amplitude = amplitude.lerp(donor, strength)
mixed = torch.fft.ifft2(
    mixed_amplitude * phase, dim=(-2, -1)
).real
```

当 batch size 为 1 时，源码沿空间维滚动幅度作为 donor；当 batch size 大于 1 时，使用相邻样本的幅度。该实现不做几何 warp，因此标签仍可直接复用。

## 8. 损失函数与实验变体

### 8.1 分割监督

原图始终使用标准像素级交叉熵：

```text
L_seg = CE(L_original, Y)
```

风格视图使用相同标签。combined 配置将总风格监督权重固定为 1：

```text
L_style = 0.5 * CE(L_photo, Y) + 0.5 * CE(L_fourier, Y)
```

单风格消融则使用权重 1，以保证总反事实监督权重可比。

### 8.2 普通预测一致性

A3 使用原图预测作为停止梯度的参考，对风格视图施加有效像素上的 KL 散度：

```text
L_pred = KL(stopgrad(P_original) || P_style)
```

权重 `lambda_pred = 1.0`。它是 CQE 的必要控制组，用来回答“约束 Query 效应是否比直接约束预测更有价值”。

源码实现（[`causalq/losses/prediction_consistency.py`](../causalq/losses/prediction_consistency.py)）先 detach 原图 teacher，再只在有效像素上求 KL：

```python
scaled_view = view_logits.float() / temperature
scaled_reference = reference_logits.detach().float() / temperature
reference_probabilities = F.softmax(scaled_reference, dim=1)
per_class = F.kl_div(
    F.log_softmax(scaled_view, dim=1),
    reference_probabilities,
    reduction="none",
)
per_pixel = per_class.sum(dim=1) * temperature**2
return per_pixel[labels != ignore_index].mean()
```

### 8.3 Causal Query Effect Distillation

A4 对原图与风格图的零消融效应图进行逐类 L2 归一化，仅在有效像素和当前样本真实出现的类别上计算 Smooth L1：

```text
L_CQE = SmoothL1(norm(stopgrad(E_original)), norm(E_style))
```

权重 `lambda_cqe = 1.0`。停止梯度只作用于原图参考分支。

CQE 的关键掩码和归一化逻辑（[`causalq/losses/causal_query_effect.py`](../causalq/losses/causal_query_effect.py)）如下：

```python
valid = labels != ignore_index
safe_labels = labels.masked_fill(~valid, 0)
present = F.one_hot(safe_labels, num_classes=num_classes).bool()
present = (present & valid.unsqueeze(-1)).any(dim=(1, 2))

view_flat = view_effect.float().reshape(batch_size, num_classes, -1)
reference_flat = reference_effect.detach().float().reshape(
    batch_size, num_classes, -1
)
view_flat = view_flat.masked_fill(~valid.reshape(batch_size, 1, -1), 0.0)
reference_flat = reference_flat.masked_fill(
    ~valid.reshape(batch_size, 1, -1), 0.0
)
view_normalized = F.normalize(view_flat, p=2, dim=-1, eps=eps)
reference_normalized = F.normalize(reference_flat, p=2, dim=-1, eps=eps)
per_element = F.smooth_l1_loss(
    view_normalized, reference_normalized,
    reduction="none", beta=beta,
)
return per_element.sum(dim=-1)[present].mean()
```

读者应注意 `present` 是“每个样本中出现过的类别”掩码，不是把 19 类全部强行平均；这正是当前实现处理类别稀疏标签的地方。

### 8.4 Query Diversity

A5 对每一类内部的 Query residual 施加非对角余弦相似度平方惩罚：

```text
L_div = mean(off_diagonal(cos(residual[c])))^2
```

权重 `lambda_div = 0.01`。它确实让同类 Query 更分散，但 mIoU 只比 A4 增加 0.037 个百分点，低于预设 0.2 个百分点保留阈值，因此已作为负消融结论拒绝进入最终方法。

## 9. A0–A5 受控实验阶梯

| ID | Query | Style | 预测一致性 | CQE | Diversity | 目的 |
|---|---:|---:|---:|---:|---:|---|
| A0 | 否 | 否 | 否 | 否 | 否 | 冻结 DINOv3 source-only 基线 |
| A1 | 是 | 否 | 否 | 否 | 否 | 验证 Query 本身 |
| A2 | 是 | 是 | 否 | 否 | 否 | 验证风格干预监督 |
| A3 | 是 | 是 | 是 | 否 | 否 | 普通预测一致性控制 |
| A4 | 是 | 是 | 受控 | 是 | 否 | 验证零消融效应一致性 |
| A5 | 是 | 是 | 是 | 是 | 是 | 验证 Query 多样性正则 |

这条阶梯遵循“每次只增加一个主要机制”的原则。不能用 A5 对 A0 的总差异替代相邻实验比较，否则无法判断哪一项有效。

## 10. 已完成实验与当前结论

### 10.1 主实验结果

| 实验 | Cityscapes mIoU | 相邻变化 | 结论 |
|---|---:|---:|---|
| A0 Baseline | 60.17% | — | 冻结 DINOv3-L 基线成立 |
| A1 Query | 61.95% | +1.78 pp | Query 分支有效 |
| A2 Query + Style | **63.96%** | +2.01 pp | 当前最强主实验 |
| A3 + Prediction Consistency | 62.12% | -1.84 pp | 普通 KL 一致性有害 |
| A4 + CQE | 59.18% | -2.95 pp | 当前 CQE 有害 |
| A5 + Diversity | 59.21% | +0.04 pp | 提升低于阈值，拒绝保留 |

相对 A0，A2 共提高约 3.79 个百分点。A4 和 A5 不应被称为最终最优模型。

### 10.2 效应稳定性分析

| 模型 | 跨风格归一化效应方差 |
|---|---:|
| A3 | 0.0019905460 |
| A4 | 0.0000105223 |

A4 相比 A3 将方差降低约 99.47%，但 mIoU 同时下降。这是项目目前最关键的诊断：

```text
效应更一致 != 效应更有语义、更充分或更能支持正确预测
```

因此，后续不能只继续调大/调小 `lambda_cqe`，而应修改效应定义并加入防退化目标。

### 10.3 风格消融

seed 0：

| 视图 | mIoU |
|---|---:|
| Combined | 63.96% |
| Photometric only | 63.83% |
| Fourier only | 62.71% |

seed 1：

| 视图 | mIoU |
|---|---:|
| Combined | 65.10% |
| Photometric only | 65.15% |

seed 2 中 combined 为 62.95%，photometric-only 为 61.88%。三 seed 汇总后，combined 为 `64.00 ± 1.08%`，photometric-only 为 `63.62 ± 1.64%`；配对差为 `+0.38 ± 0.60 pp`，且 seed 1 排序翻转。该证据不支持 combined 稳定优于 photometric-only。按预设决策树，后续优先采用计算更低的 photometric-only，combined 与 Fourier 保留为消融证据。

## 11. 项目进度总览

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 0 | 清理旧项目依赖，建立独立仓库原则 | 完成 |
| Phase 1 | 工程骨架、配置、测试框架 | 完成 |
| Phase 2 | AutoDL 环境与 RTX 4090D 验证 | 完成 |
| Phase 3 | GTA5/Cityscapes 数据检查与映射 | 完成 |
| Phase 4 | DINOv3-L 权重与主干 smoke test | 完成 |
| Phase 5 | A0 source-only baseline | 完成 |
| Phase 6 | A1 Query-only | 完成 |
| Phase 7 | A2 Style Intervention | 完成 |
| Phase 8 | A3 Prediction Consistency | 完成 |
| Phase 9 | A4 CQE 与效应方差分析 | 完成，结果为负 |
| Phase 10 | A5 Query Diversity | 完成，机制被拒绝 |
| Phase 11 | 风格消融与多种子复验 | 完成，默认选择 photometric-only |
| Phase 12 | Query 数量消融（R=1/2/3/4） | 完成；机制诊断选择 R=2 |
| Phase 13 | Query Interaction 消融 | Static Query smoke 已通过，等待40k full run |
| 扩展评估 | BDD100K、Mapillary、规模扩展 | 未执行 |
| 理论升级 | learned-null + sufficiency + specificity | 仅为拟议路线，未实现 |

## 12. 代码与文档地图

```text
CausalQ_Project/
├── README.md                       # 快速状态与入口
├── CausalQ_DG_Project_Plan.md      # 原始总体方案
├── configs/                        # A0–A5 与风格消融配置
├── causalq_dg/
│   ├── datasets/                   # GTA5、Cityscapes、标签映射与增强
│   ├── models/                     # DINOv3、基础解码器、Query 模型
│   ├── style/                      # Photometric 与 Fourier 干预
│   ├── losses/                     # CE、预测一致性、CQE、diversity
│   ├── metrics/                    # mIoU
│   └── analysis/                   # 效应方差分析
├── tools/                          # 环境、数据、权重、训练、评估工具
├── scripts/                        # 分阶段本机与 AutoDL 命令
├── tests/                          # CPU 单元与 smoke tests
├── experiments/                    # 进入 Git 的精简实验元数据与摘要
└── docs/
    ├── METHOD.md                   # 当前实现定义
    ├── CAUSAL_MODEL.md             # 当前因果假设
    ├── DATASETS.md                 # 数据布局与映射
    ├── ABLATION_PLAN.md            # 消融矩阵与门槛
    ├── UPSTREAM.md                 # 上游参考边界
    ├── AUTODL_NEXT.md              # 唯一下一步远端操作单
    └── PROJECT_GUIDE.md            # 本指导书
```

关键实现入口：

- `causalq_dg/models/dinov3_wrapper.py`：冻结主干和 token 网格恢复；
- `causalq_dg/models/baseline_segmentor.py`：A0；
- `causalq_dg/models/query_segmentor.py`：Query Bank、Cross-Attention、残差与零消融；
- `causalq_dg/style/`：干预视图；
- `causalq_dg/losses/`：A3–A5 新增目标；
- `tools/train.py`：配置验证、训练和验证主入口；
- `experiments/registry.csv`：所有正式运行的登记表。

## 13. 环境与依赖

已验证的 AutoDL 环境为：

- GPU：NVIDIA GeForce RTX 4090 D，约 23.52 GiB；
- Python：3.12.3；
- PyTorch：2.7.0+cu126；
- CUDA runtime：12.6；
- torchvision：0.22.0+cu126；
- transformers：4.56.1；
- timm：1.0.15；
- lightning：2.5.1.post0；
- torchmetrics：1.7.1。

`tools/check_environment.py` 中 `ok: false` 若只由 `data/pretrained/outputs` 尚未设置引起，并不表示 CUDA 或软件栈损坏。应先配置路径，再复查环境。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

环境检查：

```bash
python tools/check_environment.py \
  | tee /root/autodl-tmp/causalq_environment_report.txt
```

## 14. 数据与权重准备

### 14.1 推荐目录

大文件只放 AutoDL 数据盘，不进入 Git：

```text
/root/autodl-tmp/
├── CausalQ_DG/              # Git 仓库
├── datasets/
│   ├── gta5/
│   │   ├── images/
│   │   └── labels/
│   └── cityscapes/
│       ├── leftImg8bit/
│       └── gtFine/
├── pretrained/
│   └── dinov3_vitl16/
└── outputs/
```

具体可接受的数据布局、GTA5 19 类映射和 Cityscapes 文件匹配规则以 [`DATASETS.md`](DATASETS.md) 为准。

### 14.2 数据检查原则

在开始任何训练前必须确认：

- 图像与标签一一对应；
- 标签能映射到 19 个 train IDs 与 255；
- 随机裁剪后有效像素比例不低于配置门槛；
- 风格视图和原标签空间严格对齐；
- Cityscapes validation 不参与梯度更新。

### 14.3 DINOv3 权重

主模型标识：`facebook/dinov3-vitl16-pretrain-lvd1689m`。

固定权重 SHA-256：

```text
dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179
```

权重目录和散列必须同时匹配配置。不要把权重提交到 Git，也不要在不同实验间悄悄更换模型版本。

## 15. 本机与 AutoDL 协作流程

### 15.1 本机负责什么

- 实现和审查代码；
- 修改配置与文档；
- 运行静态检查和 CPU 测试；
- 整理实验摘要、指标和决策；
- 提交 Git。

### 15.2 AutoDL 负责什么

- 大规模数据检查；
- DINOv3 权重实际加载；
- CUDA/GPU smoke test；
- 训练、评估与显存测量；
- 生成原始日志和 checkpoint。

AutoDL 只运行一个明确的 Git commit。禁止直接在 AutoDL 修改源码；如果发现问题，应回到本机修改、测试、提交，再在 AutoDL 拉取该提交。

### 15.3 标准循环

```text
本机实现 -> pytest -> Git commit/push
    -> AUTODL_NEXT.md 写明 commit 与命令
    -> AutoDL fetch/checkout 精确 SHA
    -> 环境/数据/权重门禁
    -> smoke test
    -> 通过后 full run
    -> 提取精简证据
    -> 本机更新 experiments/registry.csv 与文档
```

任何需要 GPU 或远端数据才能继续验证的阶段，都必须停止本机实现并更新 `docs/AUTODL_NEXT.md`。

## 16. 训练协议

当前 A0–A5 的共同设置：

- crop：`512×512`；
- batch size：1；
- AMP：开启；
- 最大迭代：40,000；
- optimizer：AdamW；
- learning rate：`1e-4`；
- weight decay：`0.01`；
- scheduler：poly，power `0.9`；
- Cityscapes validation interval：500；
- 基础随机种子：0；
- 数据增强：scale `[0.5, 2.0]`、水平翻转概率 `0.5`。

RTX 4090D 上使用多风格视图时采用顺序前向，避免同时持有三份完整计算图。现有 full run 的峰值显存约在 2.5 GiB reserved 量级，但新机制仍必须重新测量，不能据此假设任意扩展都安全。

## 17. 复现一个正式实验

以下是通用顺序；精确命令必须从对应提交的 `docs/AUTODL_NEXT.md` 或 `scripts/` 读取。

### 17.1 锁定代码

```bash
cd /root/autodl-tmp/CausalQ_DG
git fetch origin
git checkout <EXACT_COMMIT_SHA>
test "$(git rev-parse HEAD)" = "<EXACT_COMMIT_SHA>"
git status --short
```

### 17.2 设置数据路径

```bash
export CAUSALQ_DATA_ROOT=/root/autodl-tmp/datasets
export CAUSALQ_PRETRAINED_ROOT=/root/autodl-tmp/pretrained
export CAUSALQ_OUTPUT_ROOT=/root/autodl-tmp/outputs
```

### 17.3 运行门禁

```bash
python tools/check_environment.py
python -m pytest
```

还应执行对应阶段的数据检查、权重检查和 GPU smoke test。任何门禁失败都不得直接启动 40k full run。

训练器会把配置约束落实为运行时检查。例如 Phase 11/12 禁止混入 CQE、预测一致性或 diversity：

```python
if phase in (11, 12) and (
    prediction_enabled or cqe_enabled or diversity_enabled
):
    raise ValueError("Phase 11/12 ablations disable all consistency losses")
```

正式训练的视图循环也体现了“原图先行、风格视图复用同一标签”的协议：

```python
for view_name, view_images in named_views:
    logits = model(view_images)
    view_loss = segmentation_cross_entropy(logits, labels)
    objective = view_loss * view_weights[view_name]

    if prediction_enabled and view_name != "original":
        objective += (lambda_pred / 2.0) * prediction_consistency_kl(
            logits, reference_logits, labels=labels
        )
    (objective / accumulation).backward()
```

完整训练循环还包含 finite 检查、梯度累积、梯度裁剪、验证与 JSONL 证据记录；上面只保留解释协议所需的核心路径。

### 17.4 正式训练与证据保存

每次运行至少记录：

- experiment ID 与运行目录名；
- 精确 Git SHA；
- 完整配置副本；
- GPU、Python、PyTorch、CUDA 和依赖版本；
- 随机种子；
- 数据根目录与样本计数；
- DINOv3 权重路径及 SHA-256；
- 最佳/最终 mIoU 与对应 step；
- 峰值显存；
- 各损失曲线、`alpha`、异常/恢复信息；
- 运行完成状态。

原始数据、权重、完整 checkpoints 和完整运行目录不进入 Git。Git 只保存精简后的 `summary.json`、必要日志摘录、运行清单和 `experiments/registry.csv`。

## 18. 当前下一步：Phase 13 Query Interaction 消融

Phase 12 已固定 `R=2`。Phase 13 按原项目计划验证 Query 与图像交互是否必要，保持 photometric-only、seed 0、优化器和训练长度不变。第一个受控变体是 Static Query：保留 grouped query bank 与 pixel/query residual head，但移除 Query→Image cross-attention。其500-iteration smoke 已在 `367694e` 通过；下一步运行 seed-0 40k full run，并与现有 R=2 one-way 结果 `0.643725` 比较。在 Static Query full 完成前不加入 bidirectional interaction 或 learned-null。

执行前必须使用最新 [`AUTODL_NEXT.md`](AUTODL_NEXT.md) 中的精确提交、清理保护和命令，不能从本指导书复制占位符直接运行。

Query 数量实验除 mIoU 外，还应在 full run 后比较 query similarity、active-query 行为和 query-effect variance，避免仅凭容量变化解释结果。

## 19. 创新点：已经成立与仍待验证

### 19.1 已实现且有实验支持

**类别分组的上下文自适应 Query。** 每类拥有共享锚点和多个 residual，Query 通过单向 cross-attention 读取图像上下文。A1 比 A0 提高 1.78 pp，说明这一机制在当前协议下有效。

**显式可分离的 Query logit 残差。** 基础预测与 Query 贡献在结构上分开，`alpha=0` 的初始化保证安全接入，也使类级零消融无需重跑主干。这是后续机制分析的工程基础。

**保持几何的风格干预。** Photometric 处理颜色与成像变化，Fourier 处理频谱幅度变化；A2 比 A1 提高 2.01 pp。三 seed 消融未证明 combined 稳定优于 photo-only，因此后续默认使用更简洁的 photometric-only，双风格结果保留为受控消融。

**将机制稳定性与任务性能分开测量。** A4 展示了 99.47% 方差降低却伴随显著精度下降。这一结果否定了“稳定性指标降低即可证明机制更好”的简单叙事，是有价值的机制诊断。

### 19.2 已实现但被当前证据否定

- 普通 prediction KL consistency：A3 低于 A2；
- 基于零消融残差的 CQE：A4 低于 A3/A2；
- 当前 query diversity：A5 对 A4 的提升低于保留阈值。

这些模块应保留为受控消融和失败证据，不应组合成“最终推荐方法”。

### 19.3 下一版拟议创新，尚未实现

后续理论升级建议将研究框架改为 **Style-Interventional Query Mechanism Learning**，核心变化包括：

1. 将数据生成 SCM 与模型内部 SCM 明确分开；
2. 区分 factual、zero、learned-null、semantic-counterfactual 四种 Query 状态；
3. 主效应从 `factual - zero` 改为 `factual - learned-null`；
4. zero 只作为 sanity check，不再承担主因果语义；
5. 同时优化 effect invariance、effect sufficiency 与 effect specificity；
6. 引入 EI、ES、EL、ESR 等机制指标，避免只看方差。

这一方向直接回应 A4 的失败模式：

```text
Invariance：不同风格下效应是否一致？
Sufficiency：保留 Query 机制是否足以支持正确预测？
Specificity：移除/替换该机制是否真的破坏对应语义证据？
```

只有同时回答这三个问题，才能区分“稳定且有用的机制”与“稳定但接近零的退化机制”。这部分目前只是设计路线，尚无代码、配置或 GPU 结果。

## 20. 后续研究决策树

```text
Phase 11：差异不稳定/极小
  |
  v
优先 photo，combined/Fourier 作为消融
  |
  v
完成 Query 数量 R=1/2/3/4 消融
  |
  v
冻结可复现的 A2 强基线
  |
  v
单独实现 learned-null 干预
  |
  +-- 无法避免效应塌缩 -------> 停止扩大因果叙事
  |
  v
依次加入 sufficiency、specificity（每次只加一项）
  |
  v
同时检查 mIoU + EI/ES/EL/ESR + 显存/稳定性
  |
  +-- 机制指标改善但 mIoU 下降 -> 诊断冲突，不宣称成功
  |
  +-- 多种子均改善 ------------> 扩展 BDD100K/Mapillary
```

## 21. 新成员上手清单

### 第一天：理解项目

- 阅读本指导书；
- 阅读 `README.md`、`METHOD.md`、`CAUSAL_MODEL.md`；
- 查看 `experiments/registry.csv`；
- 理解 A0–A5 只相邻比较；
- 明确 A2 是当前最强模型，A4/A5 是负结果。

### 第二天：理解实现

- 从 `DINOv3Backbone` 跟到 `BaselineSegmentor`；
- 再阅读 `QuerySegmentor` 的 factual、delta、scaled delta 和 zero ablation；
- 阅读 style bank 与三个损失模块；
- 用单元测试理解张量形状和梯度边界。

### 第三天：完成本机验证

```bash
python -m pytest
```

确认测试全部通过，并检查 `git status --short` 没有误改数据、权重或运行目录。

### 开始 GPU 工作前

- 只执行 `AUTODL_NEXT.md` 指定阶段；
- checkout 精确 SHA；
- 核对环境、数据、权重散列；
- 先 smoke test，再 full run；
- 记录运行证据，不在 AutoDL 改源码。

## 22. 常见误区

**误区 1：A5 模块最多，所以是最终模型。**

错误。研究模型由受控结果决定，不由模块数量决定。当前 A2 最好。

**误区 2：效应方差下降 99.47%，所以 CQE 成功。**

错误。稳定性目标被优化成功，但任务性能下降，且可能存在效应缩小的退化解。

**误区 3：zero ablation 等于真实因果干预。**

错误。它是模型结构内部的机制消融代理。

**误区 4：可以直接把 EoMT、REIN 或旧 DAFormer/MRM 项目复制进来。**

错误。本项目只把 EoMT/DINOv3 作为主要实现参考，把 REIN、Causal-Tune、SoMA 作为协议参考；不复制旧项目源码或历史包袱。

**误区 5：AutoDL 上修一行代码更快。**

错误。这样会破坏精确提交复现。所有源码修改必须回到本机并进入 Git。

**误区 6：单个 seed 的微小差异足以决定风格方案。**

错误。seed 0/1 排序翻转，三 seed 配对后仍未得到稳定优势，因此项目选择更简洁的 photometric-only，而不是放大微小均值差。

## 23. 完成标准

一个阶段只有同时满足以下条件才算完成：

- 代码、配置、测试和文档处于同一个 Git 提交链；
- 本机测试通过；
- GPU 阶段有精确 SHA、环境、权重与数据证据；
- smoke test 通过后才进行 full run；
- 指标和失败情况已进入实验登记；
- 结论与证据一致，不隐藏负结果；
- `AUTODL_NEXT.md` 已指向唯一下一步。

项目层面的最低可信成果不是“堆出一个复杂模型”，而是形成一条可复现、可否证的证据链：证明哪些 Query/风格机制有效，哪些一致性约束失败，失败原因是什么，以及下一版设计如何针对该失败模式。

## 24. 术语表

| 术语 | 含义 |
|---|---|
| DG | Domain Generalization，训练时不访问目标域的域泛化 |
| Query | 按类别组织、从图像 token 读取上下文的可学习向量 |
| Factual | 正常 Query 状态下的模型输出 |
| Zero ablation | 将特定 Query 残差贡献置零的模型内部消融 |
| Query effect | 当前实现中 factual 与 zero ablation 的 logit 差分 |
| Style intervention | 保持几何/标签、改变外观的代理干预 |
| CQE | Causal Query Effect Distillation，跨风格对齐 Query 效应图 |
| Effect collapse | 通过让效应接近零来满足一致性约束的退化现象 |
| pp | percentage point，百分点 |
| Smoke test | 小规模快速运行，用于验证完整链路而非报告最终精度 |
| Full run | 按正式配置完成的 40k 训练与评估 |

## 25. 权威信息优先级

出现冲突时按以下顺序判断：

1. 当前 Git 中的代码与配置；
2. `experiments/registry.csv` 及运行摘要；
3. `docs/METHOD.md`；
4. `docs/AUTODL_NEXT.md`（仅负责下一次远端操作）；
5. 本指导书；
6. 原始 `CausalQ_DG_Project_Plan.md` 与后续理论提案。

原始计划描述的是研究设计空间，不保证每项都已经实现；实验结果可以否定计划中的机制。项目应以可复现实证为准，而不是为了维护最初叙事而忽略负结果。
