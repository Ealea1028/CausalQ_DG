# CausalQ-DG：Counterfactual Query Effect Distillation for Domain-Generalized Semantic Segmentation

## 0. Codex 身份与执行规则

你是本项目的编码 Agent。

项目目标是在 **Vision Foundation Model + Domain Generalized Semantic Segmentation** 场景下研究：

> 在不同 Style Intervention 下，与语义预测真正相关的 Query causal effect 是否应该保持稳定；是否可以通过蒸馏这种 causal effect，而不是直接强制 feature 一致，提高 unseen-domain semantic segmentation 性能。

项目最终目标模型：

```text
DINOv3 ViT-L/16
        ↓
frozen / mostly frozen VFM
        ↓
image patch features
        ↓
Grouped Causal Query Module
        ↓
context-adaptive semantic queries
        ↓
Causal Query Residual Prediction
        ↓
base segmentation prediction + query residual
        ↓
semantic segmentation
```

训练期间加入：

```text
Style Intervention
        ↓
same content / same label
different appearance
        ↓
do(S)
        ↓
Query ablation
        ↓
do(Qc = 0)
        ↓
Causal Query Effect
        ↓
Cross-style Effect Distillation
```

项目简称：

```text
CausalQ_DG
```

---

# 1. 严格区分本机与 AutoDL

## 本机

本机不承担正式 GPU 训练。

本机主要任务：

- GitHub 代码管理；
- Codex 编码；
- 模型模块实现；
- 配置文件；
- 单元测试；
- 静态检查；
- 数据检查脚本；
- 实验方案；
- 结果分析脚本；
- README；
- 文档；
- 根据 AutoDL 日志修复代码。

本机目录建议：

```text
D:/Research/CausalQ_DG
```

本机只能运行：

```text
CPU smoke test
pytest
import test
shape test
config test
```

不要尝试在本机 RTX 4060 上完成 DINOv3-L 正式训练。

## AutoDL

AutoDL 主要任务：

- 数据集保存；
- DINOv3 权重保存；
- GPU 环境；
- 训练；
- 验证；
- DG 测试；
- checkpoint；
- 大型日志；
- 可视化；
- 多 seed 实验；
- ablation；
- scaling experiment。

AutoDL 项目：

```text
/root/autodl-tmp/CausalQ_DG
```

永久数据：

```text
/root/autodl-tmp/datasets
```

权重：

```text
/root/autodl-tmp/pretrained
```

训练结果：

```text
/root/autodl-tmp/outputs/CausalQ_DG
```

---

# 2. GitHub 是唯一代码真源

本机和 AutoDL 必须使用同一个 GitHub repository：

```text
YOUR_GITHUB_USER/CausalQ_DG
```

规则：

```text
本机 Codex
   ↓ commit
GitHub
   ↓ checkout exact commit
AutoDL
```

AutoDL 原则上禁止直接修改源码。

AutoDL 如果发现错误：

```text
记录 traceback
↓
反馈给本机
↓
本机 Codex 修改
↓
commit + push
↓
AutoDL git pull
```

禁止：

```text
本机代码版本 A
AutoDL 手工改成版本 B
但没有 Git commit
```

---

# 3. GitHub 不保存原始大数据

GitHub 保存：

```text
源码
配置
实验摘要
数据索引
dataset manifest
环境配置
分析结果
小型图片
```

GitHub 不保存：

```text
GTA5
Cityscapes
BDD100K
Mapillary
ACDC
DINOv3 pretrained weights
*.pth
*.pt
*.ckpt
完整 work_dir
```

使用：

```text
data_manifest.yaml
```

统一两端数据定义。

AutoDL：

```bash
export CAUSALQ_DATA_ROOT=/root/autodl-tmp/datasets
export CAUSALQ_PRETRAINED_ROOT=/root/autodl-tmp/pretrained
export CAUSALQ_OUTPUT_ROOT=/root/autodl-tmp/outputs/CausalQ_DG
```

任何代码禁止写死：

```text
/root/autodl-tmp/...
```

---

# 4. 旧 DAFormer/MRM/QK 项目如何处理

当前 AutoDL 实例为全新实例，且本项目不再以过去的 DAFormer+MRM 或 QK Adapter 项目为前置基础。

因此新实例中不要求存在：

```text
/root/autodl-tmp/DAFormer_MRM
```

也不要求上传过去 QK Adapter 项目资产。若将来另行找回旧资产，这些旧工程只允许复用：

```text
GTA5 数据
Cityscapes 数据
转换后的 trainId labels
dataset split
数据转换脚本
过去实验结果
路径结构经验
```

不直接复用：

```text
旧 mmseg 0.x 源码
MRM Rebuilder
旧 DACS 修改
旧 MiT-B5
旧 QK Adapter
旧 conda 环境
旧 checkpoint
```

原因：

```text
旧工程技术栈
        ≠
DINOv3 / modern VFM 技术栈
```

新项目必须独立。

---

# 5. 推荐工程底座

主工程参考：

```text
tue-mps/eomt
```

用途：

```text
DINOv3 backbone
segmentation query mechanism
DINOv3 semantic segmentation
modern environment
```

DGSS 数据协议参考：

```text
w1oves/Rein
```

以及：

```text
zhangyin1996/Causal-Tune
```

用途：

```text
GTA5 conversion
Mapillary conversion
BDD100K layout
Cityscapes mapping
ACDC
DG evaluation protocol
```

强基线参考：

```text
ysj9909/SoMA
```

不要直接在这些 upstream repository 中开发。

本项目自己维护：

```text
CausalQ_DG
```

在：

```text
third_party/
```

保存必要的 upstream 信息或 submodule。

---

# 6. 最终建议项目结构

```text
CausalQ_DG/
│
├── README.md
├── AGENTS.md
├── CHANGELOG.md
├── requirements.txt
├── environment.yaml
├── data_manifest.yaml
├── pretrained_manifest.yaml
├── .gitignore
│
├── configs/
│   ├── baseline/
│   │   ├── gta_dinov3l.yaml
│   │   └── citys_dinov3l.yaml
│   ├── query/
│   │   └── gta_dinov3l_query.yaml
│   ├── style/
│   │   └── gta_dinov3l_style.yaml
│   └── causalq/
│       ├── gta_dinov3l_causalq.yaml
│       └── citys_dinov3l_causalq.yaml
│
├── causalq/
│   ├── __init__.py
│   ├── models/
│   │   ├── dinov3_wrapper.py
│   │   ├── baseline_segmentor.py
│   │   ├── causal_query_bank.py
│   │   ├── query_cross_attention.py
│   │   ├── query_residual_head.py
│   │   └── causalq_segmentor.py
│   ├── interventions/
│   │   ├── style_bank.py
│   │   ├── photometric.py
│   │   └── fourier_style.py
│   ├── losses/
│   │   ├── segmentation.py
│   │   ├── prediction_consistency.py
│   │   ├── causal_query_effect.py
│   │   ├── query_diversity.py
│   │   └── query_prototype.py
│   ├── datasets/
│   │   ├── gta5.py
│   │   ├── cityscapes.py
│   │   ├── bdd100k.py
│   │   ├── mapillary.py
│   │   └── acdc.py
│   ├── metrics/
│   │   └── miou.py
│   └── utils/
│       ├── distributed.py
│       ├── checkpoint.py
│       └── seed.py
│
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── check_environment.py
│   ├── check_datasets.py
│   ├── check_backbone.py
│   ├── export_run_summary.py
│   └── visualize.py
│
├── analysis/
│   ├── effect_variance.py
│   ├── query_similarity.py
│   ├── query_activation.py
│   ├── style_sensitivity.py
│   └── make_tables.py
│
├── scripts/
│   ├── setup_autodl.sh
│   ├── train_baseline.sh
│   ├── train_query.sh
│   ├── train_style.sh
│   ├── train_causalq.sh
│   ├── eval_cbm.sh
│   └── publish_result.sh
│
├── experiments/
│   ├── README.md
│   └── registry.csv
│
├── docs/
│   ├── METHOD.md
│   ├── CAUSAL_MODEL.md
│   ├── AUTODL_NEXT.md
│   └── ABLATION_PLAN.md
│
└── tests/
    ├── test_query_bank.py
    ├── test_query_head.py
    ├── test_style_intervention.py
    └── test_causal_effect.py
```

---

# 7. 因果假设

变量：

```text
D = domain/environment
S = style
C = semantic content
X = image
F = VFM feature
Q = semantic query
Y = GT segmentation
Ŷ = model prediction
```

主因果图：

```text
          Domain D
             │
             ▼
           Style S
             │
             ▼
Content C ──► X
    │         │
    │         ▼
    │      Feature F
    │       /     \
    │      ▼       ▼
    │  Base Head   Q
    │      │       │
    │      │       ▼
    │      │   Query Effect
    │      │       │
    └──────┴──────►Ŷ
    │
    ▼
    Y
```

核心：

```text
C → X ← S
```

X 是 content 与 style 的共同结果。

在合理 style intervention 下：

```text
do(S = s1)
do(S = s2)
...
```

应满足：

```text
C fixed
Y fixed
S changes
```

理想模型：

```text
P(Y | do(S=s1), C)
≈
P(Y | do(S=s2), C)
```

---

# 8. 模型主干

主 backbone：

```text
DINOv3 ViT-L/16
```

预训练：

```text
LVD-1689M
```

第一阶段：

```text
freeze backbone
```

只训练：

```text
segmentation head
causal query bank
query cross-attention
query projection
query residual head
```

后期 ablation 才允许：

```text
unfreeze final block
```

或：

```text
LoRA final 1-4 blocks
```

绝不一开始 full fine-tuning。

---

# 9. 不直接复制 QPrompt 的 Query 结构

QPrompt：

```text
learnable queries
+
image tokens
        ↓
same transformer block
```

本项目为了获得更清楚的因果干预路径，采用：

## One-way Context-Adaptive Query

即：

```text
Query → read Image
```

但：

```text
Image feature
```

不被 Query 修改。

实现：

```text
Q = learnable queries
K = image features
V = image features

Q' = CrossAttention(Q,K,V)
```

因此：

```text
F → Q'
```

但没有：

```text
Q → F
```

这样：

```text
do(Qc = 0)
```

不会改变 base image representation。

这是与 QPrompt 的重要结构区别。

---

# 10. Grouped Causal Query Bank

Cityscapes protocol：

```text
num_classes = 19
```

每一类：

```text
queries_per_class = 3
```

总 Query：

```text
19 × 3 = 57
```

定义：

```text
Qc = {qc1, qc2, qc3}
```

推荐参数化：

```text
q(c,r) = class_anchor(c) + residual(c,r)
```

其中：

```text
class_anchor
```

学习共有类别语义；

```text
residual
```

让同类 Query 保留多样性。

目的：

```text
同一类别
↓
多个可替代语义路径
↓
降低 single-query failure
```

---

# 11. Query Cross Attention

输入 feature：

```text
F ∈ R[B,N,D]
```

Query：

```text
Q ∈ R[B,57,D]
```

执行：

```text
Q' = LN(Q + MHA(Q,F,F))
Q'' = LN(Q' + FFN(Q'))
```

只需要：

```text
1 layer
```

默认：

```text
num_heads = 8
```

或根据 D=1024 调整。

第一版不要堆多层。

---

# 12. Query Residual Head

Base segmentation branch：

```text
F
↓
Base Head
↓
Z_base
```

Query branch：

```text
F + Q'
↓
similarity
↓
ΔZ_query
```

最终：

```text
Z = Z_base + α ΔZ_query
```

α 为 learnable scalar。

初始化：

```text
α = 0
```

非常重要。

这样模型初始化时：

```text
Z = Z_base
```

新增 Query branch 不会一开始破坏 baseline。

---

# 13. Query 如何产生 pixel logits

Feature projection：

```text
fp = normalize(Wf(F))
```

Query projection：

```text
qcr = normalize(Wq(qcr))
```

计算：

```text
s(c,r,p) = fp(p)^T q(c,r) / τ
```

同类别多个 Query 聚合：

```text
ΔZc(p)
=
logsumexp_r(s(c,r,p))
```

或第一版：

```text
mean_r
```

默认先使用：

```text
logsumexp
```

因为允许不同 query 竞争。

最终：

```text
ΔZ ∈ R[B,19,H',W']
```

---

# 14. do(Q) 如何实现

因为：

```text
Z = Z_base + αΔZ
```

对于类别 c：

```text
do(Qc = 0)
```

定义为：

```text
ΔZc = 0
```

其他部分保持不变。

于是 logit-level causal effect：

```text
Ec = Z - Z_do(Qc=0)
```

理论上：

```text
Ec = α ΔZc
```

因此不需要为每一个 class 重新运行 DINOv3。

这是本项目计算可行性的关键。

---

# 15. Style Intervention

不要第一版使用 CycleGAN。

原因：

```text
成本高
变量太多
很难证明改善来自 causal loss 还是 generator
```

第一版设置三个 counterfactual views：

```text
V0 = original
V1 = photometric intervention
V2 = Fourier amplitude intervention
```

## V1 Photometric

允许：

```text
brightness
contrast
saturation
hue
gamma
grayscale
color temperature
moderate blur
```

禁止：

```text
crop差异
rotation差异
random geometric warp
object removal
```

因为 paired effect distillation 要求：

```text
pixel correspondence
```

## V2 Fourier Style

保持 phase：

```text
Phase(X)
```

主要扰动：

```text
Amplitude(X)
```

或进行 batch 内 amplitude mixing。

直觉：

```text
phase ≈ structure
amplitude ≈ appearance/style statistics
```

第一版仅作为 style counterfactual generator，不声称 Fourier 完全等于真实 causal style。

---

# 16. Style Intervention 的验证标准

必须制作：

```text
analysis/style_examples/
```

至少输出：

```text
original
photometric
fourier
GT
```

确认：

```text
semantic layout不变
label不变
appearance明显变化
```

如果出现：

```text
物体结构明显损坏
```

必须降低 intervention 强度。

---

# 17. Loss 设计

最终候选：

```text
Ltotal =
Lseg
+ λcf Lseg_cf
+ λpred Lpred
+ λcqe Lcqe
+ λdiv Ldiv
```

不要一开始全部开启。

## Lseg

Original view：

```text
CE(Z0,Y)
```

可结合现有 segmentation baseline 的损失。

## Lseg_cf

Counterfactual style：

```text
mean_k CE(Zk,Y)
```

目的：

```text
不同 style
仍然语义正确
```

## Lpred

普通 prediction consistency：

```text
KL(
softmax(Zk),
stopgrad(softmax(Z0))
)
```

它非常重要，因为这是后面验证：

```text
普通 consistency
vs
causal effect consistency
```

的控制变量。

---

# 18. 核心 Loss：Causal Query Effect Distillation

对于 style：

```text
s0
s1
s2
```

得到：

```text
E(c,s0)
E(c,s1)
E(c,s2)
```

只对当前 GT 中存在的 class 计算。

定义：

```text
E(c,s)
=
effect of Qc on class-c prediction
```

第一版使用 logit effect：

```text
E(c,s) = α ΔZ(c,s)
```

因为不同 style view 保持像素对齐，可以直接约束 effect map。

定义：

```text
E_ref = stopgrad(E(c,s0))
```

然后：

```text
Lcqe =
mean_{s≠s0,c∈present(Y)}
SmoothL1(
normalize(E(c,s)),
normalize(E_ref)
)
```

normalize 默认对每个 class effect map 做：

```text
L2 normalization
```

防止模型单纯通过缩小 effect magnitude 降低 loss。

---

# 19. 防止 Causal Effect Collapse

存在风险：

```text
ΔZ → 0
```

则：

```text
Lcqe → 0
```

但 Query 什么也没学。

必须通过：

```text
Lseg
Lseg_cf
```

保证预测能力。

如果仍 collapse，增加：

```text
Lquery_proto
```

---

# 20. Optional Query Prototype Loss

只在 Query branch 学不到明显语义时启用。

利用 GT：

```text
Pc =
mean(F[pixels where Y=c])
```

停止梯度：

```text
stopgrad(Pc)
```

要求：

```text
mean_r Q(c,r)
```

接近：

```text
Pc
```

：

```text
Lqproto =
1 - cosine(
mean_r Q(c,r),
stopgrad(Pc)
)
```

它属于辅助损失。

不能让它成为论文主要创新。

---

# 21. Query Diversity Loss

同一 class：

```text
q(c,1)
q(c,2)
q(c,3)
```

不能完全相同。

只对：

```text
residual(c,r)
```

做轻量 decorrelation：

```text
Ldiv =
mean off-diagonal cosine^2
```

避免：

```text
3 queries
=
复制3次同一个query
```

但 λ 必须很小。

---

# 22. 论文真正要验证的假设

最关键不是：

```text
Full > Baseline
```

而是：

```text
Style + CQE
>
Style + Prediction Consistency
```

即：

```text
B3 → prediction consistency
B4 → causal effect consistency
```

必须证明：

```text
B4 > B3
```

否则论文会退化成：

```text
style augmentation + consistency
```

而没有充分 causal contribution。

---

# 23. 主实验协议

## Protocol A

训练：

```text
GTA5
```

禁止使用：

```text
Cityscapes train
BDD train
Mapillary train
```

测试：

```text
Cityscapes val
BDD100K val
Mapillary val
```

指标：

```text
mIoU Cityscapes
mIoU BDD100K
mIoU Mapillary
Avg mIoU
```

所有数据统一：

```text
Cityscapes 19 classes
```

## Protocol B

训练：

```text
Cityscapes train
```

测试：

```text
ACDC
```

分别：

```text
fog
night
rain
snow
average
```

这用于验证：

```text
normal → adverse condition
```

---

# 24. 推荐实验顺序

## Phase 0：清理与迁移

### [AutoDL]

目标：

```text
旧工程退出主工作流
数据保留
```

确认：

```text
/root/autodl-tmp/datasets/gta5
/root/autodl-tmp/datasets/cityscapes
```

之后补：

```text
bdd100k
mapillary
acdc
```

若旧资产不存在：

```text
DAFormer_MRM
QK Adapter
```

直接跳过即可；新项目代码、环境、配置和实验记录均从 GitHub 的 CausalQ_DG 重新建立，不把旧工程作为依赖。

---

# 25. Phase 1：本机建立项目骨架

### [LOCAL / CODEX]

创建：

```text
CausalQ_DG
```

完成：

```text
README
AGENTS
directory structure
.gitignore
data_manifest
environment files
test framework
```

同时保存 upstream commit 信息：

```text
docs/UPSTREAM.md
```

记录：

```text
EoMT commit
DINOv3/timm version
REIN commit
Causal-Tune commit
SoMA commit
```

### 验收

```bash
pytest
```

全部通过。

然后：

```text
commit:
chore: initialize CausalQ-DG project
```

---

# 26. Phase 2：AutoDL 新环境

### [AUTODL]

在全新实例中创建本项目独立环境，不寻找或修改旧 mrm 环境。

## AutoDL 镜像选择（RTX 4090D）

创建实例时优先选择 AutoDL 官方基础镜像中的：

```text
PyTorch 2.7.0 + CUDA 12.8 + Python 3.12
```

若同名镜像有 `devel` 与 `runtime` 两种，优先选 `devel`（后续若需编译 CUDA 扩展更省事；只运行预编译包时 `runtime` 也可）。若平台只提供官方 PyTorch 基础镜像，则选 CUDA 12.8、Ubuntu 22.04（或平台当前默认 Ubuntu LTS）的那一项。不要选旧的 CUDA 10.x/11.0、PyTorch 1.x，也不要选带有 DAFormer/MRM、QK Adapter 或不明第三方预装包的社区镜像。

选择原则是：镜像 CUDA 版本不得高于该实例驱动支持的上限，且 4090D 至少需要 CUDA 11.1 及以上；本项目按 EoMT/DINOv3 的现代栈优先采用 CUDA 12.8。镜像只是底座，项目依赖仍以本仓库的 `environment.yaml`/`requirements.txt` 和锁定版本为准。

优先以 EoMT 当前依赖为参考：

```text
Python 3.12
torch 2.7
torchvision 0.22
transformers 4.56.x
lightning 2.5.x
torchmetrics
```

正式安装前 Codex 必须生成：

```text
scripts/setup_autodl.sh
```

不要要求用户手工逐行修改环境。

安装后运行：

```text
python tools/check_environment.py
```

在正式安装前先记录镜像名称、Python、PyTorch、CUDA 与驱动版本；若 `torch.cuda.is_available()` 为 false 或 CUDA/驱动不匹配，先更换镜像，不进入训练阶段。

必须打印：

```text
Python
PyTorch
CUDA
GPU
VRAM
Transformers
Timm
Lightning
DINOv3 availability
```

---

# 27. Phase 3：数据检查

### [AUTODL]

运行：

```text
python tools/check_datasets.py
```

必须验证：

```text
GTA5 image count
GTA5 labels
Cityscapes train/val
class IDs
ignore index
image-mask shape
```

并随机保存：

```text
10组 image + colored GT
```

如果旧 GTA labels 仍是原始 label IDs：

使用 REIN/Causal-Tune 的 conversion logic。

转换后保存到：

```text
datasets/gta5/labels_trainIds
```

不要覆盖原始 label。

---

# 28. Phase 4：DINOv3-L Backbone Smoke Test

### [LOCAL / CODEX]

实现：

```text
dinov3_wrapper.py
```

必须支持：

```text
DINOv3 ViT-B
DINOv3 ViT-L
```

接口：

```python
features = backbone(images)
```

至少返回：

```text
patch feature map
```

以及可选：

```text
intermediate features
```

禁止把：

```text
CLS
register tokens
```

误当 patch features。

代码必须读取模型自身：

```text
num_prefix_tokens
```

而不是硬编码“前5个 token”。

### [AUTODL]

运行：

```text
python tools/check_backbone.py
```

对于：

```text
512×512
patch=16
```

期望 patch grid 大致：

```text
32×32
```

检查：

```text
NaN = 0
feature shape correct
weights loaded
GPU memory reasonable
```

---

# 29. Phase 5：Source-only Baseline

必须先得到可信 baseline。

模型：

```text
DINOv3-L
+
baseline semantic segmentation path
```

暂时：

```text
无 Query
无 Style
无 CQE
```

### [AUTODL]

训练：

```text
GTA5
```

先：

```text
500 iter smoke test
```

确认：

```text
loss下降
gradient正常
checkpoint正常
validation正常
```

然后：

```text
完整 schedule
```

第一版建议：

```text
40k iterations
crop 512×512
batch size 根据显存设定
AMP enabled
gradient accumulation if needed
```

---

# 30. Phase 6：Query-only

### [LOCAL]

实现：

```text
GroupedCausalQueryBank
QueryCrossAttention
QueryResidualHead
```

配置：

```text
queries_per_class = 3
alpha_init = 0
cross_attention_layers = 1
```

### 单元测试

必须验证：

```text
query output shape
delta logits shape
alpha=0时：
full output == baseline output
```

误差：

```text
< 1e-6
```

### [AUTODL]

训练：

```text
Baseline + Query
```

不使用：

```text
Style
CQE
```

实验名：

```text
B1_QUERY
```

验收：

```text
不应比 baseline 平均下降 > 0.5 mIoU
```

理想：

```text
+0.5 ~ +2
```

如果明显下降：

先检查：

```text
α
query scale
temperature τ
logsumexp
learning rate
```

不要立即增加更多 loss。

---

# 31. Phase 7：Style Intervention

### [LOCAL]

实现：

```text
StyleInterventionBank
```

第一版：

```text
original
photometric
fourier
```

所有 views：

```text
完全相同 geometric transform
```

### [AUTODL]

实验：

```text
B2_QUERY_STYLE
```

只：

```text
Query + style augmentation
```

损失：

```text
Lseg + λcf Lseg_cf
```

保存 style visualizations。

---

# 32. Phase 8：普通一致性控制实验

### [LOCAL]

实现：

```text
prediction_consistency.py
```

### [AUTODL]

实验：

```text
B3_PRED_CONS
```

：

```text
Query
+ Style
+ prediction consistency
```

这是 CQE 的最重要 control baseline。

---

# 33. Phase 9：CQE

### [LOCAL]

实现：

```text
causal_query_effect.py
```

必须提供 API：

```python
effect = model.get_query_effect(...)
```

不得让分析脚本直接访问内部私有 tensor。

### [AUTODL]

实验：

```text
B4_CQE
```

：

```text
Query
+ Style
+ CQE
```

第一目标：

```text
B4 > B3
```

第二目标：

```text
effect variance B4 < B3
```

---

# 34. Phase 10：Query Diversity

实验：

```text
B5_FULL
```

增加：

```text
Ldiv
```

如果提升：保留。

如果：

```text
< +0.2 mIoU
```

或导致训练不稳定：从最终方法移除。

论文不需要为了模块数量强行保留它。

---

# 35. 核心 Ablation Table

| Exp | Query | Style | Pred Cons. | CQE | Diversity | Avg |
|---|---|---|---|---|---|---|
| B0 | × | × | × | × | × | |
| B1 | ✓ | × | × | × | × | |
| B2 | ✓ | ✓ | × | × | × | |
| B3 | ✓ | ✓ | ✓ | × | × | |
| B4 | ✓ | ✓ | ×/✓ | ✓ | × | |
| B5 | ✓ | ✓ | ✓ | ✓ | ✓ | |

论文关键：

```text
B4/B5
vs
B3
```

---

# 36. Style Ablation

比较：

```text
Photometric only
Fourier only
Photometric + Fourier
```

后期可加入：

```text
MixStyle
```

但不是第一阶段。

不得一次加入 6~8 种 style mechanism。

---

# 37. Query 数量 Ablation

```text
R = 1
R = 2
R = 3
R = 4
```

观察：

```text
mIoU
active query diversity
query similarity
effect variance
```

默认预计：

```text
R=3
```

是比较合理起点。

---

# 38. Query Interaction Ablation

比较：

```text
No Query
Static Query
(no image interaction)
Bidirectional Self-Attention
(QPrompt-like)
One-way Query→Image Cross Attention
(Ours)
```

这一张表非常重要。

它验证结构创新是否成立。

---

# 39. Effect Distillation Ablation

比较：

```text
Feature L2 Consistency
Prediction KL Consistency
Query Feature Consistency
Query Effect Consistency
```

这是论文最重要的方法论实验之一。

理想结论：

```text
Query Effect Consistency
>
直接 Feature/Query 对齐
```

这才支持：

```text
distill effect instead of representation
```

---

# 40. 因果分析实验

除了 mIoU，必须记录：

```text
Feature Style Variance
Query Style Variance
Query Effect Style Variance
```

定义：

```text
VF
VQ
VE
```

重点分析：

```text
VE ↓
DG mIoU ↑
```

最终绘制：

```text
x-axis:
Query Effect Variance

y-axis:
Unseen-domain mIoU
```

观察负相关。

---

# 41. Query Effect 可视化

对：

```text
road
car
person
vegetation
```

输出：

```text
Original image
GT
baseline prediction
style view
query effect heatmap original
query effect heatmap style
ours prediction
```

理想：

```text
Style变化明显
Effect region相对稳定
```

---

# 42. Scaling 验证

主实验：

```text
DINOv3-L
```

最终扩展：

```text
DINOv3-B
DINOv3-L
```

算力充足：

```text
DINOv3-H+
```

7B：只作为最后 scaling validation。

研究问题：

```text
VFM规模越大，
Query Effect Style Variance是否自然下降？

CQE增益是否仍然存在？
```

不要把 DINOv3-7B 当主训练平台。

---

# 43. 旧 DINOv2/REIN 验证

后期增加：

```text
DINOv2-L + REIN
```

将核心：

```text
CQE training objective
```

迁移到 REIN。

目的：

```text
证明 CQE不是DINOv3特有trick
```

不要求把全部主模型结构移植。

---

# 44. 强基线比较

至少应包含：

```text
Source-only
RobustNet
SHADE
REIN
SoMA
Causal-Tune
QPrompt-R1
```

不要盲目复制论文数字。

必须记录：

```text
resolution
backbone
training source
pretraining
target datasets
```

确保 protocol 一致。

---

# 45. AutoDL RTX 4090D 24GB 策略

主实验：

```text
crop = 512
AMP = True
batch = 1 initially
gradient accumulation
```

Style views 不要默认：

```text
B×3
```

一次送 DINOv3。

使用：

```text
sequential_style_forward = true
```

```text
view0 forward
view1 forward
view2 forward
```

逐个计算并累积 loss。

目的：

```text
降低峰值显存
```

如果后续显存足够再 stack。

---

# 46. Backbone 冻结策略

Phase 5~10：

```text
DINOv3 frozen
```

训练：

```text
head
query bank
query attention
query projections
```

完成主要结论后增加：

```text
last-block trainable
```

或者：

```text
LoRA last 4 blocks
```

作为 ablation。

不要一开始 full fine-tune。

---

# 47. Optimizer 初始建议

初始：

```text
AdamW
```

不同模块 LR：

```text
query modules:
1e-4

segmentation head:
1e-4

backbone:
0
```

如果 LoRA：

```text
1e-5 ~ 5e-5
```

weight decay：

```text
0.01
```

scheduler：

```text
poly
```

或 EoMT 当前默认 schedule。

不要在第一个实验阶段大范围搜索超参数。

---

# 48. 随机种子

开发：

```text
seed = 0
```

正式主结果：

```text
0
1
2
```

报告：

```text
mean ± std
```

至少最终主表做三 seed。

所有 ablation 不要求三 seed，除非差距：

```text
< 0.5 mIoU
```

---

# 49. 每次实验必须记录

自动写：

```text
experiment ID
date
git SHA
hostname
GPU
PyTorch
CUDA
backbone
pretrained checkpoint hash
source
targets
config
seed
trainable parameter count
total parameter count
peak VRAM
training iterations
Cityscapes mIoU
BDD mIoU
Mapillary mIoU
average
```

保存：

```text
experiments/<EXP_ID>/
```

大型 checkpoint 不进 GitHub。

---

# 50. 实验命名

统一：

```text
A0_DINOV3L_BASE
A1_QUERY
A2_QUERY_STYLE
A3_PRED_CONS
A4_CQE
A5_FULL
```

secondary：

```text
S1_STYLE_PHOTO
S2_STYLE_FOURIER
Q1_R1
Q2_R2
Q3_R3
Q4_R4
```

---

# 51. Git 工作流

开发：

```text
main
├── dev/backbone
├── dev/query
├── dev/style
├── dev/cqe
└── dev/analysis
```

本机每完成一 Phase：

```text
commit
push
```

AutoDL：

```text
checkout exact SHA
```

正式实验禁止直接追：

```text
latest main
```

而必须：

```text
git checkout <commit_sha>
```

---

# 52. AutoDL 结果如何回 GitHub

实现：

```text
scripts/publish_result.sh
```

它只允许提交：

```text
config
metrics.json
summary.md
commit.txt
small figures
```

禁止：

```text
checkpoint
full log
dataset
```

---

# 53. Codex 工作机制

每次执行项目时：

1. 阅读：

```text
README.md
AGENTS.md
docs/METHOD.md
docs/AUTODL_NEXT.md
```

2. 检查：

```text
git status
```

3. 只完成当前 Phase。

4. 不提前实现三四个后续 Phase。

5. 修改完成：

```text
pytest
```

6. 更新：

```text
CHANGELOG.md
docs/AUTODL_NEXT.md
```

7. commit。

8. 如果下一步需要 GPU：停止本地实现，并在：

```text
docs/AUTODL_NEXT.md
```

写出完整 AutoDL 命令。

---

# 54. Codex 禁止行为

禁止：

```text
猜测AutoDL训练结果
为了让测试通过而删除失败测试
同时改变backbone、loss、dataset和optimizer
未经确认重写大量upstream源码
把旧DAFormer/MRM代码整体复制进新工程
把Causal-Tune DCT模块直接加入本项目
把QPrompt GRQA直接复制作为“创新”
```

---

# 55. 项目决策树

如果：

```text
A1 Query
<
A0 - 0.5 mIoU
```

则暂停后续，修 Query branch。

如果：

```text
A2 Style
<
A1 - 1.0
```

检查 Style Intervention。

如果：

```text
A4 CQE
≈
A3 Pred Cons
```

差：

```text
< 0.3
```

则不能直接宣称 causal improvement。

优先检查：

```text
effect definition
normalization
style strength
query activation
```

如果：

```text
VE明显下降
但mIoU不升
```

说明：

```text
当前定义的“稳定effect”
不一定是task-causal effect
```

需要重新设计 effect：

```text
只计算GT region
boundary-aware
positive-negative margin
```

而不是盲目调 λ。

如果：

```text
Query Effect不稳定
且Query本身无语义
```

启用：

```text
Lqproto
```

如果最终：

```text
CQE无法稳定优于Pred Consistency
```

则停止方向 1，转向备选：

```text
Causal Q/K Head Gating Adapter
```

不要继续堆模块救实验。

---

# 56. 项目最低成功标准

最低论文价值条件：

```text
DINOv3-L
GTA → C/B/M

Ours
平均至少明显优于
同backbone source-only
```

更重要：

```text
CQE
>
Prediction Consistency
```

且：

```text
Query Effect Variance ↓
```

---

# 57. 理想结果

理想目标：

```text
Source-only
→ +2~4 Avg mIoU

strong style baseline
→ +1~2 Avg mIoU

Prediction Consistency
→ CQE further +0.5~1.5
```

这些数字只是项目目标，不得写成已有结果。

---

# 58. 最终论文贡献应控制为三个

不要写五六个贡献。

## Contribution 1

提出：

```text
Counterfactual Query Effect
```

将 DG 中需要稳定的对象从：

```text
absolute representation
```

改为：

```text
prediction causal effect
```

## Contribution 2

提出：

```text
Grouped Context-Adaptive Causal Query Module
```

通过多个 class-specific query routes 建立可以进行：

```text
do(Q)
```

干预的结构。

## Contribution 3

提出：

```text
Style Intervention based CQE Distillation
```

通过：

```text
do(S)
```

生成语义等价的 counterfactual environments，并显式稳定 Query Effect。

---

# 59. 最终论文故事

完整逻辑必须是：

```text
Domain shift
↓
appearance changes
↓
普通feature consistency
可能产生over-alignment
↓
绝对representation不一定需要相同
↓
真正应稳定的是：
semantic mechanism对预测的作用
↓
引入context-adaptive causal query
↓
定义do(Q)
↓
引入style intervention定义do(S)
↓
测量不同style下Query Effect
↓
Distill invariant causal effect
↓
提高unseen-domain DGSS
```

不要把故事写成：

```text
我们用了DINOv3
+
Query
+
Fourier
+
Consistency
+
几个Loss
```

---

# 60. 第一轮实际执行顺序

Codex 接收到本方案后，只执行：

```text
Phase 1
```

即：

```text
建立CausalQ_DG目录
建立Git配置
建立manifest
建立tests
建立README
建立AGENTS
建立AUTODL_NEXT
```

不要立即实现完整 CQE。

完成后输出：

```text
1. 新增文件
2. 文件用途
3. pytest结果
4. git diff摘要
5. 下一步AutoDL命令
```

然后停止。

等待 AutoDL Phase 2/3 的真实输出后，再继续。

---

# 61. 关键工程决策补充

主线工程底座建议参考 **EoMT-DINOv3**，而不是强行把 DINOv3 接入旧 REIN 栈。

EoMT 已支持 DINOv3，并且其“query + image patch 共同进入 ViT”的设计与 context-adaptive query 方向高度相关。

REIN/Causal-Tune 的角色主要是：

- GTA5 / BDD100K / Mapillary / ACDC 数据转换；
- DGSS 数据协议；
- 结果对照；
- 后期 plug-in baseline。

因此推荐工程组合：

```text
EoMT / DINOv3
        +
CausalQ_DG 自定义模块
        +
REIN / Causal-Tune 数据协议参考
        +
SoMA 强基线参考
```

---

# 62. Codex 使用建议

不要要求 Codex 一次实现整个项目。

第一轮只执行：

```text
严格执行第 60 节，只完成 Phase 1，完成后停止。
```

后续采用：

```text
本机 Codex 开发
↓
GitHub commit/push
↓
AutoDL checkout exact commit
↓
GPU实验
↓
返回真实日志/结果
↓
本机 Codex继续下一Phase
```

这一工作流贯穿整个项目。

# END OF PROJECT SPECIFICATION
