# 20261020_Diagnostic_Plan

## 1. 任务与执行原则

本文件是 CausalQ-DG 下一轮实验的 Codex 工作指导书，制定日期为2026-10-20。项目地址为 <https://github.com/Ealea1028/CausalQ_DG>。目标是在已有实现和正式结果上，研究**去掉 residual 中心化硬约束之后，选择性稳定 Anchor effect 或 Residual contribution 的行为与收益**，并补足幅值、相关性、梯度和 Query responsibility 诊断。

执行时，先完成第2节的项目扫描，再逐阶段实施第12节。每次只推进一个阶段。本机完成实现、CPU测试和结果分析；一旦下一步依赖真实数据、DINOv3权重或GPU，就提交并推送Git，更新 `docs/AUTODL_NEXT.md`，给用户完整AutoDL命令后停止。用户返回结果后再继续下一阶段。不得只复述方案，也不得把本文件中的拟新增入口当作已经存在的程序。

本文件明确提出一项**新的探索性研究**：不再将“未加约束的 Anchor 原始方差必须小于 Residual”作为本轮 Anchor-only 实验的启动门槛。新问题是“对哪个分量施加一致性约束更有效，以及中心化是否改变这一结果”。原研究的停止结论、原始指标和已冻结记录继续保留，不覆盖、不改成通过。当前任务只要求生成指导书；实际实施应由后续执行本指导书的Codex完成。

本轮不加入 Necessity Guard、Fourier训练视图、prototype、额外attention、diversity、新Query数量或其他模块。旧M4不自动恢复。先通过单机制消融回答选择性稳定问题。

## 2. 开始实施前先扫描项目

工作空间当前为 `D:\CrossDomainProject\CausalQ_Project`。先查看 `git status --short`、当前分支、最近提交和 `git remote -v`，保留用户未跟踪文件及未提交修改。若项目此后有更新，以实际扫描到的最新状态为准，记录相对本方案的差异。

按顺序阅读这些文件，而不是重建已有工程：

1. `AGENTS.md`、`README.md`、`docs/METHOD.md`、`docs/AUTODL_NEXT.md`。
2. `docs/CODEX_CURRENT_STATE.md`、`docs/STRICT_DG_FINAL_REPORT.md`、`experiments/registry.csv`。
3. `CausalQ_DG_1016.txt`，重点阅读分解定义、M0–M4、稳定性门槛及停止条件。
4. `configs/strict_dg/formal/{m0,m1,p1,m2,m3}_seed0.yaml` 与 `causalq/protocols/formal_source.py`。
5. `causalq/models/query_segmentor.py`、`causalq/losses/causal_query_effect.py`、`causalq/interventions/style.py`。
6. `tools/train_formal_source.py`、`tools/full_cqe_source_update.py`、`tools/trained_anchor_stability.py`、`tools/evaluate_target.py`。
7. `causalq/analysis/query_components.py`、`tools/analyze_query_components.py` 及相应测试，判断哪些指标可直接复用、哪些还缺失。
8. `experiments/STRICT_DG_PHASE25/`、`experiments/STRICT_DG_PHASE26/`、`experiments/STRICT_DG_PHASE27/` 中的小型证据与冻结合同。

扫描完成后，先写一份简短 `experiments/DIAGNOSTIC_20261020/implementation_inventory.md`，记录当前SHA、可复用模块、缺失接口、已有检查点、下一阶段及本机/实例分工。只定位和阅读相关模块，不重复遍历数据集或重做已通过的GPU工程核验。

### 2.1 当前已知进度

本方案依据的仓库快照为 `b43f683001a1131cbd49b459b65e51510f5130da`，旧研究已完成Phase27。以下均为seed0、40000更新的最终结果：

| 旧名称 | 中心化 | 一致性机制 | GTA5 holdout mIoU | Cityscapes mIoU |
| --- | --- | --- | ---: | ---: |
| M0 | 关闭 | 无 | 73.8862% | 65.2091% |
| M1 | 开启 | 无 | 73.8542% | 65.6598% |
| P1 | 开启 | 预测KL | 73.8441% | 65.7621% |
| M2 | 开启 | Full CQE | 73.7182% | 65.8755% |

正式M1源域2497张的原始Anchor/Residual方差比为27.6012，0/2497张Anchor更低。旧M3、M4未训练。该证据反对“Anchor天然更稳定”，没有直接检验“Anchor-only正则化是否有用”。新实验不得把未训练的旧M3填写为零分或失败的性能结果。

Cityscapes结果已经公开给研究者。本方案因此不再声称新设计是在完全未见目标域反馈的条件下提出。训练、诊断、权重选择仍只使用源域；所有新配置必须在新目标域评估前冻结，全部报告，不因目标域结果追加配置或挑选检查点。

## 3. 本机和AutoDL实例的边界

| 工作 | 本机Codex | AutoDL实例 |
| --- | --- | --- |
| 扫描代码、实现模块、修改配置和文档 | 执行 | 不直接编辑源码 |
| 小型合成张量测试、完整pytest | 执行，隐藏CUDA | 只保留必要运行保护 |
| 数据集规模检查、真实图片诊断、干预图导出 | 不执行 | 执行 |
| DINOv3加载、梯度诊断、正式训练、评估、显存测量 | 不执行 | 执行 |
| 表格、科学图、接收小型JSON/CSV、统计分析 | 执行 | 生成原始记录 |
| 原始数据、权重、检查点、完整运行目录 | 不进入Git | 保留在实例持久目录 |

GitHub是代码、配置、协议、清单和小型结果元数据的唯一传递入口。AutoDL运行完整40位SHA；不能在实例上临时改Python文件、YAML或损失权重。网络或CUDA问题修复也先在本机提交，再切换到新SHA。

建议实施分支为 `codex/diagnostic-20261020`。旧分支及旧运行目录保持可追溯。新输出放到 `/root/autodl-tmp/outputs/CausalQ_DG/diagnostic_20261020/` 下，子目录包含阶段、变体、种子和短SHA，禁止复用旧M0/M1/M2输出路径写入新结果。

沿用已接受的实例环境：`/root/autodl-tmp/envs/causalq-dg/bin/python`，Python3.12.3、torch2.7.0+cu126、transformers4.56.1、RTX4090D。数据根目录 `/root/autodl-tmp/datasets`，预训练根目录 `/root/autodl-tmp/pretrained`。记录实际环境；若环境已改变，明确其与旧实验不可直接视为完全匹配，不静默安装或替换版本。

## 4. 实验矩阵及唯一允许的配置差异

### 4.1 两个因素

中心化组C使用原定义：`q[c,r] = a[c] + e[c,r] - mean_r(e[c,r])`。

非中心化组U使用：`q[c,r] = a[c] + e[c,r]`，即 `centered_residuals: false`。

去掉中心化意味着不再在前向、优化器更新后或诊断中强行令 residual 均值为零。不能通过梯度投影、均值惩罚、初始化后反复重分配、隐式detach或后处理重新引入该约束。既有Query特征归一化和效应图L2归一化仍保留，它们与residual中心化是不同操作。

一致性因素包含无约束、预测KL、Full、Anchor-only、Residual-only五种。每个有约束实验只加入一种一致性项，不叠加KL与CQE。

| 本轮名称 | 中心化 | 一致性项 | 对应旧实验 | 实施方式 |
| --- | --- | --- | --- | --- |
| U0 | 关闭 | 无 | 旧M0 | 复用已完成seed0检查点与性能，补诊断 |
| UP | 关闭 | 预测KL | 旧P1的非中心化对应项 | 新训练 |
| UF | 关闭 | Full CQE | 旧M2的非中心化对应项 | 新训练 |
| UA | 关闭 | Anchor-only CQE | 新选择性实验 | 新训练 |
| UR | 关闭 | Residual-only CQE | 新选择性实验 | 新训练 |
| C0 | 开启 | 无 | 旧M1 | 复用检查点与性能，补诊断 |
| CP | 开启 | 预测KL | 旧P1 | 复用检查点与性能，补诊断 |
| CF | 开启 | Full CQE | 旧M2 | 复用检查点与性能，补诊断 |
| CA | 开启 | Anchor-only CQE | 机制对应旧M3，但属本轮新协议 | 新训练，独立命名 |
| CR | 开启 | Residual-only CQE | 新匹配对照 | 新训练 |

主任务是U组五项，其中U0已完成，故需新增UP/UF/UA/UR四次正式训练。CA/CR是补齐“中心化是否影响选择性稳定”的匹配对照，排在U组之后，共计六次新增40000步训练，即240000次新增模型更新。不训练两次相同的无中心化基线；把旧M1简单改为无中心化且不加损失就等同于U0，不另设重复实验。

如果实际预算只允许U组，完成U组后也可形成选择性稳定结果，但必须把CA/CR标为未完成，不能声称已经识别“中心化×选择性约束”的交互效应。不得看过U组目标分数后才选择是否补CA/CR；预算范围应在任何新目标域评估前确定并写入冻结合同。

### 4.2 沿用上一轮训练配置

所有新增实验从同一个预训练backbone与相同种子初始化新head开始，不能从旧M2或某个已训练模型继续优化。与seed0旧实验匹配的初始化状态及训练采样序列应能复现，不能仅设置一个相同seed却因新增构造顺序多消耗随机数而改变初始化。

| 配置项 | 固定值或要求 |
| --- | --- |
| 数据 | 既有GTA5源域训练22469张、holdout2497张，原Git冻结清单不变 |
| Backbone | 冻结DINOv3-L，原权重文件及SHA不变；中间层6/12/18/24 |
| Decoder | 原256通道decoder，dropout0.1 |
| Query | 每类2个，static interaction，cross-attention层数0，temperature0.07，logsumexp聚合 |
| Alpha | 初始0，可训练；不能固定0，也不改为非零以制造诊断梯度 |
| 训练 | seed0，40000更新，batch1，梯度累积1，workers0，BF16 |
| 几何增强 | 原512×512裁剪，random scale[0.5,2.0]，水平翻转概率0.5，原有效像素规则 |
| 外观训练视图 | original + 一个photometric，顺序前向、几何与标签对齐 |
| Photometric | brightness/contrast/saturation[0.7,1.3]，gamma[0.8,1.2]，temperature[0.9,1.1]，grayscale概率0.1 |
| 优化器 | AdamW，lr1e-4，weight decay0.01，梯度裁剪1，poly power0.9 |
| CE | 原确定性交叉熵实现，两视图权重各1，无隐藏除2 |
| 一致性 | lambda1；CQE SmoothL1 beta1；预测KL温度1，方向原图到风格图 |
| 恢复 | 每1000步不可变segment，含优化器、RNG、sampler等完整恢复状态 |
| 模型选择 | 固定最终40000步，最终全源域验证，不按源域或目标域挑best |

源域验证仍为长边1024、保持比例、pad至16倍数。最终Cityscapes评估仍为原始1024×2048整图，无resize/crop/flip/TTA/sliding；同一冻结500张清单、19类trainIds、ignore255。

本轮先沿用seed0，不自动扩张种子预算。单种子结果只能作探索性点估计。若另行安排seed1/2，应在读取其目标域结果前冻结完整配对矩阵，不能只给seed0的最高分方法补种子。

## 5. 效应与损失的精确定义

### 5.1 Full、Anchor与Residual Contribution

固定图像x和模型，令 `F(q,x)` 表示当前Query head聚合及上采样后的输出，不包含base decoder logits。定义：

```text
E_full(x)   = alpha * F(q, x)
E_anchor(x) = alpha * F(repeat(a, R=2), x)
E_res(x)    = E_full(x) - E_anchor(x)
z_full(x)   = z_base(x) + E_full(x)
```

同一图像的三个效应复用同一次backbone前向、同一特征、同一投影层。Anchor-only路径仍重复同一个anchor两次并使用同一logsumexp，因此保留原聚合常数；不能把它换成单slot，否则会人为改变Full−Anchor差值。

`E_res`是输出空间差分贡献，不是把residual向量单独送入head的输出，也不等同于独立的Residual因果效应。Query投影归一化和logsumexp非线性使两者通常不同。所有图表必须使用正确名称。

现有 `_require_anchor_mode()` 禁止非中心化路径。实施时新增明确的本轮协议入口或受控组件接口，在保留static、无learned-null等必要条件的基础上支持U组。不得简单删除全部保护，或把旧M3的统计门槛改为始终通过。旧CLI、旧配置与旧记录语义保持兼容。

### 5.2 CQE与预测一致性

对S属于Full、Anchor、Residual之一：

```text
L_S = SmoothL1(N(E_S(photo)), stopgrad(N(E_S(original))))
L_total = CE(z_full(original), y) + CE(z_full(photo), y) + 1.0 * L_S
```

N沿用既有CQE实现：FP32、忽略255、每样本每类空间L2归一化，SmoothL1空间求和，最后对GT出现类别取平均；不是只在该类别的GT区域内计算。UP/CP用原预测KL替代L_S；U0/C0无一致性项。

UR/CR中的风格侧 `E_res = E_full - E_anchor` 两项均参与求导，不单独detach风格侧Anchor。原图参考整体停止梯度。该选项必须记录，不能事后切换：Residual-only loss可能同时更新anchor和residual参数，“选择性”指被约束的效应，不保证只更新某一组参数。UA/CA也通过共享投影层影响Full输出。

只比较效应形状会掩盖幅值下降，因此训练保持旧归一化损失，诊断必须同时保留原始幅值，不能仅报告loss或归一化方差下降。

### 5.3 非中心化分解的解释边界

对U组，`a -> a+t, e_r -> e_r-t`保持完整q不变，却可能改变Anchor-only和Residual contribution。因而Anchor并非由完整预测唯一识别的分量。必须记录anchor范数、residual均值范数、原始/中心化residual范数及完整q的范数；不得把不同实验的anchor直接解释为同一个已识别语义实体。

报告初始化和训练后的这类漂移，但不在前向或优化器中自动“修正”。可以额外对冻结模型做只读合成等价变换单元测试，演示Full输出不变而分解变化；不能将重参数化后的检查点用于本轮训练或主结果。

去中心化也会改变共享anchor与residual的梯度耦合和AdamW更新轨迹；即使学习率、初始化种子完全相同，两组优化几何也不相同。这是本轮参数化因素的一部分，不能把C/U性能差全部解释为语义稳定性改变，更不能私自调整某组学习率来抵消后再称“其余配置相同”。

## 6. 统一诊断样本与风格口径

效应统计、相关性和responsibility主诊断使用全部2497张固定GTA5 holdout，模型eval、dropout关闭、无参数更新。梯度诊断单独开启autograd，仍不执行优化器step。诊断不得访问目标图像或标签。

每张图像保存一个原图视图，加4个独立photometric复合干预视图，共S=5个视图。第一干预沿用旧 `intervention_seed(source_id)`；其余按固定salt、source_id、干预序号的SHA256生成32位seed，禁止使用Python内建hash。各模型与配置复用完全相同的样本、几何和干预参数。以独立RNG上下文生成诊断视图，不消耗正式训练RNG。

主诊断计算五视图的总体方差（分母S，不是S−1）。另保存“原图+第一干预”的两视图兼容指标，严格对应旧 `(original-photo)^2 / 4`。不能把两种方差混为同一统计量。

效应统计采用源域评估几何，对所有非ignore真实像素计算，排除padding。每图每类先汇总，再对图像和19类等权宏平均；同时给出仅GT出现类的分层结果。像素加权结果单独列出，禁止在实验间切换聚合方式。缺失区域、常数相关性和零能量归一化返回null并记录有效样本数，不能填0使结果显得更好。

## 7. 效应幅值与跨风格方差

对每个图像i、类别c、分量k属于Full/Anchor/Residual，取固定有效像素集合P。定义 `E_s(p)` 为第s个视图的有符号效应，`mu(p)=mean_s E_s(p)`。

| 指标 | 计算公式 | 含义 |
| --- | --- | --- |
| 跨风格平均绝对幅值 | `mean_s mean_p abs(E_s(p))` | 保留效应的绝对规模 |
| 跨风格联合RMS | `sqrt(mean_s mean_p E_s(p)^2)` | 整体能量尺度 |
| 每风格RMS | `r_s=sqrt(mean_p E_s(p)^2)`，逐视图记录 | 区分风格变化与总体幅值 |
| 原始跨风格方差 | `mean_p mean_s (E_s(p)-mu(p))^2` | 与幅值相关的绝对波动 |
| RMS归一化方差 | 对每个有效视图设 `T_s=E_s/r_s`，再计算同一方差公式 | 相对空间形状/符号变化 |

零能量阈值固定为RMS<=1e-6。遇到零能量视图，该图类的归一化方差记null并记录collapse/near-zero计数，不能将其解释为“归一化后完美稳定”。同时保留每视图signed mean，识别大常数偏置。可派生RMS跨风格CV，均值接近零时也记null。

这里的RMS归一化用于诊断；训练使用的空间L2归一化维持原定义。两者相差有效像素数量因子，不能直接比较数值。旧 `rms_normalized_pair_mse` 是两视图均方差；在相同有效条件下，它等于RMS归一化两视图总体方差的4倍。新列名必须明确区分。

至少输出按图/类/分量的原始CSV或JSONL、类别表、数据集宏平均、median和10/90分位数，以及Anchor/Residual原始方差比和归一化方差比。比值分母为零时为null；不通过任意epsilon制造极大或极小结论。

在同一参数化内，比较UA/UR/UF与U0、CA/CR/CF与C0的幅值和方差变化。出现“方差下降但RMS/绝对幅值同步大降”时，报告可能的幅值收缩，不仅写稳定性改善。下降比例全量报告，不预设一个事后挑选的塌陷阈值来筛选方法。

## 8. Anchor与Residual相关性及差分抵消

每张图每类每个视图，在相同有效像素上计算：

1. 空间Pearson相关 `corr(E_anchor, E_res)`，同时保存未去均值cosine与协方差。任一图为空间常数或近零方差时Pearson为null。
2. 跨风格相关：`corr(E_anchor(original), E_anchor(photo_j))`、Residual对应量，以及每个photo相对原图变化的 `corr(delta_anchor, delta_res)`。
3. 差分前后幅值：同时记录 `E_full`、`E_anchor`、`E_full-E_anchor` 的mean-abs、RMS、signed mean；输出Residual/Full、Residual/Anchor比值与有效计数。
4. 抵消指标 `RMS(E_full)/(RMS(E_anchor)+RMS(E_res))`，以及差分收缩指标 `RMS(E_full-E_anchor)/(RMS(E_full)+RMS(E_anchor))`。分母近零记null；低值只提示抵消/相似性，不等于无用或因果独立。

按每个像素的风格维度计算协方差，再验证：

```text
Var_style(E_full)
  = Var_style(E_anchor) + Var_style(E_res)
    + 2 * Cov_style(E_anchor, E_res)
```

这是由定义Full=Anchor+Residual得到的恒等式，应在相同mask和归约口径下成立。它有助于判断Full稳定是因为两个分量都稳定，还是变化相互抵消。不能把空间Pearson相关直接代入这条跨风格方差公式。

效应图统一转FP32，数据集矩与协方差归约使用FP64累加器。空间去均值后的RMS<=1e-6时Pearson记null；Fisher-z聚合将有效相关系数裁剪至[-1+1e-7,1-1e-7]，记录裁剪数、有效数并同时报告median。相关性保留逐图结果与分位数，不要只给一个混合不同类别、偏置和尺度的全局相关系数。

## 9. 每个Query对Anchor的梯度贡献

### 9.1 测量条件与范围

对每个最终检查点，在全部2497张holdout上执行只读梯度诊断。每图使用独立固定seed生成一次512×512几何裁剪，再生成原图与第一photometric视图；所有模型共享裁剪、翻转和干预参数。与效应主诊断的长边1024几何不同，结果中必须标明 `gradient_geometry=deterministic_512_crop`。不因某模型梯度低或某类别缺失重新选图。

在eval模式下开启autograd，保持backbone冻结；无optimizer、无step、无gradient clipping、无GradScaler缩放。分别分析 `L_CE_original`、`L_CE_photo`、该实验的一致性项以及它们的加权总和，使用相同前向图或等价重算，结束清除梯度。无一致性项时对应记录为not_applicable。

记录的是“总损失对类别c的Query组梯度”，包含softmax竞争导致的非GT类别影响；它不等同于“仅类别c GT像素损失的梯度”。按该裁剪中GT是否出现该类分层汇总，避免把两者混淆。无需额外运行19次完整类别损失反传。

### 9.2 Query梯度与共享Anchor梯度

保留实际进入head的复合Query节点，定义每个视图 `h[c,r] = dL/dq[c,r]`，分别记录各视图的范数与方向，再把各损失、视图对同一q的梯度贡献相加。不能用 `query_residuals.grad` 冒充 `dL/dq`，尤其中心化组两者不同。

对于仅通过完整q访问anchor的损失：

```text
G_anchor_via_q[c] = sum_r h[c,r]
U组：dL/de[c,r] = h[c,r]
C组：dL/de[c,r] = h[c,r] - mean_r h[c,r]
```

UA/CA的Anchor-only路径，以及UR/CR差分中的Anchor-only路径，会额外直接访问a。因此通用分解是：

```text
G_anchor_total = G_anchor_via_q + G_anchor_direct
```

实现中显式区分完整q路径和anchor-only路径的计算图；必要时对共享同值、不同节点的anchor副本取偏导，再验证其和等于真实共享参数梯度。不能强行令所有损失满足 `anchor.grad=sum Query梯度`。图分支内部须保持可微，不能为方便统计改变训练损失。

每类别至少记录：

- 每个Query的 `||h_r||`、每个raw residual参数梯度范数、`||G_via_q||`、`||G_direct||`、`||G_total||`。
- 同类Query梯度两两cosine、各Query与G_via_q的cosine、原图/风格图同slot梯度cosine。
- 范数占比 `||h_r||/sum_s||h_s||`，表示梯度规模，不表示向量和的真实占比。
- 有符号投影贡献 `p_r=<h_r,G_via_q>/||G_via_q||^2`；有效时sum_r p_r=1，可出现负值或大于1，反映抵消。
- 各Query相对G_total的投影、direct路径相对G_total的投影；两类贡献合计为1，而仅Query项未必合计为1。
- 抵消比 `||sum_r h_r||/sum_r||h_r||`、各路径有效/近零计数，以及alpha数值。

向量范数<=1e-12时相应cosine或投影为null。初始alpha=0可使Query梯度为0，这是既有初始化行为，不据此诊断Query永久失效。梯度范数只是局部敏感度；不能直接等同于AdamW实际更新贡献或因果必要性，优化器动量、预条件和weight decay不包含在上述分解中。

### 9.3 最小实现验证

CPU合成张量测试验证U组和C组的链式梯度恒等式、direct路径项、投影贡献和、零范数、detach方向及不改变参数/RNG。真实模型诊断在AutoDL上执行，保存状态前后摘要，证明只有autograd读取，没有训练更新。

## 10. Query responsibility全数据集统计

使用Query score原生patch网格，保留既有含temperature0.07的score，不再次除温度。对同一类别c的两个slot计算：

```text
rho[c,r,p] = softmax_r(score[c,r,p])
```

主统计在下采样到score网格的GT类别c有效像素上进行，nearest标签缩放仅服务该诊断，不改变训练或分割评估GT。对所有有效位置的补充统计独立输出。不能先上采样responsibility再把插值像素当成独立测量。

对整个2497张数据集、每个类别、每个slot、每个风格视图，累加责任总和及像素数，得到像素加权平均责任；另报告每图等权结果。至少保存：

| 指标 | 定义或要求 |
| --- | --- |
| 平均责任 | `mean rho_r`，同类所有slot之和为1 |
| 像素责任熵 | `mean_p[-sum_r rho_r log(rho_r)]` |
| 全数据集使用熵 | 对数据集平均责任向量计算熵，区别于上一行 |
| 归一化熵与有效Query数 | 分别除logR，以及exp(使用熵)，R=2 |
| Top1份额 | 并列最大值平分，保留原slot编号 |
| 风格责任变化 | mean absolute delta、两视图总体方差、Jensen–Shannon divergence及top1切换率 |
| 逐slot变化 | 每个photo与原图的平均责任差，及slot间重新分配情况 |

仅在同一检查点内部跟踪slot编号。不同训练实验的slot可交换，不能把UA的slot0与UR的slot0直接当作同一语义Query。跨实验可比较熵、最大责任、排序后的份额；如展示对齐slot，匹配规则需仅依赖源域并额外保留原编号结果。

responsibility不是效应强度或因果必要性：某slot责任很高，仍可能因alpha接近零而对最终预测影响很小。必须联合第7节幅值、第9节梯度及源域分割性能解释。

## 11. 随机保存三组干预图像

从冻结源域holdout清单排序后，用Python `random.Random(20261020).sample(ids, 3)` 无放回抽取，禁止按预测成功、类别完整或视觉效果更换。当前清单对应三个ID为 **18734、12488、08744**；实施时复核清单SHA和抽样结果并写入Git清单。这里本机只处理文件名，不读取图像。

AutoDL为每个ID保存一个目录，至少包含：

```text
source_original.png                  数据集原始RGB，可与输入尺寸不同
input_original.png                   实际诊断几何后的原图
photometric_01.png ... _04.png        本轮实际使用的四次复合photometric视图
contact_sheet.png                    原图与所有实际诊断视图的对照图
interventions.json                   source_id、几何、seed、逐项参数、尺寸、配置SHA
```

所有干预图以同一几何后的输入为基础，正确反ImageNet归一化、clamp到[0,1]后导出，不对各图做独立min-max或自动对比度处理。原始RGB、模型输入原图和干预图要标清，避免把resize/crop差异当作风格变化。可另存一组确定性训练512裁剪及其原图/photo配对，目录名明确区分evaluation和training几何。

上一轮正式训练实际只用复合photometric；YAML中出现Fourier参数不代表启用了Fourier。保持本轮训练一致，不默认增加Fourier。为展示不同干预方法，在三组目录另存brightness/contrast/saturation/gamma/temperature/grayscale六种单项演示图，均以同一input_original为输入，复用第一复合视图对应参数，标记 `diagnostic_only`；灰度单项演示可强制开启，但须标明不是实际随机采样结果。单项演示不进入训练和主稳定性统计。contact sheet分区展示“实际复合视图”和“单项方法演示”，避免混淆。

如果后续确实启用了其他方法，必须先更新协议与对照预算，再在这三组里补齐该方法的真实输入/输出。不能只生成更漂亮的替代干预图。保存实际训练或诊断调用的参数和张量结果；新增参数追踪接口不得改变原photometric随机采样顺序。

导出图、原始诊断明细留在AutoDL。用户可下载三组图供本机目视分析，下载文件默认忽略Git；只将样本ID、参数、文件哈希和必要的小型结果清单提交Git。

## 12. 分阶段实施与停止点

### Phase28 本机实现与旧检查点源域诊断

先实现新协议、组件访问接口、效应/相关性/梯度/responsibility统计和干预参数追踪。复用旧U0/C0/CP/CF四个检查点，各补一次完整新诊断，不重新训练。一个实例后台任务按模型顺序执行，并保存三组干预图。诊断变体顺序和输出目录在本机冻结。

停止点：代码与CPU测试通过、提交推送并给出完整SHA后，等待用户运行AutoDL。返回旧检查点诊断后，检查统计定义与必要状态约束，不把原始Anchor方差排序作为新UA/CA的禁止条件。不得通过图像美观或新目标分数更改主指标。

### Phase29 非中心化正式消融

冻结UP、UF、UA、UR四个配置和初始化/采样合同，顺序执行完整40000步正式训练。沿用既有1000步恢复机制，无额外20步、8步或多轮工程门槛。配置仅含本方案允许差异；若确有OOM/NaN或实现错误，停止返回日志，在本机修复，不静默降低batch、分辨率或权重。

阶段末对四个最终模型进行最终源域验证；不访问目标域。正式训练只返回简明运行进度，详细诊断不插入每个训练step，以免改变RNG、速度和优化过程。

### Phase30 中心化选择性匹配对照

按预先确定预算训练CA/CR，各40000步。新探索性协议允许这两个直接实验，不要求旧“Anchor天然更稳定”的门槛为真。仍沿用旧中心化定义，唯一改变为对应一致性分量，不能同时加入Necessity Guard。

若已在新目标域评估前声明只做U组，则跳过本阶段并明确缺少交互对照；不能将未执行写成负结果。

### Phase31 新模型最终源域诊断与分析

对全部新增最终模型执行第6–10节相同诊断。复用旧模型已完成的Phase28结果，不重复运行。返回后在本机生成每类/每模型结果表、幅值与方差散点图、相关性/抵消图、梯度方向与贡献图、responsibility分布图。

诊断与训练前后状态一致性是必要检查；不再要求把整份图片目录反复打包、解包、上传或人工确认才能推进。原始明细按行流式写出、在线聚合，成功任务保留可恢复的完成标记，防止大规模数据重复计算。

### Phase32 冻结最终评估并完整报告

先在Git冻结所有完成模型的最终检查点SHA、源域结果、架构/损失/几何、目标清单和评估规则，然后才允许AutoDL按同一口径评估新增模型。旧U0/C0/CP/CF目标结果直接引用，不为“凑同批次”重复运行。目标评估无梯度、无优化器、无调参。

所有新增模型都报告，不选最高分后再决定是否展示其他模型。新旧结果按参数化和机制分层；它们是已见旧目标反馈之后的探索性对照，不能伪装成完全独立确认实验。

主要比较在任何新增目标评估前固定：UA−U0、UR−U0、UA−UR、UA−UF、UA−UP，以及CA−C0、CR−C0；交互量为 `(UA−U0)−(CA−C0)` 和 `(UR−U0)−(CR−C0)`。每项同时给出源域/目标域mIoU差与诊断变化。单种子不作训练随机性显著结论，源域图像bootstrap若使用也只能描述固定模型对该样本集的变化，不能替代训练种子不确定性。

## 13. Codex实施文件与必要测试

以下是拟新增或扩展的接口，不是当前仓库已存在的命令。实际路径可按模块结构微调，但交付时必须给出真实入口与参数。

| 文件或目录 | 用途 |
| --- | --- |
| `causalq/protocols/diagnostic_20261020.py` | 新研究ID、矩阵、固定配置及旧检查点引用 |
| `configs/diagnostic_20261020/` | U/C各变体完整解析后的YAML，明确布尔中心化和效应分量 |
| `causalq/analysis/selective_stability.py` | 幅值、方差、相关性、协方差与抵消统计 |
| `causalq/analysis/query_gradients.py` | Query/Anchor路径梯度分解与贡献统计 |
| 既有 `query_components.py` | 复用responsibility，新增多视图和全数据集聚合 |
| 既有 `query_segmentor.py` | 受控支持非中心化Anchor-only，保留旧API行为 |
| `tools/run_diagnostic_20261020.py` | 各阶段CLI编排、模型/数据流式读取、恢复和汇总 |
| `scripts/run_diagnostic_20261020.sh` | exact-SHA后台启动、互斥锁、日志及退出状态 |
| `experiments/DIAGNOSTIC_20261020/` | 协议、映射、轻量接收结果、registry与阶段记录 |
| `docs/AUTODL_NEXT.md` | 当前唯一下一步AutoDL操作入口 |

必要CPU测试覆盖：中心化开关与原API兼容；相同q的Full输出一致性；residual为零时Full=Anchor及Residual=0；效应差分与方差协方差恒等式；归一化尺度变化/零能量；相关性常数与负相关；梯度链式规则/direct路径/零范数；责任归一化、熵、并列最大值、风格交换及缺失类；三组样本抽取和随机参数追踪不改变RNG；只读诊断状态不变；恢复目录拒绝错误配置或错误SHA。

本机完整测试沿用现有可用环境，例如：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.test-deps-strict-dg')
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
& E:/Anaconda/envs/pytorch/python.exe -m pytest -p no:cacheprovider --basetemp=.test-tmp-strict-dg-diagnostic-20261020
```

若路径或环境已改变，先定位实际解释器。不得将“测试未执行”记成“通过”，不得删失败测试。完成一个模块的合成测试后再接入实例实际数据；避免重复GPU smoke门槛。

## 14. AutoDL命令交付规范

### 14.1 第一次交付

Codex完成对应阶段后，必须先推送代码，再将下面的占位SHA替换为本阶段**真实完整SHA**。新launcher应实现后台模式、run lock、返回PID、日志路径和结果目录，且同一命令可恢复；内部子shell失败不能关闭用户交互终端。下列模板在拟新增脚本实现并推送后才能使用，不能直接拿本方案文档提交的SHA启动不存在的程序。

```bash
bash <<'BASH'
set -euo pipefail
EXPECTED_SHA=REPLACE_WITH_IMPLEMENTED_40_CHARACTER_SHA
[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo '请替换为Codex给出的完整实现SHA'; exit 2; }
SOURCE_REPO=/root/autodl-tmp/CausalQ_DG
TASK_REPO=/root/autodl-tmp/CausalQ_DG_diag20261020
git -C "$SOURCE_REPO" fetch origin codex/diagnostic-20261020
if [[ ! -e "$TASK_REPO" ]]; then
  git -C "$SOURCE_REPO" worktree add --detach "$TASK_REPO" "$EXPECTED_SHA"
else
  test -z "$(git -C "$TASK_REPO" status --porcelain)" || { echo '诊断工作区有修改，请保留并回报'; exit 3; }
  git -C "$TASK_REPO" checkout --detach "$EXPECTED_SHA"
fi
cd "$TASK_REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
export PY=/root/autodl-tmp/envs/causalq-dg/bin/python
export CAUSALQ_DATA_ROOT=/root/autodl-tmp/datasets
export CAUSALQ_PRETRAINED_ROOT=/root/autodl-tmp/pretrained
export CAUSALQ_TRAINING_OUTPUT_ROOT=/root/autodl-tmp/outputs/CausalQ_DG
export CAUSALQ_OUTPUT_ROOT=/root/autodl-tmp/outputs/CausalQ_DG/diagnostic_20261020
bash scripts/run_diagnostic_20261020.sh "$EXPECTED_SHA" baseline-diagnostics
BASH
echo '本次启动检查结束，交互终端保持打开。'
```

脚本内部固定 `CUDA_VISIBLE_DEVICES=0`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、离线权重模式、原确定性/TF32选项与线程配置。不能覆盖用户shell的HOME或CODEX_HOME。切换SHA前如有本研究worker运行，先停止操作并回报，不能让正在运行的任务读取被替换源码；旧项目使用独立worktree不受影响。

### 14.2 后续阶段

每个后续阶段先由Codex给出新的完整提交SHA，重复相同的fetch、工作区clean及checkout步骤，再启动本阶段模式。模式约定如下，实际交付必须替换为已实现的真实参数：

```text
baseline-diagnostics       Phase28，旧四模型诊断和三组干预图
train-uncentered           Phase29，UP/UF/UA/UR正式训练及源域验证
train-centered-controls    Phase30，CA/CR正式训练及源域验证
final-source-diagnostics   Phase31，新增最终模型完整源域诊断
target-evaluation          Phase32，冻结合同对应的新增模型最终目标域评估
```

每次交付还必须提供无占位路径的日志查看、完成状态查看和结果导出命令。监控只读取日志，不在用户shell顶层使用 `set -e` 或 `exit`。结果导出不自动退出终端，不把整份检查点打印为base64。

不把五个阶段无条件串联为一次无人审阅的全流程；本机需接收每阶段结果、修复必要问题并提交下阶段，始终满足一次只做一个项目阶段的工作规则。

### 14.3 预估工作量

完整矩阵共十个最终模型，其中四个复用旧训练、六个新增；新增训练为240000次更新。五视图主诊断共 `10×2497×5=124850` 次图像前向，梯度诊断另有 `10×2497×2=49940` 次图像前向及head反传。效应、相关性与responsibility从同一批主诊断输出计算，不各自重新跑backbone。新增目标域评估为六个模型各500张，共3000次图像推理。

这些是完整单种子矩阵的工作量，不承诺未经测量的小时数。若只做U组，需在新目标评估前明确删除CA/CR预算。不能因诊断较慢而暗中把“全2497张”改成抽样后仍称为全数据集；如用户另行要求抽样，应记录范围并保留原统计口径。

## 15. 结果文件、恢复与验收

每阶段保存 `run_manifest.json`、完整SHA、配置SHA、源清单SHA、检查点SHA、环境、几何、干预seed/参数、开始结束时间、退出码及完成标记。复用旧检查点时记录原训练producer SHA与本轮诊断producer SHA，不能把旧训练归到新提交。

诊断原始记录建议按以下结构分文件流式写出；准确列名可由实现统一，但定义必须对应本方案：

```text
effect_metrics.jsonl          image/class/component/view幅值与方差
correlation_metrics.jsonl     同风格/跨风格相关、协方差及差分抵消
gradient_metrics.jsonl        image/class/slot/loss/path梯度及方向贡献
responsibility_metrics.jsonl  image/class/slot/view责任与有效像素数
summary.json                  全数据集聚合、null计数、状态与限制
class_summary.csv             按类统计
intervention_examples/        三组样本图与参数
SHA256SUMS                    本阶段导出清单
```

定义每个有效记录键及断点游标，断点恢复时不得重复计数；恢复需匹配模型、配置、样本、几何和风格合同。梯度与效应统计不要长时间保留全部GPU张量。不同checkpoint/variant以独立进程顺序执行，复用一次图像的backbone特征得到Full和Anchor，避免重复加载与多余前向。

本机只接收分析所需JSON/CSV、日志末尾及三组图片；检查点和全量运行目录留在实例。原始逐图文件过大时实例生成精确摘要，按需传输，不强制反复传回所有原始数据才能推进。对尚未传回的明细明确区分“实例计算、摘要已核对”与“本机已独立重算”。

本轮结项应能回答：去中心化后哪种效应约束带来何种分割变化；原始方差变化中有多少伴随幅值收缩；Anchor/Residual是否存在显著抵消现象；slot梯度是同向累积还是相互抵消；责任分布是均衡、单slot主导还是随风格切换。答案允许为负或不确定，不要求制造Anchor-only最优的故事。

## 16. 交给Codex的启动指令

将本文件放在项目根目录后，可直接向Codex发送：

> 请以 `20261020_Diagnostic_Plan.md` 为本轮工作指导书。先扫描当前仓库和最新进度，保留旧研究记录与用户文件，再实施Phase28的本机部分。新研究允许检验Anchor-only和Residual-only，不沿用旧Anchor原始方差排序作为启动门槛。保持上一轮训练配置与数据划分，明确非中心化分解不唯一及目标域已见反馈。完成必要CPU测试、提交推送、更新AUTODL_NEXT后，在首次需要AutoDL时停止，给出真实完整SHA及可直接复制的启动、监控和导出命令。等我返回AutoDL结果后继续下一个阶段，不重复无意义的工程核验，也不要直接在AutoDL修改源码。

本指导书本身不等于实验已执行、脚本已实现或新结果已产生。实施开始后，应把新研究边界写入METHOD和当前状态文档；历史冻结合同及原始证据保持不变。
