# AutoDL数据盘清理建议

## 2026年10月21日报告分析

已收到 `diagnostic_disk_report.json`，SHA256为 `b376aa4a2a6416e3a8559625d37f6b7e554b549daaa70c7870a3143fb936e58e`。这是生成报告时的快照，当前诊断仍会写入数据，实时空间以实例上的df为准。

| 项目 | 报告占用或容量 |
| --- | ---: |
| 数据盘总量 | 155.00 GiB |
| 已用 / 空闲 | 143.38 / 11.62 GiB |
| datasets | 69.66 GiB |
| pretrained | 3.72 GiB |
| CausalQ_DG outputs | 27.50 GiB |
| pip缓存 | 约25 MiB，收益很小 |
| 四组旧正式实验的156个非最终state | 合计6.64 GiB文件大小 |

四个精确最终检查点均被报告为存在，合计约174 MiB；它们需要继续保留。该文件报告并未证明检查点内容哈希，本地也没有读取远程state。清理工具会在实例上核验四个最终state的SHA，再允许删除候选。

本次最明确的候选是156个旧中间state：每组39个、每组约1.66 GiB，结束步数不超过39000。文件精确清单冻结在 `experiments/DIAGNOSTIC_20261020/cleanup_plan_20261021.json`。删除它们会失去旧完整分段恢复能力，最终40000步推理和当前Phase28诊断不依赖它们。保留小型日志、配置、segment完成记录和formal报告。

新增 `tools/prune_closed_intermediates.py` 默认只预览；只有明确传入 `--apply` 才删除。它拒绝最终检查点、清单外文件、39000步之后的状态、路径越界、软链接、大小或硬链接数不符的候选；存在旧正式训练进程时拒绝清理。删除前后核验最终state，写入逐文件审计日志，已经不存在的候选可重复运行。

在单独的新代码工作区运行，保留当前 `1086bc3` 诊断工作区和后台任务。预览与可选执行命令：

```bash
# 默认预览：不删除、不创建清理日志目录。
/root/autodl-tmp/envs/causalq-dg/bin/python -m tools.prune_closed_intermediates \
  > /root/autodl-tmp/cleanup_preview_20261021.json
cat /root/autodl-tmp/cleanup_preview_20261021.json
```

仅在接受失去旧中间状态恢复能力、已完成需要的备份后，单独执行：

```bash
/root/autodl-tmp/envs/causalq-dg/bin/python -m tools.prune_closed_intermediates --apply
df -hT /root/autodl-tmp
```

工具保留GTA5、Cityscapes、DINO权重、四个最终state、所有小型训练记录和当前诊断输出。审计日志在 `outputs/CausalQ_DG/cleanup_20261021/`。并发诊断仍在写文件，因此df变化不必等于删除文件的合计大小，不承诺当前一定增加至18.26 GiB空闲。

三大已列目录合计与文件系统已用量还相差约42.51 GiB，其中可能有未列目录或计量差异，不能直接认定为垃圾。需要进一步定位时，用 `du -x -h --max-depth=1 /root/autodl-tmp | sort -h` 查看；不要直接删除envs或其他项目。

## 原始通用建议

本轮Phase28需要旧M0/M1/P1/M2最终模型、DINOv3权重、GTA5源图像与trainIds标签；后续最终评估还需要Cityscapes。先查看实际占用，不根据目录名称推测能释放多少空间。工具只读，不执行删除。

## 先查看占用

在已拉取本轮代码的诊断工作区执行：

```bash
df -hT /root/autodl-tmp /root
/root/autodl-tmp/envs/causalq-dg/bin/python -m tools.report_diagnostic_disk > /root/autodl-tmp/diagnostic_disk_report.json
cat /root/autodl-tmp/diagnostic_disk_report.json
du -x -h --max-depth=2 /root/autodl-tmp/outputs/CausalQ_DG | sort -h | tail -n 30
```

报告包含真实字节数、四个受保护最终检查点、旧正式实验中非最终segment状态文件候选及其影响。`candidate_bytes`只表示候选文件总大小，不表示建议立即全部删除或一定释放同等物理空间。后台任务正在写入的目录不在清理范围内。

先确认挂载点：`/root/.cache/pip` 可能位于系统盘。清理系统盘缓存不会释放独立挂载的 `/root/autodl-tmp` 数据盘；只有缓存确实放在数据盘上时，才把它列为数据盘清理项。可对实际存在的缓存目录单独执行 `df -hT /root/.cache/pip` 判断。数据盘空间优先看重复压缩包和旧中间检查点。

## 清理优先级

| 优先级 | 文件类型 | 建议及影响 |
| --- | --- | --- |
| 1 | 位于所需清理磁盘上的pip下载缓存 | 先查看挂载点与缓存；不卸载已安装环境。若不再需要离线安装包，可清除pip cache |
| 2 | 已完整解压且有其他备份的数据集/权重压缩包、重复下载、临时导出zip | 确认解压后的有效数据与权重存在，再处理确切的重复文件；保留唯一的原始备份 |
| 3 | 已失败、无运行进程、无后续引用的重复attempt目录 | 先保留错误日志、配置及小型报告；当前阶段可恢复样本和任务完成目录需保留 |
| 4 | 旧四模型1000–39000步中间state.pth | 对已结项研究，保留最终40000步后可考虑转存或删除中间状态；原分段恢复将不再可用 |
| 5 | 早期smoke/pilot的重复完整模型、旧工程输出 | 根据占用报告逐项确认引用，优先转存；不能把整个phase目录一键清空 |

pip缓存的查看与可选清理命令：

```bash
/root/autodl-tmp/envs/causalq-dg/bin/python -m pip cache info
# 确认不再需要离线安装缓存后，可执行：
# /root/autodl-tmp/envs/causalq-dg/bin/python -m pip cache purge
```

不要直接清空HuggingFace缓存：先检查预训练目录是否依赖其软链接/快照；删除缓存可能同时破坏DINOv3本地权重。不要删除正在使用的conda环境，也不把多个环境目录视为普通缓存。

## 必须保留

- `/root/autodl-tmp/pretrained/dinov3_vitl16/` 及其依赖的真实权重文件。
- `/root/autodl-tmp/datasets/gta5/images/images/`、`labels_trainIds/`，固定22469/2497划分所需全部源数据。
- `/root/autodl-tmp/datasets/cityscapes/` 的既有500张val图像与trainIds标签，用于本轮最终评估。
- `experiments/STRICT_DG_PHASE26/final_evaluation_freeze.json` 引用的四个精确最终检查点。它们位于各自 `segments/39000_40000_<id>/state.pth`，不是任意名为latest/best的文件。
- 旧训练的 `initialization.json`、`metadata.json`、`config.yaml`、formal/pair/source/target报告、训练日志及segment完成记录。这些通常很小，保留其证据与删除记录。
- 当前新研究 `outputs/CausalQ_DG/diagnostic_20261020/`；尤其未完成任务的逐图恢复记录。接受并备份结果前不清理。

旧原始labelIds目录若只为生成trainIds使用，未来也可能转存；但需确认原始数据备份和重映射需求，不列为第一批清理项。方案不默认删除BDD/Mapillary等未使用数据，因为实例上是否存在、是否被其他项目引用尚未知。

## 中间检查点的处理边界

本轮只加载四个最终state，下一阶段从同一预训练权重和seed0重新初始化，不把旧1000–39000步状态当作新训练起点。因而旧中间状态可释放空间，但这不意味着旧运行目录仍可按原训练器完整恢复。

只读报告会输出精确候选路径。先备份需要保留的中间实验，再按这些确切路径处理，记录被删除文件的路径、大小、阶段及时间。不能用 `find /root/autodl-tmp -name '*.pth' -delete`、`rm -rf outputs` 或按模糊前缀清理。源码、元数据和最终检查点继续保留。

若占用主要来自旧中间检查点，优先处理这些大文件；不要花时间清理几KB报告。若主要来自唯一的数据集，清缓存并不能解决根本容量问题，应考虑外部备份或扩容。新诊断会增加压缩逐图记录和三组图片，启动前应留出实际可用空间；不足时先处理候选，再运行同一exact-SHA任务。
