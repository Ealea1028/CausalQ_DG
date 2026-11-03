# 19类逐类IoU测试设计

本次是已完成DINOv3研究的只读结果分析，沿用Phase32最终检查点和评估输出。
当前Phase33 DINOv2准备流程、冻结SHA及目标域禁用边界保持不变。

## 指标和样本

单类别报告IoU，不叫单类别mIoU。对整个评估集累加19×19混淆矩阵，行是真值、列是预测；
TP为对角线，FP为列和减TP，FN为行和减TP，IoU=TP/(TP+FP+FN)。
mIoU为19类IoU的等权平均；不能先求每图IoU再平均来替代数据集IoU。
ignore255在累加前排除，图像padding不参与源域计数。
GT和预测都无该类、union=0时为null，不填0；GT无该类但预测有该类时IoU为0。
实际两个完整评估集均有19个有效类别。

固定类别顺序为Cityscapes trainId0–18：road、sidewalk、building、wall、fence、pole、
traffic light、traffic sign、vegetation、terrain、sky、person、rider、car、truck、bus、
train、motorcycle、bicycle。不混用原始labelId和trainId。

源域使用既有GTA5 holdout2497，长边1024/pad16；目标域使用既有冻结Cityscapes500，
native1024×2048整图BF16，无resize/crop/flip/TTA/sliding。两个域分开报告。
模型固定seed0最终40000步；U0/UP/UF/UA/UR及C0/CP/CF/CA/CR全部列出，不选择best。

## 测试与交付

1. 从已接受混淆矩阵重算每类TP/FP/FN、union、IoU、GT像素数和预测像素数。
2. 每类IoU与原始报告一致，19类平均与已报告mIoU误差不超过1e-12；模型间GT类别总数一致。
3. 合成CPU测试覆盖FP/FN归约、union=0、仅假阳性、错误标签维度/负计数。
4. 输出两个域各19行×10模型的绝对IoU表、热图和含像素计数的JSON；不只展示相对提升。
5. 已有逐类差值和目标类别表可参考analysis/diagnostic_final_report_20261028/target_class_iou.csv。

现有结果无需新AutoDL推理。复现本机报告：

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:MPLCONFIGDIR=(Join-Path (Get-Location) '.test-tmp-strict-dg-classwise-mpl')
& E:/Anaconda/envs/pytorch/python.exe -m tools.report_classwise_iou
```

新DINOv2模型未来沿用同一指标定义，但必须使用自身正式评估的混淆矩阵，不能复用DINOv3
数值作为新结果。源域逐类报告可随最终源域评估生成；目标域逐类报告必须等该研究允许并
冻结最终目标评估后生成。当前不新增目标域访问，也不因类别分数低而调整本轮配置。
目前只有单训练种子，类别间差异和方法间差值均为描述性点估计。

## 本机实施结果（2026-11-03）

已生成 `analysis/classwise_iou_20261103/index.html`、两个域的PNG/SVG热图，以及
含全部380个IoU、TP、union、GT与预测像素数的classwise_iou.json。图表已视觉检查。
完整CPU测试725 passed、1 skipped，176.09秒；跳过项为Windows非特权symlink权限。
Python3.10.20、torch2.10.0+cu128、pytest8.4.2，隐藏CUDA，OMP/MKL线程1，
MKL_THREADING_LAYER=SEQUENTIAL。新增4个测试，未访问真实数据或加载权重。
Phase33配置、代码发布SHA及AutoDL交接文件未改动。
