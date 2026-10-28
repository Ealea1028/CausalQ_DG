# 十模型最终源域诊断图表

打开[index.html](index.html)查看六组科学图。模型顺序为U0/UP/UF/UA/UR、
C0/CP/CF/CA/CR。机制依次为无一致性、预测KL、Full CQE、Anchor-only、Residual-only；
U关闭中心化，C开启。旧四模型复用Phase28，新六模型来自Phase31，均为seed0最终40000步。

完整解释见[Phase31结果](../../docs/DIAGNOSTIC_PHASE31_RESULTS.md)。

- `01_amplitude_variance`：绝对幅值、联合RMS、五视图原始/逐视图RMS归一化方差。
- `02_amplitude_stability_scatter`：幅值与方差同时比较，识别尺度收缩或增长。
- `03_correlation_cancellation`：空间/变化相关、风格方差协方差分解、相加抵消/相减收缩。
- `04_query_gradients`：每slot范数、方向、direct及Query对Anchor投影、每类cosine。
- `05_query_responsibility`：原图责任/熵、风格切换/JS、GT类别区域像素加权责任。
- `06_class_stratification`：所有类别/GT出现分层的归一化A/R比值，颜色为log10并截到±3。

`model_metrics.csv`保存30行模型/分量表，`class_metrics.csv`展开导出摘要中的每类、
每损失、每视图指标及P10/median/P90、count/null_count。完整类别表约24MB，留在
本机和分析ZIP；图表、模型表及`plotted_metrics.json`输入哈希进入Git。
这些是按图像/类别等权的摘要，GT类别责任热图另用像素加权，不能混用。
每图原始JSON明细未传回本机，不声称已独立重算原始测量。P10/P90不是置信区间。

方差坐标是原始数值，无额外乘1000。pair_raw_variance是两视图兼容指标，主图是
五视图总体方差；RMS归一化必须在每个图像/类别/视图内先做，不能事后除以全局RMS。
所有结果仅描述一个训练种子和源域样本；新增模型的目标域收益尚待Phase32评估。

重现命令（本机，仅摘要分析）：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:MPLCONFIGDIR=(Join-Path (Get-Location) '.test-tmp-strict-dg-phase32-mpl')
& E:/Anaconda/envs/pytorch/python.exe -m tools.analyze_final_source_diagnostics
```
