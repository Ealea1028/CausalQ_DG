# Rein GTA5 → Cityscapes 单域复现结果

2026-11-27，操作者报告训练和测试已完成。随消息回传的 AutoDL 日志是 MMEngine 的 `Iter(val)`，覆盖 Cityscapes val 全部 500 张。报告的 **mIoU 为 66.69%**，mAcc 为 79.42%，aAcc 为 92.47%。对日志中已四舍五入的 19 类 IoU 和 Acc 重新取平均，分别得到 66.6911% 和 79.4158%，按两位小数与日志一致。

实验路线为 GTA5 训练和 Cityscapes 单域评测。沿用 Rein 官方 DINOv2-L + Mask2Former 单卡配置，batch 4、512×512 crop、40,000 步计划。派生配置只调整数据路径、GTA5 现有标签的后缀和评测范围。与论文表 2 的 Cityscapes 66.4% 对照，本次回传值高 **0.29 个百分点**。 [官方单卡训练配置](https://github.com/w1oves/Rein/blob/train/configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py) · [论文表 2](https://arxiv.org/pdf/2312.04265#page=3)

| Class | IoU (%) | Acc (%) |
|---|---:|---:|
| road | 92.24 | 94.83 |
| sidewalk | 57.44 | 79.47 |
| building | 90.65 | 95.55 |
| wall | 57.19 | 83.31 |
| fence | 54.08 | 69.90 |
| pole | 51.86 | 59.50 |
| traffic light | 58.61 | 76.02 |
| traffic sign | 51.77 | 55.65 |
| vegetation | 89.76 | 94.88 |
| terrain | 50.96 | 68.94 |
| sky | 91.09 | 98.64 |
| person | 75.02 | 85.59 |
| rider | 48.41 | 63.50 |
| car | 92.54 | 95.07 |
| truck | 63.70 | 95.05 |
| bus | 83.13 | 95.13 |
| train | 62.33 | 70.32 |
| motorcycle | 44.80 | 71.96 |
| bicycle | 51.55 | 55.59 |

日志摘要（实例显示时间）：

```text
11/27 15:27:45 - mmengine - INFO - Iter(val) [500/500]    aAcc: 92.4700  mIoU: 66.6900  mAcc: 79.4200  data_time: 0.0038  time: 0.1560
```

运行目录为 `/root/autodl-tmp/outputs/ReinOfficial_GTAV_Cityscapes`。本机目前只收到上述验证片段。对应训练步数、`iter_40000.pth` 的 SHA256 和独立 `Iter(test)` 结果待从实例日志核验。最终归档需要 `rein_sha.txt`、`inputs.sha256`、`gta_cityscapes.py`、`pip_freeze.txt`、`conda_packages.txt`、训练末段、最终测试日志和末步 checkpoint 哈希。权重保留在 AutoDL。
