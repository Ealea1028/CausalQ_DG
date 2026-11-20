# Phase28–32 单帧前向测速

QPrompt-R1 在 ICLR 2026 正式论文第 4 节与表 1 报告单张 NVIDIA RTX 4090、batch 1、512×1024 输入下 54 FPS，按倒数折算约 18.52 ms/帧。该表的 GTA5→Cityscapes mIoU 是 66.1%，Cityscapes、BDD100K、Mapillary 三域均值是 64.1%。论文未给出足以复现其前向计时边界和精度的细节，作者仓库目前只有发布说明。原 CausalQ 实例是 RTX 4090 D，因此 54 FPS 作为论文参考线。 [ICLR 正式论文](https://openreview.net/pdf?id=Vttn6pCwut) · [作者仓库](https://github.com/straybird2333/QPrompt-R1)

Phase28–32 的报告与逐图轨迹保存了准确率、诊断、覆盖和部分显存统计，缺少独立前向计时。运行总耗时还包含数据读写、归约或诊断。使用原 Phase32 代码提交 `40c73bcd6914604368c1006e6c3d7ff0e2ad2a0a` 加载 U0/UP/UF/UA/UR/C0/CP/CF/CA/CR 的固定最终权重，各测两种输入尺寸。

| H×W | 用途 |
| --- | --- |
| 512×1024 | 对照 QPrompt-R1 的输入尺寸 |
| 1024×2048 | 对应原 Phase32 的 Cityscapes 整图推理尺寸 |

每个模型和尺寸独立预热 30 帧，正式计时 3×100 帧。batch 1、单 stream、每帧同步。`model.eval()`、`torch.inference_mode()` 和原 BF16 autocast 策略。输入是 GPU 上已经就绪的固定合成张量，各线复用同一输入。前向覆盖 backbone、decoder、Query head 和模型内部 logits 上采样。预处理、传输、argmax、模型加载和文件写入在计时窗口之外。

导出内含原始逐帧 CUDA Event 时间与同步墙钟时间、均值/P50/P95、按 `1000 / 平均延迟(ms)` 换算的 FPS，以及两份代码 SHA、检查点 SHA 和运行环境。墙钟时间包含 Python 调用与同步开销。两种计时都是串行单帧口径。十条线中，多数配置差异来自训练损失，C 组另有前向 residual 中心化。速度接近时应结合 3 次重复的波动看待细小差值。

后续速度与准确率同表时，Phase32 原分数对应 1024×2048。512×1024 这一栏当前只补速度。实验在原 RTX 4090 D AutoDL 实例空闲时执行，使用 Phase32 原环境和全部最终状态。两个独立工作区分别锁定测速入口与原推理实现。完整命令见交接消息及本机 `docs/AUTODL_NEXT.md` 的补充段。成功回传 `forward_benchmark.json`、`forward_benchmark.md` 和日志尾部。
