# CausalQ-SCS P0 双线接入审计

| 用途 | 当前项目入口 | SCS 接入点 |
|---|---|---|
| DINOv2-L/p16 | causalq/models/dinov2_wrapper.py、tools/convert_dinov2_for_rein.py | 保留转换后的 REIN 权重和原生位置插值。SCS 配置显式使用零基 taps [7,11,15,23]，现有 HPI 包装器默认 [5,11,17,23]。 |
| DINOv3-L/p16 | causalq/models/dinov3_wrapper.py、causalq/models/backbone_factory.py | 同一次 forward 已返回 Patch、CLS 和 register。SCS 从逻辑 [8,12,16,24] 个 block 的归一化前输出取 tap，保留原生 RoPE、末层 norm 和特殊 token。 |
| REIN | scripts/run_rein_g2c.sh、tools/prepare_rein_g2c.py、tools/check_rein_segmentor.py | 复用固定提交的 Mask2Former 和特征金字塔。受控组使用独立的 100×256 解码 embedding。 |
| 数据 | manifests/gta5_source_split.json、manifests/cityscapes_val500.txt、scripts/run_lora_phase45_source.sh、scripts/run_lora_phase46_target.sh | 复用 GTA5 划分和 Cityscapes val500，沿用 REIN 的输入和 slide 评估设置。 |

最小实现顺序为共享 Query 与 Read–Write 核心及梯度测试，两种宿主注入与共同 Mask2Former 训练入口，双线 A0/A4 配置以及源域和锁定目标评估。首轮采用 GTA5、Cityscapes、19 类、ignore 255、p16、40,000 optimizer updates 和每步四张源图。

## AutoDL 接入条件

此前 DINOv3 在 causalq-lora1114 环境运行，原版 REIN 在单独的 OpenMMLab 环境运行。SCS 接入需要 REIN head 的 GPU backward 在同一进程内将梯度传回 SCS。

首次探测在 CausalQ 提交 `47cab11ab713a8acc52897befbe6aae84ed30ee9` 和 REIN 提交 `dc063429c4dadc0da9c6252b3db22fc55a9882ab` 执行。AutoDL 回传 Python 3.12.16、Torch 2.7.0+cu128、Transformers 4.56.1、RTX 4090（50,866,487,296 bytes）。DINOv3ViTModel 导入成功。随后在 `rein_import` 报 `ModuleNotFoundError: No module named 'mmcv'`，退出码 1，优化更新数为 0。

用 `scripts/setup_scs_rein_autodl.sh` 克隆该 DINOv3 环境，在克隆环境里按原 REIN 版本固定 MMCV 2.1.0、MMEngine 0.10.7、MMSeg 1.2.2 和 MMDet 3.3.0。MMCV 从 PyPI 发布源码和固定 SHA256 生成 CUDA 12.8、SM 8.9 wheel。优先使用实例已有 CUDA 12.8 编译工具链，缺失时使用 NVIDIA Conda 的 12.8.1 开发工具包。脚本随后向固定的 REIN head 输入四层合成特征与 100 个解码查询，验收 CUDA 前后向及输入梯度。真实权重和数据验证在 P3 执行。[MMCV 官方构建说明](https://mmcv.readthedocs.io/en/latest/get_started/build.html) · [MMCV 2.1.0 发布包](https://pypi.org/project/mmcv/2.1.0/) · [NVIDIA Conda CUDA 说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html#conda-installation)

AutoDL 在提交 `2eb18c2b1eaee0ab7cf537c8a8d50b8c681d0337` 成功安装 MMCV 2.1.0。探测进入 REIN 包初始化后，由 EVA 骨干导入未安装的 `xformers` 而退出，优化更新数仍为 0。复用 `/root/autodl-tmp/outputs/scs_p0_setup_2eb18c2_nP68O8/env/bin/python`。SCS 直接从固定外部源码加载独立的 `ReinMask2FormerHead`，其顶层依赖只来自 PyTorch、MMSeg 及标准库。用同一配置验证 100 个查询、四层特征、真实分类/掩膜/Dice 损失与梯度。REIN 官方复现命令继续使用原有入口。[固定提交的 head](https://github.com/w1oves/Rein/blob/dc063429c4dadc0da9c6252b3db22fc55a9882ab/rein/models/heads/rein_mask2former.py)

## P3 预训练权重恢复

AutoDL 在 `99f368c5b32f623b8cee4eb04e2b274225b5c00c` 读取了锁定的 GTA5 样本，随后因首个 DINOv2 转换文件字节 SHA256 与 Phase16 归档值不同而停止。回传未包含候选文件的实际 SHA256，两线均未执行工程更新。

P3 依次尝试已固定 SHA256 的 Phase16 REIN p16 转换产物和 HPI p16 转换产物。两者使用同一官方 DINOv2 ViT-L/14 源权重和 FP32 bicubic、`align_corners=False` 的 14→16 patch kernel、37→32 位置网格转换。仓库测试比较两套转换实现的张量输出。若候选均不匹配，从校验通过的官方源权重重做 REIN 转换，在本次输出目录记录实际新 SHA256。报告记录每个已检查文件的实际哈希，DINOv3 独立报告供回传。
