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
