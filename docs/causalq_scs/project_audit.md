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

用 tools/probe_scs_rein_runtime.py 在现有 DINOv3 环境导入原生模型类与固定提交的 REIN。向真实 head 输入四层合成特征和 100 个解码查询，验收 CUDA 前后向与输入梯度。环境不兼容时保留失败阶段和首个异常，用于确定独立 SCS 环境的构建方式。真实权重和数据验证在 P3 执行。
