# 官方 Rein GTA5 → Cityscapes 复现

2026-11-26 的 AutoDL 回传已报告 `REIN_SETUP_OK` 和 `CUDA_OK: NVIDIA GeForce RTX 4090`。xFormers 提示表示可用，ReinsConvNeXt 的可选依赖提示不影响 DINOv2 路线。完整安装日志、pip 清单和本次转换权重哈希仍在实例的 `/root/autodl-tmp/outputs/rein-g2c-setup`。`setup_observation.json` 记录聊天中已看到的证据。

## 实验口径

上游提交 `dc063429c4dadc0da9c6252b3db22fc55a9882ab`，DINOv2-L patch16、Rein + Mask2Former，seed 0，GTA5 全部 24,966 图，物理 batch 4、512×512 crop、40,000 更新。保留上游 `RandomChoiceResize`、`RandomCrop`、翻转和颜色增强，以及 AdamW 和 PolyLR。Cityscapes val 固定 500 张，19 类、ignore 255、滑窗 512×512 / stride 341。训练中验证保留上游间隔，最终报告固定末步权重的独立测试。

论文表 2 的 Cityscapes 对照是 66.4% mIoU。作者另行发布的 batch 8 权重报告 66.7%，历史 CausalQ 适配版完成 65.6051%。本次新训练和测试结果待回传。 [论文表 2](https://arxiv.org/pdf/2312.04265#page=3) · [官方 batch 4 配置](https://github.com/w1oves/Rein/blob/dc063429c4dadc0da9c6252b3db22fc55a9882ab/configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py) · [历史适配版](https://github.com/Ealea1028/CausalQ_DG/blob/30d3490509f151a7ce13065b3656dbdde5839103/experiments/REIN_PHASE16_RUNTIME/report.md)

## AutoDL 执行顺序

在已签出新 Git 提交的 CausalQ 根目录执行。若通过随包的 `rein-g2c.bundle` 导入，先把 bundle 上传到 `/root/autodl-tmp/rein-g2c.bundle`，在独立 checkout 导入（需要上一版已发布的基线 `30d3490509f151a7ce13065b3656dbdde5839103`）。

```bash
bash <<'BASH'
set -euo pipefail
R=/root/autodl-tmp
BUNDLE="$R/rein-g2c.bundle"
CODE=$(mktemp -d "$R/CausalQ_REIN_G2C_XXXXXX")
git clone --single-branch --branch codex/lora-causalq-1114 \
  https://github.com/Ealea1028/CausalQ_DG.git "$CODE"
git -C "$CODE" bundle verify "$BUNDLE"
read -r SHA REF < <(git bundle list-heads "$BUNDLE")
git -C "$CODE" fetch "$BUNDLE" "$REF"
git -C "$CODE" checkout --detach "$SHA"
printf '进入目录：cd %q\n' "$CODE"
BASH
```

在上述目录开启 tmux，或在已有 tmux 中打开新窗口并进入该目录。执行数据对齐，再执行 20 步试跑。保持 `RUN_ID` 不变。

```bash
SHA=$(git rev-parse HEAD)
RUN_ID=gta_citys_bs4_seed0
bash scripts/run_rein_g2c.sh prepare "$SHA" "$RUN_ID" &&
  bash scripts/run_rein_g2c.sh smoke "$SHA" "$RUN_ID"
```

`prepare` 检查 GTA5 成对 PNG、Cityscapes val500 成员和训练标签有效值。同图缩放误差在 0.2% 内的 GTA5 mask 才可用最近邻对齐。Cityscapes 有 trainIds 时复用，否则从 labelIds 生成，维持原始分辨率。派生标签写入新输出目录，配对表记录每张图用的源文件。若 `prepare` 在创建派生目录后失败，保留日志并使用新 `RUN_ID` 重试。GPU 阶段都使用 Phase46 的共享锁。

看到 `REIN_SMOKE_OK` 后，核查 `prepare.log`、`data_report.json` 和 `smoke.log`。确认真实 batch 4 运行正常后，从 DINOv2 骨干开始正式训练。训练中断时，在同一 SHA、同一 `RUN_ID` 下用 `resume` 代替 `train`，它要求已有上游完整恢复点。

```bash
bash scripts/run_rein_g2c.sh train "$SHA" "$RUN_ID"
# 收到 REIN_TRAIN_OK 后独立评测第 40,000 步
bash scripts/run_rein_g2c.sh test "$SHA" "$RUN_ID"
```

输出在 `/root/autodl-tmp/outputs/REIN_GTAV_CITYS/$RUN_ID`。测试成功将打印 `REIN_TEST_OK`。下载 `evidence.tar.gz` 和 `evidence.tar.gz.sha256` 放回此复现文件夹，内含 mIoU、逐类日志、两份提交身份、实际配置、运行环境、数据清单、checkpoint 哈希和训练日志。模型权重、原始数据与完整运行目录留在实例。若要完整归档已完成的安装，另从 `rein-g2c-setup` 下载 `setup.log`、`pip_freeze.txt`、`rein_sha.txt` 和 `backbone.sha256`。
