# 32 频部分 EoR 窗口协方差恢复测试

## 目标与边界

本测试回答一个受限问题：在不假设固定前景空间模板形态、无热噪声、且外部前景模型误差协方差可标定的条件下，是否能从复杂成像算子后的 residual 中恢复一部分 dirty-EoR 二维功率谱。

这里不声称恢复逐像素 EoR map，也不声称恢复 intrinsic EoR 功率谱。前景真值只用于构造一次“观测先验 emulator”和冻结后的验证；估计器看不到 hidden foreground residual 或 EoR 真值。

## 重新构建的 forward operator

- 使用 117.9--121.0 MHz、0.1 MHz 间隔的 32 个真实频点。
- 每个频率独立加载 stride-4、rank-64 PCA response cache，共 32 x 64 = 2048 个 tile，约 143 GB。
- 不做频率插值，manifest 明确记录 `frequency_interpolation=false`。
- operator bank 构建耗时 4322.94 s，GPU 峰值约 29.6 GB；主要耗时是共享存储加载 cache。
- 对 exact WSClean dirty image 的闭合误差：FG relative L2 为 `3.6103e-6`，EoR 为 `2.2012e-6`。
- 因 FG/EoR RMS 约为 20533，即使上述 FG 相对误差很小，其绝对残差仍为 EoR RMS 的 7.41%；这是当前结果中不能忽略的 operator discrepancy。
- 冻结的 PS2D contract hash 为 `37a984ec43857510c821de560331670d494077cea25331df3e2ba7b17176a99d`。

## 前景先验 emulator

给定模拟前景模板后，生成与模板独立的误差 realization。标量 `p` 同时控制：

- 幅度场 RMS：`p`；
- spectral-index RMS：`2p`；
- 位置扰动 RMS：`4p` pixel；
- unresolved-confusion RMS：模板 RMS 的 `0.3p`。

16 个 realization 用来估计 operator-propagated 前景误差协方差，另一个 seed 完全独立地作为 heldout。逐 `k_perp` 检查中，heldout 与 ensemble 的协方差 trace 比主要落在 0.7--1.18，说明误差幅度标定本身没有数量级错误。

这个 emulator 仍然比真实观测简单：它不含热噪声、RFI、gain 残差，也没有外部 survey 中缺失源和 diffuse morphology 的完整不确定性。因此 `p` 是受控实验参数，不应直接等同于巡天 catalogue 的 flux-scale error。

## 两种估计路线

### Wiener covariance posterior

首轮在每个 `k_perp` annulus 内拟合

\[
C_y = q_{\rm FG} C_{\rm FG} + q_{\rm EoR} C_{\rm EoR}(\ell),
\]

并对 `q_FG`、`q_EoR` 和频率相关长度 `ell` 网格边缘化，再计算 EoR Wiener posterior second moment。

这条路线没有通过。0.1% 档在标准窗口的积分功率比仅为 0.0091，在高 `k_parallel`、中等 `k_perp` 区仅为 0.0028。对照诊断显示：

- 当前 flat-transverse probe 与真实 dirty-EoR 频率协方差的归一化 Frobenius 误差约 0.40--0.75；
- 即使用 EoR 真值协方差作 oracle 数学对照，中高 `k_perp` 区仍只有约 6.8% 的功率；
- heldout 误差换符号可令当前 probe 结果从严重低估变成严重高估，表明单 realization 的 FG-EoR cross term 主导分解；
- Wiener posterior 在不可识别方向主动收缩到零，不适合作为无偏 PS2D bandpower 基线。

因此，不能把这条路线的低功率解释为“窗口内没有 EoR”。如继续该路线，需要重建按输入 `k_perp` 分带的 operator probe，并改用 Fisher-normalized bandpower likelihood，而不是继续调整同一 Wiener 超参数网格。

### 前景协方差 bias 扣除

对每个合法 PS2D bin 先分别聚合非负功率，再计算

\[
\widehat P_{\rm EoR}=P(d-A\bar f)-\mathbb{E}_{\delta f}[P(A\delta f)].
\]

第二项由 16 个独立、经过同一 32 频 operator 的前景误差 realization 估计。扣除在 bandpower 层进行，允许无偏估计出现负值；没有 Wiener shrinkage，也没有固定前景形态子空间。验证包括独立 heldout 以及 16 个 leave-one-out realization 的正负号，共 32 个 stress controls。

## 结果

预定义的 high-`k_parallel`/mid-`k_perp` 区含 40 个 bin、26,928 个独立 mode，`k_parallel=1.219--1.524 Mpc^-1`，`k_perp=0.340--0.783 Mpc^-1`。

| `p` | heldout 积分比 | heldout L2 | 最大单-bin 误差 | LOO 积分比 10--90% | 结论 |
|---:|---:|---:|---:|---:|---|
| 0.1% | 1.037 | 0.060 | 0.121 | 1.036--1.049 | 40/40 bin 通过 20% 门槛 |
| 0.3% | 1.017 | 0.152 | 0.374 | 0.972--1.091 | 积分稳定，单-bin 开始失败 |
| 1.0% | 0.829 | 0.693 | 1.635 | 0.206--1.675 | 此 40-bin 区失败 |

在最高一个 native `|k_parallel|=1.52365 Mpc^-1`、相同 `k_perp` 范围的 10-bin 子窗口中：

| `p` | heldout 积分比 | heldout L2 | LOO 落入积分 ±30% 的比例 | LOO 积分比 10--90% |
|---:|---:|---:|---:|---:|
| 0.3% | 1.040 | 0.050 | 100% | 1.020--1.061 |
| 1.0% | 1.027 | 0.101 | 100% | 0.933--1.168 |
| 1.5% | 1.006 | 0.149 | 90.6% | 0.839--1.261 |
| 2.0% | 0.975 | 0.210 | 65.6% | 0.668--1.378 |
| 3.0% | 0.881 | 0.378 | 43.8% | 0.171--1.701 |

按冻结门槛，稳健边界约为 `p=1%`；1.5% 已是边缘结果，2% 不通过。这个 10-bin 子窗口有 6732 个独立 mode，但只包含标准窗口内模拟 EoR 功率的 `8.77e-5`。它证明复杂 operator 后仍存在可识别方向，但统计覆盖很小。

更大的 128-bin high-`k_parallel` 区在 `p=0.1%` 时覆盖标准窗口 EoR 功率的 1.248%，heldout 积分比/L2 为 0.983/0.146；32 个 LOO 的积分比 10--90% 为 0.942--1.116。其逐-bin 误差仍较大，所以只能称为积分 bandpower 恢复。

## 现实性判断

GLEAM 报告大部分天区对外 flux scale 精度约 8%；GLEAM-X 也建议跨 survey 使用 8%，而同一 survey 内部频间 flux-scale 误差约 2%。因此本测试的 1% 阈值比现成全空巡天的绝对尺度更严格，只能解释为对目标深场进行场内、多频联合再拟合后的乐观目标，不能声称已有 catalog 直接满足。参见 [GLEAM](https://academic.oup.com/mnras/article/464/1/1146/2280761) 与 [GLEAM-X DRI](https://pmc.ncbi.nlm.nih.gov/articles/PMC7612673/)；不完整 sky model 还会产生 EoR 相关的校准偏差，见 [Patil et al. 2016](https://academic.oup.com/mnras/article/463/4/4317/2646512)。

## 当前结论与下一门槛

1. 在真实重建的 32 频 complex operator 后，前景协方差 bias 扣除可以恢复一部分 dirty-EoR PS2D；这不是逐像素分离，也不是全窗口恢复。
2. 0.1% 误差允许较大的 high-`k_parallel` 积分区；约 1% 误差只允许最高 `k_parallel` 的极小子窗口。
3. 当前结果是无噪声、单 EoR realization、truth-derived prior emulator 的方法验证。加入噪声或直接推广到实观测前，必须增加独立 EoR realization，并将观测派生的误差协方差作为 heldout，而不是只改变 `p`。
4. 只有独立 EoR realization 仍通过后，才值得支付成本构建低 32 频 cache/64 频 operator。若继续 covariance likelihood，则下一次 cache 加载应一次性传播按输入 `k_perp` 分带的 separable probes，不能复用当前 flat-transverse EoR probe。

机器可读结果：`docs/results/partial_window_covariance_32high_20260722_summary.json`。
