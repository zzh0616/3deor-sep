# 观测先验约束下的 operator-aware 前景分离测试

## 1. 测试问题

本测试回答一个受限问题：在不向拟合器提供仿真前景真值或 EoR 真值的条件下，外部巡天星表、
Haslam 408 MHz 图和有限宽度的前景误差先验，能否经过 cached stride4/rank64 forward operator
后，把无噪声 dirty cube 中的前景残差压到 EoR 二维功率谱以下。

这不是热噪声、RFI、gain calibration 或 split cross-power 测试。若 8 频无噪声初筛不能通过，
协议要求停止 32/64 频扩展。

## 2. 非作弊数据契约

- 频率为 `117.9, 118.3, ..., 120.7 MHz`，共 8 频，间隔 `0.4 MHz`。
- forward operator 为与该频率契约匹配的 cached stride4/rank64 PCA proxy。
- 基准前景真值只用于生成 summed dirty observation：点源来自较深 GLEAM-X DR2，diffuse
  来自 Haslam 408 MHz 形态和空间变化的谱指数图。
- 拟合器只能看到较浅 GLEAM EGC 星表、Haslam 形态和全局谱指数 `-2.55`。
- 星表 nuisance 由 32 个亮源 singleton 和按 flux/spatial cell 分组的弱源组成，共 123 组；
  每组有 fractional amplitude 与 spectral-index slope 两列。
- diffuse nuisance 使用 `4 x 4` partition-of-unity，每个 cell 有 amplitude 与 slope 两列。
- 先验标准差为 catalog amplitude `0.20`、catalog slope `0.30`、diffuse amplitude `0.30`、
  diffuse slope `0.30`。basis 已乘先验标准差，latent coefficient 使用标准正态先验。
- 拟合只使用预先冻结的 control Fourier modes；guard 和 science modes 均不参与拟合。
- EoR 分量标签只在拟合结束后计算 transfer、foreground leakage 和 PS2D gate。

## 3. 求解方法

令观测减去先验均值后的 control feature 为

```text
d = F_control [dirty_total - A(FG_prior)]
```

其中 `A` 是 cached forward operator，`F_control` 是 PS2D v2 的 control-mode Fourier
选择。对每个观测先验 nuisance basis `b_j`，显式计算

```text
X_j = F_control A(b_j).
```

线性高斯后验为

```text
z_map = (X^T W X + I)^(-1) X^T W d,
Cov(z | d) = (X^T W X + I)^(-1).
```

这里每个 Fourier feature 由 nuisance prior-predictive RMS 做有限 floor 的对角尺度化，所有
正规方程和 Cholesky 求解均使用 float64。最终修正不是在图像域直接扣多项式，而是重新组成
sky nuisance cube，再通过同一个 forward operator 一次，随后才在冻结的 science modes 上
计算二维功率谱。

严格门限要求每个目标 bin 同时满足：纯 EoR transfer 和 total residual 相对真 EoR 在 10%
以内，foreground residual power 不超过 EoR 的 10%。快速门限把三项容差放宽为 20%。

## 4. 8 频主实验

| 指标 | prior mean only | control fit 后 |
|---|---:|---:|
| 目标 bin | 32 | 32 |
| quick pass | 0/32 | 0/32 |
| foreground/EoR 集成功率比 | 3,446,205.01 | 1,364,405.77 |
| total residual PS2D 相对 L2 | 5,695,107.04 | 1,590,629.49 |
| pure-EoR transfer 集成功率比 | 1.000000 | 0.985588 |
| pure-EoR transfer 相对 L2 | 约 0 | 0.023617 |

数值闭合是正常的：分量线性闭合误差为 `8.50e-17`，Hessian condition number 为
`2.55e4`。失败来自未建模前景，而不是 double precision、forward operator 重复作用或
Cholesky 发散。后验最大偏移为 `12.575 sigma`，278 个参数中有 24 个超过 `3 sigma`。
catalog 参数后验范数为 `28.956`，其中 23 个超过 `3 sigma`；多个分组要求超过 100% 的负幅度
修正，表明浅星表无法表示深星表中的缺源和位置差异。

## 5. matched-catalog diffuse 消融

第二次实验把 truth catalog 严格替换为 prior catalog，并关闭 catalog nuisance，只保留空间
变化的 diffuse 谱指数作为未知误差。这是失败归因控制，不是可部署成功声明。

| 指标 | prior mean only | diffuse control fit 后 |
|---|---:|---:|
| nuisance 参数 | 0 | 32 |
| quick pass | 0/32 | 0/32 |
| foreground/EoR 集成功率比 | 2,933,044.85 | 2,925.35 |
| total residual PS2D 相对 L2 | 5,044,210.30 | 5,363.84 |
| pure-EoR transfer 集成功率比 | 1.000000 | 1.149607 |
| pure-EoR transfer 相对 L2 | 约 0 | 0.233480 |

该模型把 foreground contamination 降低约三数量级，但仍远高于 EoR。最安全的单 bin 仍有
foreground/EoR power `0.5923`、total/EoR power `1.4814`，因此缩窄窗口也没有 20% 门限下的
通过 bin。control/guard residual fraction 为 `0.02198/0.02230`，说明“把 control dirty
data 拟合到 2%”远不足以保证 EoR-window 的科学精度。

## 6. 结论与停止决定

当前观测先验组合不支持可用的 EoR 分离：目录不完备是最大误差源，`4 x 4` diffuse
amplitude/slope 模型本身也未达到 EoR 精度，并已造成可测的 EoR transfer bias。按预先冻结的
停止门，本轮不运行 32-low、32-high 或 full64，也不进入噪声/cross-power 测试。

这个结果只否定当前具体先验和 MAP subtraction 实现，不证明所有观测先验路线不可能。合理的
下一候选应使用实际可获得的最深星表作为 prior mean，同时用独立 holdout、catalog uncertainty
和 sub-threshold confusion covariance 测试稳健性；diffuse 侧需要来自独立多频观测的空间谱
协方差。前景不确定性更适合在 bandpower likelihood 中 marginalize，而不是继续扩大一个
确定性 MAP 修正空间。任何后续方案仍须先通过同一 8 频无噪声 gate。

## 7. 结果位置与校验

远端结果根目录：

```text
/data1/zhenghao/fg_rmw/runs/observed_prior_control_8wide_20260718/
```

主实验 `result.json` SHA256：
`4cfffc0f5c139480e0a5ab12ca4a72ec68606c5fd5a008546ef1c13797e905d6`。

matched-catalog diffuse 消融 `result.json` SHA256：
`0269fb081a96f6f18e81eebdd2b4215773c8402b56bd8a8e1499adf9b950211c`。

本地和远端均通过 8 项相关单元测试。主实验使用 A800 `cuda:1`，峰值显存约 `7.52 GB`，
总耗时 `1789.16 s`；diffuse 消融使用 A800 `cuda:0`，总耗时 `403.51 s`。
