# Visibility `Q_beta` 绝对功率低估诊断

## 结论

`128 -> 64` 测试中约 `15%` 的绝对功率低估已经定位。它不是 exact-PB
forward operator、OSKAR 设置、有限 visibility rows 或单个 response probe
造成的，而是当前 response 模型隐含的**全输入带宽平稳性假设**与 EOS
lightcone 的强频率演化不相容。

当前 source response 在完整三维 Fourier 基底中按 bandpower 标定，等价于
使用近似对角的 source covariance。对 Fourier 系数向量 `a`，任一输出二次型
可写成

```text
q_beta = a^H K_beta a .
```

随机相位 response 只标定

```text
E[q_beta] = sum_i (K_beta)_ii E[|a_i|^2] .
```

真实 lightcone 还含有

```text
sum_{i != j} a_i^* (K_beta)_ij a_j .
```

其中主要部分来自同一空间 Fourier mode 的跨频率协方差。复杂、色散的
visibility operator 并没有制造这个不一致，但使这些非对角项进入 EoR-window
输出，不能继续用全局平稳 bandpower response 忽略。

## 已排除的工程原因

### Operator 闭合

用实际 EoR 天空和 cached station-pair aperture PB 重新执行 matrix-free
exact DFT：

| 检查 | relative L2 |
|---|---:|
| 新 128 频 OSKAR visibility closure | `3.35e-7` |
| 新宽带 restricted-Q closure | `1.06e-6` |
| 新窄带 restricted-Q closure | `1.29e-6` |
| 旧 64 频 OSKAR visibility closure | `5.23e-7` |
| 旧 restricted-Q closure | `5.16e-7` |

这些误差比 `15%` 小五个数量级，且 forward operator 只作用一次，没有双重
作用。

### 行数与 response probe

20 个 partition 使用互斥 visibility rows，但使用相同的 calibration phase
seed；validation 是第二套独立 phase seed。增加 partition 因而增加的是 rows，
不是独立 phase realization 数。冻结正式窗口后，使用 calibration/validation
平均 response 得到：

| partitions | 新宽带 ratio | 新窄带 ratio | 旧窄带 ratio |
|---:|---:|---:|---:|
| 1 | `0.8553` | `0.8411` | `1.0652` |
| 4 | `0.8551` | `0.8605` | `1.0209` |
| 8 | `0.8517` | `0.8488` | `1.0113` |
| 20 | `0.8495` | `0.8465` | `1.0102` |

相应 held-out phase-ensemble mean 在 20 partitions 时为
`0.9970/0.9982/0.9947`。因此 response 可以预测其标定的随机相位总体，而
physical-lightcone 偏差随 rows 收敛到非零值。

### 新旧输入严格对齐

对重叠的 `114.7--121.0 MHz` 共 64 个频道进行了逐频道检查：

- EoR sky cache 的频道数组逐字节相同；
- OSKAR shard 的 row indices、UVW、time、split、antenna IDs、
  foreground visibility 和 EoR visibility 逐字节相同；
- 两条链使用同一无额外 PB sky template、相同天线模型和相同 OSKAR
  integration/channel 设置。

新测试因分析参考几何改变而重新按 `k_perp` 分层抽行，所以 partition 内
具体 rows 不要求与旧测试相同；response 标定已显式包含该差异。
逐频道 hashes 及比较结果保存在
`docs/results/visibility_qbeta_response_bias_20260728/{new,old}_overlap_contract.json`
和 `overlap_comparison_summary.json`。

## 振幅与相位分解

进行了两类不读取前景的 EoR surrogate：

1. `global random phase`：保留每个三维 Fourier mode 的实际振幅，随机化
   全部 Hermitian phases。它消除全部非对角 cross terms。
2. `spectral-coherence surrogate`：只给每个二维空间 Fourier mode 乘一个
   在所有频率共享的随机 Hermitian phase。它逐 mode 精确保留完整复数频率
   向量以及全部跨频率 covariance，只消除不同空间 modes 之间的相位关系。

全局随机相位检查使用一个 visibility partition 和 16 个 phase
realizations：

| 数据 | physical ratio | global-phase mean |
|---|---:|---:|
| 新宽带 | `0.8553` | `1.0291 +/- 0.0462` |
| 新窄带 | `0.8411` | `1.0397 +/- 0.0477` |
| 旧窄带 | `1.0652` | `1.0099 +/- 0.0475` |

随后在两个互斥 visibility-row partitions 上合并同一组 16 个
spectral-coherence realizations：

| 数据 | 同 rows physical ratio | spectral-coherence mean | 差值 / ensemble std |
|---|---:|---:|---:|
| 新宽带 | `0.8720` | `0.8749 +/- 0.0177` | `-0.16 sigma` |
| 新窄带 | `0.8638` | `0.8661 +/- 0.0273` | `-0.08 sigma` |
| 旧窄带 | `1.0447` | `0.9977 +/- 0.0201` | `+2.33 sigma` |

新宽/窄的 covariance-preserving ensemble 与同 rows physical 结果只差
`0.0029/0.0023`，所以跨频率 covariance 已解释几乎全部新低估。旧测试的
两分区 physical 值仍有较大的有限 rows 波动；增加到正式 20 partitions
后变为 `1.0129`，与其 spectral-coherence mean `0.9977` 相容。因此：

- mode 振幅、单位换算和 response 对角项没有 `15%` 级错误；
- 新低估由 lightcone 跨频率 covariance 主导，并能在独立 rows 上直接
  复现；
- 单 partition 中曾出现的约 `4%` physical--ensemble 差异不随 rows
  保持，是有限 rows/空间相位波动，不是稳定的第二项系统偏差。

## 为什么旧窄带反而没有低估

逐频道去均值后的空间方差显示：

| 输入/分析 | analysis/input 方差比 | 输入后半/前半方差比 |
|---|---:|---:|
| 新 `108.3--121.0 -> 111.5--117.8 MHz` | `0.8743` | `0.4316` |
| 旧 `114.7--121.0 -> 116.3--119.4 MHz` | `0.9813` | `0.7135` |

新 12.8-MHz 输入跨越了很强的 lightcone 演化：低频半段平均空间方差为
`7.93e-5 K^2`，高频半段只有 `3.42e-5 K^2`。中央分析带平均方差只是全
输入平均值的 `87.4%`，与最后约 `85%--90%` 的 response ratio 同量级。
旧 6.4-MHz 输入演化较弱，其中央分析带方差是输入平均的 `98.1%`，所以全局
平稳 response 恰好近似成立。

这意味着旧结果接近 `1` 是频段与该 lightcone realization 的有利组合，
不能据此声称当前全局 estimator 对任意更宽频段都无偏。

## 已否定的简单修补

把 128 个频道硬切成 `2/4/8` 个互不重叠块，并在每块内独立匹配局域
bandpower，得到的 predicted/global-target ratio 分别约为
`1.14/1.09/1.22`（宽带），physical/prediction ratio 反而降到
`0.75/0.78/0.70`。硬边界破坏块间频谱相干并引入高-delay 功率，所以“只
匹配各块局域方差”不足以修复问题。

不能使用 physical truth 导出的单一 `1/0.85` 标量校正；那会隐藏目标定义
错误，并且不会推广到另一条 lightcone。

## 正确的下一步

当前 128 频结果应作为全局平稳性 stress test，而不是无偏恢复结果。后续
estimator 应改为局域红移功率谱：

- 用重叠、平滑加窗的 STFT/DPSS/Slepian 频率块替代硬分块；
- source covariance basis 必须保留相邻窗口的 cross-window 项，而不是只
  拟合每块方差；
- response 和 target 在同一局域频率基底中共同标定；
- 窗口宽度及 overlap 由 observation geometry、delay support 和
  truth-blind conditioning 冻结，不能根据 EoR 恢复误差选择；
- 发表前至少增加一条独立 lightcone 或等价的空间-realization ensemble，
  检验局域 estimator 的 ensemble bias。

现有 exact-PB visibility operator、DPSS 前景抑制和 response-only EoR-window
选择仍可复用；需要替换的是 source covariance/target 参数化，而不是重新
实现 OSKAR operator。
