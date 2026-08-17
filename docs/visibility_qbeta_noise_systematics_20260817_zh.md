# Visibility `Q_beta` 的热噪声、增益残差与缺失频道测试

## 1. 测试范围

本轮在冻结的 `local_04_116p3_119p4mhz` exact station-pair PB 路线上加入：

1. 两个独立 Gaussian thermal-noise splits 与 cross-power；
2. `100/1000 h` 两档等效积分、各 512 个噪声实现；
3. 由参考格点 10/25 sigma 反解的 EoR 放大对照；
4. smooth/ripple 两类逐站复 gain 残差、四档 RMS、各 128 个实现；
5. 三种已知缺失频道 mask，并对每种 mask 重标定完整 response；
6. 复用已有的 primary-beam shape mismatch 配对结果。

基线仍是 64 个 100-kHz 输入频道、中央 32 频道分析带、1536 条固定
baseline--time rows、full-band hard-DPSS、Hann delay transform，以及
response-only `quad_kperp_response` 选择。无 flag 时有 86 个 selected coarse
groups，FG+EoR 的 ratio/L2 为 `1.03112/0.07035`。

本轮没有加入 RFI 发射本身、相关 receiver noise、电离层、真实 calibration
solution covariance 或新的 LST/uv tracks。因而它是 estimator robustness
测试，不是 SKA-Low 灵敏度预报。

## 2. 独立噪声与 cross-power

两个 split 使用

```text
q_cross = mean_rows Re[(L V_A) conj(L V_B)] / response_normalization,
V_s = FG + a EoR + n_s,
```

其中 `L` 已包含 exact operator 之后的 DPSS、Hann 和 delay transform，
`n_A/n_B` 相互独立。equal-input cross 与原 auto-power 的 relative L2 为
`9.285e-17`。

每个 split 的实部和虚部噪声标准差为

```text
sigma_n(nu) = SEFD_I(nu) / sqrt(2 Delta-nu t_row,split),
t_row,split = T / (32 x 2).
```

SEFD 来自 SKAO `ska-ost-senscalc` 的公开
`ska_station_sensitivity_AAVS2.h5`，文件 SHA-256 为
`9c53a4e5ef2257ffd97fe7f32c5d2e16d5753c6dca0c851ac3aab1eaec031c37`，
lookup revision 为 `f2905865f5d276b46dfc2f7ac9861e16de0772a0`。在本观测方向和频带，
Stokes-I SEFD 为 `6089.5--6624.3 Jy`，中位数 `6351.3 Jy`。

这里的 `100/1000 h` 是重复同一条冻结 320-s LST track：总时间均匀分给 32
个时间格和两个 split。它没有产生真实长观测中的新增 uv coverage。

## 3. 热噪声结果

| 等效积分 | median `sigma_n` | 未增强 EoR 参考 S/N | 10 sigma 振幅因子 | 10 sigma ratio/L2 | 25 sigma 振幅因子 | 25 sigma ratio/L2 |
|---:|---:|---:|---:|---:|---:|---:|
| `100 h` | `0.18936 Jy` | `1.668e-6` | `2926.9` | `1.0244/0.07087` | `5906.4` | `1.0296/0.07051` |
| `1000 h` | `0.05988 Jy` | `1.751e-5` | `939.1` | `1.0280/0.07623` | `1929.5` | `1.0307/0.07255` |

参考格点是预先请求 `k=0.2 h/Mpc` 后，在 selected support 中最近的格点；
实际为 `k=0.546 h/Mpc`，`k_perp=0.312 Mpc^-1`、
`k_parallel=0.199 Mpc^-1`。

未增强 FG+EoR 的 held-out pull 结果为：

| 等效积分 | pull mean/std | 68% coverage | 95% coverage | FG null >3/>5 sigma |
|---:|---:|---:|---:|---:|
| `100 h` | `-0.0079/0.9944` | `0.6879` | `0.9510` | `0/0` |
| `1000 h` | `-0.0077/1.0054` | `0.6843` | `0.9482` | `0/0` |

因此 cross-power 的零均值、方差和 coverage 均通过。未增强 EoR 在当前
1536-row 配置下远不可检测，所以不能用单个 noisy realization 的
`estimate/truth` 判定恢复。10/25-sigma 对照的 ratio/L2 回到无噪声的
`1.031/0.070` 附近，证明加入真实量级噪声后 estimator 链和不确定度传播
仍然闭合；巨大的注入因子同时说明这不是观测灵敏度成功。

## 4. 增益残差

残余增益按 `V_pq -> g_p g_q* V_pq` 作用于 signal 和 noise，response 保持
nominal。每个 realization 在 station-frequency 网格上归一化 log-amplitude
和 phase RMS；gain 在 320 s 内不随时间变化。

- `smooth`：每站独立的 Legendre `0--2` 阶组合；
- `ripple`：每站独立相位、完整 6.4-MHz 带宽内两个周期的正弦；
- RMS：`1e-4, 3e-4, 1e-3, 3e-3`，幅度与相位使用相同 RMS。

| profile | RMS | total ratio/L2 | integrated `|FG|/EoR` | 最坏格点 `|FG|/EoR` |
|---|---:|---:|---:|---:|
| smooth | `1e-4` | `1.03112/0.07035` | `1.513e-5` | `1.397e-4` |
| smooth | `3e-4` | `1.03111/0.07034` | `1.513e-5` | `1.396e-4` |
| smooth | `1e-3` | `1.03111/0.07034` | `1.513e-5` | `1.396e-4` |
| smooth | `3e-3` | `1.03124/0.07043` | `1.514e-5` | `1.396e-4` |
| ripple | `1e-4` | `1.03112/0.07035` | `1.519e-5` | `1.397e-4` |
| ripple | `3e-4` | `1.03112/0.07036` | `1.647e-5` | `1.406e-4` |
| ripple | `1e-3` | `1.03125/0.07065` | `1.499e-4` | `6.911e-4` |
| ripple | `3e-3` | `1.04198/0.10066` | `1.099e-2` | `5.435e-2` |

smooth gain 基本完全落在 DPSS 删除的低-delay 子空间中。两周期 ripple 到
`1e-3` 仍影响很小，但 `3e-3` 时出现明确前景泄漏，并把总 L2 从 7.0% 提高
到 10.1%。这不是通用的 calibration 精度要求，因为 profile 没有从真实
solution covariance 推导；它只说明频谱结构比单一 RMS 数字更关键。

## 5. 缺失频道：response 重标定仍不足

三种 mask 都把已知频道权重设为 0，再用完全相同的 sky-band probes、四个
row partitions 和 exact PB operator 重标定 response。频道索引相对 64 频
输入带：

- `random5`：`[5,17,41]`，占输入带 4.7%，其中两个在分析带；
- `random10`：`[5,12,17,29,41,54]`，占输入带 9.4%，其中三个在分析带；
- `cluster6`：`[29,30,31,32]`，连续四个分析频道，占输入带 6.25%。

| mask | selected/strict groups | pure-EoR ratio/L2 | FG+EoR ratio | FG+EoR L2 | 最大 FG/target | `1000 h` FG null >5 sigma |
|---|---:|---:|---:|---:|---:|---:|
| none | `86/80` | `1.0312/0.07068` | `1.0311` | `0.07035` | `7.05e-4` | `0` |
| random5 | `85/0` | `1.0323/0.07246` | `7.927e6` | `1.020e7` | `3.344e7` | `31` |
| random10 | `84/0` | `1.0416/0.06014` | `5.285e7` | `5.184e7` | `2.251e8` | `66` |
| cluster6 | `66/0` | `1.0266/0.07441` | `7.091e7` | `8.357e7` | `2.568e8` | `52` |

三个场景的 noise-only 68/95% coverage 仍为
`0.684--0.688/0.949--0.951`，cross-self closure 也保持机器精度。pure EoR
和随机 probes 仍约 6--7% L2，说明 response calibration、signal transfer
和 normalization 没有坏掉。

失败机制是：当前操作相当于先把缺失频道置零，再使用为连续频带构造的固定
DPSS projector。对强而光滑的前景，零值形成尖锐频谱断点，把巨大功率送入
高-delay 区域。重标定 response 只能描述 EoR signal loss/window mixing，
不能把数据中的 foreground discontinuity 消掉。因此“flag 已知”不等于
“可以只做 response correction”。

观测应用必须在 quadratic estimator 之前加入 mask-aware foreground
operation，例如 weighted-DPSS least squares、DAYENUREST/DPSS inpainting，
或在缺失采样上定义的 inverse-covariance filter；随后仍需重新标定完整
`Q_beta` response。本轮 zero-fill 是明确的 negative control，不应作为正式
flag 处理方案。

## 6. 已有 PB mismatch 对照

已有 10-partition paired test 固定 exact-PB OSKAR bank，只在 response 中
使用公共、径向、非色散的 PB shape error。成对结果为：

| response PB | model/truth visibility L2 | total ratio/L2 | 相对 exact 的 ratio 变化 |
|---|---:|---:|---:|
| exact | `1.15e-6` | `1.01832/0.04865` | `0` |
| static `-3%` | `1.199%` | `1.03288/0.04996` | `+0.01456` |
| static `+3%` | `1.199%` | `1.00401/0.05137` | `-0.01431` |

这表明 patch-edge `+/-3%` 公共 shape error 没有令旧配对测试整体失效，但
产生约 1.4% 有符号 normalization shift。它来自较早的 64-to-32 paired
run，不能把绝对 ratio/L2 与本轮 `local_04` 混用；它也不覆盖逐站复 Jones、
时变、极化或旁瓣误差。

## 7. 与放大信号验证的关系

Gorce et al. 的 exact-window 校正和 Aguirre et al. 的 HERA validation 已
使用约 25-sigma 量级的放大注入。因此本轮 10/25-sigma arm 是可比的验证
类型，不是“首次放大恢复”。由于本测试只使用 1536 条 stratified rows 和
重复单一 LST track，所需振幅因子不能与 HERA 的注入倍数直接比较。

可以声明的是：

- 未增强与增强输入共用同一 response、选择和 target；
- independent-split cross-power 的统计 coverage 正确；
- 在达到可检测尺度后，恢复回到已知的约 7% noiseless window/sample 误差；
- 未增强输入在当前热噪声合同下没有检测显著性。

## 8. 当前结论

1. Gaussian thermal noise 与 independent cross-power 工程链通过。
2. 低阶 smooth gain residual 在本 DPSS 设置下不危险；`3e-3` 两周期 ripple
   已造成可见的 1.1% integrated foreground contamination。
3. 缺失频道的简单 zero-fill 即使配合 exact response recalibration 也完全
   不可用；mask-aware filtering/inpainting 是正式观测前的硬门槛。
4. 既有 PB screen 只支持小幅公共 shape error 的初步容差，不能替代真实
   station-dependent beam uncertainty。
5. 当前仍不是 observation-ready detection pipeline，也没有给出 SKA-Low
   的最终 sensitivity forecast。

## 9. 可复现位置

- 核心 cross/flag response：`chips_visibility.py`、
  `ops_scripts/calibrate_visibility_qbeta_noiseless.py`；
- thermal/gain evaluator：
  `ops_scripts/evaluate_visibility_qbeta_noise_systematics.py`；
- flag exact-response launcher：
  `ops_scripts/run_visibility_qbeta_flag_stress.sh`；
- flag thermal follow-up：
  `ops_scripts/run_visibility_qbeta_flag_noise_followup.sh`；
- 统一汇总与绘图：
  `ops_scripts/summarize_visibility_qbeta_noise_systematics.py`；
- 机器摘要、NPZ、CSV 和图：
  `docs/results/visibility_qbeta_noise_systematics_20260817/`。

主 thermal/gain 正式 run 用时 `2:45`，最大常驻内存约 `771 MiB`；每个
flag thermal follow-up 用时约 `37 s`。耗时主体是 12 个 exact-PB
response partitions，而不是最终 cross-power Monte Carlo。
