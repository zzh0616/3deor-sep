# CHIPS 思路的 32 频 visibility-domain 无噪声测试

## 1. 本轮问题

本轮不再把 raw visibility 直接视为 EoR-window 功率估计，而是吸收
CHIPS 的两个关键做法：

1. 在 visibility domain 显式表示频率协方差，并对平滑、低 delay
   前景方向作 inverse-covariance down-weighting 或 marginalization。
2. 用 quadratic estimator 的 Fisher response/window function 描述
   信号混合，而不是把滤波后的 delay power 直接当成未滤波 sky PS2D。

测试目标是先回答一个有限问题：在 32 个真实频点和精确 OSKAR
visibility 下，频谱平滑先验是否仍能在 EoR window 中抑制前景；若能，
失败究竟发生在单基线频谱、uv gridding，还是 sky-bandpower 标定。

## 2. 数据与工程口径

- 频率为 117.9--121.0 MHz，共 32 频，间隔 0.1 MHz。
- 每频分别用 OSKAR 2.12.2 生成 foreground 和 EoR Measurement Set。
- 采用 isotropic station beam、无 PB、无噪声、32 个 10 s time steps；
  channel bandwidth 为 100 kHz，uv 范围为 30--2500 lambda。
- 从 XX/YY 构造 Stokes-I。标签仅用于冻结 support 后的诊断，不参与
  covariance、window 或 bin 选择。
- 每频 MS 立即压缩为两类数据并删除：
  1. 交替时间 split A/B 的 512x512 bilinear uv grid；
  2. 跨频率保持相同 baseline-time row identity 的固定样本。
- 512 网格的 cell width 为 9.765625 lambda；由 4.5511 deg 视场得到的
  natural Fourier cell 为 12.5894 lambda，二者比值 0.7757，通过预设
  `<=1.2` 的 gridding-resolution gate。128 网格比值为 3.1028，只保留为
  工程负对照。
- 32 频 bank 大小 118,160,280 bytes，SHA256 为
  `f09644cda5e22dfb5adf572ef54f1df3dd68688f8ad8f5ecfdc36772bc53a1d7`。
  3 GPU 并行生成约 28 分钟；单频 FG+EoR 仿真与压缩耗时中位数 146.2 s。

## 3. 估计器

### 3.1 DPSS nuisance subspace

对每个 kperp bin，根据视场角和预声明 supra-horizon buffer 得到
`tau_max(kperp)`。以该 delay support 构造 DPSS 矩阵 `U`，其列空间表示
允许的平滑前景频谱，不固定任何 foreground sky morphology。

有限强度的 nuisance marginalization 使用

```text
C_inv = I - U diag(rho * lambda / (1 + rho * lambda)) U^H,
```

其中 `lambda` 是 DPSS concentration eigenvalue，`rho` 扫描
`1e4, 1e8, 1e12`。`rho -> infinity` 给出 hard projection。另有完整
horizon delay support 作为更保守控制。

### 3.2 Quadratic bandpower 和 window

令 `F` 为 unitary frequency-to-delay basis，`T` 为可选 Hann taper，
分析矩阵为

```text
R = F^H T C_inv.
```

split cross-power 为

```text
q_alpha = Re[(R v_A)_alpha^* (R v_B)_alpha].
```

在 delay-diagonal signal basis 下，

```text
H = R F,
Fisher_{alpha,beta} = |H_{alpha,beta}|^2,
W_{alpha,beta} = Fisher_{alpha,beta} / sum_beta Fisher_{alpha,beta}.
```

估计量用 Fisher row sum 归一化，并把正负 delay 正确折叠到
`|k_parallel|`。support 只依赖几何 EoR window、样本数、相对 Fisher
sensitivity 和 window self-fraction；主门槛为 FG/EoR power `<0.1` 且
total-vs-EoR relative error `<0.2`。

### 3.3 完整频率 covariance 对照

为排除 DPSS basis 太简单，另按 kperp 构造完整 complex 32x32 frequency
covariance。uv angle 分成四 fold，每次用其余三个 fold 估计 covariance，
在 held-out fold 计算 bandpower。测试两种 covariance source：

- `observed_total`：只从无噪声 summed observation 估计，可部署诊断；
- `oracle_foreground`：使用真实 foreground，仅作为上限控制。

扫描 covariance eigenvalue floor `1e-2` 到 `1e-12`，不使用 diagonal
shrinkage。这里的 floor 扫描只作诊断，不能用 truth 事后选择部署参数。

## 4. 结果

### 4.1 固定 baseline-time row

单基线频谱上的结果证明 smoothness prior 本身有效：

| 方法 | support | FG/EoR L1 | total L2 | 通过 |
|---|---:|---:|---:|---:|
| raw none | 410 | 2.410e8 | 2.577e8 | 0 |
| patch covariance, rho=1e12 | 340 | 9.382e-3 | 7.590e-2 | 339 |
| patch hard projection | 337 | 7.492e-4 | 3.640e-3 | 337 |
| horizon hard projection | 24 | 1.501e-3 | 7.450e-3 | 24 |

`patch_hard` 的 summed total/EoR 为 1.00077。因此，在保持同一
baseline-time identity 时，色散前景仍集中在几何允许的低-delay 子空间，
无需 fixed sky template 也能很好地和 EoR 分开。

但这只是滤波后的 observable。相对未滤波 raw EoR，`patch_hard` 的
pure-EoR 集成功率为 0.8035，bin-wise L2 为 0.5654；即使应用当前简单
delay-diagonal window，积分闭合也只有 0.5840，L2 为 0.5998。原因是物理
baseline 随频率在 uv 平面迁移，同一天空 bandpower 的频率 covariance
不是 delay diagonal。

### 4.2 512 uv-grid cross-power

主视图 `uvgrid_cross` 完全失败：

| 方法 | support | FG/EoR L1 | total L2 | 通过 |
|---|---:|---:|---:|---:|
| raw none | 410 | 1.750e7 | 1.584e7 | 0 |
| patch covariance, rho=1e12 | 340 | 2.925e5 | 5.584e5 | 0 |
| patch hard projection | 337 | 2.845e5 | 5.637e5 | 0 |
| horizon hard projection | 24 | 2.436e5 | 6.308e5 | 0 |

`uvgrid_auto` 与 cross 结果几乎相同，例如 `patch_hard` 的 FG/EoR L1 为
`2.853e5`。因此失败不是 thermal/noise bias 或 split 构造造成，而是
不同频率进入同一 uv cell 的 baseline、time 和 w 分布发生变化；简单的
bilinear coherent gridding 把这种色散 sample migration 重新混入了高
delay。

### 4.3 Cross-fit 32x32 frequency covariance

最佳诊断 floor 为 `1e-6`，但仍没有任何 bin 通过：

| covariance | support | FG/EoR L1 | FG/EoR L2 | total L2 |
|---|---:|---:|---:|---:|
| observed total | 302 | 5.044e4 | 4.427e4 | 4.428e4 |
| true-FG oracle | 302 | 5.042e4 | 4.425e4 | 4.425e4 |

observed-total 与 true-FG oracle 几乎相同，说明失败不只是 EoR 污染了
covariance 估计。按 kperp pooling 的单个 32x32 frequency covariance
无法跨 uv-angle fold 泛化，缺少完整的 uv-frequency coupling 和 spatial
response；这正是 CHIPS 中完整 covariance/operator 不能被一个频率核替代
的部分。

## 5. 结论与停止门

本轮给出三个彼此独立的结论：

1. visibility-domain smoothness prior 没有失效。它在固定 baseline-time
   row 上能把 foreground 压到 EoR 的 `7.5e-4`，并恢复滤波后 observable。
2. 当前 uv-grid DPSS 方案不能恢复 sky PS2D。即使网格分辨率足够，简单
   gridding 仍把 chromatic baseline migration 变成未建模的高-delay
   covariance。
3. 仅增加一个 per-kperp 32x32 frequency covariance 不足；true-FG oracle
   也失败，因此不应继续调 DPSS strength、eigen floor 或窗口来寻找
   truth-selected 假通过。

所以停止当前 `bilinear uvgrid + per-kperp frequency covariance` 路线。
固定 row 的 DPSS 可保留为 foreground-filtering/avoidance 前端，但不能把
其输出直接报告为未滤波 EoR sky PS2D。

真正的下一步应构造 sky bandpower response `Q_beta`：把 exact
visibility operator、baseline migration、选定 DPSS filter、gridding 和
split cross-estimator 全部作用到 unit sky-band probes 上，再由完整
Fisher/window 做 bandpower 去卷积或 forward likelihood。可优先实现
per-baseline visibility OQE；现有 no-PB direct-DFT/OSKAR smearing
operator 已有约 `1e-7` dirty closure，可避免再次逐像素生成昂贵 response
bank。只有这一闭合通过后，才值得加入 thermal noise、station beam 和
独立 sky realization。

## 6. 可复现入口

- 核心 estimator：`chips_visibility.py`
- 单测：`test_chips_visibility.py`
- bank builder：`ops_scripts/build_chips_visibility_bank.py`
- 32 频 launcher：`ops_scripts/run_chips_visibility_32freq_noiseless.sh`
- DPSS evaluator：`ops_scripts/evaluate_chips_dpss_visibility_noiseless.py`
- cross-fit covariance evaluator：
  `ops_scripts/evaluate_chips_crossfit_covariance_noiseless.py`
- 精简机器结果：
  `docs/results/chips_dpss_visibility_32freq_20260723_summary.json`
- Genoa 完整结果：
  `/data1/zhenghao/fg_rmw/runs/chips_visibility_32freq_grid512_20260723/`
