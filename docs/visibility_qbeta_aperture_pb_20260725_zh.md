# `Q_beta` 粗分组与 OSKAR primary-beam operator

## 1. 当前结论

本轮先冻结无 PB 的功率谱基线，再实现并验证带真实 OSKAR aperture-array
PB 的 visibility operator。

无 PB 基线采用 response-weighted `4x1` 分组：固定 `k_parallel`，合并四个
相邻 `k_perp` cells。选择只使用 operator response 和几何 EoR-window
局域性，不读取 EoR 真值。该方案得到 85 个 coarse windows，覆盖 337 个
原细格；原物理 EoR、前景加 EoR 和 16 个独立物理平移视图全部通过 20%
逐窗门。

带 PB 时，不能使用单站 power beam 代替真实基线响应。对非极化 Stokes I，
每条 visibility 的正确标量 PB 因子是

```text
b_pq(s, t, f) = 0.5 Tr[E_p(s,t,f) E_q(s,t,f)^H] .
```

新 operator 直接调用 OSKAR 的 telescope settings 和
`oskar_evaluate_jones_E()`，只缓存选中站对的复数 coherency。与 station-0
对照完全相同的单频 384-row 测试相对 OSKAR visibility 的 L2 闭合为
`1.90e-7`；64 频、7680 rows 的合并闭合为 `1.10e-6`。因此自建的
`PB x DFT x w-term x channel/time smearing` 链条确实复现了 OSKAR
visibility，而不是用近似图像域 beam 代替。

64 频 PB 正式结果继续采用预先冻结的 response-only 选择。`4x1` 得到
88 个 coarse windows、覆盖 348 个名义细格，原物理 EoR+FG 和 16 个
heldout 相位视图均通过 20% 逐窗门。它无需退化到 `4x2`。该结论仍限定于
无噪声、固定 baseline-time rows；热噪声、独立 time split 和 cross-power
尚未加入。

## 2. 无 PB 的正式 coarse 基线

共同选择条件为：

```text
minimum kperp index = 4
relative response >= 0.1
geometric-window response fraction >= 0.95
aggregation weighting = response
```

主要对照如下：

| profile | coarse windows | 严格通过 | 物理平移最坏 L2 | 最大逐窗误差 |
|---|---:|---:|---:|---:|
| fine | 333 | 280 | 9.51% | 42.46% |
| `2x1` | 86 | 83 | 8.08% | 22.95% |
| `4x1` | 85 | **85** | **7.11%** | **16.90%** |
| `4x2` | 43 | 43 | 7.08% | 13.36% |

`4x2` 的误差略低，但额外牺牲径向分辨率，因此正式 no-PB 基线选
`4x1`。其详细指标是：

- 原 bank foreground+EoR：积分比 `1.01312`，L2 `1.984%`；
- 16 个 heldout 随机相位 total：最坏 L2 `5.777%`；
- 16 个物理平移 pure EoR：积分比 `0.97726--1.01758`，最坏 L2
  `7.092%`；
- 16 个物理平移 foreground+EoR：积分比 `0.97712--1.01758`，
  最坏 L2 `7.106%`；
- 最大逐 coarse-window 误差 `16.896%`；
- 最大 foreground-induced change `0.06897%`；
- response-window participation rank `33.18`。

物理平移是在天空平面对 EoR cube 做 16 个唯一、非零循环平移。该操作保持
intrinsic 3D PS2D 完全不变，但改变 EoR 相位相对 instrument response 的
关系，因此比只做 Fourier random phase 更接近独立物理视图压力测试。

## 3. 为何 station-0 beam 不够

OSKAR aperture-array station model 中，不同 station 的 element layout/
station model 不完全相同。基线 `p-q` 的响应一般是复数、随时间和方向变化，
不能写成一个公共实数 power beam。

旧的 station-0 auto-power cache 在 119.4 MHz 的 384-row 测试中得到：

```text
relative L2       = 9.5924e-2
complex corr      = 0.996721
pred/target norm  = 0.945306
```

误差随时间和站对变化，不能用一个复增益修正。它不是 DFT、`w` 项或时频平均
的误差；同一 no-PB/Gaussian 链条已分别闭合到 `1e-6` 以内。

## 4. 精确 station-pair Jones 实现

实现步骤为：

1. 读取生成目标 OSKAR visibility 时的原始 `.ini`；
2. 通过 `oskar_app_settings_tree()` 和
   `oskar_settings_to_telescope()` 载入完全相同的 telescope model；
3. 在 GPU 上按 source chunk 调用 `oskar_evaluate_jones_E()`；
4. 每个时间点只拷贝选中 row 涉及的 station Jones；
5. 计算 `0.5 Tr(E_p E_q^H)`，写为磁盘流式
   `[selected_row, direction] complex64` cache；
6. matrix-free DFT 按 row/source chunk 读取该因子，并同时计算相位、
   `w` 项、100-kHz channel smearing 和 10-s time smearing。

OSKAR 在允许 station-beam duplication 时对 parallactic rotation 使用公共
station-0 rotation。对非极化 trace，该公共 unitary rotation严格消去，因此
cache 只需保存 E-Jones station-pair coherency。

119.4 MHz、与 station-0 失败对照完全相同的 384 rows、262144 directions
结果为：

```text
relative L2             = 1.8978e-7
maximum relative error  = 2.6565e-7
complex correlation     = 0.9999999999999829
pred/target norm        = 0.9999999854
PB cache build          = 69.00 s
matrix-free apply       = 0.44 s
peak reserved GPU RAM   = 0.150 GiB
```

早期记录的 `8.34 s/1.99e-7` 使用的是文件开头连续 384 rows，不能与按
`k_perp` 分层抽样的 station-0 失败样本作耗时上的直接对照；这里改用完全
相同的分层行集。

因此 station-dependent PB 已达到远低于 EoR 恢复误差门的 operator closure。

## 5. 如何保证拟合成本可接受

不会在每次 optimizer iteration 中调用 OSKAR，也不会反复逐像素构造
visibility。计算分成两层：

```text
一次性 instrument calibration:
OSKAR Jones -> row beam cache -> exact visibility operator
-> sky-band probes -> R_alpha,beta

重复估计/拟合:
filtered visibility q_alpha -> small response solve
-> coarse windowed EoR bandpowers
```

当前完整源 context 是 1056 个 sky bands，输出是 408 个几何窗口。一次性
标定后，主要求解仅作用于 `408 x 1056` response 和其 coarse contraction。
OSKAR/Jones 不在重复拟合循环内。

为避免 20 个分区重复计算同一 Jones，正式实现先构造每个 `k_perp` bin
240 rows 的并集，然后每个 evaluator 只索引其中属于本分区的 12 rows：

- 共享 cache 为 7680 rows/frequency，正好等于 20 个分区的无重复并集；
- 单频共享 `complex64` coherency 约 15 GiB，64 频总量约 0.94 TiB；
- 每个 384-row 分区实际抽取约 48 GiB，并在 80-GiB GPU 上 materialize；
- 一次性共享 cache 实测约 4 h 45 min，替代逐分区重复计算约 24.5 h；
- 20 个 response 分区实测约 3 h 30 min；
- 合并 response 和执行四组 coarse 口径约 11 s；
- 64 频 OSKAR PB bank 本身约 3 h 33 min。

因此从零开始的研究级标定仍是约 12 小时、约 1 TiB 中间存储的重任务，但
可以在单台 A100/A800 80-GiB 机器上完成。标定完成后，重复功率谱估计是秒级
小矩阵运算。若转向生产使用，应把频率 cache 依次流过所有分区或压缩 Jones
cache，避免长期保留 0.94 TiB；这属于 I/O/存储优化，不改变测量方程。

## 6. 多频 PB 结果

### 6.1 8 频工程门

8 个真实 OSKAR PB 频点的 operator 总 L2 闭合为 `1.1538e-6`，逐频最大
`1.3665e-6`，evaluator 运行 `13.4 s`。但是 0.8-MHz 总带宽只给出 23 个
受支持输出；在冻结的 relative-response `0.1` 和 EoR-window locality
`0.8` 条件下没有科学窗口。该结果只证明多频 PB 软件链正确，不能作为
EoR 恢复结论，也没有通过降低选择门来制造窗口。

### 6.2 64 频正式测试

正式 bank 覆盖 `114.7--121.0 MHz`，输入 64 频，经全带宽 hard-DPSS 后
估计中央 32 频。20 个分区合并为 7680 rows，source response 为
`408 x 1056`、秩 408，保留条件数 `426.2`。合并 operator 对 OSKAR EoR
visibility 的 L2 闭合为 `1.1000e-6`；所有分区/频率中的最大值为
`3.4704e-6`，均通过 `1e-5` 门。

固定选择条件仍为 `kperp index >=4`、relative response `>=0.1`、几何窗口
response fraction `>=0.95`。结果为：

| profile | 选择窗口 | 覆盖名义细格 | 严格通过 | bank total L2 | heldout 最坏 L2 | 最大逐窗误差 |
|---|---:|---:|---:|---:|---:|---:|
| fine | 350 | 350 | 324 | 5.96% | 9.19% | 43.99% |
| `2x1` | 176 | 350 | 170 | 4.45% | 6.79% | 26.58% |
| `4x1` | **88** | **348** | **88** | **3.98%** | **6.28%** | **16.95%** |
| `4x2` | 45 | 348 | 45 | 3.92% | 6.13% | 16.23% |

正式 `4x1` 结果的详细指标为：

- 原 bank EoR+FG 积分比 `1.012884`，L2 `3.9788%`；
- 原 bank 最大逐窗误差 `16.946%`；
- 16 个 heldout total 积分比 `0.971269--1.018625`；
- heldout total 最坏 L2 `6.280%`；
- 最大 foreground-induced change `0.03513%`；
- response-window participation rank `36.48`；
- median effective width `28.09` 个 full-band source bands。

PB 相对 no-PB 把原 bank L2 从 `1.98%` 提高到 `3.98%`，但 heldout 最坏
L2 只从 `5.78%` 提高到 `6.28%`，仍有足够余量通过 20% 门。348 个名义
细格占 408 个几何 EoR-window cells 的 `85.29%`，但它们被聚合为 88 个
相互重叠的 broad windows，有效 participation rank 只有 36.48；不能声称
恢复了 348 个独立 PS2D cells。

## 7. 晋级条件与限制

PB 路线必须依次满足：

1. 每频 station-pair visibility closure `<1e-5`；
2. 多频 materialized operator 对 OSKAR EoR visibility closure `<1e-5`；
3. response/calibration 与 heldout probe 不读取 EoR truth 做选择；
4. 先复测无噪声 coarse-window 的 foreground+EoR 恢复；
5. 只有无噪声 PB 结果通过，才加入独立 time/noise split 和 cross-power。

前四项现已通过。第 5 项仍未执行，因此当前结果是无噪声 PB promotion，
不是最终观测 estimator。还需注意：测试固定 baseline-time rows、没有 uv
gridding，未传播 beam-model uncertainty；exact PB cache 与 OSKAR 仿真使用
同一 telescope model，因此验证的是 forward-chain correctness，而不是 PB
模型失配鲁棒性。

## 8. 复现入口

- coarse evaluator：
  `ops_scripts/evaluate_visibility_qbeta_coarse_covariance.py`
- 物理平移 evaluator：
  `ops_scripts/evaluate_visibility_qbeta_physical_shifts.py`
- Jones 收缩器：
  `ops_scripts/evaluate_oskar_aperture_row_beam_factors.cc`
- PB cache builder：
  `ops_scripts/build_oskar_aperture_row_beam_cache.py`
- PB visibility closure：
  `ops_scripts/evaluate_oskar_aperture_row_beam_closure.py`
- 多频 aperture-PB bank：
  `ops_scripts/run_chips_visibility_aperture_pb_pilot.sh`
- 多频 aperture-PB `Q_beta`：
  `ops_scripts/run_visibility_qbeta_aperture_pb_pilot.sh`
- 64 频 aperture-PB bank：
  `ops_scripts/run_chips_visibility_aperture_pb_wideband.sh`
- 64 频 aperture-PB `Q_beta`：
  `ops_scripts/run_visibility_qbeta_aperture_pb_wideband.sh`
- 机器可读总结：
  `docs/results/visibility_qbeta_aperture_pb_20260725_summary.json`
- 单频 closure 图：
  `docs/figures/oskar_aperture_beam_closure_20260725.png`
