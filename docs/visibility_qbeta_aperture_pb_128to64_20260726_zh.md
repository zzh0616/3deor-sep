# Exact-PB `128 -> 64` 增频测试

## 目标

在不加入噪声、不改变天空 realization、也不引入新的前景形态先验的条件
下，把 visibility-domain `Q_beta` 的输入/输出带宽从当前
`64 -> 32` 个 100-kHz channels 扩到 `128 -> 64`，检验更细的
`k_parallel` 采样、更多 radial combinations 和更宽 DPSS guard 是否提高
有限视场 EoR-window bandpower 恢复。

## 输入合同调整

原理想方案为输入 `111.5--124.2 MHz`、输出 `114.7--121.0 MHz`。启动前
发现当前 matched cube2 truth 只覆盖 `106.0--121.0 MHz`。独立的 `f120`
truth 覆盖 `120--135 MHz`，但重叠层 EoR 相关仅 `0.009--0.013`，不能与
cube2 拼接。

本轮采用同一 cube 内的可执行方案：

- input：`108.3--121.0 MHz`，128 channels，`12.8 MHz`；
- analysis：`111.5--117.8 MHz`，64 channels，`6.4 MHz`；
- guard：每侧 `3.2 MHz`；
- 复用 exact-PB bank：`114.7--121.0 MHz`；
- 新增 exact-PB bank：`108.3--114.6 MHz`。

功率谱几何以 analysis 中心 `114.65 MHz` 冻结；visibility-bank 的行抽样
参考频率仍使用旧 bank 的 `117.85 MHz`，只为保证新旧 shards 的
`sample_row_indices` 完全一致。

## 两阶段统计

第一阶段是 4-partition screen：

- `12 rows/(partition, kperp bin)`；
- shared cache 为 `48 rows/kperp bin`；
- exact station-pair factor仍为 `0.5 Tr(E_p E_q^H)`；
- full-band hard-DPSS 后截取中央64频，再做 Hann delay transform；
- response source scope 为 `all_in_range_with_nyquist`；
- 输出继续评估 `fine / 2x1 / 4x1 / 4x2` coarse profiles。

若 operator closure、response locality 和无噪声 coarse-window 恢复通过，
第二阶段晋级到原正式口径的 20 partitions。还必须增加同一 analysis 子带
上的 `filter-bandwidth-scope=analysis_subband` 对照，才能把提升归因于
guard bandwidth。

## 内存与执行方式

一个 partition 的 dense complex64 exact operator 大约为

`128 x 384 x 262144 x 8 bytes = 96 GiB`。

它不能常驻单张 80-GiB GPU，因此使用 `cpu_streamed`：完整 operator 放在
Genoa 主存，逐频矩阵送入 GPU。Genoa 有 2.2 TiB RAM，可以并行三个 screen
partitions；shared 4-partition PB cache 预计约 `0.38 TiB`。

## 4 分区配对结果

宽带和窄带对照均已完成。两者使用同一 128 频 visibility bank、同一天空、
同一 sampled rows 和同一 exact-PB cache；唯一受控差异是 DPSS 在完整
128 频输入上执行，还是只在中央 64 频 analysis subband 上执行。

共同检查结果：

- combined rows：`48/kperp bin`，共 1536 行；
- exact operator closure：`8.1858e-7`，覆盖 196608 个 visibilities；
- response shape：`832 x 2080`；
- calibration/validation response L2：宽带 `0.14497`，窄带 `0.14365`。

宽带带来的主要改善不是 4 分区 bank 振幅立刻达到无偏，而是数值条件、
留出稳定性和 FG 抑制同时变好：

| response-weighted `4x1` 指标 | 宽带 `full_input` | 窄带 `analysis_subband` |
|---|---:|---:|
| response rank | `832/832` | `815/832` |
| retained condition number | `565.7` | `9832.9` |
| selected groups | `166` | `179` |
| strict fraction | `0.8675` | `0.8603` |
| participation rank | `49.85` | `48.15` |
| held-out worst relative L2 | `0.09641` | `0.12645` |
| held-out integrated-ratio range | `0.96935--1.03138` | `0.95342--1.04153` |
| maximum FG effect | `8.47e-4` | `2.46e-3` |
| bank-total ratio | `0.85720` | `0.85935` |
| bank-total relative L2 | `0.14981` | `0.15444` |

因此，宽带相对窄带把条件数改善约 `17.4x`，held-out worst L2 降低约
`24%`，最大 FG effect 降低约 `2.9x`，并恢复满秩。与此同时，宽带
selected groups 数量从 179 降到 166，且两者的低行数 bank-total 偏差仍约
15%。结论是“通过正式计算的工程晋级门”，不是“4 分区已经完成科学恢复”。

机器可读结果保存在：

- `docs/results/visibility_qbeta_aperture_pb_128to64_screen4_20260727/wide_combined_result.json`；
- `docs/results/visibility_qbeta_aperture_pb_128to64_screen4_20260727/wide_coarse_summary.json`；
- `docs/results/visibility_qbeta_aperture_pb_128to64_screen4_20260727/narrow_combined_result.json`；
- `docs/results/visibility_qbeta_aperture_pb_128to64_screen4_20260727/narrow_coarse_summary.json`。

## 启动修复

首次 screen 启动暴露了两项工程集成问题，而不是科学算法失败：

1. 新 analysis config 缺少 coarse evaluator 要求的
   `reporting_masks`；
2. launcher 默认使用 base Python，远端该环境没有 PyTorch。

两项均已修复。宽/窄 launcher 现在默认使用已验证的 `torch` 环境，并在
重计算前 fail fast 检查 CUDA 和 reporting config。相关本地测试为
`9 passed`。

## 正式 20 分区

原始正式启动在选行阶段停止：最高横向 bin（zero-based index 31，
约 `1.533--1.582 Mpc^-1`）只有 72 条可用行，低于
`20 partitions x 12 rows = 240`；bins 0--30 均至少有 277 条。不能为
一个稀疏边缘 bin 把所有 bins 降到 6 partitions，因此新增了显式
`maximum_kperp_index_exclusive` support gate。该 gate 只依赖观测行支持，
不读取 EoR truth 或恢复误差；宽带和窄带对照均固定为 31。

正式运行状态：

- host：SKA-Genoa (`119.78.226.31`)；
- code snapshot：
  `/data1/zhenghao/fg_rmw/code/3dnet_128freq_20260726`；
- source root：
  `/data1/zhenghao/fg_rmw/runs/cube2_fullsky_isobeam_512_128freq_20260726`；
- bank root：
  `/data1/zhenghao/fg_rmw/runs/chips_visibility_aperture_pb_128freq_20260726`；
- formal wide root：
  `/data1/zhenghao/fg_rmw/runs/visibility_qbeta_aperture_pb_128to64_promotion20_20260727`；
- formal wide PID：`428714`；
- formal narrow root：
  `/data1/zhenghao/fg_rmw/runs/visibility_qbeta_aperture_pb_128to64_narrow_control_promotion20_20260727`；
- formal narrow watcher PID：`428873`。

正式宽带已于 `2026-07-28` 约 05:03 完成：

- partition count：`20`；
- combined rows：`240/kperp bin`，共 `7440` 行；
- exact operator closure：`8.9411e-7`，覆盖 `952320` 个 visibilities；
- response：`810 x 2080`，在当前支持域内满秩 `810/810`；
- retained condition number：`540.80`；
- calibration-validation response L2：`0.08212`。

response-weighted `4x1` 正式结果为：

| 指标 | 正式宽带 |
|---|---:|
| selected / strict groups | `147 / 130` |
| strict fraction | `0.8844` |
| minimum response locality | `0.9584` |
| held-out worst relative L2 | `0.07729` |
| held-out integrated-ratio range | `0.98602--1.02387` |
| maximum FG effect | `2.740e-4` |
| response participation rank | `46.37` |
| bank-total ratio | `0.85396` |
| bank-total relative L2 | `0.15488` |

相对 4 分区宽带预筛，calibration-validation response L2 降低 `43.4%`，
held-out worst L2 降低 `19.8%`，最大 FG effect 降低 `3.09` 倍。正式
response 的 `810/810` 与预筛的 `832/832` 都是满秩；维数差异只来自预先
冻结的 `maximum_kperp_index_exclusive=31`。

但绝对 physical-bank 偏差没有随采样量增加而消失：bank-total ratio 从
`0.85720` 变为 `0.85396`，relative L2 从 `0.14981` 变为 `0.15488`。
因此约 `14.6%` 的整体低估不是有限行 response-calibration noise。后续
振幅-相位和 lightcone stationarity 诊断已把它定位为：当前全输入带宽
Fourier-band response 隐含全局平稳、近似对角 source covariance，而
`108.3--121.0 MHz` EOS lightcone 在该带宽内明显演化。它不是 exact-PB
operator 闭合失败。正式宽带通过了工程稳定性检查，但没有通过绝对科学
恢复验收。

窄带 watcher 已正常复用宽带 exact-PB cache，并于
`2026-07-28 13:35 +0800` 完成全部 `20/20` 分区、combine 和 coarse
evaluation。两臂没有 traceback、OOM 或 fatal error，当前也没有遗留
worker。窄带同样使用 7440 rows，operator closure 与宽带完全相同：
`8.9411e-7`。

## 正式宽/窄配对结论

两臂唯一受控差异仍是 DPSS 使用完整 128 频输入，还是只使用中央 64 频
analysis subband：

| 正式 `4x1` 指标 | 宽带 `full_input` | 窄带 `analysis_subband` |
|---|---:|---:|
| response rank | `810/810` | `790/810` |
| retained condition number | `540.80` | `7906.87` |
| calibration-validation L2 | `0.08212` | `0.08192` |
| selected groups | `147` | `159` |
| strict groups | `130` | `142` |
| held-out worst L2 | `0.07729` | `0.05849` |
| maximum FG effect | `2.740e-4` | `8.791e-4` |
| bank-total ratio | `0.85396` | `0.84674` |
| bank-total relative L2 | `0.15488` | `0.17728` |

因为两臂 selected groups 不完全相同，不能直接把上表全部差异归因于
filter。集合核对显示，宽带 147 个 groups 是窄带 159 个 groups 的严格
子集。在这 147 个共同几何窗口上：

| common-support `4x1` 指标 | 宽带 | 窄带 |
|---|---:|---:|
| strict groups | `130` | `130` |
| bank-total ratio | `0.85396` | `0.84494` |
| bank-total relative L2 | `0.15488` | `0.17747` |
| held-out worst L2 | `0.07729` | `0.05854` |
| maximum FG effect | `2.740e-4` | `7.948e-4` |

因此宽带 guard 把 response condition number 改善 `14.62` 倍，在共同
窗口上把 physical-bank L2 降低 `12.7%`、最大 FG effect 降低 `2.90`
倍；但 held-out worst L2 高 `32.0%`。窄带额外保留的 12 个 groups
主要位于较高横向 bins；它们在宽带中是因为 response locality 低于冻结的
95% 门限而被排除，而不是 input response 不足。

正式结论是：宽 guard 对数值条件、前景控制和共同窗口的 physical-bank
恢复有实际收益，但会损失 12 个 response-local windows，也没有消除约
15% 的绝对低估。该低估来自 source covariance/target 参数化，不应通过
truth-derived 标量校正。下一步要改成保留 cross-window covariance 的重叠
局域频率 estimator，而不是继续增加同一全局 response 的 rows。

正式宽带机器可读结果保存在：

- `docs/results/visibility_qbeta_aperture_pb_128to64_promotion20_20260728/wide_combined_result.json`；
- `docs/results/visibility_qbeta_aperture_pb_128to64_promotion20_20260728/wide_coarse_summary.json`；
- `docs/results/visibility_qbeta_aperture_pb_128to64_promotion20_20260728/narrow_combined_result.json`；
- `docs/results/visibility_qbeta_aperture_pb_128to64_promotion20_20260728/narrow_coarse_summary.json`；
- `docs/results/visibility_qbeta_aperture_pb_128to64_promotion20_20260728/paired_summary.json`；
- `docs/results/visibility_qbeta_aperture_pb_128to64_promotion20_20260728/status_summary.json`。

## 绝对低估复核

完整诊断见
`docs/visibility_qbeta_response_bias_diagnosis_20260728_zh.md`。关键结果为：

- actual-amplitude、全局随机相位 ensemble 的宽/窄 ratio 为
  `1.029 +/- 0.046` 和 `1.040 +/- 0.048`，排除 mode 振幅、单位换算和
  response 对角项错误；
- 在两个互斥 rows partitions 上，保留每个空间 mode 完整跨频 covariance、
  只随机化空间相位后，宽/窄 ratio 为 `0.875 +/- 0.018` 和
  `0.866 +/- 0.027`；同 rows physical 值为 `0.872/0.864`，只差
  `0.16/0.08 sigma`；
- 旧 `64 -> 32` 的同类两分区 ratio 为 `0.998 +/- 0.020`，与正式
  20 分区 physical ratio `1.0129` 相容；
- 新 analysis/input 频道空间方差比为 `0.8743`，旧频段为 `0.9813`；
- 新旧重叠 64 个频道的 sky 和 OSKAR visibility 合同逐字节相同。

因此旧窄带没有明显低估，是因为其较短频段内 lightcone 近似平稳；不能把
它解释为旧 operator 比新 operator 更可靠。
