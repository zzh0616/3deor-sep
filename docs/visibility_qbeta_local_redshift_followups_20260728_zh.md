# 局域红移宽频修复与四项补充验证

## 1. 目的

全局 `128 -> 64` exact-PB 估计在增加 response rows 后仍稳定低估
EoR bandpower 约 `14.6%`。闭合、单位、PB cache 和对角 source response
均已排除；保留每个空间 Fourier mode 完整跨频复向量的 surrogate 可以重现
该低估，说明主要问题是把强演化 lightcone 当作全带平稳、近似频率对角的
source covariance。

本轮不使用仿真真值做标量修正，而是完成以下五项工作：

1. 用重叠的局域红移窗口替代单个全带 target；
2. 在固定 visibility rows 上测量 `1/2/4/8` 个 response probes 的收敛；
3. 传播一条未参与选择的独立物理 lightcone；
4. 用共享全带 realization 估计所有局域窗口的完整 covariance；
5. 加入中央方形图像以外的 foreground，并按其实际角距重新定义 DPSS
   与 EoR-window 支持。

所有窗口和筛选阈值均在查看物理 EoR 结果前冻结。前景星表、EoR template
和 truth-derived correction 均未进入 estimator。

## 2. 局域红移窗口

完整输入为 `108.3--121.0 MHz` 的 128 个 100-kHz channels。每个局域
估计使用 64 个输入 channels，在其中央估计 32 个 channels；相邻窗口按
16 channels 平移，因此分析带有 50% 重叠：

| window | input [MHz] | analysis [MHz] | selected groups |
|---|---:|---:|---:|
| 0 | 108.3--114.6 | 109.9--113.0 | 88 |
| 1 | 109.9--116.2 | 111.5--114.6 | 88 |
| 2 | 111.5--117.8 | 113.1--116.2 | 87 |
| 3 | 113.1--119.4 | 114.7--117.8 | 87 |
| 4 | 114.7--121.0 | 116.3--119.4 | 86 |

五个窗口共用以 `114.65 MHz` 为参考的 transverse geometry，避免把
坐标变化误当作 lightcone 演化。每个窗口使用四个互斥 row partitions，
每个 transverse bin 每分区 12 rows，并采用相同 exact aperture-PB
visibility operator、full-input DPSS、Hann taper 和 response-only selection。

`quad_kperp_response` 的物理 cube2 结果为：

| centre [MHz] | integrated ratio | relative L2 | median / p90 error | <20% |
|---:|---:|---:|---:|---:|
| 111.45 | 0.9016 | 0.0875 | 0.111 / 0.233 | 72/88 |
| 113.05 | 0.9060 | 0.1142 | 0.0649 / 0.195 | 79/88 |
| 114.65 | 0.9317 | 0.0692 | 0.0678 / 0.181 | 79/87 |
| 116.25 | 0.9823 | 0.0489 | 0.0603 / 0.151 | 85/87 |
| 117.85 | 1.0311 | 0.0703 | 0.0762 / 0.162 | 80/86 |

局域化把单个全带 `0.854` 的 integrated ratio 明显拉近 1，但没有得到
一个对所有红移和 realization 均严格无偏的结果。它是对全局平稳性错误的
有效缓解，不是一个可用 truth-derived scalar 替代的完全修复。

把输入和输出都进一步缩为 `32 -> 16` 的负对照得到 ratio `0.8777`、
relative L2 `0.1501`。短带减少了径向分辨率和可辨认的 delay combinations，
因此“窗口越短越局域”并不单调等于“估计越好”。

## 3. 固定 rows 的 response-probe 收敛

为了把 response Monte Carlo 误差与 row sampling 误差分开，测试始终使用
同一组 visibility rows，只改变 calibration probes。对一个中央分区枚举
全部 `1/2/4/8` probe 子集：

| probes | response row-sum L2 to 8-probe | bank-ratio std. | selection Jaccard |
|---:|---:|---:|---:|
| 1 | 0.0660 | 0.0162 | 0.9986 |
| 2 | 0.0432 | 0.0107 | 1.0000 |
| 4 | 0.0249 | 0.0062 | 1.0000 |
| 8 | 0 | 0 | 1.0000 |

这说明一个 probe 足以稳定确定报告支持，但 response normalization 仍有约
百分数级 Monte Carlo 波动。更重要的四分区对照中，四 probes 给出 ratio
`0.92950`、relative L2 `0.07137`；原单 probe 结果为 `0.93172` 和
`0.06922`。integrated ratio 只移动 `0.22` 个百分点，因此局域中央窗口的
改善不是单 probe 偶然波动。

## 4. 独立物理 lightcone

`eor_cube1.fits` 使用与 cube2 完全相同的中央 1024 crop 和 `2x2`
平均，得到 512 像素、32 arcsec 的独立 sky。它只作为 evaluation sky
传播，不参与 response、window 或阈值选择。五个窗口的 foreground+cube1
结果为：

| centre [MHz] | integrated ratio | relative L2 | <20% |
|---:|---:|---:|---:|
| 111.45 | 0.9601 | 0.0576 | 87/88 |
| 113.05 | 0.9545 | 0.0709 | 88/88 |
| 114.65 | 0.9407 | 0.0682 | 85/87 |
| 116.25 | 0.9480 | 0.0728 | 87/87 |
| 117.85 | 0.9102 | 0.1372 | 84/86 |

去掉 foreground 后结果几乎不变，说明差异仍由 EoR realization 的跨频
covariance，而不是前景残留主导。独立 lightcone 支持局域方法具有迁移性，
同时否定了“任意 realization 上均达到几乎无偏”的更强说法。

## 5. 完整跨窗口 covariance

使用 512 个共享全带 realizations。每个 realization 对每个空间 Fourier
mode 施加一个共同二维相位，同时保留该 mode 的完整 128-frequency 复向量，
因此保留物理 lightcone 的跨频 covariance。相同 realization 被传播到五个
重叠窗口，共得到 436 个 concatenated bandpowers；covariance 没有参与
窗口选择。

- 样本秩上限为 511，高于 436 个输出；
- covariance participation rank 为 `7.33`；
- 相邻窗口的 median absolute correlation 约 `0.030`，最大值为
  `0.414--0.500`；
- 非相邻窗口最大 absolute correlation 为 `0.163--0.198`；
- 在 `rcond=1e-6` 下，样本内分解保留 403 个 eigenmodes。

样本内 whitening 的非对角项约 `1.6e-12` 只是线性代数恒等式，不能当作
独立稳定性证据。两个 `256/256` train/test folds 中，test/train variance
ratio 的中位数为 `1.74/1.81`，90 分位为 `6.83/7.09`，test 最大相关为
`0.375/0.407`。因此本轮可靠交付物是完整 covariance、跨窗口相关和
正交 KL 定义；403 个样本内 modes 不能全部宣称为独立测量。未来若需要
precision matrix，应使用更多 realizations 或预注册的 shrinkage/structured
covariance，而不是直接反演经验 covariance 尾部。

## 6. 外场支持门

外场 sky 从同一个完整 `2048x2048` foreground cube 构造：保留中央
1024 方形以外的全部 196,608 个 coarse directions，像素为 64 arcsec，
最大角距 `6.44353 deg`。中央参考 cube 的相对 L2 和最大绝对差都严格为
0。64 个频率的 exact OSKAR station-pair PB cache 已全部构建。

用旧 `3.21812 deg` 图像角点作为 DPSS 支持时，base Q 和 coarse
recomputation 均严格闭合为 0，但外场引入的 integrated absolute power
为 EoR target 的 `83.07` 倍，extended-total ratio 为 `83.995`。其中中央
方形外但仍在 3.218 deg 内的区域只贡献 `0.0050`，灾难性 leakage 来自
3.218--6.444 deg 的 foreground。这是延迟 nuisance support 定义过窄，
不是 central-chain 或 outer evaluator 的闭合错误。

修复实现新增显式 `foreground_support_angle_deg`。该值同时控制：

1. 每个 transverse bin 的最大几何 delay；
2. 冻结 EoR-window 的 patch-wedge slope；
3. 分析 window energy 与 contract hash。

旧配置未设置该字段时保持逐字相同的 analysis/input hashes。设置
`6.4436 deg` 后，wedge slope 从 `0.24157` 增至 `0.49270`，可报告
independent modes 从 379,480 降至 296,716，仍保留约 78%。四分区
response 重标定得到 74 个 selected groups；中央 sky 的 ratio/L2 为
`0.90298/0.10197`，maximum foreground effect 为 `7.67e-4`。

复用既有 145-GiB outer PB cache 后，修复前后的外场结果为：

| support | groups | outer induced integrated abs. | extended ratio | extended L2 | <20% |
|---:|---:|---:|---:|---:|---:|
| 3.218 deg | 87 | 83.0658 | 83.9953 | 156.315 | 67/87 |
| 6.444 deg | 74 | 0.02973 | 0.93245 | 0.08483 | 74/74 |

因此与实际模拟角支持一致的 DPSS 将外场积分泄漏压低约 2,800 倍，最终
总谱的所有 74 个窗口都在 target 的 20% 内。修复后的外场效应中位数和
90 分位为 `0.94%/7.04%`，但仍有 6 个窗口超过 10%、3 个超过 20%，最坏
为 30.9%；这些窗口全部位于最低的 `kperp index 4--7` 组。事后去掉整个
最低 transverse 组会留下 60 个窗口，外场积分/最大效应降到
`1.11%/2.75%`，extended ratio 为 `0.9589`。该 cut 是发现外场结果后的
诊断，不能作为本轮主结果的预注册选择；若用于文章主 mask，必须先冻结
并用独立 outer-sky realization 验证。

## 7. 当前解释边界

本轮已经补齐 response Monte Carlo、独立 lightcone 和跨窗口 covariance，
并把此前隐含的 foreground angular support 变成显式、可冻结参数。
6.444-deg 复测通过积分与总体 bandpower 门，但仍有少量低-`kperp`
单窗受到较大外场影响。它只覆盖当前物理 foreground cube 和 exact PB
支持，不等于 full-horizon、热噪声、校准误差、RFI 或电离层测试。因此
现在可以把有限 patch 结果晋级为“在已模拟 6.444-deg 有限支持内验证的
noiseless feasibility”，仍不能称为 observation-ready recovery。

## 8. 可复核产物

- 汇总图：`docs/figures/visibility_qbeta_local_redshift_followups_20260728.png`
- 机器摘要：
  `docs/results/visibility_qbeta_local_redshift_followups_20260728/summary.json`
- 完整 436 维 covariance：
  `docs/results/visibility_qbeta_local_redshift_followups_20260728/covariance/products.npz`
- 汇总脚本：
  `ops_scripts/summarize_visibility_qbeta_local_redshift_followups.py`
- 远端主根：
  `/data1/zhenghao/fg_rmw/runs/visibility_qbeta_local_redshift_screen4_20260728`
- 宽支持重标定：
  `/data1/zhenghao/fg_rmw/runs/visibility_qbeta_local_redshift_support644_screen4_20260728`
- 外场 sky 与 exact-PB cache：
  `/data1/zhenghao/fg_rmw/runs/visibility_qbeta_outer_field_20260728`
