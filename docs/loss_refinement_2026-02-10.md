# Loss Refinement Note (2026-02-10)

本文档记录本次 `3dnet` loss 改动的科学动机、数学形式、实现位置和验证结果。

## 1. 总体目标

当前分离问题写作：

- 观测：`y = fg + eor`（当前阶段为纯天空，不含仪器项）
- 待优化变量：`fg, eor`

总损失结构（代码一致）：

`L_total = alpha*L_data + beta*L_smooth + gamma*L_eor + corr_weight*L_corr + s(t)*(fft_weight*L_fft + lagcorr_weight*L_lag + poly_weight*L_poly)`

其中 `s(t)` 由 `extra_loss_start_iter` 与 `extra_loss_ramp_iters` 控制。

## 2. 主要问题（改动前）

### 2.1 rFFT 项尺度与动态范围问题

旧 rFFT 项：

- 对 `fg` 沿频率轴做 rFFT，取最高频段能量图 `E_hf(x,y)`。
- 惩罚：`L_fft = mean_{x,y} [ (E_hf - mu) / sigma ]^2`

问题在于 `E_hf` 重尾明显，极少数亮像素可主导均值平方惩罚，导致：

- 该项数值巨大但方向辨识力弱（近似常数背景项）
- 与数据项/平滑项尺度严重不匹配

### 2.2 lagcorr 项采样与先验过硬

旧 lagcorr 项：

- 对每个 lag `l`，计算若干对频率切片相关系数 `rho_i^(l)`。
- 惩罚：`L_lag = 0.5 * [ mean_l mean_i ((rho_fg - m_fg,l)/s_fg,l)^2 + mean_l mean_i ((rho_eor - m_eor,l)/s_eor,l)^2 ]`

改动前实现使用“前 `max_pairs` 对”（head sampling），并且高 lag 的 `eor_lagcorr_sigma` 偏小，导致：

- 采样带频段偏置
- 高 lag 处先验过硬，对真实 EoR 波动容忍不足

## 3. 本次改动（含数学形式）

## 3.1 rFFT 稳健化

改动为：

1. 能量域改为对数域：`u = log(1 + E_hf)`  
2. 标准化后可选裁剪：`z = (u - mu_u)/sigma_u`, `z_clip = clip(z, -z_max, z_max)`  
3. 惩罚：`L_fft_new = mean(z_clip^2)`（不设 clip 时退化为 `mean(z^2)`）

并保持 prior 一致性：`mu_u, sigma_u` 在与损失同一变换域（log 域）从 FG reference 派生。

## 3.2 lagcorr 采样修正

将 lag 对采样从“前缀采样”改为“随机无放回采样”（当 `max_pairs` 启用时）：

- 旧：`pair_idx = [0, 1, ..., max_pairs-1]`
- 新：`pair_idx ~ UniformWithoutReplacement({0,...,N_lag-1}, max_pairs)`

这样减少固定频段偏置，使 lag 统计更接近全局分布。

## 3.3 lagcorr 先验放宽与重标定（mode sweep 默认）

针对两组真值诊断结果，调整了 `run_loss_mode_sweep.py` 默认先验：

- `lagcorr_pair_sampling: random`
- `lagcorr_max_pairs: 256`（从 50 提高）
- FG lag sigma 放宽
- EoR lag mean/sigma 调整，尤其高 lag sigma 显著放宽
- `corr_prior_sigma` 从 `0.2` 放宽到 `0.5`

## 4. 实现位置

- `3dnet/losses.py`
  - `L_fft` 增加 `log1p` 变换和 `fft_z_clip`
  - lagcorr 增加 `pair_sampling` 与随机采样
  - `derive_fft_prior_from_cube` 支持 `use_log_energy`
- `3dnet/separation_optim.py`
  - 新增配置项：`lagcorr_pair_sampling`, `lagcorr_random_seed`, `fft_use_log_energy`, `fft_z_clip`
  - 优化路径与诊断路径都接入新参数
- `3dnet/separation_cli.py`
  - 新增 CLI 覆盖参数
- `3dnet/run_loss_mode_sweep.py`
  - 更新默认先验与稳健参数

## 5. 先验一致性定量结果（真值上直接评估）

使用 `data/stage_1024_20260210` 真值 `fg/eor`：

- cube1:
  - lagcorr 总惩罚：`2.3071 -> 0.0685`（约 `33.7x` 降低）
  - rFFT 惩罚：`1934.38 -> 0.7907`
- cube2:
  - lagcorr 总惩罚：`1.0665 -> 0.2067`（约 `5.16x` 降低）
  - rFFT 惩罚：`22.22 -> 0.8089`

解释：新先验与真值统计更一致，且 rFFT 数值尺度回到与其他项可比区间。

## 6. 改后验证跑（两组数据）

运行：

- 输出：`runs/loss_refine_eval_20260210_v2/mode_sweep_results.csv`
- 数据：`stage_1024_20260210`，`cube1/cube2`
- 模式：`base,rfft,lagcorr`
- 迭代：`600`

主要结果：

- cube1:
  - `base`: `eor_mse=1.37584e-4`, `corr=0.44442`
  - `rfft`: `eor_mse=1.37584e-4`, `corr=0.44443`, `final_fft_loss=0.79068`
  - `lagcorr`: `eor_mse=1.46779e-4`, `corr=0.29350`, `final_lagcorr_loss=0.013142`
- cube2:
  - `base`: `eor_mse=1.51104e-4`, `corr=0.45519`
  - `rfft`: `eor_mse=1.51105e-4`, `corr=0.45520`, `final_fft_loss=0.80894`
  - `lagcorr`: `eor_mse=1.58610e-4`, `corr=0.28960`, `final_lagcorr_loss=0.011113`

## 7. 解释与结论

1. `rfft` 现在不再是巨大常数背景项，数值上已“可训练”。  
2. 在当前目标下，`rfft` 与 `base` 仍几乎重合，说明其辨识力仍弱于数据项+平滑项。  
3. `lagcorr` 与真值先验冲突已显著下降（可从真值惩罚和 `final_lagcorr_loss` 看出），但在 600 iter 下仍会牺牲图像域指标。  
4. 下一步应优先优化 `lagcorr` 调度和分项权重，而不是继续硬收紧先验。

## 8. 建议的下一轮优化

1. `lagcorr` 分 FG/EoR 两个权重（`w_lag_fg`, `w_lag_eor`），先降低 EoR 权重。  
2. 将 `lagcorr` 激活推迟到更后期，并增大 ramp。  
3. rFFT 继续尝试“相对能量”或“频谱斜率”型先验，而不只用绝对高频能量。  
