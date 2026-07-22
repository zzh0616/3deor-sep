# Ops Scripts

This directory holds operational runner/fetch/monitor helpers.
Root-level wrappers are kept for backward-compatible command paths.

## Partial-window covariance experiment

- `build_partial_window_covariance_bank.py` rebuilds an exact-frequency cached
  operator bank and propagates foreground/EoR covariance probes.
- `estimate_partial_window_covariance_ps2d.py` is the experimental Wiener
  covariance posterior. It is retained as a negative control, not the promoted
  PS2D estimator.
- `diagnose_partial_window_covariance_model.py` compares the probe covariance
  family with a truth-only oracle mathematical control.
- `estimate_partial_window_debiased_ps2d.py` is the promoted noiseless
  bandpower-level foreground-covariance bias subtraction baseline.
- `run_partial_window_covariance_32high.sh` runs the 32-high operator build and
  both estimators without frequency interpolation.
