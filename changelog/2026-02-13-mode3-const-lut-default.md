# Mode3 const-LUT default enabled (ROCm IQ4_NL)

## Summary
Enabled the mode3 const-LUT decode path by default for IQ4_NL ratio-VNNI kernels on ROCm.

## Code Changes
- `src/v2/kernels/rocm/ROCmGemvKernel.hip`
  - Changed `iq4_mode3_const_lut` default from `false` to `true`.
  - Preserved `LLAMINAR_RATIO_IQ4_MODE3_CONST_LUT` environment variable override behavior.

## Validation (Release)
- Target: `build_v2_release/tests/v2/v2_perf_rocm_ratio_vnni_kernel`
- Filter: `ROCmRatioVNNIPerfTest.Phase1Q4AndIQ4SpeedupVsInt8VNNI`
- Env: `LLAMINAR_RATIO_IQ4_CPT=1`, `LLAMINAR_RATIO_IQ4_PREFETCH_NEXT=1`, `LLAMINAR_RATIO_IQ4_DECODE_MODE=3`
- 10-run global ratio/int8 speedup values:
  - `1.52002, 1.44085, 1.43779, 1.44489, 1.52665, 1.43090, 1.43837, 1.43999, 1.44420, 1.51279`
- Aggregate:
  - Mean: `1.46365x`
  - Median: `1.44252x`
  - Min/Max: `1.43090x / 1.52665x`
  - Stddev (population): `0.03708`
- Stability:
  - `PASSED_COUNT=10/10`

## Notes
- Logs directory: `/tmp/iq4_mode3_defaulton_10run_20260213_150741`
- This keeps compressed IQ4 storage and only changes default runtime dispatch behavior for mode3.

## Next Steps
- Optional: run a paired 10-run A/B (`MODE3_CONST_LUT=0` vs default-on) and record median deltas in the next changelog entry.
