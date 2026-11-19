# ROCm NativeQuantised trainer smoke fixtures (authored test inputs)

These three CSVs are **deterministic, hand-authored test inputs** for the existing
ROCm NativeQuantised parser/generator/validator contracts. They are NOT measured
performance corpora. They live in the source repository as small device-free
fixtures, exactly as the validators require.

## Provenance

- Author: local WSL qualification, PR #43 (`feat/arm64-native-kernels`), fixing the
  model-free Unit gate. These three files were missing from every commit on
  `upstream/develop` and `b47de952d`; the four `V2_Unit_ROCmNativeQuantised*` tests
  therefore failed with "file not found" for their `fixtures/*smoke.csv` inputs.
- All timing fields (`min_us`, `median_us`, `p95_us`, `mad_us`, `relative_l2`,
  `max_abs`, `gflops`, `eff_bw_gbs`, `speedup_vs_int8`) are **synthetic labels**.
  They encode the parsing/validation/emission contracts the validators check; they
  are NOT measurements of any hardware and must NOT be interpreted as such.

## Does NOT do

- Does NOT fit or install dispatch policy.
- Does NOT update any installed include, dispatch registry, numerical/generation
  baseline, or high-water corpus.
- Does NOT imply measured performance or a certification.
- Does NOT alter the intentionally incomplete IQ4_NL / IQ4_XS alias contest: the
  decode fixture keeps IQ4_XS source codebook 4 / physical execution codebook 24 and
  IQ4_NL source 4 / physical 4 at distinct geometries, so the generator still rejects
  the incomplete alias coverage ("no eligible candidate covers every required surface")
  rather than inventing observations to fill that gap.
- Generated artifacts from these fixtures are always written to a temporary directory
  inside the test and are discarded.

## Files

- `rocm_native_quantised_prefill_alias_smoke.csv`
  - Rows: Q4_1, Q5_1, Q8_0, Q8_1, Q8_K, IQ4_XS, IQ4_NL in `prefill` phase.
    Q8_1 (source codebook 20) and Q8_K (source 21) both execute on physical
    codebook 19 and collapse into the single generated `CB=19 (Q8_0)` runtime
    branch together with Q8_0.
  - Satisfies `validate_rocm_native_quantised_trainer_csv.py`
    (`--require-alias-group --require-policy-prefill-m --require-prefill-m 600`) and the
    prefill generator's fragments `CB=5/7/19/24/4`, `{128,32,1,2}`, `M=600`,
    `qwen36_ffn_down_bucket`.
- `rocm_native_quantised_decode_alias_smoke.csv`
  - Single `decode` phase fixture carrying both the raw CSV validator columns
    (`format`, `codebook`, `eff_bw_gbs`, `speedup_vs_int8`, `variant`, `is_best`) and
    the decode adapter's `source_format`/`source_codebook`/`execution_codebook`/
    `observed_path`/digest columns.
  - Diagnostic rows Q4_1 (cb5), Q5_1 (cb7), Q8_0 (cb19) resolve to
    `rocm.native_quantised.decode.fast.kb16` producing `{16,4}` and the `alias_q4_1`/`q5_1`/`q8_0`
    shape labels.
  - IQ4_XS (source4/physical24) + IQ4_NL (source4/physical4) at different geometries
    exercise the incomplete-coverage negative control.
- `rocm_native_quantised_batched_decode_smoke.csv`
  - Two `batched_decode` rows with `projection_ns=4+12` and `8+12`, codebooks `5+7`
    (Q4_1/Q5_1), shapes `Qwen36_GDN_Q4K_qkv_z` / `Qwen36_GDN_Q5K_qkv_z`, and the
    `rel_l2=2.500e-04`, `max_abs=1.750e-03` fragments.

## Verification

Each of the four `V2_Unit_ROCmNativeQuantised*` validators passes locally against
these fixtures (see the gate evidence for the exact commands and outputs).
