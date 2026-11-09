# RoPE negative-position P2: evidence (fix beffdbe13, claims b199ccfa0)

## Regression `V2_Integration_CPURoPENegativePositionDirtyWorkspace`

Filter: `CPURoPEWorkshare.OrdinaryNegativePositionsIgnoreDirtyAngleWorkspace`.

Cells:
- FP32 full and FP32 partial: rows 1, 8, 40.
- FP16 and BF16: rows 1 and 40. The 32+ rows are the table path; rows below 32 use the direct
  path, which computes angles inline and does not read the workspace.
- `n_past`: -1 and -3.
- Workspace content before the call: Clean, NaN with flags set, finite stale, or left by a prior
  complete call at positive positions.
- Team: 1, or 3 with every member entering; one member sleeps 1 ms before entering. The sleep
  makes late arrival likely but does not prove arrival order under arbitrary scheduling.

Each cell checks three properties:
1. `width`: the observed OpenMP team width equals the request. Member 0 records it without a
   barrier. `OMP_THREAD_LIMIT=1` makes the check fail, so a serialized run cannot pass.
2. `raw`: raw output bytes equal a clean-workspace run. This is unrestricted byte equality and
   proves the output does not depend on prior workspace content.
3. `canonical`: output equals per-row rotation at each row's own position, **modulo signed
   zero**.
   - The reference uses the current one-row path at non-negative positions, plus the odd
     symmetry of rotation for negative rows. It is not independent arithmetic.
   - Equality with the original code is established separately by the cross-TU harness below.

## Negative control (unfixed RoPEPrimitives.cpp from 407347f6c with the b199ccfa0 test)

Log: `/tmp/armqual/negctl-407347f6c-rope-negative-position.log`
- canonical: 160/160 workspace/team cells fail, across every entry point, row count and start.
- raw: 112 fail. Clean-workspace cells cannot fail this check by construction, and some BF16
  workspace contents happen to coincide.
- width: 0 fail.

## Fixed tip b199ccfa0

- Mac portability: NEON_DOTPROD 271/271 and NEON 271/271, 0 skips.
- ProductionTestPreflight, repeat-until-fail:20: 215/215.
- gcc aarch64 and clang x86 AVX2/AVX512 syntax: OK.

## Cross-TU harness: base 425002104 TU vs fixed TU

Driver: `/tmp/armqual/rope_neg_driver.cpp`. argv[2] chooses the start. A negative or zero
argument is the exact `n_past`. A positive argument is an offset, so the start is argument + rows.

Scope: rows 1..64 contiguous; dims 64/128/256. Per start, the driver runs:
- FP32 full and partial (rotary dim/4);
- FP16 and BF16;
- the FP32 verifier, full and dim/4;
- the FP16/BF16/Q8_1 verifiers;
- Q8_1 integer.

Exclusion: `SKIP_Q8_ONE_ROW_NEGATIVE` omits only the Q8_1 integer call at rows==1 with a
negative start. The base program there is the stale-TLS defect, fixed earlier with its own
regression. Every other Q8 integer cell is compared.

Base one-row routes:
- The persistent-state route (`rows==1 ? &state`) aborts only at `n_past == -1`. Its
  `last_pos = -1` empty sentinel collides with the target, so `update_rope_cache` leaves empty
  planes and `apply_rope_cached_heads` throws `std::invalid_argument`. The driver does not catch
  it.
- The base's no-state API (`BASE_NO_STATE=1`) and its partial API give valid one-row
  observations at -1.

Results (97,884,192 bytes per negative start, 97,886,208 with Q8 included):

| base route | -1 | -3 | -40 | -100 |
|---|---|---|---|---|
| persistent state | base aborts | IDENTICAL | IDENTICAL | IDENTICAL |
| no-state API | IDENTICAL | IDENTICAL | IDENTICAL | IDENTICAL |

- Positive control, every cell including Q8: argument 0 (exact start 0) and argument 17231
  (start 17231 + rows): IDENTICAL.
- Unfixed 407347f6c TU vs base (no-state, -3): DIFFERS from byte 1.
