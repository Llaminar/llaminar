# Idea 31: AppleClang secondary gate at b4dd753c8 (parent 777f7a52a)

Host: M1 Max, macOS. Compiler: AppleClang-21 (`Apple clang version 21.0.0 (clang-2100.3.34.2)`).

Project: `tests/v2/portability`, Release (the documented AArch64 portability build). The candidate
and the parent were each built in both envelopes, NEON and NEON_DOTPROD: four clean builds of clean
detached worktrees. NEON_I8MM was not requested and was not run.

## (b)/(c) Unit + Integration, byte tests, 0-new/0-gone

| Tree / envelope | ctest | skips |
|---|---|---|
| cand NEON_DOTPROD | 292/292 pass | 0 |
| cand NEON | 291/291 pass | 0 |
| parent NEON_DOTPROD | 291/291 pass | 0 |
| parent NEON | 291/291 pass | 0 |

**Per-test against the parent**

- **NEON_DOTPROD:** new is exactly `V2_Integration_CPUNativeQuantisedNibblePanelReplay`
  (`V2_Integration_CPUNativeNEONNibblePanelReplay`); none gone; 0 status changes.
- **NEON:** identical test set; 0 status changes.

**Prepared-projection binary, all gtest cases rather than only the registered filters**

| Tree / envelope | Cases | Skips |
|---|---|---|
| cand NEON_DOTPROD | 15/15 | 0 |
| parent NEON_DOTPROD | 14/14 | 0 |
| cand NEON | 9/9 | 0 |
| parent NEON | 9/9 | 0 |

- The only new case is `DotProdNibbleS4RecordedInputReplay`, which passes. No case is gone.

**Byte sweep coverage, all passing in both envelopes**

- `{ASIMD,DotProd}PanelWidthsMatchTheStripProgram`: forced S1/S2/S4, byte-equal to S1 with untouched
  padding/sentinels. Covers every non-compact source format, N ∈ {1, 5, 8, 12, 15, 16, 17, 20, 31,
  33, 47, 63, 64, 65, 100}, K blocks ∈ {1, 3, 17}, both policies and rows 1–4.
- `{ASIMD,DotProd}EverySourceAndEncodingMatchesSerialAndSourceOrder` and the `Compact` variant.
- `EveryHalfAndOpaqueWordHasExactBits`, `ActivationSumsIncludeSignedMinimum` (extremes), and the
  tail-padding contracts.
- The new replay: centered and nonlinear recorded inputs × both policies, S4, dirty reset, repeated
  replay, guards and inputs unchanged.

## (a) Register witness

### Installed (canonical, Strips = 0)

- Passing in both envelopes: `V2_{Unit,Integration}_CPUNativeNEONTileRegisters` and
  `…TileRegisterEvidence{Analysis,Binding0-3}`.
- Every Rows1 program has 0 spills.
- The installed witness objects of the candidate and the parent are **instruction-identical**
  (label-canonical): 128/128 programs in NEON_DOTPROD and 64/64 in NEON. `panelStrips` is unchanged.

### Forced-width census (NON-CERTIFYING)

- Scratch probe `census/ForcedStripProbe.cpp`: explicit
  `Tile<1>::executeWith<NEONMultiScaleOps<DotProd>, S>` for S ∈ {1, 2, 4}.
- Driver `census/census_forced.py` calls the canonical module unchanged: `target_command`, the real
  target compile command plus `WITNESS_FLAGS`, `disassemble` (`/usr/bin/objdump`), `read_object`,
  `normalize_disassembly`, `noreturn_symbols` and `analyse_function`.
- Only the expected inventory is overridden: Rows1, SDOT leaf, both policies, 8 dispatch cells, S1/2/4
  = 48 programs.
- Analyzer SHA-256 `e6350f86…`, the same as the GNU census. All 48 programs are classifiable on
  AppleClang, so there are no structural failures.

Spill units, S1/S2/S4:

| Cell | Policy | Candidate | Parent |
|---|---|---|---|
| **NibbleLUT Centered** | BN | 0/0/**0** | 0/0/9 |
| **NibbleLUT Centered** | GA | 0/0/**0** | 0/0/0 |
| **NibbleLUT NonLinear** | BN | 0/0/**0** | 0/0/4 |
| **NibbleLUT NonLinear** | GA | 0/0/**0** | 0/0/0 |
| NibbleLUT Unsigned + min | BN / GA | 0/0/7, 0/0/0 | same |
| NibbleLUTSourceScale NonLinear | BN / GA | 0/0/4, 0/0/0 | same |
| ExpandedInt8 + min (Q5_K) | BN / GA | 0/0/4, 0/0/0 | same (gate-5 Q5_K BN S4 = 4 independently verified) |
| ExpandedInt8, IQ3SExpandedSourceScale | both | 0/0/0 | same |
| Q6KNativeDualScale | BN / GA | 0/0/25, 0/0/25 | same (known AppleClang spill, outside candidate; classifiable here, unlike GNU) |

**Instruction identity of the 48 forced programs, label-canonical:** 44 are identical. Exactly the 4
candidate programs changed: NibbleLUT Centered/NonLinear × BN/GA at S4. Instruction counts went
573→555, 557→552, 558→559 and 559→556.

## Verdict

AppleClang parity holds for the candidate:

- The candidate cells reach **zero spills at S4 on AppleClang-21 under both policies**. That includes
  the BN programs that spilled 9 and 4 at the parent.
- No other program changed; the installed programs are identical.
- All byte tests pass in both envelopes with 0 new and 0 gone apart from the new replay.

The census is NON-CERTIFYING. A production S4 nibble installation still needs the canonical witness
extension (plan step 3), and economy needs the GB10 tournament (gate 4).

## Artifacts

All under `/tmp/armqual-i31/`:

- `ctest-{cand,parent}-{NEON_DOTPROD,NEON}.log`
- `cases-*.txt`, `fullbin-*.log`
- `build-*.{configure,build}.log`
- `census/{ForcedStripProbe.cpp, census_forced.py, census-*-NEON_DOTPROD.{json,o}, identity-compare.txt}`
