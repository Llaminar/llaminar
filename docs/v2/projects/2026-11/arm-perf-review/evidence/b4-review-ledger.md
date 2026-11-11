
## REVIEWER VERDICT (B4 corrected spec, ~22:48Z): NOT YET SPEC-FAITHFUL

Verified at tracker f20600403: the ORIGINAL
evidence/b4-draft-vocab-budget-design-spec.md is STILL the exact old blob
69cf952677146da42b565223b1d511b92913cdac (unchanged since de7062e2). The
new evidence/b4-design-spec-corrected.md (blob
886baa6a7f07bd68473dede271b8bb925a8d6e73) is a 61-line CORRECTIONS
SUMMARY, not the corrected full spec (single-commit history 0fefe187) —
"full corrected text preserved in history" is NOT supported.
S1/S2/S3/S6: substantively right. S4: lifetime "specified" without the
actual lifecycle. S5: rider reference "resolvable" without any reference.
The contradictory Mode C/fence wording REMAINS in the claimed canonical
spec. FIX REQUIRED: apply the normative corrections to the EXISTING
original spec and identify the exact resulting blob; no more summaries.
No new design disagreement, implementation restriction or gate added.
The B4 diagnostic d7eb73708 source review proceeds separately.

## REVIEWER VERDICT (B4 diagnostic source, ~22:52Z): HOLD at d7eb73708

Static findings (complete 1000-line diff read; no execution by reviewer;
production routing/sampler/verifier untouched):
- D1: fixture contradictions (size==8 vs 3 IDs; V=96 skips 128/256/512;
  threshold 1 not 0.5; wrong-q negative evaluates identically-p).
- D2: **Mode P biased — accept_uniform reused untransformed for the
  residual draw; counterexample p=(1/2,1/4,1/8,1/8), q=delta_0 → token1
  never emitted; realizedLaw never calls rejectionStep.**
- D3: selected arithmetic ≠ production tree (one 64-block chain vs 16
  ordered partitions × 4 blocks + fold; EXPECT_EQ not byte-compare;
  M1/unsharded V96 only; GPUAlignedExpert ≠ the measured LM-head policy).
- D4: selector lifetime not enforced (no verified receipts; merge-not-
  replace; no stale/lookahead rejection; cold-start fallback needs
  explicit admission).
- D5: Modes G/C toy copies (new hash/double-CDF vs
  sampleMTPSerialEquivalentTargetRow; G tie retains first entry not
  lowest ID).
- D6: cross-arch build regression (CMakeLists:7448ff adds the target to
  ordinary Unit on every arch while the TU #errors off AArch64).
UNPROVED remaining: counted traffic, multi-N execution, 63/65/127/129/V-1
byte sweep, reset replay, normalized-logit draw.
DISPOSITION: safely non-installed; not source-ready/unfencing basis;
repair focused mechanisms in THIS diagnostic; no new gate/apparatus.
Note: the station W3 focused check is MOOT for now — the diagnostic is
under repair; re-run after the D1-D6 fix tip.
