# Idea 23 — qwen4exp HyperConnection auto-planning classification gap

Status: **Open**. Found 2026-11-08 on the .79 Qwen3.8 speed test (engine
`arm-validation/0cdcecc3c-spills` = `bef53edb6`, Release sm_121, GB10).

## Finding

**Auto planning rejects qwen4exp HC models.** `PlanningForwardWeightWork::
planningOrdinaryWeightKind` has no case for the HyperConnection roles, so
planning fails with:

```
Unclassified request work: output_hc_down.weight
```

`WeightIdentity`/`Qwen4Exp` DO support these tensors. Workaround that works:
authored `--expert-tier` apply mode. But the normal flow (plan → serve) is
broken for these models, and it is a correctness/admission gap first, a perf
one second (blocks automatic placement of models like Qwen3.8-Flash-Next).

## Proposed fix

- Classify the HC roles (`HyperConnectionProjection`, `HyperConnectionNorm`,
  `output_hc_down.weight` etc.) in `planningOrdinaryWeightKind`, pricing them
  as projections (they are weight-transport work like other projections).
- Add the HC tensors to the same coverage that `WeightIdentity`/`Qwen4Exp`
  admit, so `plan` accepts them without authored placement.

## Validation

- `plan -m Qwen3.8-Flash-Next-UD-Q4_K_XL ...` (auto) builds an apply document
  without the unclassified-work error; the resulting plan matches the
  `--expert-tier` authored workaround for the same model; admission succeeds
  on GB10 with the joint-pool observation in idea 2.
- Device-free planning tests for the HC kinds, plus the Qwen4Exp parity/
  admission cells that already cover these tensors.

### OPERATOR DECISION (~09:05Z): idea-23 PATH (b) — fix + model in parallel; the model homes on the DGX station

- **Stage 1 dispatched to addon3**: classify the HC roles (HyperConnection-
  Projection, HyperConnectionNorm, output_hc_down.weight, etc.) in
  PlanningForwardWeightWork::planningOrdinaryWeightKind, priced as
  projections; add the HC tensors to the WeightIdentity/Qwen4Exp
  coverage; device-free planning tests + the Qwen4Exp parity/admission
  cells. Normal implement → validate → handback from the PR tip.
- **Stage 2 dispatched to the station session**: download
  Qwen3.8-Flash-Next-UD-Q4_K_XL (4-split, ~111 GB) to the STATION host
  storage (256G free; the operator designated the station as the model
  home), SHA/size-verified; report paths + disk state.
- The model-level validation (plan -m ... auto without the
  unclassified-work error, matching the --expert-tier workaround)
  waits for both.
