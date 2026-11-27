# ARM physical row-program routing — implementation checkpoint

UNQUALIFIED: source checkpoints precede tests by owner instruction. Byte and
stack acceptance require the final frozen source, not a prior passing build.
No speed claim is made; payload rereads and slower-than-develop execution are
accepted for this repair. Partition-carry/shared-decode optimization remains a
separate follow-up, not discarded or implemented here. Retain economy16 and
all prior native evidence.

## Single routing authority

`kNEONRowDecompositions` contains 19 exact source keys. Its explicit policy,
source family, encoding, map, minimum, pair family and tail family select the
existing physical programs at compile time. No runtime retry or fallback is
possible. Rows 1/2 select one same-row alternate; rows 3/4 consume exact spans
[0,2) and [2,Rows), preserving each row's K partitions, scale program and FP
fold order. Slices call exact `run`/`runCoded` leaves without redispatch.

Ordered source keys (rows/family; family A=ASIMD, D=SDOT, I=I8MM):

| Source | Format | First leaf | Tail leaf | Optimization follow-up |
|---|---|---|---|---|
| 3/A | IQ3_S | 2/A | 1/A | recover shared decode across three rows |
| 3/A | IQ4_XS | 2/A | 1/A | recover shared source-scale decode |
| 4/A | IQ3_S, Q5_K, Q4_K | 2/A | 2/A | recover four-row payload reuse |
| 3/D | IQ3_S | 2/D | 1/A | recover shared decode without spilling SDOT1 |
| 3/D | Q5_K, IQ4_XS, Q4_K | 2/D | 1/D | recover three-row payload reuse |
| 4/D | IQ3_S, Q5_K, Q4_K | 2/D | 2/D | recover four-row payload reuse |
| 4/I | IQ3_S | 2/D | 2/D | economical exact SMMLA partition carry |
| 2/I | IQ3_S | 2/D | none | economical exact SMMLA partition carry |
| 1/D | IQ3_S, Q5_0, Q5_1 | 1/A | none | recover admitted exact SDOT scale program |

BackendNative additionally routes only 1/D Q5_0 and Q5_1 to 1/A. This
cross-ISA arithmetic change is **not qualified** until compared against the
retired SDOT program including NaN/Inf/zero inputs. All other unlisted keys
retain their original physical program. Source keys are rejected as targets
(no routing chains). Expected retired source counts: NEON 5, DOTPROD 17,
I8MM 19. The final replacement bytes and stack traffic must be measured anew.

## Proof inventory and numerical contracts

The checker parses the C++ table and enum values; a native CTest compares it
against the compiled constexpr table dump. The probe expects exactly the
actual-dispatch leaf union, rejects missing/extra/retired programs, and restricts
each dispatcher to its own exact replacements. Production chain evidence also
rejects retired source symbols. The test-only wide oracle is confined to the
numerical executable, never the production probe or runtime.

Only retired exact-key entries are removed from GCC attributes, register-budget
inventory and ordered-add/chain counts. Every live ceiling and count remains
unchanged. The new role-map entry is an unchanged-line remap of the previous
source-bound entry; historical maps and the develop StackTable remain immutable.
Do not waive admission or traffic accounting if an exact replacement regresses.

Focused preflight compares original and installed programs on both policies,
all affected source formats, rows 1..4 as applicable, N tails/chunks and K
partition boundaries, dirty repeated execution and guard bytes. Four input
phases cover finite data, exceptional activation scales, exceptional weight
scales and both: distinct signed NaN payloads (including signaling), infinities
and signed zeros occur at different K positions. Full production all-format,
K-tile/carry and actual-TU stack/admission gates remain mandatory.

## Evidence status

- Initial redispatch checkpoint 97acc3bbb: GNU14 I8MM Integration focused
  wide-vs-installed regression passed (51 ms); superseded, not final proof.
- All-19 checkpoint 26d511a68: build failed before numerical execution because
  a nondependent two-row private enum alias forced premature class instantiation.
  The subsequent checkpoint makes that alias dependent; no arithmetic changed.
- Remote evidence roots: `~/pr73-arm-stack-q5/runs/rowpairs97` and `rowpairs26`.
  Later frozen runs must use new paths, never overwrite these records.
