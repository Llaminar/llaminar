# Live implementation — UNVERIFIED, review required before qualification

The dated proposal and its manifest describe the immutable proposal, not the
current checker. The historical tools-v9 digest is not a current-tools digest.
Dormant imported tools have been removed; the source-owned checker is the only
authority. No production kernel, role, budget or table is changed.

## Pre-effect native inspection (2026-11-27)

Source candidate: e4add4e20b8a3f436d4ef629f6607b2b3827a720.
Retained remote root: factory@192.168.178.80:
/home/factory/pr73-arm-stack-q5/economy16/static-candidate-I8MM/.
WeightManager.cpp.o/production.o SHA256 recorded by the original diagnostic:
2e4caad3baaa10cd948e53c696be499df93c65948d0f9f3a1ed49104c16d14f0.
Native `objdump -dr` inspection before committing live TLS admission showed:

```
f4f4: d53bd040 mrs x0, tpidr_el0
f4f8: 91400000 add x0, x0, #0x0, lsl #12
       R_AARCH64_TLSLE_ADD_TPREL_HI12 _ZZN9llaminar24gemm13onednn_engineEvE15engine_instance
f4fc: 91000000 add x0, x0, #0x0
       R_AARCH64_TLSLE_ADD_TPREL_LO12_NC _ZZN9llaminar24gemm13onednn_engineEvE15engine_instance
```

WeightManager-relocations.txt records both zero addends. They are consecutive
within `_ZN4dnnl6memoryC2ERKNS0_4descERKNS_6engineEPv.constprop.0`.
Original checker rejection is retained unchanged in analysis-feasibility.json:
`ValueError: relocation type 549 is not admitted in code (.text)`.
This is inspection evidence, not a new native qualification or T-a PASS.

The live TLS slice is committed UNVERIFIED before tests, per owner direction.
Review is mandatory before native controls or using it to admit Q5. Static
transaction authority is not implemented by this TLS slice; static default/weak
binding remains rejecting. Auto-merge remains OFF; parent owns pushes.

## Post-push developer validation

Parent confirmed all three commits pushed before verification: a618058a2
(relocation), f51fc475b (dormant tools retirement), e3164760e (live TLS).

- `python3 -B tests/v2/unit/cmake/test_cpu_native_tile_register_evidence.py TLSLocalExecPairTest TLSDescriptorTest CallModelTest CallProvenanceTest RelocationTest -v`: 37 passed.
- Real `cmake -S tests/v2/portability -B /tmp/pr73-tlsle-portability -G Ninja -DCMAKE_BUILD_TYPE=Release` with resolved Homebrew Ninja and explicit Homebrew OpenMP flags/library/include configured successfully (AppleClang21, native NEON). The redundant `CPU_INSTRUCTION_SET` input was unused; native architecture detection selected NEON.
- `ctest --test-dir /tmp/pr73-tlsle-portability -N -R '^V2_Integration_CPUNativeTileTLSLocalExecPair$'`: one registered test.
- Same CTest selection with `--output-on-failure`: passed; ProductionTestPreflight label present.
- `git diff --check`: passed.

These are developer checks only, not the complete Unit/preflight gate, native
production T-a, static verifier coverage or Q5 qualification. The approved static
acquisition scope and observed provenance gaps are in STATIC-ACQUISITION.md;
no static authority has been implemented or minted.

## Bounded native parser attempt — T-a BLOCKED, not PASS

After developer validation, parent requested the small native parser control.
Copied the actual object to `/tmp/pr73-tlsle-native-inspection/WeightManager.cpp.o`
and retained remote GNU `objdump -dr` output as `WeightManager.objdump` there.
Recomputed object SHA256 equals the recorded 2e4caad3... digest above. Used the
original checker extracted with
`git show f44e67e9a:tests/v2/cmake/CPUNativeTileRegisterEvidence.py` and the live
checker separately. Native command:

```
python3 -B /tmp/pr73-tlsle-native-inspection/check_pair.py
```

The original checker rejects the **actual complete object** at relocation549:
`relocation type 549 is not admitted in code (.text)` (T-b reproduced).
The live checker advances beyond549, then **fails closed at relocation541**:

```
relocation type 541 is not admitted in code (.text._ZZNSt9once_flag18_Prepare_executionC4IZSt9call_onceIMNSt13__future_base13_State_baseV2EFvPSt8functionIFSt10unique_ptrINS3_12_Result_baseENS7_8_DeleterEEvEEPbEJPS4_SC_SD_EEvRS_OT_DpOT0_EUlvE_EERSI_ENUlvE_4_FUNEv)
```

`readelf -Wr` identifies this as R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 at0,
paired with type542 R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC at4, targeting
`_ZSt15__once_callable` with zero addends. These are **not** the approved
local-exec grammar. No new admission, slicing, relocation removal, fake
function image or parser bypass was used. Normalization and pair extraction
were never reached; T-a remains BLOCKED despite the conforming549/551 native
instruction inspection. Native parser command exits1; do not report it green.

The first scratch invocation failed because the extracted original checker
lacked sibling CPUCodegenEnvelope.py; copied complete current sibling Python
support to scratch and reran. Subsequent runs reproduced the native failure
above. The final combined stdout/stderr is retained as `result.txt` in the
scratch directory. No scratch artifacts are staged or committed.

Additional retained remote file SHA256:

- WeightManager-relocations.txt: 04e9961f80e4f595a43dd01d4cb2ccc9f57acbcc66d2d2b4841e0c4dfbdbd0db
- analysis-feasibility.json: b74bfc0d1698853db0581b4847df04b23d15f9fb46d24b2c205dc780f0da541c

Static acquisition implementation and any TLSIE extension need their own
reviewed slices. The present work closes neither native whole-object admission
nor Q5 qualification.
