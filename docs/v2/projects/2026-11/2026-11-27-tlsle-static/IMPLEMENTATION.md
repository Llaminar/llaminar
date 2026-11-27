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
