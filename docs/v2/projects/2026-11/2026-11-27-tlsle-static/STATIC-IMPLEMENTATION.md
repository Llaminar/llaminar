# Bounded static acquisition implementation — review required

The existing CPUNativeTileRegisterEvidence.py is the sole analyzer and authority.
The new API takes an exact ordered GNU14 Ninja driver edge, its working directory,
the existing production executable, and a dedicated artifact directory. Existing
link.json diagnostic records are not accepted as authority. No Q5 qualification,
kernel, budget, role table or TLS behavior is changed.

Capture retains the production bytes first, driver -### jobs and collect2 -debug
actual linker resolution, linker identity/version, default script for the actual
emulation, full environment, trace/dependency inventories and relink artifacts.
Debug, instrumented, control and map executables must equal production. Instrument
adds only trace/dependency options and output replacement; control changes only
output; map additionally adds -Map. Failed capture removes the previous recipe.
Recipes are diagnostic only; selected-object proof must be reconstructed natively.

The typed authority additionally reads actual AArch64 executable/object bytes,
checks PIE identity and dynsym absence, unique weak/default final symbol, kept map
interval, archive member ordinal/header offset and exact Unit/B1 object identity.
The CallModel consumes the ephemeral authority, not a serialized enum or digest.
The map grammar is bounded; unsupported inputs fail rather than guessing.

## Commands and qualification limits

Compiler-free preflight boundaries:

```
python3 -B tests/v2/unit/cmake/test_cpu_native_tile_static_selection.py StaticSelectionBoundaryTest
```

Explicit native integration (Linux GNU14 driver, bfd linker, binutils; no skip):

```
LLAMINAR_STATIC_TEST_GXX=/usr/bin/g++-14 python3 -B tests/v2/unit/cmake/test_cpu_native_tile_static_selection.py StaticSelectionNativeTest
```

The native test is always registered outside preflight and fails if GNU14 is
unavailable. Host acquisition alone does not qualify the AArch64 selected-object
boundary. Reviewer acceptance and actual AArch64 native adversarial controls are
mandatory before Q5 use. Required negative inventory from STATIC-ACQUISITION.md
remains the qualification contract; these initial tests do not claim to complete
that inventory. Owner pushes each unverified slice before verification.

Exactly one existing -Wl,--dependency-file=PATH is required. The production
executable and dependency file are snapshotted and rehashed unchanged. Relinks
replace only the output and dependency PATH values; instrument/map additionally
add their exact diagnostic options. Original and every redirected dependency
inventory must agree. No production side output is overwritten, restored or
deleted. Authority is single-process and bound to the captured environment.

## Native validation receipt — 2026-11-27

Pushed source 1b4130348bf95a30d4f4ae897966ec2562d57bad passed all three explicit
native tests on factory@192.168.178.80, AArch64, image
`pr73-arm-stack-repair:gnu14` (image ID `005a823c10bf`): 3 tests, 4.734 seconds.
Command (source mirror contains the existing analyzer and its dependencies):

```
docker run --rm --init --network none \
  -e LLAMINAR_STATIC_TEST_ARTIFACTS=/proof/native \
  -v /home/factory/pr73-static-selection-worker:/proof -w /proof \
  pr73-arm-stack-repair:gnu14 python3 -B \
  tests/v2/unit/cmake/test_cpu_native_tile_static_selection.py StaticSelectionNativeTest
```

Retained under `/home/factory/pr73-static-selection-worker/native/`:

- `static-comdat-x_ebplpg`: real competing template COMDAT archive objects,
  normalized selected production object, exact Unit/B1 byte identity, native
  verifier and CallModel authority success, original checker rejection; retained
  wrong-winner/discarded-only/stale map/ELF, symbol extent/binding, PIE/interpreter,
  duplicate-member, mutated-member, dynsym, replay and typed-marker mutations.
- `static-closure-dhl76e89`: actual shadow CRT/plugin/collect2 and selected libm.so
  script-stub closure; native input-byte, response, script/default-script artifact,
  environment/argv, control/map ELF, trace/dependency and original-dependency
  mutations all reject at their asserted boundaries. No installed SDK modified.
- `static-native-lbgckdwz`: independent bounded acquisition, original-dependency
  token cardinality and rewriting-option controls.

Earlier `static-closure-aqnbh_it` retains the literal native observation that
explicit `-T` scripts appear in dependencies but not `--trace`; this is rejected,
never repaired by unioning inventories. The current explicit-script regression
rejects at the earlier reviewed unsupported-option boundary.

The composed existing checker suite passed 239/239 in 121.218 seconds after c664
(including the canonical cache fix and actual typed TLS marker tests), local log
`/tmp/pr73-static-composed-full.log`. This is tool validation, not Q5 evidence.

## Actual production continuation blocker

The retained real candidate edge in
`/home/factory/pr73-arm-stack-q5/economy16/static-candidate-I8MM/final-link.txt`
contains exactly one dependency-output option and no explicit linker script.
Its production dependency output
`numtree/build/CMakeFiles/llaminar2.dir/link.d` is absent. This is ordinary Ninja
`deps=gcc` behavior: Ninja consumes the linker depfile into `.ninja_deps`.
Read-only observation in `economy16/static-ninja-observation/deps.txt` reports
exact target `llaminar2`, 43 dependencies, VALID, mtime1795780850938300935.

The reviewed explicit native-Ninja admission requires that link cwd be the Ninja
build root owning `.ninja_deps`; a mismatched cwd or missing/stale/foreign/empty
record fails closed. Supply the actual Ninja executable explicitly only when the
original depfile is absent. Its realpath/ELF/version/hash, native database hash,
exact-target query/environment/raw record and output mtime are bound. Every
redirected native relink depfile must match that record's input set; database,
tool, output bytes and mtime must remain unchanged. No synthetic inventory,
in-place production relink, automatic alternative or depfile recreation occurs.

No WeightManager/Q5 admission, budget/table change, native traffic/spill result,
no-offset economy verdict or baseline58d77 comparison is certified here.
