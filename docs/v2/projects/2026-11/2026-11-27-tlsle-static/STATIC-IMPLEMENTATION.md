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
