# Git hooks for Llaminar

The tracked hooks in this directory are used directly through Git's
`core.hooksPath`; do not install separate copies in `.git/hooks`.

## Pre-commit

Every branch requires the same two complete suites, in order:

1. The full `^V2_Unit_` CTest namespace.
2. The full `^ProductionTestPreflight$` CTest label.

The hook configures `build_v2_integration` with CUDA/ROCm enabled and the active
Ninja executable pinned in `CMAKE_MAKE_PROGRAM`. It explicitly regenerates
`compile_commands.json` for the SDK and compiler-flag audits, including after a
toolchain change resets cached settings. It builds only `v2_unit_gate`
and `v2_production_test_preflight_gate`, whose dependencies derive from the
canonical CMake test registrations. Script-only tests need no executable build;
shared fixtures are built once. Build and test parallelism is unrestricted,
with CTest retaining the registered resource locks and timeouts.

Set `LLAMINAR_PRECOMMIT_BUILD_DIR` to select an existing Integration build
without losing its pinned compiler and dependency configuration. The default
remains `build_v2_integration`. After a complete canonical prerequisite run,
`LLAMINAR_PRECOMMIT_PREREQUISITES` may name its `prerequisites.json` receipt.
The hook still configures and builds both gate targets, then uses the shared
validator to authenticate the unchanged build and complete test inventory.
A rebuilt target, changed registration, failed receipt or mismatched build
blocks the commit. Source-file and directory timestamps also reject later
changes to interpreted tests and policies that do not trigger compilation.
Explicit receipt reuse never starts another test mode or
refreshes the old evidence timestamp.

Configuration, compilation, missing tests, or test failures block the commit
immediately and leave the underlying diagnostics visible. The hook does not
run numerical model-parity campaigns, broader integration selections, E2E,
container builds, Release builds, or performance benchmarks. There are no
branch exceptions or environment switches that add those gates. Their separate
manual/CI entry points remain available; passing pre-commit is not model-parity,
HTTP, or performance certification.

## Register or re-register

Run from the repository root:

```bash
chmod +x .githooks/pre-commit
git config --local core.hooksPath .githooks
git config --show-origin --get core.hooksPath
```

This is idempotent and also retains the tracked Git LFS hooks. Old copied
hooks under `.git/hooks` are not used while this path is selected.

## Run manually

Exercise Git's registered hook without creating a commit:

```bash
git hook run pre-commit
```

After configuring Integration, the equivalent build/test commands are:

```bash
cmake --build build_v2_integration --parallel \
  --target v2_unit_gate v2_production_test_preflight_gate
ctest --test-dir build_v2_integration --output-on-failure --parallel \
  --no-tests=error -R '^V2_Unit_'
ctest --test-dir build_v2_integration --output-on-failure --parallel \
  --no-tests=error -L '^ProductionTestPreflight$'
```

To bypass pre-commit explicitly for a WIP checkpoint:

```bash
git commit --no-verify -m 'WIP: checkpoint'
```

Hook command selection and failure propagation are covered by
`V2_Unit_PreCommitHook`, using command recorders without running real builds,
models, accelerators, or modifying Git registration. Compiler-metadata
regeneration, build selection and receipt freshness are also registered in
`ProductionTestPreflight` as `V2_Integration_PreCommitCompilerMetadata`,
`V2_Integration_PreCommitSelectedBuild` and
`V2_Integration_ProductionPrerequisiteReuse`.
