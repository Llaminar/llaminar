# Git hooks for Llaminar

The tracked hooks in this directory are used directly through Git's
`core.hooksPath`; do not install separate copies in `.git/hooks`.

## Pre-commit

Every branch runs exactly the same two suites, in order:

1. The full `^V2_Unit_` CTest namespace.
2. The full `^ProductionParityPreflight$` CTest label.

The hook configures `build_v2_integration` with CUDA/ROCm enabled and the active
Ninja executable pinned in `CMAKE_MAKE_PROGRAM`. It builds only `v2_unit_gate`
and `v2_production_parity_preflight_gate`, whose dependencies derive from the
canonical CMake test registrations. Script-only tests need no executable build;
shared fixtures are built once. Build and test parallelism is unrestricted,
with CTest retaining the registered resource locks and timeouts.

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
  --target v2_unit_gate v2_production_parity_preflight_gate
ctest --test-dir build_v2_integration --output-on-failure --parallel \
  --no-tests=error -R '^V2_Unit_'
ctest --test-dir build_v2_integration --output-on-failure --parallel \
  --no-tests=error -L '^ProductionParityPreflight$'
```

To bypass pre-commit explicitly for a WIP checkpoint:

```bash
git commit --no-verify -m 'WIP: checkpoint'
```

Hook command selection and failure propagation are covered by
`V2_Unit_PreCommitHook`, using command recorders without running real builds,
models, accelerators, or modifying Git registration.
