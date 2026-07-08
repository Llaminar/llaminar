# Git Hooks for Llaminar

This directory contains Git hook templates for the Llaminar project.

## Pre-Commit Hook

The pre-commit hook builds the V2 integration and release configurations, then
runs the local correctness, E2E, and performance gates before allowing a commit.

### What It Does

Feature branches run this 7-step suite:

1. **Integration build** - Configures and builds `build_v2_integration` with CUDA and ROCm enabled.
2. **Unit tests** - Runs all `^V2_Unit_` tests in parallel.
3. **Focused integration tests** - Runs `^V2_Integration_GroupedVerifierRows_` to prove grouped verifier rows are bitwise serial-decode equivalent across CPU, CUDA, ROCm, and supported tensor formats/codebooks.
4. **Focused parity baseline** - Runs the model-family parity baseline through `.githooks/run_parity_baseline.sh`.
5. **Release build** - Configures and builds `build_v2_release` with CUDA and ROCm enabled for E2E tests.
6. **E2E server integration tests** - Runs CPU, CUDA, ROCm, long-context, and MoE rebalance probes.
7. **Performance regression benchmarks** - Runs `.githooks/run_benchmark_check.sh`.

On `develop` and `master`, the hook also runs the broader `^V2_Integration_`
suite after the focused grouped-verifier gate. The broader sweep excludes
`Parity`, `RCCL`, and `GroupedVerifierRows` because those have dedicated gates.

### Installation

```bash
cp .githooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Usage

The hook runs automatically before every commit:

```bash
git commit -m "Your commit message"
```

If any check fails, the commit is blocked and the hook prints the matching
manual command for the failed gate.

### Containerized E2E Mode

By default, the pre-commit E2E server suite launches the local Release
`build_v2_release/llaminar2` executable. To package the Release runtime image
and run that same E2E suite against the containerized server instead:

```bash
LLAMINAR_PRECOMMIT_E2E_CONTAINER=1 git commit -m "Your commit message"
```

Useful overrides:

```bash
LLAMINAR_PRECOMMIT_E2E_CONTAINER=1 \
LLAMINAR_E2E_CONTAINER_IMAGE=llaminar:precommit \
git commit -m "Your commit message"

LLAMINAR_PRECOMMIT_E2E_CONTAINER=1 \
LLAMINAR_PRECOMMIT_E2E_CONTAINER_BUILD=0 \
LLAMINAR_E2E_CONTAINER_IMAGE=llaminar:local \
git commit -m "Your commit message"
```

You can also run the E2E harness directly against a container:

```bash
scripts/docker/build-runtime-image.sh --tag llaminar:local --cuda-archs 86
tests/v2/e2e/server/test_server_e2e.sh \
  --container-image llaminar:local \
  --backends "cpu,cuda:0,rocm:0"
```

### Manual Commands

```bash
cmake -B build_v2_integration -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Integration -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_GroupedVerifierRows_" --output-on-failure --parallel
.githooks/run_parity_baseline.sh build_v2_integration

cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2_release --parallel
tests/v2/e2e/server/test_server_e2e.sh --binary build_v2_release/llaminar2 --backends "cpu,cuda:0,rocm:0"
.githooks/run_benchmark_check.sh
```

### Override

For WIP commits, bypass the hook explicitly:

```bash
git commit -m "WIP: debugging" --no-verify
```

Use `--no-verify` only for work in progress. PR merges must pass the gates.

### CI/CD Integration

GitHub Actions runs the same unit and focused grouped-verifier integration
commands in the builder-image job. Runtime-image jobs then run release-container
E2E and benchmark gates.

## Future Hooks

Additional hooks can be added here:

- `pre-push` - Run integration tests before pushing.
- `commit-msg` - Enforce commit message format.
- `post-merge` - Rebuild after pulling changes.

Git hooks are not installed automatically when cloning a repository, so each
developer must install them locally.
