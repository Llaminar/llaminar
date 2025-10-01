## Llaminar Test Agent & MPI Utilities Guide

This document explains how to write reliable, deadlock‑resistant tests for the Llaminar
inference engine using the shared utilities under `tests/`. It focuses on:

1. Canonical MPI initialization / finalization
2. Watchdog timeouts & hang diagnostics
3. Rank / world helpers and root‑only sections
4. Patterns for distributed numeric correctness tests
5. Environment flag scoping and cleanup

---

### 1. MPI Environment (`test_mpi_utils.h`)

Include the header in any test that touches distributed code:

```cpp
#include "test_mpi_utils.h"
```

Add the canonical main at the bottom of the translation unit **once per test binary**:

```cpp
LLAMINAR_DEFINE_GTEST_MPI_MAIN();
```

This guarantees:
- Idempotent `MPI_Init_thread` with `MPI_THREAD_FUNNELED`
- A world barrier before `MPI_Finalize()` (reduces OpenMPI noisy warnings)
- Automatic cleanup if a test aborts early (RAII at‑exit hook)

Available helpers (namespace `llaminar::test_util::MPIEnvironment`):

```cpp
int rank();              // MPI rank (0-based)
int world();             // world size
bool is_root();          // rank()==0 convenience
void barrier();          // MPI_Barrier(MPI_COMM_WORLD)
template<class Fn> void root_only(Fn&& fn); // executes fn on rank 0 only
void skip_unless_world(int expected);       // GTEST_SKIP if world size mismatch
void skip_unless_world_at_least(int min);   // GTEST_SKIP if world smaller
```

Example usage in a fixture:

```cpp
class DistFixture : public ::testing::Test {
  void SetUp() override {
    using Env = llaminar::test_util::MPIEnvironment;
    world_ = Env::world();
    rank_  = Env::rank();
  }
  int world_ = 1;
  int rank_  = 0;
};
```

If you need a custom `main()` (rare), call:

```cpp
MPIEnvironment::init(&argc, &argv);
// RUN_ALL_TESTS
MPIEnvironment::finalize();
```

---

### 2. Timeout / Hang Watchdog (`test_timeout_guard.h`)

Long‑running or collective‑heavy tests should guard against silent hangs:

```cpp
#include "test_timeout_guard.h"
using llaminar::test_util::TestTimeoutGuard;

TEST(DistributedOp, CompletesUnderBudget) {
  TestTimeoutGuard guard("DistributedOp", 
      TestTimeoutGuard::ResolveTimeout({"LLAMINAR_TEST_TIMEOUT_MS"}, std::chrono::milliseconds(60000)));
  // ... test body ...
}
```

If the timeout elapses you get:
- Rank & world size
- Elapsed ms vs budget
- Symbolized stack trace (where supported)
- Process abort (fast feedback in CI)

Environment override example:

```bash
export LLAMINAR_TEST_TIMEOUT_MS=15000
```

---

### 3. Root‑Only Assertions & Reference Paths

To avoid redundant heavy reference computations:

```cpp
auto &Env = llaminar::test_util::MPIEnvironment;
std::vector<float> reference;
if (Env::is_root()) {
  reference = build_reference();
}

// Broadcast reference size then data if needed.
size_t sz = reference.size();
MPI_Bcast(&sz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
reference.resize(sz);
MPI_Bcast(reference.data(), (int)sz, MPI_FLOAT, 0, MPI_COMM_WORLD);

if (Env::is_root()) {
  EXPECT_LT(rel_l2(out, reference), 1e-5) << "drift detected";
}
```

When only rank 0 asserts, failures are still reported normally (other ranks exit after barrier/finalize).

---

### 4. Mixed Zero‑Tile & Replicated Fallback Testing

Some distributed strategies yield a "mixed zero‑tile" distribution (one or more ranks hold no tiles).
The fused RMSNorm+QKV path now auto‑falls back to a replicated host implementation in that case.

Testing pattern:

```cpp
auto fused = mgr.fused_rmsnorm_qkv(...);
bool ok_norm = fused.normalized.host_owned || fused.normalized.original_row_major;
if (!ok_norm) {
  GTEST_SKIP() << "Fallback not materialized"; // defensive skip rather than crash
}
```

Add counters (already exported): `mixed_zero_tile_fallbacks` for telemetry.

---

### 5. Environment Flag Hygiene

Use the built‑in RAII helper `MPIEnvironment::ScopedEnvVar` to apply and automatically
restore environment variable overrides:

```cpp
using Env = llaminar::test_util::MPIEnvironment;

TEST(PathSelect, ForcesCOSMA) {
  Env::ScopedEnvVar force_cosma("LLAMINAR_COSMA_FORCE", "1");
  // test body with COSMA forced
}

TEST(DisableLogTemporarily, UnsetLogLevel) {
  // Temporarily unset an existing variable
  Env::ScopedEnvVar unset_log("LLAMINAR_COSMA_LOG_LEVEL");
  // ... operations with variable removed ...
}
```

Key behaviors:
- Previous value (or absence) is restored on destruction.
- Passing nullptr (or omitting the second argument) unsets the variable for the scope.
- Move operations transfer ownership; copying is disabled to prevent double restores.

---

### 6. Choosing Sequence Lengths & Dimensions

For small sequence lengths + multi‑rank runs you are more likely to trigger zero‑tile distributions.
If your test specifically wants distributed ownership across all ranks, increase `seq_len` until each
rank reports a non‑zero local tile (see logs from `CosmaPrefillManager`). Conversely, to explicitly
exercise fallback paths choose minimal lengths (e.g. 16–64) known to produce mixed layouts.

---

### 7. Pattern Reference: Fused RMSNorm+QKV Test

See `test_cosma_fused_rmsnorm_qkv.cpp` for a complete example combining:
- MPI main macro
- Root‑only assertions
- Replicated fallback tolerance
- Deterministic random seeds

---

### 8. Adding a New Distributed Test (Checklist)

1. `#include "test_mpi_utils.h"` (and `test_timeout_guard.h` if needed)
2. Define your fixture (cache rank/world in `SetUp()` if you use them repeatedly)
3. (Optional) Install `TestTimeoutGuard` first in the test body
4. Set deterministic seeds for reproducibility
5. Materialize / broadcast any reference data once (root)
6. Execute distributed path
7. Reconstruct row‑major outputs (if needed) via manager utilities
8. Root rank: perform numeric assertions (rel L2, max abs)
9. `LLAMINAR_DEFINE_GTEST_MPI_MAIN();` at file end

---

### 9. Troubleshooting

| Symptom | Likely Cause | Mitigation |
|---------|--------------|------------|
| OpenMPI warning about missing finalize | Missing barrier/finalize ordering | Use macro main or call `MPIEnvironment::finalize()` explicitly |
| Hang during collective | Mismatched ranks entering collective | Add `TestTimeoutGuard`, insert diagnostic barriers, ensure all ranks pass conditionals |
| Mixed zero‑tile fallback unexpected | Sequence length too small | Increase seq_len or adjust distribution thresholds |
| Sporadic numeric drift | Non-deterministic seed use | Fix seed (`std::mt19937 gen(42)`) across ranks + broadcast weights |

---

### 10. Future Extensions

Planned shared utilities (not yet implemented):
- Scoped environment variable helper (if adoption grows)
- Reference broadcast wrappers
- Automatic per‑rank log prefix injection

Contributions welcome—follow existing patterns in `test_mpi_utils.h` and submit a PR.

---

Maintainers: Keep this guide updated whenever test infrastructure changes (e.g. new counters,
new fallback modes, or new distributed kernels).

---

### 11. Case Study: Resolving a Persistent Hang in `test_attention_shard_parity`

We encountered a hard hang (no stdout/stderr, Ctrl+C ineffective until forced) while developing the `test_attention_shard_parity` harness. Root cause narrowed to interaction between:

1. Linking both a custom `main()` and `gtest_main` (duplicate entry / conflicting startup order)
2. Static initialization inside shared test utilities (`test_mpi_utils.h`) touching MPI state before `MPI_Init_thread`
3. Invoking weight sharding logic prior to confirmed MPI init (amplifying undefined ordering)

#### Symptoms
* `mpirun -np 2 ./build/test_attention_shard_parity` produced zero expected early logs.
* Single-process run also produced no early main prints.
* Strace showed normal dynamic loader activity, then apparent stall before our user-space diagnostics.

#### Investigation Timeline
| Step | Action | Observation |
|------|--------|-------------|
| 1 | Added static ctor probe printing before `main()` | No output → stall before probe executed or output suppressed |
| 2 | Replaced macro main with explicit custom main | Hang persisted |
| 3 | Removed `gtest_main` from linkage | Hang persisted |
| 4 | Created `minimal_mpi_probe` binary | Probe ran fine (MPI itself OK) |
| 5 | Strace on hanging binary | Loaded libs; no forward progress into our code |
| 6 | Rewrote test w/out any GTest/utilities | Hang disappeared |

#### Implemented Fix
Replaced the test with a standalone minimal MPI program:
* No GTest, no `test_mpi_utils.h`, no watchdog utilities
* Explicit `main()` performs: early print → `MPI_Init_thread` → build tensors + run kernel → aggregate + compare → `MPI_Finalize`
* Linked only core library + MPI; excluded `gtest_main`

#### Why It Worked
Eliminating layered static initialization removed hidden ordering dependencies and duplicate main symbol ambiguity. MPI now initializes deterministically before any higher-level constructs run.

#### Guidelines to Prevent Recurrence
1. If writing a custom main, never link against `gtest_main`
2. When a hang occurs pre-output:
  * Add a static ctor probe printing to `stderr`
  * Spin up a minimal probe binary (MPI init/finalize only)
  * Strip test down to raw MPI + simplest reproduction, then reintroduce utilities incrementally
3. Avoid indirect MPI calls from static/global objects in test files
4. Prefer first validating new distributed logic in a minimal harness before integrating shared utilities

#### Parity Harness Enhancements (Post-Fix)
After stabilization we added:
* Deterministic weight init + per-rank head sharding
* Rank 0 naive reference: QKV matmuls → RoPE → causal masked softmax → output projection
* Distributed partial sum reconstruction exploiting row-partitioned `Wo`
* Parity metrics (max_abs, rel_l2) with tight thresholds (1e-5) achieving ~1e-9 max_abs

#### Future Hardening Ideas
* CI stage running distributed tests once in "raw" mode (no utilities)
* Optional build flag to instrument static initializers with timestamp + rank
* Environment toggle to disable utilities (`LLAMINAR_NO_TEST_UTILS`) for isolation A/B

Use this case study as a playbook for future unexplained pre-main hangs.

---

### 12. Lesson Learned: Stale Test Binary + Misaligned GGUF Data Region (Float Tensor Parity Regression)

While adding IQ quant formats we hit a dramatic failure in the golden loader test:

* All F32 tensors (layer norms, output norm, biases) showed tiny denormal values (~5–20e-39)
* Relative L2 errors exploded (1e+36 – 1e+39) vs llama.cpp reference
* Quantized tensors appeared structurally fine, masking the root cause

#### Root Cause
Two interacting issues:
1. The loader previously recorded the raw file position immediately after parsing tensor metadata, without aligning to the GGUF-mandated 32-byte boundary before the tensor data blob. Reads began a few bytes early inside padding / preceding quant payload bytes, so F32 tensors deserialized junk that still formed valid (but meaningless) subnormal floats.
2. The golden test executable had not been relinked after updating `model_loader.cpp`; incremental build rebuilt `libllaminar_core.a` only. CTest kept executing an older binary lacking the alignment fix, prolonging the investigation.

#### Diagnostic Signals That Solved It
| Signal | Insight |
|--------|---------|
| Added first-10 tensor dump (name, offset, size, type) | Confirmed offset chain looked internally consistent but needed upstream comparison |
| Upstream gguf comparison block (mismatches=0) | Proved header + tensor metadata parsing was correct (enum realignment OK) |
| Logging aligned vs raw `data_offset` (raw_pos=... aligned_pos=...) | Showed a 3-byte padding adjustment was required (alignment=32) |
| Direct test binary run (bypassing CTest) after full rebuild | Passed immediately → stale binary suspicion confirmed |

#### Fix
* Seek to the next 32-byte boundary after metadata (`(pos + align - 1)/align*align`) and use that as `data_offset` (optionally recompute if first tensor offset is non-zero)
* Force full rebuild (not just target library) so test executables relink with updated loader

#### Preventive Invariants Added / Recommended
* Assert (in Debug) that: first tensor offset == 0; each subsequent offset equals previous offset + padded(previous_size)
* Log alignment decision only when `LLAMINAR_MODEL_LOAD_DEBUG` is set
* (Planned) Lightweight regression test verifying alignment + offset chain for a small known model

#### Takeaways
1. Always rebuild dependent test binaries after changing a static library with parsing logic
2. Add a structural comparison (offsets/types) against a known-good upstream parser early—this isolates logic vs data issues
3. Denormal floods across all floats usually mean misaligned base pointer or element-size mismatch, not numeric drift
4. Gate verbose diagnostics behind an env var to allow rapid deep dives without polluting normal test output

Use this pattern for future mysterious “all floats are tiny” failures in loaders or deserializers.

