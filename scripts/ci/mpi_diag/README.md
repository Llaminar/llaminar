# MPI startup diagnostic and CI hook (PR75/78)

For the exact-runner investigation verified by reviewers 84/85/87. The worktree
source is e155102ef (PR head); the CI gate image artifact identifies source
merge 663fcaa83 (e155102ef into 63b053401) in its baked receipt. No gates are
changed by this diagnostic; the CI hook runs only after a host-lane failure.

## Files
- run_mpi_startup_trace.py: bounded single-fixture diagnostic (reviewer-87
  revision): per-pid+starttime+role allowlisted env VALUE|ABSENT, exact
  allowlisted keys only (non-allowlisted = names only), ptrace-free /proc
  sampling (syscall raw+name+arch, wchan, stat state, fd socket inodes ->
  /proc/net/{tcp,tcp6,unix}, maps hwloc/X/OpenCL names), ALL task/*/children
  walk, setsid/reparent escape detection, root-only canonical cancel (never
  signals descendants directly), one enforced total budget
  (window = min(registered TIMEOUT,120) + 36s grace + 25s gdb), gdb ONCE
  during the sampler the moment a stall is established (8 KiB cap, explicit
  DENIED/truncation), ENVIRONMENT_MODIFICATION honored or fail, FIXTURES_REQUIRED
  or fail, TIMEOUT_PROCESSORS/RESOURCE_LOCK recorded, original return code
  preserved (diagnostic_returncode separate from original_gate_failed).
- ci_mpi_hook.py: actual CI wiring entry (reviewer-87 item 1, reviewer-90
  revision). Called by run_develop_pr_tests.py after a FAILED host lane; reads
  the host-lane JUnit, picks ONE deterministic fixture (singleton first) from
  the failure list, runs A (registered) and B (outermost -gl,-opencl export)
  3 pairs each under ONE global <=15 min deadline, skips remaining pairs when
  the budget is exhausted, refuses an A==B contrast (no fake comparison),
  retires every recorded identity via pidfd+starttime before returning, and
  never greens. Source identity requires the baked receipt: without it the
  run is recorded UNBOUND (no git fallback qualification).
- env_policy.py: THE shared CTest env-policy parser (reviewer-92 item 2).
  Exact semantics: ENVIRONMENT NAME=VALUE; ENVIRONMENT_MODIFICATION
  NAME=set|unset|string_append|string_prepend:value. ONLY these four ops are
  supported (reviewer-93); reset/path_list_*/cmakelist are deliberately
  unsupported and fail clean with 125. FIXTURES_REQUIRED is ALWAYS unsupported
  (125) - this diagnostic never executes arbitrary setup.
  Source binding: requires --image-revision, the exact 40-lower-hex
  org.opencontainers.image.revision from `docker image inspect` (VCS_REF),
  passed by the CI launcher (run_develop_image_gate.py reads the test-runner
  image's label and injects LLAMINAR_IMAGE_REVISION into the PR-tests
  container) and forwarded by the hook; the diagnostic never scans receipt
  fields heuristically and stays UNBOUND when the label is missing,
  malformed, or unrelated.
  Log streaming: fixture output is drained DURING the run through a bounded
  pipe writer thread (16 MiB on disk + explicit marker). The pump NEVER
  breaks at the cap: it keeps reading and discarding until EOF so a
  >16 MiB fixture can still EXIT 0 (a 64 KiB pipe would otherwise deadlock
  the fixture and force a fake backstop); the on-disk bound holds DURING the
  run and a real-hook control proves the fixture finishes without backstop.
- test_mpi_diag_controls.py: 21 focused controls (all pass, ~57s):
  * DiagnosticControls (8): startup failure preserved, original exit
    preserved, timeout retirement (TERM-trapped tree -> zero survivors),
    setsid grandchild escape -> survivor FAILED + identity reaped via
    pidfd+starttime+bounded poll (zombies recorded EXITED), truncation,
    missing debugger recorded, secret key names only + value never in any
    evidence file, binding.json redaction.
  * HookControls (6): selection single/deterministic order, skip on success,
    missing JUnit records error, A==B contrast refused, shared env-policy
    parse (set/unset/reset/prepend/append + two 125 paths + fixtures 125),
    gate exit unchanged.
  * RealHookControls (2): real hook drives the real diagnostic against a tiny
    CTest dir (NO mocks): correct build dir, non-125 evidence on disk,
    source_bound false (no receipt), zero survivors; and the 16 MiB log cap
    with the on-disk marker asserted.
  * SourceIdentityControls (5): explicit --image-revision with an EXACT
    40-lower-hex value binds and names the label field
    (org.opencontainers.image.revision); missing/malformed (wrong length,
    uppercase, non-hex)/unrelated values are UNBOUND (no heuristic scan, no
    git fallback qualification).
  * Non-block happy path: the A arm that finishes normally records NO backstop
    (backstop absent from the revision), and the Unit control suite asserts
    the explicit false case (fixture exits 0, no forced cancel).
  * BackstopControls (1): REAL forced backstop (reviewer-94 structured).
    The fixture publishes a persistent child + a setsid grandchild identity
    (both TERM-ignoring) BEFORE hanging; the hook deadline (20 s) is far
    below the diagnostic's own cancel path, so the hook's run_fixture wait
    times out and the hook _retire runs. The control asserts the incremental
    owned-identities.jsonl flush exists, EVERY jsonl identity is gone or
    ZOMBIE on hook return, the structured backstop record {fired:true,
    retired:[{pid,starttime,outcome}], survivors_after:[]} is on disk (never
    a raw-kill fallback outcome), and no fixture/grandchildren survive.
  * Registered CMake tests (reviewer-94 block 3), in tests/v2/CMakeLists.txt:
      V2_Unit_MPIStartupDiagnosticPolicy         (TIMEOUT 60, seconds)
          - env_policy exact 4 ops + 125 for reset/path_list_*/cmakelist
          - selection ONE + success skip + missing JUnit
          - source binding valid/missing/malformed
          - A==B contrast refused
      V2_Integration_MPIStartupDiagnosticLifecycle (TIMEOUT 60, identity
          self-cleanup, ProductionTestPreflight)
          - REAL forced backstop structured record
          - REAL >16 MiB cap-drain fixture EXITING 0 (no backstop; pump
            discards after cap without hanging; bounded + marker on disk)

## Runtime requirements
- Linux >= 5.3 (pidfd_open/pidfd_send_signal; used for identity-pinned
  retirement and the post-open starttime re-read). OSError/AttributeError
  fallbacks are recorded as unavailable outcomes, never raw kills.
- Python >= 3.9 (os.pidfd_open, signal.pidfd_send_signal).

## Usage (reviewer-approved, runs after failed host gate on the same image/uid)
  cd /src && python3 scripts/ci/run_develop_pr_tests.py --build-dir build_v2_integration \
    --installed-build-receipt /src/installed-tests.json --output /ci-results/pr-tests
  # The hook runs automatically only when V2_Integration_MPISessionRetirement_* /
  # MPICTestSessionIsolation / MoEOwnedSharedBackingNativeStartup failed.
