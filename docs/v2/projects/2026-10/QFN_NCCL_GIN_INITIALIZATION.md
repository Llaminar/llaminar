# QFN native transport: NCCL GIN initialization repair

The two-CUDA counted-transfer admission fixture crashed before graph capture.
Native `ncclNetPluginInit` successfully initialized its GDAKI probe but discovered
zero compatible devices. The automatic-selection pointer remained the `-1`
sentinel, which the next `init` call dereferenced. The ordinary IB device was
present; ordinary network availability does not imply GDAKI compatibility.

The repair starts automatic selection with NCCL's existing IB proxy candidate
and replaces it only after a successful, positive GDAKI inventory. Successful
temporary probe contexts still finalize exactly once. Explicit GDAKI/proxy and
external plugin choices keep their existing semantics. No transport policy,
arithmetic, capture path or device kernel changes.

## Installed regression contract

`scripts/docker/install-nccl.sh` applies `nccl-gin-auto-selection.patch` to the
pinned native source and builds the complete selected collective inventory.
Its device-free probe includes the actual complete `plugin/net.cc` translation
unit and native types. Provider callbacks supply authored discovery outcomes.
It covers automatic available/zero/negative/error outcomes, explicit choices,
external plugins, absent GIN and warm plugin reuse: **220 replays and 2,160
assertions**. Reversing the exact repair compiles successfully and reproduces
SIGSEGV at the zero-device case.

The installer publishes a source/DSO-bound receipt. The receipt includes the
patch chain and executable oracle identity, so installer reuse cannot silently
retain old coverage. `V2_Unit_NCCLGinSelection` and
`V2_Integration_NCCLGinSelection` verify it without the native source, CUDA SDK
or a GPU. The latter is explicitly in `ProductionTestPreflight`.
`V2_Unit_NCCLGinProof` has eleven adversarial validator/packaging tests, including
stale libraries, changed sources, incomplete execution, compilation failures
masquerading as negative controls and unrelated original-code crashes.

## Focused qualification

- The frozen two-CUDA counted-transfer fixture passes with only the GIN repair
  applied to its private SDK. It exercises cold capture and repeated changing
  live extents, including empty work, with exact payload/receipt checks. Native
  inspection confirms the selected LocalP2p entry, 256 threads, 168 registers
  and zero local storage on SM86. This private SDK also contains unpublished
  transport work; it is not the production dependency inventory.
- A separate canonical installer build succeeds with the complete collective
  inventory on SM86, matching the previously installed local architecture.
  The normal installer still receives the image's complete CUDA target set.
  Device scope, retained-parent reentry and native WHILE request-reset tests
  pass against this exact full DSO. Loader observations authenticate the selected
  library; driver windows and process retirement are clean.
- The real CMake proof target builds and both new proof registrations pass
  after the installer has removed its temporary vendor source tree. Installer
  reuse verifies the same proof before taking its cache hit.
- After installing the exact tested local DSO, both complete prerequisite
  dependency targets build and **792/792 Unit tests pass**. The new explicit
  preflight regression passes in the main build. The previous complete r13
  preflight receipt remains scoped to `6f4c742e7`; the full integration gate has
  not been rerun for this repair.

Local artifacts: `/tmp/qfn-nccl-gin-selection-r5/`,
`/tmp/qfn-native-counted-peer-cuda01-gin-fixed-r5/`, and
`/tmp/qfn-nccl-gin-canonical-r{1,2}/`. Final r2 uses whitespace-clean patch
context with identical native source hashes; its rebuilt DSO passes all three
native checks and the installed proof checks. The first canonical attempt incorrectly
disabled CUDA startup for two backend-initialized fixtures; those failed before
execution. `native-capture-r2/` preserves the corrected ordinary startup run.
The first attempt's separate device-scope test passed and remains its evidence.

This repair does not complete the broader QFN admission matrix. SDK preparation
and allocation ownership, pooled RAM/SSD provider integration, composed native
graphs and complete source-admission qualification remain open.
