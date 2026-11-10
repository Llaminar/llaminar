<!-- Reviewer #167 (2026-11-10): final-evidence composition under standing
     ruling #166. Static specification; exact combined SHA/driver/evidence
     still to be reviewed at the combined re-pin. Committed for the driver
     builder; text preserved essentially verbatim. -->

# Final-evidence composition (ruling #166)

YES to the core-free portability project and 18 required cells: GNU-14 and
AppleClang-21, each across Debug/Release/Integration ×
NEON/NEON_DOTPROD/NEON_I8MM. These certify the declared optimized witness
configurations, NOT llaminar2, complete Unit/preflight, native instruction
execution, parity or economy. Preserve the full installed probe inventory in
every cell; forced candidates are not substituted into it.

ONE ADDITION: include the EXISTING companion compiler-envelope proof and
aliases. The register checker binds CPUCodegenEnvelope.py as a helper, but
that hash is not execution of the separate ISA-envelope proof. The normal
CPUISAProfile owner already depends on both proof targets; using the two
custom targets directly keeps this campaign compile/script-only.

## Per cell

1. Establish the actual selected compiler, ISA and build configuration
   through tests/v2/portability and its normal CMake policy; retain
   configuration identity and compile database.
2. Build v2_cpu_native_tile_register_proof AND
   v2_cpu_codegen_envelope_proof.
3. Run both Unit and Integration aliases for: CPUNativeNEONTileRegisters
   (native bound register verification); CPUCodegenEnvelope (the existing
   compiler-envelope verification); CPUNativeTileRegisterEvidenceAnalysis
   and Binding0–3 (checker regressions). At the reviewed registration this
   is 14 CTest aliases per cell. Discover/reconcile the canonical
   registrations; missing aliases, skips, Not Run, zero selection or nonzero
   phase exit are failure, not reduced coverage. This remains a focused
   gate, not the full Unit/preflight suite.

## (a) Configuration / reuse

A new cell needs a successful configuration. Existing cell directories and
native proofs may be reused; deleting/reconfiguring/recompiling every
unchanged cell is NOT a requirement. The build-system configuration must
nevertheless be current for the final candidate and requested cell.
Regenerate when its CMake/configuration inputs change; never consume a stale
compile database after failed configuration. Use the existing build/verify
authorities to establish freshness, not a matching Git SHA, cached JSON, or
filename. A changed bound input requires new evidence for that cell;
unchanged bindings can be verified and reused. Native-proof reuse does not
automatically reuse checker-shard results: those must also belong to the
current test source/registration epoch.

## (b) Receipt / final verification

proof.json plus a generic CTest-success line is NOT the complete bound
receipt. Retain per cell:
- register proof.json AND proof.o, proof.dis, proof.i; retain dependency
  records as supporting build evidence;
- compiler-envelope proof.json, proof.s and its bound target.txt manifest;
- the current configured compile database/configuration, exact commands and
  actual compiler/ISA/build-type identity;
- configure/build/CTest exit statuses, complete selected-test inventory and
  results/logs, and exact candidate/policy identities.

Keep cells in distinct directories; do not overwrite or merge same-basename
proof files. The register build action already calls verify, and the native
CPUNativeNEONTileRegisters CTest aliases invoke that SAME verifier again.
Use those aliases, plus the envelope aliases, as the FINAL acceptance
verification across the retained set after its final bound inputs/policy are
fixed. If the requested CTest runs serve that final phase, NO additional
third verifier/pass is required. If earlier tests preceded later input
changes or artifact relocation, their old passes do not replace final
verification. Do not implement another spill analyser or budget comparator.

The aggregate driver still must establish exactly the complete 18-cell set,
reconcile each receipt with its declared cell and the final
candidate/policy, require every required test/phase to pass, and retain
immutable accepted artifacts or revalidate changed ones before publication.
It is a completeness/binding coordinator, not a new register authority. A
failure may publish a failure report, never a successful set receipt
harvested from old files.

Native verification is build-host/toolchain-bound: it needs the referenced
source/SDK headers, logical/resolved paths and symlinks, compiler/assembler
identities, disassembler, and configured command context. Copying JSON to
another host is not SDK-independent verification. Do not rewrite receipt
paths/hashes to make relocated evidence appear current. Retained collected
artifacts plus builder-side verification evidence may be reviewed by Main,
but that does not make the bundle self-verifying everywhere.

## (c) Debug / shared budget

Do NOT omit Debug, loosen its limits, or copy Release receipts into it. The
witness deliberately appends -O3 -fno-sanitize=all -fno-lto -std=c++20 to
the actual configured target command. Thus Debug contributes its real
configuration flags/macros to an OPTIMIZED register witness; this is not a
zero-spill claim for an ordinary -O0 Debug application. Keep the canonical
Integration defaults/validation too; an empty/stale Integration flag cache
is not an admissible cell.

Rows1 remains zero in EVERY cell. Grouped limits apply separately to every
program in every cell. Because the existing budget key lacks build type, a
tightening must use the maximum admitted count over ALL configurations
sharing that compiler/ISA/program key. Never choose the lowest count,
average cells, or offset one key with another. If limits change, final proof
must bind the final budget bytes.

No pending combined receipt has been authenticated by this specification.
The old idea-31 and B9 receipts still cannot substitute for the actual final
combination. This specifies the gate; it does not approve the new driver,
execute the campaign or grant merge admission.
