# Exact static compiler-alias authority — review and qualification boundary

The sole analyzer now accepts an optional `StaticB1Replay` through
`verify_static_executable_selection(..., b1=replay)`. This is not a name allowlist
or a serialized authority. Without B1, overlapping functions still fail.

The native transaction revalidates the driver-observed plan with the existing
`JobPaths` grammar, source dependency closure and compiler/assembler identities,
then uses `replay_plan` to retain the assembler input. The entire replayed object
must equal the audited object and ultimately the map-selected object/archive
member. Retained `production.s`, plan, source closure and programs enter the
selection's checked files before assignment parsing. Scratch B1 paths must not
name production outputs. Unsupported production compiler options fail closed.

Object authority requires exactly two unique weak/default FUNC names at the same
input section/value/size. One full-line directed `.set alias,body` or `alias=body`
is **corroboration**, not independent semantic authority: whole-object equality
cannot distinguish `.set` from two adjacent labels. Same-input-section extent
identity, not assembly spelling or identical final bytes, excludes cross-section
linker ICF. Unsupported section switching (`.pushsection`, `.popsection`,
`.previous`, `.subsection`) fails closed; this implementation does not infer its
state. No claim is made to distinguish every source spelling with equal objects.

Both final names must be unique weak/default FUNC, absent from dynsym, at the
same final section/address/size, without another function/IFUNC overlap. Existing
kept-section/map/native closure and archive-member requirements remain intact.
One typed selection owns the pair. CallModel registers both spellings to that
same authority and resolves both to the assignment's body name **before** helper
summary caching and recursion detection. Analysis is reused per body/context;
each executed call retains its actual call-site traffic (not zero or twice).

Focused controls extend the existing static boundary and GNU14 native test
classes. Their registrations are unchanged: compiler-free boundaries are already
in Unit and ProductionTestPreflight; native acquisition is the explicit
AArch64/GNU14 integration lane, not an unqualified host preflight substitute.
Neither passing fixture controls nor reviewed code qualifies Q5. Actual production
alias B1/native admission and the separate line-witness/numerical/economy gates
remain necessary. No budget, table, kernel or Q5 verdict changes here.

## Checked controls and actual production boundary

Pinned native host: `factory@192.168.178.80`, image
`pr73-arm-stack-repair:gnu14`, immutable image ID
`005a823c10bf26cf666b13bfa86be8437553c7682ac70ebef986fc3bd5aee7b5`.
After parent publication, local boundaries passed 13/13. The combined boundary
and full native classes passed 18/18 in 8.868 seconds. Retained evidence root:
`/home/factory/pr73-alias-23156/native/`, including
`static-alias-m8cka7_r`, `static-comdat-x7a7hzvm`,
`static-closure-4l_t2gy7`, `static-native-s5h7uoul`, `static-ninja-vhkbk07v`.
A final scratch-output isolation guard and after-replay mutation control were
added after this run; their subsequent validation is reported separately.

The read-only actual production command diagnostic is retained there as
`actual-production-boundary.json` (driver query only, **no authority**).
Its precise first rejection is:
`job value -MT CMakeFiles/llaminar2_core.dir/loaders/WeightManager.cpp.o not admitted`.
The executable, Ninja database and selected WeightManager object retain the
three exact SHA256 anchors in STATIC-ALIASES-PROPOSAL.md. No production output
was written and no actual Q5 alias was admitted. This is not a `.set` failure.

The independently owned production-plan lane must supply a reviewed typed
production compile authority before actual alias continuation: exact configured
launcher/driver edge, source/compile-database/Ninja identity, complete source
closure, actual dialect/target, selected compiler/assembler identities, retained
assembler input and whole selected-object identity. Adapt this B1 seam to that
shared authority after review; do not duplicate its grammar, substitute C++20,
rewrite the dependency target to `probe`, or treat old b1.json as authority.
The actual target remains the exact ISA1 lambda from the proposal, final
`0x4d6f70`, size `0x110`; the second reached lambda needs its own proof.
