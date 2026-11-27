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
