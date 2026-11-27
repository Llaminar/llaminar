# 2026-11-27 — REVIEW ONLY: TLSLE proposal / static authority design

**UNVERIFIED. Do not apply to the live checker or use for Q5 qualification.**
Auto-merge remains OFF; this task changes no GitHub settings. Owner owns push.

## Placement and scope

Authenticated tools-v9 were imported once at `ce7806d5e79bf5a756ed719a426c77761d92e2c5`.
Their historical manifest remains immutable. The actual relocation rejection and
CallModel live in `tests/v2/cmake/CPUNativeTileRegisterEvidence.py`, not imported
`elf_tools.py`. Accordingly the inert patch targets that existing authority,
its source-named tests and focused preflight registration; no second analyzer.
The parent approved this bounded first proposal: TLS implementation + tests,
static authority design **explicitly incomplete**, before broader implementation.

The patch is NOT applied. `check_proposal.py` copies the complete CMake and unit
fixture support directories to a disposable directory, applies the patch there,
and runs developer controls only. Source is read-only to that experiment; the
existing bounded-memcpy test reads the source through a temporary symlink.
No native production object, qualification receipt, role, budget, kernel, table,
traffic result or baseline comparison is changed or certified.

## TLS exact semantics for review

Only relocations 549 (`TLSLE_ADD_TPREL_HI12`) and 551
(`TLSLE_ADD_TPREL_LO12_NC`) become parseable. They are never generically skipped:
normalization requires consecutive instructions within one function, same
symbol, zero addends, exact `add xD, xB, #0, lsl #12` then
`add xD, xD, #0`, D/B in x0-x30, never SP/ZR. Both placeholders must be zero;
non-LE types, alternate low pairs, 32-bit forms, wrong shifts, orphan or mismatched
records, wrong register chains and boundary crossings fail. Normalized symbolic
TPREL operands preserve unknown arithmetic, not fictitious zero arithmetic.
Existing stack-provenance rules reject any stack-derived source. Only xD changes;
there is no resolver call, NZCV change, save exemption or call-traffic credit.

No new generic ISA forms, Mach-O relocation admission or dynamic binding rules.
The existing front end still checks printed/object relocation equality before
pair normalization. Consistent nonzero-addend ELF + printed fixtures exercise
the pair's own rejection; a separate inconsistent printed fixture exercises the
earlier parser mismatch. The fixtures use zero instruction bytes and simulated
disassembly, like the existing tests: they are not native compiler qualification.
The original checker rejects the same generated pair at relocation 549.

Five new focused tests cover these contracts. Existing descriptor, helper call,
provenance and relocation classes run alongside them (37 total). Proposed
`V2_Integration_CPUNativeTileTLSLocalExecPair` explicitly registers the focused
class in ProductionTestPreflight; the existing analysis Unit shard also includes
it automatically. No configured CTest inventory has been changed or executed.

## Static executable proof design — NOT IMPLEMENTED / approval required

A string `nonpreemptible=static-executable`, caller-provided map digest, or a
same-TU assumption is NOT sufficient. CallModel must remain rejecting static
weak/default symbols until the following native artifact authority exists.

1. **Executable identity.** Parse bounded ELF64 little-endian EM_AARCH64 bytes,
   program headers, section/symbol tables and dynamic symbols, not filename or
   build-cache flags. Admit ET_EXEC only initially. Reject ET_DYN (including PIE)
   rather than guessing DSO versus executable; extending this needs separate
   review. A static core executable may still load system DSOs: require the
   target absent from every SHT_DYNSYM entry regardless of defined/undefined
   status. Validate the sole final defined STT_FUNC's binding (STB_WEAK for this
   case), default visibility, address, size and executable section membership.
   Duplicate/alias ambiguity, IFUNC, COMMON, undefined, malformed or stripped
   insufficient evidence fails. This is static **symbol binding**, not a claim
   the entire application is fully statically linked.
2. **Exact link transaction authority.** Producer captures the real Ninja final
   edge, cwd, ordered argv (including response-file contents and search paths),
   compiler/linker identity and hashes of every transitive input (archives,
   objects, scripts, plugins, libraries). Reject unsupported/implicit input
   discovery until completely resolved. Re-run control with ONLY output changed;
   re-run map with ONLY output changed + GNU `-Map`. Original/control/map ELF
   bytes must be identical. Preserve dependency side outputs and prove originals
   and all named inputs unchanged before/after. The consumer authenticates all
   files and replays the exact bounded transaction before constructing authority;
   a map digest alone must never mint an admission. Failed/incomplete transaction
   retires any previous proof; never reuse a prior success after an input change.
3. **Kept versus discarded selection.** Parse only the explicitly supported GNU
   map grammar, separated at its unique discarded/kept boundary. Record every
   occurrence, including discarded COMDAT copies, section sizes and placement.
   Exactly one kept input-section interval must own the full final symbol extent.
   A symbol appearing only in a discarded section, duplicate kept candidate,
   unknown map grammar or unsupported linker optimization (ICF/LTO rewriting)
   fails. Do not infer kept from matching member basename or first map occurrence.
4. **Input identity / archive extraction.** Bind exact archive path+digest, member
   ordinal, raw member name and header offset (duplicate basenames are not unique
   identity), or the exact direct input object. Extract from those authenticated
   archive bytes. Thin archives require full referenced-member closure or fail.
   Compare extracted bytes with the audited unit's original object bytes; compare
   the kept input section and symbol-relative pre-relocation bytes with the exact
   B1 replay. Store the whole object digest and precise section index/name/extent
   and function digest with the Unit at construction, not a JSON assertion supplied
   to select(). Selection can legitimately choose another TU: audit that actual
   selected Unit, never substitute the caller's weak copy.
5. **Final symbol crosscheck.** Kept section placement plus original symbol offset
   must equal the native final symbol address; input and final symbol sizes must
   agree, with the full extent inside the kept section and final executable
   section. Relocated final bytes need not equal pre-relocation bytes. Whole-ELF
   identity across the transaction binds the linker result; raw pre-relocation
   byte equality binds the audited implementation. No unreviewed relocation mask
   or silent stripping may be introduced to make those bytes compare.
6. **Typed boundary.** Proposed separate `StaticExecutableSelection` authority is
   constructible only by the native transaction verifier, carrying executable,
   input object, section, symbol and selected Unit identities. CallModel consumes
   it for this new mode only and rechecks target/copy identity. No new magic enum
   inside the existing `direct` record, no public digest-as-proof constructor,
   and no changes to existing `direct`/`process`/PLT behavior. Persistence uses a
   versioned manifest whose complete inputs are revalidated before authority is
   reconstructed; serialized success booleans are not authority.

### Required static implementation tests before any use

A real bounded native link fixture must have competing weak/default copies in
*different* archive objects, one discarded and one kept; final symbol absent
from dynsym. Demonstrate the old checker rejects its valid static selection,
then the proposed verifier admits only the real kept copy. Negative controls:
forged map (including valid-looking wrong winner), stale map/input/ELF, mutated
archive member bytes, duplicate member identities, changed link argv/response
file/tool, non-identical control/map ELF, discarded-only section, ambiguous kept
interval, wrong final address/size/binding, selected Unit bytes differing from
extracted bytes, dynsym entry and DSO/PIE. Each must fail at its intended boundary,
not merely from an unrelated malformed fixture. Preserve original dynamic-rule
negative tests unchanged. Register focused static verifier tests explicitly in
ProductionTestPreflight. None of these static tests exists in this first proposal.

## Validation and resume boundary

Run developer checks only:

`python3 -B scripts/native_quantised/qualification/proposals/2026-11-27-tlsle-static/check_proposal.py`

Observed: 37 tests passed; original TLS pair rejected at 549; original static
vocabulary rejected the proposed spelling. The latter is only a vocabulary
control, NOT a native executable/static-proof test. Early scratch attempts failed
on missing fixture dependencies; an initial addend test hit printed/object
mismatch instead of pair validation and was corrected with consistent printed
addend plus a separate mismatch negative. No assertions were weakened.

Review must approve TLS grammar and static authority design before any live
application. Parent must push a new coherent implementation commit marked
unverified before native verification. Then re-run actual selected WeightManager
TU and native static negative controls with retained source/tool manifests.
Q5 route(a), baseline58d77 comparison, per-role/class traffic/spills and K-loop
callee save accounting remain BLOCKED. This document grants no qualification.
