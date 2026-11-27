# Static acquisition implementation contract — 2026-11-27

Status: approved bounded direction, **NOT IMPLEMENTED**. No static selection
admission exists. The TLS slice does not close this boundary. The earlier
REVIEW.md, patch and manifest are immutable historical proposal artifacts;
check_proposal.py was for the pre-application checkout, not the current tree.
The original manifest authenticates those historical artifacts only.

## Observed executable identity (S1)

Inspection command, on factory@192.168.178.80:

```
readelf -h -d -l /home/factory/pr73-arm-stack-q5/numtree/build/llaminar2
```

The actual candidate executable is ELF64 little-endian AArch64, **ET_DYN**,
entry 0x173640, with PT_INTERP `/lib/ld-linux-aarch64.so.1` and DT_FLAGS_1
`NOW PIE`. ET_EXEC-only support would exclude this actual executable. The
approved contract is ET_EXEC **or** ET_DYN with both DF_1_PIE in DT_FLAGS_1
and PT_INTERP; an ordinary DSO must fail. The selected function must be absent
from every dynsym entry, including undefined entries. Native static binding
does not imply a wholly statically linked application: this executable has
DT_NEEDED system DSOs.

## Existing diagnostic is not authority

Root: `/home/factory/pr73-arm-stack-q5/economy16/static-candidate-I8MM/`.
The retained link.json, final-link.txt, ninja-commands.txt, selection.json,
WeightManager.cpp.o/production.o and static-production-SHA256SUMS remain useful
observations. They must not be silently promoted into a new proof schema.

Observed gaps in static_link.py:

- Its inputs comprehension hashes existing named argv files only. It omits
  `-lmvec`, `-ldl`, driver-added CRT objects/libraries, default linker scripts,
  plugins and implicit search resolution.
- The dependency side output is restored or removed after relinks, not retained
  as an authenticated complete input inventory.
- Linker identity, relevant environment and expanded response-file contents are
  not the complete authenticated closure.
- selection.json records paths/member names and selected bytes but does not
  authenticate archive member ordinal/header-offset identity from archive bytes.

Original/control/map byte equality is necessary, not sufficient to repair these
gaps. A link_evidence digest, static-executable string or serialized success flag
must never construct a StaticExecutableSelection authority.

## Approved bounded acquisition scope

Integrate with the existing source-owned checker, not the removed tools import
or another analyzer. GNU14/Linux driver and **ld.bfd only**:

1. Capture the exact final Ninja link edge, working directory, ordered driver
   argv, expanded response files and tool identity. Resolve the driver's `-###`
   jobs to the actual linker path, version and SHA256. Reject gold/lld/mold and
   unsupported jobs rather than guessing their semantics.
2. Bind the environment affecting resolution and reproducibility, including
   LIBRARY_PATH, COMPILER_PATH, GCC_EXEC_PREFIX, LD_RUN_PATH, LD_LIBRARY_PATH,
   SOURCE_DATE_EPOCH and GCC_*; distinguish unset from empty. Keep the same
   environment for original, control and map transactions.
3. Capture the native linker trace and retain its dependency file. Their input
   path sets must agree exactly after supported canonicalization. Hash every
   resolved input, including implicit CRT objects, all selected libraries,
   scripts and text stubs. Capture the default script from `--verbose` and bind
   it to the actual linker identity. Hash plugins even when unused. Unsupported
   or unresolved implicit discovery fails, never creates partial authority.
4. Reject `-flto`, any input `.gnu.lto_*`, `--icf`, `--gc-sections`, and relocatable
   `-r` links, including equivalent forwarded/response-file forms. Inspect map
   ownership too: folded, overlapping or garbage-collected selected copies fail.
   Do not admit a discarded copy because it has the same name or bytes.
5. Re-run the control with **only** the output changed. Re-run the map transaction
   with **only** the output changed plus the supported GNU map option. Require
   original/control/map executable bytes identical. Preserve original dependency
   side outputs while retaining separately authenticated copies of every new
   transaction's dependency evidence. Rehash inputs afterwards and reject any
   mutation. Failed capture retires previous proof before attempting work.
6. Parse the supported GNU map grammar with an explicit unique discarded/kept
   boundary. Require one kept input interval owning the full final symbol.
   Bind exact archive path, archive bytes, member ordinal, raw name and header
   offset; extract that member from the authenticated archive. Reject thin
   archives until their referenced closure is supported. Compare the extracted
   whole object with the audited selected Unit and exact B1 replay object and
   function extent. Match input symbol offset plus map placement to final symbol
   address/size and executable-section membership. Aliases/ambiguity fail.
7. Only this successful native verifier constructs the typed static authority.
   Bind Unit identity, target, object, input section, function extent and final
   executable. CallModel rechecks these identities. Preserve existing direct,
   process and PLT rules unchanged; no new magic direct-proof enum. A persisted
   manifest is a versioned input recipe requiring reconstruction and revalidation,
   never a constructor for authority.

## Required tests and handoff constraints

Focused static tests must run the production verifier on a bounded native weak
copy fixture with competing archive members, kept/discarded placement and no
dynsym target. Explicitly register it in ProductionTestPreflight. Demonstrate the
original checker rejects the genuine static selection without inventing a proof.

Each negative must fail at its own boundary: forged map/wrong winner, stale
object/ELF/map, mutated member bytes, duplicate member identity, discarded-only
copy, wrong final symbol fields, DSO, missing PIE marker/interpreter, dynsym
selection, GC/ICF injection, dropped implicit `-l` input, mutated CRT/default
script/plugin, changed environment, non-bfd linker, trace/dependency mismatch,
changed response contents/tool/argv, and non-identical relinks. Retain native
fixtures and exact provenance. Synthetic booleans or fake enums do not certify
these checks.

Commit coherent implementation **UNVERIFIED**, send its SHA to parent for push
before tests, and obtain review before qualification use. No Q5 static admission,
new native checker T-a, economy verdict, budget/table or kernel change is
permitted by the TLS tests or this document. Parent owns push; auto-merge OFF.
