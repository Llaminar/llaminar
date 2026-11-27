# Separate production line-witness plan proposal / handoff

**No plan grammar, memcpy table, budget, role or kernel change is implemented
here.** Static selection tooling is complete; production economy qualification
requires this separate reviewed source-owned extension and the alias proposal.

## Reproducible actual boundary

Use the same tool commit/SHA, source/build identities and native container mounts
recorded in `STATIC-ALIASES-PROPOSAL.md`. On factory@192.168.178.80:

- Existing selected object:
  `/home/factory/pr73-arm-stack-q5/numtree/build/CMakeFiles/llaminar2_core.dir/loaders/WeightManager.cpp.o`.
- Exact B1 replay/assembly/dependencies/command:
  `economy16/static-candidate-I8MM/WeightManager.cpp.o/{production.o,production.s,production.d,b1.json}`.
- Existing source-owned witness diagnostic:
  `economy16/static-reviewed-a698-line-witness-candidate-I8MM/`.
  `result.json` records the precise rejected native `-MT` operand;
  `unvalidated-native-driver-jobs.{stdout,stderr}` retain actual GNU `-###` output;
  `unvalidated-native-job-inventory.json` retains exact original argv, cwd/env,
  same-build compileDB entry and complete native command forms. These observations
  are diagnostic, never an alternative plan authority.
- Sole source-owned invocation script retained outside the repository:
  `/home/factory/pr73-static-selection-worker/line-witness.py`.

The actual selected/B1 object SHA is
`2e4caad3baaa10cd948e53c696be499df93c65948d0f9f3a1ed49104c16d14f0`.
Retained release assembly has zero `.loc` directives, so supplying its text alone
cannot authorize any variable-size memcpy site. Header
`/num/src/v2/kernels/cpu/gemm/CPUNativeQuantisedMultiScaleKernels.h` has SHA
`44a46bf7b36830996a9b93ec0961bb08d3d862babd9194c6353a34eba62485ed`;
existing `CPUNativeTileBoundedMemcpy.json` already admits its exact sites:

- line 1110: `std::memcpy(&acc[row], output[row] + column, (columns - column) * sizeof(float));`
- line 1354: `std::memcpy(output[row] + column, &reduced[row], (columns - column) * sizeof(float));`

The diagnostic used existing `line_witness_command`/`build_line_witness` APIs,
without substituting flags. It stopped **before compiling**:

```
ValueError: job value -MT CMakeFiles/llaminar2_core.dir/loaders/WeightManager.cpp.o not admitted
```

The existing GNU plan grammar is probe-specific (`-MT probe`) and also lacks the
actual production `-isystem` form. Do not fix one flag at a time or rerun an ad-hoc
`-g` compilation. Inspect the complete retained jobs once.

## Same-build compileDB discrepancy to settle explicitly

`/num/build/compile_commands.json` SHA256:
`5fedbde190388dc1c3b4916261e673cfc8e6615b7323307f012ffef8215f7f05`.
Its exact entry is selected by both FILE `/num/src/v2/loaders/WeightManager.cpp`
and OUTPUT `CMakeFiles/llaminar2_core.dir/loaders/WeightManager.cpp.o`, cwd
`/num/build`. The entry includes the actual `-I`, three `-isystem` pairs,
`-mcpu=generic`, `-march=armv8-a+simd+dotprod+i8mm+fp16fml`, `-O3`, `-std=gnu++23`,
`-fopenmp`, defines and original `-o`/`-c` operands.

**It does not contain `-MD/-MT/-MF`.** Native Ninja's executed B1 command adds
`-MD -MT <original object> -MF <original object>.d` (and has a ccache launcher).
The new authority must explicitly bind that build-generated metadata relationship,
not silently claim the compileDB command and executed command are byte-identical.
All original codegen and include/define options must remain exact. The original
source-owned `build_line_witness` plus full original argv produced the failure
above; its guard was not bypassed.

## Reviewed bounded direction for a fresh implementation

- Bind exact same-build compileDB bytes and unique FILE+OUTPUT entry, actual
  original executed edge and all native job forms. Resolve the metadata relation
  above explicitly in the reviewed interface.
- Witness outputs must be arena-independent artifact paths, never build-tree
  object/dependency outputs. The witness command preserves original `-MT`,
  `-MD/-MMD`, `-MQ` and every codegen token, changes only reviewed output operands
  (`-o`, `-MF`) and appends the existing `LINE_WITNESS_FLAG` (`-g`). No new debug
  heuristic or generic production compiler driver is needed.
- Admit typed dependency metadata and include-search forms (`-I/-isystem/-iquote`)
  and exact defines only through the bound entry/edge. Unknown or changed
  `-m/-f/-O/-std` options fail, even if the compiler accepts them.
- Reuse `require_code_identity` on **all code and data SHF_ALLOC sections and their
  relocations** before any location context is installed. Then and only then use
  `assembly_functions(..., locations)` and `closure_sources` followed by
  `install_bounded_context`. Hash the `.loc` file table and source closure for
  the selected actual source tree. No memcpy-table or role/budget edit is needed.
- Focused negatives: stale/foreign/ambiguous compileDB entry, changed/unknown
  codegen, extra argv, duplicate/removed `-MT`, dependency output in build tree,
  wrong output/FILE, changed source closure, differing allocated code/data and
  incomplete/foreign location tables. Use the native production command forms,
  not only a synthetic probe.

After both independent extensions pass review/native controls, analyze each
actual helper/copy/class/role separately on baseline58d77 and candidate I8MM with
one identical tool commit. Include K-loop call multiplicity and callee per-call
frame/saves; prohibit offsets between roles or helpers. Stop at any new unmodelled
binding, relocation or memcpy site. Current raw diagnostic counts are not a
qualified result, and no baseline comparison has been performed by this lane.
