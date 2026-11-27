# Separate compiler-alias authority proposal / qualification handoff

**Proposal only. No alias admission is implemented by this lane. Q5 is NOT
qualified.** The completed static tool deliberately rejects overlapping aliases.
The next implementation requires its own reviewed diff and native controls.

## Completed tool checkpoint

Source commit: `a6985d38f49a09267fd2ae302e118aaba3ab0ee6`.
Sole analyzer: `tests/v2/cmake/CPUNativeTileRegisterEvidence.py`, SHA256
`ecc2b8ae455954009430c78aa45d297480992dbacb9714910b80a0af810275f7`.
No retired analyzer is imported. Native tests on AArch64 factory@192.168.178.80,
image `pr73-arm-stack-repair:gnu14` (`005a823c10bf`), passed 4/4 in 6.790 seconds.
Retained under `/home/factory/pr73-static-selection-worker/native/`:

- `static-comdat-l459fq77`: selected-object/CallModel positive, real transitive
  DSO authority and all ordered resolution, wrong-path/SONAME/parent, duplicate,
  non-DSO/PIE/target-definition and already-traced-loader negative controls.
- `static-closure-_t0lvg7y`: actual CRT/plugin/tool/response/script/defaultscript,
  environment/argv, relink byte-identity and trace/dependency fault controls.
- `static-native-env85et9`: native acquisition and option/dependency-token controls.
- `static-ninja-q4sqo3qo`: real consumed depfile/VALID native record and stale,
  wrong-target, absent opt-in, changed tool/database and incomplete-set controls.

Actual candidate verification passed, not merely a synthetic fixture:
`/home/factory/pr73-arm-stack-q5/economy16/static-reviewed-a698-candidate-I8MM/`.
`diagnostic-result.json` records 3,060 normalized WeightManager functions and
native typed authority/CallModel success for weak `reconstruct<12,I8MM>`.
Its `capture/` retains every exact driver argv/environment, original and redirected
executables/dependencies, Ninja database/tool identities, default script, plugin,
map, archive-member identity, nm checks and seven proven transitive DSO inputs.

Unchanged actual anchors:

| Artifact | SHA256 |
|---|---|
| `/num/build/llaminar2` | `092a5cc76c167414c4355c50666d0c89beea528185001266baeb515964e1982e` |
| `/num/build/.ninja_deps` | `c0ddca45b4e766f32c1feec3722140ed82ff793e4abf756e3b8cb48038ecba41` |
| selected WeightManager object and whole B1 replay | `2e4caad3baaa10cd948e53c696be499df93c65948d0f9f3a1ed49104c16d14f0` |

Production command and full environment are retained in `diagnostic-input.json`,
not reconstructed from a manifest-as-proof. Original edge:
`economy16/static-candidate-I8MM/final-link.txt`; cwd `/num/build`; actual explicit
Ninja `/opt/llaminar-tools/bin/ninja`. Original P is naturally consumed by Ninja;
its exact `llaminar2` record has 43 dependencies. Original script-option scan is
in `capture/production-policy.json`. No production output or database changed.

## Actual new alias boundary

Bounded continuation verified the real production link separately for each
newly reached weak helper, without assuming the caller's TU wins. The first
helper is rejected by the existing overlapping-alias rule. Evidence:
`economy16/static-reviewed-a698-lambdas-isa1-candidate-I8MM/`:
`lambda-0/`, `overlap-observation.json`, `diagnostic-result.json`.

Exact target:

```
_ZZN9llaminar23cpu16native_quantised11multi_scale7dotHalfILi17ELi0ELi4ELm2ENS2_17NEONMultiScaleOpsILNS2_11NEONByteDotE1EEEEEvPKhiRKSt5arrayIPKNS_9Q8_1BlockEXT2_EEiRAT2__NT3_1IEENKUlTnivE_clILi0EEEDav
```

Exact compiler alias:

```
_ZZN9llaminar23cpu16native_quantised11multi_scale7dotHalfILi17ELi0ELi4ELm2ENS2_17NEONMultiScaleOpsILNS2_11NEONByteDotE1EEEEEvPKhiRKSt5arrayIPKNS_9Q8_1BlockEXT2_EEiRAT2__NT3_1IEENKUlvE_clILi0EEEDav
```

Both are final weak/default STT_FUNC, section index 13, address `0x4d6f70`,
size `0x110`. Retained exact B1 `production.s` contains `.set ALIAS,TARGET`.
This is an observed compiler alias, not invented linker ICF equivalence. The
second reached helper ends `UlTnivE_clILi1EEEDav` with the same preceding raw
prefix; it still requires its own selected-copy/B1/alias proof. A similarly named
`NEONByteDotE0` helper also exists but is not either of these reached ISA1 calls;
do not select it by substring or count alone.

## Reviewed proposal, not installed behavior

1. Bind both input names to the same section/value/size, weak/default STT_FUNC,
   and exact selected object/B1 bytes. Require exactly one actual `.set` (or
   reviewed assignment spelling) with its explicit direction; no other overlap.
2. Require both final names unique, identical address/size and weak/default, both
   absent from dynsym, and one kept input section. Preserve complete native link
   closure, relink, map, archive ordinal/header-offset and whole-object checks.
3. One static authority owns the proven extent/alias set. CallModel resolves
   either name to the same audited body; analyze once and do not double traffic.
4. Reject different sizes/sections/bindings, IFUNC, missing/wrong/ambiguous `.set`,
   unproven extra overlap, ICF and folded/discarded-copy substitutions. Exercise
   one real compiler-alias fixture and a focused negative for each boundary.

Do not broaden the current verifier merely because two bytes or names match.

## Numerical/economy continuation remains blocked

`static-reviewed-a698-traffic-candidate-I8MM/native-traffic-diagnostic.json` is
**invalid for economy acceptance**: the three rowsProgram roots still have
structural errors from missing memcpy source context and the unadmitted weak
lambda copies. Reconstruct alone has no structural errors. Raw counters must not
be reported as accepted per-role/class counts. No spills/no-offset verdict exists.

Run the same reviewed tool on separate exact builds for baseline
`58d77f05aaa08f55fee9c43b608a4931a06d76dd` and candidate
`e4add4e20b8a3f436d4ef629f6607b2b3827a720`. Current actual production build is
`/home/factory/pr73-arm-stack-q5/numtree/build` (mounted `/num/build`), Integration,
I8MM. This is an exported tree without `.git`; source/compileDB manifests, not its
basename, must authenticate its owner-recorded candidate identity. No full
baseline production build was identified/admitted in this lane. `head2`, `head3`,
`base`, `develop`, `src` are not qualified aliases for that baseline. The existing
`work/NEON_I8MM-Integration` is only `tests/v2/portability`, not the production link.

Retained historic tools manifest SHA256 is
`72f815965e96bdc483936126b2176f55369422f110827223ff61a9a47997d46a`
(`economy16/tools-SHA256SUMS-v9`); it authenticates historical artifacts only.
Do not load its retired analyzer or treat it as new authority. Actual build
manifest `economy16/static-production-SHA256SUMS` binds original outputs,
compileDB and CMakeCache. See the separate line-witness handoff before counting.
