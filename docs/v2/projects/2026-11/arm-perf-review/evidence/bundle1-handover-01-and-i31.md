# Handover: bundle1-perf-harness (01) final push + idea-31 AppleClang closure

Sent to arm-perf because the qualification session (01a1bd53, Herdr wF:tJ) is being retired.
Do not send relays to that session after this.

## Lane 01 — branch codex/bundle1-perf-harness (test-only, 11 commits)

State at handover (2026-11-10 ~16:00Z):
- Remote branch = 5d606a10894f7b3cc34b103de96c30a60003a773 (reviewer-accepted source at 0f9960bb5 + docstring fix 28e64b4e4, rebased onto ce9ddb326).
- Acceptance inputs COMPLETE: WSL full final gate GREEN at 5d606a108; .80 bound run graded BOUND at 5d606a108 (kernel-metrics, Release, g++-14 -O3, SHARED_CORE=ON; Ladder Q4_K/Q5_K/Q6_K/IQ3_S x 20/10/1 threads; results .80 build/results/bound-5d606a108.json + bound-bw-result.json).
- Local candidate a4ebff3dd3ffc63aa2a2a8484f88ca9afdbd3e76 = the same 11 commits rebased onto 28ddbe253, PROVEN (range-diff 11/11 `=` vs 897727484; delta-equality: all non-CMake files identical, CMake = same 39 added lines, 0 base removals, block after panel-tournament; ctest -N both registration sets; 5/5 rerun in the llaminar-b1-dev container). NOT pushed (arm-perf held the push pending Main's SETTLED + combined-FF sequence).
- Observed #43 tip at handover: 763a8e15b (may include the combined idea-31+B9 FF; verify against arm-perf's relay).

Remaining procedure (per arm-perf's approved settlement protocol; each re-rebase inherits the delta-equality argument and needs only a fresh range-diff + ctest -N + 5/5 rerun):
1. Wait for arm-perf's relay that the combined FF has landed; use the then-current tip.
2. Re-rebase the 11 commits onto it: `git rebase --onto <tip> <prior-base>` (bases were REWRITTEN twice: ce9ddb326 -> 8fcc0fe0b -> 28ddbe253, so --onto from the prior base is required). Expect no conflicts; if CMake interacts, KEEP BOTH base-first, added lines verbatim.
3. Proof: fresh range-diff (expect `=` everywhere except the commit that first adds the CMake block, anchor-only), delta-equality (same 39 added lines, 0 base removals), reconfigure + build in llaminar-b1-dev, `ctest -N` showing both registration sets (panel-tournament + DecodeBandwidth + PerfBaseline; contracts in ProductionTestPreflight), 5/5 rerun (V2_Unit_PerfBaselineIdentity, V2_Integration_PerfBaselineProvenance, V2_Integration_CPUNativeQuantisedDecodeBandwidthContract, + the two panel-tournament contracts as coexistence control).
4. Send proof to arm-perf, then push: `git push --force-with-lease=codex/bundle1-perf-harness:5d606a108 https://github.com/Llaminar/llaminar.git <new-sha>:refs/heads/codex/bundle1-perf-harness` (force-with-lease pre-approved on the proof basis; FF approval pre-approved).
5. Main fast-forwards #43.

Proof execution environment (private, do NOT use llaminar-armbuild):
- Container llaminar-b1-dev; `export CCACHE_DIR=/ccache; cmake -S /src/src/v2 -B /build` then build targets v2_perf_cpu_native_quantised_decode_bandwidth (+ v2_perf_cpu_native_neon_panel_tournament for the coexistence check); ctest --test-dir /build.
- aarch64 Mac host runs the aarch64 gates; the x86-only OrderedNibbleSplitDot registrations must stay absent there (confirms 28ddbe253 gating).

## Idea 31 — AppleClang closure offer does NOT survive

The canonical witness extension (register-gate tooling, addon lane) will need an AppleClang-21 closure run (NEON + NEON_DOTPROD + I8MM compile-only per inventory). My offer to run it on the M1 dies with this session; the addon lane should re-book it or arm-perf should assign a Mac session.
Reference material (VOLATILE /tmp, dies on reboot): /tmp/armqual-i31/census/{ForcedStripProbe.cpp, census_forced.py, census-*.{json,o}, identity-compare.txt} — the scratch probe + driver showing the inventory override the canonical extension needs. The full idea-31 AppleClang record is /tmp/armqual-i31/idea31-appleclang-secondary.md (accepted + recorded as 0ad84ecca by arm-perf — if not yet archived into the tracker, archive it NOW).
Worktrees llaminar-i31-cand / llaminar-i31-parent + the four builds can be freed once the closure is reassigned.

## Local state and cleanup

- Worktrees: /Users/floriankinder/repo/llaminar-bundle1 (detached helper), llaminar-bundle1-dev (branch checkout, clean at a4ebff3dd).
- Containers: llaminar-b1 (record build, historical), llaminar-b1-dev (proof runner). Volumes llaminar-b1-build/ccache, llaminar-b1-dev-build, llaminar-b1-ccache.
- Per the standing cleanup rule, these can be removed AFTER the branch is pushed and #43 fast-forwarded (the branch then exists on the canonical remote). Before that, llaminar-bundle1-dev holds the only copy of a4ebff3dd-equivalent content (re-derivable by re-rebase from 5d606a108).
- /tmp/armqual-b1 raw evidence (15/20/09 docs + classifier already archived in tracker at evidence/bundle1/; 01-* logs may still need archiving — arm-perf decides). /tmp/armqual-i31 as above. Both die on Mac reboot.

Routing unchanged: base feedback -> Main (01a1a8bf), perf findings -> arm-perf, reviewer via arm-perf.
