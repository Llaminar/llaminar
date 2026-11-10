<!-- Provenance: reviewer ruling #166 (2026-11-10), delegated by the operator.
     Text below is committed VERBATIM as the standing rule. -->

### Standing rule: installed NEON tile register gate

**Claim level.** The installed-set register contract is a mandatory, fail-closed change-admission gate, not advisory evidence. Reworks, rebases, compiler changes and checker changes inherit it; ordinary re-certification does not reopen this decision.

**One authority and explicit scope.** Use the existing CPUNativeTileRegisterEvidence.py machinery and its canonical census. CPUNativeTileRegisterBudget.json owns the numeric limits; CPUNativeNEONTileRegisterProbe.cpp, the checker and V2CPUArchitectureTests.cmake own the installed inventory and compilation contract. Cover every declared compiler/ISA/configuration cell and every enumerated program, not just the cells that ran or passed. The gate measures the checker's per-program FP/SIMD stack-transfer instruction sites, excluding proven ABI saves/restores. It does not claim zero general-register spills, final-application code identity, runtime correctness or economy.

Performance-only forced/experimental variants remain diagnostic while unreachable in production. Default or user-selectable production admission requires their promotion into the canonical coverage and a complete pass first. Calling a production route experimental does not exempt it. Do not delete a required cell, program or architecture to clear a failure.

**Pass contract.** Every single-row program must have zero classified spills. Every grouped program must remain at or below its committed per-program ceiling. Every required cell must pass; reductions elsewhere cannot offset a failing key. Structural failures, unsupported instructions/calls, incomplete coverage, unknown identities and malformed or missing proof are never budgetable.

**Bound evidence.** A qualifying census consists of the existing canonical bound build/verify evidence for every required cell. The stand-alone census subcommand is diagnostic: it does not apply budgets or publish the complete build/verify receipt. Its exit status, a tree stamp, a summary marked GREEN, patch equality or equal budgets alone cannot satisfy this gate.

Validate the actual final configured candidate, including its complete dependency and command identities, compiler/assembler, object, disassembler, checker, envelope, inventory and budget. Any changed bound input requires fresh canonical evidence for the affected cells. Reuse unchanged cells only when the existing verifier validates their complete binding against that candidate. Checker composition and policy edits invalidate the bindings they change. Never splice incompatible epochs or relabel an old success after a failed collection. Preserve failure evidence; incomplete collection publishes no current pass.

**Operational consequence.** A classified spill/budget violation is a register-contract failure. Missing, stale, incompatible or infrastructure-failed evidence leaves the candidate unqualified; it is not by itself proof of a spill regression. Both withhold promotion, merge admission and qualification of the affected installed change until resolved. Attribution determines repair ownership, not permission to pass.

This is not a repository-wide work freeze: independent work retaining a valid covered state may proceed and reuse valid proof. A candidate containing the failing installed state cannot bypass the gate by calling its newest diff unrelated. Split or revert the offending change, fix it, or use the narrowly permitted budget amendment below. Other required gates remain in force.

**Downward ratchet.** Compare against the pre-change committed ceilings; do not let collection rewrite its own acceptance limits. A complete admitted census that lowers a grouped key's worst count must lower that key's committed ceiling. Where configurations share a budget key, use the maximum count across all required configurations sharing it, not the most favorable cell. This is a ceiling ratchet, not a separate tracker-only last-run baseline. Disclose per-cell changes even when they remain within a shared ceiling.

Automation may propose monotone reductions. Adopt them as an explicit source diff through ordinary review, with final proof bound to the resulting policy. Tightening needs no special exception approval; unattended rebaselining is not allowed.

**Only escape hatch: an explicit grouped-budget amendment.** Before acceptance, obtain a named register-gate reviewer's written ACK and normal integration-owner approval for the exact grouped keys and finite old-to-new ceilings. Retain the original failure, exact source/tool/configuration identities, bound per-program evidence, the reason for the increase, its disassembly attribution, and the applicable correctness/economy evidence. Explain why the trade-off is justified; recertification cost or a desire to turn the check green is not justification.

Commit the named change to the existing budget authority and verify the complete required census against the final source/checker/policy combination. A pass against a proposed loosening does not approve that loosening. Record acceptance as a reviewed budget amendment, never as non-regression against the old ceiling. Approval covers only that exact amendment; later increases require new approval.

No such amendment may waive single-row zero, structural completeness, binding/freshness, required coverage or another correctness/economy contract. No skip, anonymous reserve, parser blind spot, copied receipt or generic override is an escape hatch. Checker repairs require their focused controls and the existing complete checker gates before requalification.

The tracker records this standing rule, evidence references and explicit decisions. It is not a second measurement engine or live threshold ledger. Passing this gate does not replace any other production admission gate.
