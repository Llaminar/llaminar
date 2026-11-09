# Idea 32 — NEON serial/grouped NaN-precedence witness (Main follow-up)

Status: OPEN (tracked 2026-11-09; Main noted — NOT a blocker; while adding the
__AVX2__ ordered-add witness, Main is checking whether NEON serial/grouped
epilogues can DISAGREE on two-NaN payloads before adding a NEON witness
(maxim-20 point). If they can disagree -> add a NEON witness + regression;
if provably consistent -> close with evidence.
