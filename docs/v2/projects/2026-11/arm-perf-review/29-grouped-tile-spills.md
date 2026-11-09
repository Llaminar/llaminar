# Idea 29 — Pre-existing grouped-tile FP/SIMD spills on ARM (Ratchet follow-up)

Status: **Open (perf item)**. Found 2026-11-09 by widening the tile-register
detector (SOURCE114 direction).

## Finding

Production gcc-14 on .80 (probe at NEON_I8MM; cells = programs with spills /
spill instructions, out of 16 per family):

| Build | R2 ASIMD/SDOT/I8MM | R3 ASIMD/SDOT/I8MM | R4 ASIMD/SDOT/I8MM |
|---|---|---|---|
| base 302318f1e | 14/30, 8/16, 2/6 | 14/46, 14/52, 16/52 | 16/947, 16/70, 16/40 |
| R1 035d1d948 | 14/28, 12/24, 2/6 | 14/158, **14/463**, 16/58 | 16/966, **16/666**, 16/48 |
| fix b01d285dc | 14/30, 8/16, 2/10 | 14/52, 14/65, 16/54 | 16/947, 16/66, 14/34 |

- The grouped row tiles (Rows 2–4) spilled in the BASE — the worst is **R4
  ASIMD ≈ 950 spill instructions**. The earlier [sp]-only census missed these
  (gcc addresses them through x29).
- The R1 regression was real in grouped SDOT (R3 52→463, R4 70→666); the fix
  brings both back to base level.
- Rows=1 is ZERO in all three builds on Linux gcc-14 (the M=1 spill seen
  earlier was Darwin gcc output — see idea 28 for the open M=1 investigation).

## Gate design (ratchet, approved 2026-11-09)

- Rows=1 (decode): ZERO FP/SIMD spills enforced (hard rule).
- Rows 2–4: committed per-program RATCHET BUDGETS keyed by compiler identity
  (id + major version) + ISA envelope; values = today's counts (fix ≈ base);
  over budget fails; unknown compiler identities FAIL CLOSED. Never change
  production allocation to fit the scanner.

## Follow-up

This is a real perf item: reduce/eliminate the grouped-tile spills (R4 ASIMD
~950) — C4 (wider panels, addon lane) touches the same tiles; coordinate when
C4 lands.
