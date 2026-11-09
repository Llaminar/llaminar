# Idea 31 — final-base panel census + pressure table (2026-11-09, perf lane)

NON-CERTIFYING baseline compile census; no production edits/kernel experiments/
runtime timing. Base 777f7a52a; GCC14.2 (llaminar-dev); one CPU, quota1, 8GiB,
no net/GPU. 48 programs/envelope x NEON_DOTPROD + NEON_I8MM = 96; expected =
observed. Canonical witness overrides + unmodified canonical parser/classifier
(analyzer SHA e6350f86...). Scratch: artifacts/panel-census/{census.json,
ForcedStripProbe.cpp, census.py, NEON_DOTPROD.asm, NEON_I8MM.asm,
census-stdout.txt} (station). INCOMPLETE: 4 forced-S4 Q6 programs unclassifiable
('ldr x,mem,imm post-index' outside admitted grammar; 21 structural msgs each;
partial spill = UNKNOWN; nothing weakened; installed Q6 width = 2 so the .80
installed census unaffected). Tooling gap -> GitHub #56.

## Spill table (static FP/SIMD spill units, proven ABI saves excluded)
Rows=1, Family=SDOT; Policy0=BackendNative 1=GPUAlignedExpert.
NibbleLUT Centered false: S1 0 / S2 0 / S4 22 (both policies) - rejected.
NibbleLUT NonLinear false: S1 0 / S2 0 / S4 22 (both) - rejected.
NibbleLUT Unsigned true (min): S1 0 / S2 0 / S4 6 (both) - rejected.
NibbleLUTSourceScale NonLinear false: S1 0 / S2 0 / S4 31 (BN) / 20 (GA).
ExpandedInt8 Centered false (incl. Q5 min path): S4 0 (both; no GNU defect).
IQ3SExpandedSourceScale Centered false: S4 0 (both).
Q6KNativeDualScale: S4 UNKNOWN (unsupported form; no claim).
=> 92/96 classifiable; 14 classifiable S4 = 8 spilling nibble + 6 zero controls.

## Pressure/location (NEON_DOTPROD; I8MM agrees)
- 0:0:1:1:4 / 1:0:1:1:4 (6): K loop .L1c8; `str q31,[sp,#160]` before first
  SDOT chain + `ldr q1,[sp,#160]` restore for lane2; 4 more static vector
  ops around partition traversal (d8-d15 ABI proved/excluded). Hypothesis:
  reduce statically-expanded decoded-fragment overlap; keep shared
  activationBlock + immediate dotLane; full closure must reach zero.
- 0:0:0:0:4 / 1:0:0:0:4 (22): K loop .L1a8 includes per-block scale/FMA
  epilogue partial-accumulator ld/st at SP+160/176 (load q26 -> fmla -> store
  q26; backedge b.gt confirms in-loop); more after backedge (partition/tail).
  Hypothesis: free decoded/constant live ranges BEFORE strip epilogues so
  independent partials stay in registers; verify no move to another region.
- 0:4:2:0:4 (31): K loop .L1d0 source-scale coefficient; strip epilogues
  load/FMA/store acc words SP+176/192/160; hypothesis: shorten decode/
  metadata lives first; preserve base x coefficient scaling + round order.
- 1:4:2:0:4 (20): GPUAligned variant .L1a0; different rounding -> no BN
  evidence reuse; independent gate.
- Q5/expanded-min (0): no GNU spill -> unchanged regression control only.

## First slice (APPROVED 2026-11-09, arm-perf)
- ONE hypothesis: NibbleLUT Centered + NonLinear (22/22 each) decoded-fragment
  lifetime shortening in Rows1/S4 indexed-SDOT panel; all other nibble
  variants + controls improve-or-unchanged; unsigned+min 6 = regression
  control + witness; retain C3 activation-block reuse + compile-time lane
  indices; NO panelStrips change; no arithmetic reassociation (v2 invariants).
- IQ4_XS source-scale (31/20) = NEXT distinct candidate (not bundled).
- Q5 GNU = control only. Q6 = tooling gap (#56), no slice work.
- Gates per v2 (byte oracle + zero-spill closure + recorded replay + .80
  panel-tournament economy; AppleClang secondary later); isolated branch off
  777f7a52a; self-gate evidence -> reviewer -> confirm -> push.
