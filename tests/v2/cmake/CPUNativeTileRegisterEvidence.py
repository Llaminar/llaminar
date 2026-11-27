#!/usr/bin/env python3
"""Build and authenticate a register-allocation witness for the native NEON projection tiles.

Build phase: compile CPUNativeNEONTileRegisterProbe.cpp into an object with the
real target's ordered compile command at -O3, disassemble that object with the
configured objdump and audit the instructions actually assembled.

- Object: the format (ELF64 AArch64 or Mach-O arm64) is read from the object
  itself and must match the disassembly's format line. The disassembler prints
  generic AArch64 syntax (GNU objdump, or llvm-objdump --triple=aarch64, not
  Apple's abbreviated printer), so every operand is visible; assembler-level
  controls (macros, conditionals, directives) cannot change or hide audited
  instructions, because only the assembled bytes are read.
- Object image: sections (with executable extents), the complete symbol table
  (raw linker name, binding, type, defining section, value, ELF size) and the
  relocation tables of every executable section (type, raw target symbol,
  addend) are read from the object itself, which must be a relocatable object
  (ELF ET_REL / Mach-O MH_OBJECT). Relocation records are admitted by their
  native fields: complete, bounded ELF RELA records with valid section links,
  an admitted code relocation type and an aligned in-section offset; extern
  Mach-O records whose type has the r_pcrel / r_length combination required by
  mach-o/arm64/reloc.h. Anything else fails before comparison. Function starts and
  extents (ELF symbol size; Mach-O next distinct defined address) must be
  instruction aligned, lie in their section and never overlap; every
  non-padding instruction and every relocation belongs to one.
- Inventory: the expected tile K-loop programs (row tile x integer family x
  policy x encoding/map/minimum) derive from the admitted ISA envelope; the
  inventory string embedded in the object and the object's function symbols
  must agree. Every expected program must exist exactly once; any other tile
  program symbol fails. Tile programs are the run<> programs of dispatch() and
  the coded runCoded<Codebook> programs it routes CompactMultiScale codebooks
  7 (Q5_K) and 11 (IQ3_S) to (key encoding 3, map = codebook, minimum =
  Codebook == 7); both are audited alike. Every other out-of-line member
  function of the tile class fails closed, except the probe's explicit
  execute<Dot>() dispatchers, exactly one per inventory entry, each proven
  loop-free and branching or calling into the tile class only to audited tile
  programs of its own rows and integer family.
- Disassembly contract: every executable section of the object is disassembled
  at every 4-byte address of its extent, in order, exactly once (no gaps,
  missing prefixes or tails, duplicates or reordering); every heading names a
  symbol defined at that address; function bodies are cut from the object's
  extents, not from printed headings. Every printed relocation must equal the
  object's relocation at that instruction (type, target, addend) and every
  object relocation must be printed; the object's record is the one analysed.
  Every line is a format, section, symbol, instruction or relocation line;
  in-function branches become local labels, relocated branches name their
  symbol, page and low-12 data relocations become symbolic operands. Unknown
  relocations, branch addends, unrelocated address operands, data words and
  ELF data mapping symbols in code fail; relocated memory or arithmetic on a
  stack address fails. A call relocated against an executable section's
  STT_SECTION symbol (identified by section header index) plus an addend names
  a callee only when exactly one STT_FUNC extent (aliases allowed) starts at
  that address; every other branch addend fails. Such a call is still subject
  to the call rules below.
- Admitted forms: every instruction must be an admitted (mnemonic, operand
  signature) form: the forms qualified on the GNU-14 and AppleClang-21 witness
  objects plus the explicitly modelled merge definitions. This is an inventory
  of supported forms, not a complete AArch64 grammar; any other form fails.
- Calls (call model, CallModel): the exact fatal-diagnostic symbol (a defined
  function of the object under its raw linker name, whose body must consist of
  admitted forms, call only fwrite/fputs, be acyclic and end every path in a
  call to abort()) is never-returning. fwrite, fputs, abort and the other
  never-returning imports (__cxa_throw, _Unwind_Resume, __stack_chk_fail),
  memcpy and __memcpy_chk are trusted only as the object's unique, undefined,
  strongly bound imports in the exact spelling of the format's C ABI; a foreign
  spelling, local definition, alias, rebinding or weak import disproves them.
  A TU-local helper is audited in its own body by the same analysis under the
  caller's entry facts (stack-address arguments map into the caller frame;
  every access through one must be exact and inside that frame, clear of live
  pointer, save and canary slots); a weak/global definition only through a
  link-selection record naming an audited copy, bound directly (proven
  non-preemptible) or by the process resolution of the actual executable and
  DSO set (a bare PLT binding is preemptible and fails).
  The helper's frame record and callee-saved saves/restores are call_frame
  traffic and its body traffic keeps its class, all charged at the call site's
  loop; its loops need a role entry and nest under the call site (the stack
  table multiplies caller x helper trips); it must restore sp and x19-x29 and
  prove its own d8-d15 saves; x0-x18 then hold the helper's joined return facts
  (exact, as GCC -fipa-ra may rely on them; FP/SIMD registers carry no
  tracked fact), x30 and NZCV are undefined.
  memcpy needs one constant size on every path (__memcpy_chk a constant bound
  >= size) and exact stack sides; it counts ext_call, ext_bytes and
  ext_accesses. Any other returning import is admitted only on a cold path:
  every path from it (normal and exception edges) ends in a never-returning
  call without reaching a return or a loop. Exception edges come from the
  object's .eh_frame FDEs and .gcc_except_table LSDA call-site records (ELF);
  a landing pad must be cold, and a block reachable through neither normal nor
  exception edges fails. Cold traffic is listed, never counted. The stack
  protector's canary slot store and reloads are call_frame traffic; protected
  array accesses count normally. Recursion, a call depth above 8, tail calls
  and indirect calls other than the TLS descriptor call fail; so does every
  call on Mach-O (no exception tables are parsed there). Without a call
  model (analyse_function() with no model) every call boundary except the
  diagnostic is a structural failure.
- Analysis: each program is analysed over its control-flow graph. Stack
  provenance of every general register is a forward data-flow fact; merge
  definitions (bfi, bfxil, movk) keep their destination as an input; csel joins
  its alternatives; any other computation on a stack address keeps it a stack
  address at an unknown offset; a base that is a stack address on only some
  paths fails. 8-byte frame slots that may hold stack addresses are tracked
  with may/must value sets (including the non-stack alternative); every memory
  access uses its exact opcode/arrangement/lane footprint; partial or
  reinterpreting access to such a slot, any stack access at an unknown offset
  while such slots are live, a 32-bit view of a stack address and a stack
  address leaving the tracked frame (to memory, an FP/SIMD register or an
  index) fail.
- Spills: each load/store moving FP/SIMD data (any width) through a stack
  address counts once, except proven ABI saves/restores of d8-d15: whole-register
  contiguous str/stp/ldr/ldp of the d or q view only, saved to a fixed slot
  while the register holds the caller's value (never written, or restored and
  not written since - a shrink-wrapped save/restore around each iteration of a
  loop), slot untouched while live, restored from that slot, never clobbered
  after the restore, restored before return. Only the low 64 bits of v8-v15
  are callee-saved (AAPCS64): a single-lane write wholly in bits 64-127 is no
  clobber; every other FP/SIMD write is.
- FP/SIMD data in general registers: a general register defined by UMOV/FMOV
  from an FP/SIMD register carries FP/SIMD data (through register moves,
  conditional selects and merge definitions, not through other computations);
  a stack store of it counts as a spill. A general register loaded from the
  stack carries stack data the same way; moving it unchanged into an FP/SIMD
  register (INS/MOV to a lane, DUP, FMOV) counts as a reload.
- Stack traffic (recorded, schema 12; the develop-baseline gate is a separate,
  reviewed table): every program's loop nesting forest is derived from its own
  CFG (Steensgaard: each non-trivial strongly connected component is a loop,
  entered at one header or, after GCC jump threading, at several blocks; the
  edges back to its entries removed expose the nested loops) and each
  instruction is attributed to its innermost loop (0: outside every loop).
  Per program and per loop, five classes are counted statically: `fp` (the
  FP/SIMD spills and reloads above); `scalar` (every general-register
  transfer through a stack address, except proven ABI saves/restores of
  x19-x30 outside every loop: a whole 8-byte store while the register holds
  the caller's value, reloaded from its own slot; a store over a live save
  slot voids the save); `abi_loop` (proven d8-d15 or x19-x30 saves/restores
  inside a loop, e.g. a shrink-wrapped save around each iteration: per
  iteration traffic, never exempt); `alias_load` (inside a loop, a load from
  an incoming argument register x0-x7 plus a known constant on every path -
  through moves, add/sub of constants, csel and exact 8-byte stack slot round
  trips - i.e. a loop-invariant reload of caller-owned descriptor memory);
  `alias_store` (a store to such an address, at any level); `invariant_load`
  (provenance-free: inside a loop, a non-stack load with a constant -
  immediate, absent or relocated - displacement, no writeback or post-index,
  whose base register is loop invariant: computed per loop (nested loops
  included) to a fixed point from the registers the loop does not define,
  adding a register only when every in-loop definition is a load from an
  invariant non-frame address, a constant, a move, an add/sub of a constant
  or a csel of invariant registers - never self-justified, so an induction
  walk stays variant; i.e. the same address every iteration, such as a
  pointer or span loaded from the descriptor before or inside the loop, a
  chain of them, an outer-loop-invariant row pointer or a constant table;
  every alias_load is also one, so alias_load is its incoming-argument
  subset), with `invariant_reloc` its relocated-displacement subset. Loads
  through an index, a post-index/writeback walk or a base the loop varies
  are neither. A reload through a frame slot never makes a register
  invariant (the slot may be rewritten; the reload is scalar traffic). Calls
  follow the call model (Calls above): an import redefines x0-x18 and x30,
  the TLS resolver x0 and x30, an audited helper leaves its proven return
  facts in x0-x18; NZCV is undefined after every call. The call-boundary
  classes call_frame, int_call, ext_call, ext_bytes and ext_accesses are
  counted per loop like the others. A general-register stack reload later
  moved into an FP/SIMD register counts in both `scalar` and `fp`. Loops are named by the `.loc`
  source locations of their entries and latches in the executed compile's
  own assembler input (instruction counts must equal the object's).
- Compact ordered additions (S2-4, compact_ordered_add_evidence()): per production
  copy, the GPU-aligned compact programs' inline-asm sites bound like the tile
  sites, and every compiler-emitted (non-template) scalar or vector `fadd`
  proven in source order by global reaching definitions (prove_order():
  loop-carried, across joins, through register and frame-slot copies; the
  first operand only from sites, the exact +0.0 initialisation, the
  source-bound Contiguous output seed or memcpy-filled slot, or proven fadds;
  the second operand never; every proven fadd grounded in a template, zero
  or seed origin, F-1), with exact committed counts. Tile GPU-aligned
  programs (T-1, tile_chain_evidence()): no compiler fadd may touch the
  accumulator chain (every chain addition is a template site); off-chain
  within-block expression adds are listed and counted, not ordered. Fused
  multiply-adds (S2-4c, fused_evidence()): every fmla/fmls/fmadd/fmsub/
  fnmadd/fnmsub of the audited closure must sit on a located source line
  that calls an explicit FMA and holds no contractible operation; weak or
  global callees need their link-selected copy's evidence (F-2), else an FP
  callee fails; exact committed counts per copy. Synthetic qualification
  only until B1 production objects exist.
- Structural failures are fatal and never paid from a budget.
- Policy: single-row (decode) programs must not spill at all; grouped row tiles
  may not exceed the committed per-program budget for the exact compiler
  identity and ISA envelope (CPUNativeTileRegisterBudget.json, a ratchet). The
  identity is derived from the compiler's own version output and must equal the
  configured one; unknown identities/envelopes fail closed.
- Dependencies: the dependency closure is the one the compiler reports for this
  exact compilation (-MD on the -c run), parsed in the plain subset of the
  native Make depfile grammar (names that Make escaping would alter - '$$',
  escaped spaces or '#', backslashes, quotes, wildcards - fail), cross-checked against a preprocessing
  pass with the same flags; every input is hashed before and after compiling and
  bound with its logical path, its resolved path (component-wise, as the kernel
  resolves symlinks and "..") and the symlinks followed. The Ninja depfile names
  resolved paths; symlink retargeting is caught by verification. Those lists are
  preprocessing dependencies only, and the assembler can read further files or
  construct directives through macros and substitution. Rather than model the
  assembler, the preprocessed translation unit (same flags) is retained and
  C++-tokenized; every asm keyword (including ones nested in operands or
  lambdas) must be an empty-template barrier or a declaration asm label. The
  label rule is a conservative syntactic approximation (one plain symbol, no
  qualifiers or operands, at namespace or extern "..." scope, after ')' / ']'
  or a non-keyword identifier, followed by ';', ',', '=' or __attribute__);
  anything else fails closed. Opaque preprocessor input (PCH markers, module
  pragmas, remaining include/import/embed) fails, as does any precompiled
  header the GNU compile reports reading (-H).
- Ordered addition: exactly one non-empty inline-assembly template is
  admitted, and only as the complete body of the GPU-aligned ordered-add
  helper: the token sequence `addOrderedNEON ( float32x4_t earlier ,
  float32x4_t later ) noexcept { asm ( "fadd %0.4s, %0.4s, %1.4s" : "+w" (
  earlier ) : "w" ( later ) ) ; return earlier ; }`, with no asm qualifier
  (volatile, inline, goto), no clobber and no memory operand, at most once per
  translation unit. Any other template text (a second instruction, another
  opcode or arrangement, a destination other than the first source), any
  other operand or constraint list, and the same template anywhere else stay
  unproved. The template is not trusted for its effect. The assembly read is
  the executed compile's own assembler input, not a second compilation: after
  the driver compile, the validated GNU plan (cc1plus, then as) is replayed
  with only GCC's temporary cc*.s replaced by the retained proof.asm, and the
  object and dependency file the replay writes must be byte-identical to the
  driver's (so proof.asm assembles into exactly the bound object). An
  integrated clang job assembles in memory and has no input to retain, so
  ordered-add sites cannot be bound there and such a proof fails closed.
  Within that assembly every function must have exactly the instruction
  count of the bound object's function (statements split at ';' on ELF;
  assembler alignment nops set aside); every instruction in an inline-asm
  region must be an in-place `fadd vD.4s, vD.4s, vM.4s`, and the bound
  object must hold that same instruction (same D and M) at the same index.
  Backend-native tile programs hold no site; every GPU-aligned tile program
  holds exactly its committed number of sites
  (CPUNativeTileOrderedAddSites.json, per identity and envelope, hashed in
  the proof), so losing or adding one site is a reviewed diff. The per-program
  site indices are recorded and re-verified. The instruction is an ordinary admitted form (`fadd
  v,v,v`) for the register and spill analysis, which is unchanged.
- Driver plan: the expanded jobs (-###) of the exact compile are the input
  authority. Every argument of every job is admitted by its type (flag,
  checked value, include search root, closure-bound file, or one of the
  designated source/object/dependency/assembly/directory paths); unknown
  options, positionals and values fail - e.g. clang -remap-file, VFS
  overlays, profile inputs, plugins or an extra assembler input. The plan is
  one integrated clang -cc1 -emit-obj job, or one GNU cc1plus job plus one as
  job reading only the compiler's output; a driver that reads specs or
  configuration files fails, as does any failing driver query.
- Programs: the driver is the absolute program of the authenticated target
  command; every job program is identified logically - absolute, or for GNU
  looked up as the driver does (its program directories from
  -print-search-dirs, then PATH), cross-checked against the driver's
  -print-prog-name answer - with the directories examined before the
  selection, the symlinks followed, the resolved file and its hash. Search
  lists with empty or relative entries (working-directory dependent) fail.
- Epoch: the driver identity is bound before any observation the proof
  consumes; the plan and all job programs are bound before preprocessing; the
  identities must be unchanged after preprocessing and after compiling, and
  the plan is observed again after compiling and must be identical.
- The receipt records the plan and exactly one identity per program role;
  verification revalidates the plan with the same pure validator, derives the
  driver role from the authenticated target command and replays every lookup
  (content, symlink retargets, search shadowing).

Verify phase (Unit/preflight): no compiler is invoked, but every recorded input
(including absolute system/SDK headers named by the compiler) must be present
and unchanged, and the bound object is re-disassembled by the bound disassembler
(hashed) and must reproduce the retained disassembly; verification is therefore
tied to the build host's toolchain and binutils, not an SDK-independent or
installed-runner qualification. The actual target command must match and the
object must re-audit to the recorded counts. Probe paths must be absolute.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import time
import threading
import sys
import signal
import ctypes
import hashlib
import importlib.util
import json
import os
import posixpath
import pickle
from pathlib import Path
import re
from types import SimpleNamespace
from typing import NamedTuple
import shlex
import subprocess

_ENVELOPE = Path(__file__).resolve().with_name("CPUCodegenEnvelope.py")
_SPEC = importlib.util.spec_from_file_location("cpu_codegen_envelope", _ENVELOPE)
assert _SPEC and _SPEC.loader
envelope = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(envelope)

# A tile K-loop program run<Ops, Policy, Encoding, Map, WithMinimum, Strips>(): the one outlined program per
# instantiation, containing the strip walk and, for a panel tile, its decode panels. Production instantiates
# Strips = 0 (the installed width). Groups: rows, family, policy, encoding, map, minimum, strips.
PROGRAM = re.compile(r"27CPUNativeNEONProjectionTileILm(\d)EE3runINS1_11multi_scale17NEONMultiScaleOpsILNS\d+_11NEONByteDotE(\d)EEE"
                     r"LNS_28CPUProjectionNumericalPolicyE(\d)ELNS1_26CPUNativeQuantisedEncodingE(\d)ELNS\d+_9NibbleMapE(\d)ELb(\d)E"
                     r"Li(\d)E")
# Coded CompactMultiScale tile programs (runCoded<Ops, Policy, Codebook, Strips>): the source codebook takes the
# key's map position and the minimum is the one runCoded derives (Codebook == 7).
CODED = re.compile(r"27CPUNativeNEONProjectionTileILm(\d)EE8runCodedINS1_11multi_scale17NEONMultiScaleOpsILNS\d+_11NEONByteDotE"
                   r"(\d)EEELNS_28CPUProjectionNumericalPolicyE(\d)ELi(\d+)ELi(\d)E")
CODED_ENCODING = 3
CODED_DISPATCH = ((7, 1), (11, 0))  # dispatch(): CompactMultiScale codebooks 7 (Q5_K) and 11 (IQ3_S).
INVENTORY = re.compile(r"llaminar-native-neon-tile-probe:([A-Za-z0-9,:]*)")
# (encoding, nibble map, with-minimum) dispatched per policy by Tile::dispatch().
DISPATCH = ((1, 0, 0), (2, 0, 0), (4, 2, 0), (0, 0, 0), (0, 2, 0), (0, 1, 1), (0, 3, 0), (6, 1, 1), (7, 0, 0), (7, 1, 1))
DOTS = {"ASIMD": 0, "DotProd": 1, "I8MM": 2}
# The one never-returning diagnostic a tile may call, and abort. Both must also be proven never to return.
FATAL_DIAGNOSTIC = "_ZN9llaminar23cpu16native_quantised28CPUNativeQuantisedWeightView32rejectOutsideAdmittedPublicationEv"
NORETURN_LEAVES = {"abort"}

LABEL = re.compile(r"^([A-Za-z_.$][\w.$]*):(.*)$")
CONDITIONS = "eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al|nv"
CONDITION = re.compile(rf"^(?:{CONDITIONS})$")
CONDITIONAL_BRANCH = re.compile(rf"^b\.?(?:{CONDITIONS})$")
INDIRECT = re.compile(r"^(?:br|blr)(?:aa|ab|aaz|abz)?$")
RETURN = re.compile(r"^ret(?:aa|ab)?$")
BRANCHES = ("b", "bl", "b.cond", "cbz", "cbnz", "tbz", "tbnz")

# Admitted instruction forms: (mnemonic, operand signature) pairs qualified on the GNU-14 and AppleClang-21
# witnesses (all build types and envelopes) plus the forms the semantics below model explicitly. Operand kinds:
# x/w general registers, b/h/s/d/q scalar FP views, v vector, v[] element, list / list[] register lists (lane),
# mem addressing, imm, shift, cond, sym label, reloc relocated symbol. Any other form is a structural failure.
FORMS = frozenset(line.strip() for line in """
add v,v,v|add w,w,imm|add w,w,w|add x,x,imm|add x,x,imm,shift|add x,x,reloc|add x,x,w,shift|add x,x,x|add x,x,x,shift
addp v,v,v|adds x,x,imm|adrp x,reloc|adrp x,sym|and v,v,v|and w,w,imm|and x,x,imm|asr w,w,imm|b sym|b.cond sym|bic v,imm|bic v,imm,shift
bic v,v,v|bic w,w,w,shift|bics w,w,w|bif v,v,v|bit v,v,v|bfi w,w,imm,imm|bfi x,x,imm,imm|bfxil x,x,imm,imm|bl sym|blr x,reloc|bsl v,v,v
cbnz w,sym|cbnz x,sym|cbz w,sym|cbz x,sym|ccmp w,w,imm,cond|cmeq v,v,imm|cmeq v,v,v|cmhi v,v,v|cmn w,imm|cmtst v,v,v
cmp w,imm|cmp w,w|cmp w,w,shift|cmp x,imm|cmp x,w,shift|cmp x,x|csel w,w,w,cond|csel x,x,x,cond|csneg w,w,w,cond|cset x,cond|dup d,v[]
dup v,v[]|dup v,w|dup v,x|eor v,v,v|eor x,x,x|ext v,v,v,imm|fadd v,v,v|fcsel s,s,s,cond|fcvtl v,v|fcvtl2 v,v|fcvtn v,v|fmla v,v,v|fmov d,d|fmov d,x|fmov s,s
fmov s,w|fmov w,s|fmov x,d|fmul s,s,s|fmul v,v,v|fmul v,v,v[]|fneg v,v|ins v[],v[]|ins v[],x|ld1 list,mem
ld1 list,mem,imm|ld1 list[],mem|ld1 list[],mem,imm|ld1r list,mem|ld1r list,mem,imm|ld2 list,mem|ld2 list[],mem
ld2r list,mem|ldp d,d,mem|ldp d,d,mem,imm|ldp q,q,mem|ldp q,q,mem,imm|ldp s,s,mem|ldp w,w,mem|ldp x,x,mem
ldp x,x,mem,imm|ldpsw x,x,mem|ldr b,mem|ldr d,mem|ldr d,mem,imm|ldr h,mem|ldr q,mem|ldr q,mem,imm|ldr s,mem
ldr w,mem|ldr x,mem|ldrb w,mem|ldrh w,mem|ldrsh w,mem|ldrsw x,mem|ldur d,mem|ldur s,mem|ldur q,mem|ldur w,mem|ldur x,mem|lsl w,w,imm|lsl x,x,imm
lsr x,x,imm|lsr x,x,x|madd x,x,x,x|mov d,v[]|mov v,v|mov w,v[]|mov x,v[]|mov v[],v[]|mov v[],w|mov v[],x|mov w,imm|mov w,w|mov x,imm|mov x,x|movi d,imm|movi v,imm
movi v,imm,shift|movk x,imm,shift|mrs x,sym|mul w,w,w|mul x,x,x|mvn v,v|mvni v,imm|mvni v,imm,shift|neg v,v|neg w,w,shift|negs w,w|nop|orr v,imm|orr v,v,v|orr w,w,imm
orr w,w,w|orr w,w,w,shift|orr x,x,imm|orr x,x,x,shift|ret|saddlp v,v|sbfiz x,x,imm,imm|sbfx x,x,imm,imm|scvtf s,w|scvtf v,v
sdot v,v,v|sdot v,v,v[]|shl v,v,imm|sli v,v,imm|shrn v,v,imm|shll v,v,imm|smaddl x,w,w,x|smmla v,v,v|smull v,v,v|smull x,w,w|smull2 v,v,v|sshll v,v,imm|sshl v,v,v|sri v,v,imm|sshr v,v,imm
st1 list,mem|st1 list,mem,imm|st1 list[],mem|stp d,d,mem|stp q,q,mem|stp w,w,mem|stp x,x,mem|str b,mem|str d,mem|str d,mem,imm|str q,mem,imm
str h,mem|str q,mem|str s,mem|str w,mem|str x,mem|stur d,mem|stur q,mem|stur w,mem|stur x,mem|sub v,v,v|sub w,w,imm|sub w,w,w
sub x,x,imm|sub x,x,imm,shift|sub x,x,x|subs w,w,imm|subs w,w,w|sxth w,w|sxtw x,w|tbl v,list,v|tbnz w,imm,sym|tbnz x,imm,sym|tbz w,imm,sym
uaddw v,v,v|ubfiz w,w,imm,imm|ubfiz x,x,imm,imm|ubfx w,w,imm,imm|ubfx x,x,imm,imm|ucvtf v,v|umaddl x,w,w,x|umov w,v[]|umull x,w,w|ushl v,v,v|ushll v,v,imm|ushr v,v,imm
usra v,v,imm|uxtw x,w|xtn v,v|zip1 v,v,v|zip2 v,v,v
""".replace("\n", "|").split("|") if line.strip()) | frozenset(
    # Forms observed in the production compact multi-scale programs and their reached helpers (reviewer-65
    # collect inventory), each modelled by an existing semantics class of Analysis:
    #   GPR computations (destination redefined, NZCV per FLAG_WRITERS, 32-bit views of stack addresses fail):
    "subs x,x,x|and w,w,w|lsr w,w,imm|add w,w,w,shift"
    #   flag read and write without a destination (FLAG_READERS and FLAG_WRITERS):
    "|ccmp w,imm,imm,cond"
    #   FP/SIMD writes of a whole register or a scalar view (h/s/d zero the upper bits): every one changes the
    #   low 64 bits, so on v8-v15 it is a callee-saved clobber (writes_callee_saved_half):
    "|fadd s,s,s|ushr d,d,imm|saddlv h,v|uzp1 v,v,v|uzp2 v,v,v|mul v,v,v|uxtl v,v|fmov v,imm"
    #   FP lane to GPR move (sign-extended lane bits: FP_DATA taint, FP_TO_GP_MOVES):
    "|smov w,v[]"
    #   scalar FP pair store (footprint 2 x 4 bytes; through a stack address it is FP/SIMD spill traffic, never an
    #   ABI save - ABI_REGISTER admits only whole d/q views):
    "|stp s,s,mem"
    #   scalar binary16 load with an unscaled (negative) offset, the LDUR spelling of `ldr h,mem` (GNU-14 I8MM,
    #   unscheduled ordered/backend Q5_K row-pair programs 2:2:x:3:7:1 and 4:2:x:3:7:1, binary16 header staging):
    #   footprint 2 bytes; stack class from the base register's provenance; an h write zeroes bits 16-127, so on
    #   v8-v15 it is a callee-saved clobber (writes_callee_saved_half) - exactly the semantics of `ldr h,mem`:
    "|ldur h,mem".split("|"))
# Admitted instruction grammar. Anything else is a structural failure.
LOADS = frozenset("ldr ldur ldp ldpsw ldrb ldrh ldrsb ldrsh ldrsw ldurb ldurh ldursb ldursh ldursw "
                  "ld1 ld1r ld2 ld2r ld3 ld3r ld4 ld4r".split())
STORES = frozenset("str stur stp strb strh sturb sturh st1 st2 st3 st4".split())
PREFETCHES = frozenset({"prfm", "prfum"})
# Contiguous whole-register transfers: the only forms an ABI save/restore of d8-d15 may use.
ABI_TRANSFERS = frozenset({"str", "stur", "stp", "ldr", "ldur", "ldp"})
ABI_REGISTER = re.compile(r"^([dq])(8|9|1[0-5])$")
NO_DESTINATION = frozenset({"cmp", "cmn", "ccmp", "ccmn", "tst", "fcmp", "fcmpe"})
# The destination is also an input: only part of it is written.
MERGE_DEFINITIONS = frozenset({"bfi", "bfxil", "movk"})
# csinc/csinv/csneg transform one alternative, so they are admitted only in 32-bit form, where no stack address
# can take part (reading a 32-bit view of one is already fatal).
CONDITIONAL_SELECTS = frozenset({"csel"})
STRUCTURE_ACCESSES = re.compile(r"^(ld|st)([1-4])(r?)$")
# Trusted calls of the fatal diagnostic body (they return; the path must still reach abort()).
DIAGNOSTIC_CALLS = frozenset({"fwrite", "fputs"})

GP64 = re.compile(r"^(?:x([12]?[0-9]|30)|sp|fp|lr|xzr)$")
GP32 = re.compile(r"^(?:w([12]?[0-9]|30)|wsp|wzr)$")
FP = re.compile(r"^([bhsdqv])([12]?[0-9]|3[01])(?:\.(\d*)([bhsd]))?(?:\[\d+\])?$")
IMMEDIATE = re.compile(r"^#?(-?(?:0x[0-9a-fA-F]+|\d+))$")
FLOAT = re.compile(r"^#?-?\d+\.\d+(?:e[+-]?\d+)?$")
SHIFT = re.compile(r"^(lsl|lsr|asr|ror|msl|uxtb|uxth|uxtw|uxtx|sxtb|sxth|sxtw|sxtx)(?:\s*#?(\d+))?$")
RELOCATION = re.compile(r"^(?:#?:[a-z0-9_]+:[\w.$+-]+|[\w.$]+@\w+)$")
INDEX = re.compile(r"^([xw](?:[12]?[0-9]|30))(?:\s*,\s*(?:lsl|uxtw|sxtw|sxtx|uxtx)(?:\s*#?\d+)?)?$")
MEMORY = re.compile(r"^\[\s*(sp|fp|x(?:[12]?[0-9]|30))\s*(?:,\s*(.*?))?\s*\](!?)$")
SYMBOL = re.compile(r"^[A-Za-z_.$][\w.$@]*$")
WIDTH = {"b": 1, "h": 2, "s": 4, "d": 8, "q": 16}
CALLEE_SAVED = range(8, 16)
# Data origins of a general register (State.taint): FP/SIMD register bits moved out unchanged, or bits loaded
# from the stack. A stack store of FP_DATA is an FP/SIMD spill; a bit move of STACK_DATA into an FP/SIMD
# register is an FP/SIMD reload.
FP_DATA, STACK_DATA = "F", "L"
# Instructions that move FP/SIMD register bits unchanged into a general register (UMOV and its mov alias, FMOV).
FP_TO_GP_MOVES = frozenset({"mov", "umov", "smov", "fmov"})
# Instructions that move general register bits unchanged into an FP/SIMD register (INS and its mov alias, DUP,
# FMOV).
GP_TO_FP_MOVES = frozenset({"mov", "ins", "dup", "fmov"})
LANE = re.compile(r"^v(?:[12]?[0-9]|3[01])\.\d*([bhsd])\[(\d+)\]$")
LIST_LANE = re.compile(r"^\{\s*v\d+\.\d*([bhsd]).*\}\[(\d+)\]$")
NOT_STACK = frozenset({"N"})
UNKNOWN_STACK = frozenset({("S", None)})
WIDEN_AFTER = 4
# Callee-saved general registers (AAPCS64): x19-x28, the frame pointer x29 and the link register x30.
GPR_CALLEE_SAVED = tuple(f"x{n}" for n in range(19, 31))
# Argument registers: at entry each holds an incoming value (for a tile program x0 is `this`, the descriptor the
# caller owns). An address that is such a value plus a known constant names caller-owned memory.
INCOMING = tuple(f"x{n}" for n in range(8))
# Stack-traffic classes recorded per program and per natural loop (loop 0: outside every loop):
#   fp          FP/SIMD spills and reloads (the spill count, including data laundered through general registers);
#   scalar      loads/stores of general registers through a stack address, except proven ABI saves/restores of
#               x19-x30 outside every loop;
#   abi_loop    proven ABI saves/restores (d8-d15 or x19-x30) inside a loop: executed per iteration;
#   alias_load  inside a loop, a load from an incoming argument plus a known constant (caller-owned, loop
#               invariant: reloaded every iteration);
#   alias_store a store to an incoming argument plus a known constant, at any level;
#   invariant_load   inside a loop, a non-stack load from the same address every iteration: a constant (immediate,
#               absent or relocated) displacement, no writeback or post-index, and a loop-invariant base register
#               (loop_invariant_registers(): not defined in the loop, or defined only by invariant loads, constants,
#               moves, add/sub of constants and csel of invariant registers, to a fixed point) - provenance-free:
#               pointers loaded from the descriptor, chains of them reloaded every iteration, outer-loop-invariant
#               bases, constant tables - or an alias_load, its incoming-argument subset;
#   invariant_reloc  the invariant_load subset whose displacement is relocated (constant data such as .rodata).
#   tls_call    a TLS descriptor resolver call (an admitted call boundary), at any level.
# Call-boundary classes (call model; every one is charged at the call site's innermost loop, at any level):
#   call_frame  a called helper's frame: its entry-level frame record and callee-saved saves/restores (executed
#               on every call, never exempt), and every stack-protector canary slot store/reload;
#   int_call    a bare `bl` to an audited internal helper (one per call instruction, nested calls included);
#   ext_call    a bare call to an external memcpy / __memcpy_chk;
#   ext_bytes   stack bytes those calls copy (per stack side: source and/or destination);
#   ext_accesses  their access count: ceil(bytes / 8) per stack side (the GPR / unknown-data width; the q-width
#               ceil(bytes / 16) needs memory-content facts the analysis does not keep, so the strict 8-byte
#               unit is used on every tree).
CALL_TRAFFIC = ("call_frame", "int_call", "ext_call", "ext_bytes", "ext_accesses")
TRAFFIC = ("fp", "scalar", "abi_loop", "alias_load", "alias_store", "invariant_load", "invariant_reloc",
           "tls_call") + CALL_TRAFFIC
# Undefined imports proven never to return when bound as unique, undefined, strongly bound imports in the exact
# C/C++ ABI spelling of the object format (as for abort in the fatal diagnostic).
NORETURN_IMPORTS = frozenset({"abort", "__cxa_throw", "_Unwind_Resume", "__stack_chk_fail"})
# External copies with exact accounting rules (constant size, stack sides counted).
MEMCPY_IMPORTS = frozenset({"memcpy", "__memcpy_chk"})
STACK_GUARD = "__stack_chk_guard"
# Canary data origin (State.taint): the stack-protector guard value, loaded through the guard symbol.
CANARY_DATA = "C"
MAX_CALL_DEPTH = 8
# Link-selection records of a weak / global helper definition (CallModel.select()). A PLT call from inside a DSO
# exists only because the symbol is preemptible, so the executed copy is decided by process-wide resolution (the
# executable first, then the DSOs in load order), never by the DSO's own COMDAT winner:
#   direct   the symbol is proven non-preemptible: hidden or protected visibility (checked in the object's own
#            symbol table, caller and copy), or -Bsymbolic / -Bsymbolic-functions / a --dynamic-list excluding it
#            (bound by the link evidence digest the parent's process-binding artifact supplies);
#   process  a process resolution record of the actual executable + DSO set: the lookup scope in order
#            (executable first) and, per object, whether it defines the symbol; the first definer must be the
#            object the audited copy was linked into.
# Anything else ("plt" alone, "interposable", unknown) fails: it can execute an unaudited definition.
NONPREEMPTIBLE_VISIBILITY = frozenset({"hidden", "protected"})
NONPREEMPTIBLE_LINK = frozenset({"Bsymbolic", "Bsymbolic-functions", "dynamic-list"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Caller-saved general registers whose facts a call to an audited helper transfers exactly (call()).
CALL_REGISTERS = tuple(f"x{n}" for n in range(19))


CALL_CLOBBERED = frozenset(f"x{n}" for n in range(19))
# Stack bases: a frame slot can be rewritten inside the loop, so a reload through one never makes a register
# loop invariant (the reload itself is scalar traffic).
FRAME_BASES = frozenset({"sp", "x29"})


def invariant_address(instruction: "Instruction", invariant: set[str]) -> bool:
    """A memory access of the same address on every iteration: a constant (immediate, absent or relocated)
    displacement, no writeback or post-index, and a base register in the loop-invariant set."""
    if instruction.memory is None:
        return False
    operands = instruction.operands
    memory = MEMORY.match(operands[instruction.memory])
    if not memory or memory.group(3) == "!" or operands[instruction.memory + 1:]:
        return False
    offset = memory.group(2)
    if offset and immediate(offset) is None and not RELOCATION.match(offset):
        return False
    return gp(memory.group(1)) in invariant


def invariant_definition(instruction: "Instruction", invariant: set[str]) -> bool:
    """Whether every general register this instruction defines holds the same value on every iteration, given the
    registers already proven invariant: a load from an invariant non-frame address, a constant (mov #imm,
    adrp), a register move, an add/sub of a constant (immediate, shifted immediate or relocated low-12 part), or
    a csel whose alternatives are invariant. Anything else (arithmetic with a variable, an indexed or walked
    load, a merge definition, a call) is not."""
    base, operands = instruction.base, instruction.operands
    if base in LOADS:
        memory = MEMORY.match(operands[instruction.memory]) if instruction.memory is not None else None
        return bool(memory) and gp(memory.group(1)) not in FRAME_BASES and invariant_address(instruction, invariant)
    sources = operands[1:]
    def register(operand: str) -> str | None:
        return gp(operand) or gp_view(operand)
    def stable(operand: str) -> bool:
        name = register(operand)
        return name == "xzr" or name in invariant
    if base == "mov" and len(sources) == 1:
        return immediate(sources[0]) is not None or (register(sources[0]) is not None and stable(sources[0]))
    if base == "adrp" and len(sources) == 1:
        return True
    if base in ("add", "adds", "sub", "subs") and len(sources) in (2, 3) and register(sources[0]):
        constant = immediate(sources[1]) is not None or operand_kind(sources[1]) == "reloc"
        shifted = len(sources) == 2 or (SHIFT.match(sources[2]) is not None and SHIFT.match(sources[2]).group(1) == "lsl")
        return constant and shifted and stable(sources[0])
    if base in CONDITIONAL_SELECTS and len(sources) == 3:
        return all(register(operand) is not None and stable(operand) for operand in sources[:2])
    return False


def loop_invariant_registers(instructions: list["Instruction"]) -> set[str]:
    """General registers holding the same value on every iteration of a loop whose body is these instructions.

    Start from the registers no instruction of the body defines (frame bases excluded) and add a register only
    when every one of its in-loop definitions is an invariant definition over the set proven so far, until a
    fixed point. A register is never assumed invariant to prove itself, so an induction walk (x5 = x5 + 16) or a
    cycle of registers defined only through each other stays variant. (A register also defined outside the loop
    may hold that other value on the first iteration only.)
    """
    definitions: dict[str, list[Instruction]] = {}
    for instruction in instructions:
        for register in defined_registers(instruction):
            definitions.setdefault(register, []).append(instruction)
    every = {f"x{n}" for n in range(31)} | {"sp"}
    invariant = {register for register in every if register not in definitions} - FRAME_BASES
    changed = True
    while changed:
        changed = False
        for register, sites in definitions.items():
            if register in invariant or register in FRAME_BASES:
                continue
            if all(invariant_definition(instruction, invariant) for instruction in sites):
                invariant.add(register)
                changed = True
    return invariant


def defined_registers(instruction: "Instruction") -> set[str]:
    """64-bit general registers an admitted instruction writes (a w destination writes its x register).

    Loads write their general-register data operands, any memory access with writeback or post-index writes its
    base, and every other instruction with a destination writes its first operand when that is a general register.
    """
    registers: set[str] = set()
    base, operands = instruction.base, instruction.operands
    if base == "blr" and len(operands) == 2 and operands[1].startswith(":tlsdesc_call:"):
        return {"x0", "x30"}  # TLS descriptor resolver: x0 (result) and the link register only.
    if base == "bl" or base.startswith("blr"):
        # Only the proven never-returning diagnostic may be called today (any other call is structural), and its
        # block never lies on a cycle. Should a returning call ever be admitted, it clobbers the AAPCS64
        # caller-saved general registers and the link register: never implicitly preserved.
        return CALL_CLOBBERED | {"x30"}
    if instruction.memory is not None:
        memory = MEMORY.match(operands[instruction.memory])
        if memory and (memory.group(3) == "!" or operands[instruction.memory + 1:]):
            registers.add(gp(memory.group(1)))
        if base in LOADS:
            registers.update(gp(operand) or gp_view(operand) for operand in operands[:instruction.memory]
                             if gp(operand) or gp_view(operand))
    elif operands and base not in NO_DESTINATION and base not in BRANCHES and \
            not INDIRECT.match(base) and not RETURN.match(base):
        register = gp(operands[0]) or gp_view(operands[0])
        if register:
            registers.add(register)
    registers.discard("xzr")
    return registers


class Dialect:
    """Syntax of the normalised disassembly the analysis reads (generic AArch64 syntax, ELF-style labels)."""

    def local(self, label: str) -> bool:
        """Local labels (branch targets inside one function) stay inside the enclosing function."""
        return label.startswith(".L")

    def symbol(self, name: str) -> str:
        return name

    def statements(self, line: str) -> list[str]:
        return [line]


GENERIC = Dialect()
ENTRY_SECTION = re.compile(r'^\t\.section \.text\.\S+,"ax"$')

# Disassembly front end. The witness analyses the instructions actually assembled into the probe object, as
# printed by the configured disassembler in generic AArch64 syntax (GNU objdump, or llvm-objdump with
# --triple=aarch64, which does not use Apple's abbreviated printer). Assembler-level controls (macros,
# conditionals, directives) therefore cannot change or hide the audited instructions.
FORMAT_LINE = re.compile(r"^\S.*:\s+file format (\S.*?)\s*$")
SECTION_LINE = re.compile(r"^Disassembly of section (\S+):$")
SYMBOL_LINE = re.compile(r"^([0-9a-f]+) <(\S+)>:$")
RELOCATION_LINE = re.compile(r"^\s+([0-9a-f]+):\s+((?:R_AARCH64|ARM64_RELOC)_\w+)\s+(\S+)\s*$")
INSTRUCTION_LINE = re.compile(r"^\s+([0-9a-f]+):\s+(\S.*?)\s*$")
ADDRESS_OPERAND = re.compile(r"^(?:0x)?([0-9a-f]+)\s+<(\S+)>$")
BRANCH_RELOCATIONS = frozenset({"R_AARCH64_CALL26", "R_AARCH64_JUMP26", "ARM64_RELOC_BRANCH26"})
PAGE_RELOCATIONS = frozenset({"R_AARCH64_ADR_PREL_PG_HI21", "R_AARCH64_ADR_GOT_PAGE", "ARM64_RELOC_PAGE21",
                              "ARM64_RELOC_GOT_LOAD_PAGE21"})
LOW12_RELOCATIONS = frozenset({"R_AARCH64_ADD_ABS_LO12_NC", "R_AARCH64_LDST8_ABS_LO12_NC",
                               "R_AARCH64_LDST16_ABS_LO12_NC", "R_AARCH64_LDST32_ABS_LO12_NC",
                               "R_AARCH64_LDST64_ABS_LO12_NC", "R_AARCH64_LDST128_ABS_LO12_NC",
                               "R_AARCH64_LD64_GOT_LO12_NC", "ARM64_RELOC_PAGEOFF12", "ARM64_RELOC_GOT_LOAD_PAGEOFF12"})
# Instructions that read NZCV, and those that define it (a reader that also writes, e.g. ccmp, reads first).
FLAG_READERS = frozenset({"b.cond", "csel", "csinc", "csinv", "csneg", "cset", "csetm", "cinc", "cinv", "cneg",
                          "fcsel", "ccmp", "ccmn", "fccmp", "fccmpe", "adc", "adcs", "sbc", "sbcs"})
FLAG_WRITERS = frozenset({"cmp", "cmn", "tst", "ccmp", "ccmn", "adds", "subs", "ands", "bics", "negs", "ngcs",
                          "fcmp", "fcmpe", "fccmp", "fccmpe", "adcs", "sbcs"})
TLSDESC_SEQUENCE = ("R_AARCH64_TLSDESC_ADR_PAGE21", "R_AARCH64_TLSDESC_LD64_LO12", "R_AARCH64_TLSDESC_ADD_LO12",
                    "R_AARCH64_TLSDESC_CALL")
OBJECT_FORMATS = {"elf64-littleaarch64": False, "mach-o arm64": True}


def object_format(data: bytes) -> bool:
    """True for a Mach-O arm64 object, False for an ELF64 AArch64 object; anything else fails."""
    if data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1 and int.from_bytes(data[18:20], "little") == 183:
        return False
    if data[:4] == bytes.fromhex("cffaedfe") and int.from_bytes(data[4:8], "little") == 0x0100000C:
        return True
    raise ValueError("witness object is neither an ELF64 AArch64 nor a Mach-O arm64 object")


def object_inventory(data: bytes) -> list[str]:
    """The probe's instantiation inventory strings embedded in the object."""
    marker = b"llaminar-native-neon-tile-probe:"
    found, start = [], data.find(marker)
    while start >= 0:
        end = data.find(b"\0", start)
        found.append(data[start:end if end >= 0 else len(data)].decode("ascii", "replace"))
        start = data.find(marker, start + 1)
    return [entry[len(marker.decode()):] for entry in found]


class Section(NamedTuple):
    name: str
    address: int
    size: int
    executable: bool


class ObjectSymbol(NamedTuple):
    """One symbol-table entry. kind: function, object, notype, section, file, undefined, absolute, common,
    indirect or other; binding: local, global or weak; size is None where the format records none (Mach-O)."""
    name: str
    section: str | None
    value: int
    size: int | None
    kind: str
    binding: str
    visibility: str = "default"  # ELF st_other: default, internal, hidden or protected (Mach-O: default).


class ObjectImage(NamedTuple):
    """Sections, the symbol table (raw linker names) and the relocations of every executable section
    ({section: {address: (type name, raw target symbol, addend)}}), all read from the object."""
    macho: bool
    sections: dict[str, Section]
    symbols: list[ObjectSymbol]
    relocations: dict[str, dict[int, tuple[str, str, int]]]
    # (relocated section, address) -> the executable section whose STT_SECTION symbol (named by its section
    # header index) the ELF relocation there names. Mach-O code relocations are extern only.
    section_relocations: dict[tuple[str, int], str] = {}
    # Exception call-site records per function (raw name): (range start, range end, landing pad) absolute
    # addresses, parsed from the object's .eh_frame FDEs and their .gcc_except_table LSDAs (ELF). Mach-O objects
    # carry none: a landing pad there stays unreachable code and fails.
    landing: dict[str, list[tuple[int, int, int]]] = {}

    @property
    def prefix(self) -> str:
        """The C linker-name prefix of the object format (Mach-O prepends an underscore)."""
        return "_" if self.macho else ""


TLSLE_PAIR = ("R_AARCH64_TLSLE_ADD_TPREL_HI12", "R_AARCH64_TLSLE_ADD_TPREL_LO12_NC")


TLSIE_PAIR = ("R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21", "R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC")


ELF_RELOCATIONS = {541: TLSIE_PAIR[0], 542: TLSIE_PAIR[1], 549: TLSLE_PAIR[0], 551: TLSLE_PAIR[1], 257: "R_AARCH64_ABS64", 275: "R_AARCH64_ADR_PREL_PG_HI21", 277: "R_AARCH64_ADD_ABS_LO12_NC",
                   278: "R_AARCH64_LDST8_ABS_LO12_NC", 279: "R_AARCH64_TSTBR14", 280: "R_AARCH64_CONDBR19",
                   282: "R_AARCH64_JUMP26", 283: "R_AARCH64_CALL26", 284: "R_AARCH64_LDST16_ABS_LO12_NC",
                   285: "R_AARCH64_LDST32_ABS_LO12_NC", 286: "R_AARCH64_LDST64_ABS_LO12_NC",
                   299: "R_AARCH64_LDST128_ABS_LO12_NC", 311: "R_AARCH64_ADR_GOT_PAGE",
                   312: "R_AARCH64_LD64_GOT_LO12_NC", 562: "R_AARCH64_TLSDESC_ADR_PAGE21",
                   563: "R_AARCH64_TLSDESC_LD64_LO12", 564: "R_AARCH64_TLSDESC_ADD_LO12", 569: "R_AARCH64_TLSDESC_CALL"}
MACHO_RELOCATIONS = {0: "ARM64_RELOC_UNSIGNED", 1: "ARM64_RELOC_SUBTRACTOR", 2: "ARM64_RELOC_BRANCH26",
                     3: "ARM64_RELOC_PAGE21", 4: "ARM64_RELOC_PAGEOFF12", 5: "ARM64_RELOC_GOT_LOAD_PAGE21",
                     6: "ARM64_RELOC_GOT_LOAD_PAGEOFF12", 7: "ARM64_RELOC_POINTER_TO_GOT",
                     8: "ARM64_RELOC_TLVP_LOAD_PAGE21", 9: "ARM64_RELOC_TLVP_LOAD_PAGEOFF12", 10: "ARM64_RELOC_ADDEND"}


# Relocations admitted on instructions, and (for Mach-O, from mach-o/arm64/reloc.h) their required
# r_pcrel / r_length (2 = 4 bytes).
CODE_RELOCATIONS = frozenset({*TLSLE_PAIR, *TLSIE_PAIR, "R_AARCH64_CALL26", "R_AARCH64_JUMP26", "R_AARCH64_CONDBR19", "R_AARCH64_TSTBR14",
                              "R_AARCH64_ADR_PREL_PG_HI21", "R_AARCH64_ADR_GOT_PAGE", "R_AARCH64_ADD_ABS_LO12_NC",
                              "R_AARCH64_LDST8_ABS_LO12_NC", "R_AARCH64_LDST16_ABS_LO12_NC",
                              "R_AARCH64_LDST32_ABS_LO12_NC", "R_AARCH64_LDST64_ABS_LO12_NC",
                              "R_AARCH64_LDST128_ABS_LO12_NC", "R_AARCH64_LD64_GOT_LO12_NC",
                              # Admitted only as the four relocations of one exact TLS descriptor call
                              # sequence (tlsdesc_groups()); any other use fails in normalize_disassembly().
                              "R_AARCH64_TLSDESC_ADR_PAGE21", "R_AARCH64_TLSDESC_LD64_LO12",
                              "R_AARCH64_TLSDESC_ADD_LO12", "R_AARCH64_TLSDESC_CALL"})
MACHO_CODE_RELOCATIONS = {"ARM64_RELOC_BRANCH26": (1, 2), "ARM64_RELOC_PAGE21": (1, 2),
                          "ARM64_RELOC_PAGEOFF12": (0, 2), "ARM64_RELOC_GOT_LOAD_PAGE21": (1, 2),
                          "ARM64_RELOC_GOT_LOAD_PAGEOFF12": (0, 2)}


def _add_relocation(table: dict, section: str, address: int, record: tuple[str, str, int]) -> None:
    if address in table.setdefault(section, {}):
        raise ValueError(f"two relocations at {section}+{address:#x}")
    table[section][address] = record


def _field(data: bytes, offset: int, size: int) -> int:
    if offset < 0 or offset + size > len(data):
        raise ValueError("truncated witness object")
    return int.from_bytes(data[offset:offset + size], "little")


def _string(data: bytes, start: int, limit: int) -> str:
    end = data.find(b"\0", start, limit)
    if start < 0 or start >= limit or end < 0:
        raise ValueError("witness object string table entry out of bounds")
    return data[start:end].decode("utf-8", "replace")


# Pure parses of identical inputs are memoised within one process (one proof run parses the same object,
# disassembly and assembly in several witnesses); keys are content digests, so a changed input is parsed afresh.
# The cache never hands out its own objects (M1): read_object() returns a deep copy, the body and assembly parses
# fresh containers per call, so a caller mutating a result can never change another caller's (or a later) parse.
_PARSED: dict[tuple, object] = {}


def _memo(key: tuple, compute):
    if key not in _PARSED:
        if len(_PARSED) > 64:
            _PARSED.clear()
        _PARSED[key] = compute()
    return _PARSED[key]


def read_object(data: bytes) -> ObjectImage:
    """Sections (with executable extents) and the complete symbol table, read from the object itself."""
    return copy.deepcopy(_memo(("object", hashlib.sha256(data).digest()),
                               lambda: _read_macho(data) if object_format(data) else _read_elf(data)))


def image_fingerprint(image: ObjectImage) -> bytes:
    """Content digest of an ObjectImage (its full repr: sections, symbols, relocations, EH records)."""
    return hashlib.sha256(repr(image).encode()).digest()


def _read_elf(data: bytes) -> ObjectImage:
    if _field(data, 0x10, 2) != 1 or _field(data, 0x14, 4) != 1 or data[6] != 1:
        raise ValueError("the witness object is not an ELF relocatable object (ET_REL, version 1)")
    shoff, shentsize, shnum, shstrndx = (_field(data, 0x28, 8), _field(data, 0x3A, 2), _field(data, 0x3C, 2),
                                         _field(data, 0x3E, 2))
    if shentsize != 64 or not 0 < shstrndx < shnum:
        raise ValueError("malformed ELF section header table")
    headers = []
    for index in range(shnum):
        base = shoff + index * 64
        headers.append({"name": _field(data, base, 4), "type": _field(data, base + 4, 4),
                        "flags": _field(data, base + 8, 8), "address": _field(data, base + 16, 8),
                        "offset": _field(data, base + 24, 8), "size": _field(data, base + 32, 8),
                        "link": _field(data, base + 40, 4), "info": _field(data, base + 44, 4),
                        "entsize": _field(data, base + 56, 8)})
    strings = headers[shstrndx]
    names = [_string(data, strings["offset"] + header["name"], strings["offset"] + strings["size"])
             for header in headers]
    sections: dict[str, Section] = {}
    for index, header in enumerate(headers[1:], start=1):
        executable = bool(header["flags"] & 0x4)
        if names[index] in sections and (executable or sections[names[index]].executable):
            raise ValueError(f"ambiguous executable section name {names[index]}")
        sections.setdefault(names[index], Section(names[index], header["address"], header["size"], executable))
    tables = [header for header in headers if header["type"] == 2]
    if len(tables) != 1 or tables[0]["entsize"] != 24:
        raise ValueError("the witness object must have exactly one ELF symbol table")
    table, symbols = tables[0], []
    names_table = headers[table["link"]]
    kinds = {0: "notype", 1: "object", 2: "function", 3: "section", 4: "file"}
    indexed: list[str] = [""]
    section_symbols: dict[int, str] = {}  # Symbol table index -> executable section, for STT_SECTION symbols.
    placed: list[tuple[int, int]] = [(0, 0)]  # Symbol table index -> (defining section index, value).
    for entry in range(1, table["size"] // 24):
        base = table["offset"] + entry * 24
        name = _string(data, names_table["offset"] + _field(data, base, 4),
                       names_table["offset"] + names_table["size"])
        if _field(data, base + 4, 1) & 0xF == 3 and 0 < _field(data, base + 6, 2) < len(headers):
            name = names[_field(data, base + 6, 2)]  # A section symbol is named by its section.
        indexed.append(name)
        info, index = _field(data, base + 4, 1), _field(data, base + 6, 2)
        placed.append((index if 0 < index < len(headers) else 0, _field(data, base + 8, 8)))
        if info & 0xF == 3 and 0 < index < len(headers) and headers[index]["flags"] & 0x4:
            section_symbols[entry] = names[index]
        binding = {0: "local", 1: "global", 2: "weak"}.get(info >> 4, "other")
        if index == 0:
            kind, section = "undefined", None
        elif index == 0xFFF1:
            kind, section = "absolute", None
        elif index == 0xFFF2:
            kind, section = "common", None
        elif index < len(headers):
            kind, section = kinds.get(info & 0xF, "other"), names[index]
        else:
            raise ValueError(f"symbol {name} names a nonexistent section")
        visibility = ("default", "internal", "hidden", "protected")[_field(data, base + 5, 1) & 3]
        symbols.append(ObjectSymbol(name, section, _field(data, base + 8, 8), _field(data, base + 16, 8), kind, binding,
                                    visibility))
    relocations: dict[str, dict[int, tuple[str, str, int]]] = {}
    section_relocations: dict[tuple[str, int], str] = {}
    eh_relocations: dict[int, tuple[int, int]] | None = None
    for number, header in enumerate(headers):
        if header["type"] not in (4, 9):
            continue
        if not 0 < header["info"] < len(headers) or not 0 < header["link"] < len(headers):
            raise ValueError(f"relocation section {names[number]} has an invalid section reference")
        target = headers[header["info"]]
        if not target["flags"] & 0x4:
            if names[header["info"]] == ".eh_frame":
                eh_relocations = _eh_relocations(data, header, headers, placed)
            continue  # Relocations of other data sections are not analysed.
        if header["type"] == 9 or header["entsize"] != 24 or headers[header["link"]] is not table:
            raise ValueError("executable-section relocations must be RELA entries against the symbol table")
        if header["size"] % 24 or header["offset"] + header["size"] > len(data):
            raise ValueError(f"relocation section {names[number]} holds a partial or truncated record")
        section = names[header["info"]]
        for entry in range(header["size"] // 24):
            base = header["offset"] + entry * 24
            offset, info = _field(data, base, 8), _field(data, base + 8, 8)
            if not 0 < info >> 32 < len(indexed):
                raise ValueError(f"relocation in {section} names no symbol")
            kind = ELF_RELOCATIONS.get(info & 0xFFFFFFFF)
            if kind not in CODE_RELOCATIONS:
                raise ValueError(f"relocation type {info & 0xFFFFFFFF} is not admitted in code ({section})")
            if offset % 4 or offset >= target["size"]:
                raise ValueError(f"relocation offset {offset:#x} is not an instruction of {section}")
            addend = _field(data, base + 16, 8)
            addend -= (1 << 64) if addend >= 1 << 63 else 0
            _add_relocation(relocations, section, target["address"] + offset, (kind, indexed[info >> 32], addend))
            if info >> 32 in section_symbols:
                section_relocations[(section, target["address"] + offset)] = section_symbols[info >> 32]
    landing: dict[str, list[tuple[int, int, int]]] = {}
    frames = [number for number, name in enumerate(names) if name == ".eh_frame"]
    if len(frames) > 1:
        raise ValueError("the witness object has more than one .eh_frame section")
    if frames:
        frame = headers[frames[0]]
        for text, start, length, table, at in parse_eh_frame(data[frame["offset"]:frame["offset"] + frame["size"]],
                                                            eh_relocations or {}):
            if not headers[text]["flags"] & 0x4 or table is None:
                continue
            names_at = [symbol.name for symbol in symbols if symbol.kind == "function" and
                        symbol.section == names[text] and symbol.value == start]
            if not names_at:
                raise ValueError(f"an FDE with an LSDA covers no function start ({names[text]}+{start:#x})")
            lsda = headers[table]
            ranges = parse_lsda(data[lsda["offset"]:lsda["offset"] + lsda["size"]], at, length)
            for name in names_at:
                landing[name] = [(start + low, start + high, start + pad) for low, high, pad in ranges]
    return ObjectImage(False, sections, symbols, relocations, section_relocations, landing)


def _eh_relocations(data: bytes, header: dict, headers: list[dict], placed: list[tuple[int, int]]) \
        -> dict[int, tuple[int, int]]:
    """.eh_frame relocations: {field offset: (target section index, target offset)}. Only R_AARCH64_PREL32
    (the pcrel|sdata4 FDE and LSDA pointers, and the personality reference) is admitted."""
    if header["type"] != 4 or header["entsize"] != 24 or header["size"] % 24:
        raise ValueError(".eh_frame relocations must be complete RELA entries")
    result = {}
    for entry in range(header["size"] // 24):
        base = header["offset"] + entry * 24
        offset, info = _field(data, base, 8), _field(data, base + 8, 8)
        addend = _field(data, base + 16, 8)
        addend -= (1 << 64) if addend >= 1 << 63 else 0
        if info & 0xFFFFFFFF != 261 or not 0 < info >> 32 < len(placed):
            raise ValueError(f".eh_frame relocation type {info & 0xFFFFFFFF} is not R_AARCH64_PREL32")
        section, value = placed[info >> 32]
        if offset in result:
            raise ValueError(f"two .eh_frame relocations at {offset:#x}")
        result[offset] = (section, value + addend)
    return result


def _uleb(data: bytes, offset: int) -> tuple[int, int]:
    value, shift = 0, 0
    while True:
        byte = _field(data, offset, 1)
        value |= (byte & 0x7F) << shift
        offset, shift = offset + 1, shift + 7
        if not byte & 0x80:
            return value, offset


def _sleb(data: bytes, offset: int) -> tuple[int, int]:
    start = offset
    value, offset = _uleb(data, offset)
    bits = 7 * (offset - start)
    return (value - (1 << bits) if value & (1 << (bits - 1)) else value), offset


PCREL_SDATA4 = 0x1B


def parse_eh_frame(data: bytes, relocations: dict[int, tuple[int, int]]) \
        -> list[tuple[int, int, int, int | None, int]]:
    """FDEs of a relocatable object's .eh_frame: (code section index, function start, pc range, LSDA section
    index or None, LSDA offset). Pointers are the admitted pcrel|sdata4 encoding and must carry their PREL32
    relocation (the field bytes of a relocatable object hold no address). CIE augmentations other than z, P, L,
    R, S and B, 64-bit records, and any other pointer encoding fail."""
    cies: dict[int, tuple[str, int | None, int | None]] = {}
    fdes, offset = [], 0
    while offset < len(data):
        length = _field(data, offset, 4)
        if length == 0:
            offset += 4
            continue
        if length == 0xFFFFFFFF:
            raise ValueError("64-bit .eh_frame records are not admitted")
        start, end = offset + 4, offset + 4 + length
        if end > len(data):
            raise ValueError("truncated .eh_frame record")
        identifier = _field(data, start, 4)
        if identifier == 0:
            version, position = _field(data, start + 4, 1), start + 5
            terminator = data.find(b"\0", position, end)
            if version not in (1, 3) or terminator < 0:
                raise ValueError("malformed .eh_frame CIE")
            augmentation, position = data[position:terminator].decode("ascii", "replace"), terminator + 1
            _, position = _uleb(data, position)
            _, position = _sleb(data, position)
            position = position + 1 if version == 1 else _uleb(data, position)[1]
            lsda, pointer = None, None
            if augmentation:
                if augmentation[0] != "z":
                    raise ValueError(f"unsupported CIE augmentation {augmentation!r}")
                _, position = _uleb(data, position)
                for letter in augmentation[1:]:
                    if letter == "P":
                        encoding = _field(data, position, 1)
                        if encoding & 0x0F not in (0x03, 0x0B):
                            raise ValueError(f"unsupported personality encoding {encoding:#x}")
                        position += 5
                    elif letter == "L":
                        lsda, position = _field(data, position, 1), position + 1
                    elif letter == "R":
                        pointer, position = _field(data, position, 1), position + 1
                    elif letter not in "SB":
                        raise ValueError(f"unsupported CIE augmentation {augmentation!r}")
            cies[offset] = (augmentation, lsda, pointer)
        else:
            cie = cies.get(start - identifier)
            if cie is None:
                raise ValueError(f"FDE at {offset:#x} names no preceding CIE")
            augmentation, lsda, pointer = cie
            if pointer != PCREL_SDATA4 or start + 4 not in relocations:
                raise ValueError(f"FDE at {offset:#x} has no relocated pcrel|sdata4 initial location")
            section, function = relocations[start + 4]
            span, position = _field(data, start + 8, 4), start + 12
            table, at = None, 0
            if augmentation.startswith("z"):
                size, position = _uleb(data, position)
                if "L" in augmentation and lsda != 0xFF:
                    if lsda != PCREL_SDATA4 or size != 4 or position not in relocations:
                        raise ValueError(f"FDE at {offset:#x} has no relocated pcrel|sdata4 LSDA pointer")
                    table, at = relocations[position]
            fdes.append((section, function, span, table, at))
        offset = end
    return fdes


def parse_lsda(data: bytes, offset: int, size: int) -> list[tuple[int, int, int]]:
    """Call-site records with a landing pad of one LSDA: (range start, range end, pad), function relative.

    LPStart must be omitted (pads are relative to the function start); call-site entries are uleb128 or udata4;
    every range and pad must lie inside the function's extent."""
    if _field(data, offset, 1) != 0xFF:
        raise ValueError("LSDA with an explicit landing-pad base is not admitted")
    position = offset + 1
    if _field(data, position, 1) != 0xFF:
        _, position = _uleb(data, position + 1)
    else:
        position += 1
    encoding = _field(data, position, 1)
    if encoding not in (0x01, 0x03):
        raise ValueError(f"LSDA call-site encoding {encoding:#x} is not admitted")
    length, position = _uleb(data, position + 1)
    end, result = position + length, []
    while position < end:
        fields = []
        for _ in range(3):
            if encoding == 0x01:
                value, position = _uleb(data, position)
            else:
                value, position = _field(data, position, 4), position + 4
            fields.append(value)
        _, position = _uleb(data, position)
        start, span, pad = fields
        if start + span > size or pad >= size or start % 4 or span % 4 or pad % 4:
            raise ValueError("LSDA call-site record outside its function")
        if pad:
            result.append((start, start + span, pad))
    if position != end:
        raise ValueError("LSDA call-site table length does not match its records")
    return result


def _read_macho(data: bytes) -> ObjectImage:
    if _field(data, 12, 4) != 1:
        raise ValueError("the witness object is not a Mach-O relocatable object (MH_OBJECT)")
    commands, offset = _field(data, 16, 4), 32
    sections: dict[str, Section] = {}
    ordered: list[str | None] = [None]
    relocation_tables: dict[str, tuple[int, int]] = {}
    symtab = None
    for _ in range(commands):
        command, size = _field(data, offset, 4), _field(data, offset + 4, 4)
        if size < 8:
            raise ValueError("malformed Mach-O load command")
        if command == 0x19:  # LC_SEGMENT_64
            for index in range(_field(data, offset + 64, 4)):
                base = offset + 72 + index * 80
                section = data[base:base + 16].rstrip(b"\0").decode()
                segment = data[base + 16:base + 32].rstrip(b"\0").decode()
                name = f"{segment},{section}"
                flags = _field(data, base + 64, 4)
                if name in sections:
                    raise ValueError(f"ambiguous Mach-O section name {name}")
                sections[name] = Section(name, _field(data, base + 32, 8), _field(data, base + 40, 8),
                                         bool(flags & 0x80000400))
                relocation_tables[name] = (_field(data, base + 56, 4), _field(data, base + 60, 4))
                ordered.append(name)
        elif command == 0x2:  # LC_SYMTAB
            symtab = tuple(_field(data, offset + 8 + 4 * field, 4) for field in range(4))
        elif command == 0x29 and _field(data, offset + 12, 4):  # LC_DATA_IN_CODE with entries
            raise ValueError("the witness object marks data inside code")
        offset += size
    if symtab is None:
        raise ValueError("the witness object has no Mach-O symbol table")
    symoff, count, stroff, strsize = symtab
    symbols, indexed = [], []
    for index in range(count):
        base = symoff + index * 16
        name = _string(data, stroff + _field(data, base, 4), stroff + strsize)
        indexed.append(name)
        kind_bits, number, description = _field(data, base + 4, 1), _field(data, base + 5, 1), _field(data, base + 6, 2)
        value = _field(data, base + 8, 8)
        if kind_bits & 0xE0:
            continue  # Debugger (stab) entry.
        section = None
        if kind_bits & 0x0E == 0x0:
            kind = "common" if value else "undefined"
        elif kind_bits & 0x0E == 0xE:
            if not 0 < number < len(ordered):
                raise ValueError(f"symbol {name} names a nonexistent section")
            section = ordered[number]
            kind = "function" if sections[section].executable else "object"
        else:
            kind = {0x2: "absolute", 0xA: "indirect"}.get(kind_bits & 0x0E, "other")
        binding = "weak" if description & 0xC0 else ("global" if kind_bits & 0x01 else "local")
        symbols.append(ObjectSymbol(name, section, value, None, kind, binding))
    relocations: dict[str, dict[int, tuple[str, str, int]]] = {}
    for name, (offset, count) in relocation_tables.items():
        if not sections[name].executable:
            continue
        for entry in range(count):
            address, info = _field(data, offset + entry * 8, 4), _field(data, offset + entry * 8 + 4, 4)
            symbol, pcrel, length = info & 0xFFFFFF, (info >> 24) & 1, (info >> 25) & 3
            external, kind = (info >> 27) & 1, info >> 28
            name_of = MACHO_RELOCATIONS.get(kind)
            if address & 0x80000000 or not external or not symbol < len(indexed):
                raise ValueError(f"unsupported relocation form in {name} (scattered or section-relative)")
            if name_of not in MACHO_CODE_RELOCATIONS or (pcrel, length) != MACHO_CODE_RELOCATIONS[name_of]:
                raise ValueError(f"relocation {name_of or kind} with pcrel={pcrel} length={length} is not admitted "
                                 f"in code ({name})")
            if address % 4 or address >= sections[name].size:
                raise ValueError(f"relocation address {address:#x} is not an instruction of {name}")
            _add_relocation(relocations, name, sections[name].address + address, (name_of, indexed[symbol], 0))
    return ObjectImage(True, sections, symbols, relocations)


def function_extents(image: ObjectImage) -> dict[str, tuple[str, int, int]]:
    """(section, start, end) of every function symbol defined in an executable section.

    ELF extents are the recorded symbol sizes; Mach-O records none, so a symbol extends to the next distinct
    defined address in its section (or the section end). Extents must lie inside their section; distinct
    function extents in one section must not overlap; a name may be defined only once.
    """
    extents: dict[str, tuple[str, int, int]] = {}
    for symbol in image.symbols:
        # ELF notype labels (mapping symbols) and section symbols are not functions.
        if symbol.section is None or not image.sections[symbol.section].executable or symbol.kind != "function":
            continue
        section = image.sections[symbol.section]
        if image.macho:
            later = sorted({other.value for other in image.symbols if other.section == symbol.section
                            and other.value > symbol.value})
            end = later[0] if later else section.address + section.size
        else:
            if not symbol.size:
                raise ValueError(f"function symbol {symbol.name} records no size")
            end = symbol.value + symbol.size
        if symbol.value % 4 or end % 4:
            raise ValueError(f"function symbol {symbol.name} has an unaligned start or extent")
        if not section.address <= symbol.value < end <= section.address + section.size:
            raise ValueError(f"function symbol {symbol.name} lies outside its section")
        if symbol.name in extents:
            raise ValueError(f"symbol defined twice in the witness: {symbol.name}")
        extents[symbol.name] = (symbol.section, symbol.value, end)
    spans = sorted(set(extents.values()))
    for (section, start, end), (other, other_start, other_end) in zip(spans, spans[1:]):
        if section == other and other_start < end:
            raise ValueError(f"function extents overlap in {section} at {other_start:#x}")
    return extents


def trusted_imports(image: ObjectImage) -> set[str]:
    """Raw linker names of the stdio/abort exceptions bound as the object's unique, undefined, strongly bound
    imports, in the exact spelling of the object format's C ABI (abort on ELF, _abort on Mach-O)."""
    names = [symbol.name for symbol in image.symbols]
    wanted = {image.prefix + name for name in DIAGNOSTIC_CALLS | NORETURN_LEAVES}
    return {symbol.name for symbol in image.symbols if symbol.name in wanted and symbol.kind == "undefined"
            and symbol.binding == "global" and names.count(symbol.name) == 1}


def parse_disassembly(disassembly: str, macho: bool) -> tuple[dict[str, dict[int, list]], list[tuple[str, int, str]]]:
    """Instruction map per section {address: [text, printed relocation]} and (section, address, raw name)
    headings."""
    def symbol(name: str) -> str:
        return name

    lines = disassembly.splitlines()
    formats = [FORMAT_LINE.match(line).group(1) for line in lines if FORMAT_LINE.match(line)]
    if len(formats) != 1 or OBJECT_FORMATS.get(formats[0]) != macho:
        raise ValueError(f"disassembly object format {formats} does not match the witness object")
    code: dict[str, dict[int, list]] = {}
    headings: list[tuple[str, int, str]] = []
    section, last = None, None
    for line in lines:
        if not line.strip() or FORMAT_LINE.match(line):
            continue
        if SECTION_LINE.match(line):
            section, last = SECTION_LINE.match(line).group(1), None
            if section in code:
                raise ValueError(f"section disassembled twice: {section}")
            code[section] = {}
            continue
        if SYMBOL_LINE.match(line):
            if section is None:
                raise ValueError(f"disassembly symbol outside a section: {line}")
            address, name = SYMBOL_LINE.match(line).groups()
            headings.append((section, int(address, 16), symbol(name)))
            continue
        relocation = RELOCATION_LINE.match(line)
        if relocation:
            if last is None or last != int(relocation.group(1), 16):
                raise ValueError(f"relocation does not follow its instruction: {line.strip()}")
            if code[section][last][1] is not None:
                raise ValueError(f"instruction carries two relocations: {line.strip()}")
            code[section][last][1] = (relocation.group(2), symbol(relocation.group(3)))
            continue
        instruction = INSTRUCTION_LINE.match(line)
        if instruction and section is not None and headings and headings[-1][0] == section:
            address = int(instruction.group(1), 16)
            if last is not None and address <= last:
                raise ValueError(f"disassembly addresses out of order in {section} at {address:#x}")
            code[section][address] = [instruction.group(2).split("//", 1)[0].strip(), None]
            last = address
            continue
        raise ValueError(f"unparseable disassembly line: {line.strip()}")
    return code, headings


def relocation_target(symbol: str, addend: int) -> str:
    """The relocation target as disassemblers print it: symbol, symbol+0xN or symbol-0xN."""
    return symbol if not addend else f"{symbol}{'+' if addend > 0 else '-'}0x{abs(addend):x}"


class Bodies(dict):
    """Normalised function bodies {raw name: lines}, with the exception edges of each function: eh = {raw name:
    {instruction index of a call inside an LSDA call-site range: landing-pad label}}."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eh: dict[str, dict[int, str]] = {}


def normalize_disassembly(disassembly: str, image: ObjectImage) -> dict[str, list[str]]:
    """Memoised _normalize_disassembly(), keyed by the disassembly's and the image's content; every call gets fresh
    containers (bodies are lists of immutable strings; the exception edges fresh dicts)."""
    cached = _memo(("normalized", hashlib.sha256(disassembly.encode()).digest(), image_fingerprint(image)),
                   lambda: _normalize_disassembly(disassembly, copy.deepcopy(image)))
    bodies = Bodies({name: list(body) for name, body in cached.items()})
    bodies.eh = {name: dict(edges) for name, edges in cached.eh.items()}
    return bodies


def _normalize_disassembly(disassembly: str, image: ObjectImage) -> dict[str, list[str]]:
    """Translate disassembler output into one normalised body per function symbol of the object.

    Coverage is complete and bound to the object: every executable section is disassembled at every 4-byte
    address of its extent, in order, exactly once; every heading names a symbol defined at that address; and
    each function body is cut from its object symbol extent, not from the printed headings. Branch targets
    inside the function become local labels; relocated branches name their relocation symbol; page/low-12
    relocations become symbolic operands; ELF data mapping symbols inside a function mark data in code.
    """
    code, headings = parse_disassembly(disassembly, image.macho)
    for name, section in image.sections.items():
        if not section.executable:
            if name in code:
                raise ValueError(f"disassembly covers a non-executable section: {name}")
            continue
        if section.size % 4 or section.address % 4:
            raise ValueError(f"executable section {name} is not instruction aligned")
        expected = list(range(section.address, section.address + section.size, 4))
        if sorted(code.get(name, {})) != expected:
            missing = sorted(set(expected) - set(code.get(name, {})))
            raise ValueError(f"disassembly does not cover executable section {name} exactly "
                             f"(missing {[hex(address) for address in missing[:3]]})")
    for name in code:
        if name not in image.sections:
            raise ValueError(f"disassembly names a section the object does not define: {name}")
    defined = {(symbol.section, symbol.value, symbol.name) for symbol in image.symbols if symbol.section}
    for heading in headings:
        if heading not in defined:
            raise ValueError(f"disassembly heading is not a symbol defined at that address: {heading[2]}")
    for name, table in image.relocations.items():
        for address in table:
            if address not in code.get(name, {}):
                raise ValueError(f"object relocation at {name}+{address:#x} is not on a disassembled instruction")
    for name, instructions in code.items():
        table = image.relocations.get(name, {})
        for address, (_, printed) in instructions.items():
            actual = table.get(address)
            if (printed is None) != (actual is None) or \
                    (actual and printed != (actual[0], relocation_target(actual[1], actual[2]))):
                raise ValueError(f"printed relocation at {name}+{address:#x} does not match the object: "
                                 f"{printed} != {actual}")
    extents = function_extents(image)
    tlsdesc = tlsdesc_groups(code, image, extents)
    tlsdesc.update(tlsle_groups(code, image, extents))
    tlsdesc.update(tlsie_groups(code, image, extents))
    owned = {(section, address) for section, start, end in extents.values() for address in range(start, end, 4)}
    for name, table in image.relocations.items():
        orphans = [address for address in table if (name, address) not in owned]
        if orphans:
            raise ValueError(f"relocation outside every function extent in {name} at {orphans[0]:#x}")
    for name, instructions in code.items():
        orphans = [address for address, (text, _) in instructions.items()
                   if (name, address) not in owned and text != "nop"]
        if orphans:
            raise ValueError(f"code outside every function extent in {name} at {orphans[0]:#x}")
    starts: dict[tuple[str, int], str] = {}
    for name, (section, start, _) in sorted(extents.items()):
        starts.setdefault((section, start), name)
    data_marks = {(symbol.section, symbol.value) for symbol in image.symbols
                  if not image.macho and re.match(r"^\$d(?:\.|$)", symbol.name)}
    spans: dict[tuple[str, int], set[tuple[str, int, int]]] = {}
    for extent in extents.values():
        spans.setdefault(extent[:2], set()).add(extent)
    bodies = Bodies()
    for name in image.landing:
        if name not in extents:
            raise ValueError(f"exception call-site records name no function extent: {name}")
    for name, (section, start, end) in extents.items():
        function = {"section": section, "start": start, "spans": spans}
        rewritten, targets = [], set()
        for low, high, pad in image.landing.get(name, []):
            if not start <= low <= high <= end or not start <= pad < end:
                raise ValueError(f"exception call-site record outside {name}")
            targets.add(pad)
            for address in range(low, high, 4):
                if code[section][address][0].split(None, 1)[0] == "bl":
                    bodies.eh.setdefault(name, {})[(address - start) // 4] = f".L{pad:x}"
        for address in range(start, end, 4):
            text = code[section][address][0]
            relocation = image.relocations.get(section, {}).get(address)
            owner = image.section_relocations.get((section, address))
            if (section, address) in tlsdesc:
                rewritten.append((address, tlsdesc[(section, address)]))
                continue
            rewritten.append((address, rewrite(text, relocation, function, end, starts, targets, owner)))
        body = [f'\t.section .text.{section.replace(",", ".")},"ax"']
        for address, text in rewritten:
            if address in targets:
                body.append(f".L{address:x}:")
            if (section, address) in data_marks:
                body.append("\t.data-mapping-symbol")
            body.append(f"\t{text}")
        bodies[name] = body
    return bodies


TLSDESC_TEXT = (re.compile(r"^adrp\s+x0,\s*(?:0x)?[0-9a-f]+\s+<\S+>$"),
                re.compile(r"^ldr\s+x([1-9]|1[0-8]),\s*\[x0(?:,\s*#0(?:x0)?)?\]$"),
                re.compile(r"^add\s+x0,\s*x0,\s*#0(?:x0)?$"),
                re.compile(r"^blr\s+x([1-9]|1[0-8])$"))


def tlsdesc_groups(code: dict, image: ObjectImage, extents: dict) -> dict[tuple[str, int], str]:
    """The exact TLS descriptor call sequences of the object, as normalised instruction text per address.

    Every R_AARCH64_TLSDESC_* relocation must belong to one sequence of four consecutive instructions of one
    function, relocated in order ADR_PAGE21, LD64_LO12, ADD_LO12, CALL against the same symbol with addend 0:
    `adrp x0, sym` / `ldr xR, [x0, #0]` / `add x0, x0, #0` / `blr xR` (R in x1-x18, the same register in the
    load and the call). A stray, reordered, mixed-symbol or partial sequence, a CALL relocation on anything but
    that `blr`, or a mismatched register fails. The normalised call `blr xR, :tlsdesc_call:sym` is the only
    admitted indirect call: under the AArch64 TLS descriptor ABI the resolver clobbers x0 and x30 only (and
    the flags) and preserves every other general and FP/SIMD register. The analysis treats NZCV as undefined
    after the call: a flag reader before any flag writer fails."""
    owner = {}
    for function, (section, start, end) in extents.items():
        for address in range(start, end, 4):
            owner.setdefault((section, address), function)
    result: dict[tuple[str, int], str] = {}
    for section, instructions in code.items():
        for address, (text, _) in instructions.items():
            if "tlsdesc" in text:
                raise ValueError(f"disassembly text names a TLS descriptor without a relocation: {text}")
        table = image.relocations.get(section, {})
        heads = sorted(address for address, record in table.items() if record[0] == TLSDESC_SEQUENCE[0])
        claimed = set()
        for head in heads:
            records = [table.get(head + 4 * step) for step in range(4)]
            if any(record is None or record[0] != kind for record, kind in zip(records, TLSDESC_SEQUENCE)):
                raise ValueError(f"TLS descriptor sequence at {section}+{head:#x} is not ADR_PAGE21, LD64_LO12, "
                                 f"ADD_LO12, CALL on four consecutive instructions")
            if len({record[1] for record in records}) != 1 or any(record[2] for record in records):
                raise ValueError(f"TLS descriptor sequence at {section}+{head:#x} mixes symbols or has an addend")
            if len({owner.get((section, head + 4 * step)) for step in range(4)} - {None}) != 1 or \
                    owner.get((section, head)) is None:
                raise ValueError(f"TLS descriptor sequence at {section}+{head:#x} crosses a function boundary")
            texts = [" ".join(instructions[head + 4 * step][0].split()) for step in range(4)]
            matches = [pattern.match(text) for pattern, text in zip(TLSDESC_TEXT, texts)]
            if not all(matches) or matches[1].group(1) != matches[3].group(1):
                raise ValueError(f"TLS descriptor sequence at {section}+{head:#x} is not the exact adrp x0 / ldr xR, "
                                 f"[x0] / add x0, x0 / blr xR form: {texts}")
            symbol, register = records[0][1], matches[1].group(1)
            result[(section, head)] = f"adrp\tx0, :tlsdesc:{symbol}"
            result[(section, head + 4)] = f"ldr\tx{register}, [x0, :tlsdesc_lo12:{symbol}]"
            result[(section, head + 8)] = f"add\tx0, x0, :tlsdesc_lo12:{symbol}"
            result[(section, head + 12)] = f"blr\tx{register}, :tlsdesc_call:{symbol}"
            claimed.update(head + 4 * step for step in range(4))
        stray = sorted(address for address, record in table.items()
                       if record[0] in TLSDESC_SEQUENCE and address not in claimed)
        if stray:
            raise ValueError(f"TLS descriptor relocation outside an exact sequence at {section}+{stray[0]:#x}")
    return result


def tlsle_groups(code: dict, image: ObjectImage, extents: dict) -> dict[tuple[str, int], str]:
    """Model only the consecutive ELF AArch64 local-exec HI12/LO12_NC ADD pair.

    Both relocations name the same symbol with zero addend, inside one function.
    HI12 is `add xD, xB, #0, lsl #12`; LO12_NC is `add xD, xD, #0`.
    D/B are x0-x30 (never SP/ZR); no other TLSLE forms are admitted. Preserve
    symbolic operands, not numeric zero: normal provenance analysis must still
    reject relocated arithmetic on any stack-derived input. This is arithmetic,
    not a resolver call: it changes only xD, not NZCV or any call-frame counter.
    """
    result = {}
    high = re.compile(r"^add x([0-9]|[12][0-9]|30), x([0-9]|[12][0-9]|30), #0(?:x0)?, lsl #12$")
    low = re.compile(r"^add x([0-9]|[12][0-9]|30), x([0-9]|[12][0-9]|30), #0(?:x0)?$")
    for section, table in image.relocations.items():
        claimed = set()
        for head, record in sorted(table.items()):
            if record[0] != TLSLE_PAIR[0]:
                continue
            tail = table.get(head + 4)
            if image.macho or tail != (TLSLE_PAIR[1], record[1], 0) or record[2]:
                raise ValueError("TLS local-exec pair is incomplete, mixed-symbol or has an addend")
            if not any(sec == section and start <= head and head + 8 <= end
                       for sec, start, end in extents.values()):
                raise ValueError("TLS local-exec pair crosses a function boundary")
            texts = [" ".join(code[section][address][0].split()) for address in (head, head + 4)]
            first, second = high.fullmatch(texts[0]), low.fullmatch(texts[1])
            if not first or not second or first[1] != second[1] or first[1] != second[2]:
                raise ValueError("TLS local-exec pair has unsupported ADD form or register chain")
            dest, base = first[1], first[2]
            result[(section, head)] = f"add\tx{dest}, x{base}, :tprel_hi12:{record[1]}"
            result[(section, head + 4)] = f"add\tx{dest}, x{dest}, :tprel_lo12_nc:{record[1]}"
            claimed.update((head, head + 4))
        if any(record[0] in TLSLE_PAIR and address not in claimed for address, record in table.items()):
            raise ValueError("TLS local-exec relocation outside an exact HI12/LO12_NC pair")
    return result


def tlsie_groups(code: dict, image: ObjectImage, extents: dict) -> dict[tuple[str, int], str]:
    """Admit only an ELF initial-exec GOT page/load pair in one function.

    The consecutive relocations must name one symbol with zero addends. ADRP
    defines a non-stack symbolic GOT page; the 64-bit LDR reads its symbolic
    TP offset, not a stack slot or a resolver. Register aliasing is legal.
    """
    result = {}
    register = r"([0-9]|[12][0-9]|30)"
    page = re.compile(r"^adrp x" + register + r", (?:0x)?[0-9a-f]+ <\S+>$")
    load = re.compile(r"^ldr x" + register + r", \[x" + register + r"(?:, #0(?:x0)?)?\]$")
    for section, table in image.relocations.items():
        claimed = set()
        for head, record in sorted(table.items()):
            if record[0] != TLSIE_PAIR[0]:
                continue
            if image.macho or record[2] or table.get(head + 4) != (TLSIE_PAIR[1], record[1], 0):
                raise ValueError("TLS initial-exec pair is incomplete, mixed-symbol or has an addend")
            if not any(sec == section and start <= head and head + 8 <= end
                       for sec, start, end in extents.values()):
                raise ValueError("TLS initial-exec pair crosses a function boundary")
            texts = [" ".join(code[section][address][0].split()) for address in (head, head + 4)]
            first, second = page.fullmatch(texts[0]), load.fullmatch(texts[1])
            if not first or not second or first[1] != second[2]:
                raise ValueError("TLS initial-exec pair has unsupported ADRP/LDR form or register chain")
            result[(section, head)] = f"adrp\tx{first[1]}, :gottprel:{record[1]}"
            result[(section, head + 4)] = f"ldr\tx{second[1]}, [x{first[1]}, :gottprel_lo12:{record[1]}]"
            claimed.update((head, head + 4))
        if any(record[0] in TLSIE_PAIR and address not in claimed for address, record in table.items()):
            raise ValueError("TLS initial-exec relocation outside an exact PAGE21/LO12_NC pair")
    return result


def rewrite(text: str, relocation: tuple[str, str, int] | None, function: dict, end: int,
            starts: dict[tuple[str, int], str], targets: set[int], owner: str | None = None) -> str:
    """One disassembled instruction in normalised form (symbolic branch targets and relocations).

    owner is the executable section whose STT_SECTION symbol (identified by its section header index, not by
    name) the instruction's ELF relocation names, if any.
    """
    parts = text.split(None, 1)
    mnemonic, operands = parts[0], (split_operands(parts[1]) if len(parts) > 1 else [])
    branch = mnemonic in ("b", "bl", "cbz", "cbnz", "tbz", "tbnz") or CONDITIONAL_BRANCH.match(mnemonic)
    kind, name, addend = relocation if relocation else (None, None, 0)
    if relocation and addend and kind in BRANCH_RELOCATIONS:
        # An ELF call to an internal-linkage function of this object is relocated against its section
        # symbol: the target is that section's address `addend`. It names a function only when a defined
        # function of the object starts exactly there; any other addend (a function symbol plus an offset,
        # an interior or unaligned address, another section) stays unsupported.
        # The callee must be a defined STT_FUNC symbol (the only kind with an extent) starting exactly there;
        # several names are admitted only as aliases of one extent.
        callee = None
        if owner is not None and owner == name and len(function.get("spans", {}).get((owner, addend), ())) == 1:
            callee = starts.get((owner, addend))
        if callee is None:
            raise ValueError(f"branch relocation with an addend is not supported: {text} {name}{addend:+#x}")
        name, addend = callee, 0
    if branch:
        target = ADDRESS_OPERAND.match(operands[-1]) if operands else None
        if not target:
            raise ValueError(f"unparseable branch target: {text}")
        if relocation:
            if kind not in BRANCH_RELOCATIONS or mnemonic not in ("b", "bl"):
                raise ValueError(f"unsupported branch relocation: {text} {kind}")
            operands[-1] = name
        else:
            address = int(target.group(1), 16)
            if function["start"] <= address < end:
                targets.add(address)
                operands[-1] = f".L{address:x}"
            else:
                operands[-1] = starts.get((function["section"], address), f".Loutside{address:x}")
    elif relocation and kind in PAGE_RELOCATIONS and mnemonic == "adrp" and len(operands) == 2 \
            and ADDRESS_OPERAND.match(operands[1]):
        operands[1] = name
    elif relocation and kind in LOW12_RELOCATIONS:
        memory = next((index for index, operand in enumerate(operands) if operand.startswith("[")), None)
        if memory is not None:
            inner = MEMORY.match(operands[memory])
            if not inner or (inner.group(2) and immediate(inner.group(2)) != 0) or inner.group(3):
                raise ValueError(f"unsupported relocated addressing: {text}")
            operands[memory] = f"[{inner.group(1)}, :lo12:{name}]"
        elif mnemonic == "add" and len(operands) == 3 and immediate(operands[2]) == 0:
            operands[2] = f":lo12:{name}"
        else:
            raise ValueError(f"unsupported low-12 relocation: {text}")
    elif relocation:
        raise ValueError(f"unsupported relocation {kind} on {text}")
    elif mnemonic == "adr" and len(operands) == 2 and ADDRESS_OPERAND.match(operands[1]) and \
            function["start"] <= int(ADDRESS_OPERAND.match(operands[1]).group(1), 16) < end:
        # The assembler resolves an ADR of a label of the same function (a jump table's base) without a
        # relocation: the address of this function's own code, named by its local label like a branch target.
        # An unrelocated ADR anywhere else stays unsupported.
        address = int(ADDRESS_OPERAND.match(operands[1]).group(1), 16)
        targets.add(address)
        operands[1] = f".L{address:x}"
    elif any(ADDRESS_OPERAND.match(operand) for operand in operands):
        raise ValueError(f"unrelocated address operand: {text}")
    return f"{mnemonic}\t{', '.join(operands)}" if operands else mnemonic


def split_operands(text: str) -> list[str]:
    """Split instruction operands on top-level commas (not inside [] or {})."""
    operands, depth, current = [], 0, []
    for character in text:
        if character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
        if character == "," and depth == 0:
            operands.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if "".join(current).strip():
        operands.append("".join(current).strip())
    return operands


def gp(operand: str) -> str | None:
    """Canonical 64-bit general register name of an x/sp operand (fp=x29, lr=x30)."""
    if not GP64.match(operand):
        return None
    return {"fp": "x29", "lr": "x30"}.get(operand, operand)


def gp_view(operand: str) -> str | None:
    """The 64-bit register a w operand is the low half of."""
    match = GP32.match(operand)
    if not match:
        return None
    return {"wsp": "sp", "wzr": "xzr"}.get(operand, f"x{match.group(1)}")


def immediate(text: str) -> int | None:
    match = IMMEDIATE.match(text.strip())
    return int(match.group(1), 0) if match else None


def fp_registers(operand: str) -> list[tuple[int, int]] | None:
    """(register number, byte width) of an FP/SIMD operand or register list; None if not FP/SIMD."""
    operand = re.sub(r"\[\d+\]$", "", operand.strip())
    if operand.startswith("{"):
        inner = operand[1:operand.rindex("}")]
        result = []
        for element in (part.strip() for part in inner.split(",")):
            bounds = [part.strip() for part in element.split("-")]
            parsed = [FP.match(bound) for bound in bounds]
            if not all(parsed) or len(parsed) > 2:
                raise ValueError(f"unclassifiable register list: {operand}")
            first, last = int(parsed[0].group(2)), int(parsed[-1].group(2))
            width = _fp_width(parsed[0])
            number = first
            while True:
                result.append((number, width))
                if number == last:
                    break
                number = (number + 1) % 32
        return result
    match = FP.match(operand)
    return [(int(match.group(2)), _fp_width(match))] if match else None


def writes_callee_saved_half(operand: str) -> bool:
    """Whether a write to this FP/SIMD destination operand changes the low 64 bits of its register.

    AAPCS64 preserves only bits 0-63 of v8-v15 (d8-d15). A single-lane write (INS, MOV to a lane, a lane load)
    whose lane lies wholly in bits 64-127 leaves them unchanged; every other write - a whole, 64-bit or scalar
    register, or a lane in the low half - changes them.
    """
    operand = operand.strip()
    lane = LANE.match(operand) or LIST_LANE.match(operand)
    if lane:
        return int(lane.group(2)) * WIDTH[lane.group(1)] < 8
    return True


def _fp_width(match: re.Match) -> int:
    if match.group(1) != "v":
        return WIDTH[match.group(1)]
    if match.group(4):
        return int(match.group(3) or 1) * WIDTH[match.group(4)]
    return 16


WHOLE_X = frozenset({"ldr", "ldur", "ldp", "str", "stur", "stp"})
NARROW = {"b": 1, "h": 2, "sb": 1, "sh": 2, "sw": 4}


def memory_width(mnemonic: str, operand: str) -> int:
    """Bytes a general-register operand of a load/store transfers (sign-extending forms read less)."""
    match = re.match(r"^(?:ldr|ldur|str|stur|ldp|stp)(b|h|sb|sh|sw)?$", mnemonic)
    if match and match.group(1):
        return NARROW[match.group(1)]
    if mnemonic == "ldpsw":
        return 4
    return 8 if gp(operand) else 4


def operand_kind(operand: str) -> str | None:
    """Kind of one operand in the admitted grammar; None if unclassifiable."""
    if gp(operand):
        return "x"
    if GP32.match(operand):
        return "w"
    if operand.startswith("["):
        return "mem"
    if operand.startswith("{"):
        try:
            fp_registers(operand)
        except ValueError:
            return None
        return "list[]" if operand.endswith("]") else "list"
    if re.match(r"^[bhsdq](?:[12]?[0-9]|3[01])$", operand):
        return operand[0]
    if re.match(r"^v(?:[12]?[0-9]|3[01])(?:\.\d+[bhsd])?$", operand):
        return "v"
    if re.match(r"^v(?:[12]?[0-9]|3[01])(?:\.\d*[bhsd])?\[\d+\]$", operand):
        return "v[]"
    if IMMEDIATE.match(operand) or FLOAT.match(operand):
        return "imm"
    if SHIFT.match(operand):
        return "shift"
    if CONDITION.match(operand):
        return "cond"
    if RELOCATION.match(operand):
        return "reloc"
    if SYMBOL.match(operand):
        return "sym"
    return None


class Instruction:
    """One parsed machine instruction."""

    def __init__(self, text: str, dialect: Dialect) -> None:
        self.text = text.strip()
        parts = self.text.split(None, 1)
        mnemonic = parts[0].lower()
        self.operands = split_operands(parts[1]) if len(parts) > 1 else []
        self.base = "b.cond" if CONDITIONAL_BRANCH.match(mnemonic) else mnemonic
        self.memory = next((index for index, operand in enumerate(self.operands) if operand.startswith("[")), None)
        kinds = [operand_kind(operand) for operand in self.operands]
        self.form = None if None in kinds else f"{self.base} {','.join(kinds)}".strip()

    def admitted(self) -> bool:
        return self.form in FORMS


def footprint(instruction: Instruction, operand: str) -> int | None:
    """Bytes of memory one data operand of a load/store transfers; None if not exactly known."""
    base = instruction.base
    scalar = re.match(r"^([bhsdq])\d+$", operand)
    if scalar and base in ("ldr", "ldur", "ldp", "str", "stur", "stp"):
        return WIDTH[scalar.group(1)]
    if gp(operand) or GP32.match(operand):
        return memory_width(base, operand)
    structure = STRUCTURE_ACCESSES.match(base)
    if not structure or not operand.startswith("{"):
        return None
    count, replicate = int(structure.group(2)), bool(structure.group(3))
    lane = re.search(r"\]\s*$", operand) is not None and not operand.rstrip().endswith("}")
    registers = []
    for element in (part.strip() for part in operand[1:operand.rindex("}")].split(",")):
        bounds = [bound.strip() for bound in element.split("-")]
        if len(bounds) == 2:  # A consecutive register range v<a>.T-v<b>.T of one arrangement.
            first, last = (re.match(r"^v(\d+)(\.\w+)?$", bound) for bound in bounds)
            if not first or not last or first.group(2) != last.group(2):
                return None
            span = (int(last.group(1)) - int(first.group(1))) % 32 + 1
            registers += [f"v{(int(first.group(1)) + step) % 32}{first.group(2) or ''}" for step in range(span)]
        elif len(bounds) == 1:
            registers.append(element)
        else:
            return None
    if len(registers) != (count if count > 1 else len(registers)) or not 1 <= len(registers) <= 4:
        return None
    shapes = {re.sub(r"^v\d+", "", element) for element in registers}
    if len(shapes) != 1:
        return None
    shape = shapes.pop().lstrip(".")
    arrangement = re.match(r"^(\d*)([bhsd])$", shape)
    if not arrangement:
        return None
    element = WIDTH[arrangement.group(2)]
    if lane or replicate:
        if arrangement.group(1) and not replicate:
            return None
        return len(registers) * element
    if not arrangement.group(1):
        return None
    return len(registers) * int(arrangement.group(1)) * element


class Block:
    def __init__(self) -> None:
        self.labels: list[str] = []
        self.instructions: list[Instruction] = []
        self.successors: list[int] = []
        self.falls = True


def build_cfg(body: list[str], dialect: Dialect) -> tuple[list[Block], list[str]]:
    """Basic blocks of one normalised function body; data words or directives inside code fail."""
    blocks: list[Block] = [Block()]
    if not body or not ENTRY_SECTION.match(body[0]):
        return blocks, ["function body has no executable entry section"]
    errors: list[str] = []
    for line in body[1:]:
        label = LABEL.match(line)
        if label:
            if blocks[-1].instructions:
                blocks.append(Block())
            blocks[-1].labels.append(label.group(1))
            line = label.group(2)
        for statement in dialect.statements(line):
            statement = statement.strip()
            if not statement:
                continue
            if statement.startswith("."):
                errors.append(f"data or directive inside code: {statement}")
                continue
            instruction = Instruction(statement, dialect)
            blocks[-1].instructions.append(instruction)
            if instruction.base in BRANCHES or INDIRECT.match(instruction.base) or RETURN.match(instruction.base):
                blocks.append(Block())  # Every branch ends its block.
    return blocks, errors


class State:
    """Per-program-point facts: general register stack provenance, pointer slots, ABI saves, data taint.

    pointers maps the offset of each 8-byte frame slot that may hold a stack address to the set of values it
    may hold, including "N" when a non-stack value reaches the slot on some path (may/must information).
    taint maps a general register to the may-set of its data origins: FP_DATA when it may hold FP/SIMD
    register bits moved out unchanged (UMOV/FMOV), STACK_DATA when it may hold bits loaded from the stack.
    """

    def __init__(self, registers: dict[str, frozenset], pointers: dict[int, frozenset],
                 saved: dict[int, frozenset], taint: dict[str, frozenset] | None = None,
                 gsaved: dict[str, frozenset] | None = None, incoming: dict[str, frozenset] | None = None,
                 islots: dict[int, frozenset] | None = None) -> None:
        self.registers, self.pointers, self.saved = registers, pointers, saved
        self.taint = taint if taint is not None else {}
        # ABI status of x19-x30 (U caller value, S saved, D saved and clobbered, R restored, X clobbered unsaved).
        self.gsaved = gsaved if gsaved is not None else {r: frozenset({("U", None)}) for r in GPR_CALLEE_SAVED}
        # Incoming-argument provenance of general registers: ("I", argument, offset or None); "N" otherwise.
        self.incoming = incoming if incoming is not None else {}
        # Exact 8-byte stack slots holding an incoming-argument value.
        self.islots = islots if islots is not None else {}
        # NZCV after a call on some path (a TLS descriptor resolver or any called function does not preserve the
        # flags), until a flag-setting instruction redefines them.
        self.flags_clobbered = False
        # Known constant values of general registers on every path (mov #imm), for exact memcpy sizes.
        self.consts: dict[str, int] = {}
        # Frame offsets of 8-byte slots holding the stack-protector canary on every path.
        self.canary: frozenset = frozenset()

    @staticmethod
    def entry() -> "State":
        """Function entry: sp at offset 0 of the new frame, x29 the caller's frame pointer."""
        registers = {"sp": frozenset({("S", 0)}), "x29": UNKNOWN_STACK}
        incoming = {r: frozenset({("I", r, 0)}) for r in INCOMING}
        return State(registers, {}, {n: frozenset({("U", None)}) for n in CALLEE_SAVED}, incoming=incoming)

    def copy(self) -> "State":
        copied = State(dict(self.registers), dict(self.pointers), dict(self.saved), dict(self.taint),
                       dict(self.gsaved), dict(self.incoming), dict(self.islots))
        copied.flags_clobbered = self.flags_clobbered
        copied.consts, copied.canary = dict(self.consts), self.canary
        return copied

    def tainted(self, register: str | None) -> frozenset:
        return self.taint.get(register, frozenset()) if register and register != "xzr" else frozenset()

    def define(self, register: str | None, origins: frozenset) -> None:
        """The data origins of a general register's new value (empty: a computed value).

        Every definition of a general register passes here: it also clobbers x19-x30 for the ABI model.
        """
        if not register or register == "xzr":
            return
        self.consts.pop(register, None)
        if register in self.gsaved:
            self.gsaved[register] = frozenset(("D", slot) if status in ("S", "D") else ("X", None)
                                              for status, slot in self.gsaved[register])
        if origins:
            self.taint[register] = origins
        else:
            self.taint.pop(register, None)

    def arrival(self, register: str | None) -> frozenset:
        """Incoming-argument provenance of a general register (NOT_STACK: none)."""
        return self.incoming.get(register, NOT_STACK) if register and register != "xzr" else NOT_STACK

    def arrive(self, register: str | None, values: frozenset) -> None:
        if not register or register == "xzr":
            return
        if values == NOT_STACK:
            self.incoming.pop(register, None)
        else:
            self.incoming[register] = widen_incoming(values)

    def value(self, register: str) -> frozenset:
        return self.registers.get(register, NOT_STACK) if register != "xzr" else NOT_STACK

    def assign(self, register: str, values: frozenset) -> None:
        if register == "xzr":
            return
        if values == NOT_STACK:
            self.registers.pop(register, None)
        else:
            self.registers[register] = widen(values)

    def slot(self, offset: int) -> frozenset:
        return self.pointers.get(offset, NOT_STACK)

    def store_slot(self, offset: int, values: frozenset) -> None:
        if values == NOT_STACK:
            self.pointers.pop(offset, None)
        else:
            self.pointers[offset] = widen(values)

    def join(self, other: "State") -> "State":
        registers = {register: widen(self.value(register) | other.value(register))
                     for register in set(self.registers) | set(other.registers)}
        pointers = {offset: widen(self.slot(offset) | other.slot(offset))
                    for offset in set(self.pointers) | set(other.pointers)}
        saved = {n: self.saved[n] | other.saved[n] for n in CALLEE_SAVED}
        taint = {register: self.tainted(register) | other.tainted(register)
                 for register in set(self.taint) | set(other.taint)}
        gsaved = {r: self.gsaved[r] | other.gsaved[r] for r in GPR_CALLEE_SAVED}
        incoming = {register: widen_incoming(self.arrival(register) | other.arrival(register))
                    for register in set(self.incoming) | set(other.incoming)}
        islots = {offset: widen_incoming(self.islots.get(offset, NOT_STACK) | other.islots.get(offset, NOT_STACK))
                  for offset in set(self.islots) | set(other.islots)}
        joined = State(registers, pointers, saved, taint, gsaved, incoming, islots)
        joined.flags_clobbered = self.flags_clobbered or other.flags_clobbered
        joined.consts = {register: value for register, value in self.consts.items()
                         if other.consts.get(register) == value}
        joined.canary = self.canary & other.canary
        return joined

    def key(self) -> tuple:
        return (tuple(sorted((r, tuple(sorted(v, key=repr))) for r, v in self.registers.items())),
                tuple(sorted((o, tuple(sorted(v, key=repr))) for o, v in self.pointers.items())),
                tuple(tuple(sorted(self.saved[n], key=repr)) for n in CALLEE_SAVED),
                tuple(sorted((r, tuple(sorted(v))) for r, v in self.taint.items())),
                tuple(tuple(sorted(self.gsaved[r], key=repr)) for r in GPR_CALLEE_SAVED),
                tuple(sorted((r, tuple(sorted(v, key=repr))) for r, v in self.incoming.items())),
                tuple(sorted((o, tuple(sorted(v, key=repr))) for o, v in self.islots.items())),
                self.flags_clobbered, tuple(sorted(self.consts.items())), tuple(sorted(self.canary)))


def widen(values: frozenset) -> frozenset:
    """Keep the provenance lattice finite: many or unknown stack offsets collapse to one unknown offset."""
    stack = [value for value in values if value != "N"]
    if ("S", None) in values or len(stack) > WIDEN_AFTER:
        return frozenset({value for value in values if value == "N"} | {("S", None)})
    return values


def widen_incoming(values: frozenset) -> frozenset:
    """Finite incoming-provenance lattice: more than WIDEN_AFTER offsets of an argument become an unknown offset."""
    known = [value for value in values if value != "N"]
    if len(known) <= WIDEN_AFTER:
        return values
    return frozenset({value for value in values if value == "N"} | {("I", value[1], None) for value in known})


def invariant_alias(values: frozenset) -> bool:
    """An address that is, on every path, an incoming argument plus a known constant: caller-owned, call invariant."""
    return bool(values) and all(value != "N" and value[2] is not None for value in values)


def provenance(values: frozenset) -> str:
    """N: never a stack address; S: always a stack address; A: path-dependent (ambiguous)."""
    if values == NOT_STACK:
        return "N"
    return "S" if "N" not in values else "A"


def single_offset(values: frozenset) -> int | None:
    return next(iter(values))[1] if len(values) == 1 and "N" not in values else None


def import_symbols(image: ObjectImage) -> frozenset:
    """Raw names of the object's unique, undefined, strongly bound (global) imports."""
    count: dict[str, int] = {}
    for symbol in image.symbols:
        count[symbol.name] = count.get(symbol.name, 0) + 1
    return frozenset(symbol.name for symbol in image.symbols if symbol.kind == "undefined"
                     and symbol.binding == "global" and count[symbol.name] == 1)


# Static executable selection deliberately has no JSON/digest constructor. The native
# transaction below is the only producer; persisted records are diagnostic recipes.
_STATIC_SELECTION_KEY = object()


class TypedStaticExecutableSelection:
    """Ephemeral authority for one audited Unit selected by a complete native link."""

    def __init__(self, key, unit: "Unit", target: str, files: dict[str, str], environment: dict) -> None:
        if key is not _STATIC_SELECTION_KEY:
            raise ValueError("static selection requires the native verifier")
        self.unit, self.target = unit, target
        self.files, self.environment = dict(files), dict(environment)

    def check(self, unit: "Unit", target: str) -> None:
        """Recheck retained bytes and process environment at the call-model boundary."""
        if unit != self.unit or target != self.target:
            raise ValueError("static selected Unit/target changed")
        if dict(os.environ) != self.environment:
            raise ValueError("static link environment changed")
        for path, digest in self.files.items():
            if _static_hash(Path(path).read_bytes()) != digest:
                raise ValueError(f"static input changed: {path}")


def _static_hash(data: bytes) -> str:
    """SHA256 of an actual artifact, not a caller assertion."""
    return hashlib.sha256(data).hexdigest()


def _static_elf(data: bytes) -> tuple[list[dict], list[str]]:
    """Read bounded ELF64 little-endian section records, including non-code inputs."""
    if data[:6] != b"\x7fELF\x02\x01" or _field(data, 58, 2) != 64:
        raise ValueError("static input requires ELF64 little endian")
    count, offset, strings = _field(data, 60, 2), _field(data, 40, 8), _field(data, 62, 2)
    if not 0 < strings < count or offset + count * 64 > len(data):
        raise ValueError("static ELF section bounds")
    headers = []
    for i in range(count):
        b = offset + i * 64
        h = dict(name=_field(data, b, 4), type=_field(data, b+4, 4), flags=_field(data, b+8, 8),
                 address=_field(data, b+16, 8), offset=_field(data, b+24, 8), size=_field(data, b+32, 8),
                 link=_field(data, b+40, 4), entsize=_field(data, b+56, 8))
        if h["type"] != 8 and h["offset"] + h["size"] > len(data):
            raise ValueError("static ELF section payload bounds")
        headers.append(h)
    s = headers[strings]
    return headers, [_string(data, s["offset"] + h["name"], s["offset"] + s["size"]) for h in headers]


def _static_symbols(data: bytes, headers: list[dict], kind: int) -> list[tuple]:
    """Return raw symbol identity including dynamic undefined entries."""
    result = []
    for h in headers:
        if h["type"] != kind:
            continue
        if h["entsize"] != 24 or h["size"] % 24 or not 0 < h["link"] < len(headers):
            raise ValueError("static symbol table bounds")
        s = headers[h["link"]]
        for b in range(h["offset"] + 24, h["offset"] + h["size"], 24):
            result.append((_string(data, s["offset"] + _field(data, b, 4), s["offset"] + s["size"]),
                           _field(data, b+4, 1), _field(data, b+5, 1), _field(data, b+6, 2),
                           _field(data, b+8, 8), _field(data, b+16, 8)))
    return result


def _static_members(data: bytes) -> list[tuple[int, int, bytes, str, bytes]]:
    """Extract regular ar members with ordinal/header-offset/raw-name identity; thin fails."""
    if not data.startswith(b"!<arch>\n"):
        raise ValueError("static archive is not a regular archive")
    offset, ordinal, names, result = 8, 0, b"", []
    while offset < len(data):
        h = data[offset:offset+60]
        if len(h) != 60 or h[58:] != b"`\n":
            raise ValueError("static archive header")
        try:
            size = int(h[48:58])
        except ValueError:
            raise ValueError("static archive size") from None
        body = data[offset+60:offset+60+size]
        if len(body) != size or size < 0:
            raise ValueError("static archive bounds")
        raw = h[:16]
        name = raw.rstrip().decode("ascii")
        if name == "//":
            names = body
        elif name not in ("/", "/SYM64/"):
            if name.startswith("/"):
                n = int(name[1:])
                end = names.find(b"/\n", n)
                if end < n:
                    raise ValueError("static archive long name")
                name = names[n:end].decode("utf-8")
            else:
                name = name.removesuffix("/")
            if name.startswith("#1/"):
                raise ValueError("unsupported BSD archive member")
            result.append((ordinal, offset, raw, name, body))
        offset += 60 + size + size % 2
        ordinal += 1
    if offset != len(data):
        raise ValueError("static archive trailing bytes")
    return result


def _static_response(argv: list[str], cwd: Path, files: dict) -> list[str]:
    """Expand bounded GNU response tokens; preserve every response file as an input."""
    result = []
    for arg in argv:
        if arg.startswith("@"):
            p = (cwd / arg[1:]).resolve(strict=True)
            if str(p) in files:
                raise ValueError("recursive/duplicate static response file")
            data = p.read_bytes()
            files[str(p)] = _static_hash(data)
            result.extend(_static_response(shlex.split(data.decode()), cwd, files))
        else:
            result.append(arg)
    return result


def _static_options(argv: list[str]) -> None:
    """Reject rewriting/selection-changing options even through forwarding and responses."""
    tokens = [token for arg in argv for token in arg.removeprefix("-Wl,").split(",")]
    for t in tokens:
        if (t in ("-r", "--relocatable", "--gc-sections", "-shared", "--shared") or
                t.startswith(("-flto", "--icf", "--plugin-opt", "-fuse-linker-plugin"))):
            raise ValueError(f"unsupported static link option: {t}")


def capture_static_link(argv: list[str], cwd: Path, output: Path, artifacts: Path) -> dict:
    """Capture a bounded GNU14/ld.bfd production edge; diagnostics never mint authority.

    Caller supplies the exact ordered Ninja driver edge, not a shell command. Every
    preparation/relink changes only the mechanically specified tokens. Artifact
    directories must be dedicated: prior authority is retired before any operation.
    All subprocesses inherit the same captured environment, including unset/empty
    distinctions (the complete mapping is retained, not a selected variable subset).
    """
    artifacts.mkdir(parents=True, exist_ok=True)
    proof = artifacts / "selection-recipe.json"
    proof.unlink(missing_ok=True)
    cwd, output, artifacts = cwd.resolve(strict=True), output.resolve(strict=True), artifacts.resolve()
    env, files = dict(os.environ), {}
    args = _static_response(list(argv), cwd, files)
    _static_options(args)
    if args.count("-o") != 1 or args.index("-o") + 1 >= len(args):
        raise ValueError("static edge needs exactly one separate -o output")
    oi = args.index("-o") + 1
    if (cwd / args[oi]).resolve() != output:
        raise ValueError("static output does not match edge")
    if any("--dependency-file" in a or "-Map" in a or "--trace" in a for a in args):
        raise ValueError("production edge already has unsupported diagnostic side outputs")
    driver = Path(args[0]).resolve(strict=True)
    args[0] = str(driver)
    files[str(driver)] = _static_hash(driver.read_bytes())
    production = output.read_bytes()
    (artifacts / "production.elf").write_bytes(production)

    def run(label: str, command: list[str]) -> str:
        """Retain exact argv, environment and both output streams, even on failure."""
        (artifacts / f"{label}.json").write_text(json.dumps(dict(argv=command, cwd=str(cwd), env=env), indent=2))
        r = subprocess.run(command, cwd=cwd, env=env, capture_output=True, timeout=120)
        (artifacts / f"{label}.stdout").write_bytes(r.stdout)
        (artifacts / f"{label}.stderr").write_bytes(r.stderr)
        if r.returncode:
            raise ValueError(f"static {label} failed ({r.returncode})")
        return (r.stdout + r.stderr).decode("utf-8")

    version = run("driver-version", [str(driver), "-dumpfullversion"])
    if not version.startswith("14."):
        raise ValueError("static capture supports GNU14 only")
    jobs = run("driver-jobs", args + ["-###"])
    collect = [shlex.split(line)[0] for line in jobs.splitlines() if line.lstrip().startswith('"/')
               and "collect2" in line]
    if len(collect) != 1:
        raise ValueError("static driver must expose one collect2 job")
    cp = Path(collect[0]).resolve(strict=True)
    files[str(cp)] = _static_hash(cp.read_bytes())
    debug_args = args.copy()
    debug_args[oi] = str(artifacts / "debug.elf")
    debug = run("collect2-debug", debug_args + ["-debug"])
    match = re.findall(r"^ld_file_name\s*=\s*(\S+)\s*$", debug, re.M)
    if len(match) != 1:
        raise ValueError("collect2 did not identify its actual linker")
    linker = Path(match[0]).resolve(strict=True)
    if linker.name != "ld.bfd":
        raise ValueError("static selection requires actual ld.bfd")
    files[str(linker)] = _static_hash(linker.read_bytes())
    if "GNU ld" not in run("linker-version", [str(linker), "--version"]):
        raise ValueError("static linker is not GNU bfd")
    if (artifacts / "debug.elf").read_bytes() != production:
        raise ValueError("collect2 debug preparation changed production bytes")
    default_script = run("default-script", [str(linker), "--verbose"])
    (artifacts / "default-script.txt").write_text(default_script)
    # GNU14 exposes the plugin on the actual job even when there is no LTO work.
    job_tokens = [token for line in jobs.splitlines() if "collect2" in line for token in shlex.split(line)]
    for i, token in enumerate(job_tokens):
        if token in ("-plugin", "--plugin"):
            p = Path(job_tokens[i+1]).resolve(strict=True)
            files[str(p)] = _static_hash(p.read_bytes())
    dep = artifacts / "native.d"
    instrument = args + ["-Wl,--trace", f"-Wl,--dependency-file={dep}"]
    transactions = []
    for label in ("instrumented", "control", "map"):
        command = instrument.copy()
        command[oi] = str(artifacts / f"{label}.elf")
        if label == "map":
            command += [f"-Wl,-Map={artifacts / 'native.map'}"]
        trace = run(label, command)
        if (artifacts / f"{label}.elf").read_bytes() != production:
            raise ValueError(f"static {label} differs from production")
        dependency = dep.read_bytes()
        (artifacts / f"{label}.d").write_bytes(dependency)
        # Restrict grammar rather than guessing make quoting or trace diagnostics.
        text = dependency.decode().replace("\\\n", "")
        first = text.split("\n\n", 1)[0]
        if ":" not in first or "\\" in first:
            raise ValueError("unsupported static dependency grammar")
        paths = first.split(":", 1)[1].split()
        traced = [line.strip() for line in trace.splitlines() if line.strip()]
        def canonical(items):
            return {(cwd / p).resolve(strict=True) for p in items}
        dependencies, traces = canonical(paths), canonical(traced)
        if dependencies != traces or not dependencies:
            raise ValueError("static trace/dependency input set mismatch")
        inventory = {}
        for p in dependencies:
            data = p.read_bytes()
            inventory[str(p)] = _static_hash(data)
            objects = [m[4] for m in _static_members(data)] if data.startswith(b"!<arch>\n") else [data]
            if data.startswith(b"!<thin>"):
                raise ValueError("thin static archive unsupported")
            for obj in objects:
                if obj.startswith(b"\x7fELF"):
                    _, names = _static_elf(obj)
                    if any(n.startswith(".gnu.lto_") for n in names):
                        raise ValueError("GNU LTO input is not static selection evidence")
        if transactions and transactions[0] != inventory:
            raise ValueError("static input changed between transactions")
        transactions.append(inventory)
        files.update(inventory)
    if dict(os.environ) != env:
        raise ValueError("static environment changed during capture")
    if run("default-script-after", [str(linker), "--verbose"]) != default_script:
        raise ValueError("static default script changed")
    files[str(output)] = _static_hash(production)
    for path, digest in files.items():
        if _static_hash(Path(path).read_bytes()) != digest:
            raise ValueError(f"static input changed: {path}")
    for p in artifacts.iterdir():
        if p.is_file() and p != proof:
            files[str(p)] = _static_hash(p.read_bytes())
    recipe = dict(schema=1, argv=argv, cwd=str(cwd), output=str(output), env=env, files=files)
    proof.write_text(json.dumps(recipe, indent=2))
    return recipe


def verify_static_executable_selection(argv: list[str], cwd: Path, output: Path, artifacts: Path,
                                       unit: "Unit", object_bytes: bytes, replay_bytes: bytes,
                                       target: str) -> TypedStaticExecutableSelection:
    """Reconstruct native closure and prove the kept input owns the actual AArch64 callee."""
    recipe = capture_static_link(argv, cwd, output, artifacts)
    # A capture recipe is never itself sufficient: the following proof uses bytes.
    image = read_object(object_bytes)
    if object_bytes != replay_bytes or unit.object_identity != image_fingerprint(image):
        raise ValueError("static selected object differs from audited Unit/B1 replay")
    data = output.read_bytes()
    headers, names = _static_elf(data)
    if _field(data, 18, 2) != 183:
        raise ValueError("static selected executable must be AArch64")
    typ = _field(data, 16, 2)
    po, pe, pn = _field(data, 32, 8), _field(data, 54, 2), _field(data, 56, 2)
    if pe != 56 or po + pe * pn > len(data):
        raise ValueError("static program header bounds")
    interp = any(_field(data, po+i*pe, 4) == 3 for i in range(pn))
    pie = any(_field(data, b, 8) == 0x6ffffffb and _field(data, b+8, 8) & 0x8000000
              for h in headers if h["type"] == 6
              for b in range(h["offset"], h["offset"] + h["size"], 16))
    if typ != 2 and not (typ == 3 and interp and pie):
        raise ValueError("static executable requires ET_EXEC or PIE with PT_INTERP and DF_1_PIE")
    if any(s[0] == target for s in _static_symbols(data, headers, 11)):
        raise ValueError("static selected target appears in dynsym")
    symbols = [s for s in _static_symbols(data, headers, 2) if s[0] == target]
    if len(symbols) != 1:
        raise ValueError("ambiguous final static symbol")
    _, info, visibility, section, address, size = symbols[0]
    if info != 0x22 or visibility != 0 or not 0 < section < len(headers) or size == 0:
        raise ValueError("static final symbol is not unique weak/default STT_FUNC")
    final = headers[section]
    if not final["flags"] & 4 or not final["address"] <= address < address+size <= final["address"]+final["size"]:
        raise ValueError("static final symbol outside executable section")
    originals = [s for s in image.symbols if s.name == target]
    if len(originals) != 1 or originals[0].binding != "weak" or originals[0].visibility != "default":
        raise ValueError("ambiguous static input symbol")
    original = originals[0]
    mapping = (artifacts / "native.map").read_text()
    boundary = "Linker script and memory map"
    if mapping.count(boundary) != 1 or mapping.count("Discarded input sections") != 1:
        raise ValueError("ambiguous GNU map kept/discarded boundary")
    discarded, kept = mapping.split(boundary)
    if discarded.index("Discarded input sections") > len(discarded):
        raise ValueError("static discarded boundary")
    # GNU map input-section lines may wrap once between section name and address.
    intervals = re.findall(r"^ (\.[^\s]+)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+([^\n]+)$", kept, re.M)
    owners = [(n, int(a, 16), int(s, 16), p.strip()) for n,a,s,p in intervals
              if int(a, 16) <= address < address+size <= int(a, 16)+int(s, 16)]
    if len(owners) != 1:
        raise ValueError("static map has ambiguous/missing kept owner")
    name, placed, extent, owner = owners[0]
    member = re.fullmatch(r"(.+)\(([^()]+)\)", owner)
    p = (cwd / (member[1] if member else owner)).resolve(strict=True)
    if str(p) not in recipe["files"]:
        raise ValueError("static map owner outside input closure")
    selected = p.read_bytes()
    if member:
        matches = [m for m in _static_members(selected) if m[3] == member[2]]
        if len(matches) != 1:
            raise ValueError("ambiguous archive member identity")
        selected = matches[0][4]
        (artifacts / "selected-member.json").write_text(json.dumps(dict(
            archive=str(p), ordinal=matches[0][0], header_offset=matches[0][1], raw_name=matches[0][2].hex())))
    if selected != object_bytes or name != original.section or extent != image.sections[name].size:
        raise ValueError("static kept member/section differs from audited object")
    if placed + original.value != address or original.size != size:
        raise ValueError("static input/final symbol extent mismatch")
    files = dict(recipe["files"])
    for p in artifacts.iterdir():
        if p.is_file():
            files[str(p.resolve())] = _static_hash(p.read_bytes())
    return TypedStaticExecutableSelection(_STATIC_SELECTION_KEY, unit, target, files, recipe["env"])


class Unit(NamedTuple):
    """One audited object (a translation unit's copy): its normalised bodies, defined function bindings, unique
    strongly bound imports and the symbols proven never to return."""
    copy: str
    bodies: dict
    prefix: str
    functions: dict
    imports: frozenset
    noreturn: frozenset
    visibility: dict = {}  # Raw name -> visibility of each defined function.
    macho: bool = False
    object_identity: bytes = b""


def make_unit(copy: str, image: ObjectImage, bodies: dict) -> Unit:
    imports = import_symbols(image)
    functions: dict[str, set] = {}
    visibility: dict[str, set] = {}
    for symbol in image.symbols:
        if symbol.kind == "function" and symbol.name in bodies:
            functions.setdefault(symbol.name, set()).add(symbol.binding)
            visibility.setdefault(symbol.name, set()).add(symbol.visibility)
    noreturn = noreturn_symbols(bodies, GENERIC, trusted_imports(image), image.prefix) | \
        {image.prefix + name for name in NORETURN_IMPORTS if image.prefix + name in imports}
    return Unit(copy, bodies, image.prefix, {name: bindings.pop() if len(bindings) == 1 else "ambiguous"
                                             for name, bindings in functions.items()},
                imports, frozenset(noreturn),
                {name: values.pop() if len(values) == 1 else "ambiguous" for name, values in visibility.items()},
                image.macho, image_fingerprint(image))


class CallContext(NamedTuple):
    """Entry facts of a called helper, in the helper's coordinates: stack provenance, data taint and caller
    incoming provenance of x0-x7; the caller frame [0, frame) above the helper's entry sp (None: unknown); whether
    the call site lies in a caller loop; the call depth."""
    registers: tuple
    taint: tuple
    incoming: tuple
    frame: int | None
    looped: bool
    depth: int


class CallModel:
    """Interprocedural resolution of `bl` targets (no name allowlist: every admission is derived from the
    object's symbols, relocations and the injected link-selection records).

    - A defined TU-local (STB_LOCAL / Mach-O non-external) function is audited in that unit's own body.
    - A weak or global definition executes the link-selected copy: admitted only with a link-selection record
      {symbol: {"copy": unit, "binding": "direct"|"process", ...}} (select(); NONPREEMPTIBLE_* above) naming an
      audited unit that holds the body. No record, a bare "plt" or "interposable" binding, a process resolution
      that selects another object, an unproven non-preemptibility or a missing copy fails (an un-audited copy).
      The parent's process-binding artifact (P2) supplies the records; a DSO link map alone is not enough.
    - Undefined imports: the never-returning set (NORETURN_IMPORTS), memcpy/__memcpy_chk, or a returning import
      admitted only on a cold path (every path from it ends in a never-returning call). Anything else fails.
    - Helpers are analysed with the same Analysis under the caller's entry facts (CallContext), cached per
      context; recursion and a call depth above MAX_CALL_DEPTH fail. Loops inside a helper need a role entry
      {symbol: {"source": path, "source_sha256": sha, "loops": {helper loop id: role}}} (source-validated by the
      stack table) or the call fails.
    """

    def __init__(self, units: dict[str, Unit], selection: dict | None = None,
                 helper_roles: dict | None = None) -> None:
        self.units, self.selection, self.helper_roles = units, selection or {}, helper_roles or {}
        self.cache: dict[tuple, "Analysis"] = {}
        self.active: list[tuple[str, str]] = []

    def resolve(self, unit: Unit, target: str) -> tuple[str, object, str]:
        """(kind, unit or None, name): helper, memcpy, import - or ("fail", None, reason)."""
        binding = unit.functions.get(target)
        if binding == "local":
            return "helper", unit, target
        if binding not in (None, "global", "weak"):
            return "fail", None, f"unsupported call boundary (ambiguous or unknown binding {binding!r}): {target}"
        if binding is not None:
            record = self.selection.get(target)
            if record is None:
                return "fail", None, (f"unsupported call boundary (a {binding} definition executes the link-selected "
                                       f"copy; no link-selection record: un-audited weak copy): {target}")
            if isinstance(record, TypedStaticExecutableSelection):
                copy = record.unit
                try:
                    record.check(copy, target)
                except ValueError as error:
                    return "fail", None, str(error)
                if self.units.get(copy.copy) != copy:
                    return "fail", None, "static selected Unit is not registered"
                return "helper", copy, target
            copy = self.units.get(record.get("copy"))
            if copy is None or target not in copy.bodies or copy.functions.get(target) is None:
                return "fail", None, f"unsupported call boundary (the link-selected copy is not audited): {target}"
            reason = self.select(unit, copy, target, record)
            if reason:
                return "fail", None, f"unsupported call boundary ({reason}): {target}"
            return "helper", copy, target
        if target in unit.imports:
            bare = target[len(unit.prefix):] if target.startswith(unit.prefix) else None
            if bare in MEMCPY_IMPORTS:
                return "memcpy", None, bare
            return "import", None, target
        return "fail", None, f"unsupported call boundary (undefined, weak or unbound symbol): {target}"

    @staticmethod
    def select(unit: Unit, copy: Unit, target: str, record: dict) -> str | None:
        """None when the link-selection record proves the audited copy is the executed one, else the reason."""
        binding = record.get("binding")
        if binding == "direct":
            proof = record.get("nonpreemptible")
            if proof in NONPREEMPTIBLE_VISIBILITY:
                if unit.visibility.get(target) != proof or copy.visibility.get(target) != proof:
                    return f"direct binding claims {proof} visibility the objects do not record"
                return None
            if proof in NONPREEMPTIBLE_LINK:
                if not SHA256.match(str(record.get("link_evidence", ""))):
                    return f"direct binding by {proof} without the link evidence digest"
                return None
            return "direct binding without a non-preemptibility proof"
        if binding == "process":
            resolution = record.get("resolution") or {}
            scope = resolution.get("scope") or []
            if not scope or any(set(entry) != {"object", "role", "sha256", "defines"} or
                                entry["role"] not in ("executable", "dso") or not SHA256.match(str(entry["sha256"]))
                                or not isinstance(entry["defines"], bool) for entry in scope):
                return "malformed process resolution record"
            if scope[0]["role"] != "executable" or any(entry["role"] != "dso" for entry in scope[1:]):
                return "process lookup scope must start with the executable, then the DSOs in load order"
            definer = next((entry["object"] for entry in scope if entry["defines"]), None)
            if definer is None or definer != resolution.get("copy_object"):
                return (f"process resolution selects {definer!r}, not the audited copy's object "
                        f"{resolution.get('copy_object')!r} (interposition)")
            return None
        if binding == "plt":
            return "a PLT binding is preemptible: it needs a process resolution record"
        return f"binding {binding!r}"

    def summary(self, unit: Unit, name: str, context: CallContext) -> "Analysis | str":
        identity = (unit.copy, name)
        if identity in self.active:
            return f"recursion through {name}"
        if context.depth > MAX_CALL_DEPTH:
            return f"call depth above {MAX_CALL_DEPTH} at {name}"
        key = (identity, context)
        if key not in self.cache:
            self.active.append(identity)
            try:
                analysis = Analysis(name, unit.bodies[name], GENERIC, set(), model=self, unit=unit, context=context)
                if not any(block.instructions for block in analysis.blocks):
                    analysis.errors.append("empty or unclassifiable body")
                else:
                    analysis.run()
            finally:
                self.active.pop()
            self.cache[key] = analysis
        return self.cache[key]


def call_model(image: ObjectImage, bodies: dict, copy: str = "self", selection: dict | None = None,
               helper_roles: dict | None = None, units: dict[str, Unit] | None = None) -> tuple[CallModel, Unit]:
    """The call model of one audited object (plus other audited copies a link-selection record may name)."""
    unit = make_unit(copy, image, bodies)
    return CallModel({**(units or {}), copy: unit}, selection, helper_roles), unit


class Analysis:
    """Control-flow-aware audit of one function: counts FP/SIMD spills, records structural failures.

    Without a call model the only admitted call is to a proven never-returning diagnostic. With one (CallModel),
    calls resolve as described there; a callee's frame and body traffic is charged at the call site's loop.
    Indirect calls other than the TLS descriptor call and tail calls are structural failures.
    """

    def __init__(self, name: str, body: list[str], dialect: Dialect, noreturn: set[str],
                 model: CallModel | None = None, unit: Unit | None = None, context: CallContext | None = None,
                 eh: dict[int, str] | None = None) -> None:
        self.name, self.dialect = name, dialect
        self.noreturn = set(noreturn) | (set(unit.noreturn) if unit else set())
        self.model, self.unit, self.context = model, unit, context
        self.eh = eh if eh is not None else (getattr(unit.bodies, "eh", {}).get(name, {}) if unit else {})
        self.helper_loops: list[dict] = []
        self.cold: set[int] = set()
        self.cold_calls: set[int] = set()
        self.cold_traffic: dict[str, int] = {}
        self.outer: set[tuple[int, int, bool]] = set()
        self.returned: dict[str, tuple[frozenset, frozenset, frozenset]] = {}
        self.pads: set[int] = set()
        self.blocks, self.errors = build_cfg(body, dialect)
        self.positions: dict[int, int] = {}
        for block in self.blocks:
            for instruction in block.instructions:
                if instruction.base != "nop":
                    self.positions[id(instruction)] = len(self.positions)
        self.bounded: list[dict] = []  # Admitted variable-size memcpy sites (M-1..M-3), report pass only.
        self.spills = 0
        self.report = False
        self.labels = {label: index for index, block in enumerate(self.blocks) for label in block.labels}
        # Stack traffic per class and per natural loop (0: outside every loop), counted in the report pass.
        self.traffic: dict[str, dict[int, int]] = {kind: {} for kind in TRAFFIC}
        self.loops: list[dict] = []
        self.innermost: dict[int, int] = {}
        self.loop_invariants: dict[int, set[str]] = {}
        self.entry_invariants: set[str] = set()
        self.block = 0

    def tally(self, kind: str, amount: int = 1, loop: int | None = None) -> None:
        if not self.report or not amount:
            return
        if self.block in self.cold:
            self.cold_traffic[kind] = self.cold_traffic.get(kind, 0) + amount  # Listed, never counted.
            return
        loop = self.innermost.get(self.block, 0) if loop is None else loop
        self.traffic[kind][loop] = self.traffic[kind].get(loop, 0) + amount

    def looped(self) -> bool:
        """Inside a loop of this function or, for a helper, at a call site inside a caller loop."""
        return self.in_loop() or (self.context is not None and self.context.looped)

    def in_loop(self) -> bool:
        return self.innermost.get(self.block, 0) != 0

    def fail(self, instruction: Instruction | None, message: str) -> None:
        if self.report:
            where = f" at '{instruction.text}'" if instruction else ""
            self.errors.append(f"{message}{where}")

    # Control flow --------------------------------------------------------------------------------------------
    def classify_branch(self, instruction: Instruction) -> tuple[str, str | None]:
        """('fall'|'branch'|'jump'|'return'|'call'|'noreturn'|'indirect'|'invalid', target)."""
        base = instruction.base
        if base == "blr" and len(instruction.operands) == 2 and instruction.operands[1].startswith(":tlsdesc_call:"):
            return "tlscall", instruction.operands[1]
        if RETURN.match(base):
            return ("return", None) if not instruction.operands else ("indirect", None)
        if INDIRECT.match(base):
            return "indirect", None
        if base not in BRANCHES:
            return "fall", None
        target = instruction.operands[-1] if instruction.operands else ""
        if not SYMBOL.match(target):
            return "indirect", None
        if target in self.labels:
            if base == "bl":
                return "invalid", target
            return ("jump" if base == "b" else "branch"), target
        if self.dialect.local(target):
            return "invalid", target
        if base == "bl" and target in self.noreturn:
            return "noreturn", target
        return "call", target

    def link(self) -> None:
        ordinal = 0
        for index, block in enumerate(self.blocks):
            block.falls = True
            for instruction in block.instructions:
                kind, target = self.classify_branch(instruction)
                if kind in ("jump", "branch"):
                    block.successors.append(self.labels[target])
                if kind in ("jump", "return", "noreturn", "indirect", "invalid") or \
                        (kind == "call" and instruction.base != "bl"):
                    block.falls = False
                pad = self.eh.get(ordinal)
                if pad is not None and instruction.base == "bl":
                    # Exception edge (LSDA call-site range): the call may unwind to its landing pad.
                    if pad not in self.labels:
                        self.errors.append(f"landing pad {pad} is not code of this function")
                    else:
                        block.successors.append(self.labels[pad])
                        self.pads.add(self.labels[pad])
                ordinal += 1
            if block.falls:
                block.successors.append(index + 1 if index + 1 < len(self.blocks) else -1)

    # Data flow ---------------------------------------------------------------------------------------------------
    def find_loops(self) -> None:
        """Loop nesting forest of the reachable CFG (Steensgaard): every non-trivial strongly connected component
        is a loop; its entries are the members with a predecessor outside it (or the function entry); removing
        the edges from inside the component to its entries exposes the nested loops, recursively. A reducible
        loop has one entry, its natural-loop header; a loop GCC enters at more than one block (an irreducible
        region, e.g. after jump threading) is still one loop, attributed the same way. Each block's innermost
        loop is the deepest component containing it."""
        successors = {index: [s for s in block.successors if s >= 0] for index, block in enumerate(self.blocks)}
        reachable, work = {0}, [0]
        while work:
            for successor in successors[work.pop()]:
                if successor not in reachable:
                    reachable.add(successor)
                    work.append(successor)
        predecessors: dict[int, set[int]] = {index: set() for index in reachable}
        for index in reachable:
            for successor in successors[index]:
                predecessors[successor].add(index)
        found: list[dict] = []
        self.members_by_id: dict[int, set[int]] = {}

        def components(nodes: set[int], removed: set[tuple[int, int]]) -> list[list[int]]:
            index_of, low, on_stack, stack, result, counter = {}, {}, set(), [], [], [0]
            for root in sorted(nodes):
                if root in index_of:
                    continue
                frames = [(root, iter(sorted(successors[root])))]
                index_of[root] = low[root] = counter[0]
                counter[0] += 1
                stack.append(root)
                on_stack.add(root)
                while frames:
                    node, children = frames[-1]
                    child = next(children, None)
                    if child is not None:
                        if child not in nodes or (node, child) in removed:
                            continue
                        if child not in index_of:
                            index_of[child] = low[child] = counter[0]
                            counter[0] += 1
                            stack.append(child)
                            on_stack.add(child)
                            frames.append((child, iter(sorted(successors[child]))))
                        elif child in on_stack:
                            low[node] = min(low[node], index_of[child])
                        continue
                    frames.pop()
                    if frames:
                        low[frames[-1][0]] = min(low[frames[-1][0]], low[node])
                    if low[node] == index_of[node]:
                        component = []
                        while True:
                            member = stack.pop()
                            on_stack.discard(member)
                            component.append(member)
                            if member == node:
                                break
                        result.append(sorted(component))
            return result

        def nest(nodes: set[int], removed: set[tuple[int, int]], parent: int, depth: int) -> None:
            for component in components(nodes, removed):
                members = set(component)
                cyclic = len(component) > 1 or any(successor == component[0] and (component[0], successor) not in removed
                                                   for successor in successors[component[0]])
                if not cyclic:
                    continue
                entries = sorted(member for member in members
                                 if member == 0 or any(p not in members for p in predecessors[member]))
                if not entries:
                    entries = [component[0]]
                back = {(source, entry) for entry in entries for source in predecessors[entry]
                        if source in members and (source, entry) not in removed}
                loop = {"id": len(found) + 1, "parent": parent, "depth": depth, "entries": entries,
                        "latches": sorted({source for source, _ in back}), "members": members}
                found.append(loop)
                self.members_by_id[loop["id"]] = members
                for member in members:
                    self.innermost[member] = loop["id"]
                nest(members, removed | back, loop["id"], depth + 1)

        nest(reachable, set(), 0, 1)
        # Stable identities: number loops by their first entry block (program order), keeping parents consistent.
        order = sorted(found, key=lambda loop: (loop["entries"][0], loop["depth"]))
        renumber = {loop["id"]: number + 1 for number, loop in enumerate(order)}
        renumber[0] = 0
        self.innermost = {block: renumber[loop] for block, loop in self.innermost.items()}
        # Registers some instruction of each loop defines (its whole body, nested loops included).
        self.loop_invariants = {renumber[number]: loop_invariant_registers(
                                    [instruction for member in sorted(members)
                                     for instruction in self.blocks[member].instructions])
                                for number, members in self.members_by_id.items()}
        # At a helper's entry level (a call site inside a caller loop): registers the helper defines only by adrp
        # hold one constant address on every call.
        defined: dict[str, set[str]] = {}
        for block in self.blocks:
            for instruction in block.instructions:
                for register in defined_registers(instruction):
                    defined.setdefault(register, set()).add(instruction.base)
        self.entry_invariants = {register for register, bases in defined.items() if bases == {"adrp"}}
        self.loops = [{"id": renumber[loop["id"]], "parent": renumber[loop["parent"]], "depth": loop["depth"],
                       "entries": loop["entries"], "latches": loop["latches"], "blocks": len(loop["members"])}
                      for loop in order]

    def run(self) -> None:
        self.link()
        self.find_loops()
        if self.errors:
            return
        entry = State.entry()
        if self.context is not None:
            # A helper's arguments carry the caller's facts: stack provenance (helper coordinates), data taint and
            # the caller's incoming provenance (an argument without one is not caller-owned memory).
            entry.incoming = {register: values for register, values in self.context.incoming}
            for register, values in self.context.registers:
                entry.assign(register, values)
            for register, values in self.context.taint:
                entry.taint[register] = values
        states: dict[int, State] = {0: entry}
        pending = [0]
        while pending:
            index = pending.pop()
            state = self.transfer(index, states[index].copy())
            if state is None:
                continue
            for successor in self.blocks[index].successors:
                if successor < 0:
                    continue
                merged = state if successor not in states else states[successor].join(state)
                if successor not in states or merged.key() != states[successor].key():
                    states[successor] = merged
                    pending.append(successor)
        self.find_cold(states)
        self.report = True
        for index, block in enumerate(self.blocks):
            if index not in states:
                # Alignment padding after the last return is never executed; anything else is unclassifiable.
                if block.instructions and not all(instruction.base == "nop" for instruction in block.instructions):
                    self.errors.append(f"unreachable instructions cannot be classified: '{block.instructions[0].text}'")
                continue
            self.block = index
            if self.transfer(index, states[index].copy()) is not None and block.falls and index + 1 >= len(self.blocks):
                self.errors.append("control falls off the end of the function")
        for pad in sorted(self.pads - self.cold):
            self.errors.append(f"a landing pad path reaches a return or a loop (block {pad}): it must end in "
                               f"_Unwind_Resume or a never-returning call")
        for block in sorted(self.cold_calls - self.cold):
            self.errors.append(f"a returning external call outside a never-returning sequence: "
                               f"'{self.blocks[block].instructions[-1].text}'")

    def find_cold(self, states: dict[int, State]) -> None:
        """Cold blocks: reachable blocks from which no path (normal or exception edges) reaches a return or a
        loop - every path ends in a never-returning call. Their traffic is listed, never counted."""
        anchors = {index for index in states if index in self.innermost or
                   any(RETURN.match(instruction.base) for instruction in self.blocks[index].instructions)}
        predecessors: dict[int, set[int]] = {}
        for index in states:
            for successor in self.blocks[index].successors:
                predecessors.setdefault(successor, set()).add(index)
        hot, work = set(anchors), list(anchors)
        while work:
            for predecessor in predecessors.get(work.pop(), ()):
                if predecessor not in hot:
                    hot.add(predecessor)
                    work.append(predecessor)
        self.cold = set(states) - hot

    def transfer(self, index: int, state: State) -> State | None:
        self.block = index
        for instruction in self.blocks[index].instructions:
            if not self.step(instruction, state):
                return None
        return state

    def step(self, instruction: Instruction, state: State) -> bool:
        """Apply one instruction; False when control leaves the function or the path cannot be classified."""
        if not instruction.admitted():
            self.fail(instruction, f"instruction form outside the admitted grammar: {instruction.form or instruction.base}")
            return False
        if instruction.base in FLAG_READERS and state.flags_clobbered:
            self.fail(instruction, "reads NZCV clobbered by a call (TLS descriptor or function call)")
            return False
        if instruction.base in FLAG_WRITERS:
            state.flags_clobbered = False
        kind, target = self.classify_branch(instruction)
        if kind == "indirect":
            self.fail(instruction, "indirect or uninspectable branch")
            return False
        if kind == "invalid":
            self.fail(instruction, "branch to a label that is not code of this function")
            return False
        if kind == "noreturn":
            return False
        if kind == "tlscall":
            # TLS descriptor resolver (AAPCS64 TLSDESC): defines x0 (the TLS offset) and x30; preserves every
            # other general and FP/SIMD register and the stack. A stack address cannot be the descriptor.
            if not self.reads(instruction, state, instruction.operands[:1]):
                return False
            if provenance(state.value("x0")) != "N":
                self.fail(instruction, "a stack address is passed to a TLS descriptor call")
                return False
            for register in ("x0", "x30"):
                state.define(register, frozenset())
                state.assign(register, NOT_STACK)
                state.arrive(register, NOT_STACK)
            state.flags_clobbered = True
            self.tally("tls_call")
            return True
        if instruction.base == "mrs" and instruction.operands[1:] != ["tpidr_el0"]:
            self.fail(instruction, "system register read other than tpidr_el0")
            return False
        if kind == "call":
            if instruction.base != "bl":
                self.fail(instruction, f"unsupported call boundary (tail call): {target}")
                return False
            if self.model is None or self.unit is None:
                self.fail(instruction, f"unsupported call boundary (only the proven fatal diagnostic may be called): "
                                       f"{target}")
                return False
            return self.call(instruction, state, target)
        if kind == "return":
            self.leave(instruction, state)
            return False
        if kind in ("jump", "branch"):
            return self.reads(instruction, state, instruction.operands[:-1])
        if instruction.base == "ldr" and instruction.memory is not None and \
                ":gottprel_lo12:" in instruction.operands[instruction.memory]:
            # This pair's GOT load is not stack traffic or an invariant-load credit.
            memory = MEMORY.fullmatch(instruction.operands[instruction.memory])
            source = gp(memory.group(1)) if memory else None
            if source is None or provenance(state.value(source)) != "N":
                self.fail(instruction, "TLS initial-exec load uses a stack-derived operand")
                return False
            destination = instruction.operands[0]
            state.define(destination, frozenset())
            state.assign(destination, NOT_STACK)
            state.arrive(destination, NOT_STACK)
            return True
        if instruction.memory is not None:
            return self.access(instruction, state)
        return self.compute(instruction, state)

    def reads(self, instruction: Instruction, state: State, operands: list[str]) -> bool:
        """A 32-bit view of a stack address cannot be classified: reading one is a structural failure."""
        for operand in operands:
            view = gp_view(operand)
            if view and provenance(state.value(view)) != "N":
                self.fail(instruction, f"32-bit view of a stack address {operand}")
                return False
        return True

    def leave(self, instruction: Instruction, state: State) -> None:
        for n in CALLEE_SAVED:
            if any(status == "D" for status, _ in state.saved[n]):
                self.fail(instruction, f"returns without restoring callee-saved d{n}")
        if self.context is None:
            return
        # A helper returns to audited code: sp, x19-x29 must hold the caller's values. Its final facts of x0-x18
        # (joined over every return) are what the caller sees after the call (see call()).
        if state.value("sp") != frozenset({("S", 0)}):
            self.fail(instruction, "a helper returns with sp not restored")
        for register in GPR_CALLEE_SAVED[:-1]:
            if any(status not in ("U", "S", "R") for status, _ in state.gsaved[register]):
                self.fail(instruction, f"a helper returns without preserving callee-saved {register}")
        for register in CALL_REGISTERS:
            values, taint, arrival = self.returned.get(register, (frozenset(), frozenset(), frozenset()))
            self.returned[register] = (values | state.value(register), taint | state.tainted(register),
                                       arrival | state.arrival(register))

    def clobber_call(self, state: State) -> None:
        """AAPCS64 at a call's return: x0-x18 and x30 are redefined (provenance, taint, incoming provenance and
        constants reset) and NZCV is undefined. v0-v7 and v16-v31 hold no tracked fact; v8-v15 keep their ABI
        status only because every audited callee proves it preserves their low halves."""
        for register in sorted(CALL_CLOBBERED | {"x30"}):
            state.define(register, frozenset())
            state.assign(register, NOT_STACK)
            state.arrive(register, NOT_STACK)
        state.flags_clobbered = True

    def overlaps_live(self, state: State, start: int, width: int) -> bool:
        """Whether [start, start + width) overlaps a live pointer slot, an ABI save slot or a canary slot."""
        slots = [(slot, 8) for slot in state.pointers] + [(slot, 8) for slot in state.canary]
        slots += [entry for n in CALLEE_SAVED for status, entry in state.saved[n] if status in ("S", "D") and entry]
        slots += [(slot, 8) for register in GPR_CALLEE_SAVED for status, slot in state.gsaved[register]
                  if status in ("S", "D") and slot is not None]
        return any(slot < start + width and start < slot + size for slot, size in slots)

    def retire_incoming(self, state: State, start: int, width: int) -> None:
        for slot in [slot for slot in state.islots if slot < start + width and start < slot + 8]:
            del state.islots[slot]

    def outer_access(self, instruction: Instruction, address: int, width: int, store: bool) -> bool:
        """A helper's access at or above its entry sp: it must lie inside the caller's frame exactly."""
        frame = self.context.frame if self.context else None
        if frame is None or address + width > frame:
            self.fail(instruction, "a helper accesses stack memory outside the caller frame")
            return False
        if self.report:
            self.outer.add((address, width, store))
        return True

    def call(self, instruction: Instruction, state: State, target: str) -> bool:
        """A direct call under the call model (see CallModel)."""
        if self.unit.macho:
            # No Mach-O exception tables (__eh_frame / __compact_unwind) are parsed: call edges cannot be complete.
            self.fail(instruction, f"unsupported call boundary (the call model fails closed on Mach-O): {target}")
            return False
        kind, unit, name = self.model.resolve(self.unit, target)
        if kind == "fail":
            self.fail(instruction, name)
            return False
        if kind == "memcpy":
            return self.memcpy(instruction, state, name)
        if kind == "import":
            self.cold_calls.add(self.block)  # Proven cold after the fixed point (run()), else fatal.
            self.clobber_call(state)
            return True
        sp = single_offset(state.value("sp"))
        registers, taint, incoming = [], [], []
        # x0-x18 (arguments and caller-saved values a helper may read) must be exact; x19-x28 keep the caller's
        # facts too (a callee could use them as bases), translated with their non-stack alternative, so any use
        # of a path-dependent one fails inside the helper.
        for register in CALL_REGISTERS + GPR_CALLEE_SAVED[:-2]:
            values = state.value(register)
            if provenance(values) == "A" and register in CALL_REGISTERS:
                self.fail(instruction, f"{register} is a stack address on only some paths at a call")
                return False
            if provenance(values) != "N":
                if sp is None:
                    self.fail(instruction, "a stack address is passed while sp is not at a known offset")
                    return False
                values = frozenset(value if value == "N" else ("S", None if value[1] is None else value[1] - sp)
                                   for value in values)
                registers.append((register, values))
            if state.tainted(register):
                taint.append((register, state.tainted(register)))
            if state.arrival(register) != NOT_STACK:
                incoming.append((register, state.arrival(register)))
        depth = (self.context.depth if self.context else 0) + 1
        context = CallContext(tuple(registers), tuple(taint), tuple(incoming), -sp if sp is not None else None,
                              self.looped(), depth)
        callee = self.model.summary(unit, name, context)
        if isinstance(callee, str):
            self.fail(instruction, f"unsupported call boundary ({callee})")
            return False
        if callee.errors:
            self.fail(instruction, f"helper {name}: {callee.errors[0]}")
            return False
        roles = self.model.helper_roles.get(name, {}).get("loops", {})
        for loop in callee.loops:
            if str(loop["id"]) not in roles:
                self.fail(instruction, f"helper {name}: loop {loop['id']} has no role entry (helper loop without a "
                                       f"role map)")
                return False
        for address, width, store in sorted(callee.outer):
            start = address + (sp or 0)
            if self.overlaps_live(state, start, width):
                self.fail(instruction, f"helper {name} accesses a live pointer, save or canary slot of the caller")
                return False
            if self.context is not None and start >= 0 and not self.outer_access(instruction, start, width, store):
                return False
            if store:
                self.retire_incoming(state, start, width)
        if self.report:
            self.charge(name, callee, roles)
        self.clobber_call(state)
        # The audited executed body (TU-local, or the link-selected copy select() proved) decides x0-x18 exactly
        # (GCC's -fipa-ra may keep a caller value live in a caller-saved register across a call to a known body,
        # so an ABI reset could hide its later stack use): the helper's joined return facts - stack provenance,
        # FP/stack data taint, incoming provenance - with caller-frame addresses mapped back and own-frame (dead)
        # or unknown-offset addresses kept as stack addresses at an unknown offset. Imports keep the AAPCS reset.
        # FP/SIMD registers carry no tracked fact the analysis could reset or keep (a stack address entering one
        # already fails; every FP/SIMD stack transfer is counted whatever the value): v0-v7 and v16-v31 need no
        # transfer, and v8-v15's low halves are preserved because the helper's own analysis proves its saves.
        for register, (values, taint, arrival) in callee.returned.items():
            state.assign(register, frozenset(value if value == "N" else
                                             ("S", None if value[1] is None or value[1] < 0 or sp is None
                                              else value[1] + sp) for value in values))
            state.define(register, taint)
            state.arrive(register, arrival or NOT_STACK)
        return True

    def charge(self, name: str, callee: "Analysis", roles: dict) -> None:
        """Charge a helper's traffic at this call site: its entry-level traffic (frame included) at the call site's
        innermost loop, its loops (and its own helpers' loops) as nested helper loops of that loop."""
        self.tally("int_call")
        here = self.innermost.get(self.block, 0)
        if self.block in self.cold:
            for kind, counts in callee.traffic.items():
                self.tally(kind, sum(counts.values()))
            return
        depths = {0: 0, **{loop["id"]: loop["depth"] for loop in self.loops + self.helper_loops}}
        mapping = {0: here}
        for loop in sorted(callee.loops + callee.helper_loops, key=lambda loop: loop["id"]):
            number = len(self.loops) + len(self.helper_loops) + 1
            mapping[loop["id"]] = number
            parent = mapping[loop["parent"]]
            record = {"id": number, "parent": parent, "depth": depths[parent] + 1, "entries": [], "latches": [],
                      "helper": loop.get("helper", name), "helper_loop": loop.get("helper_loop", loop["id"]),
                      "helper_role": loop.get("helper_role", roles.get(str(loop["id"])))}
            depths[number] = record["depth"]
            self.helper_loops.append(record)
        for kind, counts in callee.traffic.items():
            for loop, count in counts.items():
                self.tally(kind, count, mapping[loop])
        for kind, count in callee.cold_traffic.items():
            self.cold_traffic[kind] = self.cold_traffic.get(kind, 0) + count

    def memcpy(self, instruction: Instruction, state: State, name: str) -> bool:
        """memcpy / __memcpy_chk: one constant size on every path (__memcpy_chk: a constant bound >= size); each
        stack side at an exact offset clear of live pointer, save and canary slots, counted in ext_bytes and
        ext_accesses; the call itself in ext_call. Returns its destination in x0."""
        size = state.consts.get("x2")
        site = None
        if size is None:
            key = compact_program_key(self.name, self.unit.prefix if self.unit else "")
            lanes = int(key.split(":")[2]) if key and self.context is None else None
            site = bounded_memcpy_site(self.name, self.positions.get(id(instruction)), lanes)
            if isinstance(site, str):
                self.fail(instruction, f"{name} size is not one constant on every path ({site})")
                return False
            size = site["upper_bytes"]  # M-2: charged and footprint-checked at sizeof(F).
        if name == "__memcpy_chk" and (state.consts.get("x3") is None or state.consts["x3"] < size):
            self.fail(instruction, f"{name} bound is not a constant at least the size")
            return False
        sides = []
        for register, store in (("x0", True), ("x1", False)):
            values = state.value(register)
            if provenance(values) == "A":
                self.fail(instruction, f"{name} {register} is a stack address on only some paths")
                return False
            if provenance(values) == "S":
                start = single_offset(values)
                if start is None:
                    self.fail(instruction, f"{name} {register} is a stack address at an unknown offset")
                    return False
                if self.overlaps_live(state, start, size):
                    self.fail(instruction, f"{name} {register} overlaps a live pointer, save or canary slot")
                    return False
                if self.context is not None and start >= 0 and not self.outer_access(instruction, start, size, store):
                    return False
                sides.append((start, store))
        destination = state.value("x0")
        for start, store in sides:
            if store:
                self.retire_incoming(state, start, size)  # Live pointer, save and canary slots already failed.
        self.tally("ext_call")
        self.tally("ext_bytes", size * len(sides))
        self.tally("ext_accesses", -(-size // 8) * len(sides))
        if site is not None and self.report and self.block not in self.cold:
            self.bounded.append({**site, "loop": self.innermost.get(self.block, 0), "sides": len(sides),
                                 "index": self.positions[id(instruction)]})
        self.clobber_call(state)
        state.assign("x0", destination)
        return True

    def clobber(self, instruction: Instruction, state: State, number: int) -> None:
        if number not in CALLEE_SAVED:
            return
        result = set()
        for status, slot in state.saved[number]:
            if status == "U":
                self.fail(instruction, f"clobbers callee-saved v{number} before saving it")
                result.add((status, slot))
                continue
            if status == "R":
                self.fail(instruction, f"clobbers callee-saved v{number} after its epilogue restore")
            result.add(("D", slot))
        state.saved[number] = frozenset(result)

    def access(self, instruction: Instruction, state: State) -> bool:
        """Loads/stores: FP/SIMD stack traffic, ABI saves/restores, pointer slots, with exact footprints."""
        name = instruction.base
        load, store = name in LOADS, name in STORES
        if not (load or store):
            self.fail(instruction, "unclassifiable memory instruction")
            return False
        memory = MEMORY.match(instruction.operands[instruction.memory])
        if not memory:
            self.fail(instruction, "unclassifiable addressing mode")
            return False
        base = gp(memory.group(1))
        offset_text, writeback = memory.group(2), memory.group(3) == "!"
        post = instruction.operands[instruction.memory + 1:]
        displacement: int | None = 0
        if offset_text:
            if immediate(offset_text) is not None:
                displacement = immediate(offset_text)
            elif RELOCATION.match(offset_text):
                displacement = None
                if provenance(state.value(base)) != "N":
                    self.fail(instruction, "relocated offset from a stack base")
                    return False
            elif INDEX.match(offset_text):
                displacement = None
                index = INDEX.match(offset_text).group(1)
                if provenance(state.value(gp(index) or gp_view(index))) != "N":
                    self.fail(instruction, "stack address used as an index register")
                    return False
            else:
                self.fail(instruction, "unclassifiable addressing offset")
                return False
        if post and (writeback or offset_text or len(post) != 1 or immediate(post[0]) is None):
            self.fail(instruction, "unclassifiable post-index form")
            return False
        base_values = state.value(base)
        kind = provenance(base_values)
        if kind == "A":
            self.fail(instruction, "base register is a stack address on only some paths")
            return False
        stack = kind == "S"
        origin = single_offset(base_values)
        address = None if origin is None or displacement is None else origin + (0 if post else displacement)
        data = instruction.operands[:instruction.memory]
        if store and not self.reads(instruction, state, data):
            return False
        items, position = [], address
        for operand in data:
            width = footprint(instruction, operand)
            registers = fp_registers(operand) if not (gp(operand) or GP32.match(operand)) else None
            if width is None or (registers is None and not (gp(operand) or GP32.match(operand))):
                self.fail(instruction, f"memory footprint of {operand} is not exactly known")
                return False
            items.append(("fp" if registers is not None else ("x" if gp(operand) else "w"),
                          operand, registers, position, width))
            position = None if position is None else position + width
        if stack and self.context is not None:
            if address is None and self.context.registers:
                self.fail(instruction, "a helper accesses the stack at an unknown offset while a caller stack "
                                       "address is live")
                return False
            if address is not None and position > 0:  # Bytes at or above the helper's entry sp: caller frame.
                if address < 0:
                    self.fail(instruction, "a helper access straddles its own and the caller frame")
                    return False
                if store and any(provenance(state.value(gp(operand))) != "N"
                                 for category, operand, *_ in items if category == "x"):
                    self.fail(instruction, "a helper stores a stack address into the caller frame")
                    return False
                if not self.outer_access(instruction, address, position - address, store):
                    return False
        if stack and not self.slots(instruction, state, load, store, address, items):
            return False
        if store and not stack and any(provenance(state.value(gp(operand))) != "N"
                                       for category, operand, *_ in items if category == "x"):
            self.fail(instruction, "a stack address is stored where it cannot be tracked")
            return False
        spilled = False
        eligible = name in ABI_TRANSFERS and all(ABI_REGISTER.match(operand) for _, operand, *_ in items)
        # Caller-owned memory: an address that is an incoming argument plus a known constant on every path.
        alias = (not stack and displacement is not None and not writeback and not post and
                 invariant_alias(state.arrival(base)))
        looped = self.looped()
        if alias and load and looped:
            self.tally("alias_load")
        relocated = bool(offset_text) and RELOCATION.match(offset_text) is not None
        constant = not offset_text or immediate(offset_text) is not None or relocated
        invariant = self.loop_invariants.get(self.innermost.get(self.block, 0), self.entry_invariants)
        if load and not stack and looped and not writeback and not post and constant and \
                (alias or base in invariant):
            self.tally("invariant_load")
            if relocated:
                self.tally("invariant_reloc")
        if alias and store:
            self.tally("alias_store")
        general = [item for item in items if item[0] != "fp"]
        # Stack-protector canary: the guard value (loaded through the guard symbol, or through a register holding
        # its GOT address) stored to an exact 8-byte slot, and every whole reload of that slot.
        prefix = self.unit.prefix if self.unit else ""
        guard = load and not stack and (offset_text == f":lo12:{prefix}{STACK_GUARD}" or
                                        CANARY_DATA in state.tainted(base))
        canary = stack and bool(general) and len(general) == len(items) and all(
            category == "x" and width == 8 and start is not None and
            ((store and CANARY_DATA in state.tainted(gp(operand))) or (load and start in state.canary))
            for category, operand, _, start, width in general)
        if stack and store:
            self.track_incoming_slots(state, items)
            self.retire_general_saves(state, items)
            for _, _, _, start, width in items:
                state.canary = frozenset(slot for slot in state.canary
                                         if start is not None and not (slot < start + width and start < slot + 8))
            if canary:
                state.canary = state.canary | {start for _, _, _, start, _ in general}
        abi_general = bool(general) and stack and name in ABI_TRANSFERS
        fp_abi = False
        for category, operand, registers, start, width in items:
            if category != "fp":
                register = gp(operand) or gp_view(operand)
                if load:
                    exact = stack and start is not None and category == "x" and name in WHOLE_X
                    restoring = exact and self.restore_general(state, register, start)
                    state.assign(register, state.slot(start) if exact else NOT_STACK)
                    state.define(register, frozenset({CANARY_DATA}) if guard or canary else
                                 frozenset({STACK_DATA}) if stack else frozenset())
                    state.arrive(register, state.islots.get(start, NOT_STACK) if exact else NOT_STACK)
                    if restoring:
                        state.gsaved[register] = frozenset({("R", start)})
                    abi_general = abi_general and restoring
                else:
                    if stack and FP_DATA in state.tainted(register):
                        spilled = True  # FP/SIMD data stored through a general register.
                    saved = stack and category == "x" and name in ABI_TRANSFERS and \
                        self.save_general(state, register, start)
                    abi_general = abi_general and saved
                continue
            for number, register_width in registers:
                if stack and eligible and store and self.save(state, number, register_width, start):
                    fp_abi = True
                elif stack and eligible and load and self.restore(state, number, register_width, start):
                    fp_abi = True
                elif stack:
                    spilled = True
                    if load and writes_callee_saved_half(operand):
                        self.clobber(instruction, state, number)
                elif load and writes_callee_saved_half(operand):
                    self.clobber(instruction, state, number)
                if eligible and start is not None:
                    start += register_width
        if spilled and self.report:
            self.spills += 1
            self.tally("fp")
        if general and stack:
            if canary:
                self.tally("call_frame")
            elif not abi_general:
                self.tally("scalar")
            elif self.in_loop():
                self.tally("abi_loop")
            elif self.context is not None:
                self.tally("call_frame")  # A helper's entry-level frame runs on every call: charged, never exempt.
        if fp_abi and self.in_loop():
            self.tally("abi_loop")
        elif fp_abi and self.context is not None:
            self.tally("call_frame")
        if writeback or post:
            arrival = state.arrival(base)
            state.define(base, frozenset())
            step = displacement if writeback else immediate(post[0])
            if step is None:
                state.assign(base, UNKNOWN_STACK if stack else NOT_STACK)
                state.arrive(base, NOT_STACK)
            else:
                state.assign(base, frozenset(value if value == "N" or value[1] is None else ("S", value[1] + step)
                                             for value in base_values))
                state.arrive(base, frozenset(value if value == "N" or value[2] is None else
                                             ("I", value[1], value[2] + step) for value in arrival))
        return True

    def track_incoming_slots(self, state: State, items: list) -> None:
        """A stack store retires every incoming-value slot it overlaps; an exact 8-byte store of an x register
        holding an incoming value records it (so a reload keeps the provenance)."""
        for category, operand, _, start, width in items:
            if start is None:
                state.islots.clear()
                return
            for slot in [slot for slot in state.islots if slot < start + width and start < slot + 8]:
                del state.islots[slot]
            if category == "x" and width == 8:
                values = state.arrival(gp(operand))
                if values != NOT_STACK:
                    state.islots[start] = values

    @staticmethod
    def retire_general_saves(state: State, items: list) -> None:
        """A stack store over a live x19-x30 save slot (or at an unknown offset) voids that save: its reload is
        then ordinary scalar traffic, never an exempt restore."""
        for _, _, _, start, width in items:
            for register in GPR_CALLEE_SAVED:
                current = state.gsaved[register]
                if any(status in ("S", "D") and (start is None or (slot < start + width and start < slot + 8))
                       for status, slot in current):
                    state.gsaved[register] = frozenset({("X", None)})

    def save_general(self, state: State, register: str, slot: int | None) -> bool:
        """An ABI save of x19-x30: a whole 8-byte store to a fixed frame slot while it holds the caller's value."""
        current = state.gsaved.get(register)
        if not current or slot is None or any(status not in ("U", "R") for status, _ in current):
            return False
        state.gsaved[register] = frozenset({("S", slot)})
        return True

    def restore_general(self, state: State, register: str, slot: int | None) -> bool:
        """An ABI restore of x19-x30: a whole 8-byte reload from its own save slot."""
        current = state.gsaved.get(register)
        return bool(current) and slot is not None and all(status in ("S", "D") and saved == slot
                                                          for status, saved in current)

    def slots(self, instruction: Instruction, state: State, load: bool, store: bool, address: int | None,
              items: list) -> bool:
        """Pointer slots and live ABI save slots: any access that cannot be proven disjoint or exact fails."""
        live = [(n, slot) for n in CALLEE_SAVED for status, slot in state.saved[n]
                if status in ("S", "D") and slot is not None]
        if address is None:
            if state.pointers or (store and live):
                self.fail(instruction, "stack access at an unknown offset while pointer or save slots are live")
                return False
            if store and any(provenance(state.value(gp(operand))) != "N"
                             for category, operand, *_ in items if category == "x"):
                self.fail(instruction, "a stack address is stored where it cannot be tracked")
                return False
            return True
        for category, operand, _, start, width in items:
            end = start + width
            for slot in list(state.pointers):
                if not (slot < end and start < slot + 8):
                    continue
                exact = start == slot and width == 8 and category == "x" and instruction.base in WHOLE_X
                if load and not exact:
                    self.fail(instruction, "partial or reinterpreting reload of a stored stack address")
                    return False
                if store and not exact and not (start <= slot and slot + 8 <= end):
                    self.fail(instruction, "partial overwrite of a stored stack address")
                    return False
                if store and not exact:
                    state.store_slot(slot, NOT_STACK)
            if store:
                for number, (slot, slot_width) in live:
                    if start < slot + slot_width and slot < end:
                        self.fail(instruction, f"overwrites the live save slot of d{number}")
                        return False
                if category == "x":
                    values = state.value(gp(operand))
                    if provenance(values) == "A":
                        self.fail(instruction, "stores a stack address on only some paths")
                        return False
                    state.store_slot(start, values)
        return True

    def save(self, state: State, number: int, width: int, slot: int | None) -> bool:
        """An ABI save stores d8-d15 (or its q view) whole to a fixed frame slot while it holds the caller's value.

        The register holds the caller's value on every path when it is unsaved (U) or restored and not
        clobbered since (R): a save inside a loop whose epilogue restore precedes the back edge (shrink-wrapped
        save/restore around one iteration) saves the caller's value again.
        """
        if number not in CALLEE_SAVED or width < 8 or slot is None or \
                not state.saved[number] or any(status not in ("U", "R") for status, _ in state.saved[number]):
            return False
        state.saved[number] = frozenset({("S", (slot, width))})
        return True

    def restore(self, state: State, number: int, width: int, slot: int | None) -> bool:
        """An epilogue restore reloads a saved register whole from its own save slot, exactly once."""
        current = state.saved.get(number) if number in CALLEE_SAVED else None
        if not current or slot is None or any(status not in ("S", "D") or saved != (slot, width)
                                              for status, saved in current):
            return False
        state.saved[number] = frozenset({("R", (slot, width))})
        return True

    def origins(self, instruction: Instruction, state: State) -> frozenset:
        """Data origins of a general register an instruction defines.

        UMOV/FMOV from an FP/SIMD register yields FP/SIMD data; a register move or a conditional select keeps
        the origins of its sources; a merge definition (bfi, bfxil, movk) also keeps its destination's; any
        other computation yields a new value with no tracked origin.
        """
        base, operands = instruction.base, instruction.operands
        sources = operands[1:]
        if base in FP_TO_GP_MOVES and len(sources) == 1 and fp_registers(sources[0]) is not None:
            return frozenset({FP_DATA})
        registers = [gp(operand) or gp_view(operand) for operand in sources]
        if base == "mov" and len(sources) == 1 and registers[0]:
            return state.tainted(registers[0])
        if base in CONDITIONAL_SELECTS:
            return frozenset().union(*(state.tainted(register) for register in registers[:2] if register))
        if base in MERGE_DEFINITIONS:
            return frozenset().union(*(state.tainted(gp(operand) or gp_view(operand)) for operand in operands))
        return frozenset()

    @staticmethod
    def arrival_of(instruction: Instruction, state: State) -> frozenset:
        """Incoming-argument provenance of a computed 64-bit value: a move keeps it, add/sub of a constant shifts
        its offset, csel joins its alternatives; any other computation yields no incoming provenance."""
        base, sources = instruction.base, instruction.operands[1:]
        if base == "mov" and len(sources) == 1 and gp(sources[0]):
            return state.arrival(gp(sources[0]))
        if base in CONDITIONAL_SELECTS and len(sources) >= 2:
            return state.arrival(gp(sources[0])) | state.arrival(gp(sources[1]))
        if base in ("add", "adds", "sub", "subs") and len(sources) in (2, 3) and gp(sources[0]):
            amount = immediate(sources[1])
            if amount is not None and len(sources) == 3:
                shift = SHIFT.match(sources[2])
                amount = amount << int(shift.group(2) or 0) if shift and shift.group(1) == "lsl" else None
            values = state.arrival(gp(sources[0]))
            if amount is None or values == NOT_STACK:
                return NOT_STACK
            sign = 1 if base.startswith("add") else -1
            return frozenset(value if value == "N" or value[2] is None else ("I", value[1], value[2] + sign * amount)
                             for value in values)
        return NOT_STACK

    def compute(self, instruction: Instruction, state: State) -> bool:
        """Register instructions: propagate stack provenance; FP writes clobber d8-d15."""
        base = instruction.base
        operands = instruction.operands
        if base in NO_DESTINATION or not operands:
            return self.reads(instruction, state, operands)
        destination = operands[0]
        merges_destination = (base in MERGE_DEFINITIONS or operand_kind(destination) == "v[]" or
                              (base in {"bic", "orr"} and len(operands) > 1 and operand_kind(operands[1]) == "imm"))
        inputs = operands if merges_destination else operands[1:]
        if not self.reads(instruction, state, inputs):
            return False
        values = [state.value(gp(operand)) for operand in inputs if gp(operand)]
        stack_inputs = any(provenance(value) != "N" for value in values)
        if stack_inputs and any(operand_kind(operand) == "reloc" for operand in inputs):
            self.fail(instruction, "relocated arithmetic on a stack address")
            return False
        registers = fp_registers(destination)
        if registers is not None:
            if stack_inputs:
                self.fail(instruction, "a stack address moves into an FP/SIMD register")
                return False
            if base in GP_TO_FP_MOVES and self.report and \
                    any(STACK_DATA in state.tainted(gp(operand) or gp_view(operand)) for operand in operands[1:]):
                self.spills += 1  # Stack data moved unchanged into an FP/SIMD register: an FP/SIMD reload.
                self.tally("fp")
            if writes_callee_saved_half(destination):
                for number, _ in registers:
                    self.clobber(instruction, state, number)
            return True
        target = gp(destination)
        arrival = self.arrival_of(instruction, state) if target else NOT_STACK
        state.define(target or gp_view(destination), self.origins(instruction, state))
        state.arrive(target or gp_view(destination), arrival)
        if base == "mov" and len(operands) == 2 and immediate(operands[1]) is not None and \
                (target or gp_view(destination)) not in (None, "xzr", "sp"):
            value = immediate(operands[1])
            state.consts[target or gp_view(destination)] = value if target else value & 0xFFFFFFFF
        if target is None:
            if gp_view(destination) is None:
                self.fail(instruction, f"unclassifiable destination {destination}")
                return False
            if stack_inputs:
                self.fail(instruction, "a stack address is truncated to 32 bits")
                return False
            state.assign(gp_view(destination), NOT_STACK)
            return True
        if not stack_inputs:
            state.assign(target, NOT_STACK)
            return True
        sources = operands[1:]
        if base == "mov" and len(sources) == 1:
            state.assign(target, values[0])
            return True
        if base in CONDITIONAL_SELECTS:
            selected = state.value(gp(sources[0]) or "xzr") | state.value(gp(sources[1]) or "xzr")
            state.assign(target, frozenset(value if value == "N" else ("S", value[1]) for value in selected))
            return True
        if base in ("add", "adds", "sub", "subs") and len(values) == 1 and gp(sources[0]) and len(sources) <= 3:
            amount = immediate(sources[1]) if len(sources) >= 2 else None
            if amount is not None and len(sources) == 3:
                shift = SHIFT.match(sources[2])
                amount = amount << int(shift.group(2) or 0) if shift and shift.group(1) == "lsl" else None
            if amount is not None:
                sign = 1 if base.startswith("add") else -1
                state.assign(target, frozenset(value if value == "N" or value[1] is None
                                               else ("S", value[1] + sign * amount) for value in values[0]))
                return True
        # Any other computation on a stack address: still a stack address, at an unknown offset.
        combined = UNKNOWN_STACK
        if any(provenance(value) == "A" for value in values):
            combined = combined | NOT_STACK
        state.assign(target, combined)
        return True


def proven_noreturn(symbol: str, bodies: dict[str, list[str]], dialect: Dialect, imports: set[str],
                    prefix: str) -> bool:
    """The fatal diagnostic is exempt only if every reachable instruction is an admitted form, its only calls are
    the trusted diagnostic calls, and its body is acyclic with every path ending in a call to abort()."""
    body = bodies.get(symbol)
    if body is None:
        return False
    blocks, errors = build_cfg(body, dialect)
    if errors:
        return False
    labels = {label: index for index, block in enumerate(blocks) for label in block.labels}
    successors: dict[int, list[int] | None] = {}
    for index, block in enumerate(blocks):
        edges, ends, valid = [], False, True
        for instruction in block.instructions:
            target = instruction.operands[-1] if instruction.operands else ""
            if not instruction.admitted():
                valid = False
                break
            if instruction.base == "bl":
                if target in labels or dialect.local(target):
                    valid = False
                    break
                if target not in imports:
                    valid = False  # Trusted only as an undefined, strongly bound import; never a local body.
                    break
                if target in {prefix + leaf for leaf in NORETURN_LEAVES}:
                    ends = True
                    break
                if target not in {prefix + call for call in DIAGNOSTIC_CALLS}:
                    valid = False
                    break
                continue
            if instruction.base in ("b", "b.cond", "cbz", "cbnz", "tbz", "tbnz"):
                if target not in labels:
                    valid = False
                    break
                edges.append(labels[target])
                if instruction.base == "b":
                    ends = True
                    break
            elif instruction.base == "ret":
                valid = False
                break
        if not valid:
            successors[index] = None
            continue
        if not ends:
            if index + 1 >= len(blocks):
                successors[index] = None
                continue
            edges.append(index + 1)
        successors[index] = edges
    colors: dict[int, str] = {}

    def visit(index: int) -> bool:
        if successors.get(index) is None or colors.get(index) == "active":
            return False
        if colors.get(index) == "done":
            return True
        colors[index] = "active"
        result = all(visit(successor) for successor in successors[index])
        colors[index] = "done"
        return result

    return visit(0)


def analysed(name: str, bodies: dict[str, list[str]], dialect: Dialect, noreturn: set[str],
             model: CallModel | None = None, unit: Unit | None = None) -> Analysis:
    """Analyse one top-level program; with a call model (call_model()) its calls resolve under the call model."""
    analysis = Analysis(name, bodies[name], dialect, noreturn, model, unit,
                        eh=getattr(bodies, "eh", {}).get(name, {}))
    if not any(block.instructions for block in analysis.blocks):
        analysis.errors.append("empty or unclassifiable body")
    else:
        analysis.run()
    return analysis


def analyse_function(name: str, bodies: dict[str, list[str]], dialect: Dialect,
                     noreturn: set[str], model: CallModel | None = None,
                     unit: Unit | None = None) -> tuple[int, list[str]]:
    """Spills and structural errors of one tile program."""
    analysis = analysed(name, bodies, dialect, noreturn, model, unit)
    return analysis.spills, analysis.errors


def traffic_record(analysis: Analysis) -> dict:
    """Stack traffic of one analysed program: its natural loops and the per-class, per-loop static counts.

    Loops are named by instruction indices in the order object_instructions() uses (labels, markers and nops
    dropped): the first instruction of every entry block (one for a reducible loop: its header) and the last
    instruction of every latch (the back-edge branch, or the instruction before a fall-through into an entry).
    """
    index, positions, starts = 0, {}, {}
    for number, block in enumerate(analysis.blocks):
        real, starts[number] = [], index
        for instruction in block.instructions:
            if instruction.base != "nop":
                real.append(index)
                index += 1
        positions[number] = real

    def first(block: int) -> int:
        # A block of padding nops only names the next instruction.
        return positions[block][0] if positions[block] else starts[block]

    def last(block: int) -> int:
        # A latch of padding nops only falls through: it is named by the instruction before it.
        return positions[block][-1] if positions[block] else starts[block] - 1
    loops = [{"id": loop["id"], "parent": loop["parent"], "depth": loop["depth"],
              "entries": [first(entry) for entry in loop["entries"]],
              "latches": [last(latch) for latch in loop["latches"]]} for loop in analysis.loops]
    # Helper loops (call model): nested under the call site's loop, named by the helper's raw symbol and loop id,
    # with the helper role entry the call was admitted under (validated against its source by the stack table).
    loops += [dict(loop) for loop in analysis.helper_loops]
    counts = {kind: {str(loop): count for loop, count in sorted(analysis.traffic[kind].items())} for kind in TRAFFIC}
    record = {"loops": loops, "counts": counts}
    if analysis.bounded:
        record["bounded"] = sorted(analysis.bounded, key=lambda site: site["index"])  # M-1..M-3 sites.
    if analysis.cold_traffic:
        record["cold"] = dict(sorted(analysis.cold_traffic.items()))  # Listed only: never-returning paths.
    return record


def analyse_traffic(name: str, bodies: dict[str, list[str]], dialect: Dialect,
                    noreturn: set[str], model: CallModel | None = None,
                    unit: Unit | None = None) -> tuple[dict, list[str]]:
    """Stack traffic record and structural errors of one tile program."""
    analysis = analysed(name, bodies, dialect, noreturn, model, unit)
    return traffic_record(analysis), analysis.errors


def noreturn_symbols(bodies: dict[str, list[str]], dialect: Dialect, imports: set[str], prefix: str) -> set[str]:
    """The only symbol a tile program may call: the exact fatal diagnostic (raw linker name of the object
    format), when proven never to return."""
    return {name for name in bodies if name == prefix + FATAL_DIAGNOSTIC and
            proven_noreturn(name, bodies, dialect, imports, prefix)}


def program_fields(name: str) -> tuple[int, ...] | None:
    """(rows, family, policy, encoding, map, minimum, strips) of a run<> or runCoded<> tile program symbol; a
    coded program's map field is its source codebook and its minimum is runCoded's (Codebook == 7)."""
    match = PROGRAM.search(name)
    if match:
        return tuple(int(field) for field in match.groups())
    match = CODED.search(name)
    if match:
        rows, family, policy, codebook, strips = (int(field) for field in match.groups())
        return (rows, family, policy, CODED_ENCODING, codebook, int(codebook == 7), strips)
    return None


def program_key(name: str) -> str:
    """Stable budget key rows:family:policy:encoding:map:minimum for an installed (Strips = 0) tile program."""
    fields = program_fields(name)
    if fields is None:
        raise ValueError(f"not a tile program: {name}")
    if fields[-1] != 0:
        raise ValueError(f"tile program pins a non-installed panel width: {name}")
    return ":".join(str(field) for field in fields[:-1])


def key_tuple(match: re.Match) -> tuple:
    """Integer key of a matched program, including its Strips argument."""
    return program_fields(match.group(0))


def tile_match(name: str, prefix: str) -> re.Match | None:
    """The exact run<> or runCoded<> tile program symbol match of a raw linker name (None otherwise)."""
    if not name.startswith(prefix):
        return None
    return TILE_PROGRAM.fullmatch(name[len(prefix):]) or TILE_CODED.fullmatch(name[len(prefix):])


ENVELOPE_PROGRAMS = {
    # Row tiles x integer family the probe instantiates in each admitted ISA envelope.
    "NEON": ("1:ASIMD", "2:ASIMD", "3:ASIMD", "4:ASIMD"),
    "NEON_DOTPROD": ("1:ASIMD", "2:ASIMD", "3:ASIMD", "4:ASIMD", "1:DotProd", "2:DotProd", "3:DotProd", "4:DotProd"),
    "NEON_I8MM": ("1:ASIMD", "2:ASIMD", "3:ASIMD", "4:ASIMD", "1:DotProd", "2:DotProd", "3:DotProd", "4:DotProd",
                  "2:I8MM", "3:I8MM", "4:I8MM"),
}
TILE_RUN = re.compile(r"27CPUNativeNEONProjectionTileILm\d+EE(?:3run|6strips|5panel|8runCoded)")
TILE_PROGRAM = re.compile(r"_ZNK9llaminar23cpu16native_quantised" + PROGRAM.pattern + r"EEvv")
TILE_CODED = re.compile(r"_ZNK9llaminar23cpu16native_quantised" + CODED.pattern + r"EEvv")
# Every out-of-line member function of the tile class must be admitted: an audited tile program (run, runCoded)
# or the probe's explicit execute<Dot>() dispatcher of one inventory entry. Anything else fails closed.
TILE_MEMBER = re.compile(r"27CPUNativeNEONProjectionTileILm\d+EE")
DISPATCHER = re.compile(r"_ZNK9llaminar23cpu16native_quantised27CPUNativeNEONProjectionTileILm(\d)EE7executeILNS1_11multi_scale"
                        r"11NEONByteDotE(\d)EEEvv")


# Compact multi-scale K-loop programs (rowsProgram<OrderedNaN, Codebook, Lanes, Rows, Tree, Packed, Ops>), the one
# exact symbol shape production instantiates: Packed = CPUNativeQuantisedWeightView, Ops = NEONMultiScaleOps<Dot>,
# optionally a GCC clone (.isra/.constprop/.part). Groups: ordered, codebook, lanes, rows, tree, dot, clone suffix.
COMPACT_PROGRAM = re.compile(
    r"_ZN9llaminar23cpu16native_quantised11multi_scale11rowsProgramILb([01])ELi(\d+)ELi(\d+)ELm([1-4])ELNS2_9Reduction"
    r"E([01])ENS1_28CPUNativeQuantisedWeightViewENS2_17NEONMultiScaleOpsILNS2_11NEONByteDotE([0-2])EEEEEvRKT4_RKSt5"
    r"arrayIPKNS_9Q8_1BlockEXT2_EERKSC_IPfXT2_EEiiibiii((?:\.(?:isra|constprop|part)\.\d+)*)")
# The develop tree's policy-free predecessor rows<Codebook, Lanes, Rows, Tree, Packed, Ops>() (one program serves
# both policies; its key's ordered field is "-"), in the shape g++-14 emits for it (confirmed on a develop
# instantiation). Groups: codebook, lanes, rows, tree, dot, clone suffix.
COMPACT_DEVELOP = re.compile(
    r"_ZN9llaminar23cpu16native_quantised11multi_scale4rowsILi(\d+)ELi(\d+)ELm([1-4])ELNS2_9ReductionE([01])ENS1_28"
    r"CPUNativeQuantisedWeightViewENS2_17NEONMultiScaleOpsILNS2_11NEONByteDotE([0-2])EEEEEvRKT3_RKSt5arrayIPKNS_9"
    r"Q8_1BlockEXT1_EERKSC_IPfXT1_EEiiibiii((?:\.(?:isra|constprop|part)\.\d+)*)")
# Any symbol of either compact program template: it must match one exact shape (compact_programs()).
COMPACT_MEMBER = re.compile(r"11multi_scale(?:11rowsProgram|4rows)I")


def compact_program_key(name: str, prefix: str = "") -> str | None:
    """Stable key compact:codebook:lanes:rows:tree:ordered:dot of a rowsProgram symbol (None if not one); the
    clone suffix is not part of the key (a TU holds one copy per key, compact_programs())."""
    raw = name[len(prefix):] if name.startswith(prefix) else None
    match = COMPACT_PROGRAM.fullmatch(raw) if raw is not None else None
    if match:
        ordered, codebook, lanes, rows, tree, dot, _ = match.groups()
        return f"compact:{codebook}:{lanes}:{rows}:{tree}:{ordered}:{dot}"
    match = COMPACT_DEVELOP.fullmatch(raw) if raw is not None else None
    if match:
        codebook, lanes, rows, tree, dot, _ = match.groups()
        return f"compact:{codebook}:{lanes}:{rows}:{tree}:-:{dot}"
    return None


def compact_programs(image: ObjectImage) -> dict[str, tuple[str, str]]:
    """Per compact program key, the (raw symbol, binding) of the TU's one defined copy. Any other rowsProgram
    symbol shape, an undefined or non-function rowsProgram and two copies of one key fail. A local copy is audited
    in this TU (CallModel); a weak/global one needs a link-selection record like any other helper."""
    found: dict[str, tuple[str, str]] = {}
    for symbol in image.symbols:
        if symbol.kind == "section" or not COMPACT_MEMBER.search(symbol.name):
            continue
        key = compact_program_key(symbol.name, image.prefix)
        if key is None or symbol.kind != "function":
            raise ValueError(f"unrecognized compact program symbol ({symbol.kind}): {symbol.name}")
        if key in found:
            raise ValueError(f"compact program {key} is defined twice in one object: {symbol.name}")
        found[key] = (symbol.name, symbol.binding)
    return found


def audit_dispatcher(name: str, body: list[str], rows: int, leaf: int, programs: dict[tuple, str]) -> list[str]:
    """The execute<Dot>() dispatcher admits no loop and no branch, tail call or call into the tile class other
    than an audited tile program of its own rows and integer family (calls outside the class - the compact
    implementation, the FPCR profile, the stack protector - are its own concern, not tile code)."""
    analysis = Analysis(name, body, GENERIC, set())
    analysis.link()
    analysis.find_loops()
    errors = list(analysis.errors)
    if analysis.loops:
        errors.append("the dispatcher holds a loop")
    admitted = {symbol for key, symbol in programs.items() if key[0] == rows and key[1] == leaf}
    for block in analysis.blocks:
        for instruction in block.instructions:
            if instruction.base not in BRANCHES or not instruction.operands:
                continue
            target = instruction.operands[-1]
            if TILE_MEMBER.search(target) and target not in admitted:
                errors.append(f"the dispatcher branches to an unaudited tile member: {target}")
    return [f"{name}: {error}" for error in errors]


# Per-program parallel analysis (V2). Each tile program's analysis and T-1 chain witness is an independent pure
# function of the parsed object; map_programs() runs that same serial reference function per program in forked
# worker processes and merges the results in input order, so every field is byte-identical to the serial run.
# Workers: one per physical core of the admitted CPU set (physical_workers()), each pinned to one logical CPU of
# its own core; never a fixed count and never hyperthread siblings. A worker's exception is re-raised exactly as
# the serial run would raise it (the first failing program in order); a worker that dies, returns a missing or a
# duplicate result fails the whole analysis: no partial result is ever used.
def physical_workers(admitted: list[int] | None = None, root: Path = Path("/sys/devices/system/cpu")) -> dict:
    """The analysis worker observation: Linux - the admitted CPU set (sched_getaffinity) reduced to one logical CPU
    per physical core (sysfs topology: package and core id); macOS - the documented equivalent hw.physicalcpu (no
    affinity API). Without a topology the analysis stays serial (one worker), never one worker per logical CPU."""
    if admitted is None and hasattr(os, "sched_getaffinity"):
        admitted = sorted(os.sched_getaffinity(0))
    if admitted is not None:
        cores: dict[tuple[int, int], int] = {}
        for cpu in sorted(admitted):
            try:
                package = int((root / f"cpu{cpu}" / "topology" / "physical_package_id").read_text())
                core = int((root / f"cpu{cpu}" / "topology" / "core_id").read_text())
            except (OSError, ValueError):
                return {"method": "affinity without topology (serial)", "admitted": len(admitted), "workers": 1,
                        "cpus": []}
            cores.setdefault((package, core), cpu)
        cpus = sorted(cores.values())
        return {"method": "sched_getaffinity + sysfs topology", "admitted": len(admitted), "workers": len(cpus),
                "cpus": cpus}
    try:
        count = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        count = 1
    return {"method": "sysctl hw.physicalcpu", "admitted": None, "workers": max(1, count), "cpus": []}


def snapshot(analysis) -> SimpleNamespace:
    """The fields of an Analysis its consumers read after the run (loop annotations, spills, errors, traffic), as
    plain picklable values: blocks and instructions become namespaces of their attributes."""
    blocks = [SimpleNamespace(**{**vars(block), "instructions": [SimpleNamespace(**vars(instruction))
                                                                 for instruction in block.instructions]})
              for block in analysis.blocks]
    return SimpleNamespace(name=analysis.name, blocks=blocks, errors=list(analysis.errors), spills=analysis.spills,
                           loops=analysis.loops, innermost=analysis.innermost, helper_loops=analysis.helper_loops,
                           traffic=analysis.traffic, cold_traffic=analysis.cold_traffic)


def map_programs(function, items: list, workers: int | None = None) -> list:
    """[function(item) for item in items], computed by forked workers when more than one is admitted.

    Items are dealt to the workers interleaved (worker w: items w, w + n, ...); a worker stops at its first failing
    item. The merge is by input position: the first failure in input order is re-raised (all items before it were
    computed by some worker), else every position must hold exactly one result."""
    observation = physical_workers() if workers is None else None
    count = min(observation["workers"] if observation else workers, len(items))
    if count <= 1 or not hasattr(os, "fork"):
        return [function(item) for item in items]
    cpus = observation["cpus"] if observation else []
    children: list[tuple[int, int]] = []
    reaped: set[int] = set()
    parent = os.getpid()
    try:
        return _run_workers(function, items, count, cpus, parent, children, reaped)
    finally:
        # W1: on any exit - a result, an exception, KeyboardInterrupt - no worker outlives this call: every
        # unreaped worker is killed and reaped (a finished one is only reaped).
        for pid, read in children:
            if pid in reaped:
                continue
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
            with contextlib.suppress(OSError):
                os.close(read)


def _die_with_parent(parent: int) -> None:
    """In a worker: die with the parent. Linux: PR_SET_PDEATHSIG = SIGKILL, then re-check getppid() (the parent may
    have died before the call). Everywhere: a daemon thread polls getppid() and exits the worker when the parent
    is gone (reparented)."""
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != parent:
        os._exit(1)

    def watch() -> None:
        while True:
            if os.getppid() != parent:
                os._exit(1)
            time.sleep(0.05)
    threading.Thread(target=watch, daemon=True).start()


def _run_workers(function, items: list, count: int, cpus: list[int], parent: int, children: list,
                 reaped: set) -> list:
    for worker in range(count):
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:  # Worker: never returns into the caller.
            os.close(read)
            status = 0
            try:
                _die_with_parent(parent)
                if worker < len(cpus) and hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, {cpus[worker]})
                done = []
                for index in range(worker, len(items), count):
                    try:
                        done.append((index, True, function(items[index])))
                    except Exception as error:
                        done.append((index, False, error))
                        break
                try:
                    payload = pickle.dumps(done)
                except Exception as error:
                    payload = pickle.dumps([(-1, False, ValueError(f"an analysis worker result cannot be "
                                                                   f"transferred: {error}"))])
                with os.fdopen(write, "wb") as handle:
                    handle.write(payload)
            except BaseException:
                status = 1
            finally:
                os._exit(status)
        os.close(write)
        children.append((pid, read))
    payloads, failures = [], []
    for pid, read in children:
        handle = os.fdopen(read, "rb", closefd=False)
        payloads.append(handle.read())
        _, status = os.waitpid(pid, 0)
        reaped.add(pid)
        os.close(read)
        if status:
            failures.append(f"worker {pid} exit status {status}")
    if failures:
        raise ValueError(f"an analysis worker died ({'; '.join(failures)}): no partial result is used")
    results: dict[int, object] = {}
    first_failure = None
    for payload in payloads:
        try:
            done = pickle.loads(payload)
        except Exception as error:
            raise ValueError(f"an analysis worker returned no readable result ({error})") from None
        for index, succeeded, value in done:
            if not succeeded:
                if index < 0:
                    raise value
                if first_failure is None or index < first_failure[0]:
                    first_failure = (index, value)
                continue
            if index in results:
                raise ValueError(f"an analysis worker returned item {index} twice")
            results[index] = value
    if first_failure is not None:
        if any(index not in results for index in range(first_failure[0])):
            raise ValueError("an analysis worker result is missing before the first failure")
        raise first_failure[1]
    if sorted(results) != list(range(len(items))):
        raise ValueError("an analysis worker result is missing")
    return [results[index] for index in range(len(items))]


def audit(data: bytes, disassembly: str, mode: str, traffic: dict | None = None, selection: dict | None = None,
          helper_roles: dict | None = None, units: dict | None = None,
          analyses: dict | None = None, workers: int | None = None) -> dict[str, int]:
    """Per-program FP/SIMD spill counts of the tile programs actually assembled into the witness object.

    The expected programs derive from the admitted envelope; the object's embedded inventory, its function
    symbols and the complete disassembly of its executable sections must all agree. Structural failures
    (missing/duplicate/unexpected/empty programs, incomplete coverage, unsupported calls, indirect branches,
    unclassifiable forms, ambiguous or unclassifiable stack addressing, unproven ABI saves) are fatal and
    never paid from a spill budget.
    """
    if mode not in ENVELOPE_PROGRAMS:
        raise ValueError(f"no tile program inventory for ISA envelope {mode}")
    image = read_object(data)
    inventory = object_inventory(data)
    if inventory != [",".join(ENVELOPE_PROGRAMS[mode])]:
        raise ValueError(f"probe instantiation inventory {inventory} does not match the {mode} envelope")
    expected = set()
    for entry in ENVELOPE_PROGRAMS[mode]:
        rows, dot = entry.split(":")
        leaf = 1 if dot == "I8MM" and rows == "1" else DOTS[dot]
        for policy in (0, 1):
            for encoding, nibble_map, minimum in DISPATCH:
                expected.add((int(rows), leaf, policy, encoding, nibble_map, minimum, 0))
            for codebook, minimum in CODED_DISPATCH:
                expected.add((int(rows), leaf, policy, CODED_ENCODING, codebook, minimum, 0))
    bodies = normalize_disassembly(disassembly, image)
    tile_symbols = sorted({symbol.name for symbol in image.symbols
                           if symbol.kind != "section" and TILE_RUN.search(symbol.name)})
    for symbol in tile_symbols:
        if symbol not in bodies:
            raise ValueError(f"tile program symbol is not a function in an executable section: {symbol}")
    noreturn = noreturn_symbols(bodies, GENERIC, trusted_imports(image), image.prefix)
    found: dict[tuple, str] = {}
    for name in tile_symbols:
        match = tile_match(name, image.prefix)
        if not match:
            raise ValueError(f"unrecognized tile program symbol: {name}")
        key = key_tuple(match)
        if key in found:
            raise ValueError(f"tile program appears twice: {name}")
        if key not in expected:
            raise ValueError(f"tile program outside the {mode} inventory: {name}")
        found[key] = name
    missing = expected - found.keys()
    if missing:
        raise ValueError(f"register witness is missing {len(missing)} expected tile programs, e.g. {sorted(missing)[:3]}")
    dispatchers = set()
    for name in sorted(bodies):
        if not TILE_MEMBER.search(name) or name in tile_symbols:
            continue
        match = DISPATCHER.fullmatch(name[len(image.prefix):]) if name.startswith(image.prefix) else None
        if not match:
            raise ValueError(f"out-of-line tile member outside the audited programs and dispatchers: {name}")
        rows, dot = int(match.group(1)), int(match.group(2))
        entry = f"{rows}:{next(label for label, value in DOTS.items() if value == dot)}"
        if entry not in ENVELOPE_PROGRAMS[mode] or entry in dispatchers:
            raise ValueError(f"tile dispatcher outside the {mode} inventory or duplicated: {name}")
        dispatchers.add(entry)
        leaf = 1 if dot == DOTS["I8MM"] and rows == 1 else dot
        errors = audit_dispatcher(name, bodies[name], rows, leaf, found)
        if errors:
            raise ValueError(f"tile dispatcher is not a proven dispatcher: {errors[:3]}")
    if dispatchers != set(ENVELOPE_PROGRAMS[mode]):
        raise ValueError(f"tile dispatchers {sorted(dispatchers)} do not match the {mode} inventory")
    result: dict[str, int] = {}
    structural: list[str] = []
    model, unit = call_model(image, bodies, selection=selection, helper_roles=helper_roles, units=units)
    keys = sorted(expected)

    def one(key: tuple) -> tuple:
        # One program's analysis (the serial reference computation, also run per program in a worker, V2).
        analysis = analysed(found[key], bodies, GENERIC, noreturn, model, unit)
        return (analysis.spills, list(analysis.errors), traffic_record(analysis) if traffic is not None else None,
                snapshot(analysis) if analyses is not None else None)
    for key, (spills, errors, record, view) in zip(keys, map_programs(one, keys, workers)):
        program = found[key]
        result[program] = spills
        if analyses is not None:
            analyses[program] = view  # The caller's single analysis (V1): reused, never recomputed.
        if traffic is not None:
            traffic[program_key(program)] = record
        structural.extend(f"{program_key(program)}: {error}" for error in errors)
    if structural:
        raise ValueError(f"native NEON tile witness is structurally unprovable ({len(structural)}): "
                         + "; ".join(structural[:6]))
    return result


ASM_BLOCK_START = re.compile(r"^\s*(?:#|//|;)\s*(?:APP|InlineAsm Start)\s*$")
ASM_BLOCK_END = re.compile(r"^\s*(?:#|//|;)\s*(?:NO_APP|InlineAsm End)\s*$")
ASM_LABEL_LINE = re.compile(r"^([^\s:#;/\\\"]+):\s*(?:(?://|;|#).*)?$")
ORDERED_ADD = re.compile(r"^fadd\s+v(\d+)\.4s,\s*v\1\.4s,\s*v(\d+)\.4s$")


ASM_FILE = re.compile(r'^\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
ASM_LOC = re.compile(r"^\.loc\s+(\d+)\s+(\d+)")
SOURCE_ROOTS = ("src/v2/", "tests/v2/")


def source_name(path: str) -> str:
    """Tree-independent spelling of a source path: from its src/v2/ or tests/v2/ component, else the base name.
    A path with ./.. components (a production TU's relative include, e.g. src/v2/a/b/../../kernels/x.h) is
    resolved lexically first, so one source file has one name whichever directory includes it."""
    if "/../" in path or "/./" in path or path.startswith(("../", "./")):
        path = posixpath.normpath(path)
    for root in SOURCE_ROOTS:
        if path.startswith(root):
            return path
        position = path.rfind("/" + root)
        if position >= 0:
            return path[position + 1:]
    return path.rsplit("/", 1)[-1]


def assembly_functions(assembly: str, names: set[str], macho: bool = False,
                       locations: dict[str, list[str]] | None = None) -> tuple[dict[str, list[tuple[str, int]]],
                                                                               list[str]]:
    """Memoised _assembly_functions() (shallow copies per call)."""
    found: dict[str, list[str]] = {}
    key = ("assembly", hashlib.sha256(assembly.encode()).digest(), frozenset(names), macho)
    functions, stray, where = _memo(key, lambda: _assembly_functions(assembly, names, macho, found) + (found,))
    if locations is not None:
        locations.update({name: list(lines) for name, lines in where.items()})
    return {name: list(lines) for name, lines in functions.items()}, list(stray)


def _assembly_functions(assembly: str, names: set[str], macho: bool = False,
                        locations: dict[str, list[str]] | None = None) -> tuple[dict[str, list[tuple[str, int]]],
                                                                                list[str]]:
    """Instructions of each named function of compiler assembly output, with their inline-asm block number (0 outside).

    A function runs from its label to the next label naming another object function; local labels,
    directives, comments and blank lines are not instructions. ELF assembly separates statements on one
    line with ';' (GNU as for AArch64); in Mach-O assembly ';' starts a comment. Returns the bodies and the
    inline-asm blocks found outside every named function. When `locations` is given, it receives for every
    instruction of every named function the source location (file:line, see source_name()) of the last `.loc`
    directive before it, or "?" when none precedes it.
    """
    bodies: dict[str, list[tuple[str, int]]] = {}
    current, inside, block, stray = None, False, 0, []
    files: dict[str, str] = {}
    # source_name() of each .file entry, resolved once per .file directive instead of once per .loc (a file number
    # never seen resolves as source_name("?") == "?", exactly as files.get(number, "?") did).
    resolved: dict[str, str] = {}
    location = "?"
    start_match, end_match, label_match = ASM_BLOCK_START.match, ASM_BLOCK_END.match, ASM_LABEL_LINE.match
    file_match, loc_match = ASM_FILE.match, ASM_LOC.match
    for raw in assembly.splitlines():
        # Fast path (exact): a line starting with tab + "." is a directive. No inline-asm marker can match it (its
        # first non-space character is "."), it is no label (it starts with a tab), and its text starts with ".",
        # so after the .file/.loc bookkeeping the full path below would skip it. Only .file/.loc lines are read.
        if raw.startswith("\t."):
            if raw.startswith(("\t.file", "\t.loc")):
                line = raw.split("\t//", 1)[0].rstrip() if "\t//" in raw else raw.rstrip()
                text = line.strip()
                directive = file_match(text)
                if directive:
                    files[directive.group(1)] = directive.group(3) if directive.group(3) is not None \
                        else directive.group(2)
                    resolved[directive.group(1)] = source_name(files[directive.group(1)])
                directive = loc_match(text)
                if directive:
                    location = f"{resolved.get(directive.group(1), '?')}:{directive.group(2)}"
            continue
        first = raw[:1]
        # An inline-asm marker needs optional whitespace then '#', '//' or ';' (re \s is str.isspace()).
        marker = not first or first.isspace() or first in "#/;"
        if "\t//" in raw and not (marker and start_match(raw)):
            line = raw.split("\t//", 1)[0].rstrip()
        else:
            line = raw.rstrip()
        if marker and start_match(line):
            if inside:
                raise ValueError("nested inline-asm block in the probe assembly")
            inside, block = True, block + 1
            if current is None:
                stray.append(line)
            continue
        if marker and end_match(line):
            if not inside:
                raise ValueError("unbalanced inline-asm block end in the probe assembly")
            inside = False
            continue
        label = label_match(line.strip()) if not line.startswith(("\t", " ")) else None
        if label:
            if label.group(1) in names:
                if inside:
                    raise ValueError("a function label inside an inline-asm block")
                current = label.group(1)
                bodies.setdefault(current, [])
            continue
        text = line.strip()
        directive = file_match(text)
        if directive:
            files[directive.group(1)] = directive.group(3) if directive.group(3) is not None else directive.group(2)
            resolved[directive.group(1)] = source_name(files[directive.group(1)])
        directive = loc_match(text)
        if directive:
            location = f"{resolved.get(directive.group(1), '?')}:{directive.group(2)}"
        if not text or text.startswith((".", "#", "//", ";", "@")):
            continue
        statements = [text.split(";", 1)[0].strip()] if macho else [part.strip() for part in text.split(";")]
        for statement in statements:
            if not statement or statement.startswith((".", "//", "#")):
                if statement.startswith(".") and inside:
                    raise ValueError(f"directive inside an inline-asm block: {statement}")
                continue
            if current is None:
                if inside:
                    stray.append(statement)
                continue
            bodies[current].append((" ".join(statement.split()), block if inside else 0))
            if locations is not None:
                locations.setdefault(current, []).append(location)
    if inside:
        raise ValueError("unterminated inline-asm block in the probe assembly")
    return bodies, stray


def object_instructions(body: list[str]) -> list[str]:
    """Instruction texts of one normalised object body (labels, markers and alignment nops dropped).

    The assembler pads `.p2align` inside a function with nop; the compiler assembly names no nop of its
    own in these functions, so both sides are compared without nops.
    """
    return [" ".join(line.split()) for line in body
            if line.startswith("\t") and not line.startswith(("\t.section", "\t.data-mapping-symbol"))
            and line.strip() != "nop"]


def ordered_add_evidence(data: bytes, disassembly: str, assembly: str) -> dict[str, list[int]]:
    """Ordered-add sites of every tile program, proven against the bound object.

    Every inline-asm block of the compiler's assembly must be exactly one in-place `fadd vD.4s, vD.4s,
    vM.4s` inside a function of the object; such a function must have the object's instruction count, and
    the object must hold the identical instruction at the same index. GPU-aligned tile programs (policy 1)
    need at least one site; backend-native ones none. Returns raw tile symbol -> site indices.
    """
    image, bodies, sites = bound_ordered_sites(data, disassembly, assembly)
    tiles = [name for name in bodies if TILE_RUN.search(name)]
    for name in tiles:
        match = tile_match(name, image.prefix)
        if not match:
            continue  # audit() rejects unrecognised tile symbols.
        policy = int(match.group(3))
        if policy == 1 and not sites.get(name):
            raise ValueError(f"GPU-aligned tile program without an ordered add: {program_key(name)}")
        if policy == 0 and sites.get(name):
            raise ValueError(f"backend-native tile program holds an ordered add: {program_key(name)}")
    return {program_key(name): indices for name, indices in sorted(sites.items()) if name in tiles}


def bound_ordered_sites(data: bytes, disassembly: str, assembly: str) -> tuple[ObjectImage, dict, dict[str, list[int]]]:
    """(image, bodies, raw function -> inline-asm ordered-add site indices), every site bound to the object (the
    rules of ordered_add_evidence())."""
    image = read_object(data)
    bodies = normalize_disassembly(disassembly, image)
    functions, stray = assembly_functions(assembly, set(bodies), image.macho)
    if stray:
        raise ValueError(f"inline assembly outside every object function: {stray[:2]}")
    sites: dict[str, list[int]] = {}
    for name, lines in functions.items():
        # GCC opens one inline-asm region for consecutive asm statements, so a region may hold several
        # expansions. The translation-unit check admits only the one-instruction template, so every
        # instruction of a region must be one in-place fadd .4s (judged on the compiler's own text,
        # before any nop is set aside).
        blocks = [[position] for position, (_, block) in enumerate(lines) if block]
        if not blocks:
            continue
        for block in blocks:
            text = lines[block[0]][0]
            if not ORDERED_ADD.match(text):
                raise ValueError(f"{name}: inline-asm instruction is not an in-place fadd .4s: {text!r}")
        if any(text == "nop" for text, _ in lines):
            raise ValueError(f"{name}: compiler assembly names a nop; alignment padding cannot be told apart")
        compiled = object_instructions(bodies[name])
        if len(compiled) != len(lines):
            raise ValueError(f"{name}: compiler assembly ({len(lines)}) and object ({len(compiled)}) instruction "
                             f"counts differ; the ordered-add sites cannot be bound")
        indices = []
        for block in blocks:
            match = ORDERED_ADD.match(lines[block[0]][0])
            emitted = ORDERED_ADD.match(compiled[block[0]])
            if not emitted or emitted.groups() != match.groups():
                raise ValueError(f"{name}: the object does not hold the ordered add at instruction {block[0]}: "
                                 f"{compiled[block[0]]!r} != {lines[block[0]][0]!r}")
            indices.append(block[0])
        sites[name] = indices
    return image, bodies, sites


COMPILER_FADD = re.compile(r"^fadd ([sdv])(\d+)(\.\d*[bhsd])?, [sdv](\d+)(?:\.\d*[bhsd])?, [sdv](\d+)(?:\.\d*[bhsd])?$")
FP_COPY = re.compile(r"^(?:mov v(\d+)\.16b, v(\d+)\.16b|orr v(\d+)\.16b, v(\d+)\.16b, v\4\.16b|fmov ([sd])(\d+), \5(\d+))$")
FP_ZERO = re.compile(r"^(?:movi [vd](\d+)(?:\.\d*[bhsd])?, #0(?:x0)?|fmov s(\d+), wzr|fmov d(\d+), xzr)$")
FRAME_SLOT = re.compile(r"^\[(sp|x29), #(-?\d+)\]$")
# The Contiguous accumulator seed statement of rowsProgram (S2-4a), compared on the located source line with all
# whitespace removed: `acc[row] = load<F>(output[row] + column);`.
SEED_STATEMENT = "acc[row]=load<F>(output[row]+column);"
EXPLICIT_FMA = re.compile(r"\b(?:vfmaq?_\w+|vfmsq?_\w+|std::fmaf?|__builtin_fmaf?|__builtin_aarch64_fn?ma\w*)\s*\(")
CONTRACTIBLE = re.compile(r"\bvmul\w*|\bvadd\w*|\bvsub\w*|[\w)\]]\s*\*\s*[\w(]")
FUSED = frozenset({"fmla", "fmls", "fmadd", "fmsub", "fnmadd", "fnmsub"})
GRANULE = 4


def program_instructions(name: str, body: list[str]) -> tuple[list["Instruction"], list[list[int]], list[int]]:
    """(instructions without nops, in object_instructions() order; successors per instruction index; block-start
    indices) of one normalised body, from the same CFG Analysis builds (calls fall through)."""
    analysis = Analysis(name, body, GENERIC, set())
    analysis.link()
    if analysis.errors:
        raise ValueError(f"{name}: {analysis.errors[0]}")
    instructions, first_of, last_of = [], {}, {}
    for number, block in enumerate(analysis.blocks):
        first_of[number] = len(instructions)
        instructions += [instruction for instruction in block.instructions if instruction.base != "nop"]
        last_of[number] = len(instructions) - 1
    successors: list[list[int]] = [[index + 1] for index in range(len(instructions))]
    for number, block in enumerate(analysis.blocks):
        end = last_of[number]
        if end < first_of[number]:
            continue  # An empty (nop-only) block forwards to the next real instruction.
        targets = []
        for successor in block.successors:
            while successor >= 0 and successor < len(analysis.blocks) and last_of[successor] < first_of[successor]:
                successor += 1
            targets.append(first_of[successor] if 0 <= successor < len(analysis.blocks) else None)
        successors[end] = sorted({target for target in targets if target is not None})
    if successors and successors[-1] == [len(instructions)]:
        successors[-1] = []
    starts = sorted({first_of[number] for number, block in enumerate(analysis.blocks) if last_of[number] >= first_of[number]})
    return instructions, successors, starts


def _granules(base: str, offset: int, width: int) -> list[str]:
    return [f"{base}:{offset - offset % GRANULE + step}" for step in range(0, max(width, 1), GRANULE)]


def fp_effects(instruction: "Instruction", index: int, instructions: list["Instruction"], prefix: str) \
        -> list[tuple[str, tuple]]:
    """Definitions one instruction makes for the order proof: (key, value) with keys v<n> (FP/SIMD registers) and
    frame granules <base>:<offset>; values ("site"|"fadd"|"init"|"seed"|"other",) or ("copy", key...) (a
    transparent move: the reaching definitions of those keys at this instruction). ("kill", base) clears every
    granule of a frame base whose register the instruction redefines."""
    base, operands, text = instruction.base, instruction.operands, " ".join(instruction.text.split())
    effects: list[tuple[str, tuple]] = []
    registers = defined_registers(instruction)
    for frame in ("sp", "x29"):
        if frame in registers:
            effects.append(("kill", (frame,)))
    if base == "bl":
        target = operands[-1] if operands else ""
        bare = target[len(prefix):] if target.startswith(prefix) else target
        effects += [(f"v{n}", ("other",)) for n in list(range(8)) + list(range(16, 32))]
        effects.append(("kill", ("calls",)))
        if bare in MEMCPY_IMPORTS:
            # memcpy(&acc[row], ...): the copied bytes of a constant-size frame destination seed the accumulator.
            destination, size = None, None
            for previous in reversed(instructions[max(0, index - 8):index]):
                parts = " ".join(previous.text.split())
                found = re.fullmatch(r"add x0, (sp|x29), #(\d+)", parts) or re.fullmatch(r"mov x0, (sp|x29)()", parts)
                if destination is None and found:
                    destination = (found.group(1), int(found.group(2) or 0))
                found = re.fullmatch(r"mov x2, #(\d+)", parts)
                if size is None and found:
                    size = int(found.group(1))
            if destination and size:
                effects += [(key, ("seed",)) for key in _granules(destination[0], destination[1], size)]
        return effects
    if instruction.memory is not None:
        memory = FRAME_SLOT.match(operands[instruction.memory])
        data = operands[:instruction.memory]
        position = int(memory.group(2)) if memory and not operands[instruction.memory + 1:] else None
        if base in STORES and memory:
            for operand in data:
                width = footprint(instruction, operand) or GRANULE
                registers_ = fp_registers(operand) if not (gp(operand) or GP32.match(operand)) else None
                if position is None:
                    effects.append(("kill", (memory.group(1),)))
                else:
                    value = ("copy", f"v{registers_[0][0]}") if registers_ and len(registers_) == 1 else ("other",)
                    effects += [(key, value) for key in _granules(memory.group(1), position, width)]
                    position += width
        elif base in LOADS:
            for operand in data:
                registers_ = fp_registers(operand) if not (gp(operand) or GP32.match(operand)) else None
                if not registers_:
                    continue
                width = footprint(instruction, operand) or GRANULE
                if memory and position is not None and len(registers_) == 1 and LANE.match(operand) is None:
                    effects.append((f"v{registers_[0][0]}", ("copy",) + tuple(_granules(memory.group(1), position, width))))
                    position += width
                else:
                    effects += [(f"v{number}", ("load",)) for number, _ in registers_]
        return effects
    if not operands or base in NO_DESTINATION or base in BRANCHES:
        return effects
    destination = fp_registers(operands[0]) if not (gp(operands[0]) or GP32.match(operands[0])) else None
    if not destination:
        return effects
    number = destination[0][0]
    copy, zero, fadd = FP_COPY.match(text), FP_ZERO.match(text), COMPILER_FADD.match(text)
    if copy:
        source = copy.group(2) or copy.group(4) or copy.group(7)
        effects.append((f"v{number}", ("copy", f"v{source}")))
    elif zero:
        effects.append((f"v{number}", ("init",)))
    elif fadd and LANE.match(operands[0]) is None:
        effects.append((f"v{number}", ("fadd",)))
    else:
        effects += [(f"v{n}", ("other",)) for n, _ in destination]
    return effects


def reaching(instructions: list["Instruction"], successors: list[list[int]], prefix: str) \
        -> tuple[list[dict[str, frozenset]], dict[tuple[int, str], tuple]]:
    """Global reaching definitions (loop-carried and across joins) of FP registers and frame granules: per
    instruction the in-state {key: {defining instruction index, -1 = entry}} and the value of each definition."""
    values: dict[tuple[int, str], tuple] = {}
    effects = [fp_effects(instruction, index, instructions, prefix) for index, instruction in enumerate(instructions)]
    for index, items in enumerate(effects):
        for key, value in items:
            if key != "kill":
                values[(index, key)] = value
    states: list[dict[str, frozenset] | None] = [None] * len(instructions)
    if instructions:
        states[0] = {}
    work = [0] if instructions else []
    while work:
        index = work.pop()
        state = dict(states[index])
        for key, value in effects[index]:
            if key == "kill":
                prefixes = ("sp:", "x29:") if value[0] == "calls" else (f"{value[0]}:",)
                for slot in [slot for slot in state if slot.startswith(prefixes)]:
                    state[slot] = frozenset({-1})
                continue
            state[key] = frozenset({index})
        for successor in successors[index]:
            if successor >= len(instructions):
                continue
            previous = states[successor]
            merged = dict(state) if previous is None else {
                key: previous.get(key, frozenset({-1})) | state.get(key, frozenset({-1}))
                for key in set(previous) | set(state)}
            if merged != previous:
                states[successor] = merged
                work.append(successor)
    return states, values


def terminals(index: int, key: str, states: list, values: dict, seen: set | None = None) -> set:
    """The non-copy origins of the value `key` holds at instruction `index`: ("entry"), ("site", i), ("fadd", i),
    ("init", i), ("seed", i), ("other", i); copies are followed transparently (cycles contribute nothing)."""
    seen = set() if seen is None else seen
    state = states[index] or {}
    result: set = set()
    for definition in state.get(key, frozenset({-1})):
        if definition < 0:
            result.add(("entry", -1))
            continue
        if (definition, key) in seen:
            continue
        seen.add((definition, key))
        value = values[(definition, key)]
        if value[0] == "copy":
            for source in value[1:]:
                result |= terminals(definition, source, states, values, seen)
        else:
            result.add((value[0], definition))
    return result


def fadd_origins(name: str, body: list[str], sites: list[int], prefix: str, located: list[str],
                 sources: dict[str, str], seeds: tuple[str, ...] = (SEED_STATEMENT,), memcpy_seed: bool = True,
                 context: dict | None = None) -> tuple[list["Instruction"], dict[int, tuple[set, set]]]:
    """(instructions, {compiler fadd index: (origins of A, origins of B)}) of one program by global reaching
    definitions (reaching(); copies followed transparently). Origins: ("site", i) template ordered add, ("init", i)
    exact +0.0, ("seed", i) a load whose located source line (whitespace removed) ends in one of `seeds` or (when
    memcpy_seed) a frame slot filled by memcpy, ("fadd", i) a compiler fadd, ("other", i), ("entry", -1). The
    function-entry pseudo-definition reaches every register or slot not defined on every path and is never a
    chain origin."""
    instructions, successors, _ = program_instructions(name, body)
    states, values = reaching(instructions, successors, prefix)
    site_set = set(sites)
    for index in site_set:
        values[(index, f"v{int(ORDERED_ADD.match(' '.join(instructions[index].text.split())).group(1))}")] = ("site",)
    for (index, key), value in list(values.items()):
        if value == ("seed",) and not memcpy_seed:
            values[(index, key)] = ("other",)
        if value == ("load",):
            location = located[index] if index < len(located) else "?"
            file, _, line = location.rpartition(":")
            text = sources.get(file, "").split("\n")[int(line) - 1] if line.isdigit() and file in sources else ""
            compact = re.sub(r"\s+", "", text)
            values[(index, key)] = ("seed",) if any(compact.endswith(seed) for seed in seeds) else ("other",)
    if context is not None:
        context.update(states=states, values=values, sites=site_set)
    origins = {}
    for index, instruction in enumerate(instructions):
        match = COMPILER_FADD.match(" ".join(instruction.text.split()))
        if index in site_set or not match:
            continue
        _, _, _, earlier, later = match.groups()
        origins[index] = (terminals(index, f"v{earlier}", states, values),
                          terminals(index, f"v{later}", states, values))
    return instructions, origins


ANCHORS = ("site", "init", "seed")


def grounded(accepted: set, origins: dict[int, tuple[set, set]]) -> set:
    """F-1: the accepted fadds whose A chain reaches a non-fadd chain origin (template, +0.0, seed): the least
    fixpoint, so a cycle of fadds justified only by each other is never grounded."""
    result: set = set()
    changed = True
    while changed:
        changed = False
        for index in accepted - result:
            if any(kind in ANCHORS or (kind == "fadd" and source in result) for kind, source in origins[index][0]):
                result.add(index)
                changed = True
    return result


def order_verdicts(origins: dict[int, tuple[set, set]]) -> tuple[set, dict[int, str]]:
    """(proven fadds, {failing fadd: commuted|unprovable|ungrounded}): the greatest fixpoint of "every A origin is a
    chain origin (at least one) and no B origin is", then grounded() (F-1)."""
    def chain(origin: tuple, accepted: set) -> bool:
        return origin[0] in ANCHORS or (origin[0] == "fadd" and origin[1] in accepted)
    accepted = set(origins)
    while True:
        rejected = {index for index in accepted
                    if not origins[index][0] or not all(chain(origin, accepted) for origin in origins[index][0])
                    or any(chain(origin, accepted) for origin in origins[index][1])}
        if not rejected:
            break
        accepted -= rejected
    ground = grounded(accepted, origins)
    failures = {}
    for index in sorted(set(origins) - ground):
        first, second = origins[index]
        if index in accepted:
            failures[index] = "ungrounded"
        elif any(chain(origin, ground) for origin in second) and not all(chain(origin, ground) for origin in first):
            failures[index] = "commuted"
        else:
            failures[index] = "unprovable"
    return ground, failures


def prove_order(name: str, body: list[str], sites: list[int], prefix: str, located: list[str],
                sources: dict[str, str]) -> list[int]:
    """S2-4a/b: every compiler-emitted (non-template) `fadd` - scalar s/d or vector v - of a GPU-aligned compact
    program in source order, by global reaching definitions (fadd_origins()): every origin of the first operand is
    a chain value - a template site, the exact +0.0 accumulator initialisation (movi #0, fmov from wzr/xzr), the
    Contiguous output seed (SEED_STATEMENT, or a frame slot filled by memcpy) or a proven fadd - and no origin of
    the second operand is. The fadds form a greatest fixpoint, grounded (F-1): every proven fadd's A chain must
    reach a template, zero or seed origin; the entry pseudo-definition is never a chain origin. Returns the proven
    fadd indices; a commuted, unprovable or ungrounded fadd fails."""
    instructions, origins = fadd_origins(name, body, sites, prefix, located, sources)
    proven, failures = order_verdicts(origins)
    for index, verdict in failures.items():
        raise ValueError(f"{name}: compiler fadd at instruction {index} is {verdict} against source order: "
                         f"{instructions[index].text!r}")
    return sorted(proven)


LANE_MOVES = frozenset({"ins", "mov", "dup", "umov", "smov", "fmov"})
MERGING = frozenset({"bif", "bit", "bsl", "ins"}) | FUSED


def fp_reads(instruction: "Instruction") -> tuple[str, list[int]] | None:
    """("move"|"op", FP/SIMD registers read) of one instruction for the T-1 chain classification, or None when it
    reads none or is outside it: loads/stores (spills and the final output store), branches and calls. A move is
    pure data movement (whole-register copies are already transparent in reaching(); lane, scalar-view and FP->GP
    moves are moves here): it propagates the chain but never fails. Every other FP read is an op: arithmetic,
    compares, selects (fcsel, bsl/bif/bit), conversions, min/max, abs/neg and fused forms (whose destination is
    also read)."""
    base, operands = instruction.base, instruction.operands
    if instruction.memory is not None or base in BRANCHES or base == "blr" or RETURN.match(base) or not operands:
        return None
    text = " ".join(instruction.text.split())
    if FP_COPY.match(text) or FP_ZERO.match(text):
        return None
    destination = operands[0]
    lane_destination = operand_kind(destination) == "v[]"
    sources = operands if base in NO_DESTINATION or base in MERGING or lane_destination else operands[1:]
    registers = []
    for operand in sources:
        if gp(operand) or GP32.match(operand) or operand_kind(operand) in (None, "imm", "shift", "cond", "reloc", "sym"):
            continue
        found = fp_registers(operand)
        registers += [number for number, _ in found or []]
    if not registers:
        return None
    plain = all(operand_kind(operand) in ("x", "w", "v", "v[]", "b", "h", "s", "d", "q") for operand in operands)
    kind = "move" if base in LANE_MOVES and plain and len(operands) == 2 else "op"
    return kind, sorted(set(registers))


def tile_chain_evidence(data: bytes, disassembly: str, assembly: str, copy: str,
                        sources: dict[str, str] | None = None, workers: int | None = None) -> dict[str, dict]:
    """T-1: the accumulator chain of every GPU-aligned tile program of one copy is template-only. By the same global
    reaching definitions, an instruction touches the chain when an origin of an FP register it reads is a template
    site, an exact +0.0 initialisation or the result of an instruction that touches it (derived, transitively).
    The chain's zero is only an exact +0.0 initialisation that reaches a template site's earlier operand (the
    accumulator); other zero constants are not on it. Any touching FP operation - fadd, fsub, fmul, fneg, fabs, fmax/fmin, fcsel and the bit selects, conversions,
    compares, fused forms - fails: every chain addition must be a template site. Allowed on the chain: register
    and lane moves (fp_reads(); they propagate the chain), spills and the final store, and the template sites
    themselves. Every compiler fadd off the chain is within-block expression arithmetic (e.g. completed_block =
    dot_contribution + correction): listed and counted, never ordered (its NaN payload order is the T-2
    definition's inventory item). Tile programs have no output seed. Returns {key@copy: {"sites", "offchain"}}."""
    image, bodies, sites = bound_ordered_sites(data, disassembly, assembly)
    locations: dict[str, list[str]] = {}
    assembly_functions(assembly, set(bodies), image.macho, locations)
    names = [name for name in sorted(bodies) if TILE_RUN.search(name) and tile_match(name, image.prefix)
             and int(tile_match(name, image.prefix).group(3)) == 1]

    def one(name: str) -> tuple[str, dict]:
        # One GPU-aligned program (the serial reference computation, also run per program in a worker, V2).
        key = program_key(name)
        indices = sites.get(name, [])
        if not indices:
            raise ValueError(f"{key}@{copy}: GPU-aligned tile program without an ordered add")
        context: dict = {}
        instructions, origins = fadd_origins(name, bodies[name], indices, image.prefix, locations.get(name, []),
                                             sources or {}, seeds=(), memcpy_seed=False, context=context)
        readers = {}
        for index, instruction in enumerate(instructions):
            read = None if index in context["sites"] else fp_reads(instruction)
            if read:
                readers[index] = (read[0], set().union(*(terminals(index, f"v{number}", context["states"],
                                                                     context["values"]) for number in read[1])))
        # Accumulator initialisations: the exact +0.0 values reaching a template site's earlier operand. Other zero
        # constants (integer widening, decode masks) are not on the chain.
        inits = {source for site in context["sites"]
                 for kind, source in terminals(site, "v" + ORDERED_ADD.match(
                     " ".join(instructions[site].text.split())).group(1), context["states"], context["values"])
                 if kind == "init"}

        def anchored(kind: str, source: int) -> bool:
            return kind == "site" or (kind == "init" and source in inits) or source in touching
        touching: set = set()
        changed = True
        while changed:
            changed = False
            for index, (_, found) in readers.items():
                if index not in touching and any(anchored(kind, source) for kind, source in found):
                    touching.add(index)
                    changed = True
        for index in sorted(touching):
            if readers[index][0] == "op":
                raise ValueError(f"{key}@{copy}: compiler FP operation at instruction {index} touches the accumulator "
                                 f"chain (only template sites may add to it): {instructions[index].text!r}")
        return f"{key}@{copy}", {"sites": indices, "offchain": sorted(set(origins) - touching)}
    return dict(map_programs(one, names, workers))


def require_tile_chain(evidence: dict[str, dict], expected: dict[str, dict]) -> None:
    """Exact committed per-copy counts {key@copy: {"sites": n, "offchain": n}}."""
    counts = {key: {"sites": len(value["sites"]), "offchain": len(value["offchain"])} for key, value in evidence.items()}
    if counts != expected:
        raise ValueError(f"tile chain counts {counts} differ from the committed {expected}")


def compact_ordered_add_evidence(data: bytes, disassembly: str, assembly: str, copy: str,
                                 sources: dict[str, str] | None = None) -> dict[str, dict]:
    """S2-4: the ordered additions of every compact program of one production copy (TU object), bound to the
    executed compile's assembler input (B1 replay) like the tile sites. GPU-aligned programs (ordered 1) need at
    least one inline-asm site (addOrderedNEON); every compiler-emitted scalar or vector `fadd` must be proven in
    source order by prove_order() (global reaching definitions); a commuted or unprovable one fails.
    Backend-native (ordered 0) and develop's policy-free (-) programs make no order claim and must hold no inline
    site. sources: {located file (source_name()) : text} of the closure, for the seed binding. Returns
    {key@copy: {"sites", "scalar"}} (instruction indices; "scalar" lists every proven compiler fadd) for exact
    committed counts (require_compact_ordered_adds())."""
    image, bodies, sites = bound_ordered_sites(data, disassembly, assembly)
    locations: dict[str, list[str]] = {}
    assembly_functions(assembly, set(bodies), image.macho, locations)
    result: dict[str, dict] = {}
    for key, (name, _) in sorted(compact_programs(image).items()):
        ordered = key.split(":")[5]
        indices = sites.get(name, [])
        if ordered != "1":
            if indices:
                raise ValueError(f"{key}@{copy}: a program without the ordered policy holds an ordered add")
            continue
        if not indices:
            raise ValueError(f"{key}@{copy}: GPU-aligned compact program without an ordered add")
        try:
            proven = prove_order(name, bodies[name], indices, image.prefix, locations.get(name, []), sources or {})
        except ValueError as error:
            raise ValueError(f"{key}@{copy}: {error}".replace(f"{name}: ", "")) from None
        result[f"{key}@{copy}"] = {"sites": indices, "scalar": proven}
    return result


FP_ARITHMETIC = frozenset({"fadd", "fsub", "fmul", "fdiv", "fneg", "fabs", "fmax", "fmin", "fcvt", "fcvtl", "fcvtl2",
                           "fcvtn", "scvtf", "ucvtf", "frecpe", "frsqrte", "fsqrt"}) | FUSED


def audited_closure(image: ObjectImage, bodies: dict, roots: tuple[str, ...] = (), follow_nonlocal: bool = False) \
        -> tuple[dict[str, str], dict[str, str]]:
    """(local closure {raw name: key}, non-local callees {raw name: binding}) of one object for the fused witness:
    every tile and compact program (by key), plus `roots`, and every TU-local function they reach by direct call,
    transitively. A weak or global callee executes the link-selected copy, so it is not followed here: it is
    returned for fused_evidence() to resolve (F-2) - unless follow_nonlocal (the pre-link scope): then this object's
    own copy of every defined weak/global callee is followed like a TU-local one."""
    unit = make_unit("closure", image, bodies)
    roots_ = {name: program_key(name) for name in bodies if TILE_RUN.search(name) and tile_match(name, image.prefix)}
    roots_.update({name: key for key, (name, _) in compact_programs(image).items()})
    roots_.update({name: name for name in roots})
    closure, external, work = dict(roots_), {}, list(roots_)
    while work:
        name = work.pop()
        for line in bodies[name]:
            parts = line.split()
            if not parts or parts[0] != "bl" or parts[-1] not in bodies:
                continue
            callee = parts[-1]
            if unit.functions.get(callee) != "local" and not (follow_nonlocal and
                                                                unit.functions.get(callee) in ("weak", "global")):
                external[callee] = unit.functions.get(callee, "unknown")
            elif callee not in closure:
                closure[callee] = callee
                work.append(callee)
    return closure, external


def fused_evidence(data: bytes, disassembly: str, assembly: str, copy: str, sources: dict[str, str],
                   selection: dict | None = None, units: dict | None = None, selected: dict | None = None,
                   roots: tuple[str, ...] = (), scope: str = "postlink") -> dict[str, dict]:
    """S2-4c: every fused multiply-add of the audited closure of one production copy (fmla, fmls, fmadd, fmsub,
    fnmadd, fnmsub, element forms included) must be an explicit source FMA: its located source line (the executed
    compile's own .loc, see assembly_functions(); arm_neon.h lines included) calls one (vfma*/vfms*, std::fma,
    __builtin_fma, the arm_neon.h __builtin_aarch64_fma/fnma bodies) and holds no contractible operation (vmul*,
    vadd*, vsub*, a binary `*`). Anything else is a contraction (-ffp-contract=fast fused a separate multiply and
    add) and fails; so does an unlocated one. Returns {function key@copy: {"fused": {mnemonic: count}}} for exact
    committed counts (require_fused()).

    F-2: a weak/global callee executes its link-selected copy. With a P2 link-selection record (selection, as
    CallModel.select() admits it, with the audited `units`) the selected copy's own fused evidence must be
    supplied (`selected` {"symbol@copy": {"fused": ...}}, from fused_evidence(..., roots=(symbol,)) on that copy's
    object) and is included; without a record, a callee whose copy here holds FP arithmetic leaves the proof
    incomplete and fails; one without is listed under "unselected" (never followed or assumed).

    scope (reviewer 102): "postlink" is the production P2 rule above (the selected copy). "prelink" - a probe
    compiled before any link exists, with no link authority - is selection-independent and stricter: no selection
    record is accepted, and this object's own copy of every weak/global callee is followed into the closure and
    must satisfy the witness itself, whichever copy a later link selects (no helper exemption)."""
    if scope not in ("postlink", "prelink"):
        raise ValueError(f"unknown fused proof scope {scope!r}")
    if scope == "prelink" and (selection or units or selected):
        raise ValueError("a pre-link fused proof has no link authority: selection records are not admitted")
    image = read_object(data)
    bodies = normalize_disassembly(disassembly, image)
    locations: dict[str, list[str]] = {}
    functions, _ = assembly_functions(assembly, set(bodies), image.macho, locations)
    closure, external = audited_closure(image, bodies, roots, follow_nonlocal=scope == "prelink")
    result: dict[str, dict] = {}
    unselected = []
    unit = make_unit(copy, image, bodies)
    for callee in sorted(external):
        record = (selection or {}).get(callee)
        if record is None:
            if any(line.split()[0] in FP_ARITHMETIC for line in bodies[callee][1:]
                   if line.startswith("\t") and line.split()):
                raise ValueError(f"{callee}@{copy}: a {external[callee]} callee with FP arithmetic has no "
                                 f"link-selection record (P2): the fused proof is incomplete")
            unselected.append(callee)
            continue
        copy_unit = (units or {}).get(record.get("copy"))
        if copy_unit is None or callee not in copy_unit.bodies:
            raise ValueError(f"{callee}@{copy}: the link-selected copy {record.get('copy')!r} is not audited")
        reason = CallModel.select(unit, copy_unit, callee, record)
        if reason:
            raise ValueError(f"{callee}@{copy}: {reason}")
        evidence = (selected or {}).get(f"{callee}@{record['copy']}")
        if not isinstance(evidence, dict) or set(evidence) != {"fused"}:
            raise ValueError(f"{callee}@{copy}: the selected copy's fused evidence {callee}@{record['copy']} is not "
                             f"supplied")
        result[f"{callee}@{record['copy']}"] = evidence
    for name, key in sorted(closure.items(), key=lambda item: item[1]):
        compiled = object_instructions(bodies[name])
        located = locations.get(name, [])
        if len(compiled) != len(functions.get(name, [])):
            raise ValueError(f"{key}@{copy}: compiler assembly and object instruction counts differ; fused "
                             f"instructions cannot be located")
        counts: dict[str, int] = {}
        for index, text in enumerate(compiled):
            mnemonic = text.split(" ", 1)[0]
            if mnemonic not in FUSED:
                continue
            file, _, line = located[index].rpartition(":")
            source = sources.get(file)
            if source is None or not line.isdigit():
                raise ValueError(f"{key}@{copy}: fused {text!r} at {located[index]} has no bound source line")
            statement = source.split("\n")[int(line) - 1]
            if not EXPLICIT_FMA.search(statement) or CONTRACTIBLE.search(EXPLICIT_FMA.sub("(", statement)):
                raise ValueError(f"{key}@{copy}: fused {text!r} at {located[index]} is not an explicit source FMA "
                                 f"(contraction): {statement.strip()!r}")
            counts[mnemonic] = counts.get(mnemonic, 0) + 1
        result[f"{key}@{copy}"] = {"fused": dict(sorted(counts.items()))}
    if unselected:
        result[f"unselected@{copy}"] = {"fused": {}, "callees": unselected}
    return result


def require_fused(evidence: dict[str, dict], expected: dict[str, dict]) -> None:
    """Exact committed fused-instruction counts per function key@copy: an unexpected, lost or moved one fails."""
    if evidence != expected:
        raise ValueError(f"fused instructions {evidence} differ from the committed {expected}")


def require_compact_ordered_adds(evidence: dict[str, dict], expected: dict[str, dict]) -> None:
    """Exact committed per-copy counts {key@copy: {"sites": n, "scalar": n}}: a lost, added or moved site fails."""
    counts = {key: {"sites": len(value["sites"]), "scalar": len(value["scalar"])} for key, value in evidence.items()}
    if counts != expected:
        raise ValueError(f"compact ordered-add sites {counts} differ from the committed {expected}")


def name_loops(traffic: dict, data: bytes, disassembly: str, assembly: str) -> dict:
    """Name every loop of every tile program by source location, from the executed compile's own assembler input.

    The retained assembly is the input the bound object was assembled from (see replay_plan()), so its
    instruction sequence is the object's: every tile program must have the object's instruction count (nops
    set aside), and a loop's entries and latches (instruction indices of traffic_record()) take the `.loc`
    location of the instruction at that index. Returns the traffic with "entry_sources"/"latch_sources" added.
    """
    image = read_object(data)
    bodies = normalize_disassembly(disassembly, image)
    locations: dict[str, list[str]] = {}
    functions, _ = assembly_functions(assembly, set(bodies), image.macho, locations)
    named = {}
    keys = {program_key(name): name for name in bodies if TILE_RUN.search(name) and tile_match(name, image.prefix)}
    for key, record in traffic.items():
        name = keys.get(key)
        if name is None:
            raise ValueError(f"traffic names a program the object does not hold: {key}")
        compiled = object_instructions(bodies[name])
        lines = functions.get(name, [])
        where = locations.get(name, [])
        if len(compiled) != len(lines) or len(where) != len(lines):
            raise ValueError(f"{key}: compiler assembly ({len(lines)}) and object ({len(compiled)}) instruction counts "
                             f"differ; loops cannot be named")
        loops = []
        for loop in record["loops"]:
            loop = dict(loop)
            loop["entry_sources"] = [where[index] if 0 <= index < len(where) else "?" for index in loop["entries"]]
            loop["latch_sources"] = [where[index] if 0 <= index < len(where) else "?" for index in loop["latches"]]
            loops.append(loop)
        named[key] = {"loops": loops, "counts": record["counts"],
                      **({"bounded": record["bounded"]} if record.get("bounded") else {})}
    return named


BUDGET_PATH = Path(__file__).resolve().with_name("CPUNativeTileRegisterBudget.json")
ORDERED_ADD_SITES_PATH = Path(__file__).resolve().with_name("CPUNativeTileOrderedAddSites.json")
CHAIN_COUNTS_PATH = Path(__file__).resolve().with_name("CPUNativeTileChainCounts.json")
PROBE_COPY = "probe"


BOUNDED_MEMCPY_PATH = Path(__file__).resolve().with_name("CPUNativeTileBoundedMemcpy.json")
# Per analysed function the instruction source locations ("file:line", instruction indices as traffic_record()
# numbers them) and the closure-bound source texts by located name: installed by the caller of an analysis
# (install_bounded_context()) so a variable-size memcpy can be admitted by its exact source line (M-1).
_BOUNDED: dict = {"where": {}, "sources": {}}
COLUMN_ASSIGNMENT = re.compile(r"\bcolumn\s*(?:\+=|-=|\*=|/=|=(?!=)|\+\+|--)|(?:\+\+|--)\s*column\b")


def install_bounded_context(where: dict[str, list[str]], sources: dict[str, str]) -> None:
    _BOUNDED["where"], _BOUNDED["sources"] = dict(where), dict(sources)


def bounded_memcpy_site(function: str, index: int | None, lanes: int | None) -> dict | str:
    """M-1..M-3 (reviewer 91): the admitted variable-size memcpy at one call, else the failure reason.

    M-1: the call's .loc is a line of the compact kernel source whose closure-bound text (sha256) the committed
    admission file names, at one of that file's admitted lines, and the line with whitespace removed is one of the
    admitted texts. M-2: bytes lie in [sizeof(float), sizeof(F)], sizeof(F) = 4 x Lanes of the program key.
    M-3: the site is the column loop's tail: the line before it is `else`, three lines before it the guard
    `if (column + Lanes <= columns)`, the nearest enclosing for header steps `column += Lanes` while
    `column < columns`, and nothing between that header and the site assigns column - so the tail branch is
    true on at most one column iteration (charged once per column-loop entry by the stack table)."""
    where = _BOUNDED["where"].get(function)
    if lanes is None:
        return "no program key gives sizeof(F) (variable memcpy outside a compact program)"
    if where is None or index is None or index >= len(where):
        return "no source location is bound for the call"
    path, _, line = where[index].rpartition(":")
    admission = json.loads(BOUNDED_MEMCPY_PATH.read_text())
    source = _BOUNDED["sources"].get(path)
    if path != admission["file"] or source is None or not line.isdigit():
        return f"the call's source {where[index]} is not the bound compact kernel source"
    entry = admission["trees"].get(hashlib.sha256(source.encode()).hexdigest())
    number = int(line)
    lines = source.splitlines()
    if entry is None or number not in entry["lines"]:
        return f"{where[index]} is not an admitted line of this source version"
    text = re.sub(r"\s+", "", lines[number - 1])
    if text not in admission["texts"]:
        return f"{where[index]} does not hold an admitted text"
    if re.sub(r"\s+", "", lines[number - 2]) != "else" or re.sub(r"\s+", "", lines[number - 4]) != admission["guard"]:
        return f"{where[index]} is not the else branch of the column tail guard"
    header = next((k for k in range(number - 1, 0, -1) if re.sub(r"\s+", "", lines[k - 1]).startswith("for(")
                   and "column" in lines[k - 1]), None)
    if header is None or not re.sub(r"\s+", "", lines[header - 1]).endswith(admission["column_loop"]):
        return f"{where[index]}: the enclosing column loop does not step column += Lanes while column < columns"
    if any(COLUMN_ASSIGNMENT.search(lines[k - 1]) for k in range(header + 1, number)):
        return f"{where[index]}: column is assigned inside the column loop before the tail"
    return {"line": where[index], "text": text, "lower_bytes": 4, "upper_bytes": 4 * lanes,
            "placement": "column-tail", "column_header": f"{path}:{header}"}


def closure_sources(inputs: dict) -> dict[str, str]:
    """The dependency closure's texts by located name (source_name(), as assembly_functions() names `.loc`
    files). A name two closure files share is ambiguous and left out (its lines are then unbound and fail)."""
    found: dict[str, list[str]] = {}
    for path in inputs:
        found.setdefault(source_name(path), []).append(path)
    return {name: Path(paths[0]).read_text(errors="replace") for name, paths in found.items() if len(paths) == 1}


def chain_witness(data: bytes, disassembly: str, assembly: str, inputs: dict, workers: int | None = None) -> dict:
    """The probe's T-1 tile chain witness (mandatory in build/verify): per GPU-aligned tile program {"sites",
    "offchain"} (tile_chain_evidence()). The probe is one copy (PROBE_COPY); diagnostic, not a production
    certification."""
    chain = tile_chain_evidence(data, disassembly, assembly, PROBE_COPY, closure_sources(inputs), workers)
    suffix = f"@{PROBE_COPY}"
    return {"tile_chain": {key[:-len(suffix)]: {"sites": len(value["sites"]), "offchain": len(value["offchain"])}
                           for key, value in sorted(chain.items())}}


def require_chain_counts(witness: dict, identity: str, mode: str, expected: dict | None = None) -> None:
    """The probe's chain witness equals its committed counts (CPUNativeTileChainCounts.json, per compiler identity
    and ISA envelope), like the ordered-add sites: an unexpected, lost or moved chain site, off-chain fadd or fused
    instruction fails until the committed counts change in a reviewed diff."""
    if expected is None:
        expected = json.loads(CHAIN_COUNTS_PATH.read_text())
    counts = expected.get(identity, {}).get(mode)
    if counts is None:
        raise ValueError(f"no committed chain witness counts for compiler {identity} envelope {mode}")
    for part in witness:
        if part not in counts:
            raise ValueError(f"no committed {part} counts for compiler {identity} envelope {mode}")
        # Canonical (sorted-key) values: the detail must be identical whether the witness was just computed or
        # re-read from a JSON record (the unqualified analysis record re-check compares these strings).
        failures = [f"{key}: {canonical(witness[part].get(key))} != committed {canonical(counts.get(part, {}).get(key))}"
                    for key in sorted(set(witness[part]) | set(counts.get(part, {})))
                    if witness[part].get(key) != counts.get(part, {}).get(key)]
        if failures:
            raise ValueError(f"{part} witness differs from the committed counts ({len(failures)}): "
                             + "; ".join(failures[:4]))


def require_ordered_adds(sites: dict[str, list[int]], identity: str, mode: str, expected: dict | None = None) -> None:
    """Every GPU-aligned tile program holds exactly its committed number of ordered-add sites.

    The site counts are owned like the spill budget (CPUNativeTileOrderedAddSites.json, per compiler identity
    and ISA envelope): a change that turns one ordered addition of a program back into a plain one, or adds
    one, changes the count and fails until the committed count is changed in a reviewed diff. The programs
    with sites must be exactly the committed ones (backend-native programs hold none, see
    ordered_add_evidence()).
    """
    if expected is None:
        expected = json.loads(ORDERED_ADD_SITES_PATH.read_text())
    counts = expected.get(identity, {}).get(mode)
    if counts is None:
        raise ValueError(f"no committed ordered-add site counts for compiler {identity} envelope {mode}")
    actual = {key: len(indices) for key, indices in sites.items()}
    failures = [f"{key} holds {actual.get(key, 0)} ordered adds, committed {counts.get(key, 0)}"
                for key in sorted(set(actual) | set(counts)) if actual.get(key, 0) != counts.get(key, 0)]
    if failures:
        raise ValueError(f"ordered-add site counts differ from the committed ones ({len(failures)}): "
                         + "; ".join(failures[:6]))


def require_registers(counts: dict[str, int], identity: str, mode: str, budget: dict | None = None) -> None:
    """Single-row programs must not spill; grouped programs must stay within the committed ratchet."""
    if budget is None:
        budget = json.loads(BUDGET_PATH.read_text())
    limits = budget.get(identity, {}).get(mode)
    if limits is None:
        raise ValueError(f"no qualified tile register budget for compiler {identity} envelope {mode}")
    # The coded Q4_K leaf replaces the former Q4_K identity-nibble leaf;
    # its same-width Q4_1 neighbour retains the identical qualified ceiling.
    limits = dict(limits)
    for key, ceiling in tuple(limits.items()):
        if key.endswith(":0:1:1"):
            limits[key.removesuffix(":0:1:1") + ":6:1:1"] = ceiling
    failures = [failure["detail"] for failure in register_failures(counts, limits)]
    if failures:
        raise ValueError(f"native NEON tile register policy violated ({len(failures)}): " + "; ".join(failures[:6]))


def register_failures(counts: dict[str, int], limits: dict) -> list[dict]:
    """Typed register-policy admission failures (reason "spill": a single-row program spills; "register_budget":
    a grouped program exceeds its committed budget or has none; "stale_budget": the budget names programs the
    witness did not emit), each with its program keys."""
    failures = []
    for name, count in counts.items():
        key = program_key(name)
        if key.startswith("1:"):
            if count:
                failures.append({"reason": "spill", "programs": [key],
                                 "detail": f"{key} spills {count} (single-row programs must not spill)"})
        elif key not in limits:
            failures.append({"reason": "register_budget", "programs": [key], "detail": f"{key} has no budget"})
        elif count > limits[key]:
            failures.append({"reason": "register_budget", "programs": [key],
                             "detail": f"{key} spills {count} > budget {limits[key]}"})
    stale = set(limits) - {program_key(name) for name in counts}
    if stale:
        failures.append({"reason": "stale_budget", "programs": sorted(stale),
                         "detail": f"budget names programs the witness did not emit: {sorted(stale)[:3]}"})
    return failures


# Proof documents. The seal (proof.json) is written only when every admission passes; an analysis that completed
# structurally but failed admission is written as an unqualified analysis record (proof.analysis.json): a distinct
# schema and kind, never a seal (verify() rejects it under any name), diagnostic only (UA-1..UA-4).
SEALED_KIND, ANALYSIS_KIND = "sealed-proof", "unqualified-analysis"
ANALYSIS_SCHEMA = "12-unqualified-analysis"
ANALYSIS_FIELDS = ("program_violations", "ordered_adds", "stack_traffic", "chain_witness")
ADMISSION_REASONS = frozenset({"spill", "register_budget", "stale_budget", "budget_missing", "ordered_add_count",
                               "chain_count"})
BOUND_FIELDS = ("signature", "identity", "mode", "checker_sha256", "envelope_sha256", "budget_sha256",
                "ordered_add_sites_sha256", "chain_counts_sha256", "compiler_version", "executed_command",
                "executed_directory", "target_command_sha256", "driver_plan", "programs", "inputs", "object_file",
                "object_sha256", "disassembler", "disassembler_version", "disassembler_sha256",
                "disassembly_command", "disassembly_sha256", "translation_unit_sha256", "assembly_sha256",
                "line_witness")


def analysis_path(proof: Path) -> Path:
    """The unqualified analysis record next to a proof path (proof.json -> proof.analysis.json)."""
    return Path(proof).with_suffix(".analysis.json")


def analysis_fields(data: bytes, disassembly: str, assembly: str, inputs: dict, mode: str,
                    analyses: dict | None = None, workers: int | None = None) -> dict:
    """The structurally complete analysis of one bound object: spills, ordered-add sites, named stack traffic and
    the T-1 chain witness. Any structural failure raises (then no record of any kind is written). `analyses`
    receives the per-program Analysis objects of this one analysis."""
    traffic: dict = {}
    image = read_object(data)
    locations: dict[str, list[str]] = {}
    assembly_functions(assembly, set(normalize_disassembly(disassembly, image)), image.macho, locations)
    install_bounded_context(locations, closure_sources(inputs))
    counts = audit(data, disassembly, mode, traffic, analyses=analyses, workers=workers)
    return {"program_violations": counts, "ordered_adds": ordered_add_evidence(data, disassembly, assembly),
            "stack_traffic": name_loops(traffic, data, disassembly, assembly),
            "chain_witness": chain_witness(data, disassembly, assembly, inputs, workers)}


def admission_failures(fields: dict, identity: str, mode: str) -> list[dict]:
    """Every admission failure of a structurally complete analysis, typed (ADMISSION_REASONS) with program keys:
    the register policy, the committed ordered-add site counts and the committed chain counts."""
    failures = []
    budget = json.loads(BUDGET_PATH.read_text()).get(identity, {}).get(mode)
    if budget is None:
        failures.append({"reason": "budget_missing", "programs": sorted(map(program_key, fields["program_violations"])),
                         "detail": f"no qualified tile register budget for compiler {identity} envelope {mode}"})
    else:
        limits = dict(budget)
        for key, ceiling in tuple(limits.items()):
            if key.endswith(":0:1:1"):
                limits[key.removesuffix(":0:1:1") + ":6:1:1"] = ceiling
        failures += register_failures(fields["program_violations"], limits)
    try:
        require_ordered_adds(fields["ordered_adds"], identity, mode)
    except ValueError as error:
        committed = json.loads(ORDERED_ADD_SITES_PATH.read_text()).get(identity, {}).get(mode) or {}
        actual = {key: len(indices) for key, indices in fields["ordered_adds"].items()}
        failures.append({"reason": "ordered_add_count", "detail": str(error),
                         "programs": sorted(key for key in set(actual) | set(committed)
                                            if actual.get(key, 0) != committed.get(key, 0))})
    try:
        require_chain_counts(fields["chain_witness"], identity, mode)
    except ValueError as error:
        committed = json.loads(CHAIN_COUNTS_PATH.read_text()).get(identity, {}).get(mode) or {}
        witness = fields["chain_witness"]["tile_chain"]
        failures.append({"reason": "chain_count", "detail": str(error),
                         "programs": sorted(key for key in set(witness) | set(committed.get("tile_chain", {}))
                                            if witness.get(key) != committed.get("tile_chain", {}).get(key))})
    # Canonical order (independent of the analysis fields' key order after a JSON round trip).
    return sorted(failures, key=lambda failure: (failure["reason"], failure["programs"], failure["detail"]))


def check_admission(admission) -> list[dict]:
    """The admission of an unqualified analysis record: status "failed" with a non-empty list of typed failures,
    each {"reason" in ADMISSION_REASONS, "programs": [str], "detail": str}; anything else fails."""
    if not isinstance(admission, dict) or admission.get("status") != "failed":
        raise ValueError("the unqualified analysis record has no failed admission status")
    failures = admission.get("failures")
    if not isinstance(failures, list) or not failures or any(
            not isinstance(failure, dict) or set(failure) != {"reason", "programs", "detail"} or
            failure["reason"] not in ADMISSION_REASONS or not isinstance(failure["programs"], list) or
            not all(isinstance(key, str) for key in failure["programs"]) or not isinstance(failure["detail"], str)
            for failure in failures):
        raise ValueError("the unqualified analysis record carries no typed admission failures")
    return failures


WITNESS_FLAGS = ["-O3", "-fno-sanitize=all", "-fno-lto", "-std=c++20"]


def compiler_identity(version: str) -> str:
    """CMake-style compiler identity (id-major) derived from the compiler's own --version output."""
    apple = re.search(r"^Apple clang version (\d+)\.", version, re.MULTILINE)
    if apple:
        return f"AppleClang-{apple.group(1)}"
    clang = re.search(r"^(?:\S+ )?clang version (\d+)\.", version, re.MULTILINE)
    if clang:
        return f"Clang-{clang.group(1)}"
    gnu = re.search(r"^\S*(?:g\+\+|c\+\+|gcc)\S* \(.*\) (\d+)\.\d+(?:\.\d+)?\s*$", version, re.MULTILINE)
    if gnu and "Free Software Foundation" in version:
        return f"GNU-{gnu.group(1)}"
    raise ValueError(f"unrecognized compiler identity: {version.splitlines()[0] if version else version!r}")


DEPFILE_UNSUPPORTED = re.compile(r"[$#\\'\"%*?\[\]]")
# Every character a plain-subset name may contain: printable ASCII except the space separator.
DEPFILE_NAME = re.compile(r"[!-~]+")


def read_depfile(path: Path, directory: str) -> list[str]:
    """Logical (unresolved, absolute) paths named by a compiler (Make-syntax) dependency file.

    Only the plain subset of the native depfile grammar is admitted: one rule, prerequisites separated by
    whitespace, backslash-newline continuations. Any name that Make escaping would alter or that is not plain
    ('$$' for '$', escaped spaces or '#', backslashes, quotes, wildcards) fails before anything is bound,
    instead of being decoded with another grammar.
    """
    text = path.read_bytes().decode("utf-8", "replace")
    # Separators are exactly the native ones (ASCII space, tab, newline). Any other character outside printable
    # ASCII - e.g. a no-break space, which Make keeps inside a name but a Unicode split would treat as a
    # separator, or a vertical tab / carriage return a line split would treat as a line end - fails before
    # anything is split.
    if re.search(r"[^\t\n !-~]", text):
        raise ValueError(f"dependency file {path} contains characters outside the plain depfile subset")
    text = text.replace("\\\n", " ")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"dependency file {path} is not exactly one rule")
    target, separator, prerequisites = lines[0].partition(": ")
    if not separator:
        raise ValueError(f"malformed compiler dependency file {path}")
    items = [item for item in re.split(r"[ \t]+", prerequisites) if item]
    unsupported = [item for item in items if DEPFILE_UNSUPPORTED.search(item) or not DEPFILE_NAME.fullmatch(item)]
    if not items or unsupported or DEPFILE_UNSUPPORTED.search(target):
        raise ValueError(f"dependency file {path} uses unsupported escaped names: {unsupported[:3]}")
    # No lexical normalisation: "link/../h.h" must keep the filesystem's meaning (".." of the link target).
    return sorted({os.path.join(directory, item) for item in items})


def resolve(path: str) -> tuple[str, list[list[str]]]:
    """Resolve a logical path component by component as the kernel does, recording every symlink followed."""
    if not os.path.isabs(path):
        raise ValueError(f"dependency path is not absolute: {path}")
    links, current, pending = [], os.sep, list(Path(path).parts[1:])
    while pending:
        part = pending.pop(0)
        if part in ("", "."):
            continue
        if part == "..":
            current = os.path.dirname(current) or os.sep
            continue
        candidate = os.path.join(current, part)
        if os.path.islink(candidate):
            if len(links) >= 64:
                raise ValueError(f"too many symbolic links resolving {path}")
            target = os.readlink(candidate)
            links.append([candidate, target])
            pending = list(Path(target).parts[1:] if os.path.isabs(target) else Path(target).parts) + pending
            current = os.sep if os.path.isabs(target) else current
        else:
            current = candidate
    if current != os.path.realpath(path):
        raise ValueError(f"dependency path resolution is ambiguous: {path}")
    return current, links


def input_identity(path: str) -> dict:
    """Content hash of the file a logical input path names, its resolved path and the symlinks followed."""
    resolved, links = resolve(path)
    return {"sha256": envelope.digest(Path(resolved)), "resolved": resolved, "links": links}


def bind_inputs(paths: list[str]) -> dict[str, dict]:
    """Identity of every dependency; a missing input fails closed."""
    return {path: input_identity(path) for path in paths}


def executed_command(command: list[str], source: str, obj: Path, dependencies: Path) -> list[str]:
    """The witness compile; -H reports every header and precompiled header the compiler actually read."""
    return command + WITNESS_FLAGS + ["-c", source, "-o", str(obj), "-MD", "-MF", str(dependencies), "-MT", "probe",
                                      "-H"]


# Release line witness (L-1..L-4). Loop roles and fused-instruction sites are named by the `.loc` locations of
# the retained assembly. A compile without debug info (a Release argv: `-O3 -DNDEBUG`) emits none, so every
# location is "?". Such a compile gets a second, line-table witness compile:
#   L-1  its argv is the executed Release argv with only the output operands renamed and exactly one appended
#        LINE_WITNESS_FLAG (`-g`, the debug level of the Integration envelope, whose `.loc` semantics the role
#        map was qualified on); any other delta fails;
#   L-2  every SHF_ALLOC section (name, type, flags, alignment, entry size and bytes) and every relocation of an
#        SHF_ALLOC section (offset, type, addend and the target symbol by name, kind, binding, section and value)
#        of the witness object must equal the exact Release object's; any difference fails with both digests;
#   L-3  admission and analysis still read the exact Release object and its disassembly; only the `.loc`
#        locations come from the witness compile's retained assembly, whose instructions L-2 proves identical;
#   L-4  an argv that already selects debug info is analysed directly (no witness, flag never doubled); `-g0`
#        (debug info disabled explicitly) fails closed.
LINE_WITNESS_FLAG = "-g"
DEBUG_FLAG = re.compile(r"^-g(?:[0-3]|gdb[0-3]?|dwarf(?:-[2-5])?|line-tables-only|line-directives-only)?$")
SHF_ALLOC = 0x2
SHT_RELA, SHT_REL, SHT_NOBITS, SHT_SYMTAB = 4, 9, 8, 2


def line_witness_needed(executed: list[str]) -> bool:
    """Whether the executed argv lacks line tables and needs the Release line witness (L-4)."""
    flags = [argument for argument in executed if DEBUG_FLAG.match(argument)]
    if "-g0" in flags:
        raise ValueError("the witness argv disables debug information (-g0): no line witness is admitted for it")
    return not flags


def line_witness_command(executed: list[str], obj: Path, dependencies: Path, lines_obj: Path,
                         lines_dependencies: Path) -> list[str]:
    """The line witness argv: the executed argv, its object and dependency operands renamed, plus `-g` (L-1)."""
    renamed = {str(obj): str(lines_obj), str(dependencies): str(lines_dependencies)}
    witness = [renamed.get(argument, argument) for argument in executed] + [LINE_WITNESS_FLAG]
    require_line_witness_delta(executed, witness, renamed)
    return witness


def require_line_witness_delta(executed: list[str], witness: list[str], renamed: dict[str, str]) -> None:
    """L-1: the witness argv differs from the executed one only by the renamed outputs and one trailing `-g`."""
    if len(witness) != len(executed) + 1 or witness[-1] != LINE_WITNESS_FLAG or \
            any(after != renamed.get(before, before) for before, after in zip(executed, witness)) or \
            sum(argument in renamed for argument in executed) != len(renamed):
        raise ValueError(f"the line witness argv is not the executed argv plus exactly {LINE_WITNESS_FLAG!r}: "
                         f"{sorted(set(witness) ^ set(executed))[:6]}")


def allocated_image(data: bytes) -> dict:
    """L-2: every SHF_ALLOC section of an ELF relocatable object and the relocations applied to it, by name."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("the line witness needs ELF64 little-endian relocatable objects")
    shoff, shnum, shstrndx = _field(data, 0x28, 8), _field(data, 0x3C, 2), _field(data, 0x3E, 2)
    headers = [{"name": _field(data, shoff + 64 * i, 4), "type": _field(data, shoff + 64 * i + 4, 4),
                "flags": _field(data, shoff + 64 * i + 8, 8), "offset": _field(data, shoff + 64 * i + 24, 8),
                "size": _field(data, shoff + 64 * i + 32, 8), "link": _field(data, shoff + 64 * i + 40, 4),
                "info": _field(data, shoff + 64 * i + 44, 4), "align": _field(data, shoff + 64 * i + 48, 8),
                "entsize": _field(data, shoff + 64 * i + 56, 8)} for i in range(shnum)]
    if not 0 < shstrndx < shnum:
        raise ValueError("malformed ELF section header table")
    names = headers[shstrndx]

    def name(offset: int, table: dict) -> str:
        return _string(data, table["offset"] + offset, table["offset"] + table["size"])

    for header in headers:
        header["text"] = name(header["name"], names)

    def content(header: dict) -> bytes:
        if header["type"] == SHT_NOBITS:
            return b""
        if header["offset"] + header["size"] > len(data):
            raise ValueError("truncated witness object")
        return data[header["offset"]:header["offset"] + header["size"]]

    sections = {}
    for header in headers:
        if header["flags"] & SHF_ALLOC:
            if header["text"] in sections:
                raise ValueError(f"duplicate allocated section name {header['text']}")
            sections[header["text"]] = {"type": header["type"], "flags": header["flags"], "size": header["size"],
                                        "align": header["align"], "entsize": header["entsize"],
                                        "sha256": hashlib.sha256(content(header)).hexdigest()}
    relocations: dict[str, list] = {}
    for header in headers:
        if header["type"] not in (SHT_RELA, SHT_REL) or not 0 < header["info"] < shnum or \
                not headers[header["info"]]["flags"] & SHF_ALLOC:
            continue
        symtab = headers[header["link"]]
        if symtab["type"] != SHT_SYMTAB:
            raise ValueError("a relocation section does not link a symbol table")
        strings = headers[symtab["link"]]
        width = 24 if header["type"] == SHT_RELA else 16
        entries = []
        for base in range(header["offset"], header["offset"] + header["size"], width):
            info = _field(data, base + 8, 8)
            symbol = symtab["offset"] + (info >> 32) * 24
            sym_info, shndx = data[symbol + 4], _field(data, symbol + 6, 2)
            target = headers[shndx]["text"] if 0 < shndx < shnum else str(shndx)
            label = target if sym_info & 0xF == 3 else name(_field(data, symbol, 4), strings)
            entries.append((_field(data, base, 8), info & 0xFFFFFFFF,
                            _field(data, base + 16, 8) if width == 24 else None,
                            label, sym_info, target, _field(data, symbol + 8, 8)))
        relocations.setdefault(headers[header["info"]]["text"], []).extend(entries)
    return {"sections": sections, "relocations": {key: sorted(value) for key, value in sorted(relocations.items())}}


def require_code_identity(release: bytes, witness: bytes) -> str:
    """L-2: fail unless the witness object's allocated image equals the Release object's; returns its digest."""
    image, other = allocated_image(release), allocated_image(witness)
    if image != other:
        moved = sorted(name for name in set(image["sections"]) | set(other["sections"])
                       if image["sections"].get(name) != other["sections"].get(name))
        moved += sorted(f"relocations of {name}" for name in set(image["relocations"]) | set(other["relocations"])
                        if image["relocations"].get(name) != other["relocations"].get(name))
        raise ValueError(f"the line witness changed code generation: allocated image differs in {moved[:6]} "
                         f"(Release object {hashlib.sha256(release).hexdigest()}, "
                         f"witness object {hashlib.sha256(witness).hexdigest()})")
    return hashlib.sha256(canonical_image(image).encode()).hexdigest()


def canonical_image(image: dict) -> str:
    return json.dumps(image, sort_keys=True, separators=(",", ":"))


def require_absolute_source(source: str) -> None:
    """The probe path is passed to the compiler as given; relative spellings depend on the cwd and are rejected."""
    if not os.path.isabs(source):
        raise ValueError(f"the witness probe path must be absolute: {source}")


def disassembler_command(tool: str, obj: Path) -> tuple[list[str], str]:
    """Disassembly command for the configured objdump, and its version text."""
    version = subprocess.check_output([tool, "--version"], text=True)
    if "GNU objdump" in version:
        options = ["-d", "-r", "-z", "--no-show-raw-insn"]
    elif "LLVM" in version:
        options = ["-d", "-r", "-z", "--no-show-raw-insn", "--triple=aarch64"]  # Generic, not Apple, printer.
    else:
        raise ValueError(f"unsupported disassembler: {version.splitlines()[0] if version else tool}")
    return [tool] + options + [str(obj)], version


def disassemble(tool: str, obj: Path) -> tuple[str, list[str], str]:
    command, version = disassembler_command(tool, obj)
    return subprocess.check_output(command, text=True), command, version


ASM_KEYWORDS = frozenset({"asm", "__asm", "__asm__"})
ASM_QUALIFIERS = frozenset({"volatile", "__volatile__", "__volatile", "inline", "__inline__", "__inline", "goto"})
# One C++ token of a preprocessed translation unit (comments are already removed by -E). Linemarkers and
# pragmas are skipped; pp-numbers include digit separators; string and character literals carry their
# encoding prefix; raw strings use their delimiter.
CXX_TOKEN = re.compile(r"""
    (?P<directive>^[ \t]*\#[^\n]*)
  | (?P<space>[ \t\r\f\v]*\n|[ \t\r\f\v]+)
  | (?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^()\\\s"]{0,16})\((?P<body>.*?)\)(?P=delimiter)")
  | (?P<string>(?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*")
  | (?P<character>(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)+')
  | (?P<number>\.?[0-9](?:[0-9A-Za-z_.]|'[0-9A-Za-z_]|[eEpP][+-])*)
  | (?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*)
  | (?P<punctuator>.)
""", re.MULTILINE | re.DOTALL | re.VERBOSE)
ASM_LABEL = re.compile(r"[A-Za-z_$][\w$]*")
ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def decode_literal(token: re.Match) -> str:
    """Contents of a string literal token, with escapes decoded."""
    if token.lastgroup == "raw":
        return token.group("body")
    text = token.group()
    body, decoded, position = text[text.index('"') + 1:-1], [], 0
    while position < len(body):
        character = body[position]
        if character != "\\":
            decoded.append(character)
            position += 1
            continue
        escape = body[position + 1]
        if escape == "x":
            digits = re.match(r"[0-9a-fA-F]+", body[position + 2:]).group()
            decoded.append(chr(int(digits, 16) % 0x110000))
            position += 2 + len(digits)
        elif escape in "01234567":
            digits = re.match(r"[0-7]{1,3}", body[position + 1:]).group()
            decoded.append(chr(int(digits, 8)))
            position += 1 + len(digits)
        elif escape in "uU":
            width = 4 if escape == "u" else 8
            decoded.append(chr(int(body[position + 2:position + 2 + width], 16) % 0x110000))
            position += 2 + width
        else:
            decoded.append(ESCAPES.get(escape, escape))
            position += 2
    return "".join(decoded)


OPAQUE_DIRECTIVES = re.compile(r"^[ \t]*#[ \t]*(?:pragma[ \t]+(?:GCC[ \t]+pch_preprocess|clang[ \t]+module)\b|import\b"
                               r"|include\b|include_next\b|embed\b)")
NAMESPACE_INTRODUCERS = frozenset({"namespace"})
LABEL_PRECEDERS = frozenset({")", "]"})
LABEL_FOLLOWERS = frozenset({";", ",", "=", "__attribute__"})
# Identifiers that can never be the end of a declarator.
KEYWORDS = frozenset("""__extension__ alignas alignof asm auto break case catch char class co_await co_return
    co_yield concept const consteval constexpr constinit continue decltype default delete do double else enum
    explicit export extern false float for friend goto if inline int long mutable namespace new noexcept nullptr
    operator private protected public register requires return short signed sizeof static static_assert
    struct switch template this throw true try typedef typename union unsigned using virtual void volatile
    while __asm __asm__ __volatile__ __inline __inline__ __restrict __restrict__ _Pragma""".split())


def cxx_tokens(preprocessed: str) -> list[re.Match]:
    """Significant C++ tokens of a preprocessed translation unit.

    Linemarkers and ordinary pragmas are skipped. Directives that hand the compiler content -E does not show
    (a PCH marker, a clang module pragma, a remaining include/import/embed) fail; so does an unterminated
    literal.
    """
    tokens = []
    for token in CXX_TOKEN.finditer(preprocessed):
        kind = token.lastgroup
        if kind == "directive":
            if OPAQUE_DIRECTIVES.match(token.group()):
                raise ValueError(f"translation unit uses opaque compiler input: {token.group().strip()[:80]}")
            continue
        if kind == "space":
            continue
        if kind == "punctuator" and token.group() in "\"'":
            raise ValueError(f"unterminated literal at offset {token.start()}")
        tokens.append(token)
    return tokens


def scope_kinds(tokens: list[re.Match]) -> list[bool]:
    """For every token, whether it lies outside namespace scope (code, class bodies, lambdas, initializers).

    A brace opens a namespace-level scope only when the declaration preceding it (since the last ';', '{' or
    '}') is exactly 'namespace' [name] / 'inline namespace' [name] / 'extern "..."'; every other brace -
    function bodies, lambdas, class bodies, initializers - counts as non-declaration scope, which only makes the
    label rule stricter.
    """
    stack, kinds, start = [], [], 0
    for index, token in enumerate(tokens):
        text = token.group()
        kinds.append(bool(stack) and stack[-1])
        if text == "{":
            head = [other.group() for other in tokens[start:index]]
            words = [word for word in head if word not in ("inline", "export")]
            declaration = (bool(words) and words[0] in NAMESPACE_INTRODUCERS and len(words) <= 2 and
                           all(re.fullmatch(r"[A-Za-z_]\w*", word) for word in words)) or \
                (len(words) == 2 and words[0] == "extern" and words[1].startswith('"'))
            stack.append(not declaration or bool(stack and stack[-1]))
            start = index + 1
        elif text == "}":
            if not stack:
                raise ValueError(f"unbalanced braces at offset {token.start()}")
            stack.pop()
            start = index + 1
        elif text == ";":
            start = index + 1
    return kinds


ORDERED_ADD_TEMPLATE = "fadd %0.4s, %0.4s, %1.4s"
ORDERED_ADD_PREFIX = ("addOrderedNEON", "(", "float32x4_t", "earlier", ",", "float32x4_t", "later", ")", "noexcept",
                      "{")
ORDERED_ADD_SUFFIX = (":", '"+w"', "(", "earlier", ")", ":", '"w"', "(", "later", ")", ")", ";", "return", "earlier",
                      ";", "}")


def ordered_add_definition(tokens: list[re.Match], index: int, position: int, text: str) -> bool:
    """Whether the asm at tokens[index] is the complete body of the one admitted ordered-add helper.

    tokens[position] follows the (single) template literal. Everything is matched token by token: the
    helper's exact signature immediately before `asm`, the exact template text, then exactly one "+w"
    output tied to `earlier`, one "w" input `later`, no clobber list, and `return earlier; }`.
    """
    prefix, suffix = len(ORDERED_ADD_PREFIX), len(ORDERED_ADD_SUFFIX)
    if text != ORDERED_ADD_TEMPLATE or index < prefix or position + suffix > len(tokens):
        return False
    if tokens[index + 1].group() != "(" or tokens[index + 2].lastgroup != "string" or index + 3 != position:
        return False
    return (tuple(token.group() for token in tokens[index - prefix:index]) == ORDERED_ADD_PREFIX and
            tuple(token.group() for token in tokens[position:position + suffix]) == ORDERED_ADD_SUFFIX)


def inline_assembly_violations(preprocessed: str) -> list[str]:
    """Inline-assembly constructs the witness cannot prove free of assembler-side inputs.

    The compiler's dependency lists name preprocessing inputs only; the assembler can read further files and
    construct directives through macros or substitution. Rather than model the assembler, every asm keyword of
    the tokenized translation unit (including one nested in another asm's operands or in a lambda) must be
    either an empty-template statement (a compiler barrier) or a declaration asm label. The label rule is a
    conservative syntactic approximation, not a C++ parse: a single plain symbol, without qualifiers or
    operands, at namespace or extern "..." scope, after ')' or ']' or a non-keyword identifier, and followed by
    ';', ',', '=' or __attribute__. Anything else fails.
    """
    tokens = cxx_tokens(preprocessed)
    code = scope_kinds(tokens)
    findings = []
    ordered_definitions = 0
    for index, token in enumerate(tokens):
        if token.lastgroup != "identifier" or token.group() not in ASM_KEYWORDS:
            continue
        position, qualified = index + 1, False
        while position < len(tokens) and tokens[position].group() in ASM_QUALIFIERS:
            position, qualified = position + 1, True
        if position >= len(tokens) or tokens[position].group() != "(":
            raise ValueError(f"unclassifiable inline assembly at offset {token.start()}")
        position += 1
        template = []
        while position < len(tokens) and tokens[position].lastgroup in ("string", "raw"):
            template.append(decode_literal(tokens[position]))
            position += 1
        if not template or position >= len(tokens) or tokens[position].group() not in (")", ":"):
            raise ValueError(f"inline assembly without a literal template at offset {token.start()}")
        text = "".join(template)
        if not text.strip():
            continue  # Empty template: emits nothing; operands are visited as ordinary tokens.
        if not qualified and ordered_add_definition(tokens, index, position, text):
            ordered_definitions += 1
            if ordered_definitions > 1:
                findings.append(f"offset {token.start()}: second ordered-add definition")
            continue
        previous = tokens[index - 1] if index else None
        following = tokens[position + 1] if position + 1 < len(tokens) else None
        label = (not qualified and tokens[position].group() == ")" and not code[index] and previous is not None
                 and (previous.group() in LABEL_PRECEDERS or
                      (previous.lastgroup == "identifier" and previous.group() not in KEYWORDS))
                 and following is not None and following.group() in LABEL_FOLLOWERS
                 and ASM_LABEL.fullmatch(text))
        if not label:
            findings.append(f"offset {token.start()}: {text[:80]!r}")
    return findings


# Driver plan. The expanded jobs of the exact compile (-###) are the authority for what the compiler and
# assembler read, and every argument of every job must be admitted by its type. Kinds: flag (no value);
# value (a regex-checked opaque value); search (an include search root - files found through it are reported
# by -MD and bound); closure (a file that must be in the bound dependency closure); source, object, deps,
# assembly, dumpdir (the designated witness paths); directory (the executed directory, debug metadata only).
# Unknown options, unknown positionals and any other value shape fail.
WARNING = re.compile(r"^-W(?:no-)?(?:error=)?[a-z][a-z0-9-]*$")
CLANG_CC1 = {
    "-cc1": "flag", "-emit-obj": "flag", "-disable-free": "flag", "-clear-ast-before-backend": "flag",
    "-disable-llvm-verifier": "flag", "-discard-value-names": "flag", "-fdefine-target-os-macros": "flag",
    "-fdeprecated-macro": "flag", "-fdwarf2-cfi-asm": "flag", "-fencode-extended-block-signature": "flag",
    "-fexceptions": "flag", "-fcxx-exceptions": "flag", "-fblocks": "flag", "-fcommon": "flag",
    "-fno-assume-unique-vtables": "flag", "-fno-cxx-modules": "flag", "-fno-implicit-modules": "flag",
    "-fno-modulemap-allow-subdirectory-search": "flag", "-fno-odr-hash-protocols": "flag",
    "-fno-rounding-math": "flag", "-fno-sized-deallocation": "flag", "-fno-strict-return": "flag",
    "-fobjc-msgsend-selector-stubs": "flag", "-fregister-global-dtors-with-atexit": "flag",
    "-fskip-odr-check-in-gmf": "flag", "-fstack-check": "flag", "-fvisibility-inlines-hidden-static-local-var": "flag",
    "-mdarwin-stkchk-strong-link": "flag", "-skip-unused-modulemap-deps": "flag", "-sys-header-deps": "flag",
    "-vectorize-loops": "flag", "-vectorize-slp": "flag", "-enable-tlsdesc": "flag", "-O3": "flag", "-H": "flag",
    "-triple": r"arm64-apple-macosx[\d.]+", "-main-file-name": "main-file-name", "-mrelocation-model": "pic",
    "-pic-level": "2", "-target-sdk-version=": r"[\d.]+", "-fmax-type-align=": "16", "-mframe-pointer=": "non-leaf",
    "-ffp-contract=": "on", "-funwind-tables=": "1", "-target-cpu": "generic", "-target-feature": r"[+-][\w.-]+",
    "-target-abi": "darwinpcs", "-target-linker-version": r"[\d.]+", "-debug-info-kind=": "standalone",
    "-dwarf-version=": "5", "-debugger-tuning=": "lldb", "-std=": r"c\+\+20", "-fgnuc-version=": r"[\d.]+",
    "-ferror-limit": r"\d+", "-stack-protector": "1", "-mllvm": "-extra-vectorizer-passes",
    "-clang-vendor-feature=": r"\+\w+", "-Werror=": r"[a-z][a-z0-9-]*", "-Wno-error=": r"[a-z][a-z0-9-]*",
    "-Wundef-prefix=": r"[A-Z_]+", "-D": r"[A-Za-z_]\w*(?:=[\w.+-]*)?", "-x": r"c\+\+", "-MT": "probe",
    "-I": "search", "-isystem": "search", "-internal-isystem": "search", "-internal-externc-isystem": "search",
    "-internal-iframework": "search", "-isysroot": "search", "-resource-dir": "search",
    "-fdepfile-entry=": "closure", "-dependency-file": "deps", "-o": "object",
    "-fdebug-compilation-dir=": "directory", "-fcoverage-compilation-dir=": "directory",
}
GNU_CC1PLUS = {
    "-quiet": "flag", "-mlittle-endian": "flag", "-g": "flag", "-O3": "flag", "-fPIE": "flag", "-fPIC": "flag", "-fopenmp": "flag", "-fno-lto": "flag",
    "-fasynchronous-unwind-tables": "flag", "-fstack-protector-strong": "flag", "-fstack-clash-protection": "flag",
    "-H": "flag", "-imultiarch": r"aarch64-linux-gnu", "-mcpu=": "generic", "-march=": r"armv8-a(?:\+\w+)*",
    "-mabi=": "lp64", "-std=": r"(?:gnu|c)\+\+(?:20|23)", "-fno-sanitize=": "all", "-D": r"[A-Za-z_]\w*(?:=[\w.+-]*)?",
    "-MT": "probe", "-MD": "md", "-MF": "deps", "-I": "search", "-dumpdir": "dumpdir",
    "-dumpbase": r"[\w.+-]+", "-dumpbase-ext": r"\.cpp", "-o": "assembly",
}
GNU_AS = {"-EL": "flag", "--gdwarf-5": "flag", "-march=": r"armv8-a(?:\+\w+)*", "-mabi=": "lp64", "-I": "search",
          "-o": "object"}
CONFIGURATION_LINES = re.compile(r"^(?:Reading specs from|Configuration file:|Using specs)", re.MULTILINE)


class JobPaths(NamedTuple):
    source: str
    obj: str
    deps: str
    directory: str
    closure: frozenset | None  # None while the closure is not yet observed; validated again once it is.


def admit_job(job: list[str], table: dict[str, str], paths: JobPaths, positional: str) -> dict[str, list[str]]:
    """Admit every argument of one expanded job by its type; return the designated paths it names."""
    named: dict[str, list[str]] = {}
    position = 1
    while position < len(job):
        argument = job[position]
        if not argument.startswith("-") or argument == "-":
            named.setdefault("positional", []).append(argument)
            position += 1
            continue
        if argument in table:
            option, kind = argument, table[argument]
            if kind == "flag":
                position += 1
                continue
            if position + 1 >= len(job):
                raise ValueError(f"job option {argument} lacks its value")
            value, position = job[position + 1], position + 2
        else:
            joined = next((name for name in sorted(table, key=len, reverse=True)
                           if name.endswith("=") and argument.startswith(name)), None)
            if joined is None and argument[:2] in ("-D", "-I") and argument[:2] in table:
                joined = argument[:2]
            if joined is None:
                if WARNING.match(argument) and table is not GNU_AS:
                    position += 1
                    continue
                raise ValueError(f"job argument not admitted: {argument}")
            option, kind, value, position = joined, table[joined], argument[len(joined):], position + 1
        if kind in ("search", "closure", "deps", "md", "object", "assembly", "dumpdir", "directory", "main-file-name"):
            named.setdefault(kind, []).append(value)
            if kind == "search" and not os.path.isabs(value):
                raise ValueError(f"relative include search root {value}")
            if kind == "closure" and paths.closure is not None and value not in paths.closure:
                raise ValueError(f"job input {value} is not in the bound dependency closure")
            if kind == "directory" and value != paths.directory:
                raise ValueError(f"job directory {value} is not the executed directory")
            if kind == "main-file-name" and value != os.path.basename(paths.source):
                raise ValueError(f"job main file name {value} is not the probe")
            continue
        if not re.fullmatch(kind, value):
            raise ValueError(f"job value {option} {value} not admitted")
    if named.get("positional", []) != [positional]:
        raise ValueError(f"job positional inputs {named.get('positional')} are not exactly {positional}")
    return named


def validate_plan(jobs: list[list[str]], identity: str, paths: JobPaths) -> None:
    """Pure validation of a recorded driver plan (used at build and in compiler-free verification)."""
    if not isinstance(jobs, list) or not all(isinstance(job, list) and job and all(isinstance(a, str) for a in job)
                                             for job in jobs):
        raise ValueError("malformed driver plan")
    if identity.startswith(("AppleClang-", "Clang-")):
        if len(jobs) != 1 or Path(jobs[0][0]).name != "clang" or not os.path.isabs(jobs[0][0]):
            raise ValueError(f"the clang driver must run exactly one absolute integrated clang job: {jobs}")
        named = admit_job(jobs[0], CLANG_CC1, paths, paths.source)
        if named.get("object") != [paths.obj] or named.get("deps") != [paths.deps] or \
                "-emit-obj" not in jobs[0] or jobs[0][1] != "-cc1":
            raise ValueError("the clang job does not compile the probe into the witness object and deps")
        return
    if len(jobs) != 2 or Path(jobs[0][0]).name != "cc1plus" or not os.path.isabs(jobs[0][0]) or jobs[1][0] != "as":
        raise ValueError(f"the GNU driver must run exactly one cc1plus job and one as job: {[job[:1] for job in jobs]}")
    compiler = admit_job(jobs[0], GNU_CC1PLUS, paths, paths.source)
    # GCC retains earlier dialect switches in -### even when the witness's
    # final -std=c++20 overrides the production target's C++23 dialect.
    standards = [arg for arg in jobs[0] if arg.startswith("-std=")]
    if not standards or standards[-1] != "-std=c++20":
        raise ValueError("GNU witness requires final effective -std=c++20")
    # GNU passes -MD the object-derived name; -MF selects the file actually written.
    if compiler.get("deps") != [paths.deps] or len(compiler.get("md", [])) != 1 or \
            compiler["md"][0] not in (paths.deps, os.path.splitext(paths.obj)[0] + ".d") or \
            len(compiler.get("assembly", [])) != 1 or \
            len(compiler.get("dumpdir", [])) > 1:
        raise ValueError("the GNU compiler job does not write the witness dependencies and one assembly")
    if gnu_temporary(jobs) != compiler["assembly"][0]:
        raise ValueError("the GNU compiler output is not its temporary assembly file")
    assembler = admit_job(jobs[1], GNU_AS, paths, compiler["assembly"][0])
    if assembler.get("object") != [paths.obj]:
        raise ValueError("the GNU assembler job does not write the witness object")


GNU_TEMPORARY = re.compile(r"cc[A-Za-z0-9]{6}\.s")  # Used with fullmatch: no trailing newline.


def gnu_temporary(jobs: list[list[str]]) -> str:
    """GCC's per-expansion temporary assembly file: the cc1plus -o value, which must be an absolute path named
    cc + six alphanumerics + .s, occur exactly once in the compiler job (as its output) and exactly once in the
    assembler job (as its sole, final input)."""
    compiler, assembler = jobs
    outputs = [index for index, argument in enumerate(compiler[:-1]) if argument == "-o"]
    if len(outputs) != 1:
        raise ValueError("the GNU compiler job must name exactly one output")
    temporary = compiler[outputs[0] + 1]
    if not os.path.isabs(temporary) or not GNU_TEMPORARY.fullmatch(os.path.basename(temporary)):
        raise ValueError(f"the GNU compiler output is not a driver temporary assembly file: {temporary}")
    if compiler.count(temporary) != 1 or assembler.count(temporary) != 1 or assembler[-1] != temporary:
        raise ValueError("the GNU assembler job must read exactly the compiler's temporary output, once, last")
    return temporary


def plan_shape(plan: dict) -> dict:
    """The plan with only GCC's ephemeral temporary name masked (the driver picks a fresh one per expansion).

    The directory stays; only the basename of the one shared cc*.s temporary, at its two validated positions,
    becomes a placeholder. Any other difference - program, option, value, search list - remains visible.
    """
    jobs = [list(job) for job in plan["jobs"]]
    if len(jobs) == 2 and Path(jobs[0][0]).name == "cc1plus":
        temporary = gnu_temporary(jobs)
        masked = os.path.join(os.path.dirname(temporary), "cc<temporary>.s")
        jobs = [[masked if argument == temporary else argument for argument in job] for job in jobs]
    return dict(plan, jobs=jobs)


def observe(command: list[str], directory: str) -> str:
    """One driver query; a failing or erroring query is fatal."""
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"the compiler driver query failed: {' '.join(command[-1:])}: {result.stderr[-300:]}")
    return result.stdout + result.stderr


def driver_plan(executed: list[str], directory: str, identity: str, paths: JobPaths) -> dict:
    """Observe and validate the driver plan: its jobs and, for GNU, its program lookup authority."""
    expansion = observe(executed + ["-###"], directory)
    if CONFIGURATION_LINES.search(expansion):
        raise ValueError("the compiler driver reads a specs or configuration file")
    jobs = [shlex.split(line) for line in expansion.splitlines() if line.startswith(" ")]
    validate_plan(jobs, identity, paths)
    plan = {"jobs": jobs, "program_dirs": [], "path": []}
    if not identity.startswith(("AppleClang-", "Clang-")):
        dirs = re.findall(r"^programs: =(.*)$", observe(executed + ["-print-search-dirs"], directory), re.MULTILINE)
        if len(dirs) != 1:
            raise ValueError("the GNU driver did not report its program search directories")
        plan["program_dirs"] = absolute_search(dirs[0], "program directory list")
        plan["path"] = absolute_search(os.environ.get("PATH", ""), "PATH")
        for job in jobs:
            answer = observe(executed + [f"-print-prog-name={Path(job[0]).name}"], directory).strip()
            selected = lookup_program(job[0], plan)["selected"]
            if (os.path.isabs(answer) and resolve(answer)[0] != resolve(selected)[0]) or \
                    (not os.path.isabs(answer) and (answer != job[0] or selection_source(job[0], plan) != "path")):
                raise ValueError(f"the driver selects {answer} for {job[0]}, the witness lookup selects {selected}")
    return plan


def absolute_search(value: str, what: str) -> list[str]:
    """A search list whose every entry is absolute. Empty entries (the current directory of the executing
    process) and relative entries depend on the working directory: rejected, not reinterpreted."""
    entries = value.split(os.pathsep)
    if not value or any(not entry or not os.path.isabs(entry) for entry in entries):
        raise ValueError(f"the {what} has empty or relative entries (working-directory dependent): {value!r}")
    return entries


def selection_source(program: str, plan: dict) -> str:
    if os.path.isabs(program):
        return "absolute"
    for entry in plan["program_dirs"]:
        if os.path.isfile(os.path.join(entry, program)):
            return "program_dirs"
    return "path"


def lookup_program(program: str, plan: dict) -> dict:
    """Logical selection of a job program, replayable without the compiler.

    An absolute program is itself; a bare name is looked up as the GNU driver does - its program directories,
    then the executable search PATH. The record names the logical program, the directories examined before
    the selection (which must stay empty of it), the selected path, the symlinks followed resolving it, the
    resolved file and its content hash.
    """
    if os.path.isabs(program):
        examined, selected = [], program
    else:
        examined, selected = [], None
        for entry in plan["program_dirs"] + plan["path"]:
            candidate = os.path.join(entry, program)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                selected = candidate
                break
            examined.append(entry)
        if selected is None or not os.path.isabs(selected):
            raise ValueError(f"cannot select job program {program}")
    resolved, links = resolve(selected)
    return {"program": program, "examined": examined, "selected": selected, "links": links,
            "resolved": resolved, "sha256": envelope.digest(Path(resolved))}


def driver_identity(command: list[str]) -> dict:
    """Identity of the driver named by the authenticated target command (which must name it absolutely)."""
    if not os.path.isabs(command[0]):
        raise ValueError(f"the target command names its compiler driver relatively: {command[0]}")
    return lookup_program(command[0], {"program_dirs": [], "path": []})


def program_identities(plan: dict, command: list[str]) -> dict[str, dict]:
    """Identity of the driver (from the target command) and of every job program of the plan."""
    identities = {"driver": driver_identity(command)}
    for index, job in enumerate(plan["jobs"]):
        identities[f"job{index}"] = lookup_program(job[0], plan)
    return identities


def precompiled_headers(header_trace: str) -> list[str]:
    """PCH files the GNU compiler used or tried: -H prints them after the include depth dots (possibly none)
    marked '!' (valid) or 'x' (invalid)."""
    return [line for line in header_trace.splitlines() if re.match(r"^\.*[!x] \S", line)]


def replay_plan(plan: dict, directory: str, obj: Path, dependencies: Path, assembly: Path) -> None:
    """Retain the assembler input of the executed compile: replay its validated GNU plan with the temporary bound.

    The driver compile has just written the bound object and dependency file. The plan's own jobs - each
    program the selection the plan binds, every argument as observed - are executed again with only GCC's
    temporary assembly file (the compiler job's output and the assembler job's sole input) replaced by
    `assembly`. The assembler job therefore rewrites the bound object from exactly the retained assembly, and
    that object and the compiler job's dependency file must be byte-identical to what the driver compile
    wrote: the retained file is the program assembled into the bound object, not a second compilation's.
    The compiler job's -H trace must name no precompiled header. An integrated clang job assembles in memory
    and has no assembler input to retain, so ordered-add sites cannot be bound and clang plans fail closed.
    """
    jobs = plan["jobs"]
    if len(jobs) != 2 or Path(jobs[0][0]).name != "cc1plus":
        raise ValueError("the witness plan has no assembler input to retain (an integrated clang job assembles in "
                         "memory); ordered-add sites cannot be bound to the executed compile")
    temporary = gnu_temporary(jobs)
    compiled, recorded = obj.read_bytes(), dependencies.read_bytes()
    # The replay must write both again: a job that writes nothing cannot leave the driver's files standing.
    obj.unlink()
    dependencies.unlink()
    for index, job in enumerate(jobs):
        program = lookup_program(job[0], plan)["selected"]
        replayed = subprocess.run([program] + [str(assembly) if argument == temporary else argument
                                               for argument in job[1:]],
                                  cwd=directory, capture_output=True, text=True)
        if replayed.returncode:
            raise ValueError(f"the witness plan job {index} failed on replay: {replayed.stderr[-2000:]}")
        if index == 0 and precompiled_headers(replayed.stderr):
            raise ValueError("the replayed witness compiler job read precompiled headers")
    if not assembly.is_file() or not obj.is_file() or not dependencies.is_file() or \
            obj.read_bytes() != compiled or dependencies.read_bytes() != recorded:
        raise ValueError("the replayed witness plan does not reproduce the bound object and dependencies from the "
                         "retained assembly")


def line_witness_paths(output: Path) -> tuple[Path, Path, Path]:
    """The Release line witness artifacts next to a proof: object, dependency file and retained assembly."""
    return output.with_suffix(".lines.o"), output.with_suffix(".lines.deps"), output.with_suffix(".lines.asm")


LINE_WITNESS_KEYS = frozenset({"flag_delta", "command", "driver_plan", "object_file", "object_sha256",
                               "assembly_sha256", "release_object_sha256", "allocated_image_sha256"})


def build_line_witness(args: argparse.Namespace, command: list[str], directory: str, identity: str,
                       executed: list[str], outputs: tuple[Path, Path], output: Path, before: dict,
                       programs: dict) -> dict:
    """L-1..L-3: compile the executed argv plus `-g`, retain its assembler input through the same validated GNU
    plan replay, and prove its allocated image equal to the exact Release object's. Returns the bound record."""
    obj, dependencies = outputs
    lines_obj, lines_dependencies, lines_assembly = line_witness_paths(output)
    witness = line_witness_command(executed, obj, dependencies, lines_obj, lines_dependencies)
    plan = driver_plan(witness, directory, identity, JobPaths(args.source, str(lines_obj), str(lines_dependencies),
                                                              directory, None))
    validate_plan(plan["jobs"], identity, JobPaths(args.source, str(lines_obj), str(lines_dependencies), directory,
                                                   frozenset(before)))
    if program_identities(plan, command) != programs:
        raise ValueError("the line witness compile selects other compiler or assembler programs")
    compiled = subprocess.run(witness, cwd=directory, capture_output=True, text=True)
    if compiled.returncode:
        raise ValueError(f"the line witness compile failed: {compiled.stderr[-2000:]}")
    if precompiled_headers(compiled.stderr):
        raise ValueError("the line witness compile read precompiled headers")
    closure = read_depfile(lines_dependencies, directory)
    if closure != sorted(before) or bind_inputs(closure) != before:
        raise ValueError("the line witness compile read another dependency closure or an input changed")
    replay_plan(plan, directory, lines_obj, lines_dependencies, lines_assembly)
    if bind_inputs(closure) != before or program_identities(plan, command) != programs:
        raise ValueError("an input or program changed while the line witness plan was replayed")
    image = require_code_identity(obj.read_bytes(), lines_obj.read_bytes())
    return {"flag_delta": [LINE_WITNESS_FLAG], "command": witness, "driver_plan": plan,
            "object_file": lines_obj.name, "object_sha256": envelope.digest(lines_obj),
            "assembly_sha256": envelope.digest(lines_assembly), "release_object_sha256": envelope.digest(obj),
            "allocated_image_sha256": image}


def bind_line_witness(args: argparse.Namespace, document: dict, document_path: Path, command: list[str],
                      paths: "JobPaths") -> Path | None:
    """Re-check a document's Release line witness without a compiler (L-1..L-4); returns its retained assembly,
    or None when the executed argv already carries line tables (then no witness may be recorded)."""
    witness = document.get("line_witness")
    executed = document["executed_command"]
    if not line_witness_needed(executed):
        if witness is not None:
            raise ValueError("a line witness is recorded for an argv that already carries line tables")
        return None
    obj, dependencies = document_path.with_suffix(".o"), document_path.with_suffix(".deps")
    lines_obj, lines_dependencies, lines_assembly = line_witness_paths(document_path)
    if not isinstance(witness, dict) or set(witness) != LINE_WITNESS_KEYS or \
            witness["flag_delta"] != [LINE_WITNESS_FLAG] or witness["object_file"] != lines_obj.name or \
            witness["command"] != line_witness_command(executed, obj, dependencies, lines_obj, lines_dependencies) or \
            witness["release_object_sha256"] != document.get("object_sha256"):
        raise ValueError("the Release argv lacks line tables and the document binds no valid line witness")
    plan = witness["driver_plan"]
    if not isinstance(plan, dict) or set(plan) != {"jobs", "program_dirs", "path"}:
        raise ValueError("the line witness lacks a well-formed driver plan")
    validate_plan(plan["jobs"], args.identity, JobPaths(paths.source, str(lines_obj), str(lines_dependencies),
                                                        paths.directory, paths.closure))
    if program_identities(plan, command) != document.get("programs"):
        raise ValueError("the line witness plan selects other compiler or assembler programs")
    if not lines_obj.is_file() or witness["object_sha256"] != envelope.digest(lines_obj) or \
            not lines_assembly.is_file() or witness["assembly_sha256"] != envelope.digest(lines_assembly):
        raise ValueError("the line witness object or its retained assembly is missing or changed")
    if require_code_identity(obj.read_bytes(), lines_obj.read_bytes()) != witness["allocated_image_sha256"]:
        raise ValueError("the line witness allocated image digest changed")
    return lines_assembly


def located_assembly(proof_path: Path, document: dict) -> tuple[Path, str]:
    """The retained assembly whose `.loc` locations a document's analysis read, with its bound digest."""
    witness = document.get("line_witness")
    if witness:
        return line_witness_paths(proof_path)[2], witness.get("assembly_sha256")
    return proof_path.with_suffix(".asm"), document.get("assembly_sha256")


def build(args: argparse.Namespace) -> None:
    """Compile the probe object with the target command, disassemble it, audit it, publish the proof last."""
    require_absolute_source(args.source)
    output = Path(args.proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    obj, dependencies = output.with_suffix(".o"), output.with_suffix(".deps")
    disassembly_path, translation_unit = output.with_suffix(".dis"), output.with_suffix(".i")
    assembly_path = output.with_suffix(".asm")
    preprocessed, depfile, temporary = output.with_suffix(".pre.deps"), output.with_suffix(".d"), output.with_suffix(".tmp")
    # proof.s / proof.sym: retired artifacts of earlier schemas.
    # The seal and the analysis record are retired before the attempt: nothing of an earlier run survives a
    # failure (DR-1).
    for stale in (output, analysis_path(output), temporary, obj, dependencies, preprocessed, depfile, disassembly_path,
                  translation_unit, assembly_path, output.with_suffix(".s"), output.with_suffix(".sym"),
                  *line_witness_paths(output)):
        stale.unlink(missing_ok=True)
    command, directory = envelope.target_command(args)
    # Epoch: the driver is bound before any observation the proof consumes; the plan and every job program are
    # bound before any job runs; all of it must be unchanged after compiling (and the plan observed again).
    driver = driver_identity(command)
    version = subprocess.check_output([command[0], "--version"], cwd=directory, text=True)
    identity = compiler_identity(version)
    if identity != args.identity:
        raise ValueError(f"compiler reports identity {identity}, configured budget identity is {args.identity}")
    executed = executed_command(command, args.source, obj, dependencies)
    plan = driver_plan(executed, directory, identity, JobPaths(args.source, str(obj), str(dependencies), directory,
                                                               None))
    programs = program_identities(plan, command)
    if programs["driver"] != driver:
        raise ValueError("the compiler driver changed while its plan was observed")
    subprocess.run(command + WITNESS_FLAGS + ["-M", "-MF", str(preprocessed), "-MT", "probe", args.source],
                   cwd=directory, check=True, stdout=subprocess.DEVNULL)
    before = bind_inputs(read_depfile(preprocessed, directory))
    paths = JobPaths(args.source, str(obj), str(dependencies), directory, frozenset(before))
    validate_plan(plan["jobs"], identity, paths)
    subprocess.run(command + WITNESS_FLAGS + ["-E", args.source, "-o", str(translation_unit)], cwd=directory, check=True)
    unproved = inline_assembly_violations(translation_unit.read_text(errors="replace"))
    if unproved:
        raise ValueError(f"inline assembly the dependency closure cannot bind (non-empty template): {unproved[:2]}")
    if program_identities(plan, command) != programs:
        raise ValueError("a compiler or assembler program changed during preprocessing")
    compiled = subprocess.run(executed, cwd=directory, capture_output=True, text=True)
    if compiled.returncode:
        raise ValueError(f"the witness compile failed: {compiled.stderr[-2000:]}")
    used = precompiled_headers(compiled.stderr)
    if used:
        raise ValueError(f"the witness compile read precompiled headers: {used[:2]}")
    closure = read_depfile(dependencies, directory)
    if closure != sorted(before):
        raise ValueError("the compilation's dependency closure differs from its preprocessing closure")
    after = bind_inputs(closure)
    if after != before:
        changed = sorted(path for path in closure if before[path] != after[path])
        raise ValueError(f"inputs changed while the witness compiled: {changed[:3]}")
    if program_identities(plan, command) != programs or \
            plan_shape(driver_plan(executed, directory, identity, paths)) != plan_shape(plan) or \
            program_identities(plan, command) != programs:
        # The identities are checked again after the final plan observation, so nothing observed before
        # publication escapes the epoch.
        raise ValueError("a compiler or assembler program, or the driver plan, changed while the witness compiled")
    replay_plan(plan, directory, obj, dependencies, assembly_path)
    if bind_inputs(closure) != before or program_identities(plan, command) != programs:
        raise ValueError("an input or a compiler or assembler program changed while the witness plan was replayed")
    line_witness, located = None, assembly_path
    if line_witness_needed(executed):
        line_witness = build_line_witness(args, command, directory, identity, executed, (obj, dependencies),
                                          output, before, programs)
        located = line_witness_paths(output)[2]
    disassembly, disassembly_command, tool_version = disassemble(args.disassembler, obj)
    disassembly_path.write_text(disassembly)
    # Structural analysis first (a failure here writes nothing), then the admissions. The analysis reads the
    # exact executed object and its disassembly; a Release line witness supplies only the `.loc` locations (L-3).
    fields = analysis_fields(obj.read_bytes(), disassembly, located.read_text(errors="replace"), after, args.mode)
    failures = admission_failures(fields, args.identity, args.mode)
    result = {
        "schema": 12, "kind": SEALED_KIND, "admission": {"status": "passed", "failures": []},
        "signature": args.signature, "identity": identity, "mode": args.mode,
        "checker_sha256": envelope.digest(Path(__file__)),
        "envelope_sha256": envelope.digest(_ENVELOPE),
        "budget_sha256": envelope.digest(BUDGET_PATH),
        "ordered_add_sites_sha256": envelope.digest(ORDERED_ADD_SITES_PATH),
        "chain_counts_sha256": envelope.digest(CHAIN_COUNTS_PATH),
        "compiler_version": version,
        "executed_command": executed, "executed_directory": directory,
        "target_command_sha256": envelope.command_digest(command, directory),
        "driver_plan": plan, "programs": programs,
        "inputs": after,
        "object_file": obj.name, "object_sha256": envelope.digest(obj),
        "disassembler": args.disassembler, "disassembler_version": tool_version,
        "disassembler_sha256": envelope.digest(Path(os.path.realpath(args.disassembler))),
        "disassembly_command": disassembly_command, "disassembly_sha256": envelope.digest(disassembly_path),
        "translation_unit_sha256": envelope.digest(translation_unit),
        "assembly_sha256": envelope.digest(assembly_path),
        "line_witness": line_witness,
        **fields,
    }
    # Ninja depfile: resolved paths (Ninja canonicalises ".." lexically, which would name another file).
    target = args.depfile_target or str(output)
    resolved = sorted({identity["resolved"] for identity in after.values()})
    depfile.write_text(f"{target}: " + " ".join(path.replace(" ", "\\ ") for path in resolved) + "\n")
    if failures:
        # Structurally complete but not admitted: the unqualified analysis record (diagnostic), never the seal.
        record = {**result, "schema": ANALYSIS_SCHEMA, "kind": ANALYSIS_KIND, "diagnostic": True,
                  "admission": {"status": "failed", "failures": failures}}
        temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        temporary.replace(analysis_path(output))
        raise ValueError(f"tile register admission failed ({len(failures)}; unqualified analysis record written): "
                         + "; ".join(failure["detail"] for failure in failures)[:3000])
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)


class VerifiedAnalysis(NamedTuple):
    """One attempt's verified analysis, frozen (V1): only immutable values (str, bytes). Produced once per driver
    attempt after every binding was re-hashed and the analysis re-run and found equal to the document; the stack
    gate and the fused witness read it instead of verifying and analysing again. Never written to disk, never
    reused across attempts or processes."""
    kind: str
    document_text: str
    data: bytes
    disassembly: str
    assembly: str
    fields_text: str
    checker_sha256: str


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def verify(args: argparse.Namespace) -> None:
    """Validate the retained seal (the standalone verify; the driver uses verified_analysis())."""
    verified_analysis(args)


def verified_analysis(args: argparse.Namespace, analyses: dict | None = None) -> VerifiedAnalysis:
    """Validate the retained seal: inputs, command, the bound object, and its re-disassembly by the bound tool, then
    the re-run analysis and every admission. An unqualified analysis record is never a seal: verify rejects one
    under any name, and a seal next to an analysis record (never written by one sound run) is an error."""
    require_absolute_source(args.source)
    proof_path = Path(args.proof)
    if analysis_path(proof_path).exists():
        raise ValueError("an unqualified analysis record lies next to the proof: the last build failed admission (or "
                         "the outputs are stale); the proof cannot verify")
    proof = json.loads(proof_path.read_text())
    if proof.get("kind") == ANALYSIS_KIND or proof.get("schema") == ANALYSIS_SCHEMA:
        raise ValueError("an unqualified analysis record is not a sealed proof")
    if (proof.get("schema") != 12 or proof.get("kind") != SEALED_KIND or proof.get("signature") != args.signature or
            proof.get("identity") != args.identity or proof.get("mode") != args.mode or
            proof.get("admission") != {"status": "passed", "failures": []}):
        raise ValueError("tile register proof has a foreign schema/kind/configuration")
    document_text = proof_path.read_text()
    data, disassembly, assembly = bind_document(args, proof, proof_path)
    fields = analysis_fields(data, disassembly, assembly, proof["inputs"], args.mode, analyses)
    require_fields(proof, fields)
    failures = admission_failures(fields, args.identity, args.mode)
    if failures:
        raise ValueError(f"tile register admission failed ({len(failures)}): "
                         + "; ".join(failure["detail"] for failure in failures)[:3000])
    return VerifiedAnalysis(SEALED_KIND, document_text, data, disassembly, assembly,
                            canonical({name: fields[name] for name in ANALYSIS_FIELDS}), envelope.digest(Path(__file__)))


def require_fields(document: dict, fields: dict | None) -> None:
    """The document's analysis fields equal the re-run analysis of its bound object."""
    messages = {"program_violations": "tile register evidence changed",
                "ordered_adds": "retained probe assembly or its ordered-add sites changed",
                "stack_traffic": "recorded stack traffic (classes, loops or their source names) changed",
                "chain_witness": "the probe's chain witness (tile chain) changed"}
    for name in ANALYSIS_FIELDS:
        if name not in document or fields is None or document[name] != fields[name]:
            raise ValueError(messages[name])


def bind_document(args: argparse.Namespace, document: dict, document_path: Path) -> tuple[bytes, str, str]:
    """(object bytes, disassembly, assembly) of a seal or an analysis record, after checking every binding it
    records: the tool and committed-count hashes, the dependency closure (sources included), the target command,
    the object, the driver plan and its programs, the disassembler and the re-disassembly, the translation unit
    and the retained assembly."""
    if compiler_identity(str(document.get("compiler_version", ""))) != args.identity:
        raise ValueError("recorded compiler identity does not match the configured budget identity")
    for key, path in (("checker_sha256", Path(__file__)), ("envelope_sha256", _ENVELOPE),
                      ("budget_sha256", BUDGET_PATH), ("ordered_add_sites_sha256", ORDERED_ADD_SITES_PATH),
                      ("chain_counts_sha256", CHAIN_COUNTS_PATH)):
        if document.get(key) != envelope.digest(path):
            raise ValueError(f"stale tile register tool: {path}")
    inputs = document.get("inputs")
    source = resolve(args.source)[0]
    if not isinstance(inputs, dict) or not any(isinstance(identity, dict) and identity.get("resolved") == source
                                               for identity in inputs.values()):
        raise ValueError("tile register proof does not bind the probe's dependency closure")
    for path, expected in inputs.items():
        if not os.path.isfile(path) or input_identity(path) != expected:
            raise ValueError(f"stale tile register input (source, header or symlink changed): {path}")
    command, directory = envelope.target_command(args)
    if document.get("target_command_sha256") != envelope.command_digest(command, directory):
        raise ValueError("actual target compile command changed after the register witness")
    obj = document_path.with_suffix(".o")
    if document.get("object_file") != obj.name or document.get("executed_directory") != directory or \
            document.get("executed_command") != executed_command(command, args.source, obj, document_path.with_suffix(".deps")):
        raise ValueError("recorded witness invocation is not the target command over the probe")
    if not obj.is_file() or document.get("object_sha256") != envelope.digest(obj):
        raise ValueError("tile register object is missing or changed")
    plan = document.get("driver_plan")
    if not isinstance(plan, dict) or set(plan) != {"jobs", "program_dirs", "path"} or \
            not all(isinstance(plan[key], list) and all(isinstance(entry, str) for entry in plan[key])
                    for key in ("program_dirs", "path")):
        raise ValueError("tile register proof lacks a well-formed driver plan")
    paths = JobPaths(args.source, str(obj), str(document_path.with_suffix(".deps")), directory, frozenset(inputs))
    validate_plan(plan["jobs"], args.identity, paths)
    expected_keys = {"driver"} | {f"job{index}" for index in range(len(plan["jobs"]))}
    programs = document.get("programs")
    if not isinstance(programs, dict) or set(programs) != expected_keys or \
            any(programs[f"job{index}"].get("program") != job[0] for index, job in enumerate(plan["jobs"])):
        raise ValueError("tile register proof does not identify exactly the driver and every job program")
    for key in ("program_dirs", "path"):
        if plan[key] and absolute_search(os.pathsep.join(plan[key]), key) != plan[key]:
            raise ValueError("recorded search list is not absolute")
    try:
        current = program_identities(plan, command)
    except (ValueError, OSError) as error:
        raise ValueError(f"a compiler or assembler program of the witness compile changed: {error}")
    if current != programs:
        raise ValueError("a compiler or assembler program of the witness compile changed (content, symlink, "
                         "selection or search shadowing)")
    if document.get("disassembler") != args.disassembler or \
            document.get("disassembler_sha256") != envelope.digest(Path(os.path.realpath(args.disassembler))):
        raise ValueError("configured disassembler changed after the register witness")
    disassembly, disassembly_command, tool_version = disassemble(args.disassembler, obj)
    if disassembly_command != document.get("disassembly_command") or tool_version != document.get("disassembler_version"):
        raise ValueError("disassembler invocation changed after the register witness")
    retained = document_path.with_suffix(".dis")
    if not retained.is_file() or retained.read_text() != disassembly or \
            document.get("disassembly_sha256") != envelope.digest(retained):
        raise ValueError(f"retained disassembly does not match the bound object: {retained.name}")
    unit = document_path.with_suffix(".i")
    if not unit.is_file() or document.get("translation_unit_sha256") != envelope.digest(unit) or \
            inline_assembly_violations(unit.read_text(errors="replace")):
        raise ValueError("retained translation unit is missing, changed or has unproved inline assembly")
    assembly = document_path.with_suffix(".asm")
    if not assembly.is_file() or document.get("assembly_sha256") != envelope.digest(assembly):
        raise ValueError("retained probe assembly or its ordered-add sites changed")
    located = bind_line_witness(args, document, document_path, command, paths)
    return obj.read_bytes(), disassembly, (located or assembly).read_text(errors="replace")


def check_analysis_record(args: argparse.Namespace) -> dict:
    """The standalone record check (check-analysis); returns the record."""
    return json.loads(unqualified_analysis(args).document_text)


def unqualified_analysis(args: argparse.Namespace, analyses: dict | None = None) -> VerifiedAnalysis:
    """An unqualified analysis record (proof.analysis.json), bound exactly like a seal and re-analysed (UA-1, UA-2):
    its kind and schema, its failed admission with typed failures, every bound field, every analysis field equal
    to the re-run analysis of its object, and the failures equal to the recomputed ones. A seal (under either
    name), a seal next to the record, a missing field or status, a stale tool, count file, source or object fail.
    Returns the record (diagnostic only: never a seal, never Gate7 or candidate evidence)."""
    require_absolute_source(args.source)
    proof_path = Path(args.proof)
    record_path = analysis_path(proof_path)
    if proof_path.exists():
        raise ValueError("a sealed proof and an unqualified analysis record are both present" if record_path.exists()
                         else "this run is sealed: a sealed proof is never consumed as an unqualified analysis record")
    if not record_path.exists():
        raise ValueError("no unqualified analysis record of this run")
    document_text = record_path.read_text()
    record = json.loads(document_text)
    if record.get("kind") != ANALYSIS_KIND or record.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("not an unqualified analysis record (a sealed proof is never consumed as one)")
    missing = [name for name in BOUND_FIELDS + ANALYSIS_FIELDS + ("admission",) if name not in record]
    if missing:
        raise ValueError(f"the unqualified analysis record lacks fields {missing[:4]}")
    failures = check_admission(record["admission"])
    if record.get("signature") != args.signature or record.get("identity") != args.identity or \
            record.get("mode") != args.mode:
        raise ValueError("the unqualified analysis record has a foreign configuration")
    data, disassembly, assembly = bind_document(args, record, proof_path)
    fields = analysis_fields(data, disassembly, assembly, record["inputs"], args.mode, analyses)
    require_fields(record, fields)
    recomputed = admission_failures({name: record[name] for name in ANALYSIS_FIELDS}, args.identity, args.mode)
    if not recomputed:
        raise ValueError("the analysis is admissible: the build must seal it, not record it as unqualified")
    if recomputed != failures:
        raise ValueError("the record's admission failures are not the recomputed ones")
    return VerifiedAnalysis(ANALYSIS_KIND, document_text, data, disassembly, assembly,
                            canonical({name: fields[name] for name in ANALYSIS_FIELDS}), envelope.digest(Path(__file__)))


def fused_witness(args: argparse.Namespace, data: bytes, disassembly: str, assembly: str, inputs: dict) -> dict:
    """The fused witness of bound artifacts against the committed counts (raises on any difference)."""
    fused = fused_evidence(data, disassembly, assembly, PROBE_COPY, closure_sources(inputs), scope="prelink")
    suffix = f"@{PROBE_COPY}"
    witness = {"fused": {key[:-len(suffix)]: value for key, value in sorted(fused.items())}}
    require_chain_counts(witness, args.identity, args.mode)
    return witness


def fused_from_analysis(args: argparse.Namespace, verified: VerifiedAnalysis) -> dict:
    """The fused stage on this attempt's verified analysis (V1): its artifacts were re-hashed when it was made; the
    receipt is written as by fused_stage(). Raises on failure (after writing the failed receipt)."""
    receipt = Path(args.proof).with_name("fused-stage.json")
    receipt.unlink(missing_ok=True)
    document = json.loads(verified.document_text)
    base = {"stage": "fused", "input": verified.kind, "verified": verified.kind == SEALED_KIND,
            "scope": "prelink-selection-independent (probe: no link authority, every weak copy here audited)",
            "checker_sha256": verified.checker_sha256, "document_sha256": hashlib.sha256(
                verified.document_text.encode()).hexdigest()}
    try:
        witness = fused_witness(args, verified.data, verified.disassembly, verified.assembly, document["inputs"])
    except (ValueError, OSError) as error:
        receipt.write_text(json.dumps({**base, "status": "failed", "reason": str(error)}, indent=1) + "\n")
        raise
    result = {**base, "status": "passed", "witness": witness}
    receipt.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def run_document(proof_path: Path) -> tuple[str, dict]:
    """(kind, document) of this run's stage-1 output: exactly one of the seal and the unqualified analysis record
    (the driver retires both before the attempt, DR-1). Neither (stage 1 failed structurally) or both fail."""
    seal, record = proof_path.exists(), analysis_path(proof_path).exists()
    if seal and record:
        raise ValueError("a sealed proof and an unqualified analysis record are both present")
    if not seal and not record:
        raise ValueError("no sealed proof and no unqualified analysis record of this run (stage 1 failed "
                         "structurally): nothing is bound to read")
    document = json.loads((proof_path if seal else analysis_path(proof_path)).read_text())
    kind = SEALED_KIND if seal else ANALYSIS_KIND
    if document.get("kind") != kind:
        raise ValueError(f"the stage-1 document is not a {kind}")
    return kind, document


def fused_stage(args: argparse.Namespace) -> None:
    """S2-4c as its own fail-closed stage of the proof driver (CPUNativeTileProofDriver.py, after the proof and the
    stack gate): the fused-instruction witness of the bound object (fused_evidence(): explicit source FMAs only;
    weak/global callees need P2 link selection, which an unlinked probe object cannot have) must equal the
    committed `fused` counts. It reads only this run's artifacts (DR-1): the object, disassembly and assembly must
    have the hashes this run's seal or unqualified analysis record binds, the record's checker must be this
    checker and every closure input must be unchanged; "verified" says whether that document is a seal. A receipt
    records the outcome; any failure exits non-zero. Diagnostic only, never a production certification."""
    proof_path = Path(args.proof)
    receipt = proof_path.with_name("fused-stage.json")
    receipt.unlink(missing_ok=True)
    kind = None
    try:
        kind, document = run_document(proof_path)
        located, located_sha256 = located_assembly(proof_path, document)
        files = {"object_sha256": proof_path.with_suffix(".o"), "disassembly_sha256": proof_path.with_suffix(".dis"),
                 "assembly_sha256": proof_path.with_suffix(".asm")}
        for field, path in files.items():
            if not path.is_file() or document.get(field) != envelope.digest(path):
                raise ValueError(f"{path.name} is not the artifact this run's {kind} binds (stale or replaced)")
        if not located.is_file() or located_sha256 != envelope.digest(located):
            raise ValueError(f"{located.name} is not the line witness assembly this run's {kind} binds")
        if document.get("checker_sha256") != envelope.digest(Path(__file__)):
            raise ValueError(f"this run's {kind} was written by another checker")
        inputs = document.get("inputs")
        if not isinstance(inputs, dict) or any(not os.path.isfile(path) or input_identity(path) != identity
                                               for path, identity in inputs.items()):
            raise ValueError("a closure input changed after this run's stage 1")
        witness = fused_witness(args, proof_path.with_suffix(".o").read_bytes(),
                                proof_path.with_suffix(".dis").read_text(),
                                located.read_text(errors="replace"), inputs)
    except (ValueError, OSError) as error:
        receipt.write_text(json.dumps({"stage": "fused", "status": "failed", "input": kind,
                                       "verified": kind == SEALED_KIND, "reason": str(error),
                                       "checker_sha256": envelope.digest(Path(__file__))}, indent=1) + "\n")
        raise
    receipt.write_text(json.dumps({"stage": "fused", "status": "passed", "input": kind, "verified": kind == SEALED_KIND,
                                   "witness": witness, "checker_sha256": envelope.digest(Path(__file__))},
                                  indent=1, sort_keys=True) + "\n")


def census(args: argparse.Namespace) -> None:
    """Maintainer tool: print the per-program spill census for qualifying a budget (no budget applied)."""
    require_absolute_source(args.source)
    command, directory = envelope.target_command(args)
    version = subprocess.check_output([command[0], "--version"], cwd=directory, text=True)
    obj = Path(args.proof).with_suffix(".census.o")
    obj.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command + WITNESS_FLAGS + ["-c", args.source, "-o", str(obj)], cwd=directory, check=True)
    disassembly, *_ = disassemble(args.disassembler, obj)
    counts = audit(obj.read_bytes(), disassembly, args.mode)
    print(json.dumps({"identity": compiler_identity(version), "mode": args.mode,
                      "programs": {program_key(name): count for name, count in counts.items()}},
                     indent=1, sort_keys=True))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """The checker's command line (also parsed in-process by the proof driver)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "census", "fused", "check-analysis"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--proof", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--compile-database", required=True)
    parser.add_argument("--target-source", required=True)
    parser.add_argument("--identity", required=True, help="compiler id and major version, e.g. GNU-14")
    parser.add_argument("--mode", required=True, help="native ISA envelope, e.g. NEON_I8MM")
    parser.add_argument("--disassembler", required=True, help="objdump (GNU binutils or LLVM) for the target")
    parser.add_argument("--depfile-target", help="depfile output path as the generator names it")
    return parser.parse_args(argv)


def main() -> None:
    """Run the requested build/verification phase; errors are fatal, never skips."""
    args = parse_arguments()
    if args.action == "census":
        census(args)
        return
    if args.action == "fused":
        fused_stage(args)
        return
    if args.action == "check-analysis":
        record = check_analysis_record(args)
        print(f"UNQUALIFIED: the analysis record is bound and re-analysed; admission failed "
              f"({len(record['admission']['failures'])} typed failures)")
        return
    if args.action == "build":
        build(args)
    verify(args)
    count = len(json.loads(Path(args.proof).read_text())["program_violations"])
    print(f"PASS: {count} native NEON tile programs: single-row programs spill nothing, grouped programs within budget")


if __name__ == "__main__":
    main()
