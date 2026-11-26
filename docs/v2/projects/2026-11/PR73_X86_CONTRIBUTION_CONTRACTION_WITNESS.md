# PR73 bounded x86 contraction witness

Source: `e4bcdac3e4ce4ee4363ba60baa2de0aba85ddf52` plus the committed witness fixture.
Image: `sha256:00fea393290d0e867abb0c32fb45d1960ad3aad950cd766dab9093073c455300`.

One helper TU per ISA, compile/disassembly only; no production build or device execution.
The actual single-scale contribution helpers retain zero fused expert instructions,
while intentional native FMA and deliberately unguarded negative controls each fuse once.
No production flag or arithmetic change was needed at these sites.

This does NOT close B5 globally: corrected-scale helpers, coded leaves,
planning copies and all per-TU/post-link production copies remain unwitnessed.
D0 workspaceFloats COMDAT evidence is unrelated to this contraction proof.

## Exact compiler output

```text
sha256:00fea393290d0e867abb0c32fb45d1960ad3aad950cd766dab9093073c455300
g++ (Ubuntu 14.2.0-4ubuntu2~24.04.1) 14.2.0
Copyright (C) 2024 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


g++ -std=c++20 -O3 -ffp-contract=fast -mfma -mavx2 -I/work/src/v2 -c /work/tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp -o /tmp/cpu-contraction-e1vjq2pm/avx2.o
avx2 expertAVX2: fused=0, expected=0
avx2 nativeAVX2: fused=1, expected=1
avx2 negativeAVX2: fused=1, expected=1
g++ -std=c++20 -O3 -ffp-contract=fast -mfma -mavx512f -I/work/src/v2 -c /work/tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp -o /tmp/cpu-contraction-e1vjq2pm/avx512f.o
avx512f expertAVX2: fused=0, expected=0
avx512f nativeAVX2: fused=1, expected=1
avx512f negativeAVX2: fused=1, expected=1
avx512f expertAVX512: fused=0, expected=0
avx512f nativeAVX512: fused=1, expected=1
avx512f negativeAVX512: fused=1, expected=1

```
