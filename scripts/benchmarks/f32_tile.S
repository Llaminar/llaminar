# 16 rows x 16 cols FMA32 tile (512-bit vfmadd231ps, M-in-lanes).
# Apack [mb][k][16 rows x 4B]=64B/elem; Bpack [g][k][16 cols x 4B]=64B/elem.
.intel_syntax noprefix
.text
.globl f32tile
f32tile:
    push rbx; push r12; push r13; push r14; push r15
    vpxord zmm0,zmm0,zmm0
    vpxord zmm1,zmm1,zmm1
    vpxord zmm2,zmm2,zmm2
    vpxord zmm3,zmm3,zmm3
    vpxord zmm4,zmm4,zmm4
    vpxord zmm5,zmm5,zmm5
    vpxord zmm6,zmm6,zmm6
    vpxord zmm7,zmm7,zmm7
    vpxord zmm8,zmm8,zmm8
    vpxord zmm9,zmm9,zmm9
    vpxord zmm10,zmm10,zmm10
    vpxord zmm11,zmm11,zmm11
    vpxord zmm12,zmm12,zmm12
    vpxord zmm13,zmm13,zmm13
    vpxord zmm14,zmm14,zmm14
    vpxord zmm15,zmm15,zmm15
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r10, rcx
    mov r9, r8
    test r10, r10
    je .Ldone
.Lloop:
    vmovups zmm24, [r12]
    vbroadcastss zmm25, DWORD PTR [r13+0]
    vfmadd231ps zmm0, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+4]
    vfmadd231ps zmm1, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+8]
    vfmadd231ps zmm2, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+12]
    vfmadd231ps zmm3, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+16]
    vfmadd231ps zmm4, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+20]
    vfmadd231ps zmm5, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+24]
    vfmadd231ps zmm6, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+28]
    vfmadd231ps zmm7, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+32]
    vfmadd231ps zmm8, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+36]
    vfmadd231ps zmm9, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+40]
    vfmadd231ps zmm10, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+44]
    vfmadd231ps zmm11, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+48]
    vfmadd231ps zmm12, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+52]
    vfmadd231ps zmm13, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+56]
    vfmadd231ps zmm14, zmm24, zmm25
    vbroadcastss zmm25, DWORD PTR [r13+60]
    vfmadd231ps zmm15, zmm24, zmm25
    add r12, 64
    add r13, 64
    dec r10
    jnz .Lloop
.Ldone:
.Lst:
    xor r11, r11
    vextractf32x4 xmm27, zmm0, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm0, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm0, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm0, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm1, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm1, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm1, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm1, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm2, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm2, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm2, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm2, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm3, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm3, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm3, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm3, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm4, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm4, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm4, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm4, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm5, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm5, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm5, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm5, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm6, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm6, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm6, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm6, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm7, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm7, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm7, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm7, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm8, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm8, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm8, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm8, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm9, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm9, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm9, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm9, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm10, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm10, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm10, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm10, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm11, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm11, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm11, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm11, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm12, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm12, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm12, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm12, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm13, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm13, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm13, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm13, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm14, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm14, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm14, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm14, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextractf32x4 xmm27, zmm15, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm15, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm15, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm15, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    pop r15; pop r14; pop r13; pop r12; pop rbx
    ret
