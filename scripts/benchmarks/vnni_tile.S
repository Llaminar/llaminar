# 16 rows x 24 cols VNNI int8 tile (vpdpbusd, M-in-lanes, s8 x u8 -> s32).
# Apack [mb][k4][16 rows x 4B]=64B/4k; Bpack [g][k4][24 cols x 4B]=96B/4k.
.intel_syntax noprefix
.text
.globl vntile8
vntile8:
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
    vpxord zmm16,zmm16,zmm16
    vpxord zmm17,zmm17,zmm17
    vpxord zmm18,zmm18,zmm18
    vpxord zmm19,zmm19,zmm19
    vpxord zmm20,zmm20,zmm20
    vpxord zmm21,zmm21,zmm21
    vpxord zmm22,zmm22,zmm22
    vpxord zmm23,zmm23,zmm23
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r10, rcx
    mov r9, r8
    test r10, r10
    je .Ldone
.Lloop:
    vmovdqu64 zmm24, [r12]
    vpbroadcastd zmm25, DWORD PTR [r13+0]
    vpdpbusd zmm0, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+4]
    vpdpbusd zmm1, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+8]
    vpdpbusd zmm2, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+12]
    vpdpbusd zmm3, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+16]
    vpdpbusd zmm4, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+20]
    vpdpbusd zmm5, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+24]
    vpdpbusd zmm6, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+28]
    vpdpbusd zmm7, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+32]
    vpdpbusd zmm8, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+36]
    vpdpbusd zmm9, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+40]
    vpdpbusd zmm10, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+44]
    vpdpbusd zmm11, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+48]
    vpdpbusd zmm12, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+52]
    vpdpbusd zmm13, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+56]
    vpdpbusd zmm14, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+60]
    vpdpbusd zmm15, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+64]
    vpdpbusd zmm16, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+68]
    vpdpbusd zmm17, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+72]
    vpdpbusd zmm18, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+76]
    vpdpbusd zmm19, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+80]
    vpdpbusd zmm20, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+84]
    vpdpbusd zmm21, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+88]
    vpdpbusd zmm22, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+92]
    vpdpbusd zmm23, zmm24, zmm25
    add r12, 64
    add r13, 96
    dec r10
    jnz .Lloop
.Ldone:
.Lst:
    xor r11, r11
    vextracti32x4 xmm27, zmm0, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm0, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm0, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm0, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm1, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm1, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm1, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm1, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm2, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm2, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm2, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm2, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm3, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm3, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm3, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm3, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm4, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm4, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm4, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm4, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm5, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm5, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm5, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm5, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm6, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm6, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm6, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm6, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm7, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm7, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm7, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm7, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm8, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm8, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm8, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm8, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm9, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm9, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm9, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm9, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm10, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm10, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm10, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm10, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm11, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm11, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm11, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm11, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm12, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm12, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm12, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm12, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm13, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm13, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm13, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm13, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm14, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm14, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm14, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm14, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm15, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm15, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm15, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm15, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm16, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm16, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm16, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm16, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm17, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm17, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm17, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm17, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm18, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm18, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm18, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm18, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm19, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm19, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm19, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm19, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm20, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm20, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm20, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm20, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm21, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm21, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm21, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm21, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm22, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm22, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm22, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm22, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    add r14, 4
    xor r11, r11
    xor r11, r11
    vextracti32x4 xmm27, zmm23, 0
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm23, 1
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm23, 2
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextracti32x4 xmm27, zmm23, 3
    vpextrd DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vpextrd DWORD PTR [r14+r11], xmm27, 3
    pop r15; pop r14; pop r13; pop r12; pop rbx
    ret
