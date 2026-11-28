# 16 rows x 24 cols M-in-lanes tile, register-broadcast packed.
# Apack [mb][kp][16 rows x 4B] = 64B per pair; Bpack [g][kp][24 cols x 4B] = 96B per pair.
# rdi=Ap rsi=Bp rdx=C rcx=kpairs r8=Nbytes; accs zmm0-23 (lanes = 16 rows of col c);
# r12=A r13=B r14=C r10=cnt r9=tmp.
.intel_syntax noprefix
.text
.globl m16k
m16k:
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
    mov r9, r8
    mov r10, rcx
    test r10, r10
    je .Ldone
.Lloop:
    vmovdqu64 zmm24, [r12]
    vpbroadcastd zmm25, DWORD PTR [r13]
    vdpbf16ps zmm0, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+4]
    vdpbf16ps zmm1, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+8]
    vdpbf16ps zmm2, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+12]
    vdpbf16ps zmm3, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+16]
    vdpbf16ps zmm4, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+20]
    vdpbf16ps zmm5, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+24]
    vdpbf16ps zmm6, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+28]
    vdpbf16ps zmm7, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+32]
    vdpbf16ps zmm8, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+36]
    vdpbf16ps zmm9, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+40]
    vdpbf16ps zmm10, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+44]
    vdpbf16ps zmm11, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+48]
    vdpbf16ps zmm12, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+52]
    vdpbf16ps zmm13, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+56]
    vdpbf16ps zmm14, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+60]
    vdpbf16ps zmm15, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+64]
    vdpbf16ps zmm16, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+68]
    vdpbf16ps zmm17, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+72]
    vdpbf16ps zmm18, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+76]
    vdpbf16ps zmm19, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+80]
    vdpbf16ps zmm20, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+84]
    vdpbf16ps zmm21, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+88]
    vdpbf16ps zmm22, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+92]
    vdpbf16ps zmm23, zmm24, zmm25
    add r12, 64
    add r13, 96
    dec r10
    jnz .Lloop
.Ldone:
    # store: acc c (16 rows) to C + row*Nbytes + c*4  (scattered 4B stores)
    xor r11, r11
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
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
    add r14, 4
    xor r11, r11
    vextractf32x4 xmm27, zmm16, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm16, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm16, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm16, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r14, 4
    xor r11, r11
    vextractf32x4 xmm27, zmm17, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm17, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm17, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm17, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r14, 4
    xor r11, r11
    vextractf32x4 xmm27, zmm18, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm18, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm18, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm18, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r14, 4
    xor r11, r11
    vextractf32x4 xmm27, zmm19, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm19, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm19, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm19, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r14, 4
    xor r11, r11
    vextractf32x4 xmm27, zmm20, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm20, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm20, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm20, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r14, 4
    xor r11, r11
    vextractf32x4 xmm27, zmm21, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm21, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm21, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm21, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r14, 4
    xor r11, r11
    vextractf32x4 xmm27, zmm22, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm22, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm22, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm22, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r14, 4
    xor r11, r11
    vextractf32x4 xmm27, zmm23, 0
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm23, 1
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm23, 2
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    add r11, r9
    vextractf32x4 xmm27, zmm23, 3
    vextractps DWORD PTR [r14+r11], xmm27, 0
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 1
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 2
    add r11, r9
    vextractps DWORD PTR [r14+r11], xmm27, 3
    pop r15; pop r14; pop r13; pop r12; pop rbx
    ret
