# VNNI-16 16x24 tile (vpdpwssd) staging stores. Apack int16 [mb][k2][16 rows x 4B]=64B; Bpack int32 [g][k2][24 cols x 4B]=96B.
.intel_syntax noprefix
.text
.globl vnni16
vnni16:
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
    test r10, r10
    je .Ldone
.Lloop:
    vmovdqu64 zmm24, [r12]
    vpbroadcastd zmm25, DWORD PTR [r13+0]
    vpdpwssd zmm0, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+4]
    vpdpwssd zmm1, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+8]
    vpdpwssd zmm2, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+12]
    vpdpwssd zmm3, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+16]
    vpdpwssd zmm4, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+20]
    vpdpwssd zmm5, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+24]
    vpdpwssd zmm6, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+28]
    vpdpwssd zmm7, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+32]
    vpdpwssd zmm8, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+36]
    vpdpwssd zmm9, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+40]
    vpdpwssd zmm10, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+44]
    vpdpwssd zmm11, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+48]
    vpdpwssd zmm12, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+52]
    vpdpwssd zmm13, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+56]
    vpdpwssd zmm14, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+60]
    vpdpwssd zmm15, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+64]
    vpdpwssd zmm16, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+68]
    vpdpwssd zmm17, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+72]
    vpdpwssd zmm18, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+76]
    vpdpwssd zmm19, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+80]
    vpdpwssd zmm20, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+84]
    vpdpwssd zmm21, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+88]
    vpdpwssd zmm22, zmm24, zmm25
    vpbroadcastd zmm25, DWORD PTR [r13+92]
    vpdpwssd zmm23, zmm24, zmm25
    add r12, 64
    add r13, 96
    dec r10
    jnz .Lloop
.Ldone:
    vmovups [r14+0], zmm0
    vmovups [r14+64], zmm1
    vmovups [r14+128], zmm2
    vmovups [r14+192], zmm3
    vmovups [r14+256], zmm4
    vmovups [r14+320], zmm5
    vmovups [r14+384], zmm6
    vmovups [r14+448], zmm7
    vmovups [r14+512], zmm8
    vmovups [r14+576], zmm9
    vmovups [r14+640], zmm10
    vmovups [r14+704], zmm11
    vmovups [r14+768], zmm12
    vmovups [r14+832], zmm13
    vmovups [r14+896], zmm14
    vmovups [r14+960], zmm15
    vmovups [r14+1024], zmm16
    vmovups [r14+1088], zmm17
    vmovups [r14+1152], zmm18
    vmovups [r14+1216], zmm19
    vmovups [r14+1280], zmm20
    vmovups [r14+1344], zmm21
    vmovups [r14+1408], zmm22
    vmovups [r14+1472], zmm23
    pop r15; pop r14; pop r13; pop r12; pop rbx
    ret
