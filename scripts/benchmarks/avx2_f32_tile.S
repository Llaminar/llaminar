# AVX2 VEX-ONLY 8x8 tile (no regs 16-31): accs ymm0-7, A ymm8, bcast ymm9; contiguous staging stores.
# Apack [mb][k][8 rows x 4B]=32B; Bpack [g][k][8 cols x 4B]=32B; staging out = 8 accs x 32B.
.intel_syntax noprefix
.text
.globl avx2f32
avx2f32:
    push rbx; push r12; push r13; push r14; push r15
    vxorps ymm0,ymm0,ymm0
    vxorps ymm1,ymm1,ymm1
    vxorps ymm2,ymm2,ymm2
    vxorps ymm3,ymm3,ymm3
    vxorps ymm4,ymm4,ymm4
    vxorps ymm5,ymm5,ymm5
    vxorps ymm6,ymm6,ymm6
    vxorps ymm7,ymm7,ymm7
    mov r12, rdi
    mov r13, rsi
    mov r14, rdx
    mov r10, rcx
    test r10, r10
    je .Ldone
.Lloop:
    vmovups ymm8, [r12]
    vbroadcastss ymm9, DWORD PTR [r13+0]
    vfmadd231ps ymm0, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [r13+4]
    vfmadd231ps ymm1, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [r13+8]
    vfmadd231ps ymm2, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [r13+12]
    vfmadd231ps ymm3, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [r13+16]
    vfmadd231ps ymm4, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [r13+20]
    vfmadd231ps ymm5, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [r13+24]
    vfmadd231ps ymm6, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [r13+28]
    vfmadd231ps ymm7, ymm8, ymm9
    add r12, 32
    add r13, 32
    dec r10
    jnz .Lloop
.Ldone:
    vmovups [r14+0], ymm0
    vmovups [r14+32], ymm1
    vmovups [r14+64], ymm2
    vmovups [r14+96], ymm3
    vmovups [r14+128], ymm4
    vmovups [r14+160], ymm5
    vmovups [r14+192], ymm6
    vmovups [r14+224], ymm7
    pop r15; pop r14; pop r13; pop r12; pop rbx
    ret
