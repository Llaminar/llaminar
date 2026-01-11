#include <gtest/gtest.h>
#include "kernels/cpu/jit/JitMicrokernelBase.h"
#include <vector>
#include <random>

using namespace llaminar::v2::kernels::jit;
using namespace Xbyak;

class WoProjectionTestGenerator : public JitMicrokernelBase
{
public:
    WoProjectionTestGenerator(int d_model) : JitMicrokernelBase(4096)
    {
        // Function signature: void func(float* Wo, float* output, float* context)
        // System V AMD64: rdi=Wo, rsi=output, rdx=context

        Reg64 reg_Wo = rdi;
        Reg64 reg_out_ptr = rsi;
        Reg64 reg_ctx_ptr = rdx;

        emit_wo_projection_fp32(reg_Wo, reg_out_ptr, reg_ctx_ptr, d_model);

        ret();
    }

    // Copied and adapted from JitFusedAttentionWo.h
    void emit_wo_projection_fp32(
        const Xbyak::Reg64 &reg_Wo,
        const Xbyak::Reg64 &reg_out_ptr,
        const Xbyak::Reg64 &reg_ctx_base, // Changed from offset to register
        int d_model)
    {
        using namespace Xbyak;

        // Use local labels for loops to avoid conflicts when called multiple times
        inLocalLabel();

        // Use runtime loop for output rows
        // Save scratch GPRs we'll use
        Reg64 reg_out_idx = r8; // Output row index
        Reg64 reg_j = r9;       // Inner loop index
        Reg64 reg_wo_row = r10; // Wo row pointer

        xor_(reg_out_idx, reg_out_idx);

        L(".outer_loop");
        cmp(reg_out_idx, d_model);
        jge(".outer_end", T_NEAR);

        // Zero accumulator
        Zmm zmm_acc = zmm_scratch0();
        vxorps(zmm_acc, zmm_acc, zmm_acc);

        // Calculate Wo row pointer: Wo + out_idx * d_model * 4
        mov(reg_wo_row, reg_out_idx);
        imul(reg_wo_row, reg_wo_row, d_model * 4);
        add(reg_wo_row, reg_Wo);

        // Inner loop: dot product context * Wo_row
        xor_(reg_j, reg_j);

        L(".inner_loop");
        cmp(reg_j, d_model);
        jge(".inner_end", T_NEAR);

        // Load context chunk from pointer
        Zmm zmm_ctx = zmm_scratch1();
        // Original: lea(rdi, ptr[rsp + context_buffer_offset]); vmovups(zmm_ctx, ptr[rdi + reg_j * 4]);
        // Adapted:
        vmovups(zmm_ctx, ptr[reg_ctx_base + reg_j * 4]);

        // Load Wo row chunk
        Zmm zmm_wo = zmm_scratch2();
        vmovups(zmm_wo, ptr[reg_wo_row + reg_j * 4]);

        // FMA: acc += ctx * wo
        vfmadd231ps(zmm_acc, zmm_ctx, zmm_wo);

        add(reg_j, 16);
        jmp(".inner_loop", T_NEAR);

        L(".inner_end");

        // Horizontal sum to get final output value
        emit_horizontal_sum_to_scalar(zmm_acc);

        // Store result
        // Use r11 as scratch for address calculation to avoid clobbering rdi/rsi/rdx
        Reg64 reg_tmp = r11;
        lea(reg_tmp, ptr[reg_out_ptr + reg_out_idx * 4]);
        vmovss(ptr[reg_tmp], Xmm(zmm_acc.getIdx()));

        inc(reg_out_idx);
        jmp(".outer_loop", T_NEAR);

        L(".outer_end");

        outLocalLabel();
    }

    void emit_horizontal_sum_to_scalar(const Xbyak::Zmm &zmm)
    {
        using namespace Xbyak;

        Zmm zmm_tmp = Zmm(25);
        Ymm ymm_tmp = Ymm(zmm_tmp.getIdx());
        Xmm xmm_tmp = Xmm(zmm_tmp.getIdx());

        // zmm -> ymm: add upper 256 to lower
        vextractf32x8(ymm_tmp, zmm, 1);
        vaddps(Ymm(zmm.getIdx()), Ymm(zmm.getIdx()), ymm_tmp);

        // ymm -> xmm: add upper 128 to lower
        vextractf32x4(xmm_tmp, Zmm(zmm.getIdx()), 1);
        vaddps(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);

        // xmm -> scalar: horizontal sum
        vshufps(xmm_tmp, Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), 0x4E); // Swap high/low 64 bits
        vaddps(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);
        vshufps(xmm_tmp, Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), 0xB1); // Swap high/low 32 bits
        vaddss(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);
    }
};

TEST(Test__JitWoProjectionIsolated, FP32_Correctness)
{
    int d_model = 64; // Small size for testing
    WoProjectionTestGenerator gen(d_model);
    auto kernel = gen.getCode<void (*)(float *, float *, float *)>();

    std::vector<float> context(d_model);
    std::vector<float> Wo(d_model * d_model);
    std::vector<float> output(d_model);

    // Initialize with simple values
    for (int i = 0; i < d_model; ++i)
        context[i] = 1.0f;
    for (int i = 0; i < d_model * d_model; ++i)
        Wo[i] = 1.0f;

    // Expected output: each element should be dot(1s, 1s) = d_model = 64.0
    kernel(Wo.data(), output.data(), context.data());

    for (int i = 0; i < d_model; ++i)
    {
        EXPECT_NEAR(output[i], (float)d_model, 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(Test__JitWoProjectionIsolated, FP32_Random)
{
    int d_model = 128;
    WoProjectionTestGenerator gen(d_model);
    auto kernel = gen.getCode<void (*)(float *, float *, float *)>();

    std::vector<float> context(d_model);
    std::vector<float> Wo(d_model * d_model);
    std::vector<float> output(d_model);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &x : context)
        x = dist(rng);
    for (auto &x : Wo)
        x = dist(rng);

    kernel(Wo.data(), output.data(), context.data());

    // Verify
    for (int r = 0; r < d_model; ++r)
    {
        float expected = 0.0f;
        for (int c = 0; c < d_model; ++c)
        {
            expected += context[c] * Wo[r * d_model + c];
        }
        EXPECT_NEAR(output[r], expected, 1e-3f) << "Mismatch at row " << r;
    }
}
