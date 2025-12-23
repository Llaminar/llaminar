/**
 * @file Perf__JitMicrokernels.cpp
 * @brief Performance benchmark for individual JIT microkernels
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

#include "kernels/cpu/attention/q8_1/jit/JitMicrokernelBase.h"
#include "kernels/cpu/attention/q8_1/jit/JitQ8DotProduct.h"
#include "kernels/cpu/attention/q8_1/jit/JitOnlineSoftmax.h"
#include "kernels/cpu/attention/q8_1/jit/JitVWeightedAccum.h"
#include "kernels/cpu/attention/q8_1/jit/JitWoProjection.h"
#include "kernels/cpu/attention/q8_1/jit/JitFastExp.h"
#include "tensors/Tensors.h"

using namespace llaminar::v2::kernels::jit;
using namespace llaminar2;

// ============================================================================
// JIT Wrappers for Benchmarking
// ============================================================================

// Wrapper for Q8DotProduct
class JitQ8DotProductRunner : public JitMicrokernelBase
{
public:
    using FuncType = float (*)(const void *q_ptr, const void *k_ptr, float scale);
    FuncType func_;

    JitQ8DotProductRunner(int num_blocks)
    {
        using namespace Xbyak;

        // Args: rdi=q_ptr, rsi=k_ptr, xmm0=scale

        // Initialize constants
        init_constant_registers();

        // Move scale to xmm_scale (xmm10 is safe? let's check clobbers)
        // emit_dot_product clobbers: ymm4-6, xmm6-9, xmm14-15
        // It takes scale_xmm as input.
        Xmm scale_xmm = xmm0;

        JitQ8DotProductEmitter emitter;
        // We treat q_ptr (rdi) as the "stack base" for Q blocks
        emitter.emit_dot_product(*this, xmm0, rsi, rdi, 0, num_blocks, scale_xmm);

        ret();
        ready();
        func_ = getCode<FuncType>();
    }

    void init_constant_registers()
    {
        using namespace Xbyak;
        // Initialize 0x80808080 for unsigned conversion
        mov(eax, 0x80808080);
        vpbroadcastd(zmm_128(), eax);
    }
};

// Wrapper for VWeightedAccum
class JitVWeightedAccumRunner : public JitMicrokernelBase
{
public:
    using FuncType = void (*)(const void *v_ptr, float weight, float *accum_ptr);
    FuncType func_;
    int num_blocks_;

    JitVWeightedAccumRunner(int num_blocks) : num_blocks_(num_blocks)
    {
        using namespace Xbyak;

        // Args: rdi=v_ptr, xmm0=weight, rsi=accum_ptr

        // Broadcast weight to ZMM_WEIGHT
        vbroadcastss(zmm_weight(), xmm0);

        // Zero accumulators (we'll just accumulate into zeroed regs for the test)
        // In real kernel, these hold running sums.
        // For benchmark, we want to measure the accumulation cost.
        // We'll load from accum_ptr, accumulate, and store back?
        // Or just accumulate into registers and store at the end.

        // Let's simulate the loop body: load accum from memory (or assume in regs), accumulate, store.
        // But emit_weighted_accum assumes accumulators are in zmm_accum(0..N) or stack.

        // For simplicity, we'll zero the accumulators first
        for (int i = 0; i < 2; ++i)
        {
            vxorps(zmm_accum(i), zmm_accum(i), zmm_accum(i));
        }

        // We need to handle spilling if num_blocks > 2 (64 elements)
        // emit_weighted_accum takes spill_base_offset.
        // We'll use rsi (accum_ptr) as the spill base?
        // No, spill base is offset from RSP.
        // We should probably just use registers for small blocks and ignore spilling for now
        // or set up a proper stack frame.

        // Let's assume num_blocks <= 2 for basic test, or handle stack.
        // If num_blocks > 2, we need stack space.

        JitVWeightedAccumEmitter emitter;
        // We'll use rdx as a scratch register if needed, but here we just pass rdi as v_ptr

        // We need to trick the spill offset.
        // If we pass a pointer to a buffer in rsi, we can't use it as RSP offset directly.
        // But we can set RSP to rsi? No, dangerous.

        // Let's just support up to 2 blocks (head_dim=64) for now in registers.
        // If head_dim=128 (4 blocks), we need spilling.

        // For the benchmark, let's allocate stack space.
        sub(rsp, 1024); // Plenty of space

        emitter.emit_weighted_accum(*this, rdi, num_blocks, 0);

        add(rsp, 1024);
        ret();
        ready();
        func_ = getCode<FuncType>();
    }
};

// Wrapper for WoProjection
class JitWoProjectionRunner : public JitMicrokernelBase
{
public:
    using FuncType = void (*)(const void *ctx_ptr, const void *wo_ptr, float *out_ptr);
    FuncType func_;

    JitWoProjectionRunner(int head_dim, WoFormat format)
    {
        using namespace Xbyak;

        // Args: rdi=ctx_ptr, rsi=wo_ptr, rdx=out_ptr

        JitWoProjectionEmitter emitter;

        if (format == WoFormat::FP32)
        {
            emitter.emit_project_fp32(*this, rdi, rsi, rdx, head_dim);
        }
        else if (format == WoFormat::Q8_1)
        {
            // Initialize constants for Q8_1
            mov(eax, 0x80808080);
            vpbroadcastd(zmm_128(), eax);

            emitter.emit_project_q8_1(*this, rdi, rsi, rdx, head_dim);
        }

        ret();
        ready();
        func_ = getCode<FuncType>();
    }
};

// ============================================================================
// Test Fixture
// ============================================================================

class Perf__JitMicrokernels : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    void SetUp() override
    {
        rng_.seed(42);
    }

    void generate_q8_1_blocks(std::vector<Q8_1Block> &blocks, size_t num_elements)
    {
        size_t num_blocks = (num_elements + 31) / 32;
        blocks.resize(num_blocks);

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t b = 0; b < num_blocks; ++b)
        {
            blocks[b].d = 1.0f; // Simplified
            blocks[b].sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                blocks[b].qs[i] = (int8_t)(dist(rng_) * 127);
            }
        }
    }

    void generate_fp32_data(std::vector<float> &data, size_t size)
    {
        data.resize(size);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : data)
            v = dist(rng_);
    }
};

// ============================================================================
// Benchmarks
// ============================================================================

TEST_F(Perf__JitMicrokernels, Q8DotProduct)
{
    const int iterations = 1000000;
    const std::vector<int> head_dims = {64, 128};

    std::cout << "\n[ JitQ8DotProduct Performance ]" << std::endl;
    std::cout << std::setw(10) << "Head Dim"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "ns/iter" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    for (int head_dim : head_dims)
    {
        int num_blocks = head_dim / 32;
        std::vector<Q8_1Block> Q, K;
        generate_q8_1_blocks(Q, head_dim);
        generate_q8_1_blocks(K, head_dim);

        JitQ8DotProductRunner runner(num_blocks);

        // Warmup
        for (int i = 0; i < 1000; ++i)
        {
            runner.func_(Q.data(), K.data(), 1.0f);
        }

        auto start = std::chrono::high_resolution_clock::now();

        float sum = 0.0f;
        for (int i = 0; i < iterations; ++i)
        {
            sum += runner.func_(Q.data(), K.data(), 1.0f);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ns_per_iter = (duration_ms * 1e6) / iterations;

        std::cout << std::setw(10) << head_dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << duration_ms
                  << std::setw(15) << ns_per_iter << std::endl;

        // Prevent optimization
        if (sum == 12345.0f)
            std::cout << "dummy" << std::endl;
    }
}

TEST_F(Perf__JitMicrokernels, VWeightedAccum)
{
    const int iterations = 1000000;
    const std::vector<int> head_dims = {64, 128};

    std::cout << "\n[ JitVWeightedAccum Performance ]" << std::endl;
    std::cout << std::setw(10) << "Head Dim"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "ns/iter" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    for (int head_dim : head_dims)
    {
        int num_blocks = head_dim / 32;
        std::vector<Q8_1Block> V;
        generate_q8_1_blocks(V, head_dim);
        std::vector<float> accum(head_dim, 0.0f);

        JitVWeightedAccumRunner runner(num_blocks);

        // Warmup
        for (int i = 0; i < 1000; ++i)
        {
            runner.func_(V.data(), 0.5f, accum.data());
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            runner.func_(V.data(), 0.5f, accum.data());
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ns_per_iter = (duration_ms * 1e6) / iterations;

        std::cout << std::setw(10) << head_dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << duration_ms
                  << std::setw(15) << ns_per_iter << std::endl;
    }
}

TEST_F(Perf__JitMicrokernels, WoProjection_Q8_1)
{
    const int iterations = 1000000;
    const std::vector<int> head_dims = {64, 128};

    std::cout << "\n[ JitWoProjection (Q8_1) Performance ]" << std::endl;
    std::cout << std::setw(10) << "Head Dim"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "ns/iter" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    for (int head_dim : head_dims)
    {
        std::vector<float> context;
        generate_fp32_data(context, head_dim);

        std::vector<Q8_1Block> Wo;
        generate_q8_1_blocks(Wo, head_dim);

        float output = 0.0f;

        JitWoProjectionRunner runner(head_dim, WoFormat::Q8_1);

        // Warmup
        for (int i = 0; i < 1000; ++i)
        {
            runner.func_(context.data(), Wo.data(), &output);
        }

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            runner.func_(context.data(), Wo.data(), &output);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double ns_per_iter = (duration_ms * 1e6) / iterations;

        std::cout << std::setw(10) << head_dim
                  << std::setw(15) << std::fixed << std::setprecision(2) << duration_ms
                  << std::setw(15) << ns_per_iter << std::endl;
    }
}
