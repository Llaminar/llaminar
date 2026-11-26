/**
 * @file Test__RetainedCPUMegakernelReplay.cpp
 * @brief Device-free replay and contract regressions for QFN CPU megakernels and kernels.
 *
 * Covers every new/modified CPU execution path across Scalar, AVX2, and AVX-512
 * runtime ISAs (`LLAMINAR_ISA_LEVEL`), all 18+ NativeQuantised codebooks,
 * arbitrary topology participant counts, and multi-thread scaling:
 *   - FloatingPointGemmKernel 4x4 AVX-512 prefill, M=1 one-row, and FP16-RNE helper
 *   - RMSNormPrimitives::rmsnorm_fused_vectorized_row and CPUQwen4Exp non-temporal residualRead
 *   - CPUMoEKernel::sharedExpertGateAddFromTensors and CPUQ8ResidualCorrection M=1 staging
 *   - StandaloneMoEOneRowInTeam across all codebooks, intermediate={256,320,640}, mixed policies
 *   - GroupedMoE execute_moe_grouped_fused_ffn across all codebooks and shrinking replay
 *   - NativeCPUCollectiveProgram same-node AUTO -> UPI topology flexibility
 *   - CPUNativeQuantisedTileConfig floor-K-tiles across physical-core counts
 *   - RetainedCPUMegakernel policy and Q16_1 serial quantization inside OMP_SINGLE (ON6 deadlock fix)
 */

#include <gtest/gtest.h>
#include <omp.h>

#include "collective/NativeCPUCollectiveProgram.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemv.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedTileConfig.h"
#include "kernels/cpu/gemm/CPUProjectionQ8Publication.h"
#include "kernels/cpu/gemm/CPUQ8ResidualCorrection.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "kernels/cpu/primitives/SwiGLUPrimitives.h"
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include "models/qwen4exp/Qwen4ExpRetainedGDNPreAttentionPolicy.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/TensorClasses.h"
#include "utils/NativeQuantisedTestPartialStorage.h"
#include "utils/OpenMPUtils.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

namespace llaminar2::test
{
namespace
{
    std::vector<float> makeDeterministicFloats(size_t n, int seed = 0)
    {
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i)
        {
            const int v = static_cast<int>((i * 37u + static_cast<size_t>(seed) * 13u) % 97u) - 48;
            out[i] = static_cast<float>(v) * 0.03125f;
        }
        return out;
    }

    std::unique_ptr<FP32Tensor> makeFP32Tensor(size_t rows, size_t cols, const std::vector<float> &vals)
    {
        auto t = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        std::copy_n(vals.data(), rows * cols, t->mutable_data());
        return t;
    }

    uint16_t floatToBF16(float f) noexcept
    {
        uint32_t u = 0;
        std::memcpy(&u, &f, sizeof(u));
        return static_cast<uint16_t>(u >> 16);
    }

    float bf16ToFloat(uint16_t h) noexcept
    {
        uint32_t u = static_cast<uint32_t>(h) << 16;
        float f = 0.0f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }

    void runGemvOracle(
        const cpu::native_quantised::CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *in_q8,
        float *out,
        std::span<float> partial_storage,
        int workers)
    {
        if (activeISALevel() != ISALevel::Scalar)
        {
            cpu::native_quantised::gemv_native_quantised_preq(packed, in_q8, out, partial_storage);
            return;
        }
        const auto &cache = CPUExecutionGeometry::local().cache;
        const int k_tiles = std::max(
            1,
            cpu::native_quantised::serialTileConfigForPreparedProjection(
                packed.preparedFootprint(), packed.codebook_id, packed.numerical_policy,
                packed.N, packed.K, std::max(1, workers), cache).k_tiles);
        const int k_blocks = packed.blocks_per_row;
        const int bpt = (k_blocks + k_tiles - 1) / k_tiles;
        const int n_chunks = (packed.N + 63) / 64;
        for (int chunk = 0; chunk < n_chunks; ++chunk)
        {
            const int cols = std::min(64, packed.N - chunk * 64);
            float *dst = out + chunk * 64;
            alignas(64) float part[64];
            for (int p = 0; p < k_tiles; ++p)
            {
                const int kb_start = p * bpt;
                if (kb_start >= k_blocks)
                    break;
                const int kb_end = std::min(kb_start + bpt, k_blocks);
                cpu::native_quantised::gemv_native_quantised_scalar_chunk(
                    packed, in_q8, (p == 0) ? dst : part, chunk, kb_start, kb_end, cols);
                if (p > 0)
                {
                    for (int c = 0; c < cols; ++c)
                        dst[c] = device_fp32_contract::add(dst[c], part[c]);
                }
            }
        }
    }

    /** @brief Exact serial Q16_1 block quantization mirroring RetainedCPUQSAAttentionStage (ON6). */
    void quantizeFP32ToQ16Serial(const float *src_data, Q16_1Block *blocks, size_t total_elements) noexcept
    {
        constexpr size_t BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
        const size_t n_blocks = (total_elements + BLOCK_SIZE - 1u) / BLOCK_SIZE;
        for (size_t block_idx = 0; block_idx < n_blocks; ++block_idx)
        {
            const size_t offset = block_idx * BLOCK_SIZE;
            const size_t count = std::min(BLOCK_SIZE, total_elements - offset);
            Q16_1Block &block = blocks[block_idx];
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
                max_abs = std::max(max_abs, std::abs(src_data[offset + i]));
            const float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            block.d = d;
            int64_t sum_i64 = 0;
            if (d > 1e-10f)
            {
                const float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    const float scaled = src_data[offset + i] * inv_d;
                    const float clamped = std::max(-32767.0f, std::min(32767.0f, scaled));
                    const int16_t quantized = static_cast<int16_t>(std::round(clamped));
                    block.qs[i] = quantized;
                    sum_i64 += static_cast<int64_t>(quantized);
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                    block.qs[i] = 0;
            }
            for (size_t i = count; i < BLOCK_SIZE; ++i)
                block.qs[i] = 0;
            block.sum_qs = static_cast<int32_t>(sum_i64);
        }
    }
} // namespace

/** @test FP16/BF16/FP32 4x4 AVX-512 prefill, M=1 one-row, and HalfRNE match scalar tree byte-for-byte. */
TEST(RetainedCPUMegakernelReplay, FP16GPUAligned4x4AndOneRowMatchScalarOracle)
{
    using namespace llaminar2::gemm;
    const ISALevel isa = activeISALevel();

    for (const int k : {32, 64, 256, 512})
    {
        const auto a_fp32 = makeDeterministicFloats(static_cast<size_t>(19 * k), k + 1);
        std::vector<float> a_half_rne(static_cast<size_t>(k));
        round_fp32_to_fp16_rne_as_fp32_range(a_fp32.data(), a_half_rne.data(), 0, k, isa);
        for (int i = 0; i < k; ++i)
        {
            const float ref = roundFloatingOperandHalfRNE(a_fp32[static_cast<size_t>(i)]);
            EXPECT_EQ(std::memcmp(&a_half_rne[static_cast<size_t>(i)], &ref, sizeof(float)), 0)
                << "k=" << k << " i=" << i;
        }

        const int n = 17;
        const auto w_fp32 = makeDeterministicFloats(static_cast<size_t>(n * k), k + 7);
        std::vector<uint16_t> w_fp16(static_cast<size_t>(n * k));
        std::vector<uint16_t> w_bf16(static_cast<size_t>(n * k));
        for (size_t i = 0; i < w_fp32.size(); ++i)
        {
            w_fp16[i] = fp32_to_fp16(w_fp32[i]);
            w_bf16[i] = floatToBF16(w_fp32[i]);
        }

        // Verify M=1 dot_gpu_aligned_expert_one_row across FP16, BF16, FP32
        for (int col = 0; col < n; ++col)
        {
            const size_t w_off = static_cast<size_t>(col) * static_cast<size_t>(k);
            // FP16 with pre-rounded HalfRNE activation
            const float got16 = dot_gpu_aligned_expert_one_row<TensorType::FP16>(
                a_half_rne.data(), w_fp16.data(), w_off, k, isa);
            std::array<float, 1> ref16{};
            floating_expert_numerical_contract::dotRows<1>(
                1, k,
                [&](int, int idx) { return roundFloatingOperandHalfRNE(a_fp32[static_cast<size_t>(idx)]); },
                [&](int idx) { return fp16_to_fp32(w_fp16[w_off + static_cast<size_t>(idx)]); },
                ref16);
            EXPECT_EQ(std::memcmp(&got16, &ref16[0], sizeof(float)), 0) << "FP16 k=" << k << " col=" << col;

            // BF16
            const float gotbf = dot_gpu_aligned_expert_one_row<TensorType::BF16>(
                a_fp32.data(), w_bf16.data(), w_off, k, isa);
            std::array<float, 1> refbf{};
            floating_expert_numerical_contract::dotRows<1>(
                1, k,
                [&](int, int idx) { return a_fp32[static_cast<size_t>(idx)]; },
                [&](int idx) { return bf16ToFloat(w_bf16[w_off + static_cast<size_t>(idx)]); },
                refbf);
            EXPECT_EQ(std::memcmp(&gotbf, &refbf[0], sizeof(float)), 0) << "BF16 k=" << k << " col=" << col;

            // FP32
            const float got32 = dot_gpu_aligned_expert_one_row<TensorType::FP32>(
                a_fp32.data(), w_fp32.data(), w_off, k, isa);
            std::array<float, 1> ref32{};
            floating_expert_numerical_contract::dotRows<1>(
                1, k,
                [&](int, int idx) { return a_fp32[static_cast<size_t>(idx)]; },
                [&](int idx) { return w_fp32[w_off + static_cast<size_t>(idx)]; },
                ref32);
            EXPECT_EQ(std::memcmp(&got32, &ref32[0], sizeof(float)), 0) << "FP32 k=" << k << " col=" << col;
        }

#if defined(__AVX512F__) && defined(__AVX512BW__)
        if (isa >= ISALevel::AVX512)
        {
            float out4x4[4][4]{};
            dot_gpu_aligned_expert_4x4_avx512<TensorType::FP16, true, false>(
                a_fp32.data(), w_fp16.data(), static_cast<size_t>(4) * static_cast<size_t>(k), k, out4x4);
            for (int c = 0; c < 4; ++c)
            {
                std::array<float, 4> ref_col{};
                const size_t w_off = static_cast<size_t>(4 + c) * static_cast<size_t>(k);
                floating_expert_numerical_contract::dotRows<4>(
                    4, k,
                    [&](int r, int idx) {
                        return roundFloatingOperandHalfRNE(
                            a_fp32[static_cast<size_t>(r) * static_cast<size_t>(k) + static_cast<size_t>(idx)]);
                    },
                    [&](int idx) { return fp16_to_fp32(w_fp16[w_off + static_cast<size_t>(idx)]); },
                    ref_col);
                for (int r = 0; r < 4; ++r)
                {
                    EXPECT_EQ(std::memcmp(&out4x4[r][c], &ref_col[static_cast<size_t>(r)], sizeof(float)), 0)
                        << "4x4 FP16 k=" << k << " r=" << r << " c=" << c;
                }
            }
        }
#endif
    }
}

/** @test RMSNorm single-row primitive and aligned non-temporal residualRead match scalar oracle byte-for-byte. */
TEST(RetainedCPUMegakernelReplay, RMSNormFusedRowAndNonTemporalResidualReadMatchScalar)
{
    auto qfn = qwen4exp::createCPUQwen4ExpKernels();
    for (const int threads : {1, 4})
    {
        llaminar::v2::ThreadCountGuard team(threads);
        for (const int hidden : {19, 64, 256, 2560})
        {
            const auto src = makeDeterministicFloats(static_cast<size_t>(hidden), hidden + 3);
            const auto gamma = makeDeterministicFloats(static_cast<size_t>(hidden), hidden + 11);
            std::vector<float> full(static_cast<size_t>(hidden), -77.0f);
            std::vector<float> row_out(static_cast<size_t>(hidden), -88.0f);
            primitives::rmsnorm_fused_vectorized(
                src.data(), gamma.data(), full.data(), 1u, static_cast<size_t>(hidden), 1e-6f);
            primitives::rmsnorm_fused_vectorized_row(
                src.data(), gamma.data(), row_out.data(), static_cast<size_t>(hidden), 1e-6f);
            EXPECT_EQ(std::memcmp(full.data(), row_out.data(), full.size() * sizeof(float)), 0)
                << "threads=" << threads << " hidden=" << hidden;

            // Exercise both 64B-aligned (non-temporal M=1 path) and unaligned destinations for residualRead
            const int channels = hidden * 4;
            const auto branches = makeDeterministicFloats(static_cast<size_t>(5 * channels), hidden + 5);
            const auto logits = makeDeterministicFloats(static_cast<size_t>(5 * channels), hidden + 9);
            std::vector<float> aligned_out(static_cast<size_t>(hidden + 32), -99.0f);
            std::vector<float> grouped_out(static_cast<size_t>(5 * hidden), -99.0f);
            qfn->residualRead({{5}, hidden, branches.data(), logits.data(), grouped_out.data()}, nullptr);
            // Offset aligned_out to an exact 64B boundary and also test an unaligned +1 float offset.
            float *aligned_ptr = aligned_out.data();
            while ((reinterpret_cast<uintptr_t>(aligned_ptr) & 63u) != 0u)
                ++aligned_ptr;
            qfn->residualRead({{1}, hidden, branches.data(), logits.data(), aligned_ptr}, nullptr);
            EXPECT_EQ(std::memcmp(aligned_ptr, grouped_out.data(), static_cast<size_t>(hidden) * sizeof(float)), 0)
                << "aligned M=1 residualRead threads=" << threads << " hidden=" << hidden;
            qfn->residualRead({{1}, hidden, branches.data(), logits.data(), aligned_ptr + 1}, nullptr);
            EXPECT_EQ(std::memcmp(aligned_ptr + 1, grouped_out.data(), static_cast<size_t>(hidden) * sizeof(float)), 0)
                << "unaligned M=1 residualRead threads=" << threads << " hidden=" << hidden;
        }
    }
}

/** @test CPUMoEKernel::sharedExpertGateAddFromTensors matches scalar sigmoid-gate + residual add byte-for-byte. */
TEST(RetainedCPUMegakernelReplay, SharedExpertGateAddFromTensorsMatchesScalarOracle)
{
    CPUMoEKernel moe;
    for (const int threads : {1, 4})
    {
        llaminar::v2::ThreadCountGuard team(threads);
        for (const int seq : {1, 4, 9})
        {
            for (const int d : {32, 256, 512})
            {
                const auto inp_vals = makeDeterministicFloats(static_cast<size_t>(seq * d), d + 1);
                const auto gate_vals = makeDeterministicFloats(static_cast<size_t>(d), d + 2);
                const auto sh_vals = makeDeterministicFloats(static_cast<size_t>(seq * d), d + 3);
                const auto rt_vals = makeDeterministicFloats(static_cast<size_t>(seq * d), d + 4);

                auto t_inp = makeFP32Tensor(static_cast<size_t>(seq), static_cast<size_t>(d), inp_vals);
                auto t_gate = makeFP32Tensor(1u, static_cast<size_t>(d), gate_vals);
                auto t_sh = makeFP32Tensor(static_cast<size_t>(seq), static_cast<size_t>(d), sh_vals);
                auto t_rt = makeFP32Tensor(static_cast<size_t>(seq), static_cast<size_t>(d), rt_vals);
                auto t_out = makeFP32Tensor(
                    static_cast<size_t>(seq), static_cast<size_t>(d),
                    std::vector<float>(static_cast<size_t>(seq * d), -77.0f));

                moe.sharedExpertGateAddFromTensors(t_inp.get(), t_gate.get(), t_sh.get(), t_rt.get(), t_out.get(), seq, d);

                // Compare against sequential sharedExpertGate + residual add oracle
                auto ref_sh = makeFP32Tensor(static_cast<size_t>(seq), static_cast<size_t>(d), sh_vals);
                moe.sharedExpertGate(t_inp->data(), t_gate->data(), ref_sh->mutable_data(), seq, d);
                for (int i = 0; i < seq * d; ++i)
                {
                    const float expected = ref_sh->data()[i] + rt_vals[static_cast<size_t>(i)];
                    EXPECT_NEAR(t_out->data()[i], expected, 1e-6f)
                        << "threads=" << threads << " seq=" << seq << " d=" << d << " i=" << i;
                }
            }
        }
    }
}

/** @test StandaloneMoEOneRowInTeam covers all 18+ codebooks, intermediate={256,320,640}, and mixed policies. */
TEST(RetainedCPUMegakernelReplay, StandaloneMoEOneRowInTeamAllCodebooksAndGeometriesReplay)
{
    using Kernel = cpu::native_quantised::CPUNativeQuantisedGemmKernel;
    using Desc = Kernel::BatchedPrequantizedProjectionDesc;
    using Plan = Kernel::StandaloneMoEOneRowPlan;

    const auto &formats = quantizedMoEVerifierFormats();
    ASSERT_GE(formats.size(), 18u);

    constexpr int hidden = 256; // Divisible by both 64 and 256 superblock width
    const int hidden_blocks = hidden / Q8_1Block::BLOCK_SIZE;
    const auto base_input = makeDeterministicFloats(static_cast<size_t>(hidden), 41);

    for (const int intermediate : {256, 320, 640})
    {
        const int inter_blocks = intermediate / Q8_1Block::BLOCK_SIZE;
        // Cycle through every format (for superblock formats, skip non-256-divisible Down K).
        for (size_t fi = 0; fi < formats.size(); ++fi)
        {
            const auto &fmt = formats[fi];
            if (fmt.source_is_superblock && (intermediate % 256) != 0)
                continue;
            SCOPED_TRACE(::testing::Message() << "format=" << fmt.label << " inter=" << intermediate);

            const auto policy = (fi % 2 == 0)
                ? CPUProjectionNumericalPolicy::GPUAlignedExpert
                : CPUProjectionNumericalPolicy::BackendNative;

            auto w_gate = fmt.create({static_cast<size_t>(intermediate), static_cast<size_t>(hidden)}, 100u + static_cast<uint32_t>(fi));
            auto w_up = fmt.create({static_cast<size_t>(intermediate), static_cast<size_t>(hidden)}, 200u + static_cast<uint32_t>(fi));
            auto w_down = fmt.create({static_cast<size_t>(hidden), static_cast<size_t>(intermediate)}, 300u + static_cast<uint32_t>(fi));

            const float scale = composedMoEVerifierInputMultiplier(*w_gate, *w_up, 1.5f);
            std::vector<float> input_fp32(static_cast<size_t>(hidden));
            for (int i = 0; i < hidden; ++i)
                input_fp32[static_cast<size_t>(i)] = base_input[static_cast<size_t>(i)] * scale;

            Kernel k_gate(w_gate.get(), 0, -1, policy);
            Kernel k_up(w_up.get(), 0, -1, policy);
            Kernel k_down(w_down.get(), 0, -1, policy);
            ASSERT_TRUE(k_gate.isValid());
            ASSERT_TRUE(k_up.isValid());
            ASSERT_TRUE(k_down.isValid());

            std::vector<Q8_1Block> in_q8(static_cast<size_t>(hidden_blocks));
            cpu::native_quantised::quantize_activations_to_q8_1(
                input_fp32.data(), in_q8.data(), 1, hidden, hidden_blocks, policy);

            std::vector<float> splitk_buf(Kernel::kStandaloneMoESplitKScratchFloats, 0.0f);
            for (const int threads : {1, 4})
            {
                llaminar::v2::ThreadCountGuard team(threads);

                // Ground-truth oracle under the active worker count (exact K-partition contract)
                NativeQuantisedTestPartialStorage p_gate(k_gate.packedWeights(), 1);
                NativeQuantisedTestPartialStorage p_up(k_up.packedWeights(), 1);
                NativeQuantisedTestPartialStorage p_down(k_down.packedWeights(), 1);
                std::vector<float> ref_gate(static_cast<size_t>(intermediate));
                std::vector<float> ref_up(static_cast<size_t>(intermediate));
                std::vector<float> ref_act(static_cast<size_t>(intermediate));
                std::vector<Q8_1Block> ref_act_q8(static_cast<size_t>(inter_blocks));
                std::vector<float> ref_down(static_cast<size_t>(hidden));
                runGemvOracle(k_gate.packedWeights(), in_q8.data(), ref_gate.data(), p_gate.span(), threads);
                runGemvOracle(k_up.packedWeights(), in_q8.data(), ref_up.data(), p_up.span(), threads);
                if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                    primitives::compute_swiglu_gpu_aligned_expert_serial(ref_gate.data(), ref_up.data(), ref_act.data(), intermediate);
                else
                    primitives::compute_swiglu_serial(ref_gate.data(), ref_up.data(), ref_act.data(), intermediate);
                cpu::native_quantised::quantize_activations_to_q8_1(
                    ref_act.data(), ref_act_q8.data(), 1, intermediate, inter_blocks, policy);
                runGemvOracle(k_down.packedWeights(), ref_act_q8.data(), ref_down.data(), p_down.span(), threads);

                for (const size_t routes : {size_t{1}, size_t{3}})
                {
                    ASSERT_TRUE(Kernel::supportsStandaloneMoEOneRowInTeam(hidden, intermediate, routes));
                    std::vector<float> gate_out(routes * static_cast<size_t>(intermediate), -71.0f);
                    std::vector<float> up_out(routes * static_cast<size_t>(intermediate), -72.0f);
                    std::vector<float> down_out(routes * static_cast<size_t>(hidden), -73.0f);
                    std::vector<Q8_1Block> act_q8(routes * static_cast<size_t>(inter_blocks));

                    std::vector<Desc> gate_descs(2 * routes);
                    std::vector<Desc> down_descs(routes);
                    for (size_t r = 0; r < routes; ++r)
                    {
                        gate_descs[2 * r] = {.kernel = &k_gate, .input_q8 = in_q8.data(),
                            .output = gate_out.data() + r * static_cast<size_t>(intermediate),
                            .rows = 1, .n = intermediate, .ldc = intermediate};
                        gate_descs[2 * r + 1] = {.kernel = &k_up, .input_q8 = in_q8.data(),
                            .output = up_out.data() + r * static_cast<size_t>(intermediate),
                            .rows = 1, .n = intermediate, .ldc = intermediate};
                        down_descs[r] = {.kernel = &k_down, .input_q8 = act_q8.data() + r * static_cast<size_t>(inter_blocks),
                            .output = down_out.data() + r * static_cast<size_t>(hidden),
                            .rows = 1, .n = hidden, .ldc = hidden};
                    }

                    // Test both prepareStandaloneMoEOneRowPlan AND cached prepareStandaloneMoEOneRowEntry + finalize
                    Plan plan{};
                    ASSERT_TRUE(Kernel::prepareStandaloneMoEOneRowPlan(
                        plan, gate_descs, down_descs, hidden, intermediate,
                        threads, activeISALevel(), splitk_buf.size()));
                    std::atomic<bool> ok{true};
                    auto run = [&]() {
                        if (!Kernel::executePreparedStandaloneMoEOneRowInTeam(
                                plan, act_q8, splitk_buf.data(), splitk_buf.size()))
                            ok.store(false, std::memory_order_relaxed);
                    };
                    OMP_LAYER_PARALLEL(run(););
                    ASSERT_TRUE(ok.load());
                    for (size_t r = 0; r < routes; ++r)
                    {
                        EXPECT_EQ(std::memcmp(down_out.data() + r * static_cast<size_t>(hidden),
                                              ref_down.data(), static_cast<size_t>(hidden) * sizeof(float)), 0);
                    }

                    // Cached-entry assembly path (used by RetainedCPUFFNStage!)
                    Plan::RowEntry e_gate{}, e_up{}, e_down{};
                    ASSERT_TRUE(Kernel::prepareStandaloneMoEOneRowEntry(e_gate, &k_gate, intermediate, hidden, threads, activeISALevel()));
                    ASSERT_TRUE(Kernel::prepareStandaloneMoEOneRowEntry(e_up, &k_up, intermediate, hidden, threads, activeISALevel()));
                    ASSERT_TRUE(Kernel::prepareStandaloneMoEOneRowEntry(e_down, &k_down, hidden, intermediate, threads, activeISALevel()));
                    Plan cached_plan{};
                    for (size_t r = 0; r < routes; ++r)
                    {
                        cached_plan.gate_plans[2 * r] = e_gate;
                        cached_plan.gate_plans[2 * r].input = in_q8.data();
                        cached_plan.gate_plans[2 * r].output = gate_out.data() + r * static_cast<size_t>(intermediate);
                        cached_plan.gate_plans[2 * r + 1] = e_up;
                        cached_plan.gate_plans[2 * r + 1].input = in_q8.data();
                        cached_plan.gate_plans[2 * r + 1].output = up_out.data() + r * static_cast<size_t>(intermediate);
                        cached_plan.down_plans[r] = e_down;
                        cached_plan.down_plans[r].input = act_q8.data() + r * static_cast<size_t>(inter_blocks);
                        cached_plan.down_plans[r].output = down_out.data() + r * static_cast<size_t>(hidden);
                    }
                    ASSERT_TRUE(Kernel::finalizeStandaloneMoEOneRowPlan(
                        cached_plan, routes, hidden, intermediate, threads, activeISALevel(), splitk_buf.size()));
                    std::fill(down_out.begin(), down_out.end(), -81.0f);
                    ok.store(true);
                    auto run_cached = [&]() {
                        if (!Kernel::executePreparedStandaloneMoEOneRowInTeam(
                                cached_plan, act_q8, splitk_buf.data(), splitk_buf.size()))
                            ok.store(false, std::memory_order_relaxed);
                    };
                    OMP_LAYER_PARALLEL(run_cached(););
                    ASSERT_TRUE(ok.load());
                    for (size_t r = 0; r < routes; ++r)
                    {
                        EXPECT_EQ(std::memcmp(down_out.data() + r * static_cast<size_t>(hidden),
                                              ref_down.data(), static_cast<size_t>(hidden) * sizeof(float)), 0);
                    }
                }
            }
        }
    }
}

/** @test execute_moe_grouped_fused_ffn matches serial decode-equivalent oracle across all formats and M. */
TEST(RetainedCPUMegakernelReplay, GroupedMoEFusedFFNAllFormatsReplay)
{
    using Kernel = cpu::native_quantised::CPUNativeQuantisedGemmKernel;
    using Desc = Kernel::BatchedPrequantizedProjectionDesc;

    const auto &formats = quantizedMoEVerifierFormats();
    constexpr int hidden = 256;
    constexpr int intermediate = 256;
    const int hidden_blocks = hidden / Q8_1Block::BLOCK_SIZE;
    const int inter_blocks = intermediate / Q8_1Block::BLOCK_SIZE;
    const auto policy = CPUProjectionNumericalPolicy::GPUAlignedExpert;

    constexpr int unique_rows = 5;
    const auto base_hidden = makeDeterministicFloats(static_cast<size_t>(unique_rows * hidden), 77);

    for (size_t fi = 0; fi < formats.size(); ++fi)
    {
        const auto &fmt = formats[fi];
        SCOPED_TRACE(::testing::Message() << "grouped_ffn format=" << fmt.label);
        auto w_gate = fmt.create({static_cast<size_t>(intermediate), static_cast<size_t>(hidden)}, 500u + static_cast<uint32_t>(fi));
        auto w_up = fmt.create({static_cast<size_t>(intermediate), static_cast<size_t>(hidden)}, 600u + static_cast<uint32_t>(fi));
        auto w_down = fmt.create({static_cast<size_t>(hidden), static_cast<size_t>(intermediate)}, 700u + static_cast<uint32_t>(fi));

        const float scale = composedMoEVerifierInputMultiplier(*w_gate, *w_up, 1.5f);
        std::vector<float> hidden_fp32(static_cast<size_t>(unique_rows * hidden));
        for (size_t i = 0; i < hidden_fp32.size(); ++i)
            hidden_fp32[i] = base_hidden[i] * scale;

        std::vector<Q8_1Block> unique_q8(static_cast<size_t>(unique_rows * hidden_blocks));
        cpu::native_quantised::quantize_activations_to_q8_1(
            hidden_fp32.data(), unique_q8.data(), unique_rows, hidden, hidden_blocks, policy);

        Kernel k_gate(w_gate.get(), 0, -1, policy);
        Kernel k_up(w_up.get(), 0, -1, policy);
        Kernel k_down(w_down.get(), 0, -1, policy);
        ASSERT_TRUE(k_gate.isValid());
        ASSERT_TRUE(k_up.isValid());
        ASSERT_TRUE(k_down.isValid());

        // Serial per-row oracle for each unique row
        NativeQuantisedTestPartialStorage p_g(k_gate.packedWeights(), 1);
        NativeQuantisedTestPartialStorage p_u(k_up.packedWeights(), 1);
        NativeQuantisedTestPartialStorage p_d(k_down.packedWeights(), 1);
        std::vector<float> ref_down(static_cast<size_t>(unique_rows * hidden));
        std::vector<float> tmp_g(intermediate), tmp_u(intermediate), tmp_act(intermediate);
        std::vector<Q8_1Block> tmp_q8(static_cast<size_t>(inter_blocks));
        for (int r = 0; r < unique_rows; ++r)
        {
            const Q8_1Block *rq = unique_q8.data() + static_cast<size_t>(r * hidden_blocks);
            runGemvOracle(k_gate.packedWeights(), rq, tmp_g.data(), p_g.span(), 1);
            runGemvOracle(k_up.packedWeights(), rq, tmp_u.data(), p_u.span(), 1);
            primitives::compute_swiglu_gpu_aligned_expert_serial(tmp_g.data(), tmp_u.data(), tmp_act.data(), intermediate);
            cpu::native_quantised::quantize_activations_to_q8_1(tmp_act.data(), tmp_q8.data(), 1, intermediate, inter_blocks, policy);
            runGemvOracle(k_down.packedWeights(), tmp_q8.data(), ref_down.data() + static_cast<size_t>(r * hidden), p_d.span(), 1);
        }

        // Test shrinking replay (5 route rows -> 1 route row) across 1 and 4 threads
        for (const int threads : {1, 4})
        {
            llaminar::v2::ThreadCountGuard team(threads);
            for (const int route_rows : {5, 1})
            {
                std::vector<int> row_map(static_cast<size_t>(route_rows));
                for (int i = 0; i < route_rows; ++i)
                    row_map[static_cast<size_t>(i)] = (i * 2) % unique_rows;
                std::vector<float> gate_buf(static_cast<size_t>(route_rows * intermediate));
                std::vector<float> up_buf(static_cast<size_t>(route_rows * intermediate));
                std::vector<Q8_1Block> act_q8(static_cast<size_t>(route_rows * inter_blocks));
                std::vector<float> down_out(static_cast<size_t>(route_rows * hidden), -91.0f);
                std::array<Desc, 2> gu_descs{{
                    {.kernel = &k_gate, .input_q8 = unique_q8.data(), .output = gate_buf.data(),
                     .rows = route_rows, .n = intermediate, .ldc = intermediate,
                     .purpose = cpu::native_quantised::ProjectionRowsPurpose::Prefill},
                    {.kernel = &k_up, .input_q8 = unique_q8.data(), .output = up_buf.data(),
                     .rows = route_rows, .n = intermediate, .ldc = intermediate,
                     .purpose = cpu::native_quantised::ProjectionRowsPurpose::Prefill},
                }};
                std::array<Desc, 1> d_descs{{
                    {.kernel = &k_down, .input_q8 = act_q8.data(), .output = down_out.data(),
                     .rows = route_rows, .n = hidden, .ldc = hidden,
                     .purpose = cpu::native_quantised::ProjectionRowsPurpose::Prefill},
                }};
                ASSERT_TRUE(Kernel::execute_moe_grouped_fused_ffn(
                    gu_descs, unique_q8.data(), unique_rows, hidden_blocks, row_map,
                    hidden, act_q8.data(), route_rows, intermediate, inter_blocks, d_descs));
                for (int i = 0; i < route_rows; ++i)
                {
                    const int src_row = row_map[static_cast<size_t>(i)];
                    EXPECT_EQ(std::memcmp(down_out.data() + static_cast<size_t>(i * hidden),
                                          ref_down.data() + static_cast<size_t>(src_row * hidden),
                                          static_cast<size_t>(hidden) * sizeof(float)), 0);
                }
            }
        }
    }
}

/** @test NativeCPUCollectiveProgram AUTO selects UPI for arbitrary same-node counts and MPI cross-node. */
TEST(RetainedCPUMegakernelReplay, NativeCPUCollectiveProgramAutoSelectsUPIOnSameNode)
{
    for (const int degree : {2, 3, 4, 5, 8})
    {
        std::vector<NativeCPUCollectiveProgram::Participant> same_node;
        for (int r = 0; r < degree; ++r)
        {
            same_node.push_back({
                .world_rank = r,
                .address = {.hostname = "node0", .numa_node = r,
                            .device_type = DeviceType::CPU, .device_ordinal = 0},
            });
        }
        const auto prog = NativeCPUCollectiveProgram::forBoundExpertShardDomain(
            10 + degree, CollectiveBackendType::AUTO, same_node);
        EXPECT_EQ(prog.backend(), CollectiveBackendType::UPI) << "degree=" << degree;
        EXPECT_EQ(prog.participants().size(), static_cast<size_t>(degree));
    }

    // Cross-node must not select UPI
    std::vector<NativeCPUCollectiveProgram::Participant> cross_node{
        {.world_rank = 0, .address = {.hostname = "node0", .numa_node = 0, .device_type = DeviceType::CPU, .device_ordinal = 0}},
        {.world_rank = 1, .address = {.hostname = "node1", .numa_node = 0, .device_type = DeviceType::CPU, .device_ordinal = 0}},
    };
    const auto cross_prog = NativeCPUCollectiveProgram::forBoundExpertShardDomain(
        99, CollectiveBackendType::AUTO, cross_node);
    EXPECT_EQ(cross_prog.backend(), CollectiveBackendType::MPI);
}

/** @test CPUNativeQuantisedTileConfig floor-K-tiles stays positive and covers all K-blocks across thread counts. */
TEST(RetainedCPUMegakernelReplay, CPUNativeQuantisedTileConfigFloorKTilesContract)
{
    using namespace cpu::native_quantised;
    const auto &cache = CPUExecutionGeometry::local().cache;
    const NativeQuantisedPreparedFootprint fp{
        .encoding = CPUNativeQuantisedEncoding::NibbleLUT,
        .is_asymmetric = false,
        .weight_bytes_per_n_chunk_k_block = 512u,
        .activation_bytes_per_row_k_block = 36u,
        .output_bytes_per_row_n_chunk = 128u,
    };
    ASSERT_TRUE(fp.isValid());
    for (const int workers : {1, 4, 8, 14, 28})
    {
        for (const int n : {64, 320, 640, 2560, 10240})
        {
            for (const int k : {256, 320, 640, 2560})
            {
                const auto cfg = serialTileConfigForPreparedProjection(
                    fp, /*codebook_id=*/5, CPUProjectionNumericalPolicy::BackendNative,
                    n, k, workers, cache);
                EXPECT_GE(cfg.n_block_chunks, 1);
                EXPECT_GE(cfg.k_tiles, 0);
                EXPECT_LE(cfg.k_tiles, k / 32);
            }
        }
    }
}

/** @test Q8ResidualCorrection M=1 staggered stack copy matches single-thread oracle byte-for-byte. */
TEST(RetainedCPUMegakernelReplay, Q8ResidualCorrectionSingleRowStaggeredMatchesOracle)
{
    using Kernel = cpu::native_quantised::CPUNativeQuantisedGemmKernel;
    constexpr int n = 256, k = 320;
    const int bpr = k / Q8_1Block::BLOCK_SIZE;
    auto w = TestTensorFactory::createQ8_0Random({static_cast<size_t>(n), static_cast<size_t>(k)}, 9123u);
    Kernel gemm(w.get(), 0, -1, CPUProjectionNumericalPolicy::BackendNative);
    ASSERT_TRUE(gemm.isValid());

    const auto x = makeDeterministicFloats(static_cast<size_t>(4 * k), 19);
    std::vector<float> act1(static_cast<size_t>(k)), out_ref(static_cast<size_t>(n));
    std::vector<Q8_1Block> p_ref(static_cast<size_t>(bpr)), r_ref(static_cast<size_t>(bpr));
    {
        llaminar::v2::ThreadCountGuard team(1);
        cpu::native_quantised::cpuHCResidualDownQ8Projection(
            gemm.packedWeights(), x.data(), act1.data(), out_ref.data(),
            DeviceRowRange::fullyActive(1), p_ref, r_ref);
    }
    for (const int threads : {1, 4, 8})
    {
        llaminar::v2::ThreadCountGuard team(threads);
        std::vector<float> act_mt(static_cast<size_t>(k)), out_mt(static_cast<size_t>(n), -99.0f);
        std::vector<Q8_1Block> p_mt(static_cast<size_t>(bpr)), r_mt(static_cast<size_t>(bpr));
        cpu::native_quantised::cpuHCResidualDownQ8Projection(
            gemm.packedWeights(), x.data(), act_mt.data(), out_mt.data(),
            DeviceRowRange::fullyActive(1), p_mt, r_mt);
        EXPECT_EQ(std::memcmp(out_mt.data(), out_ref.data(), out_ref.size() * sizeof(float)), 0)
            << "threads=" << threads;
    }
}

/** @test CPU megakernel policy gates CPU-only and Q16 serial quantize inside OMP_SINGLE never deadlocks (ON6). */
TEST(RetainedCPUMegakernelReplay, PolicySelectionAndQ16SerialInsideOMPSingleNoDeadlock)
{
    EXPECT_TRUE(usesRetainedCPUMegakernels(
        true, DeviceId::cpu(), ForwardExecutionPhase::Decode, ForwardExecutionRole::MainInference, 1, 1));
    EXPECT_FALSE(usesRetainedCPUMegakernels(
        false, DeviceId::cpu(), ForwardExecutionPhase::Decode, ForwardExecutionRole::MainInference, 1, 1));
    EXPECT_FALSE(usesRetainedCPUMegakernels(
        true, DeviceId::cuda(0), ForwardExecutionPhase::Decode, ForwardExecutionRole::MainInference, 1, 1));
    EXPECT_FALSE(usesRetainedCPUMegakernels(
        true, DeviceId::rocm(0), ForwardExecutionPhase::Decode, ForwardExecutionRole::MainInference, 1, 1));
    EXPECT_FALSE(usesRetainedCPUMegakernels(
        true, DeviceId::cpu(), ForwardExecutionPhase::Prefill, ForwardExecutionRole::MainInference, 64, 1));

    // ON6 deadlock regression: serial Q16_1 quantization inside OMP_LAYER_PARALLEL + OMP_SINGLE
    // must complete without entering OMP_WORKSHARE_REGION and match Q16_1Tensor::copyFrom_fp32.
    constexpr int cols = 512;
    const auto src = makeDeterministicFloats(static_cast<size_t>(cols), 55);
    Q16_1Tensor ref_q16({1u, static_cast<size_t>(cols)});
    ASSERT_TRUE(ref_q16.copyFrom_fp32(src.data()));

    for (const int threads : {1, 4, 8})
    {
        llaminar::v2::ThreadCountGuard team(threads);
        Q16_1Tensor single_q16({1u, static_cast<size_t>(cols)});
        auto *dst_blocks = reinterpret_cast<Q16_1Block *>(single_q16.raw_mutable_data());
        auto run = [&]() {
            auto quantize_fn = [&]() {
                quantizeFP32ToQ16Serial(src.data(), dst_blocks, static_cast<size_t>(cols));
            };
            OMP_SINGLE(quantize_fn);
        };
        OMP_LAYER_PARALLEL(run(););
        EXPECT_EQ(std::memcmp(single_q16.raw_data(), ref_q16.raw_data(),
                              static_cast<size_t>(cols / Q16_1Block::BLOCK_SIZE) * sizeof(Q16_1Block)), 0)
            << "Q16 serial parity threads=" << threads;
    }
}

} // namespace llaminar2::test
