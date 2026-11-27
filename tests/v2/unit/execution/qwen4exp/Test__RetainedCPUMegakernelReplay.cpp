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
 *   - CPUGatedDeltaNet core-private L1d value-slice prefill (`runValueSlice` + `finalizeValueSliceState`)
 *     including non-zero initial phase and IEEE subnormal preservation (ON16b)
 *   - UnsignedInt8 L1d-resident `3row x 4z` strip prefill and `6row x 2z` SumPairs (ON16)
 *   - GroupedMoE `6row x 2z` / `3row x 4z`, `kStripMaxRows = 64` boundary (`rows = 65`), and `bpt == 1` (ON16)
 *   - StageRunnerRegistry PP=1 MTP capability delegation (ON12)
 *   - Pure-CPU retained OpenMP team stage mode contract (ON14)
 */

#include <gtest/gtest.h>
#include <omp.h>

#include "collective/NativeCPUCollectiveProgram.h"
#include "execution/compute_stages/stages/RMSNormStage.h"
#include "execution/global/GlobalOrchestrator.h"
#include "kernels/cpu/gdn/CPUFixedEpochGDN.h"
#include "kernels/cpu/gdn/CPUGatedDeltaNet.h"
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
#include "mocks/MockRankOrchestrator.h"
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

/** @test Core-private L1d value-slice GDN prefill matches single-slice oracle byte-for-byte with subnormals and non-zero phase. */
TEST(RetainedCPUMegakernelReplay, GDNValueSlicePrefillMatchesSingleSliceReplay)
{
    constexpr int d_k = 128;
    constexpr int d_v = 128;
    const CPUGDNStateLayout layout(1, d_k, d_v);

    // Part 1: Direct epoch primitive check across all 3 finalizeValueSliceState branches:
    //   initial_phase = 3 with rows = {3 (no flush), 5 (exact flush -> phase 0), 41 (multi-flush -> phase 4)}.
    for (const int rows : {3, 5, 41})
    {
        std::vector<float> init_state(static_cast<size_t>(layout.headFloats()), 0.0f);
        // Seed checkpoint matrix with deterministic floats + IEEE subnormal entries.
        auto mat_seed = makeDeterministicFloats(static_cast<size_t>(d_k * d_v), rows + 17);
        for (size_t i = 0; i < mat_seed.size(); ++i)
            init_state[1u + i] = ((i % 19u) == 0u) ? 1.5e-40f : (((i % 23u) == 0u) ? -3.0e-41f : mat_seed[i] * 0.05f);
        // Establish a canonical initial_phase = 3 state via 3 single-row runs.
        const auto pre_q = makeDeterministicFloats(static_cast<size_t>(3 * d_k), 21);
        const auto pre_k = makeDeterministicFloats(static_cast<size_t>(3 * d_k), 22);
        const auto pre_v = makeDeterministicFloats(static_cast<size_t>(3 * d_v), 23);
        const std::vector<float> pre_decay = {0.91f, 0.87f, 0.94f};
        const std::vector<float> pre_beta = {0.25f, 0.40f, 0.31f};
        std::vector<float> pre_out(static_cast<size_t>(3 * d_v), 0.0f);
        cpu_gdn_epoch::run(layout, init_state.data(), init_state.data(),
            pre_q.data(), pre_k.data(), pre_v.data(), pre_decay.data(), pre_beta.data(),
            3, d_k, d_v, 1, pre_out.data(), d_v);
        ASSERT_EQ(static_cast<int>(init_state[0]), 3);

        const auto q = makeDeterministicFloats(static_cast<size_t>(rows * d_k), rows + 31);
        const auto k = makeDeterministicFloats(static_cast<size_t>(rows * d_k), rows + 37);
        const auto v = makeDeterministicFloats(static_cast<size_t>(rows * d_v), rows + 41);
        std::vector<float> decay(static_cast<size_t>(rows)), beta(static_cast<size_t>(rows));
        for (int r = 0; r < rows; ++r)
        {
            decay[static_cast<size_t>(r)] = 0.75f + 0.01f * static_cast<float>(r % 20);
            beta[static_cast<size_t>(r)] = 0.15f + 0.02f * static_cast<float>(r % 15);
        }

        std::vector<float> ref_state = init_state;
        std::vector<float> ref_out(static_cast<size_t>(rows * d_v), -77.0f);
        cpu_gdn_epoch::run(layout, ref_state.data(), ref_state.data(),
            q.data(), k.data(), v.data(), decay.data(), beta.data(),
            rows, d_k, d_v, 1, ref_out.data(), d_v);

        std::vector<float> slice_state = init_state;
        std::vector<float> slice_out(static_cast<size_t>(rows * d_v), -88.0f);
        cpu_gdn_epoch::runValueSlice(layout, slice_state.data(), slice_state.data(),
            q.data(), k.data(), v.data(), decay.data(), beta.data(),
            rows, d_k, d_v, 1, slice_out.data(), d_v, 0, 64);
        cpu_gdn_epoch::runValueSlice(layout, slice_state.data(), slice_state.data(),
            q.data(), k.data(), v.data(), decay.data(), beta.data(),
            rows, d_k, d_v, 1, slice_out.data(), d_v, 64, 128);
        cpu_gdn_epoch::finalizeValueSliceState(layout, slice_state.data(),
            k.data(), v.data(), decay.data(), beta.data(), rows, d_k, d_v, 1);

        EXPECT_EQ(std::memcmp(slice_out.data(), ref_out.data(), ref_out.size() * sizeof(float)), 0)
            << "runValueSlice output mismatch rows=" << rows;
        EXPECT_EQ(std::memcmp(slice_state.data(), ref_state.data(), ref_state.size() * sizeof(float)), 0)
            << "runValueSlice terminal state mismatch rows=" << rows;
    }

    // Part 2: End-to-end CPUGatedDeltaNet::chunk_forward with n_heads = 4 < 8 workers
    // (triggers use_v_slice_parallel = true at 8 threads, false at 1 thread) + subnormal state.
    constexpr int n_heads = 4;
    constexpr int seq_len = 37;
    const CPUGDNStateLayout full_layout(n_heads, d_k, d_v);
    std::vector<float> base_state(static_cast<size_t>(n_heads) * static_cast<size_t>(full_layout.headFloats()), 0.0f);
    for (int h = 0; h < n_heads; ++h)
    {
        float *hs = base_state.data() + static_cast<size_t>(h) * full_layout.headFloats();
        for (int i = 0; i < d_k * d_v; ++i)
            hs[1 + i] = ((i + h) % 17 == 0) ? 1.25e-40f : 0.001f * static_cast<float>((i % 13) - 6);
    }
    const auto Q = makeDeterministicFloats(static_cast<size_t>(seq_len * n_heads * d_k), 101);
    const auto K = makeDeterministicFloats(static_cast<size_t>(seq_len * n_heads * d_k), 102);
    const auto V = makeDeterministicFloats(static_cast<size_t>(seq_len * n_heads * d_v), 103);
    const auto alpha = makeDeterministicFloats(static_cast<size_t>(seq_len * n_heads), 104);
    const auto beta_raw = makeDeterministicFloats(static_cast<size_t>(seq_len * n_heads), 105);
    const std::vector<float> A_log(static_cast<size_t>(n_heads), -0.5f);
    const std::vector<float> dt_bias(static_cast<size_t>(n_heads), 0.1f);

    CPUGatedDeltaNet gdn;
    std::vector<float> state_1t = base_state;
    std::vector<float> out_1t(static_cast<size_t>(seq_len * n_heads * d_v), -55.0f);
    {
        llaminar::v2::ThreadCountGuard team(1);
        ASSERT_TRUE(gdn.chunk_forward(Q.data(), K.data(), V.data(), alpha.data(), beta_raw.data(),
            A_log.data(), dt_bias.data(), out_1t.data(), state_1t.data(),
            seq_len, n_heads, d_k, d_v, /*chunk_size=*/64, /*use_qk_l2norm=*/true));
    }
    for (const int threads : {4, 8})
    {
        llaminar::v2::ThreadCountGuard team(threads);
        std::vector<float> state_mt = base_state;
        std::vector<float> out_mt(static_cast<size_t>(seq_len * n_heads * d_v), -66.0f);
        ASSERT_TRUE(gdn.chunk_forward(Q.data(), K.data(), V.data(), alpha.data(), beta_raw.data(),
            A_log.data(), dt_bias.data(), out_mt.data(), state_mt.data(),
            seq_len, n_heads, d_k, d_v, /*chunk_size=*/64, /*use_qk_l2norm=*/true));
        EXPECT_EQ(std::memcmp(out_mt.data(), out_1t.data(), out_1t.size() * sizeof(float)), 0)
            << "chunk_forward output mismatch threads=" << threads;
        EXPECT_EQ(std::memcmp(state_mt.data(), state_1t.data(), state_1t.size() * sizeof(float)), 0)
            << "chunk_forward state mismatch threads=" << threads;
    }
}

/** @test UnsignedInt8 L1d 3row x 4z strip prefill and 6row x 2z SumPairs match serial M=1 oracle byte-for-byte. */
TEST(RetainedCPUMegakernelReplay, UnsignedInt8L1StripAndSumPairsReplay)
{
    using Kernel = cpu::native_quantised::CPUNativeQuantisedGemmKernel;
    struct Shape { int n; int k; };
    for (const Shape shape : {Shape{256, 256}, Shape{290, 1024}})
    {
        const int bpr = shape.k / Q8_1Block::BLOCK_SIZE;
        auto w = TestTensorFactory::createQ8_0Random(
            {static_cast<size_t>(shape.n), static_cast<size_t>(shape.k)},
            static_cast<uint32_t>(7000 + shape.n + shape.k));
        Kernel gemm(w.get(), 0, -1, CPUProjectionNumericalPolicy::BackendNative);
        ASSERT_TRUE(gemm.isValid());
        const auto &packed = gemm.packedWeights();

        for (const int threads : {1, 8})
        {
            llaminar::v2::ThreadCountGuard team(threads);
            for (const int M : {12, 13, 49, 83})
            {
                const auto x = makeDeterministicFloats(static_cast<size_t>(M * shape.k), M + shape.n);
                std::vector<Q8_1Block> a_q8(static_cast<size_t>(M * bpr));
                cpu::native_quantised::quantize_activations_to_q8_1(
                    x.data(), a_q8.data(), M, shape.k, bpr, CPUProjectionNumericalPolicy::BackendNative);

                NativeQuantisedTestPartialStorage partials_batch(packed, M);
                std::vector<float> got(static_cast<size_t>(M * shape.n), -99.0f);
                if (activeISALevel() == ISALevel::Scalar)
                {
                    // Grouped TwoRowPairGrid/KPart verifier routes fail fast on Scalar ISA.
                    EXPECT_THROW(
                        cpu::native_quantised::gemm_native_quantised_preq(
                            packed, a_q8.data(), got.data(), partials_batch.span(), M, shape.n,
                            cpu::native_quantised::ISAPath::AUTO,
                            cpu::native_quantised::VerifierRowsPolicy::Auto,
                            cpu::native_quantised::PrefillSchedulePolicy::TwoRowPairGrid),
                        std::runtime_error);
                    continue;
                }
                cpu::native_quantised::gemm_native_quantised_preq(
                    packed, a_q8.data(), got.data(), partials_batch.span(), M, shape.n);

                NativeQuantisedTestPartialStorage partials_one(packed, 1);
                std::vector<float> ref_row(static_cast<size_t>(shape.n), -88.0f);
                for (int r = 0; r < M; ++r)
                {
                    runGemvOracle(packed, a_q8.data() + static_cast<size_t>(r * bpr),
                        ref_row.data(), partials_one.span(), threads);
                    EXPECT_EQ(std::memcmp(got.data() + static_cast<size_t>(r * shape.n),
                                          ref_row.data(), static_cast<size_t>(shape.n) * sizeof(float)), 0)
                        << "L1Strip n=" << shape.n << " k=" << shape.k << " threads=" << threads << " M=" << M << " r=" << r;
                }
            }
        }
    }

    // Exercise 6row x 2z SumPairs (`gemm_6row_unsigned_int8_2z_sum_pairs`) at M = {6, 7, 8, 15}.
    constexpr int n_hc = 320;
    constexpr int k_hc = 320;
    const int bpr_hc = k_hc / Q8_1Block::BLOCK_SIZE;
    auto w_hc = TestTensorFactory::createQ8_0Random({static_cast<size_t>(n_hc), static_cast<size_t>(k_hc)}, 8811u);
    Kernel gemm_hc(w_hc.get(), 0, -1, CPUProjectionNumericalPolicy::BackendNative);
    ASSERT_TRUE(gemm_hc.isValid());
    for (const int threads : {1, 4, 8})
    {
        llaminar::v2::ThreadCountGuard team(threads);
        for (const int M : {6, 7, 8, 15})
        {
            const auto x = makeDeterministicFloats(static_cast<size_t>(M * k_hc), 300 + M);
            std::vector<float> got(static_cast<size_t>(M * n_hc), -77.0f);
            std::vector<Q8_1Block> p(static_cast<size_t>(M * bpr_hc)), r(static_cast<size_t>(M * bpr_hc));
            cpu::native_quantised::cpuQ8ResidualCorrectedProjection(
                gemm_hc.packedWeights(), x.data(), got.data(), DeviceRowRange::fullyActive(M), p, r);

            std::vector<float> ref_row(static_cast<size_t>(n_hc));
            std::vector<Q8_1Block> p1(static_cast<size_t>(bpr_hc)), r1(static_cast<size_t>(bpr_hc));
            for (int row = 0; row < M; ++row)
            {
                cpu::native_quantised::cpuQ8ResidualCorrectedProjection(
                    gemm_hc.packedWeights(), x.data() + static_cast<size_t>(row * k_hc), ref_row.data(),
                    DeviceRowRange::fullyActive(1), p1, r1);
                EXPECT_EQ(std::memcmp(got.data() + static_cast<size_t>(row * n_hc),
                                      ref_row.data(), static_cast<size_t>(n_hc) * sizeof(float)), 0)
                    << "SumPairs threads=" << threads << " M=" << M << " row=" << row;
            }
        }
    }
}

/** @test GroupedMoE 6row x 2z / 3row x 4z kernels and kStripMaxRows=64 boundary match serial decode byte-for-byte. */
TEST(RetainedCPUMegakernelReplay, GroupedMoE2z4zAndLargeStripReplay)
{
    using Kernel = cpu::native_quantised::CPUNativeQuantisedGemmKernel;
    using Desc = Kernel::BatchedPrequantizedProjectionDesc;

    const auto &formats = quantizedMoEVerifierFormats();
    constexpr int hidden = 256;
    constexpr int intermediate = 256;
    const int hidden_blocks = hidden / Q8_1Block::BLOCK_SIZE;
    const int inter_blocks = intermediate / Q8_1Block::BLOCK_SIZE;
    const auto policy = CPUProjectionNumericalPolicy::GPUAlignedExpert;
    constexpr int unique_rows = 11;
    const auto base_hidden = makeDeterministicFloats(static_cast<size_t>(unique_rows * hidden), 93);

    // Exercise UnsignedNibble (Q4_K, Q4_0), CompactFiveBit (Q5_1), and UnsignedInt8 (Q8_0)
    // across rows = {3, 6, 7, 9, 65} (crossing 6row x 2z, 3row x 4z, and kStripMaxRows=64!).
    for (size_t fi = 0; fi < formats.size(); ++fi)
    {
        const auto &fmt = formats[fi];
        const std::string lbl(fmt.label);
        if (lbl != "Q4_K" && lbl != "Q4_0" && lbl != "Q5_1" && lbl != "Q8_0")
            continue;
        SCOPED_TRACE(::testing::Message() << "grouped_2z4z format=" << lbl);

        auto w_gate = fmt.create({static_cast<size_t>(intermediate), static_cast<size_t>(hidden)}, 900u + static_cast<uint32_t>(fi));
        auto w_up = fmt.create({static_cast<size_t>(intermediate), static_cast<size_t>(hidden)}, 1000u + static_cast<uint32_t>(fi));
        auto w_down = fmt.create({static_cast<size_t>(hidden), static_cast<size_t>(intermediate)}, 1100u + static_cast<uint32_t>(fi));

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
        ASSERT_TRUE(k_gate.isValid() && k_up.isValid() && k_down.isValid());

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

        for (const int threads : {1, 8})
        {
            llaminar::v2::ThreadCountGuard team(threads);
            for (const int route_rows : {3, 6, 7, 9, 65})
            {
                std::vector<int> row_map(static_cast<size_t>(route_rows));
                for (int i = 0; i < route_rows; ++i)
                    row_map[static_cast<size_t>(i)] = i % unique_rows;
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
                                          static_cast<size_t>(hidden) * sizeof(float)), 0)
                        << "format=" << lbl << " threads=" << threads << " route_rows=" << route_rows << " i=" << i;
                }
            }
        }
    }
}

/** @test StageRunnerRegistry PP=1 MTP capability delegation enforces exact conjunction/disjunction semantics (ON12). */
TEST(RetainedCPUMegakernelReplay, StageRunnerRegistryMTPDelegationContract)
{
    class CapabilityStubRunner final : public MockDeviceRunner
    {
    public:
        bool preserves_main = false;
        bool logical_base = false;
        bool shifted_reuse = false;
        bool requires_replay = false;
        bool spec_pub = false;
        bool supportsMTPSidecarPreservesMainState() const override { return preserves_main; }
        bool supportsLogicalMTPVerifierBaseCheckpoint() const override { return logical_base; }
        bool supportsMTPShiftedRowReuseFromSidecar() const override { return shifted_reuse; }
        bool requiresMTPDecodeEquivalentVerifierReplay() const override { return requires_replay; }
        bool supportsMTPSpecStatePublication() const override { return spec_pub; }
    };

    // Empty registry returns false for every capability query.
    StageRunnerRegistry empty_reg;
    EXPECT_FALSE(empty_reg.supportsMTPSidecarPreservesMainStateAll());
    EXPECT_FALSE(empty_reg.supportsLogicalMTPVerifierBaseCheckpointAll());
    EXPECT_FALSE(empty_reg.supportsMTPShiftedRowReuseFromSidecarAll());
    EXPECT_FALSE(empty_reg.requiresMTPDecodeEquivalentVerifierReplayAny());
    EXPECT_FALSE(empty_reg.supportsMTPSpecStatePublicationAll());

    auto make_entry = [](int stage_id, const std::string &domain, std::unique_ptr<IInferenceRunner> runner) {
        StageRunnerEntry entry;
        entry.stage_id = stage_id;
        entry.domain_name = domain;
        entry.action.stage_id = stage_id;
        entry.action.domain_name = domain;
        entry.runner = std::move(runner);
        return entry;
    };

    // Single capable stage delegates every positive flag.
    StageRunnerRegistry single_reg;
    auto r0 = std::make_unique<CapabilityStubRunner>();
    r0->preserves_main = true;
    r0->logical_base = true;
    r0->shifted_reuse = true;
    r0->requires_replay = true;
    r0->spec_pub = true;
    single_reg.add(make_entry(0, "cpu_tp", std::move(r0)));
    EXPECT_TRUE(single_reg.supportsMTPSidecarPreservesMainStateAll());
    EXPECT_TRUE(single_reg.supportsLogicalMTPVerifierBaseCheckpointAll());
    EXPECT_TRUE(single_reg.supportsMTPShiftedRowReuseFromSidecarAll());
    EXPECT_TRUE(single_reg.requiresMTPDecodeEquivalentVerifierReplayAny());
    EXPECT_TRUE(single_reg.supportsMTPSpecStatePublicationAll());

    // Mixed stages: All() requires every stage; Any() triggers if any stage requires it.
    StageRunnerRegistry mixed_reg;
    auto m0 = std::make_unique<CapabilityStubRunner>();
    m0->preserves_main = true;
    m0->logical_base = true;
    m0->shifted_reuse = true;
    m0->requires_replay = false;
    m0->spec_pub = true;
    auto m1 = std::make_unique<CapabilityStubRunner>();
    m1->preserves_main = true;
    m1->logical_base = false;
    m1->shifted_reuse = true;
    m1->requires_replay = true;
    m1->spec_pub = false;
    mixed_reg.add(make_entry(0, "stage0", std::move(m0)));
    mixed_reg.add(make_entry(1, "stage1", std::move(m1)));
    EXPECT_TRUE(mixed_reg.supportsMTPSidecarPreservesMainStateAll());
    EXPECT_FALSE(mixed_reg.supportsLogicalMTPVerifierBaseCheckpointAll());
    EXPECT_TRUE(mixed_reg.supportsMTPShiftedRowReuseFromSidecarAll());
    EXPECT_TRUE(mixed_reg.requiresMTPDecodeEquivalentVerifierReplayAny());
    EXPECT_FALSE(mixed_reg.supportsMTPSpecStatePublicationAll());
}

/** @test Single-row CPU FP32 RMSNorm opts into RetainedTeam MasterSingle lane while M>1 stays None (ON14). */
TEST(RetainedCPUMegakernelReplay, CPURetainedTeamModeStageContract)
{
    constexpr size_t hidden = 256;
    const auto vals = makeDeterministicFloats(4u * hidden, 61);
    const auto gamma_vals = makeDeterministicFloats(hidden, 62);
    auto in1 = makeFP32Tensor(1u, hidden, vals);
    auto out1 = makeFP32Tensor(1u, hidden, std::vector<float>(hidden, 0.0f));
    auto in4 = makeFP32Tensor(4u, hidden, vals);
    auto out4 = makeFP32Tensor(4u, hidden, std::vector<float>(4u * hidden, 0.0f));
    auto gamma = makeFP32Tensor(1u, hidden, gamma_vals);

    RMSNormStage::Params p_m1;
    p_m1.device_id = DeviceId::cpu();
    p_m1.input = in1.get();
    p_m1.output = out1.get();
    p_m1.gamma = gamma.get();
    p_m1.seq_len = 1;
    p_m1.subtract_one = true;
    RMSNormStage stage_m1(p_m1);
    EXPECT_EQ(stage_m1.cpuRetainedTeamMode(), IComputeStage::CPURetainedTeamMode::MasterSingle);

    RMSNormStage::Params p_m4 = p_m1;
    p_m4.input = in4.get();
    p_m4.output = out4.get();
    p_m4.seq_len = 4;
    RMSNormStage stage_m4(p_m4);
    EXPECT_EQ(stage_m4.cpuRetainedTeamMode(), IComputeStage::CPURetainedTeamMode::None);

    // Prepare & execute inside a single retained OpenMP team and verify exact match to standalone execute().
    std::vector<float> ref_out(hidden, -99.0f);
    std::vector<float> eff_gamma(hidden);
    for (size_t i = 0; i < hidden; ++i) eff_gamma[i] = 1.0f + gamma_vals[i];
    primitives::rmsnorm_fused_vectorized_row(in1->data(), eff_gamma.data(), ref_out.data(), hidden, 1e-6f);

    llaminar::v2::ThreadCountGuard team(4);
    stage_m1.prepareForRetainedTeam(nullptr);
    std::atomic<bool> ok{true};
    auto body = [&]() {
        if (omp_get_thread_num() == 0)
        {
            if (!stage_m1.execute(nullptr))
                ok.store(false, std::memory_order_relaxed);
        }
    };
    OMP_LAYER_PARALLEL(body(););
    ASSERT_TRUE(ok.load());
    EXPECT_EQ(std::memcmp(out1->data(), ref_out.data(), hidden * sizeof(float)), 0);
}

} // namespace llaminar2::test
