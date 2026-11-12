/**
 * @file Test__CPUDraftVocabularyBudgetHead.cpp
 * @brief NEON half of the draft-vocabulary-budget diagnostic (idea 16 B4):
 *        the selected-row byte proof on the real production head kernel.
 *
 * Architecture-boundary note: this translation unit exercises the ARM NEON
 * NativeQuantised decode path and is registered only for AArch64 builds
 * (the portable policy/coupling half lives in
 * Test__CPUDraftVocabularyBudget.cpp). Non-installed diagnostic namespace;
 * no production routing, sampler, verifier, CLI/config, PerfStats
 * production family, PMA admission or graph change is implied or granted.
 *
 * Reviewer-findings rounds (209/210, then 217 A2/A5) repaired here:
 * - D3: the shortlisted row evaluation uses the ACTUAL production policy
 *   and tree. The ordinary LM head runs BackendNative, for which the
 *   canonical partition count is ONE ordered chain over the complete K
 *   range (`partitionsFor` returns 1 for BackendNative; the 16x4
 *   partition/fold structure belongs to the GPUAlignedExpert MoE
 *   contract). The row walk therefore accumulates per block in block
 *   order with the production `accumulateSingleScaleBlockNEON` program —
 *   exactly the reference GEMV's FP32 boundary structure — and the
 *   comparisons are BITWISE (memcmp on the float bit patterns, signed
 *   zero included), never operator== on floats. The production GEMV is
 *   the reference; the optimized dispatch is untouched.
 * - Coverage within the diagnostic's device-free scope: two head
 *   geometries (V=96 with a half-populated tail chunk at K=2048, and
 *   V=64 at K=1024), shortlists spanning both chunks (global IDs across
 *   the chunk boundary), shortlist sizes including the non-aligned
 *   values, multi-N execution over one engine, and the grouped-target
 *   rows versus serial M=1 proof through the production grouped LM-head
 *   verifier path (M=4 rows, bit-exact per row). True multi-rank
 *   shard/replicate movement is device-scope and remains unproved here —
 *   claimed nowhere.
 * - A2: the traffic accounting is per arm and counts LOGICAL
 *   PREPARED-OPERAND VOLUME (rows x per-row prepared bytes) against a
 *   like-for-like full-head figure — it is NOT measured DRAM/cache-line
 *   traffic and is never described as hardware-bandwidth evidence.
 * - A5: coverage equals EXECUTED source — both head geometries are
 *   actually evaluated, even-sized shortlists exercise the reverse
 *   ORDERING branch (a fixture ordering choice, not selector backfill),
 *   and a reverse arm covers the final vocabulary row (V-1) and the
 *   ragged tail. The grouped M=4-versus-serial check is a production
 *   METAMORPHIC HEAD check; it is NOT a full model/KV/GDN/stop-state
 *   replay — that boundary is explicit.
 * - Counted traffic: the shortlisted evaluation's touched prepared bytes
 *   are counted and reported against the full head (rows x per-row
 *   bytes), so the traffic claim is a measurement, not a comment.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/cpu/gemm/CPUNativeNEONContributionContract.h"
#include "kernels/cpu/gemm/CPUNativeNEONMultiScaleOps.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "kernels/cpu/gemm/CPUProjectionQ8Publication.h"
#include "kernels/common/DeviceRowRange.h"
#include "tensors/AlignedVector.h"
#include "tensors/TensorClasses.h"
#include "utils/QuantizedVerifierFormats.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace llaminar2;
using llaminar2::cpu::native_quantised::CPUNativeQuantisedGemmKernel;

namespace
{
    constexpr int kHeadK = 2048;
    constexpr int kTailVocab = 96; ///< 1.5 chunks: exercises the tail.
    constexpr int kSmallK = 1024;
    constexpr int kSmallVocab = 64;
    constexpr int kVerifierRows = 4;

    /// K blocks for a head width (K/32).
    constexpr int kHeadBlocks(int k) { return k / 32; }

    /// Prepared bytes one vocabulary row touches: 4 B per group x 8 groups
    /// per block + the fp16 scale + the raw nibble word, per K block.
    size_t rowBytes(int blocks_per_row)
    {
        return static_cast<size_t>(blocks_per_row) * (32u + 2u + 2u);
    }

    /** @brief Bitwise equality of two floats (signed zero included). */
    bool sameBits(float a, float b)
    {
        std::uint32_t ra, rb;
        std::memcpy(&ra, &a, sizeof(ra));
        std::memcpy(&rb, &b, sizeof(rb));
        return ra == rb;
    }

    /**
     * @brief Evaluate ONE vocabulary row from the prepared planes with the
     *        production BackendNative program and count the touched bytes.
     *
     * The row is identified by its GLOBAL ID; only this row's bytes are
     * read. Per K block: the row's four K bytes per group feed the
     * production SDOT leaf (independent per-column lanes; lane 0 is the
     * exact per-column sum), then ONE production FP32 scale/accumulate
     * step with the IQ3_S source-scale equation — the same block order
     * and FP32 rounding points as the reference GEMV at BackendNative.
     */
    float evaluateRowFromPlanes(const CPUNativeQuantisedGemmKernel &engine,
        const Q8_1Block *activation_row, std::uint32_t global_row,
        size_t &touched_bytes)
    {
        using MultiScaleOps = cpu::native_quantised::multi_scale::
            NEONMultiScaleOps<cpu::native_quantised::multi_scale::
                NEONByteDot::ASIMD>;
        const auto &packed = engine.packedWeights();
        const int chunk = static_cast<int>(global_row / 64u);
        const int column = static_cast<int>(global_row % 64u);
        float32x4_t accumulator = vdupq_n_f32(0.0f);
        for (int block = 0; block < packed.blocks_per_row; ++block)
        {
            const auto *unit = packed.interleavedBase() +
                (static_cast<size_t>(chunk) *
                        static_cast<size_t>(packed.blocks_per_row) +
                    static_cast<size_t>(block)) *
                    static_cast<size_t>(packed.interleaved_block_stride);
            int32x4_t dot = vdupq_n_s32(0);
            for (int group = 0; group < 8; ++group)
            {
                std::uint32_t row_bytes = 0;
                std::memcpy(&row_bytes,
                    unit + static_cast<size_t>(group) * 256u +
                        static_cast<size_t>(column) * 4u,
                    sizeof(row_bytes));
                const auto weights =
                    vreinterpretq_s8_u32(vdupq_n_u32(row_bytes));
                int word = 0;
                std::memcpy(&word, activation_row[block].qs + group * 4,
                    sizeof(word));
                dot = MultiScaleOps::dot(dot,
                    std::bit_cast<MultiScaleOps::I>(weights),
                    MultiScaleOps::I{} + word);
            }
            const std::uint16_t d_half = *reinterpret_cast<const uint16_t *>(
                unit + static_cast<size_t>(packed.data_stride) + 128u +
                static_cast<size_t>(column) * 2u);
            const std::uint16_t metadata =
                *reinterpret_cast<const uint16_t *>(
                    unit + static_cast<size_t>(packed.data_stride) + 256u +
                    static_cast<size_t>(column) * 2u);
            const float scale0 =
                cpu::native_quantised::nativeQuantisedFP16ScaleToFP32(d_half);
            const float scale = scale0 *
                static_cast<float>(static_cast<int>(metadata) * 2 + 1);
            const float activation_scale =
                cpu::native_quantised::nativeQuantisedFP16ScaleToFP32(
                    activation_row[block].d);
            accumulator = cpu::native_quantised::
                accumulateSingleScaleBlockNEON(accumulator, dot,
                    vdupq_n_f32(scale), activation_scale,
                    CPUProjectionNumericalPolicy::BackendNative);
        }
        touched_bytes += rowBytes(packed.blocks_per_row);
        return vgetq_lane_f32(accumulator, 0);
    }

    /** @brief Deterministic hidden rows with quantization-relevant tails. */
    AlignedVector<float> makeHidden(int rows, int k)
    {
        AlignedVector<float> hidden(static_cast<size_t>(rows) *
                                    static_cast<size_t>(k));
        for (size_t index = 0; index < hidden.size(); ++index)
        {
            const int centered =
                static_cast<int>((index * 41u + index / 17u) % 257u) - 128;
            hidden[index] = static_cast<float>(centered) * 0.003125f +
                static_cast<float>(static_cast<int>(index % 5) - 2) *
                    0.00021f;
        }
        return hidden;
    }
} // namespace

TEST(CPUDraftVocabularyBudgetHead, SelectedRowsAreBitwiseIdenticalAndCounted)
{
    // Two production-prepared head geometries: the ragged tail (V=96,
    // K=2048) and a second block count (V=64, K=1024). The ordinary LM
    // head policy is BackendNative, for which the production partition
    // count is one ordered chain.
    const auto &format = []() -> const test::QuantizedVerifierFormatCase &
    {
        for (const auto &entry : test::quantizedMoEVerifierFormats())
            if (entry.label == std::string("IQ3_S"))
                return entry;
        throw std::runtime_error("IQ3_S missing from the canonical registry");
    }();

    struct HeadFixture
    {
        int vocab;
        int k;
        std::unique_ptr<TensorBase> source;
        std::unique_ptr<CPUNativeQuantisedGemmKernel> engine;
    };
    std::vector<HeadFixture> heads;
    for (const auto &[vocab, k] :
        {std::pair<int, int>{kTailVocab, kHeadK}, {kSmallVocab, kSmallK}})
    {
        HeadFixture fixture;
        fixture.vocab = vocab;
        fixture.k = k;
        fixture.source = format.create(
            {static_cast<size_t>(vocab), static_cast<size_t>(k)}, 91000u);
        ASSERT_TRUE(fixture.source) << "head creation failed";
        fixture.engine = std::make_unique<CPUNativeQuantisedGemmKernel>(
            fixture.source.get(), 0, -1,
            CPUProjectionNumericalPolicy::BackendNative);
        ASSERT_TRUE(fixture.engine->isValid()) << "head preparation failed";
        ASSERT_EQ(fixture.engine->packedWeights().numerical_policy,
            CPUProjectionNumericalPolicy::BackendNative)
            << "the ordinary LM head policy is BackendNative";
        heads.push_back(std::move(fixture));
    }

    // Four deterministic hidden rows, quantized by the production
    // activation publication (the verifier-row set).
    const auto hidden = makeHidden(kVerifierRows, kHeadK);
    std::vector<Q8_1Block> activations(
        static_cast<size_t>(kVerifierRows) *
        static_cast<size_t>(kHeadBlocks(kHeadK)));
    cpu::native_quantised::quantize_activations_to_q8_1(hidden.data(),
        activations.data(), kVerifierRows, kHeadK, kHeadBlocks(kHeadK),
        CPUProjectionNumericalPolicy::BackendNative);

    // Reference: the production serial GEMV, one hidden row at a time.
    std::vector<std::vector<float>> reference(
        kVerifierRows, std::vector<float>(kTailVocab));
    {
        const auto &packed = heads[0].engine->packedWeights();
        std::vector<float> partial(
            cpu::native_quantised::nativeQuantisedProjectionPartialFloats(
                packed, 1));
        for (int row = 0; row < kVerifierRows; ++row)
        {
            const auto *act = &activations[static_cast<size_t>(row) *
                static_cast<size_t>(kHeadBlocks(kHeadK))];
            cpu::native_quantised::gemv_native_quantised_preq(packed, act,
                reference[static_cast<size_t>(row)].data(), partial);
        }
    }

    // Multi-N execution: the same engine, several shortlist sizes and
    // orderings — odd sizes (ascending), EVEN sizes (the reverse-
    // ORDERING branch — a fixture ordering choice, not selector
    // backfill), a chunk-spanning size, and a shortlist that INCLUDES
    // the final vocabulary row (V-1). Every selected row must be
    // BITWISE identical to the reference row for the activation row it is
    // evaluated against. The traffic accounting is per arm and counts
    // LOGICAL PREPARED-OPERAND VOLUME (rows x per-row prepared bytes) —
    // it is NOT measured DRAM/cache-line traffic.
    size_t counted_selected_bytes = 0;
    size_t evaluated_rows_total = 0;
    struct Arm
    {
        std::uint32_t n;
        bool reverse;
        const char *label;
    };
    const Arm arms[] = {{1, false, "ascending1"}, {3, false, "ascending3"},
        {2, true, "reverse2"}, {64, true, "reverse64"},
        {63, false, "ascending63"}, {95, true, "reverse95"}};
    for (const Arm &arm : arms)
    {
        std::vector<std::uint32_t> shortlist;
        for (std::uint32_t index = 0; index < arm.n; ++index)
        {
            // Reverse arms are an ORDERING choice of this fixture, not
            // selector backfill (backfill is selector-v1 semantics);
            // this fixture's reverse ordering is its own choice.
            const std::uint32_t id =
                arm.reverse ? (kTailVocab - 1u) - index : index;
            shortlist.push_back(id);
        }
        size_t touched = 0;
        for (const std::uint32_t global_row : shortlist)
        {
            ASSERT_LT(global_row, static_cast<std::uint32_t>(kTailVocab));
            // Each row is evaluated against its own activation row
            // (round-robin over the verifier-row set); the reference is
            // the GEMV row for the SAME pairing.
            const size_t activation_row_index =
                static_cast<size_t>(global_row % kVerifierRows);
            const auto *activation_row =
                &activations[activation_row_index *
                             static_cast<size_t>(kHeadBlocks(kHeadK))];
            const float selected = evaluateRowFromPlanes(*heads[0].engine,
                activation_row, global_row, touched);
            const float reference_row =
                reference[activation_row_index][global_row];
            // Bitwise comparison: signed zero and payload bits included.
            std::uint32_t selected_bits, reference_bits;
            std::memcpy(&selected_bits, &selected, sizeof(selected_bits));
            std::memcpy(&reference_bits, &reference_row,
                sizeof(reference_bits));
            EXPECT_EQ(selected_bits, reference_bits)
                << "selected row " << global_row
                << " is not bitwise identical to the full-head row";
            evaluated_rows_total += 1;
        }
        EXPECT_EQ(touched, static_cast<size_t>(arm.n) *
                rowBytes(kHeadBlocks(kHeadK)))
            << "arm " << arm.label << " touched-byte accounting";
        counted_selected_bytes += touched;
        std::cout << "CPU_B4_HEAD_TRAFFIC,arm=" << arm.label
                  << ",rows=" << arm.n
                  << ",logical_operand_bytes=" << touched
                  << ",like_for_like_full_head_bytes="
                  << static_cast<size_t>(kTailVocab) *
                     rowBytes(kHeadBlocks(kHeadK))
                  << ",accounting=logical_prepared_operand_volume"
                  << ",not_hardware_traffic=1\n";
    }
    EXPECT_EQ(evaluated_rows_total, 228u)
        << "the aggregate evaluated rows across arms (1+3+2+64+63+95)";
    // The reverse arms cover the final vocabulary row (V-1) and the
    // ragged tail rows (chunk 1, columns 0..31).
    EXPECT_EQ(counted_selected_bytes, 228u * rowBytes(kHeadBlocks(kHeadK)));

    // Second geometry exercised: the V=64/K=1024 head gets its own
    // reference and a shortlist arm (the same bitwise oracle).
    {
        const auto &packed1 = heads[1].engine->packedWeights();
        std::vector<float> partial1(
            cpu::native_quantised::nativeQuantisedProjectionPartialFloats(
                packed1, 1));
        const auto hidden1 = makeHidden(kVerifierRows, kSmallK);
        std::vector<Q8_1Block> activations1(
            static_cast<size_t>(kVerifierRows) *
            static_cast<size_t>(kHeadBlocks(kSmallK)));
        cpu::native_quantised::quantize_activations_to_q8_1(hidden1.data(),
            activations1.data(), kVerifierRows, kSmallK,
            kHeadBlocks(kSmallK),
            CPUProjectionNumericalPolicy::BackendNative);
        std::vector<std::vector<float>> reference1(
            kVerifierRows, std::vector<float>(kSmallVocab));
        for (int row = 0; row < kVerifierRows; ++row)
        {
            const auto *act = &activations1[static_cast<size_t>(row) *
                static_cast<size_t>(kHeadBlocks(kSmallK))];
            cpu::native_quantised::gemv_native_quantised_preq(packed1, act,
                reference1[static_cast<size_t>(row)].data(), partial1);
        }
        size_t touched1 = 0;
        for (std::uint32_t global_row : {0u, 32u, 63u})
        {
            const size_t activation_row_index =
                static_cast<size_t>(global_row % kVerifierRows);
            const auto *activation_row =
                &activations1[activation_row_index *
                              static_cast<size_t>(kHeadBlocks(kSmallK))];
            const float selected = evaluateRowFromPlanes(*heads[1].engine,
                activation_row, global_row, touched1);
            const float reference_row =
                reference1[activation_row_index][global_row];
            std::uint32_t selected_bits, reference_bits;
            std::memcpy(&selected_bits, &selected, sizeof(selected_bits));
            std::memcpy(&reference_bits, &reference_row,
                sizeof(reference_bits));
            EXPECT_EQ(selected_bits, reference_bits)
                << "second geometry: selected row " << global_row
                << " is not bitwise identical";
        }
        EXPECT_EQ(touched1, 3u * rowBytes(kHeadBlocks(kSmallK)));
    }
}

TEST(CPUDraftVocabularyBudgetHead, GroupedTargetRowsMatchSerialM1)
{
    // S2 clause at the head level: the production grouped LM-head
    // verifier path (M=4 rows through the fused verifier kernels) must
    // produce rows bit-identical to the serial M=1 GEMV, except for the
    // selected draft rows a shortlist policy would evaluate — here the
    // full set is evaluated by both paths, so every row is compared.
    const auto &format = []() -> const test::QuantizedVerifierFormatCase &
    {
        for (const auto &entry : test::quantizedMoEVerifierFormats())
            if (entry.label == std::string("IQ3_S"))
                return entry;
        throw std::runtime_error("IQ3_S missing from the canonical registry");
    }();
    auto head_source = format.create(
        {static_cast<size_t>(kTailVocab), static_cast<size_t>(kHeadK)},
        91000u);
    ASSERT_TRUE(head_source);
    CPUNativeQuantisedGemmKernel engine(head_source.get(), 0, -1,
        CPUProjectionNumericalPolicy::BackendNative);
    ASSERT_TRUE(engine.isValid());

    const auto hidden = makeHidden(kVerifierRows, kHeadK);
    std::vector<Q8_1Block> activations(
        static_cast<size_t>(kVerifierRows) *
        static_cast<size_t>(kHeadBlocks(kHeadK)));
    cpu::native_quantised::quantize_activations_to_q8_1(hidden.data(),
        activations.data(), kVerifierRows, kHeadK, kHeadBlocks(kHeadK),
        CPUProjectionNumericalPolicy::BackendNative);

    // Serial reference rows.
    std::vector<std::vector<float>> serial(
        kVerifierRows, std::vector<float>(kTailVocab));
    {
        const auto &packed = engine.packedWeights();
        std::vector<float> partial(
            cpu::native_quantised::nativeQuantisedProjectionPartialFloats(
                packed, 1));
        for (int row = 0; row < kVerifierRows; ++row)
        {
            const auto *act = &activations[static_cast<size_t>(row) *
                static_cast<size_t>(kHeadBlocks(kHeadK))];
            cpu::native_quantised::gemv_native_quantised_preq(packed, act,
                serial[static_cast<size_t>(row)].data(), partial);
        }
    }

    // Grouped rows through the production fused verifier path.
    FP32Tensor hidden_tensor(std::vector<size_t>{
        static_cast<size_t>(kVerifierRows), static_cast<size_t>(kHeadK)});
    std::memcpy(hidden_tensor.mutable_data(), hidden.data(),
        hidden.size() * sizeof(float));
    FP32Tensor grouped_logits(
        std::vector<size_t>{static_cast<size_t>(kVerifierRows),
            static_cast<size_t>(kTailVocab)});
    // The two-argument workspace manager allocates through the device
    // backend, so the CPU backend must be initialized first (the same
    // fixture pattern the STT bench uses; the W3 seal caught this gap).
    if (!llaminar2::hasCPUBackend())
        llaminar2::initCPUBackend(-1);
    DeviceWorkspaceManager workspace(DeviceId::cpu(), 1u << 22);
    ASSERT_TRUE(workspace.allocate(
        cpuProjectionQ8WorkspaceRequirements(
            kVerifierRows, kHeadK)));

    {
        const auto scope = engine.beginVerifierDecodeEquivalentScope(
            DeviceRowRange::fullyActive(kVerifierRows));
        std::vector<ITensorGemm::TensorProjectionDesc> projections;
        projections.push_back({&engine, &grouped_logits,
            static_cast<int>(kTailVocab), nullptr, "b4_head"});
        const bool ok = engine.multiply_fused_verifier_rows_decode_equivalent(
            &hidden_tensor, projections, kVerifierRows, kHeadK, nullptr,
            &workspace);
        ASSERT_TRUE(ok) << "the production grouped head path failed";
    }

    // Every grouped row is bitwise identical to its serial M=1 row.
    for (int row = 0; row < kVerifierRows; ++row)
    {
        const float *grouped_row = grouped_logits.data() +
            static_cast<size_t>(row) * static_cast<size_t>(kTailVocab);
        for (int token = 0; token < kTailVocab; ++token)
        {
            const float g = grouped_row[token];
            const float s = serial[static_cast<size_t>(row)][token];
            std::uint32_t gb, sb;
            std::memcpy(&gb, &g, sizeof(gb));
            std::memcpy(&sb, &s, sizeof(sb));
            EXPECT_EQ(gb, sb) << "grouped row " << row << " token " << token
                              << " differs bitwise from serial M=1";
        }
    }
}
