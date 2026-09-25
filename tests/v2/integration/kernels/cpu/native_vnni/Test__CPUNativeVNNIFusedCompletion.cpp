/**
 * @file Test__CPUNativeVNNIFusedCompletion.cpp
 * @brief Prove fused projection completion before standalone or team callers resume.
 *
 * The projection, optional bias consumer and caller form a publication DAG.
 * Bias-free bundles must not require a dummy epilogue, but every worker must
 * still observe complete output when the API returns. This model-free suite
 * exercises all three layer-wide schedules with the canonical format registry,
 * mixed biased/unbiased members, padded tails and repeated nested-team calls.
 * It asserts native bytes and ownership, never a noisy timing threshold.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "utils/NativeVNNITestPartialStorage.h"
#include "utils/QuantizedVerifierFormats.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;

/** @brief Restore the caller's worker and arithmetic-partition test controls. */
class FusedCompletionScope final
{
public:
    /**
     * @brief Bind one stable team and an explicit serial arithmetic partition.
     * @param workers Positive width shared by the oracle and grouped invocation.
     * @param partitions Number of K partials in the serial arithmetic contract.
     */
    FusedCompletionScope(int workers, int partitions)
        : workers_(omp_get_max_threads()), dynamic_(omp_get_dynamic())
    {
        if (const char *value = std::getenv("LLAMINAR_CPU_VNNI_K_TILES"))
            prior_partitions_ = value;
        omp_set_dynamic(0);
        omp_set_num_threads(workers);
        setenv("LLAMINAR_CPU_VNNI_K_TILES", std::to_string(partitions).c_str(), 1);
        mutableDebugEnv().cpu_vnni.reload();
    }

    /** @brief Restore both runtime controls before the next test starts. */
    ~FusedCompletionScope()
    {
        if (prior_partitions_)
            setenv("LLAMINAR_CPU_VNNI_K_TILES", prior_partitions_->c_str(), 1);
        else
            unsetenv("LLAMINAR_CPU_VNNI_K_TILES");
        mutableDebugEnv().cpu_vnni.reload();
        omp_set_num_threads(workers_);
        omp_set_dynamic(dynamic_);
    }

    FusedCompletionScope(const FusedCompletionScope &) = delete;
    FusedCompletionScope &operator=(const FusedCompletionScope &) = delete;

private:
    int workers_; ///< Restored maximum team width.
    int dynamic_; ///< Restored dynamic-team policy.
    std::optional<std::string> prior_partitions_; ///< Absent and empty differ.
};

/**
 * @brief Check native output bytes immediately after each participant returns.
 * @param route Expected layer-wide storage/scheduling authority.
 *
 * All workers read the complete output before the fixture's next barrier.
 * An omitted completion edge therefore cannot be hidden by the caller's join.
 * Oracle projections execute independently through the public serial API.
 */
void proveFusedCompletion(NativeVNNIFusedPartialRoute route)
{
    constexpr int k = 512;
    constexpr int workers = 7;
    constexpr float guard = -12345.0f;
    const int n0 = route == NativeVNNIFusedPartialRoute::SharedBank ? 31 : 257;
    const int n1 = route == NativeVNNIFusedPartialRoute::SharedBank ? 61 : 321;
    FusedCompletionScope runtime(workers,
        route == NativeVNNIFusedPartialRoute::FullK ? 1 : 16);
    const auto &formats = quantizedVerifierFormats();
    for (size_t index = 0; index < formats.size(); ++index)
    {
        const auto &format0 = formats[index];
        const auto &format1 = formats[(index + 1) % formats.size()];
        SCOPED_TRACE(std::string(format0.label) + "/" + format1.label);
        auto weights0 = format0.create({size_t(n0), size_t(k)}, 1307 + index);
        auto weights1 = format1.create({size_t(n1), size_t(k)}, 2909 + index);
        CPUNativeVNNIGemmKernel kernel0(weights0.get());
        CPUNativeVNNIGemmKernel kernel1(weights1.get());
        ASSERT_TRUE(kernel0.isValid());
        ASSERT_TRUE(kernel1.isValid());
        const auto &packed0 = kernel0.packedWeights();
        const auto &packed1 = kernel1.packedWeights();

        for (int rows : {1, 2, 3, 15, 31})
        {
            if (route == NativeVNNIFusedPartialRoute::SharedBank && rows > 3)
                continue; // More row tiles legitimately select local ownership.
            SCOPED_TRACE("rows=" + std::to_string(rows));
            std::vector<float> source(size_t(rows) * k);
            for (size_t value = 0; value < source.size(); ++value)
                source[value] = float(int((value * 37 + index * 13) % 257) - 128) * 0.003125f;
            std::vector<Q8_1Block> input(size_t(rows) * packed0.blocks_per_row);
            quantize_activations_to_q8_1(source.data(), input.data(), rows, k, packed0.blocks_per_row);
            std::array<std::vector<float>, 2> outputs = {
                std::vector<float>(size_t(rows) * (n0 + 7), guard),
                std::vector<float>(size_t(rows) * (n1 + 7), guard)};
            std::array<std::vector<float>, 2> references = outputs;
            std::array<std::vector<float>, 2> biases = {
                std::vector<float>(n0), std::vector<float>(n1)};
            const std::array<const CPUNativeVNNIPackedWeights *, 2> packed = {&packed0, &packed1};
            const std::array<int, 2> widths = {n0, n1};
            for (int member = 0; member < 2; ++member)
            {
                NativeVNNITestPartialStorage serial_bank(*packed[member], 1);
                for (int column = 0; column < widths[member]; ++column)
                    biases[member][column] = float(column % 19 - 9) * 0.015625f;
                for (int row = 0; row < rows; ++row)
                    gemv_native_vnni_preq(*packed[member],
                        input.data() + size_t(row) * packed[member]->blocks_per_row,
                        references[member].data() + size_t(row) * (widths[member] + 7),
                        serial_bank.span());
            }
            // The second member alone has bias, proving mixed bundles wait
            // for producers without applying an epilogue to the first member.
            for (const bool biased : {false, true})
            {
                SCOPED_TRACE(biased ? "mixed bias" : "bias-free");
                auto expected = references;
                if (biased)
                    for (int row = 0; row < rows; ++row)
                        for (int column = 0; column < n1; ++column)
                            expected[1][size_t(row) * (n1 + 7) + column] += biases[1][column];
                std::array<FusedVerifierRowsDesc, 2> descriptors;
                for (int member = 0; member < 2; ++member)
                    descriptors[member] = FusedVerifierRowsDesc{
                        .packed = packed[member],
                        .output = outputs[member].data(),
                        .bias = biased && member == 1 ? biases[member].data() : nullptr,
                        .N = widths[member],
                        .ldc = widths[member] + 7,
                        .rows = rows,
                        .verifier_schedule = rows == 1 ? VerifierRowsPolicy::Auto : VerifierRowsPolicy::Pairwise};
                const auto plan = planNativeVNNIFusedRows(descriptors, rows, workers, ISAPath::AUTO);
                ASSERT_EQ(plan.partial_route, route);
                NativeVNNITestPartialStorage partials(plan.sharedPartialFloats());
                auto invoke = [&]() {
                    return gemm_native_vnni_fused_verifier_rows_preq(input.data(),
                        descriptors.data(), partials.span(), descriptors.size(), rows,
                        packed0.blocks_per_row);
                };
                auto complete = [&]() {
                    for (int member = 0; member < 2; ++member)
                        if (std::memcmp(outputs[member].data(), expected[member].data(),
                            expected[member].size() * sizeof(float)) != 0)
                            return false;
                    return true;
                };
                ASSERT_TRUE(invoke());
                ASSERT_TRUE(complete());
                std::array<int, workers> success;
                success.fill(1);
#pragma omp parallel
                {
                    const int worker = omp_get_thread_num();
                    for (int repetition = 0; repetition < 4; ++repetition)
                    {
#pragma omp single
                        {
                            for (auto &output : outputs)
                                std::fill(output.begin(), output.end(), guard);
                        }
                        if (!invoke() || !complete())
                            success[worker] = 0;
                        // Protect each reader from the next repetition's poison.
                        // This is deliberately AFTER the immediate output check.
#pragma omp barrier
                    }
                }
                EXPECT_TRUE(std::all_of(success.begin(), success.end(), [](int value) { return value == 1; }));
            }
        }
    }
}
} // namespace

/** @test Full-K bundles publish every biased and unbiased output before return. */
TEST(CPUNativeVNNIFusedCompletion, FullK)
{
    proveFusedCompletion(NativeVNNIFusedPartialRoute::FullK);
}

/** @test Per-output ordered trees need only the real optional bias dependency. */
TEST(CPUNativeVNNIFusedCompletion, OutputTileLocal)
{
    proveFusedCompletion(NativeVNNIFusedPartialRoute::OutputTileLocal);
}

/** @test A shared partial bank retains producer/reducer and completion edges. */
TEST(CPUNativeVNNIFusedCompletion, SharedBank)
{
    proveFusedCompletion(NativeVNNIFusedPartialRoute::SharedBank);
}
} // namespace llaminar2::test
