/**
 * @file Test__CPUEmbeddingGroupedVerifier.cpp
 * @brief Exhaustive CPU embedding batch-invariance tests for MTP verifier rows.
 *
 * The production CPU embedding API accepts one compact M-row token list. These
 * tests compare that call with M independent production M=1 calls for every
 * supported embedding-table codebook and every CPU activation output format.
 * Equality is checked on native storage bytes, including BF16, FP16, and Q8_1
 * block metadata. A perfstats counter proves that the grouped side used one
 * economical lookup invocation rather than a hidden serial replay API.
 *
 * This is deliberately a CPU-only unit binary. CUDA and ROCm execution lives in
 * explicit integration suites because unit tests must not initialize GPUs.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "kernels/cpu/ops/CPUEmbeddingKernelT.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "../../utils/EmbeddingVerifierFormats.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Enable and isolate structured route counters for one test. */
    class ScopedPerfStats
    {
    public:
        ScopedPerfStats()
        {
            const char *old = std::getenv("LLAMINAR_PERF_STATS_SUMMARY");
            if (old)
            {
                had_old_value_ = true;
                old_value_ = old;
            }
            setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

        ~ScopedPerfStats()
        {
            if (had_old_value_)
                setenv("LLAMINAR_PERF_STATS_SUMMARY", old_value_.c_str(), 1);
            else
                unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            mutableDebugEnv().reload();
            PerfStatsCollector::reset();
        }

    private:
        bool had_old_value_ = false;
        std::string old_value_;
    };

    /** @brief Report the first native byte that differs between grouped and serial output. */
    void expectByteExact(
        const void *grouped,
        const void *serial,
        size_t byte_count,
        const std::string &context)
    {
        if (std::memcmp(grouped, serial, byte_count) == 0)
            return;

        const auto *grouped_bytes = static_cast<const uint8_t *>(grouped);
        const auto *serial_bytes = static_cast<const uint8_t *>(serial);
        for (size_t byte = 0; byte < byte_count; ++byte)
        {
            if (grouped_bytes[byte] != serial_bytes[byte])
            {
                ADD_FAILURE() << context << " first mismatch at byte " << byte
                              << " grouped=" << static_cast<unsigned>(grouped_bytes[byte])
                              << " serial=" << static_cast<unsigned>(serial_bytes[byte]);
                return;
            }
        }
    }

    /** @brief Require the exact grouped CPU route and format tags for one case. */
    void expectGroupedCounter(
        const char *weight_format,
        const char *output_format,
        int verifier_rows,
        int d_model,
        const char *weight_route)
    {
        bool found = false;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.cpu_embedding_grouped_verifier_rows_calls"}))
        {
            auto tagEquals = [&](const char *name, const std::string &expected)
            {
                const auto it = record.tags.find(name);
                return it != record.tags.end() && it->second == expected;
            };
            found = found ||
                    (tagEquals("weight_format", weight_format) &&
                     tagEquals("output_format", output_format) &&
                     tagEquals("verifier_rows", std::to_string(verifier_rows)) &&
                     tagEquals("d_model", std::to_string(d_model)) &&
                     tagEquals("weight_route", weight_route) &&
                     tagEquals("token_source", "host") &&
                     tagEquals("invocation_policy", "single_grouped_call") &&
                     record.value == 1.0);
        }
        EXPECT_TRUE(found)
            << "CPU embedding did not publish the expected grouped route for "
            << weight_format << " -> " << output_format << " M=" << verifier_rows << "\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.cpu_embedding_grouped_verifier_rows_calls"}, 20);
    }

    /**
     * @brief Run M2/M3/M4 grouped-vs-serial proof for one weight/output pair.
     *
     * The same production kernel object serves both sides. This matters for
     * quantized tables because it exercises the real cached EmbedQ8 preparation
     * used by repeated decode calls while changing only the invocation width.
     */
    template <typename OutputTensorT>
    void runOutputFormatSweep(
        const EmbeddingVerifierFormatCase &format,
        TensorBase *embedding_table,
        int vocab_size,
        int d_model)
    {
        CPUEmbeddingKernelT<OutputTensorT> kernel;
        const std::array<int, 4> tokens = {3, vocab_size - 1, 17, vocab_size / 2};

        for (int verifier_rows : {2, 3, 4})
        {
            SCOPED_TRACE(std::string(format.label) + " M=" + std::to_string(verifier_rows));

            OutputTensorT grouped_output(
                {static_cast<size_t>(verifier_rows), static_cast<size_t>(d_model)});
            OutputTensorT serial_output(
                {static_cast<size_t>(verifier_rows), static_cast<size_t>(d_model)});
            OutputTensorT one_row_output({1u, static_cast<size_t>(d_model)});

            PerfStatsCollector::reset();
            ASSERT_TRUE(kernel.apply_tensor(
                embedding_table,
                tokens.data(),
                verifier_rows,
                d_model,
                &grouped_output));

            const size_t row_bytes = one_row_output.size_bytes();
            auto *serial_bytes = static_cast<uint8_t *>(serial_output.raw_mutable_data());
            for (int row = 0; row < verifier_rows; ++row)
            {
                ASSERT_TRUE(kernel.apply_tensor(
                    embedding_table,
                    tokens.data() + row,
                    1,
                    d_model,
                    &one_row_output));
                std::memcpy(
                    serial_bytes + static_cast<size_t>(row) * row_bytes,
                    one_row_output.raw_data(),
                    row_bytes);
            }

            expectByteExact(
                grouped_output.raw_data(),
                serial_output.raw_data(),
                grouped_output.size_bytes(),
                std::string(format.label) + " -> " +
                    tensorTypeName(grouped_output.native_type()) +
                    " grouped embedding");
            expectGroupedCounter(
                format.label,
                tensorTypeName(grouped_output.native_type()),
                verifier_rows,
                d_model,
                format.cpu_weight_route);
        }
    }
} // namespace

class Test__CPUEmbeddingGroupedVerifier : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (!getCPUBackend())
            initCPUBackend(0);
    }
};

/**
 * @brief Prove all weight codebooks and all native CPU outputs are batch invariant.
 */
TEST_F(Test__CPUEmbeddingGroupedVerifier,
       AllWeightAndOutputFormatsM234MatchSerialDecodeBytes)
{
    constexpr int vocab_size = 67;
    constexpr int d_model = 256;
    ScopedPerfStats perfstats;

    uint32_t seed = 7100;
    for (const auto &format : embeddingVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto embedding_table = format.create({vocab_size, d_model}, seed++);
        ASSERT_NE(embedding_table, nullptr);
        ASSERT_STREQ(tensorTypeName(embedding_table->native_type()), format.label);

        runOutputFormatSweep<FP32Tensor>(
            format, embedding_table.get(), vocab_size, d_model);
        runOutputFormatSweep<BF16Tensor>(
            format, embedding_table.get(), vocab_size, d_model);
        runOutputFormatSweep<FP16Tensor>(
            format, embedding_table.get(), vocab_size, d_model);
        runOutputFormatSweep<Q8_1Tensor>(
            format, embedding_table.get(), vocab_size, d_model);
    }
}

/**
 * @brief Lock the canonical sweep registry to tensor and GPU packing metadata.
 *
 * This CPU-only contract test catches the exact failure mode that previously
 * hid Q8_K: a tensor could be loadable and constructible while an all-format
 * test table or GPU repack map omitted it. No device is initialized here.
 */
TEST_F(Test__CPUEmbeddingGroupedVerifier,
       CanonicalQuantizedRegistryMatchesTensorAndDevicePackingContracts)
{
    constexpr size_t expected_quantized_formats = 21;
    ASSERT_EQ(quantizedVerifierFormats().size(), expected_quantized_formats);
    ASSERT_EQ(
        embeddingVerifierFormats().size(),
        expected_quantized_formats + 3u)
        << "Embedding adds FP32, FP16, and BF16 to the canonical quantized matrix";

    std::unordered_set<std::string> labels;
    uint32_t seed = 8100;
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        ASSERT_TRUE(labels.emplace(format.label).second)
            << "Duplicate canonical verifier format";

        auto tensor = format.create({2u, 256u}, seed++);
        ASSERT_NE(tensor, nullptr);
        EXPECT_EQ(tensor->native_type(), format.tensor_type);
        EXPECT_STREQ(tensorTypeName(tensor->native_type()), format.label);

        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(tensor.get());
        ASSERT_NE(unpackable, nullptr);
        const NativeVnniFormatInfo *info = unpackable->vnniFormatInfo();
        ASSERT_NE(info, nullptr)
            << "Every canonical grouped weight format must be preparation-capable";
        EXPECT_EQ(info->codebook_id, format.source_codebook_id);
        EXPECT_EQ(info->is_superblock, format.source_is_superblock);
        EXPECT_EQ(
            canonicalDeviceVnniCodebookId(info->codebook_id),
            format.device_execution_codebook_id);
        EXPECT_TRUE(codebookIdToRepackFormat(
                        info->codebook_id,
                        info->is_superblock)
                        .has_value())
            << "Every canonical source format must have a device repack implementation";
    }
}

/**
 * @brief Guard against vocabulary-row index narrowing in the Qwen3.6 Q6_K path.
 *
 * Qwen3.6 has a 248,320-row token table. A compact synthetic table larger than
 * 65,536 rows catches accidental 16-bit index truncation without loading a
 * model, while the all-format sweep above covers M2/M3/M4 scheduling.
 */
TEST_F(Test__CPUEmbeddingGroupedVerifier,
       Q6KHighVocabularyRowIsByteIdenticalAtM1AndM2)
{
    constexpr int vocab_size = 65'540;
    constexpr int d_model = 256;
    constexpr int high_token = vocab_size - 1;

    auto embedding_table =
        TestTensorFactory::createQ6_KRandom({vocab_size, d_model}, 73);
    FP32Tensor grouped_output({2u, static_cast<size_t>(d_model)});
    FP32Tensor serial_output({1u, static_cast<size_t>(d_model)});
    const std::array<int, 2> grouped_tokens = {3, high_token};

    CPUEmbeddingKernelT<FP32Tensor> kernel;
    ASSERT_TRUE(kernel.apply_tensor(
        embedding_table.get(), grouped_tokens.data(), 2, d_model, &grouped_output));
    ASSERT_TRUE(kernel.apply_tensor(
        embedding_table.get(), grouped_tokens.data() + 1, 1, d_model, &serial_output));

    expectByteExact(
        grouped_output.data() + d_model,
        serial_output.data(),
        static_cast<size_t>(d_model) * sizeof(float),
        "Q6_K high vocabulary row");
}

/** @brief Confirm the CPU implementation remains workspace-free. */
TEST_F(Test__CPUEmbeddingGroupedVerifier, IsCPUOnlyAndWorkspaceFree)
{
    CPUEmbeddingKernelT<FP32Tensor> kernel;
    EXPECT_TRUE(kernel.supports_device(-1));
    EXPECT_FALSE(kernel.supports_device(0));
    auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(&kernel);
    ASSERT_NE(workspace_consumer, nullptr)
        << "CPUKernelBase exposes the uniform no-op workspace contract";
    EXPECT_TRUE(workspace_consumer->getWorkspaceRequirements(4, 64, 0).buffers.empty());
    EXPECT_FALSE(workspace_consumer->hasWorkspace());
}
