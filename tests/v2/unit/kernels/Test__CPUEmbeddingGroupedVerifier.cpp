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
#include "tensors/TensorSlice.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "../../utils/EmbeddingVerifierFormats.h"
#include "../../utils/VerifierRowTestInventory.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
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
     * @brief Run the runtime-M grouped-vs-serial proof for one weight/output pair.
     *
     * The same production kernel object serves both sides. This matters for
     * quantized tables because it proves that independent requested source rows
     * retain the same values while only the invocation width changes.
     */
    template <typename OutputTensorT>
    void runOutputFormatSweep(
        const EmbeddingVerifierFormatCase &format,
        TensorBase *embedding_table,
        int vocab_size,
        int d_model)
    {
        CPUEmbeddingKernelT<OutputTensorT> kernel;
        std::vector<int> tokens(
            static_cast<size_t>(kGroupedVerifierRuntimeRows.back()));
        for (size_t row = 0; row < tokens.size(); ++row)
        {
            tokens[row] = static_cast<int>(
                (3u + row * 17u) % static_cast<size_t>(vocab_size));
        }

        for (int verifier_rows : kGroupedVerifierRuntimeRows)
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

/** @brief Nested TP wrappers retain exact CPU results for every source format. */
TEST_F(Test__CPUEmbeddingGroupedVerifier, NestedSlicesMatchUnwrappedSourceBytes)
{
    constexpr int vocab_size = 19, d_model = 256;
    const std::array<int, 3> tokens{0, 7, vocab_size - 1};
    for (const auto &format : embeddingVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto table = format.create({vocab_size, d_model}, 462);
        FP32Tensor reference({tokens.size(), d_model});
        {
            CPUEmbeddingKernelT<FP32Tensor> kernel;
            ASSERT_TRUE(kernel.apply_tensor(table.get(), tokens.data(), tokens.size(),
                                            d_model, &reference));
        }
        for (int nesting = 0; nesting != 2; ++nesting)
        {
            table = std::make_unique<TensorSlice>(std::move(table), SliceMetadata{});
            FP32Tensor actual({tokens.size(), d_model});
            CPUEmbeddingKernelT<FP32Tensor> kernel;
            ASSERT_TRUE(kernel.apply_tensor(table.get(), tokens.data(), tokens.size(),
                                            d_model, &actual));
            expectByteExact(actual.raw_data(), reference.raw_data(), reference.size_bytes(),
                            std::string(format.label) + " nested CPU source");
        }
    }
}

/**
 * @brief Lock the canonical sweep registry to tensor and GPU packing metadata.
 *
 * This CPU-only contract test catches the exact failure mode that previously
 * hid Q8_K: a tensor could be loadable and constructible while an all-format
 * test table or GPU repack map omitted it. Source4+superblock must remain
 * distinct from physical24 without changing source-policy4. No device is initialized here.
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
            canonicalDeviceVnniCodebookId(*info),
            format.device_execution_codebook_id);
        if (format.tensor_type == TensorType::IQ4_XS)
        {
            EXPECT_EQ(info->codebook_id, 4); EXPECT_TRUE(info->is_superblock);
            EXPECT_EQ(format.device_execution_codebook_id, 24);
        }
        if (format.tensor_type == TensorType::IQ4_NL)
        {
            EXPECT_EQ(info->codebook_id, 4); EXPECT_FALSE(info->is_superblock);
            EXPECT_EQ(format.device_execution_codebook_id, 4);
        }
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

/** @test Native prepared layout has exact source bytes and no floating conversion route. */
TEST_F(Test__CPUEmbeddingGroupedVerifier, NativeLayoutRetainsEverySourceFormatAndRejectsInvalidGeometry)
{
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto source = format.create({3, 256}, 981);
        const auto layout = NativeEmbeddingLayout::forSource(format.tensor_type, 3, 256);
        EXPECT_EQ(layout.source_type, format.tensor_type);
        EXPECT_EQ(layout.byte_size, source->size_bytes());
        EXPECT_EQ(layout.byte_size, 3u * layout.row_stride_bytes);
        EXPECT_EQ(layout, NativeEmbeddingLayout::forFormat(format.label, 3, 256));
        EXPECT_EQ(NativeEmbeddingLayout::forSource(format.tensor_type, 0, 256).byte_size, 0u);
        EXPECT_THROW(NativeEmbeddingLayout::forSource(format.tensor_type, 3, 0), std::invalid_argument);
        EXPECT_THROW(NativeEmbeddingLayout::forSource(format.tensor_type,
            std::numeric_limits<size_t>::max(), 256), std::overflow_error);
    }
    for (const auto source : {TensorType::FP32, TensorType::FP16, TensorType::BF16,
             TensorType::INT32, TensorType::TQ4, TensorType::TQ8, TensorType::AQ8})
    {
        EXPECT_FALSE(NativeEmbeddingLayout::supports(source));
        EXPECT_THROW(NativeEmbeddingLayout::forSource(source, 3, 256), std::invalid_argument);
    }
    EXPECT_THROW(NativeEmbeddingLayout::forFormat("FP8", 3, 256), std::invalid_argument);
}

namespace
{
    /**
     * @brief Compare native lookup against independent source decoding and activation conversion.
     * @tparam Output Activation tensor precision under test.
     * @param source Immutable quantized source with at least seven native rows.
     */
    template <typename Output>
    void requireOriginalEmbeddingRows(TensorBase &source)
    {
        const int width = static_cast<int>(source.cols());
        const std::array<int, 7> tokens{6, 0, 3, 6, 2, 0, 1};
        FP32Tensor independently_decoded({source.rows(), static_cast<size_t>(width)});
        for (size_t row = 0; row < source.rows(); ++row)
            source.to_fp32_row(row, independently_decoded.mutable_data() + row * width);
        CPUEmbeddingKernelT<Output> actual_kernel, reference_kernel;
        for (const int rows : {7, 3, 1, 7, 2})
        {
            SCOPED_TRACE(rows);
            Output actual({static_cast<size_t>(rows), static_cast<size_t>(width)});
            Output expected({static_cast<size_t>(rows), static_cast<size_t>(width)});
            ASSERT_TRUE(actual_kernel.apply_tensor(&source, tokens.data(), rows, width, &actual));
            ASSERT_TRUE(reference_kernel.apply_tensor(&independently_decoded, tokens.data(), rows, width, &expected));
            expectByteExact(actual.raw_data(), expected.raw_data(), actual.size_bytes(),
                            "original source embedding values and declared activation conversion");
        }
    }
}

/** @test Every quantized source retains original row values in all CPU activation precisions. */
TEST_F(Test__CPUEmbeddingGroupedVerifier, NativeQuantizedSourceValuesMatchIndependentRows)
{
    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto source = format.create({7, 256}, 1637);
        const auto *raw = static_cast<const uint8_t *>(source->raw_data());
        const std::vector<uint8_t> original(raw, raw + source->size_bytes());
        requireOriginalEmbeddingRows<FP32Tensor>(*source);
        requireOriginalEmbeddingRows<FP16Tensor>(*source);
        requireOriginalEmbeddingRows<BF16Tensor>(*source);
        requireOriginalEmbeddingRows<Q8_1Tensor>(*source);
        EXPECT_EQ(std::memcmp(source->raw_data(), original.data(), original.size()), 0);
    }
    // Activation conversion follows the same rounding as a direct FP32 row,
    // including halfway cases among binary16 subnormals. These original
    // binary32 Q8_K scales must reach the activation converter unchanged.
    std::vector<uint8_t> bytes(7 * sizeof(Q8_KBlock));
    auto *blocks = reinterpret_cast<Q8_KBlock *>(bytes.data());
    constexpr std::array<float, 7> scales{
        0x1.8p-24f, -0x1.8p-24f, 0x1.8p-23f, -0x1.8p-23f,
        0x1.fffffep-15f, -0x1.fffffep-15f, 0x1.000002p0f};
    for (size_t row = 0; row < scales.size(); ++row)
    {
        blocks[row].d = scales[row];
        std::memset(blocks[row].qs, 1, sizeof(blocks[row].qs));
        std::fill(std::begin(blocks[row].bsums), std::end(blocks[row].bsums), 16);
    }
    Q8_KTensor source({7, 256}, bytes);
    requireOriginalEmbeddingRows<FP32Tensor>(source);
    requireOriginalEmbeddingRows<FP16Tensor>(source);
    requireOriginalEmbeddingRows<BF16Tensor>(source);
    requireOriginalEmbeddingRows<Q8_1Tensor>(source);
}

/** @test Unequal Q6_K subscales and Q4_K derived minima expose the old lossy embedding conversion. */
TEST_F(Test__CPUEmbeddingGroupedVerifier, NativeQ6KAndQ4KAdversarialScalesPreserveSourceValues)
{
    std::vector<uint8_t> q6_bytes(7 * sizeof(Q6_KBlock));
    auto *q6 = reinterpret_cast<Q6_KBlock *>(q6_bytes.data());
    for (size_t row = 0; row < 7; ++row)
    {
        q6[row].d = fp32_to_fp16(0.1f);
        std::memset(q6[row].ql, 0xff, sizeof(q6[row].ql));
        std::memset(q6[row].qh, 0xff, sizeof(q6[row].qh));
        for (size_t scale = 0; scale < 16; ++scale)
            q6[row].scales[scale] = static_cast<int8_t>(scale % 2 == 0 ? 1 : 11);
    }
    Q6_KTensor q6_source({7, 256}, q6_bytes);
    requireOriginalEmbeddingRows<FP32Tensor>(q6_source);
    std::vector<uint8_t> q4_bytes(7 * sizeof(Q4_KBlock));
    auto *q4 = reinterpret_cast<Q4_KBlock *>(q4_bytes.data());
    for (size_t row = 0; row < 7; ++row)
    {
        q4[row].d = fp32_to_fp16(0.1f);
        q4[row].dmin = fp32_to_fp16(0.333f);
        std::memset(q4[row].scales, 0x3f, sizeof(q4[row].scales));
        std::memset(q4[row].qs, 0x21, sizeof(q4[row].qs));
    }
    Q4_KTensor q4_source({7, 256}, q4_bytes);
    requireOriginalEmbeddingRows<FP32Tensor>(q4_source);
    // Activation conversion must stay block-bounded rather than imposing a
    // whole-row stack limit on an otherwise supported native source width.
    auto wide_source = TestTensorFactory::createQ6_KRandom({7, 8448}, 2267);
    requireOriginalEmbeddingRows<FP16Tensor>(*wide_source);
    requireOriginalEmbeddingRows<BF16Tensor>(*wide_source);
    requireOriginalEmbeddingRows<Q8_1Tensor>(*wide_source);
}
