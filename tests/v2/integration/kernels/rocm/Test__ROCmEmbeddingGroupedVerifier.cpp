/**
 * @file Test__ROCmEmbeddingGroupedVerifier.cpp
 * @brief ROCm production-path embedding batch-invariance and residency proof.
 *
 * Every supported quantized embedding table is prepared through the same
 * model-owned PreparedEmbeddingWeights API used by graph construction. Token
 * IDs are uploaded once by the fixture and then supplied to the kernel as a
 * device pointer. For every certified runtime M, one grouped lookup is compared
 * byte-for-byte against M production M=1 lookups on the same explicit stream.
 *
 * The route counter is part of the assertion: equality alone is insufficient
 * if a test accidentally exercises host token upload or non-prepared weights.
 */

#include <gtest/gtest.h>

#include "backends/BackendManager.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/KernelFactory.h"
#include "kernels/rocm/ops/ROCmEmbeddingKernelT.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "../../../utils/EmbeddingVerifierFormats.h"
#include "../../../utils/VerifierRowTestInventory.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar::v2::kernels;

#ifdef HAVE_ROCM
namespace
{
    /** @brief Explicit non-default stream with deterministic lifetime. */
    class ScopedHipStream
    {
    public:
        bool create()
        {
            return hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) == hipSuccess;
        }

        ~ScopedHipStream()
        {
            if (stream_)
                (void)hipStreamDestroy(stream_);
        }

        hipStream_t get() const { return stream_; }

    private:
        hipStream_t stream_ = nullptr;
    };

    /** @brief Small RAII wrapper for test-owned HIP buffers. */
    class HipAllocation
    {
    public:
        bool allocate(size_t bytes)
        {
            return hipMalloc(&pointer_, bytes) == hipSuccess;
        }

        ~HipAllocation()
        {
            if (pointer_)
                (void)hipFree(pointer_);
        }

        void *get() const { return pointer_; }

    private:
        void *pointer_ = nullptr;
    };

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

    /** @brief Emit a precise diagnostic for the first FP32 byte mismatch. */
    void expectByteExact(
        const std::vector<float> &grouped,
        const std::vector<float> &serial,
        const std::string &context)
    {
        ASSERT_EQ(grouped.size(), serial.size()) << context;
        if (std::memcmp(grouped.data(), serial.data(), grouped.size() * sizeof(float)) == 0)
            return;

        for (size_t index = 0; index < grouped.size(); ++index)
        {
            if (std::memcmp(&grouped[index], &serial[index], sizeof(float)) != 0)
            {
                uint32_t grouped_bits = 0;
                uint32_t serial_bits = 0;
                std::memcpy(&grouped_bits, &grouped[index], sizeof(grouped_bits));
                std::memcpy(&serial_bits, &serial[index], sizeof(serial_bits));
                ADD_FAILURE() << context << " first mismatch at element " << index
                              << " grouped=" << grouped[index]
                              << " serial=" << serial[index]
                              << " grouped_bits=0x" << std::hex << grouped_bits
                              << " serial_bits=0x" << serial_bits << std::dec;
                return;
            }
        }
    }

    /** @brief Require the prepared, device-token grouped ROCm route. */
    void expectGroupedCounter(
        const char *weight_format,
        int verifier_rows,
        int d_model,
        const char *weight_route)
    {
        bool found = false;
        for (const auto &record : PerfStatsCollector::snapshot(
                 {"kernel.rocm_embedding_grouped_verifier_rows_calls"}))
        {
            auto tagEquals = [&](const char *name, const std::string &expected)
            {
                const auto it = record.tags.find(name);
                return it != record.tags.end() && it->second == expected;
            };
            found = found ||
                    (tagEquals("weight_format", weight_format) &&
                     tagEquals("verifier_rows", std::to_string(verifier_rows)) &&
                     tagEquals("d_model", std::to_string(d_model)) &&
                     tagEquals("weight_route", weight_route) &&
                     tagEquals("token_source", "device") &&
                     tagEquals("invocation_policy", "single_grouped_launch") &&
                     record.value == 1.0);
        }
        EXPECT_TRUE(found)
            << "ROCm embedding did not publish the required grouped route for "
            << weight_format << " M=" << verifier_rows << "\n"
            << PerfStatsCollector::summaryString(
                   {"kernel.rocm_embedding_grouped_verifier_rows_calls"}, 20);
    }

    /**
     * @brief Exercise one table format through grouped and serial production calls.
     */
    enum class SourceView { Direct, NestedVocabularyShard };

    /** @brief Release only this test's retained native graph after its stream joins. */
    struct CapturedEmbeddingGraph
    {
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        ~CapturedEmbeddingGraph()
        {
            if (executable) (void)hipGraphExecDestroy(executable);
            if (graph) (void)hipGraphDestroy(graph);
        }
    };

    /** @brief Compare a direct or wrapped source on its real captured device path. */
    void runFormat(
        const EmbeddingVerifierFormatCase &format,
        uint32_t seed,
        hipStream_t stream,
        SourceView source_view = SourceView::Direct)
    {
        constexpr int vocab_size = 67;
        constexpr int d_model = 256;
        constexpr int max_rows = kGroupedVerifierRuntimeRows.back();
        const DeviceId device = DeviceId::rocm(0);
        std::array<int, max_rows> tokens{};
        for (int row = 0; row < max_rows; ++row)
            tokens[static_cast<size_t>(row)] = (3 + row * 17) % vocab_size;
        tokens[1] = vocab_size - 1;

        auto embedding_table = format.create({vocab_size, d_model}, seed);
        ASSERT_NE(embedding_table, nullptr);
        ASSERT_STREQ(tensorTypeName(embedding_table->native_type()), format.label);

        std::vector<float> floating_reference;
        if (!format.prepared_embed_q8)
        {
            floating_reference.resize(vocab_size * d_model);
            embedding_table->to_fp32(floating_reference.data());
        }
        const int vocab_offset = source_view == SourceView::NestedVocabularyShard ? 17 : 0;
        if (source_view == SourceView::NestedVocabularyShard)
        {
            // The backing table is already a vocabulary shard. Nested wrappers
            // preserve its bytes and cannot create an INT8 unpack capability.
            SliceMetadata metadata;
            metadata.mode = SliceMode::ROW_PARALLEL;
            metadata.original_rows = vocab_size + vocab_offset;
            metadata.original_cols = d_model;
            metadata.slice_start = vocab_offset;
            metadata.slice_end = vocab_offset + vocab_size;
            metadata.inner_is_presliced = true;
            embedding_table = std::make_unique<TensorSlice>(std::move(embedding_table), metadata);
            embedding_table = std::make_unique<TensorSlice>(std::move(embedding_table), metadata);
            for (auto &token : tokens) token += vocab_offset;
        }
        ASSERT_EQ(IINT8Unpackable::fromTensor(embedding_table.get()) != nullptr,
                  format.prepared_embed_q8);

        std::shared_ptr<PreparedEmbeddingHandle> prepared;
        if (format.prepared_embed_q8)
        {
            prepared = KernelFactory::prepareEmbeddingHandleLocal(
                embedding_table.get(), d_model, device, vocab_offset, vocab_size + vocab_offset);
            ASSERT_NE(prepared, nullptr) << format.label;
            ASSERT_NE(prepared->weights, nullptr) << format.label;
            ASSERT_NE(prepared->weights->device_data, nullptr) << format.label;
        }
        else
        {
            ASSERT_TRUE(embedding_table->ensureOnDevice(device, stream));
        }

        HipAllocation device_tokens;
        ASSERT_TRUE(device_tokens.allocate(tokens.size() * sizeof(int)));
        ASSERT_EQ(
            hipMemcpyAsync(
                device_tokens.get(),
                tokens.data(),
                tokens.size() * sizeof(int),
                hipMemcpyHostToDevice,
                stream),
            hipSuccess);

        ROCmEmbeddingKernelT kernel(0);
        kernel.setGPUStream(stream);
        kernel.setVocabRange(vocab_offset, vocab_size);
        if (prepared)
            kernel.setPreparedEmbeddingHandle(prepared.get());

        DeviceWorkspaceManager workspace(device, 4096);
        const auto requirements = kernel.getWorkspaceRequirements(max_rows, vocab_size, d_model);
        ASSERT_EQ(requirements.buffers.size(), 1u);
        ASSERT_EQ(requirements.buffers.front().name, EmbeddingWorkspaceBuffers::TOKEN_IDS);
        ASSERT_TRUE(workspace.allocate(requirements));
        kernel.bindWorkspace(&workspace);

        for (int verifier_rows : kGroupedVerifierRuntimeRows)
        {
            SCOPED_TRACE(std::string(format.label) + " M=" + std::to_string(verifier_rows));
            FP32Tensor grouped_output(
                {static_cast<size_t>(verifier_rows), static_cast<size_t>(d_model)});
            FP32Tensor one_row_output({1u, static_cast<size_t>(d_model)});
            ASSERT_TRUE(grouped_output.allocateOnDevice(device, stream));
            ASSERT_TRUE(one_row_output.allocateOnDevice(device, stream));

            PerfStatsCollector::reset();
            kernel.setDynamicDeviceTokenIds(device_tokens.get(), verifier_rows);
            CapturedEmbeddingGraph captured;
            const bool capture = source_view == SourceView::NestedVocabularyShard;
            std::optional<GraphCaptureGuard> capture_guard;
            if (capture) capture_guard.emplace();
            if (capture)
                ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
                          hipSuccess);
            const bool applied = kernel.apply_tensor(
                embedding_table.get(), nullptr, verifier_rows, d_model, &grouped_output, nullptr, 0);
            // End recording before asserting so a rejected operation cannot
            // leave the stream recording during test-fixture destruction.
            if (capture)
                ASSERT_EQ(hipStreamEndCapture(stream, &captured.graph), hipSuccess);
            ASSERT_TRUE(applied);
            capture_guard.reset();
            if (capture)
            {
                ASSERT_EQ(hipGraphInstantiate(&captured.executable, captured.graph,
                          nullptr, nullptr, 0), hipSuccess);
                ASSERT_EQ(hipGraphLaunch(captured.executable, stream), hipSuccess);
            }

            std::vector<float> grouped(
                static_cast<size_t>(verifier_rows) * d_model);
            std::vector<float> serial(grouped.size());
            ASSERT_EQ(
                hipMemcpyAsync(
                    grouped.data(),
                    grouped_output.gpu_data_ptr(),
                    grouped.size() * sizeof(float),
                    hipMemcpyDeviceToHost,
                    stream),
                hipSuccess);

            auto *token_base = static_cast<int *>(device_tokens.get());
            for (int row = 0; row < verifier_rows; ++row)
            {
                kernel.setDynamicDeviceTokenIds(token_base + row, 1);
                ASSERT_TRUE(kernel.apply_tensor(
                    embedding_table.get(), nullptr, 1, d_model, &one_row_output, nullptr, 0));
                ASSERT_EQ(
                    hipMemcpyAsync(
                        serial.data() + static_cast<size_t>(row) * d_model,
                        one_row_output.gpu_data_ptr(),
                        static_cast<size_t>(d_model) * sizeof(float),
                        hipMemcpyDeviceToHost,
                        stream),
                    hipSuccess);
            }

            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            expectByteExact(grouped, serial, std::string(format.label) + " ROCm embedding");
            if (!floating_reference.empty())
            {
                std::vector<float> expected(grouped.size());
                for (int row = 0; row < verifier_rows; ++row)
                    std::memcpy(expected.data() + row * d_model,
                                floating_reference.data() + (tokens[row] - vocab_offset) * d_model,
                                d_model * sizeof(float));
                // A grouped/serial comparison alone would accept both paths
                // returning zeros. Check against the untouched source bytes.
                expectByteExact(grouped, expected, std::string(format.label) + " native source");
            }
            expectGroupedCounter(
                format.label,
                verifier_rows,
                d_model,
                format.gpu_weight_route);
        }
    }
} // namespace

class Test__ROCmEmbeddingGroupedVerifier : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess || count <= 0)
            GTEST_SKIP() << "No ROCm device available";
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_NE(getROCmBackend(), nullptr);
    }
};

/** @brief Sweep every embedding codebook through real prepared/device-owned execution. */
TEST_F(Test__ROCmEmbeddingGroupedVerifier,
       AllWeightFormatsDeviceTokensRuntimeMMatchSerialDecodeBytes)
{
    ScopedHipStream stream;
    ASSERT_TRUE(stream.create());
    ScopedPerfStats perfstats;

    uint32_t seed = 9100;
    for (const auto &format : embeddingVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        runFormat(format, seed++, stream.get());
    }
}

/** @brief Every format preserves its representation through captured TP wrappers. */
TEST_F(Test__ROCmEmbeddingGroupedVerifier, NestedVocabularySlicesPreserveCapturedRepresentation)
{
    ScopedHipStream stream;
    ASSERT_TRUE(stream.create());
    ScopedPerfStats perfstats;
    uint32_t seed = 9200;
    for (const auto &format : embeddingVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        runFormat(format, seed++, stream.get(), SourceView::NestedVocabularyShard);
    }
}

/** @brief Prove the removed lazy workspace-repack route cannot execute. */
TEST_F(Test__ROCmEmbeddingGroupedVerifier,
       QuantizedExecutionRequiresMatchingPreparedDeviceWeights)
{
    constexpr int vocab_size = 17;
    constexpr int d_model = 256;
    const DeviceId device = DeviceId::rocm(0);
    auto embedding_table = TestTensorFactory::createQ4_0Random({vocab_size, d_model}, 99);

    ScopedHipStream stream;
    ASSERT_TRUE(stream.create());
    ROCmEmbeddingKernelT kernel(0);
    kernel.setGPUStream(stream.get());
    DeviceWorkspaceManager workspace(device, 4096);
    const auto requirements = kernel.getWorkspaceRequirements(1, vocab_size, d_model);
    ASSERT_EQ(requirements.buffers.size(), 1u);
    ASSERT_TRUE(workspace.allocate(requirements));
    kernel.bindWorkspace(&workspace);

    const int token = 3;
    HipAllocation device_token;
    ASSERT_TRUE(device_token.allocate(sizeof(token)));
    ASSERT_EQ(hipMemcpyAsync(device_token.get(), &token, sizeof(token), hipMemcpyHostToDevice,
                             stream.get()),
              hipSuccess);
    kernel.setDynamicDeviceTokenIds(device_token.get(), 1);

    FP32Tensor output({1u, static_cast<size_t>(d_model)});
    ASSERT_TRUE(output.allocateOnDevice(device, stream.get()));
    EXPECT_FALSE(kernel.apply_tensor(
        embedding_table.get(), nullptr, 1, d_model, &output, nullptr, 0));
    ASSERT_EQ(hipStreamSynchronize(stream.get()), hipSuccess);
}
#else
TEST(Test__ROCmEmbeddingGroupedVerifier, ROCmUnavailable)
{
    GTEST_SKIP() << "ROCm support is not compiled";
}
#endif
