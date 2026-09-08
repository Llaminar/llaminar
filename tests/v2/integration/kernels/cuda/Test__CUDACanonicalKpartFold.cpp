/**
 * @file Test__CUDACanonicalKpartFold.cpp
 * @brief Captured CTA-local KPAR byte certificates against production KPAR.
 *
 * Both sides use the live sharded CUDA bridge and immutable IQ tables. Test
 * every physical codebook (including migration forms), partial column tiles,
 * empty partitions, the full CTA limit, epilogues, and replay with poisoned
 * outputs. Serial-row replay is only the diagnostic oracle in this fixture.
 * No performance threshold belongs in this model-free functional gate.
 */
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "backends/cuda/CUDAGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/cuda/gemm/CUDACanonicalKpartFold.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <array>
#include <bit>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C"
{
    bool cudaNativeVNNIInitIQGridTables_tuned();
    void cudaNativeVNNIGemvSweep_setConfig(int, int, int, int, int, int, int, int);
    void cudaNativeVNNIGemvSweep_clearConfig();
    void cudaNativeVNNIGemvSweep_setGroupedRows(int);
    void cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(int);
    bool cudaNativeVNNIGemvTuned_small_m_fp32(
        const int8_t *, const uint8_t *, const uint16_t *, const uint16_t *,
        const uint32_t *, float *, const float *, int, int, int, float, float,
        const float *, const float *, uint8_t, int, void *, CUDAGemvContext *,
        CUDARowMajorWeights **);
    bool cudaNativeVNNIGemvTuned_fp32(
        const int8_t *, const uint8_t *, const uint16_t *, const uint16_t *,
        const uint32_t *, float *, const float *, int, int, float, float,
        const float *, const float *, uint8_t, int, void *, CUDAGemvContext *,
        CUDARowMajorWeights **);
}

namespace
{
using llaminar2::CUDACanonicalKpartColumns;
using llaminar2::CUDACanonicalKpartFoldPlan;

/** @brief Attribute errors to their originating operation, never stale output. */
void checked(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/** @brief Fixture storage allocated before recording, retired after all graphs. */
template<class T> class Storage final
{
public:
    /** @brief Allocate a fixed element count outside capture. */
    explicit Storage(size_t count) : count_(count) { checked(cudaMalloc(&data, count * sizeof(T))); }
    /** @brief The later-declared execution owner has already joined. */
    ~Storage() { (void)cudaFree(data); }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    /** @brief Publish complete fixture bytes on the exact execution stream. */
    void upload(const std::vector<T> &values, cudaStream_t stream)
    {
        if (!stream || values.size() != count_) throw std::logic_error("invalid fixture publication");
        checked(cudaMemcpyAsync(data, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    T *data = nullptr;
private:
    size_t count_;
};

/** @brief Own terminal joining and graph/context lifetime, after referenced storage. */
class Execution final
{
public:
    /** @brief Create one explicit stream and one borrowed-partials context. */
    Execution() : context(cudaGemvContext_create(0), cudaGemvContext_destroy)
    {
        if (!context) throw std::runtime_error("missing native GEMV context");
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    /** @brief Join work before retiring captured pointers and releasing storage. */
    ~Execution()
    {
        (void)cudaStreamSynchronize(stream);
        graphs = {};
        (void)cudaStreamDestroy(stream);
    }
    Execution(const Execution &) = delete;
    Execution &operator=(const Execution &) = delete;
    std::unique_ptr<CUDAGemvContext, decltype(&cudaGemvContext_destroy)> context;
    cudaStream_t stream = nullptr;
    std::array<std::unique_ptr<llaminar2::CUDAGraphCapture>, 7> graphs;
};

/** @brief Scope a forceable physical family while preserving exact K partitions. */
class OraclePolicy final
{
public:
    /** @brief Force the existing two-phase KPAR bridge, never a private oracle. */
    explicit OraclePolicy(int partitions, int path, int columns)
    {
        if (path < 3)
            cudaNativeVNNIGemvSweep_setConfig(1, 64, 1, 1, 1, partitions, partitions, 1);
        else
            cudaNativeVNNIGemvSweep_setConfig(6, columns, 1, 0, 0, 0, partitions, 1);
        if (path >= 5)
        {
            cudaNativeVNNIGemvSweep_setGroupedRows(4);
            cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(1);
        }
    }
    /** @brief Restore automatic dispatch on every return/exception. */
    ~OraclePolicy()
    {
        cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(0);
        cudaNativeVNNIGemvSweep_clearConfig();
    }
    OraclePolicy(const OraclePolicy &) = delete;
    OraclePolicy &operator=(const OraclePolicy &) = delete;
};

/** @brief Stable signed data for every format, without a loaded model. */
uint32_t mixed(uint32_t value)
{
    value ^= value >> 16; value *= 0x7feb352dU;
    value ^= value >> 15; value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

/** @brief One explicit epilogue behavior, including the permitted output alias. */
enum class Epilogue { Identity, AffineBias, NegativeZero, InPlaceAffineBias };

/**
 * @brief Prove all output bytes for both compiled widths at one arithmetic identity.
 * @param codebook Canonical physical execution format.
 * @param payload_bytes Packed bytes for one 32-element weight block.
 * @param m Independent rows in this diagnostic captured transaction.
 * @param n Logical output width, including tile tails.
 * @param k Reduction length, a positive multiple of 32.
 * @param partitions Exact partition tree shared by oracle and candidate.
 * @param epilogue Explicit alpha/beta/bias/alias contract.
 */
void verify(int codebook, int payload_bytes, int m, int n, int k,
            int partitions, Epilogue epilogue)
{
    const size_t blocks = static_cast<size_t>(n) * (k / 32);
    const size_t outputs = static_cast<size_t>(m) * n;
    std::vector<uint8_t> payload(blocks * payload_bytes);
    std::vector<uint16_t> scales(blocks), secondary(blocks);
    std::vector<uint32_t> minima(blocks);
    std::vector<int8_t> activations(static_cast<size_t>(m) * k);
    std::vector<float> activation_scales(activations.size() / 32), prior(outputs), bias(n);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint8_t>(mixed(i + 19));
    for (size_t i = 0; i < blocks; ++i)
    {
        scales[i] = static_cast<uint16_t>(0x1000U + (mixed(i + 31) & 0x1fffU));
        secondary[i] = static_cast<uint16_t>(0x1000U + (mixed(i + 41) & 0x1fffU));
        minima[i] = scales[i] | (static_cast<uint32_t>(secondary[i]) << 16);
    }
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = int(mixed(i + 71) % 255) - 127;
    for (size_t i = 0; i < activation_scales.size(); ++i)
        activation_scales[i] = float(1 + mixed(i + 99) % 31) / 256;
    for (size_t i = 0; i < outputs; ++i) prior[i] = float(int(mixed(i + 213) % 255) - 127) / 128;
    for (int i = 0; i < n; ++i) bias[i] = float(int(mixed(i + 119) % 255) - 127) / 256;
    const bool affine = epilogue == Epilogue::AffineBias || epilogue == Epilogue::InPlaceAffineBias;
    const bool in_place = epilogue == Epilogue::InPlaceAffineBias;
    const float alpha = affine ? -0.625f : epilogue == Epilogue::NegativeZero ? -0.0f : 1.0f;
    const float beta = affine ? 0.375f : 0.0f;
    Storage<uint8_t> p(payload.size());
    Storage<uint16_t> s(blocks), secondary_s(blocks);
    Storage<uint32_t> em(blocks);
    Storage<int8_t> a(activations.size());
    Storage<float> sa(activation_scales.size()), old(outputs), b(n),
        partials(outputs * partitions), output(outputs + 8);
    Execution execution;
    const auto stream = execution.stream;
    p.upload(payload, stream); s.upload(scales, stream); secondary_s.upload(secondary, stream);
    em.upload(minima, stream); a.upload(activations, stream); sa.upload(activation_scales, stream);
    old.upload(prior, stream); b.upload(bias, stream);
    cudaGemvContext_bindWorkspace(execution.context.get(), partials.data,
                                 outputs * partitions * sizeof(float));
    checked(cudaStreamSynchronize(stream));
    std::vector<float> expected(outputs + 8), actual(outputs + 8);
    for (int path = 0; path < 7; ++path)
    {
        const bool wide_columns = path == 2 || path == 4 || path == 6;
        const auto plan = CUDACanonicalKpartFoldPlan::create(
            wide_columns ? CUDACanonicalKpartColumns::ThirtyTwo : CUDACanonicalKpartColumns::Sixteen,
            partitions, k);
        // Width 32 genuinely has a smaller physical limit; no replacement is launched.
        if (!plan) { ASSERT_TRUE(wide_columns); ASSERT_GT(partitions, 32); continue; }
        if (path)
        {
            llaminar2::CUDACanonicalKpartFoldResources resources;
            ASSERT_TRUE(cudaNativeVNNIGemvTuned_fusedKpar_resources(codebook, *plan, resources));
            ASSERT_EQ(resources.local_bytes, 0U);
            ASSERT_GE(resources.maximum_threads, plan->threads());
            ASSERT_GT(resources.active_blocks_per_sm, 0);
        }
        auto &graph = execution.graphs[path];
        graph = std::make_unique<llaminar2::CUDAGraphCapture>(stream, 0);
        OraclePolicy policy(partitions, path, static_cast<int>(plan->columns()));
        llaminar2::ScopedBackendGraphCapture recording(*graph, "canonical CTA-local fold certificate");
        ASSERT_TRUE(recording.begin());
        checked(cudaMemsetAsync(output.data, 0x7e, (outputs + 8) * sizeof(float), stream));
        if (in_place)
            checked(cudaMemcpyAsync(output.data, old.data, outputs * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
        if (path >= 5 && m >= 2)
        {
            // One real grouped launch inherits the fused serial schedule; it
            // must not become a row-replay implementation under this policy.
            ASSERT_TRUE(cudaNativeVNNIGemvTuned_small_m_fp32(
                a.data, p.data, s.data, secondary_s.data, em.data, output.data,
                sa.data, m, n, k, alpha, beta,
                affine ? (in_place ? output.data : old.data) : nullptr,
                affine ? b.data : nullptr, codebook, 0, stream,
                execution.context.get(), nullptr));
        }
        else for (int row = 0; row < m; ++row)
        {
            const size_t offset = static_cast<size_t>(row) * n;
            const float *existing = affine ? (in_place ? output.data : old.data) + offset : nullptr;
            const auto *activation = a.data + static_cast<size_t>(row) * k;
            const auto *row_scales = sa.data + static_cast<size_t>(row) * (k / 32);
            if (path == 0 || path >= 3)
                ASSERT_TRUE(cudaNativeVNNIGemvTuned_fp32(
                    activation, p.data, s.data, secondary_s.data, em.data, output.data + offset,
                    row_scales, n, k, alpha, beta, existing, affine ? b.data : nullptr,
                    codebook, 0, stream, execution.context.get(), nullptr));
            else
                ASSERT_TRUE(cudaNativeVNNIGemvTuned_fusedKpar_fp32(
                    activation, p.data, s.data, secondary_s.data, em.data, output.data + offset,
                    row_scales, n, alpha, beta, existing, affine ? b.data : nullptr,
                    codebook, 0, stream, *plan));
        }
        recording.finish();
        ASSERT_TRUE(graph->instantiate());
        for (int replay = 0; replay < 20; ++replay)
        {
            ASSERT_TRUE(graph->launch());
            checked(cudaMemcpyAsync(actual.data(), output.data, (outputs + 8) * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
            checked(cudaStreamSynchronize(stream));
            for (size_t guard = outputs; guard < outputs + 8; ++guard)
                ASSERT_EQ(std::bit_cast<uint32_t>(actual[guard]), 0x7e7e7e7eU);
            if (path == 0 && replay == 0) expected = actual;
            else ASSERT_EQ(std::memcmp(expected.data(), actual.data(), outputs * sizeof(float)), 0)
                << "path=" << path << " replay=" << replay;
        }
    }
}

/** @brief Use the production catalog to include all physical migration representations. */
std::map<int, int> formats()
{
    std::map<int, int> result;
    for (const auto &source : llaminar2::native_vnni_formats::kAllSourceFormats)
    {
        const auto &f = *source.metadata;
        result.emplace(llaminar2::canonicalDeviceVnniCodebookId(f.codebook_id), f.payload_bytes);
        const auto promoted = llaminar2::migrationStableDeviceVnniFormat(f);
        result.emplace(promoted.codebook_id, promoted.payload_bytes_per_block);
    }
    return result;
}
} // namespace

/** @brief All formats/epilogues, repeated captured rows, and ragged arithmetic boundaries. */
TEST(CUDACanonicalKpartFold, CapturedAllCodebooksRowsAndEpilogues)
{
    checked(cudaSetDevice(0));
    ASSERT_TRUE(cudaNativeVNNIInitIQGridTables_tuned());
    for (const auto &[codebook, bytes] : formats())
        for (int m : {1, 2, 3, 16, 31})
            for (const auto geometry : {std::array{1, 32, 1}, std::array{33, 160, 3},
                                        std::array{129, 672, 20}})
                for (auto epilogue : {Epilogue::Identity, Epilogue::AffineBias, Epilogue::NegativeZero})
                {
                    SCOPED_TRACE(::testing::Message() << "CB=" << codebook << " M=" << m
                        << " N=" << geometry[0] << " K=" << geometry[1] << " KB=" << geometry[2]
                        << " epilogue=" << static_cast<int>(epilogue));
                    verify(codebook, bytes, m, geometry[0], geometry[1], geometry[2], epilogue);
                    if (::testing::Test::HasFatalFailure()) return;
                }
}

/** @brief Native CTA limits and output aliasing must work for every physical codebook. */
TEST(CUDACanonicalKpartFold, CapturedCapacityAndInPlacePublication)
{
    checked(cudaSetDevice(0));
    ASSERT_TRUE(cudaNativeVNNIInitIQGridTables_tuned());
    for (const auto &[codebook, bytes] : formats())
        for (int partitions : {31, 32, 33, 63, 64})
        {
            SCOPED_TRACE(::testing::Message() << "CB=" << codebook << " KB=" << partitions);
            verify(codebook, bytes, 1, 33, (partitions + 1) * 32, partitions,
                   Epilogue::InPlaceAffineBias);
            if (::testing::Test::HasFatalFailure()) return;
        }
}

/** @brief Rejected metadata/stream/device contracts must not enqueue any work. */
TEST(CUDACanonicalKpartFold, RejectsInvalidBindingsBeforeSubmission)
{
    checked(cudaSetDevice(0));
    Storage<int8_t> a(32);
    Storage<uint8_t> p(32);
    Storage<uint16_t> s(1);
    Storage<float> out(1), scales(1);
    Execution execution;
    const auto plan = *CUDACanonicalKpartFoldPlan::create(CUDACanonicalKpartColumns::Sixteen, 1, 32);
    const auto submit = [&](uint8_t cb, void *stream, int device, const int8_t *activation,
                            float beta, const uint16_t *primary) {
        return cudaNativeVNNIGemvTuned_fusedKpar_fp32(activation, p.data, primary, nullptr,
            nullptr, out.data, scales.data, 1, 1.0f, beta, nullptr, nullptr, cb, device, stream, plan);
    };
    // Admission failures must leave a live recording usable. In particular,
    // stream ownership queries may not invalidate capture as a side effect.
    llaminar2::CUDAGraphCapture graph(execution.stream, 0);
    llaminar2::ScopedBackendGraphCapture recording(graph, "rejected fold binding capture isolation");
    ASSERT_TRUE(recording.begin());
    checked(cudaMemsetAsync(out.data, 0, sizeof(float), execution.stream));
    EXPECT_FALSE(submit(0, nullptr, 0, a.data, 0, s.data));
    EXPECT_FALSE(submit(0, cudaStreamLegacy, 0, a.data, 0, s.data));
    EXPECT_FALSE(submit(0, cudaStreamPerThread, 0, a.data, 0, s.data));
    EXPECT_FALSE(submit(0, execution.stream, 1, a.data, 0, s.data));
    EXPECT_FALSE(submit(0, execution.stream, 0, a.data + 1, 0, s.data));
    EXPECT_FALSE(submit(0, execution.stream, 0, a.data, 1, s.data));
    EXPECT_FALSE(submit(0, execution.stream, 0, a.data, 0, nullptr));
    EXPECT_FALSE(submit(255, execution.stream, 0, a.data, 0, s.data));
    for (uint8_t cb : {5, 7, 8, 9, 10, 13, 14, 16, 17, 23})
        EXPECT_FALSE(submit(cb, execution.stream, 0, a.data, 0, s.data));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    checked(cudaStreamIsCapturing(execution.stream, &capture_status));
    EXPECT_EQ(capture_status, cudaStreamCaptureStatusActive);
    recording.finish();
    ASSERT_TRUE(graph.instantiate());
    ASSERT_TRUE(graph.launch());
    // This local graph is declared after the execution owner; retire its work
    // explicitly before its destructor, rather than relying on that later join.
    checked(cudaStreamSynchronize(execution.stream));
}
